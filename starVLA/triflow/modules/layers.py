"""PR2: TriFlow 基础层 (ELF 块设计的 PyTorch 移植).

复用 (逻辑移植, 不 import JAX 代码):
- Third_github/ELF/src/modules/layers.py: RMSNorm / SwiGLUFFN / Attention(qk_norm) /
  TimestepEmbedder / 初始化纪律 (xavier + N(0,0.02) + 输出层零初始化)

说明:
全程 fp32 训练 (88M 小模型, 避开 v1 bf16 NaN 坑), RMSNorm 内部仍显式提升 fp32 计算。
注意力用 torch SDPA + 4D additive float mask ([B,1,T,T], 可见=0 / 屏蔽=-1e4),
不用 RoPE —— 序列短 (≤628)、块异构, 位置信息由 learned pos/type embedding 提供。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


######### // code // ##########
# 中文注释：RMSNorm（ELF 同款）。输入 [..., D] → fp32 里做 rsqrt(mean(x²)) 再乘可学习权重。
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * x.to(self.weight.dtype)).to(input_dtype)
######### // code // ##########


######### // code // ##########
# 中文注释：标量 timestep → 向量嵌入（ELF 同款：sinusoidal(256) → Linear → SiLU → Linear）。
# 输入 t [B] (连续值, ELF 直接喂 [0,1] 原值, 不乘 1000 —— 保持与其调好的 P_mean/P_std 一致)。
# 输出 [B, hidden_size]。kernel 初始化 N(0, 0.02)，bias 0。
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.reshape(-1, 1).float() * freqs.reshape(1, -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))
######### // code // ##########


######### // code // ##########
# 中文注释：SwiGLU FFN（ELF 同款）：hidden = int(dim * mlp_ratio * 2/3)；
# w12 一次性出 2*hidden 再 split，silu(x1)*x2 → w3 回 dim。
class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden, bias=bias)
        self.w3 = nn.Linear(hidden, dim, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(x1) * x2))
######### // code // ##########


######### // code // ##########
# 中文注释：双向多头自注意力。
# 输入 x [B,T,C]，attn_bias [B,1,T,T] additive float mask（可见=0 / 屏蔽=-1e4，None=全可见）。
# qk_norm：per-head RMSNorm（ELF 同款，from-scratch 稳定性关键件之一）。
class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must divide num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).reshape(bsz, seq, dim)
        return self.proj_drop(self.proj(out))
######### // code // ##########


######### // code // ##########
# 中文注释：pre-norm transformer 块（ELFBlock 同款）：
# x = x + Attn(RMSNorm(x)); x = x + SwiGLU(RMSNorm(x))
class TriBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads, qk_norm=qk_norm, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_bias)
        x = x + self.ffn(self.norm2(x))
        return x
######### // code // ##########


######### // code // ##########
# 中文注释：初始化助手（ELF 纪律）：Linear → xavier_uniform + bias 0；Embedding → N(0,0.02)。
# 输出投影的零初始化由调用方在构造后单独覆盖（见 BlockEmbedder）。
def init_xavier_(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
######### // code // ##########
