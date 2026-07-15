# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""WAM World→Action guidance building blocks.

中文注释：World→Action 引导大计划（M0–M6+）的统一构件库。所有模块都是自包含、可单测的
nn.Module，不引用 QwenGR00T；由 QwenGR00T 在 ``framework.wam.guidance.enabled=true`` 时
实例化并编排。默认全关时这里的任何代码都不会被构造/调用，保证旧 WAM/原生 QwenGR00T 行为不变。

设计原则（与 plan §2 对齐）：
  - 保留原始 action 条件 M_a=[h_act; H_qwen-context]，world 信号只「增」不「删」；
  - 直接 guidance 用 passive world（o_t,l,Q_W→ẑ_{t+H}），不引入需动作输入的 FDM；
  - 提供双 query 控制组所需的 fusion / 压缩 / pooling / bridge 构件。

各构件与实验的对应关系：
  WorldTokenAdapter  —— M1.2/M1.3：把 DINO 潜变量(d_dino) 或 h_future(d_model) 投到 action DiT
                        cross-attn 维度（= Qwen hidden），并做 LN 归一化对齐尺度。
  CompactSAFusion    —— M2：只对 [h_act; h_future] 做轻量 self-attention 融合，再拼回 Qwen context。
  WorldQFormer       —— M3：把变长空间 world tokens(如 196 个 DINO patch) 压成 n_query 个 learned token。
  WorldTokenPooler   —— M6：把空间 world tokens 池化成单个全局向量，喂给 action DiT 的 AdaLN。
  mix_world_tokens   —— DIAL warm-up：按 oracle_ratio 在 oracle / predicted world 间逐样本混合。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


# ──────────────────────────────────────────────────────────────────────
#  WorldCondition：QwenGR00T 组装好后传给 action head 的统一容器
# ──────────────────────────────────────────────────────────────────────
@dataclass
class WorldCondition:
    """统一的 world 条件容器（plan §4.1）。

    tokens:        [B, N_w, D_cross] 空间 world tokens，作 action DiT 的 cross-attn memory
                   （M1.x concat 走 action memory 拼接；M4/M5 走 dual/alternate cross-attn）。
    mask:          [B, N_w] bool，True=有效（与 attention keep-mask 同口径）。None=全有效。
    global_vector: [B, D_cross] 池化后的全局 world 向量，喂 AdaLN（M6/M6+）。
    source:        "predicted" | "oracle" | "scheduled" | "none"，仅作日志/调试标记。
    """

    tokens: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    global_vector: Optional[torch.Tensor] = None
    source: str = "none"

    def is_empty(self) -> bool:
        return self.tokens is None and self.global_vector is None


# ──────────────────────────────────────────────────────────────────────
#  WorldTokenAdapter —— 投影 world 信号到 action cross-attn 维度（M1.2/M1.3）
# ──────────────────────────────────────────────────────────────────────
class WorldTokenAdapter(nn.Module):
    """把 world 信号投到 action DiT 的 cross-attn 维度。

    中文注释：world 信号有两种尺度——① DINO 潜变量（d_dino≈384，原始/标准化尺度）；
    ② h_future（Qwen hidden，d_model）。两者进入 action memory 前都要对齐到 cross_attention_dim
    （= Qwen hidden）。LN 先把不同来源拉到可比尺度，再两层 MLP 投影。in==out 时仍保留 LN+MLP，
    让 world 信号与 h_act 的统计量更接近（h_act 来自 Qwen 末层，world 来自 DINO/中间层）。
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, in_dim] → [B, N, out_dim]
        return self.proj(self.norm(x))


# ──────────────────────────────────────────────────────────────────────
#  CompactSAFusion —— [h_act; h_future] 轻量自注意力融合（M2）
# ──────────────────────────────────────────────────────────────────────
class CompactSAFusion(nn.Module):
    """对 action query 与 future query 的拼接做轻量 self-attention 融合。

    中文注释（plan §5 M2）：只对 [h_act; h_future]（都在 Qwen hidden 维）做几层 SA，让两组 query
    互通信息，再由调用方与原始 Qwen context 拼成 action memory。**不要**对全部 Qwen hidden +
    196 个 DINO token 做全量 SA（开销大且偏离「轻量」初衷）。pre-norm 残差结构。
    """

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(dim),
                        "attn": nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                        "norm2": nn.LayerNorm(dim),
                        "ff": nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)),
                    }
                )
            )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, N, dim]; key_padding_mask: [B, N] True=忽略(pad)。返回同形状融合后序列。
        for blk in self.layers:
            xn = blk["norm1"](x)
            attn_out, _ = blk["attn"](xn, xn, xn, key_padding_mask=key_padding_mask, need_weights=False)
            x = x + attn_out
            x = x + blk["ff"](blk["norm2"](x))
        return x


# ──────────────────────────────────────────────────────────────────────
#  WorldQFormer —— 空间 world tokens 压缩成固定数量 learned token（M3）
# ──────────────────────────────────────────────────────────────────────
class WorldQFormer(nn.Module):
    """用 n_query 个 learned query 通过 cross-attention 压缩变长空间 world tokens。

    中文注释（plan §5 M3）：Q-Former 只作用于**空间 DINO tokens**（不要作用于已压缩的 h_future）。
    每层 = (learned query → cross-attn 到 world tokens) + FFN，pre-norm 残差。输出 [B, n_query, out_dim]。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_query: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_query = int(n_query)
        self.query = nn.Parameter(torch.randn(self.n_query, out_dim) * 0.02)
        self.in_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm_q": nn.LayerNorm(out_dim),
                        "norm_kv": nn.LayerNorm(out_dim),
                        "attn": nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True),
                        "norm_ff": nn.LayerNorm(out_dim),
                        "ff": nn.Sequential(nn.Linear(out_dim, 4 * out_dim), nn.GELU(), nn.Linear(4 * out_dim, out_dim)),
                    }
                )
            )

    def forward(self, world: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # world: [B, N, in_dim]; key_padding_mask: [B, N] True=忽略。返回 [B, n_query, out_dim]。
        bsz = world.shape[0]
        kv = self.in_proj(world)
        q = self.query.unsqueeze(0).expand(bsz, -1, -1).to(kv.dtype)
        for blk in self.layers:
            qn = blk["norm_q"](q)
            kvn = blk["norm_kv"](kv)
            attn_out, _ = blk["attn"](qn, kvn, kvn, key_padding_mask=key_padding_mask, need_weights=False)
            q = q + attn_out
            q = q + blk["ff"](blk["norm_ff"](q))
        return q


# ──────────────────────────────────────────────────────────────────────
#  WorldTokenPooler —— 空间 world tokens 池化成全局向量（M6 AdaLN）
# ──────────────────────────────────────────────────────────────────────
class WorldTokenPooler(nn.Module):
    """learned-query attention pooling：把空间 world tokens 池化成单个全局向量。

    中文注释（plan §5 M6）：输出向量交给 action DiT 的 world_to_temb，加到 timestep embedding 上
    驱动 AdaLN。用单 learned query 的 attention pool（比 mean-pool 更能聚焦动态区域）。
    """

    def __init__(self, dim: int, out_dim: Optional[int] = None, num_heads: int = 8):
        super().__init__()
        out_dim = out_dim or dim
        self.query = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, world: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # world: [B, N, dim] → [B, out_dim]
        bsz = world.shape[0]
        q = self.query.unsqueeze(0).expand(bsz, -1, -1).to(world.dtype)
        kv = self.norm_kv(world)
        pooled, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        return self.proj(pooled.squeeze(1))


# ──────────────────────────────────────────────────────────────────────
#  bridge —— DIAL 风格 oracle/predicted 混合（warm-up Stage 1→3）
# ──────────────────────────────────────────────────────────────────────
def mix_world_tokens(
    predicted: Optional[torch.Tensor],
    oracle: Optional[torch.Tensor],
    oracle_ratio: float,
    generator: Optional[torch.Generator] = None,
) -> Optional[torch.Tensor]:
    """逐样本伯努利在 oracle / predicted world 之间选择（plan §6 scheduled bridge）。

    中文注释：oracle_ratio=1.0 全用 oracle（Stage 1 decoupled warm-up）；0.0 全用 predicted
    （Stage 2/3）；中间值按样本独立以 oracle_ratio 概率取 oracle，让 action branch 平滑适应
    预测误差。要求 predicted 与 oracle 形状一致（M1.2/M1.3 下两者都是 [B, N_patch, d_dino]，
    天然对齐）。逐样本硬选择（非特征插值），避免造出训练分布外的「半真半假」潜变量。
    """
    if predicted is None:
        return oracle
    if oracle is None or oracle_ratio <= 0.0:
        return predicted
    if oracle_ratio >= 1.0:
        return oracle
    bsz = predicted.shape[0]
    rand = torch.rand(bsz, device=predicted.device, generator=generator)
    sel = (rand < oracle_ratio).view(bsz, *([1] * (predicted.ndim - 1)))
    return torch.where(sel, oracle.to(predicted.dtype), predicted)


def linear_gradient_ramp(
    global_step: int,
    start_step: int,
    end_step: int,
    start_scale: float = 0.0,
    end_scale: float = 1.0,
) -> float:
    """Return a deterministic linear scale for action-to-world gradients.

    The forward signal is never scheduled: the action branch always consumes
    the predicted future.  Only its backward gradient into the world predictor
    is scaled, which avoids an oracle/predicted distribution switch.
    """

    step = max(int(global_step), 0)
    start = max(int(start_step), 0)
    end = max(int(end_step), start)
    lo = float(start_scale)
    hi = float(end_scale)
    if lo < 0.0 or hi < 0.0:
        raise ValueError(f"Gradient-ramp scales must be non-negative, got {lo} -> {hi}")
    if step <= start:
        return lo
    if end == start or step >= end:
        return hi
    alpha = float(step - start) / float(end - start)
    return lo + alpha * (hi - lo)


def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Keep ``x`` unchanged in forward while multiplying its backward by scale."""

    value = float(scale)
    if value < 0.0:
        raise ValueError(f"Gradient scale must be non-negative, got {value}")
    if value == 1.0:
        return x
    detached = x.detach()
    return detached + value * (x - detached)


def shuffle_along_batch(x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """batch 内打乱（plan §11 因果消融 Shuffled World）。

    中文注释：把 world 信号在 batch 维 roll/permute，使每个样本拿到「别人的未来」。若 action 真在
    用 world，shuffle 后 SR 应明显下降；否则说明 world 被忽略。保证至少错排（避免恒等置换）。
    """
    bsz = x.shape[0]
    if bsz <= 1:
        return x
    perm = torch.randperm(bsz, device=x.device, generator=generator)
    # 避免恰好恒等置换（小 batch 时概率不可忽略）
    if bool(torch.all(perm == torch.arange(bsz, device=x.device))):
        perm = torch.roll(perm, shifts=1, dims=0)
    return x[perm]
