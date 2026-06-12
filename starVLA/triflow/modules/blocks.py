"""PR2: TriFlow 模态块嵌入器 (BlockEmbedder).

复用:
- starVLA.triflow.modules.layers.TimestepEmbedder (ELF 同款时间嵌入)

职责 (单一入口管理所有模态相关参数, 便于锚点/覆盖断言枚举):
- E             从头学的文本 embedding 表 [vocab, d_text], 查表行 RMS 归一 (防塌缩/防漂移)
- in_proj_*     各模态 latent → 塔 hidden (V:384→D 与 V' 共享; L:d_text→D; A:act_dim→D)
- out_proj_*    塔 hidden → 各模态 x0 预测 (零初始化, ELF FinalLayer 纪律; V 从不做目标)
- dec_proj/gain L 解码分支: h → GELU → d_text, 与归一化 E 表 tied 出 logits
- pos_patch     [n_patches,D] 各视角与 V' 共享空间先验; view_emb 区分当前帧多视角
- pos_text/pos_action/type_emb(4)  learned, N(0,0.02)
- time_embedder 每 token 加所属块的 t 嵌入 (干净块 t=1) —— 支持多目标块独立 t
- stride_embedder V' 块的 k 步预测间隔条件 (nwm rel-time 思路)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from starVLA.triflow.modules.layers import TimestepEmbedder


BLOCK_KIND_TO_TYPE_ID = {"v": 0, "l": 1, "a": 2, "vf": 3}


######### // code // ##########
# 中文注释：模态块嵌入器。embed_block 输入各模态 latent（干净或加噪后的 z）：
#   v  [B, V*P, d_dino]   (V 视角已展平; P=n_patches)
#   l  [B, L_max, d_text]
#   a  [B, H, action_dim]
#   vf [B, P, d_dino]
# 输出 [B, n, hidden]：in_proj(z) + pos + type + time(t)（vf 另加 stride 嵌入）。
class BlockEmbedder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_text: int = 256,
        max_text_len: int = 32,
        d_dino: int = 384,
        n_patches: int = 196,
        max_views: int = 4,
        action_dim: int = 7,
        action_horizon: int = 8,
        hidden_size: int = 768,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_text = int(d_text)
        self.max_text_len = int(max_text_len)
        self.d_dino = int(d_dino)
        self.n_patches = int(n_patches)
        self.max_views = int(max_views)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.hidden_size = int(hidden_size)

        # --- 文本 embedding 表（从头学；查表时行 RMS 归一到单位 RMS） ---
        self.E = nn.Embedding(self.vocab_size, self.d_text)

        # --- 各模态输入投影 ---
        self.in_proj_v = nn.Linear(self.d_dino, self.hidden_size)
        self.in_proj_l = nn.Linear(self.d_text, self.hidden_size)
        self.in_proj_a = nn.Linear(self.action_dim, self.hidden_size)

        # --- 各模态 x0 输出投影（零初始化；塔尾已有 RMSNorm） ---
        self.out_proj_v = nn.Linear(self.hidden_size, self.d_dino)
        self.out_proj_l = nn.Linear(self.hidden_size, self.d_text)
        self.out_proj_a = nn.Linear(self.hidden_size, self.action_dim)

        # --- L 解码分支（tied unembedding 用归一化 E 表） ---
        self.dec_proj = nn.Linear(self.hidden_size, self.d_text)
        self.dec_gain = nn.Parameter(torch.tensor(1.0))

        # --- learned 位置/视角/类型嵌入 ---
        self.pos_patch = nn.Embedding(self.n_patches, self.hidden_size)
        self.view_emb = nn.Embedding(self.max_views, self.hidden_size)
        self.pos_text = nn.Embedding(self.max_text_len, self.hidden_size)
        self.pos_action = nn.Embedding(self.action_horizon, self.hidden_size)
        self.type_emb = nn.Embedding(len(BLOCK_KIND_TO_TYPE_ID), self.hidden_size)

        # --- 时间 / stride 嵌入 ---
        self.time_embedder = TimestepEmbedder(self.hidden_size)
        self.stride_embedder = TimestepEmbedder(self.hidden_size)

        self._init_weights()

    def _init_weights(self) -> None:
        # 文本表 N(0,0.02)（查表后行 RMS 归一，初始幅值不重要）
        nn.init.normal_(self.E.weight, std=0.02)
        # 位置/视角/类型嵌入 N(0,0.2)：内容路径经 in_proj 后 RMS≈0.7~1.4，
        # 若按惯例 0.02 初始化，位置信号只有内容的 ~2%，文本去噪会"知道词袋排不对位"
        # （v2l 过拟合实测卡 0.2 平台、输出乱序）。提到同量级的 ~20% 后位置可分辨，
        # 仍可学习。ELF 原版靠 RoPE 回避此问题（旋转编码不与内容幅值竞争）。
        for emb in (self.pos_patch, self.view_emb, self.pos_text, self.pos_action, self.type_emb):
            nn.init.normal_(emb.weight, std=0.2)
        # 输入/解码投影 xavier
        for lin in (self.in_proj_v, self.in_proj_l, self.in_proj_a, self.dec_proj):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        # 输出投影零初始化（x0 预测从 0 出发 → v 初值有界）
        for lin in (self.out_proj_v, self.out_proj_l, self.out_proj_a):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        # TimestepEmbedder 构造时已按 N(0,0.02) 自初始化，这里不再覆盖。

    # ---- 文本 latent：查表 + 行 RMS 归一（结构性防塌缩） ----
    @staticmethod
    def _rms_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def embed_text_x0(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [B,L] → 单位 RMS 文本 latent [B,L,d_text]。"""
        return self._rms_normalize(self.E(ids))

    def normalized_embedding_table(self) -> torch.Tensor:
        """归一化 E 表 Ê [vocab, d_text]（tied unembedding 与解码端共用）。"""
        return self._rms_normalize(self.E.weight)

    # ---- 块嵌入 ----
    def embed_block(
        self,
        kind: str,
        z: torch.Tensor,
        t: torch.Tensor,
        stride01: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, n_tokens, _ = z.shape
        if kind == "v":
            assert n_tokens % self.n_patches == 0, f"v block tokens={n_tokens} 不是 n_patches={self.n_patches} 整数倍"
            views = n_tokens // self.n_patches
            assert views <= self.max_views, f"views={views} 超过 max_views={self.max_views}"
            proj = self.in_proj_v(z)
            patch_ids = torch.arange(self.n_patches, device=z.device).repeat(views)
            view_ids = torch.repeat_interleave(torch.arange(views, device=z.device), self.n_patches)
            pos = self.pos_patch(patch_ids) + self.view_emb(view_ids)
        elif kind == "vf":
            assert n_tokens == self.n_patches, f"vf block tokens={n_tokens} != n_patches={self.n_patches}"
            proj = self.in_proj_v(z)
            pos = self.pos_patch(torch.arange(self.n_patches, device=z.device))
        elif kind == "l":
            assert n_tokens <= self.max_text_len
            proj = self.in_proj_l(z)
            pos = self.pos_text(torch.arange(n_tokens, device=z.device))
        elif kind == "a":
            assert n_tokens <= self.action_horizon
            proj = self.in_proj_a(z)
            pos = self.pos_action(torch.arange(n_tokens, device=z.device))
        else:
            raise ValueError(f"Unknown block kind `{kind}`")

        type_id = torch.full((1,), BLOCK_KIND_TO_TYPE_ID[kind], device=z.device, dtype=torch.long)
        tok = proj + pos.unsqueeze(0) + self.type_emb(type_id).reshape(1, 1, -1)
        tok = tok + self.time_embedder(t.to(z.device)).reshape(bsz, 1, -1)
        if kind == "vf" and stride01 is not None:
            tok = tok + self.stride_embedder(stride01.to(z.device)).reshape(bsz, 1, -1)
        return tok

    # ---- 输出投影（x0 预测） ----
    def project_out(self, kind: str, hidden: torch.Tensor) -> torch.Tensor:
        if kind == "vf":
            return self.out_proj_v(hidden)
        if kind == "l":
            return self.out_proj_l(hidden)
        if kind == "a":
            return self.out_proj_a(hidden)
        raise ValueError(f"Block kind `{kind}` is never a generation target (v is condition-only)")

    # ---- L 解码分支 logits（tied） ----
    def decoder_logits(self, hidden_l: torch.Tensor) -> torch.Tensor:
        u = F.gelu(self.dec_proj(hidden_l))                       # [B,L,d_text]
        table = self.normalized_embedding_table()                 # [vocab,d_text]
        return self.dec_gain * (u @ table.t()) / math.sqrt(self.d_text)
######### // code // ##########
