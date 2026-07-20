# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Self-contained building blocks for dual-query World-to-Action guidance.

The modules do not import ``QwenGR00T`` and are instantiated only when
``framework.wam.guidance.enabled=true``. Disabling guidance therefore retains
legacy WAM and native QwenGR00T behavior.

The original action memory ``[h_act; qwen_context]`` is always retained. World
features are additive conditions, and the passive predictor avoids an
action-conditioned ``action -> world -> action`` shortcut. The module provides
the projection, compact fusion, Q-Former, pooling, and scheduled-bridge
components used by the M0--M6+ ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
@dataclass
class WorldCondition:
    """Canonical world condition passed to the action branch.

    ``tokens`` are spatial cross-attention memory, ``mask`` is a boolean keep
    mask, and ``global_vector`` is the pooled AdaLN condition. ``source`` is
    diagnostic metadata and does not alter computation.
    """

    tokens: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    global_vector: Optional[torch.Tensor] = None
    source: str = "none"

    def is_empty(self) -> bool:
        return self.tokens is None and self.global_vector is None


# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
class WorldTokenAdapter(nn.Module):
    """Normalize and project world features into the action cross-attention space.

    The LayerNorm and MLP remain active even when input and output dimensions
    match so DINO, intermediate-Qwen, and final-Qwen feature statistics can be
    aligned explicitly.
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
# ──────────────────────────────────────────────────────────────────────
class CompactSAFusion(nn.Module):
    """Fuse ``[h_act; h_future]`` with a compact pre-norm attention stack.

    Only the two query groups participate; raw Qwen context and dense DINO
    tokens stay outside this intentionally lightweight fusion module.
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
        for blk in self.layers:
            xn = blk["norm1"](x)
            attn_out, _ = blk["attn"](xn, xn, xn, key_padding_mask=key_padding_mask, need_weights=False)
            x = x + attn_out
            x = x + blk["ff"](blk["norm2"](x))
        return x


# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
class WorldQFormer(nn.Module):
    """Compress spatial world tokens with learned cross-attention queries.

    The Q-Former operates on dense DINO tokens, not the already compact
    ``h_future`` representation, and returns ``[B, n_query, out_dim]``.
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
# ──────────────────────────────────────────────────────────────────────
class WorldTokenPooler(nn.Module):
    """Pool spatial world tokens into one learned-query global vector.

    The action DiT maps this vector into its timestep embedding for AdaLN
    conditioning. Learned attention can focus on dynamic regions more readily
    than an unconditional mean.
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
# ──────────────────────────────────────────────────────────────────────
def mix_world_tokens(
    predicted: Optional[torch.Tensor],
    oracle: Optional[torch.Tensor],
    oracle_ratio: float,
    generator: Optional[torch.Generator] = None,
) -> Optional[torch.Tensor]:
    """Select oracle or predicted world tokens independently per sample.

    A ratio of one implements decoupled oracle warmup; zero uses predictions
    exclusively. Intermediate ratios perform hard sample-level selection, not
    feature interpolation, so no out-of-distribution hybrid latent is created.
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
    """Shuffle world conditions across samples for the causal-use ablation.

    The permutation is forced to be non-identity so each sample receives an
    incorrect future whenever the batch contains more than one sample.
    """
    bsz = x.shape[0]
    if bsz <= 1:
        return x
    perm = torch.randperm(bsz, device=x.device, generator=generator)
    if bool(torch.all(perm == torch.arange(bsz, device=x.device))):
        perm = torch.roll(perm, shifts=1, dims=0)
    return x[perm]
