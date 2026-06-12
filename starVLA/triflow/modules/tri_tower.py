"""PR2: TriFlow 单塔 transformer.

复用:
- starVLA.triflow.modules.layers (TriBlock / RMSNorm / init_xavier_)

说明:
N 层 pre-norm 双向块 + 输出 RMSNorm（ELF FinalLayer 的 norm 部分放在塔尾，
零初始化的 per-modality 输出投影放在 BlockEmbedder.project_out —— 等价拆分）。
"""

from __future__ import annotations

import torch
from torch import nn

from starVLA.triflow.modules.layers import RMSNorm, TriBlock, init_xavier_


######### // code // ##########
# 中文注释：单塔主干。输入 tokens [B,T,D] + attn_bias [B,1,T,T] → hidden [B,T,D]（已过输出 RMSNorm）。
class TriFlowTower(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.blocks = nn.ModuleList(
            [
                TriBlock(self.hidden_size, int(num_heads), mlp_ratio=float(mlp_ratio), qk_norm=bool(qk_norm), dropout=float(dropout))
                for _ in range(int(depth))
            ]
        )
        self.norm_out = RMSNorm(self.hidden_size)
        init_xavier_(self)

    def forward(self, tokens: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        x = tokens
        for block in self.blocks:
            x = block(x, attn_bias)
        return self.norm_out(x)
######### // code // ##########
