"""CED C5: Δ 可辨识性 probe 与 policy 对齐投影。

复用: 无（纯 nn 小模块）。
所属: starVLA.jointflow.framework.qwen_joint_flow 的 effect 任务。
"""

from __future__ import annotations

import torch
from torch import nn


######### // code // ##########
# 中文注释：CedProbeMLP——从 Δ 池化向量 d_vec [B, d_dino] 回归动作 chunk a.flatten(1) [B, H·D]。
# 2 层 MLP，参数量 << 1M；它是"可辨识性引擎"的读出端，梯度刻意穿透 d_vec 回传到
# backbone/visual head（迫使 FDM 真正依赖动作）。
class CedProbeMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))

    def forward(self, d_vec: torch.Tensor) -> torch.Tensor:
        return self.net(d_vec)


# 中文注释：CedProjA——policy 路径 h_action 池化向量 [B, D_qwen] → Δ 空间 [B, d_dino] 的线性投影，
# 供 loss_align 的 cosine 对齐（teacher 侧 detach，学生侧只动 proj_A + policy 路径）。
class CedProjA(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, h_pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(h_pooled)
######### // code // ##########
