"""PR2: 参数 EMA (eval 用 EMA 权重, ELF ema_decay1=0.9999 同款).

说明:
- shadow 只跟踪 requires_grad 参数 (88M fp32, 显存可忽略)。
- update() 只应在"真正 optimizer.step 成功"的步调用 (梯度累积/跳步时不调),
  否则有效 decay 变成 decay^accum (ELF train_step 同款注意事项)。
- 提供 state_dict/load_state_dict 以便 accelerate register_for_checkpointing。
"""

from __future__ import annotations

import torch
from torch import nn


######### // code // ##########
# 中文注释：极简 EMA。shadow 键 = named_parameters 名（构造时以"未被 DDP 包裹"的原始模型为准，
# 这样键与 model.state_dict() 对齐，merged_state_dict 可直接被 from_pretrained strict 加载）。
class EmaModel:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            shadow = self.shadow[name]
            if shadow.device != param.device:
                shadow = shadow.to(param.device)
                self.shadow[name] = shadow
            shadow.mul_(d).add_(param.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def merged_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """完整 state_dict（含 buffer/冻结参数）+ EMA 覆盖可训练参数 → 可直接存为 *_ema.pt。"""
        full = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for name, shadow in self.shadow.items():
            full[name] = shadow.detach().cpu().clone()
        return full

    # ---- accelerate register_for_checkpointing 契约 ----
    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        loaded = state.get("shadow", {})
        for name, tensor in loaded.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor.to(self.shadow[name].device))
            else:
                self.shadow[name] = tensor.clone()
######### // code // ##########
