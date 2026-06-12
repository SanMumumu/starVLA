"""PR2: TriFlow 任务注册表.

任务 = (cond 干净块集合, target 加噪块集合)。块种类 ∈ {v, l, a, vf}，
序列按固定 canonical 顺序 [v, l, a, vf] 拼接（任务只决定块的有无与噪声标志）。
config 驱动 (framework.tasks.specs)，新任务 = 加一行 yaml。
"""

from __future__ import annotations

from dataclasses import dataclass


BLOCK_ORDER = ("v", "l", "a", "vf")


######### // code // ##########
# 中文注释：单个任务规格。blocks = cond ∪ target 按 canonical 顺序；noisy_flags 与 blocks 对齐。
@dataclass(frozen=True)
class TaskSpec:
    name: str
    cond: tuple[str, ...]
    target: tuple[str, ...]

    @property
    def blocks(self) -> tuple[str, ...]:
        present = set(self.cond) | set(self.target)
        return tuple(k for k in BLOCK_ORDER if k in present)

    @property
    def noisy_flags(self) -> tuple[bool, ...]:
        target = set(self.target)
        return tuple(k in target for k in self.blocks)


def build_task_registry(tasks_cfg) -> dict[str, TaskSpec]:
    """解析 framework.tasks.specs → {name: TaskSpec}，并做静态校验。"""
    specs_cfg = tasks_cfg.get("specs", None)
    if specs_cfg is None:
        raise ValueError("framework.tasks.specs missing; TriFlow tasks are config-driven.")

    registry: dict[str, TaskSpec] = {}
    for name in specs_cfg:
        node = specs_cfg[name]
        cond = tuple(str(k) for k in node.get("cond", []))
        target = tuple(str(k) for k in node.get("target", []))
        for kind in cond + target:
            if kind not in BLOCK_ORDER:
                raise ValueError(f"task `{name}`: unknown block kind `{kind}` (expect {BLOCK_ORDER})")
        if not target:
            raise ValueError(f"task `{name}`: target must be non-empty")
        if set(cond) & set(target):
            raise ValueError(f"task `{name}`: cond and target overlap: {set(cond) & set(target)}")
        if "v" in target:
            raise ValueError(f"task `{name}`: `v` (当前观测) 只能做条件，未来帧请用 `vf`")
        registry[str(name)] = TaskSpec(name=str(name), cond=cond, target=target)
    if not registry:
        raise ValueError("framework.tasks.specs is empty")
    return registry
######### // code // ##########
