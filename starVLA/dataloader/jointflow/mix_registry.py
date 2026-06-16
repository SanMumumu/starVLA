"""JointFlow-local dataset mixture aliases."""

from __future__ import annotations

from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES


JOINTFLOW_NAMED_MIXTURES = {
    "libero_object": [("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka")],
    "libero_goal": [("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka")],
    "libero_spatial": [("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka")],
    "libero_10": [("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka")],
}


######### // code // ##########
# 中文注释：RoboTwin 按 split 的派生混合。examples/Robotwin data_registry 注册的
# robotwin_all 用 "Clean/<task>"、"Randomized/<task>" 前缀（data_root 指 RoboTwin 根，
# 与本仓配置一致）；而无前缀的 robotwin/robotwin_task1 需要 root 指到单个 split 目录。
# 这里从 robotwin_all 惰性派生（避免硬编码 50 任务清单随上游漂移）：
#   robotwin_clean / robotwin_randomized   单 split 全部 50 任务
#   robotwin_clean_task1                   单任务调试用
def _derived_robotwin_mixtures() -> dict:
    base = DATASET_NAMED_MIXTURES.get("robotwin_all", [])
    clean = [entry for entry in base if str(entry[0]).startswith("Clean/")]
    randomized = [entry for entry in base if str(entry[0]).startswith("Randomized/")]
    derived = {}
    if clean:
        derived["robotwin_clean"] = clean
        derived["robotwin_clean_task1"] = clean[:1]
    if randomized:
        derived["robotwin_randomized"] = randomized
    return derived
######### // code // ##########


def register_jointflow_mixtures() -> None:
    """Expose JointFlow-local aliases to code that only reads the global registry.

    中文注释：派生的 robotwin_clean/randomized/clean_task1 也注册进全局表——
    eval server 的 _resolve_robot_type 只查 DATASET_NAMED_MIXTURES，不走本模块的
    resolve_data_mix 兜底。
    """
    for name, mixture in JOINTFLOW_NAMED_MIXTURES.items():
        DATASET_NAMED_MIXTURES.setdefault(name, mixture)
    for name, mixture in _derived_robotwin_mixtures().items():
        DATASET_NAMED_MIXTURES.setdefault(name, mixture)


def resolve_data_mix(data_mix: str):
    if data_mix in DATASET_NAMED_MIXTURES:
        return DATASET_NAMED_MIXTURES[data_mix]
    if data_mix in JOINTFLOW_NAMED_MIXTURES:
        return JOINTFLOW_NAMED_MIXTURES[data_mix]
    derived = _derived_robotwin_mixtures()
    if data_mix in derived:
        return derived[data_mix]
    known = sorted(set(DATASET_NAMED_MIXTURES) | set(JOINTFLOW_NAMED_MIXTURES) | set(derived))
    raise KeyError(f"Unknown data_mix={data_mix!r}. Known mixtures include: {known}")
