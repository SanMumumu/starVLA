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

    Derived RoboTwin clean/randomized aliases must also be registered globally
    because deployment resolves robot types through ``DATASET_NAMED_MIXTURES``
    rather than this module's fallback resolver.
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
