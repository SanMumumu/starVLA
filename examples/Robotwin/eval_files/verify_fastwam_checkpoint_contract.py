#!/usr/bin/env python3
"""Fail-fast audit for StarVLA checkpoints trained with FastWAM's ABI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.model_server.checkpoint_contract import (  # noqa: E402
    find_run_file,
    load_checkpoint_contract_config,
    resolve_config_expects_state,
)
from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP  # noqa: E402
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionTransform  # noqa: E402


def _get(config: dict, path: str):
    value = config
    for key in path.split("."):
        value = value[key]
    return value


def verify(checkpoint: Path, replan_steps: int) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config, contract_config_path = load_checkpoint_contract_config(checkpoint)
    accessed_config_path = find_run_file(checkpoint, "config.yaml")
    stats_path = find_run_file(checkpoint, "dataset_statistics.json")
    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)

    expected = {
        "framework.name": "QwenGR00T",
        "framework.action_model.action_dim": 14,
        "framework.action_model.state_dim": 14,
        "framework.action_model.action_horizon": 32,
        "datasets.vla_data.dataset_py": "lerobot_datasets",
        "datasets.vla_data.data_mix": "robotwin_fastwam",
        "datasets.vla_data.fastwam_expected_fps": 50,
        "datasets.vla_data.fastwam_val_fraction": 0.01,
        "datasets.vla_data.fastwam_split": "train",
        "datasets.vla_data.fastwam_split_seed": 42,
        "datasets.vla_data.fastwam_direct_frame_sampling": True,
        "datasets.vla_data.action_mode": "abs",
        "datasets.vla_data.obs_image_size": [320, 384],
    }
    errors = []
    for path, wanted in expected.items():
        try:
            actual = _get(config, path)
        except (KeyError, TypeError):
            actual = "<missing>"
        if actual != wanted:
            errors.append(f"{path}: expected {wanted!r}, got {actual!r}")

    expects_state, state_contract_source = resolve_config_expects_state(config)
    if replan_steps != 24:
        errors.append(f"replan_steps must be 24 for this FastWAM evaluation, got {replan_steps}")
    if not 1 <= replan_steps <= 32:
        errors.append(f"replan_steps must lie in [1,32], got {replan_steps}")

    if len(stats) != 1:
        errors.append(f"dataset_statistics.json must contain one embodiment, got {list(stats)}")
    else:
        tag, tag_stats = next(iter(stats.items()))
        required_modalities = ("state", "action") if expects_state else ("action",)
        for modality in required_modalities:
            modality_stats = tag_stats.get(modality, {})
            for name in ("min", "max", "mean", "std", "q01", "q99"):
                values = np.asarray(modality_stats.get(name), dtype=np.float64)
                if values.shape != (14,) or not np.isfinite(values).all():
                    errors.append(f"stats[{tag!r}].{modality}.{name} must be finite shape (14,), got {values.shape}")
        mask = np.asarray(tag_stats.get("action", {}).get("mask"), dtype=np.bool_)
        if mask.shape != (14,) or not mask.all():
            errors.append(f"stats[{tag!r}].action.mask must contain 14 true z-score dimensions")

    action_cfg = config.get("framework", {}).get("action_model", {})
    if bool(action_cfg.get("use_correlated_noise", False)):
        cholesky = accessed_config_path.parent / "action_correlation_cholesky.npy"
        if not cholesky.is_file():
            errors.append(f"correlated checkpoint is missing {cholesky}")
        else:
            matrix = np.load(cholesky, allow_pickle=False)
            if matrix.shape != (448, 448) or not np.isfinite(matrix).all():
                errors.append(f"invalid correlated-noise Cholesky: shape={matrix.shape}, path={cholesky}")

    # Z-score is implemented by the robot data registry rather than a YAML
    # scalar.  Check the exact transforms used by both training and server
    # un-normalization so a stale uploaded registry cannot silently fall back
    # to min-max/binary normalization.
    data_config = ROBOT_TYPE_CONFIG_MAP.get("robotwin_fastwam")
    if data_config is None:
        errors.append("robotwin_fastwam data registry is missing")
    else:
        transforms = [
            transform
            for transform in data_config.transform().transforms
            if isinstance(transform, StateActionTransform)
        ]
        if len(transforms) != 2 or any(
            set(transform.normalization_modes.values()) != {"fastwam_zscore"}
            for transform in transforms
        ):
            errors.append("robotwin_fastwam state/action transforms must both use fastwam_zscore")

    if errors:
        raise ValueError("FastWAM checkpoint contract failed:\n- " + "\n- ".join(errors))
    return {
        "checkpoint": str(checkpoint),
        "config": str(accessed_config_path),
        "contract_config": str(contract_config_path),
        "statistics": str(stats_path),
        "chunk": 32,
        "replan": replan_steps,
        "image": "one 320x384 composite",
        "state": "14-D release order" if expects_state else "disabled (request state is omitted)",
        "expects_state": expects_state,
        "state_contract_source": state_contract_source,
        "action": "14-D release order",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replan-steps", type=int, default=24)
    args = parser.parse_args()
    summary = verify(args.checkpoint, args.replan_steps)
    print("FastWAM checkpoint contract PASS")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
