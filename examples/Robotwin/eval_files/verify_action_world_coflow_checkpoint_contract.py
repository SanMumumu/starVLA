#!/usr/bin/env python3
"""Fail-fast audit for the opt-in RoboTwin Action--World Co-Flow ABI."""

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
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config, contract_config_path = load_checkpoint_contract_config(checkpoint)
    stats_path = find_run_file(checkpoint, "dataset_statistics.json")
    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)

    expected = {
        "framework.name": "QwenActionWorldCoFlow",
        "framework.enable_action_world_coflow": True,
        "framework.action_model.action_dim": 14,
        "framework.action_model.state_dim": 14,
        "framework.action_model.action_horizon": 32,
        "framework.action_model.prediction_type": "jit_x",
        "framework.action_model.use_correlated_noise": False,
        "framework.action_world_coflow.segment_boundaries": [16, 32],
        "framework.action_world_coflow.future_strides": [16, 32],
        "framework.action_world_coflow.freeze_qwen_vision_encoder": True,
        "datasets.vla_data.dataset_py": "lerobot_datasets",
        "datasets.vla_data.data_mix": "robotwin_fastwam",
        "datasets.vla_data.lerobot_version": "v2.0",
        "datasets.vla_data.fastwam_expected_fps": 50,
        "datasets.vla_data.fastwam_direct_frame_sampling": True,
        "datasets.vla_data.fastwam_wam_targets": False,
        "datasets.vla_data.fastwam_action_world_coflow_targets": True,
        "datasets.vla_data.fastwam_coflow_future_strides": [16, 32],
        "datasets.vla_data.include_state": True,
        "datasets.vla_data.action_type": "abs_qpos",
        "datasets.vla_data.action_mode": "abs",
        "datasets.vla_data.image_layout": "fastwam_composite",
        "datasets.vla_data.composite_source_view_keys": [
            "video.cam_high",
            "video.cam_left_wrist",
            "video.cam_right_wrist",
        ],
        "datasets.vla_data.composite_view_key": "video.robotwin_composite",
        "datasets.vla_data.obs_image_size": [320, 384],
        "datasets.vla_data.video_backend": "pyav",
    }
    errors = []
    for path, wanted in expected.items():
        try:
            actual = _get(config, path)
        except (KeyError, TypeError):
            actual = "<missing>"
        if actual != wanted:
            errors.append(f"{path}: expected {wanted!r}, got {actual!r}")

    coflow = config.get("framework", {}).get("action_world_coflow", {})
    layer_strategy = coflow.get("world_feature_layer_strategy")
    if layer_strategy not in {"evenly_spaced", "explicit", "all_layers"}:
        errors.append(f"unsupported world_feature_layer_strategy={layer_strategy!r}")
    fusion_type = coflow.get("world_layer_fusion_type")
    if fusion_type not in {"fixed_mean", "fixed_weighted_mean", "trainable_weighted_mean"}:
        errors.append(f"unsupported world_layer_fusion_type={fusion_type!r}")
    if coflow.get("world_spatial_pool_type") != "adaptive_avg_pool2d":
        errors.append(
            "world_spatial_pool_type must preserve a deterministic adaptive grid, got "
            f"{coflow.get('world_spatial_pool_type')!r}"
        )
    grid = coflow.get("world_token_grid")
    try:
        grid_values = [int(value) for value in grid] if isinstance(grid, list) else []
    except (TypeError, ValueError):
        grid_values = []
    if len(grid_values) != 2 or min(grid_values, default=0) <= 0:
        errors.append(f"world_token_grid must be two positive integers, got {grid!r}")
    elif int(coflow.get("num_world_tokens", -1)) != grid_values[0] * grid_values[1]:
        errors.append(
            f"num_world_tokens={coflow.get('num_world_tokens')!r} disagrees with world_token_grid={grid!r}"
        )
    source = coflow.get("intermediate_state_source")
    if source not in {"ground_truth", "predicted_detach", "predicted_e2e", "scheduled"}:
        errors.append(f"unsupported intermediate_state_source={source!r}")
    ratios = coflow.get("noise_plane_sampling", {})
    ratio_names = ("policy", "forward", "inverse", "joint", "diagonal")
    ratio_sum = sum(float(ratios.get(f"{name}_ratio", 0.0)) for name in ratio_names)
    if abs(ratio_sum - 1.0) > 1.0e-6:
        errors.append(f"noise_plane_sampling ratios must sum to 1, got {ratio_sum}")
    if float(ratios.get("policy_ratio", 0.0)) <= 0:
        errors.append("policy_ratio must be positive so actions train without clean future")
    if coflow.get("world_bridge_type") not in {"qantara_brownian_bridge", "qantara_linear_bridge"}:
        errors.append(f"world_bridge_type must be Qantara-aligned, got {coflow.get('world_bridge_type')!r}")
    if coflow.get("world_prediction_type") not in {"qantara_x_delta", "qantara_x"}:
        errors.append(f"unsupported world_prediction_type={coflow.get('world_prediction_type')!r}")
    if int(coflow.get("action_inference_steps", 0)) <= 0 or int(coflow.get("world_inference_steps", 0)) <= 0:
        errors.append("action/world inference steps must both be positive")

    expects_state, state_source = resolve_config_expects_state(config)
    if not expects_state:
        errors.append(f"Co-Flow checkpoint must require state, resolved from {state_source}")
    if replan_steps != 24:
        errors.append(f"replan_steps must be 24 for the FastWAM protocol, got {replan_steps}")

    if len(stats) != 1:
        errors.append(f"dataset_statistics.json must contain one embodiment, got {list(stats)}")
    else:
        tag, tag_stats = next(iter(stats.items()))
        for modality in ("state", "action"):
            modality_stats = tag_stats.get(modality, {})
            for name in ("min", "max", "mean", "std", "q01", "q99"):
                values = np.asarray(modality_stats.get(name), dtype=np.float64)
                if values.shape != (14,) or not np.isfinite(values).all():
                    errors.append(f"stats[{tag!r}].{modality}.{name} must be finite shape (14,)")
        mask = np.asarray(tag_stats.get("action", {}).get("mask"), dtype=np.bool_)
        if mask.shape != (14,) or not mask.all():
            errors.append("action z-score mask must contain 14 true values")

    data_config = ROBOT_TYPE_CONFIG_MAP.get("robotwin_fastwam")
    transforms = (
        [
            transform
            for transform in data_config.transform().transforms
            if isinstance(transform, StateActionTransform)
        ]
        if data_config is not None
        else []
    )
    if len(transforms) != 2 or any(
        set(transform.normalization_modes.values()) != {"fastwam_zscore"} for transform in transforms
    ):
        errors.append("robotwin_fastwam state/action transforms must both use fastwam_zscore")

    action_cfg = config.get("framework", {}).get("action_model", {})
    if bool(action_cfg.get("use_correlated_noise", False)):
        chol_path = find_run_file(checkpoint, "action_correlation_cholesky.npy")
        matrix = np.load(chol_path, allow_pickle=False)
        if matrix.shape != (448, 448) or not np.isfinite(matrix).all():
            errors.append(f"invalid correlated-noise Cholesky at {chol_path}")

    if errors:
        raise ValueError("Action--World Co-Flow checkpoint contract failed:\n- " + "\n- ".join(errors))
    return {
        "checkpoint": str(checkpoint),
        "contract_config": str(contract_config_path),
        "statistics": str(stats_path),
        "framework": "QwenActionWorldCoFlow",
        "chunk": 32,
        "segments": [16, 32],
        "world_grid": grid,
        "intermediate_state_source_train_only": source,
        "replan": replan_steps,
        "image": "one 320x384 three-camera composite",
        "state": "14-D z-score release order",
        "action": "14-D z-score release order",
        "eval_future_source": "predicted Z16 only (no GT future input API)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replan-steps", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(verify(args.checkpoint, args.replan_steps), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
