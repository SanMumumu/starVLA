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
# Keep this verifier usable in the RoboTwin client image.  Importing the
# training-time data registry here pulls StateActionTransform and therefore
# PyTorch3D, even though evaluation never uses rotation transforms.  The
# checkpoint's exact data_mix, state/action statistics, and deployment ABI are
# checked below without importing any training-only dataloader modules.


def _get(config: dict, path: str):
    value = config
    for key in path.split("."):
        value = value[key]
    return value


def verify(
    checkpoint: Path,
    replan_steps: int,
    *,
    inference_mode: str = "diagonal",
    inference_horizon: int | None = None,
) -> dict:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config, contract_config_path = load_checkpoint_contract_config(checkpoint)
    stats_path = find_run_file(checkpoint, "dataset_statistics.json")
    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)

    action_cfg = config.get("framework", {}).get("action_model", {})
    coflow = config.get("framework", {}).get("action_world_coflow", {})
    data_cfg = config.get("datasets", {}).get("vla_data", {})
    try:
        checkpoint_horizon = int(action_cfg.get("action_horizon"))
    except (TypeError, ValueError):
        checkpoint_horizon = -1
    requested_horizon = checkpoint_horizon if inference_horizon is None else int(inference_horizon)
    inference_mode = str(inference_mode).lower()

    expected = {
        "framework.name": "QwenActionWorldCoFlow",
        "framework.enable_action_world_coflow": True,
        "framework.action_model.action_dim": 14,
        "framework.action_model.state_dim": 14,
        "framework.action_model.prediction_type": "jit_x",
        "framework.action_model.use_correlated_noise": False,
        "framework.action_world_coflow.freeze_qwen_vision_encoder": True,
        "datasets.vla_data.dataset_py": "lerobot_datasets",
        "datasets.vla_data.lerobot_version": "v2.0",
        "datasets.vla_data.fastwam_expected_fps": 50,
        "datasets.vla_data.fastwam_direct_frame_sampling": True,
        "datasets.vla_data.fastwam_wam_targets": False,
        "datasets.vla_data.fastwam_action_world_coflow_targets": True,
        "datasets.vla_data.data_mix": "robotwin_fastwam_h16",
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

    if checkpoint_horizon != 16:
        errors.append(f"single-bridge Co-Flow requires action_horizon=16, got {checkpoint_horizon}")
    removed_multibridge_keys = {
        "segment_boundaries",
        "future_strides",
        "z32_loss_weight",
        "intermediate_state_source",
        "predicted_z16_detach",
        "predicted_action_prefix_detach",
        "z16_teacher_ratio_start",
        "z16_teacher_ratio_end",
        "z16_teacher_decay_steps",
    }
    present_removed = sorted(key for key in removed_multibridge_keys if key in coflow)
    if present_removed:
        errors.append(f"removed multi-bridge fields are present: {present_removed}")
    if "fastwam_coflow_future_strides" in data_cfg:
        errors.append("datasets fastwam_coflow_future_strides was removed; t+16 is fixed")
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
    if inference_mode == "diagonal" and int(coflow.get("action_inference_steps", 0)) != int(
        coflow.get("world_inference_steps", 0)
    ):
        errors.append("diagonal inference requires equal action/world inference steps")

    expects_state, state_source = resolve_config_expects_state(config)
    if not expects_state:
        errors.append(f"Co-Flow checkpoint must require state, resolved from {state_source}")
    if inference_mode not in {"policy", "diagonal"}:
        errors.append(f"inference_mode must be policy or diagonal, got {inference_mode!r}")
    if requested_horizon != 16:
        errors.append(f"single-bridge inference_horizon must be 16, got {requested_horizon}")
    if not 1 <= int(replan_steps) <= requested_horizon:
        errors.append(
            f"replan_steps must be in [1,{requested_horizon}] for the returned action chunk, "
            f"got {replan_steps}"
        )

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

    if bool(action_cfg.get("use_correlated_noise", False)):
        chol_path = find_run_file(checkpoint, "action_correlation_cholesky.npy")
        matrix = np.load(chol_path, allow_pickle=False)
        flat_action_dim = checkpoint_horizon * 14
        if matrix.shape != (flat_action_dim, flat_action_dim) or not np.isfinite(matrix).all():
            errors.append(f"invalid correlated-noise Cholesky at {chol_path}")

    if errors:
        raise ValueError("Action--World Co-Flow checkpoint contract failed:\n- " + "\n- ".join(errors))
    return {
        "checkpoint": str(checkpoint),
        "contract_config": str(contract_config_path),
        "statistics": str(stats_path),
        "framework": "QwenActionWorldCoFlow",
        "checkpoint_chunk": checkpoint_horizon,
        "bridge_horizon": 16,
        "world_grid": grid,
        "replan": replan_steps,
        "inference_mode": inference_mode,
        "inference_horizon": requested_horizon,
        "image": "one 320x384 three-camera composite",
        "state": "14-D z-score release order",
        "action": "14-D z-score release order",
        "eval_path": (
            "policy edge: tau_action 0->1, tau_world=0; return actions only"
            if inference_mode == "policy"
            else "diagonal: tau_action=tau_world 0->1; jointly evolve actions and future state"
        ),
        "eval_future_source": "model-predicted only (no GT future input API)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument("--inference-mode", choices=("policy", "diagonal"), default="diagonal")
    parser.add_argument("--inference-horizon", type=int, default=16)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.checkpoint,
                args.replan_steps,
                inference_mode=args.inference_mode,
                inference_horizon=args.inference_horizon,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
