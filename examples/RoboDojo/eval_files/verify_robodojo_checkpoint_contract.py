#!/usr/bin/env python3
"""Fail-fast validation for a RoboDojo StarVLA checkpoint and its run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from starVLA.dataloader.action_correlation import validate_action_correlation_cholesky


EXPECTED_SOURCE_VIEWS = [
    "video.cam_high",
    "video.cam_left_wrist",
    "video.cam_right_wrist",
]


def _run_dir(checkpoint: Path) -> Path:
    for parent in checkpoint.parents:
        if (parent / "config.yaml").is_file():
            return parent
    raise FileNotFoundError(f"Could not find config.yaml above checkpoint: {checkpoint}")


def _expect(actual, expected, field: str) -> None:
    if actual != expected:
        raise ValueError(f"RoboDojo checkpoint {field}={actual!r}; expected {expected!r}.")


def _check_modality_stats(stats: dict, modality: str) -> None:
    modality_stats = stats.get(modality)
    if not isinstance(modality_stats, dict):
        raise ValueError(f"dataset_statistics new_embodiment.{modality} is missing or invalid.")
    for name in ("mean", "std"):
        values = np.asarray(modality_stats.get(name), dtype=np.float64)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError(
                f"dataset_statistics new_embodiment.{modality}.{name} must be finite 14-D, "
                f"got shape={values.shape}."
            )


def verify(checkpoint_path: str) -> dict:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RoboDojo checkpoint does not exist: {checkpoint}")
    if checkpoint.suffix not in {".pt", ".safetensors"}:
        raise ValueError(f"Unsupported checkpoint suffix: {checkpoint.suffix}")

    run_dir = _run_dir(checkpoint)
    accessed_config = run_dir / "config.yaml"
    full_config = run_dir / "config.full.yaml"
    config_path = full_config if full_config.is_file() else accessed_config
    stats_path = run_dir / "dataset_statistics.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing dataset statistics beside checkpoint: {stats_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with stats_path.open("r", encoding="utf-8") as handle:
        all_stats = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Checkpoint config is not a mapping: {config_path}")
    if not isinstance(all_stats, dict):
        raise TypeError(f"Checkpoint dataset statistics are not a mapping: {stats_path}")

    framework = config.get("framework") or {}
    action = framework.get("action_model") or {}
    data = ((config.get("datasets") or {}).get("vla_data") or {})
    framework_name = str(framework.get("name", ""))
    if framework_name not in {"QwenGR00T", "QwenWorldActionMoT"}:
        raise ValueError(
            "RoboDojo checkpoint framework.name="
            f"{framework_name!r}; expected QwenGR00T or QwenWorldActionMoT."
        )
    _expect(int(action.get("action_dim", -1)), 14, "framework.action_model.action_dim")
    _expect(int(action.get("action_horizon", -1)), 16, "framework.action_model.action_horizon")
    if framework_name == "QwenWorldActionMoT":
        mot = framework.get("world_action_mot") or {}
        _expect(
            bool(framework.get("enable_world_action_mot", False)),
            True,
            "framework.enable_world_action_mot",
        )
        if str(mot.get("architecture", "legacy")).lower() == "causal_dino_mot":
            if str(mot.get("interaction_mode", "")).lower() not in {"base", "joint"}:
                raise ValueError(
                    "framework.world_action_mot.interaction_mode must be "
                    "'base' or 'joint'"
                )
            _expect(
                mot.get("world_attention_mask_mode"),
                "first_frame_causal",
                "framework.world_action_mot.world_attention_mask_mode",
            )
            for field, expected in (
                ("world_hidden_size", 512),
                ("action_hidden_size", 1024),
                ("world_ffn_dim", 2048),
                ("action_ffn_dim", 4096),
                ("num_layers", 30),
                ("num_attention_heads", 24),
                ("attention_head_dim", 128),
                ("time_frequency_dim", 256),
                ("world_grid_height", 12),
                ("world_grid_width", 10),
                ("world_num_train_timesteps", 1000),
                ("action_num_train_timesteps", 1000),
            ):
                _expect(
                    int(mot.get(field, -1)),
                    expected,
                    f"framework.world_action_mot.{field}",
                )
            for field, expected in (
                ("norm_eps", 1.0e-6),
                ("world_train_shift", 5.0),
                ("world_infer_shift", 5.0),
                ("action_train_shift", 5.0),
                ("action_infer_shift", 5.0),
            ):
                _expect(
                    float(mot.get(field, -1.0)),
                    expected,
                    f"framework.world_action_mot.{field}",
                )
        else:
            _expect(
                mot.get("attention_pattern"),
                "alternating_condition_joint",
                "framework.world_action_mot.attention_pattern",
            )
        _expect(int(action.get("state_dim", -1)), 14, "framework.action_model.state_dim")
        _expect(bool(data.get("include_state", False)), True, "datasets.vla_data.include_state")
        _expect(bool(data.get("online_dino", False)), True, "datasets.vla_data.online_dino")
        _expect(bool(data.get("decode_future_video", False)), True, "datasets.vla_data.decode_future_video")
    else:
        # Preserve the historical baseline/WAM checkpoint contract verbatim.
        _expect(int(action.get("state_dim", -1)), 14, "framework.action_model.state_dim")
        _expect(bool(data.get("include_state", False)), True, "datasets.vla_data.include_state")
    data_mix = data.get("data_mix")
    allowed_data_mixes = {"robodojo_v21", "robodojo_v21_language_optional"}
    if data_mix not in allowed_data_mixes:
        raise ValueError(
            "RoboDojo checkpoint datasets.vla_data.data_mix="
            f"{data_mix!r}; expected one of {sorted(allowed_data_mixes)!r}."
        )
    composite_contracts = {
        ("fastwam_composite", "video.fastwam_composite"),
        ("tri_view_composite", "video.tri_view_composite"),
    }
    composite_contract = (
        data.get("image_layout"),
        data.get("composite_view_key"),
    )
    if composite_contract not in composite_contracts:
        raise ValueError(
            "RoboDojo checkpoint composite layout/key mismatch: "
            f"got {composite_contract!r}, expected one of "
            f"{sorted(composite_contracts)!r}"
        )
    _expect(
        list(data.get("composite_source_view_keys") or []),
        EXPECTED_SOURCE_VIEWS,
        "datasets.vla_data.composite_source_view_keys",
    )
    _expect(list(data.get("obs_image_size") or []), [320, 384], "datasets.vla_data.obs_image_size")
    _expect(data.get("action_type"), "abs_qpos", "datasets.vla_data.action_type")

    embodiment_stats = all_stats.get("new_embodiment")
    if not isinstance(embodiment_stats, dict):
        raise ValueError(
            "dataset_statistics.json must contain top-level key 'new_embodiment'; "
            f"available={sorted(all_stats)}."
        )
    _check_modality_stats(embodiment_stats, "state")
    _check_modality_stats(embodiment_stats, "action")

    uses_correlated_noise = bool(action.get("use_correlated_noise", False))
    correlation_path = run_dir / "action_correlation_cholesky.npy"
    if uses_correlated_noise:
        if not correlation_path.is_file():
            raise FileNotFoundError(
                "Checkpoint declares correlated noise but its Cholesky artifact is missing: "
                f"{correlation_path}"
            )
        matrix = validate_action_correlation_cholesky(
            np.load(correlation_path, allow_pickle=False),
            expected_size=16 * 14,
        )
        correlation_shape = list(matrix.shape)
    else:
        correlation_shape = None

    summary = {
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "contract_config": str(config_path),
        "framework": framework_name,
        "include_state": bool(data.get("include_state", False)),
        "data_mix": data_mix,
        "state_action_normalization": "fastwam_zscore via new_embodiment statistics",
        "image": "head+left_wrist+right_wrist -> FastWAM 320x384 composite",
        "action_chunk": [16, 14],
        "use_correlated_noise": uses_correlated_noise,
        "correlation_shape": correlation_shape,
    }
    print("[RoboDojo] checkpoint contract PASS")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    verify(args.checkpoint)


if __name__ == "__main__":
    main()
