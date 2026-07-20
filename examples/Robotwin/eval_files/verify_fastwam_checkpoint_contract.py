#!/usr/bin/env python3
"""Fail-fast audit for StarVLA checkpoints trained with FastWAM's ABI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.model_server.checkpoint_contract import (  # noqa: E402
    find_run_file,
    load_checkpoint_contract_config,
    resolve_config_expects_state,
)


DEPLOY_CONFIG_PATH = Path(__file__).with_name("deploy_policy_fastwam.yml")


def _get(config: dict, path: str):
    value = config
    for key in path.split("."):
        value = value[key]
    return value


def verify(
    checkpoint: Path,
    replan_steps: int,
    *,
    expected_wam_phase: str | None = None,
    expected_world_to_action: bool | None = None,
) -> dict:
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

    framework_cfg = config.get("framework", {})
    action_cfg = framework_cfg.get("action_model", {})
    if bool(action_cfg.get("use_correlated_noise", False)):
        cholesky = accessed_config_path.parent / "action_correlation_cholesky.npy"
        if not cholesky.is_file():
            errors.append(f"correlated checkpoint is missing {cholesky}")
        else:
            matrix = np.load(cholesky, allow_pickle=False)
            if matrix.shape != (448, 448) or not np.isfinite(matrix).all():
                errors.append(f"invalid correlated-noise Cholesky: shape={matrix.shape}, path={cholesky}")

    trainer_cfg = config.get("trainer", {})
    wam_cfg = framework_cfg.get("wam", {})
    guidance_cfg = wam_cfg.get("guidance", {})
    actual_wam_phase = str(trainer_cfg.get("wam_two_stage_phase", "") or "").lower()
    actual_wam_recipe = str(
        trainer_cfg.get("wam_two_stage_recipe", "legacy_v1") or "legacy_v1"
    ).lower()
    expected_wam_phase = (
        None if expected_wam_phase is None else str(expected_wam_phase).lower()
    )
    if expected_wam_phase not in {None, "predictor_warmup", "gate_ft"}:
        errors.append(
            "expected_wam_phase must be predictor_warmup or gate_ft, got "
            f"{expected_wam_phase!r}"
        )
    if actual_wam_phase and actual_wam_recipe not in {
        "legacy_v1",
        "policy_first_v2",
        "baseline_preserving_v3",
    }:
        errors.append(
            "trainer.wam_two_stage_recipe must be legacy_v1, policy_first_v2, "
            "or baseline_preserving_v3, got "
            f"{actual_wam_recipe!r}"
        )
    if expected_wam_phase is not None:
        if actual_wam_phase != expected_wam_phase:
            errors.append(
                f"trainer.wam_two_stage_phase: expected {expected_wam_phase!r}, "
                f"got {actual_wam_phase or '<missing>'!r}"
            )
        common_wam_expected = {
            "framework.wam.enabled": True,
            "framework.wam.guidance.enabled": True,
            "framework.wam.guidance.prompt_mode": "dual_query",
            "framework.wam.guidance.exclude_post_query_context": True,
            "framework.wam.guidance.signal": "z_pred",
            "framework.wam.guidance.bridge_source": "predicted",
            "framework.wam.guidance.detach_world": True,
            "framework.wam.guidance.detached_prediction_eval_mode": True,
            "framework.wam.guidance.world_condition_on_state": True,
            "framework.wam.guidance.include_context_in_action_memory": True,
            "framework.wam.guidance.include_context_in_world_memory": True,
            "framework.wam.guidance.world_eval_mode": "correct",
        }
        for path, wanted in common_wam_expected.items():
            try:
                actual = _get(config, path)
            except (KeyError, TypeError):
                actual = "<missing>"
            if actual != wanted:
                errors.append(f"{path}: expected {wanted!r}, got {actual!r}")

        task_weights = framework_cfg.get("tasks", {}).get("weights", {})
        active_tasks = [
            str(name) for name, weight in task_weights.items() if float(weight) > 0.0
        ]
        if expected_wam_phase == "gate_ft":
            if bool(guidance_cfg.get("action_world_bypass", True)):
                errors.append("gate_ft checkpoint must have guidance.action_world_bypass=false")
            if active_tasks != ["policy"]:
                errors.append(f"gate_ft checkpoint must train only policy, got {active_tasks}")
            if int(trainer_cfg.get("max_train_steps", 0)) != 20000:
                errors.append("gate_ft checkpoint must record max_train_steps=20000")
            if not trainer_cfg.get("pretrained_checkpoint"):
                errors.append("gate_ft checkpoint is missing its Stage-1 pretrained_checkpoint provenance")
            if not bool(trainer_cfg.get("reset_world_gates_after_pretrained_load", False)):
                errors.append("gate_ft checkpoint must record reset_world_gates_after_pretrained_load=true")
            if actual_wam_recipe in {"policy_first_v2", "baseline_preserving_v3"}:
                if bool(guidance_cfg.get("detach_action_backbone", False)):
                    errors.append(f"{actual_wam_recipe} gate_ft must keep detach_action_backbone=false")
                if not bool(guidance_cfg.get("detach_world_backbone", False)):
                    errors.append(f"{actual_wam_recipe} gate_ft must preserve detach_world_backbone=true")
                if actual_wam_recipe == "baseline_preserving_v3" and not bool(
                    guidance_cfg.get("baseline_action_context", False)
                ):
                    errors.append(
                        "baseline_preserving_v3 gate_ft must use the native baseline action context"
                    )
        elif expected_wam_phase == "predictor_warmup":
            if not bool(guidance_cfg.get("action_world_bypass", False)):
                errors.append("predictor_warmup checkpoint must have guidance.action_world_bypass=true")
            if actual_wam_recipe in {"policy_first_v2", "baseline_preserving_v3"}:
                if bool(guidance_cfg.get("detach_action_backbone", False)):
                    errors.append(
                        f"{actual_wam_recipe} predictor_warmup must let action loss update Qwen"
                    )
                if not bool(guidance_cfg.get("detach_world_backbone", False)):
                    errors.append(
                        f"{actual_wam_recipe} predictor_warmup must detach world loss from Qwen"
                    )
                if int(trainer_cfg.get("num_warmup_steps", 0)) != 2000:
                    errors.append(
                        f"{actual_wam_recipe} predictor_warmup must use baseline-matched 2000 LR warmup steps"
                    )
                if actual_wam_recipe == "baseline_preserving_v3" and not bool(
                    guidance_cfg.get("baseline_action_context", False)
                ):
                    errors.append(
                        "baseline_preserving_v3 predictor_warmup must use the native baseline action context"
                    )
            elif not bool(guidance_cfg.get("detach_action_backbone", False)):
                errors.append("legacy predictor_warmup checkpoint must detach the action backbone")
            if active_tasks != ["joint_detached"]:
                errors.append(
                    f"predictor_warmup checkpoint must train only joint_detached, got {active_tasks}"
                )
            if int(trainer_cfg.get("max_train_steps", 0)) != 80000:
                errors.append("predictor_warmup checkpoint must record max_train_steps=80000")

    actual_world_to_action = bool(guidance_cfg.get("world_to_action_enabled", True))
    if (
        expected_world_to_action is not None
        and actual_world_to_action is not expected_world_to_action
    ):
        errors.append(
            "framework.wam.guidance.world_to_action_enabled: expected "
            f"{expected_world_to_action!r}, got {actual_world_to_action!r}"
        )
    if not actual_world_to_action and not bool(
        guidance_cfg.get("action_world_bypass", False)
    ):
        errors.append(
            "world_to_action_enabled=false requires action_world_bypass=true"
        )

    # Keep this verifier safe to run in the lightweight RoboTwin client image.
    # Importing the training registry transitively requires geometry packages
    # that evaluation never uses.  The checkpoint statistics establish the
    # z-score ABI above; the adjacent deployment template establishes that the
    # client applies that ABI and selects the matching embodiment statistics.
    deploy_config: dict = {}
    if not DEPLOY_CONFIG_PATH.is_file():
        errors.append(f"FastWAM deployment config is missing: {DEPLOY_CONFIG_PATH}")
    else:
        with DEPLOY_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            loaded_deploy_config = yaml.safe_load(handle)
        if not isinstance(loaded_deploy_config, dict):
            errors.append(
                "FastWAM deployment config must be a mapping, "
                f"got {type(loaded_deploy_config).__name__}"
            )
        else:
            deploy_config = loaded_deploy_config
            expected_deploy = {
                "policy_name": "model2robotwin_fastwam_interface",
                "action_mode": "abs",
                "normalization_mode": "fastwam_zscore",
            }
            for name, wanted in expected_deploy.items():
                actual = deploy_config.get(name, "<missing>")
                if actual != wanted:
                    errors.append(
                        f"deploy_policy_fastwam.yml {name}: expected {wanted!r}, got {actual!r}"
                    )
            unnorm_key = deploy_config.get("unnorm_key")
            if not isinstance(unnorm_key, str) or unnorm_key not in stats:
                errors.append(
                    "deploy_policy_fastwam.yml unnorm_key must select the checkpoint's "
                    f"statistics entry; got {unnorm_key!r}, available={list(stats)}"
                )

    if errors:
        raise ValueError("FastWAM checkpoint contract failed:\n- " + "\n- ".join(errors))
    return {
        "checkpoint": str(checkpoint),
        "config": str(accessed_config_path),
        "contract_config": str(contract_config_path),
        "statistics": str(stats_path),
        "deploy_config": str(DEPLOY_CONFIG_PATH),
        "normalization": deploy_config.get("normalization_mode", "<missing>"),
        "unnorm_key": deploy_config.get("unnorm_key", "<missing>"),
        "chunk": 32,
        "replan": replan_steps,
        "image": "one 320x384 composite",
        "state": "14-D release order" if expects_state else "disabled (request state is omitted)",
        "expects_state": expects_state,
        "state_contract_source": state_contract_source,
        "action": "14-D release order",
        "wam_two_stage_phase": actual_wam_phase or None,
        "wam_two_stage_recipe": actual_wam_recipe if actual_wam_phase else None,
        "wam_guidance_enabled": bool(guidance_cfg.get("enabled", False)),
        "wam_action_world_bypass": bool(guidance_cfg.get("action_world_bypass", False)),
        "wam_world_to_action_enabled": actual_world_to_action,
        "wam_baseline_action_context": bool(
            guidance_cfg.get("baseline_action_context", False)
        ),
        "wam_bridge_source": guidance_cfg.get("bridge_source"),
        "wam_world_eval_mode": guidance_cfg.get("world_eval_mode"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument(
        "--expected-wam-phase",
        choices=("predictor_warmup", "gate_ft"),
        default=None,
        help="Optionally require the strict two-stage WAM checkpoint phase.",
    )
    parser.add_argument(
        "--expected-world-to-action",
        choices=("enabled", "disabled"),
        default=None,
        help="Optionally require or forbid the physical world-to-action path.",
    )
    args = parser.parse_args()
    summary = verify(
        args.checkpoint,
        args.replan_steps,
        expected_wam_phase=args.expected_wam_phase,
        expected_world_to_action=(
            None
            if args.expected_world_to_action is None
            else args.expected_world_to_action == "enabled"
        ),
    )
    print("FastWAM checkpoint contract PASS")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
