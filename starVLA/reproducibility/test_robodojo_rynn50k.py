from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from starVLA.reproducibility.robodojo_rynn50k import (
    ROBODOJO_RYNN50K_BASE_H25_CURRENT_DINO_FULLRES_PROFILE,
    ROBODOJO_RYNN50K_BASE_H25_PROFILE,
    ROBODOJO_RYNN50K_BASE_HISTORY_H25_MEM_PROFILE,
    ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_BF16_PROFILE,
    ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_PROFILE,
    ROBODOJO_RYNN50K_WEIGHT_CONTRACT,
    validate_base_h25_current_dino_fullres_config,
    validate_base_h25_config,
    validate_base_history_h25_mem_config,
    validate_base_text_h25_mem_bf16_config,
    validate_base_text_h25_mem_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE_DIR = REPO_ROOT / "examples/RoboDojo/train_files/released_rynn50k"


def _load(name: str) -> dict:
    return yaml.safe_load((RECIPE_DIR / name).read_text(encoding="utf-8"))


def test_retired_fp32_ablation_entrypoints_are_absent() -> None:
    retired_recipes = {
        "rynn_base_h25_50k_fp32.yaml",
        "rynn_base_h25_50k_fp32_frame_uniform.yaml",
    }
    assert all(not (RECIPE_DIR / name).exists() for name in retired_recipes)

    job_dir = REPO_ROOT / "执行脚本/Robodojo/released_rynn50k"
    retired_jobs = {
        "job_base_h25_50k_fp32.yaml",
        "job_base_h25_50k_fp32_frame_uniform.yaml",
    }
    assert all(not (job_dir / name).exists() for name in retired_jobs)


def test_base_h25_contract_is_frozen() -> None:
    config = _load("rynn_base_h25_50k.yaml")
    summary = validate_base_h25_config(config)
    assert summary["profile"] == ROBODOJO_RYNN50K_BASE_H25_PROFILE
    assert summary["mode"] == "base"
    assert summary["weight_contract"] == ROBODOJO_RYNN50K_WEIGHT_CONTRACT


def test_base_h25_profile_rejects_wall_x_drift() -> None:
    config = _load("rynn_base_h25_50k.yaml")
    config["framework"]["world_action_mot"]["action_velocity_target"] = "clean_minus_noise"
    with pytest.raises(ValueError, match="action_velocity_target"):
        validate_base_h25_config(config)


def test_base_h25_profile_rejects_partial_horizon_changes() -> None:
    variant = _load("rynn_base_h25_50k.yaml")
    variant["datasets"]["vla_data"]["action_horizon"] = 16
    with pytest.raises(ValueError, match="datasets.vla_data.action_horizon"):
        validate_base_h25_config(variant)


def test_base_h25_current_dino_fullres_changes_only_current_grid() -> None:
    control = _load("rynn_base_h25_50k.yaml")
    variant = _load("rynn_base_h25_current_dino_fullres_50k.yaml")
    summary = validate_base_h25_current_dino_fullres_config(variant)
    assert (
        summary["profile"]
        == ROBODOJO_RYNN50K_BASE_H25_CURRENT_DINO_FULLRES_PROFILE
    )
    assert variant["framework"]["dino"]["current_dino_pool"] == 1
    assert variant["framework"]["dino"]["dino_pool"] == 2

    control["run_id"] = variant["run_id"]
    control["framework"]["reproduction_profile"] = variant["framework"][
        "reproduction_profile"
    ]
    control["framework"]["dino"]["current_dino_pool"] = 1
    assert control == variant


def test_base_h25_current_dino_fullres_rejects_future_grid_drift() -> None:
    variant = _load("rynn_base_h25_current_dino_fullres_50k.yaml")
    variant["framework"]["dino"]["dino_pool"] = 1
    with pytest.raises(ValueError, match="framework.dino.dino_pool"):
        validate_base_h25_current_dino_fullres_config(variant)


def test_base_text_h25_mem_variant_keeps_released_physical_recipe() -> None:
    base = _load("rynn_base_h25_50k.yaml")
    variant = _load("rynn_base_text_h25_mem_50k.yaml")
    summary = validate_base_text_h25_mem_config(variant)
    assert summary["profile"] == ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_PROFILE
    assert summary["mode"] == "base"

    base["run_id"] = variant["run_id"]
    base["framework"]["reproduction_profile"] = variant["framework"][
        "reproduction_profile"
    ]
    base["framework"]["planner"]["text_supervision"] = variant["framework"][
        "planner"
    ]["text_supervision"]
    base["framework"]["world_action_mot"][
        "action_precision_mode"
    ] = "fp32_shell"
    base["framework"]["world_action_mot"]["text_loss_weight"] = 0.005
    base["datasets"]["vla_data"]["data_mix"] = "robodojo_v21_language"
    base["datasets"]["vla_data"]["text_annotations"] = variant["datasets"][
        "vla_data"
    ]["text_annotations"]
    base["trainer"]["action_eval_enabled"] = False
    assert base == variant
    assert "mem_vision_encoder" not in variant["framework"]["qwenvl"]
    assert variant["framework"]["world_action_mot"]["action_precision_mode"] == "fp32_shell"
    assert variant["framework"]["planner"]["text_supervision"]["history"]["enabled"] is False


def test_base_text_h25_mem_profile_requires_event_memory_and_text_loss() -> None:
    variant = _load("rynn_base_text_h25_mem_50k.yaml")
    variant["datasets"]["vla_data"]["text_annotations"]["event_memory"][
        "enabled"
    ] = False
    with pytest.raises(ValueError, match="text_annotations.event_memory.enabled"):
        validate_base_text_h25_mem_config(variant)

    variant = _load("rynn_base_text_h25_mem_50k.yaml")
    variant["framework"]["world_action_mot"]["text_loss_weight"] = 0.0
    with pytest.raises(ValueError, match="text_loss_weight"):
        validate_base_text_h25_mem_config(variant)


def test_base_text_h25_mem_bf16_is_a_precision_only_counterpart() -> None:
    fp32 = _load("rynn_base_text_h25_mem_50k.yaml")
    bf16 = _load("rynn_base_text_h25_mem_bf16_50k.yaml")
    summary = validate_base_text_h25_mem_bf16_config(bf16)
    assert summary["profile"] == ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_BF16_PROFILE
    assert summary["mode"] == "base"

    fp32["run_id"] = bf16["run_id"]
    fp32["framework"]["reproduction_profile"] = bf16["framework"][
        "reproduction_profile"
    ]
    fp32["framework"]["world_action_mot"][
        "action_precision_mode"
    ] = "inherit"
    assert fp32 == bf16
    assert bf16["framework"]["world_action_mot"]["action_precision_mode"] == (
        "inherit"
    )


def test_history_h25_mem_is_strict_text_plan_ablation() -> None:
    control = _load("rynn_base_h25_50k.yaml")
    ablation = _load("rynn_base_history_h25_mem_50k.yaml")
    summary = validate_base_history_h25_mem_config(ablation)
    assert summary["profile"] == ROBODOJO_RYNN50K_BASE_HISTORY_H25_MEM_PROFILE
    assert summary["mode"] == "base"

    control["run_id"] = ablation["run_id"]
    control["framework"]["reproduction_profile"] = ablation["framework"][
        "reproduction_profile"
    ]
    control["framework"]["qwenvl"]["mem_vision_encoder"] = ablation[
        "framework"
    ]["qwenvl"]["mem_vision_encoder"]
    control["framework"]["planner"]["text_supervision"] = ablation[
        "framework"
    ]["planner"]["text_supervision"]
    control["datasets"]["vla_data"]["data_mix"] = "robodojo_v21_language"
    control["datasets"]["vla_data"]["text_annotations"] = ablation[
        "datasets"
    ]["vla_data"]["text_annotations"]
    control["trainer"]["action_eval_enabled"] = False
    assert control == ablation
