#!/usr/bin/env python3
"""Validate the maintained RoboDojo H25 recipes."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.reproducibility.robodojo_rynn50k import (  # noqa: E402
    ROBODOJO_RYNN50K_BASE_H25_CURRENT_DINO_FULLRES_PROFILE,
    ROBODOJO_RYNN50K_BASE_H25_PROFILE,
    ROBODOJO_RYNN50K_BASE_HISTORY_H25_MEM_PROFILE,
    ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_BF16_PROFILE,
    ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_PROFILE,
    validate_base_h25_current_dino_fullres_config,
    validate_base_h25_config,
    validate_base_history_h25_mem_config,
    validate_base_text_h25_mem_bf16_config,
    validate_base_text_h25_mem_config,
    validate_dataset_statistics,
)


HERE = Path(__file__).resolve().parent
BASE_H25_CONFIG = HERE / "rynn_base_h25_50k.yaml"
BASE_H25_CURRENT_DINO_FULLRES_CONFIG = (
    HERE / "rynn_base_h25_current_dino_fullres_50k.yaml"
)
BASE_TEXT_H25_MEM_CONFIG = HERE / "rynn_base_text_h25_mem_50k.yaml"
BASE_TEXT_H25_MEM_BF16_CONFIG = HERE / "rynn_base_text_h25_mem_bf16_50k.yaml"
BASE_HISTORY_H25_MEM_CONFIG = HERE / "rynn_base_history_h25_mem_50k.yaml"
EXPECTED_MIXTURE = [
    ("RoboDojo_lerobot_v21_language_v1", 1.0, "robodojo_arx_x5")
]
EXPECTED_TEXT_MIXTURE = [
    ("RoboDojo_lerobot_v21_language_v2", 1.0, "robodojo_arx_x5")
]


def _load(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return payload


def _assert_base_h25_current_dino_fullres_variant() -> None:
    control = _load(BASE_H25_CONFIG)
    variant = _load(BASE_H25_CURRENT_DINO_FULLRES_CONFIG)
    control["run_id"] = variant["run_id"]
    control["framework"]["reproduction_profile"] = variant["framework"][
        "reproduction_profile"
    ]
    control["framework"]["dino"]["current_dino_pool"] = 1
    if control != variant:
        raise ValueError(
            "The current-DINO-fullres YAML must differ from the base-H25 "
            "control only in run_id, reproduction_profile, and "
            "framework.dino.current_dino_pool"
        )


def _assert_base_text_h25_mem_variant() -> None:
    base = _load(BASE_H25_CONFIG)
    variant = _load(BASE_TEXT_H25_MEM_CONFIG)
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
    if base != variant:
        raise ValueError(
            "The event-memory text YAML drifted from the verified best-H25 "
            "recipe outside its explicit fp32-shell/text/data/eval opt-ins"
        )


def _assert_base_text_h25_mem_bf16_variant() -> None:
    control = _load(BASE_TEXT_H25_MEM_CONFIG)
    variant = _load(BASE_TEXT_H25_MEM_BF16_CONFIG)
    control["run_id"] = variant["run_id"]
    control["framework"]["reproduction_profile"] = variant["framework"][
        "reproduction_profile"
    ]
    control["framework"]["world_action_mot"][
        "action_precision_mode"
    ] = "inherit"
    if control != variant:
        raise ValueError(
            "The BF16 event-memory text YAML must differ from its FP32 "
            "counterpart only in run_id, reproduction_profile, and "
            "framework.world_action_mot.action_precision_mode"
        )


def _assert_base_history_h25_mem_variant() -> None:
    control = _load(BASE_H25_CONFIG)
    ablation = _load(BASE_HISTORY_H25_MEM_CONFIG)
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
    if control != ablation:
        raise ValueError(
            "The history-only MEM ablation drifted from its frozen "
            "six-frame no-text recipe"
        )


def _assert_registry_alias() -> None:
    # Read the literal registration without importing the data pipeline.  The
    # verifier must remain runnable in the historical launcher image before
    # optional training dependencies (notably pytorch3d) are initialized.
    registry_path = (
        REPO_ROOT / "examples/RoboDojo/train_files/data_registry/data_config.py"
    )
    tree = ast.parse(registry_path.read_text(encoding="utf-8"), registry_path)
    mixtures = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DATASET_NAMED_MIXTURES"
            for target in node.targets
        ):
            mixtures = ast.literal_eval(node.value)
            break
    if mixtures is None:
        raise ValueError(f"DATASET_NAMED_MIXTURES is missing from {registry_path}")
    actual = mixtures.get("robodojo_v21_language_optional")
    if actual != EXPECTED_MIXTURE:
        raise ValueError(
            "Historical data alias drifted: "
            f"actual={actual!r}, expected={EXPECTED_MIXTURE!r}"
        )
    actual_text = mixtures.get("robodojo_v21_language")
    if actual_text != EXPECTED_TEXT_MIXTURE:
        raise ValueError(
            "Required text data alias drifted: "
            f"actual={actual_text!r}, expected={EXPECTED_TEXT_MIXTURE!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--statistics", type=Path)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = _load(config_path)
    if config_path == BASE_H25_CURRENT_DINO_FULLRES_CONFIG.resolve():
        _assert_base_h25_current_dino_fullres_variant()
    profile = str(config["framework"].get("reproduction_profile", ""))
    if profile == ROBODOJO_RYNN50K_BASE_H25_CURRENT_DINO_FULLRES_PROFILE:
        summary = validate_base_h25_current_dino_fullres_config(config)
    elif profile == ROBODOJO_RYNN50K_BASE_H25_PROFILE:
        summary = validate_base_h25_config(config)
    elif profile == ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_PROFILE:
        summary = validate_base_text_h25_mem_config(config)
        _assert_base_text_h25_mem_variant()
    elif profile == ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_BF16_PROFILE:
        summary = validate_base_text_h25_mem_bf16_config(config)
        _assert_base_text_h25_mem_bf16_variant()
    elif profile == ROBODOJO_RYNN50K_BASE_HISTORY_H25_MEM_PROFILE:
        summary = validate_base_history_h25_mem_config(config)
        _assert_base_history_h25_mem_variant()
    else:
        raise ValueError(f"Unsupported maintained reproduction profile: {profile!r}")
    _assert_registry_alias()
    if args.statistics is not None:
        summary["statistics_sha256"] = validate_dataset_statistics(
            args.statistics.expanduser().resolve()
        )
    summary["config"] = str(config_path)
    print("[released-rynn50k] PASS")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
