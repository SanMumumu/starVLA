#!/usr/bin/env python3
"""Verify the FastWAM-aligned RoboTwin Rynn H32 / BS768 / 50K contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Rynn keeps the FastWAM H32 data ABI while pinning the confirmed BS768/50K
# schedule and its own causal-DINO MoT model contract.
EXPECTED = {
    "framework.name": "QwenWorldActionMoT",
    "framework.enable_world_action_mot": True,
    "framework.planner.num_action_queries": 32,
    "framework.planner.text_supervision.enabled": False,
    "framework.world_action_mot.architecture": "causal_dino_mot",
    "framework.world_action_mot.world_grid_height": 12,
    "framework.world_action_mot.world_grid_width": 10,
    "framework.world_action_mot.max_world_tokens": 120,
    "framework.world_action_mot.text_loss_weight": 0.0,
    "framework.dino.image_size": [384, 320],
    "framework.dino.patch_size": 16,
    "framework.dino.dino_pool": 2,
    "framework.action_model.action_dim": 14,
    "framework.action_model.state_dim": 14,
    "framework.action_model.action_horizon": 32,
    "datasets.vla_data.data_mix": "robotwin_fastwam",
    "datasets.vla_data.image_layout": "fastwam_composite",
    "datasets.vla_data.composite_view_key": "video.robotwin_composite",
    "datasets.vla_data.include_state": True,
    "datasets.vla_data.action_type": "abs_qpos",
    "datasets.vla_data.action_mode": "abs",
    "datasets.vla_data.action_horizon": 32,
    "datasets.vla_data.world_model.future_stride": 32,
    "datasets.vla_data.fastwam_direct_frame_sampling": True,
    "datasets.vla_data.text_annotations.enabled": False,
    "datasets.vla_data.per_device_batch_size": 12,
    "datasets.vla_data.num_workers": 8,
    "trainer.expected_global_batch_size": 768,
    "trainer.max_train_steps": 50000,
    "trainer.num_warmup_steps": 2000,
    "trainer.save_interval": 10000,
    "trainer.eval_interval": 1000,
    "trainer.learning_rate.base": 1.0e-05,
    "trainer.learning_rate.qwen_vl_interface": 1.0e-05,
    "trainer.learning_rate.action_model": 1.0e-04,
    "trainer.lr_scheduler_type": "cosine_with_min_lr",
    "trainer.scheduler_specific_kwargs.min_lr": 5.0e-07,
    "trainer.freeze_modules": "",
    "trainer.logging_frequency": 200,
    "trainer.gradient_accumulation_steps": 1,
}


def select(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return "<MISSING>"
        value = value[key]
    return value


def verify(config: dict[str, Any], *, num_processes: int) -> dict[str, Any]:
    mismatches = {
        path: {"actual": select(config, path), "expected": expected}
        for path, expected in EXPECTED.items()
        if select(config, path) != expected
    }
    if mismatches:
        details = "; ".join(
            f"{path}={values['actual']!r} (expected {values['expected']!r})"
            for path, values in sorted(mismatches.items())
        )
        raise ValueError(f"Invalid RoboTwin H32 recipe: {details}")

    if num_processes != 64:
        raise ValueError(
            "RoboTwin H32 BS768 recipe requires exactly 64 processes, "
            f"got {num_processes}"
        )
    yaml_micro = int(select(config, "datasets.vla_data.per_device_batch_size"))
    global_batch = int(select(config, "trainer.expected_global_batch_size"))
    accumulation = int(
        select(config, "trainer.gradient_accumulation_steps")
    )
    computed_global = yaml_micro * num_processes * accumulation
    if computed_global != global_batch:
        raise ValueError(
            "global batch contract mismatch: "
            f"{yaml_micro} x {num_processes} x {accumulation} = "
            f"{computed_global}, declared {global_batch}"
        )

    height, width = select(config, "framework.dino.image_size")
    patch = int(select(config, "framework.dino.patch_size"))
    pool = int(select(config, "framework.dino.dino_pool"))
    pooled_grid = (height // patch // pool, width // patch // pool)
    configured_current_pool = select(config, "framework.dino.current_dino_pool")
    current_pool = pool if configured_current_pool is None else int(configured_current_pool)
    current_grid = (height // patch // current_pool, width // patch // current_pool)
    configured_grid = (
        int(select(config, "framework.world_action_mot.world_grid_height")),
        int(select(config, "framework.world_action_mot.world_grid_width")),
    )
    if pooled_grid != configured_grid:
        raise ValueError(
            f"DINO pooled grid {pooled_grid} != physical grid {configured_grid}"
        )

    return {
        "profile": "rynn_h32_bs768_50k",
        "num_processes": num_processes,
        "micro_batch_size": yaml_micro,
        "gradient_accumulation_steps": accumulation,
        "global_batch_size": global_batch,
        "max_train_steps": 50000,
        "save_interval": 10000,
        "current_dino_grid": list(current_grid),
        "pooled_dino_grid": list(pooled_grid),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    print(json.dumps(verify(config, num_processes=args.num_processes), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
