#!/usr/bin/env python3
"""Verify Rynn H8 model ABI + WAM-exp5-aligned non-model train controls."""

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

from deployment.libero_image import (
    LIBERO_COMPOSITE_LAYOUT,
    LIBERO_COMPOSITE_SIZE,
    LIBERO_COMPOSITE_SOURCE_VIEW_KEYS,
    LIBERO_COMPOSITE_VIEW_KEY,
)


OFFICIAL_COT_PROMPT = (
    "Your task is {instruction}. To identify the key objects for your task. "
    "Locate their bounding boxes in [x1,y1,x2,y2] format."
)

# Shared data/optimizer fields that still match StarVLA's upstream LIBERO YAML.
# Schedule / freeze / action LR intentionally follow WAM exp5 instead.
UPSTREAM_PARITY_PATHS = (
    "seed",
    "is_debug",
    "version_id",
    "framework.qwenvl.vl_hidden_dim",
    "framework.action_model.action_dim",
    "framework.action_model.action_horizon",
    "datasets.vla_data.data_mix",
    "datasets.vla_data.action_type",
    "datasets.vla_data.sequential_step_sampling",
    "datasets.vla_data.CoT_prompt",
    "datasets.vla_data.per_device_batch_size",
    "datasets.vla_data.load_all_data_for_training",
    "datasets.vla_data.video_backend",
    "trainer.save_interval",
    "trainer.learning_rate.base",
    "trainer.learning_rate.qwen_vl_interface",
    "trainer.lr_scheduler_type",
    "trainer.scheduler_specific_kwargs.min_lr",
    "trainer.loss_scale.vla",
    "trainer.max_grad_norm",
    "trainer.weight_decay",
    "trainer.gradient_clipping",
    "trainer.gradient_checkpointing",
    "trainer.optimizer.name",
    "trainer.optimizer.betas",
    "trainer.optimizer.eps",
    "trainer.optimizer.weight_decay",
)

OFFICIAL_REFERENCE_PATH = Path(__file__).with_name("starvla_cotrain_libero.yaml")

# Non-model controls aligned to
# outputs/.../0620_wam_exp5_2b_bigbs_policy/config.full.yaml
EXPECTED = {
    "seed": 42,
    "is_debug": False,
    "version_id": "0.21",
    "framework.name": "QwenWorldActionMoT",
    "framework.enable_world_action_mot": True,
    "framework.qwenvl.vl_hidden_dim": 2048,
    "framework.planner.num_action_queries": 8,
    "framework.planner.text_supervision.enabled": False,
    "framework.world_action_mot.architecture": "causal_dino_mot",
    "framework.world_action_mot.interaction_mode": "base",
    "framework.world_action_mot.world_grid_height": 8,
    "framework.world_action_mot.world_grid_width": 16,
    "framework.world_action_mot.max_world_tokens": 128,
    "framework.world_action_mot.num_world_views": 1,
    "framework.world_action_mot.text_loss_weight": 0.0,
    "framework.dino.image_size": [256, 512],
    "framework.dino.patch_size": 16,
    "framework.dino.dino_pool": 2,
    "framework.action_model.action_dim": 7,
    "framework.action_model.state_dim": 0,
    "framework.action_model.action_horizon": 8,
    "datasets.vla_data.data_mix": "libero_all",
    "datasets.vla_data.image_layout": LIBERO_COMPOSITE_LAYOUT,
    "datasets.vla_data.composite_source_view_keys": list(
        LIBERO_COMPOSITE_SOURCE_VIEW_KEYS
    ),
    "datasets.vla_data.composite_view_key": LIBERO_COMPOSITE_VIEW_KEY,
    "datasets.vla_data.include_state": False,
    "datasets.vla_data.action_type": "delta_qpos",
    "datasets.vla_data.action_mode": "abs",
    "datasets.vla_data.action_horizon": 8,
    "datasets.vla_data.world_model.future_stride": 8,
    "datasets.vla_data.text_annotations.enabled": False,
    "datasets.vla_data.sequential_step_sampling": False,
    "datasets.vla_data.CoT_prompt": OFFICIAL_COT_PROMPT,
    "datasets.vla_data.per_device_batch_size": 16,
    "datasets.vla_data.load_all_data_for_training": True,
    "datasets.vla_data.obs_image_size": list(LIBERO_COMPOSITE_SIZE),
    "datasets.vla_data.video_backend": "torchvision_av",
    "datasets.vla_data.num_workers": 16,
    "datasets.vla_data.prefetch_factor": 4,
    "datasets.vla_data.pin_memory": True,
    "datasets.vla_data.persistent_workers": True,
    "trainer.expected_global_batch_size": 256,
    "trainer.max_train_steps": 60000,
    "trainer.num_warmup_steps": 3000,
    "trainer.save_interval": 5000,
    "trainer.eval_interval": 100000,
    "trainer.learning_rate.base": 2.5e-05,
    "trainer.learning_rate.qwen_vl_interface": 1.0e-05,
    "trainer.learning_rate.action_model": 5.0e-05,
    "trainer.lr_scheduler_type": "cosine_with_min_lr",
    "trainer.scheduler_specific_kwargs.min_lr": 1.0e-06,
    "trainer.freeze_modules": "",
    "trainer.loss_scale.vla": 1.0,
    "trainer.max_grad_norm": 1.0,
    "trainer.weight_decay": 0.0,
    "trainer.logging_frequency": 100,
    "trainer.gradient_clipping": 1.0,
    "trainer.gradient_accumulation_steps": 1,
    "trainer.gradient_checkpointing": True,
    "trainer.optimizer.name": "AdamW",
    "trainer.optimizer.betas": [0.9, 0.95],
    "trainer.optimizer.eps": 1.0e-08,
    "trainer.optimizer.weight_decay": 1.0e-08,
}

FORBIDDEN = (
    # Precision is a StarVLA runtime concern, not part of the LIBERO recipe.
    "framework.world_action_mot.action_precision_mode",
)


def select(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return "<MISSING>"
        value = value[key]
    return value


def verify(config: dict[str, Any], *, num_processes: int) -> dict[str, Any]:
    forbidden = {
        path: select(config, path)
        for path in FORBIDDEN
        if select(config, path) != "<MISSING>"
    }
    if forbidden:
        details = "; ".join(
            f"{path}={value!r} must be omitted"
            for path, value in sorted(forbidden.items())
        )
        raise ValueError(
            "Invalid LIBERO H8 recipe: "
            f"{details}; precision must inherit the StarVLA runtime"
        )

    official = yaml.safe_load(
        OFFICIAL_REFERENCE_PATH.read_text(encoding="utf-8")
    ) or {}
    parity_mismatches = {
        path: {
            "actual": select(config, path),
            "official": select(official, path),
        }
        for path in UPSTREAM_PARITY_PATHS
        if select(config, path) != select(official, path)
    }
    if parity_mismatches:
        details = "; ".join(
            f"{path}={values['actual']!r} "
            f"(official {values['official']!r})"
            for path, values in sorted(parity_mismatches.items())
        )
        raise ValueError(f"LIBERO upstream-control drift: {details}")

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
        raise ValueError(f"Invalid LIBERO H8 recipe: {details}")

    if num_processes not in {1, 16}:
        raise ValueError(
            f"LIBERO H8 supports 1 or 16 processes, got {num_processes}"
        )
    yaml_micro = int(select(config, "datasets.vla_data.per_device_batch_size"))
    global_batch = int(select(config, "trainer.expected_global_batch_size"))
    micro = 4 if num_processes == 1 else yaml_micro
    denominator = micro * num_processes
    if global_batch % denominator:
        raise ValueError(
            f"global batch {global_batch} is not divisible by {micro} x {num_processes}"
        )
    accumulation = global_batch // denominator

    height, width = select(config, "framework.dino.image_size")
    patch = int(select(config, "framework.dino.patch_size"))
    pool = int(select(config, "framework.dino.dino_pool"))
    pooled_grid = (height // patch // pool, width // patch // pool)
    configured_current_pool = select(config, "framework.dino.current_dino_pool")
    current_pool = pool if configured_current_pool is None else int(configured_current_pool)
    if (
        current_pool <= 0
        or (height // patch) % current_pool
        or (width // patch) % current_pool
    ):
        raise ValueError(
            f"current_dino_pool={configured_current_pool!r} does not divide "
            f"the raw DINO grid {(height // patch, width // patch)}"
        )
    if configured_current_pool is not None and current_pool >= pool:
        raise ValueError(
            "current_dino_pool must be smaller than dino_pool for a dense-current "
            f"variant, got current={current_pool}, future={pool}"
        )
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
        "profile": "libero_rynnbrain11_dualview_h8_bs256_60k_nostate_wamhyper_v1",
        "num_processes": num_processes,
        "micro_batch_size": micro,
        "gradient_accumulation_steps": accumulation,
        "global_batch_size": global_batch,
        "max_train_steps": 60000,
        "save_interval": 5000,
        "upstream_control_fields": len(UPSTREAM_PARITY_PATHS),
        "composite_size": list(LIBERO_COMPOSITE_SIZE),
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
