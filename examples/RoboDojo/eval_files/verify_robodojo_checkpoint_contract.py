#!/usr/bin/env python3
"""Fail-fast validation for a RoboDojo StarVLA checkpoint and its run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch
import yaml

from starVLA.dataloader.action_correlation import validate_action_correlation_cholesky


EXPECTED_SOURCE_VIEWS = [
    "video.cam_high",
    "video.cam_left_wrist",
    "video.cam_right_wrist",
]

_CAUSAL_MOT_CONSTRUCTION_DEFAULTS = {
    "world_hidden_size": 3072,
    "action_hidden_size": 1024,
    "world_ffn_dim": 14336,
    "action_ffn_dim": 4096,
    "num_layers": 30,
    "num_attention_heads": 24,
    "attention_head_dim": 128,
}

_RELEASED_50K_CONTRACTS = {
    "starvla_qwen3_robodojo_dino_mot_joint_text_50k": {
        "planner": "qwen3",
        "text": True,
    },
    "starvla_qwen3_robodojo_dino_mot_joint_50k": {
        "planner": "qwen3",
        "text": False,
    },
    "starvla_rynnbrain11_robodojo_dino_mot_joint_50k": {
        "planner": "rynnbrain",
        "text": False,
    },
}


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


def _checkpoint_tensor_shapes(checkpoint: Path) -> dict[str, tuple[int, ...]]:
    """Read checkpoint tensor metadata without materializing model weights."""

    if checkpoint.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required to preflight this checkpoint"
            ) from exc
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            return {
                key: tuple(int(value) for value in handle.get_slice(key).get_shape())
                for key in handle.keys()
            }

    load_kwargs = {
        "map_location": "meta",
        "weights_only": True,
        "mmap": True,
    }
    try:
        state = torch.load(checkpoint, **load_kwargs)
    except (TypeError, RuntimeError):
        try:
            state = torch.load(
                checkpoint,
                map_location="meta",
                weights_only=True,
            )
        except TypeError:
            # Trusted, local training artifact on older PyTorch versions that
            # do not expose weights_only yet. map_location=meta still avoids
            # allocating the multi-billion-parameter checkpoint.
            state = torch.load(checkpoint, map_location="meta")
    if not isinstance(state, dict):
        raise TypeError(
            f"Checkpoint must contain a raw state_dict mapping, got {type(state).__name__}"
        )
    shapes = {
        str(key): tuple(int(value) for value in tensor.shape)
        for key, tensor in state.items()
        if torch.is_tensor(tensor)
    }
    if not shapes:
        raise ValueError(f"Checkpoint contains no tensors: {checkpoint}")
    return shapes


def _required_shape(
    shapes: dict[str, tuple[int, ...]],
    key: str,
) -> tuple[int, ...]:
    if key not in shapes:
        raise ValueError(
            f"Checkpoint is missing required causal-MoT tensor {key!r}"
        )
    return shapes[key]


def _infer_causal_mot_weight_contract(
    shapes: dict[str, tuple[int, ...]],
) -> dict:
    """Infer the physical architecture encoded by the actual state_dict."""

    layer_pattern = re.compile(r"^action_model\.layers\.(\d+)\.")
    layer_indices = sorted(
        {
            int(match.group(1))
            for key in shapes
            if (match := layer_pattern.match(key)) is not None
        }
    )
    if not layer_indices or layer_indices != list(range(layer_indices[-1] + 1)):
        raise ValueError(
            "Checkpoint causal-MoT layer indices must be contiguous from zero, "
            f"got {layer_indices[:8]}{'...' if len(layer_indices) > 8 else ''}"
        )

    layerwise_world_key = "action_model.world_context.0.0.weight"
    legacy_world_key = "action_model.world_context.0.weight"
    has_layerwise_context = layerwise_world_key in shapes
    has_legacy_context = legacy_world_key in shapes
    if has_layerwise_context == has_legacy_context:
        raise ValueError(
            "Checkpoint must contain exactly one causal-MoT context layout: "
            "per-layer or legacy shared"
        )
    expected_action_context = (
        "action_model.action_context.0.0.weight"
        if has_layerwise_context
        else "action_model.action_context.0.weight"
    )
    _required_shape(shapes, expected_action_context)

    world_input = _required_shape(shapes, "action_model.world_input.weight")
    action_input = _required_shape(shapes, "action_model.action_input.weight")
    world_ffn = _required_shape(
        shapes,
        "action_model.layers.0.world.ffn.0.weight",
    )
    action_ffn = _required_shape(
        shapes,
        "action_model.layers.0.action.ffn.0.weight",
    )
    world_query = _required_shape(
        shapes,
        "action_model.layers.0.world.self_attn.q.weight",
    )
    # Current layer-wise policies are query/vision-only and therefore have no
    # state projection. Historical state-conditioned checkpoints retain this
    # tensor and are reconstructed from its input dimension.
    state_projection = shapes.get("action_model.state_to_planner.weight")
    action_queries = _required_shape(
        shapes,
        "action_plan_queries.embedding",
    )
    world_queries = _required_shape(
        shapes,
        "world_plan_queries.embedding",
    )
    matrix_shapes = [
        world_input,
        action_input,
        world_ffn,
        action_ffn,
        world_query,
    ]
    if state_projection is not None:
        matrix_shapes.append(state_projection)
    if not all(len(shape) == 2 for shape in matrix_shapes):
        raise ValueError("Checkpoint causal-MoT projection tensors must be matrices")
    if len(action_queries) != 3 or len(world_queries) != 3:
        raise ValueError("Checkpoint planner query banks must have shape [1,N,H]")

    obsolete_action_memory = sorted(
        key
        for key in shapes
        if key.startswith("action_model.action_dino_")
    )
    if obsolete_action_memory:
        raise ValueError(
            "Checkpoint contains the obsolete action-only DINO memory branch. "
            "The supported FastWAM-style contract places full-resolution "
            "current DINO directly in the clean world prefix; obsolete keys="
            f"{obsolete_action_memory}"
        )
    layerwise = has_layerwise_context
    version = (
        "layerwise_query_only_world_v2"
        if layerwise
        else "legacy_shared_context_state_world_v1"
    )
    return {
        "version": version,
        "layerwise_planner_coupling": layerwise,
        "multires_world_input": False,
        "world_hidden_size": world_input[0],
        "world_dim": world_input[1],
        "action_hidden_size": action_input[0],
        "action_dim": action_input[1],
        "state_dim": 0 if state_projection is None else state_projection[1],
        "num_action_queries": action_queries[1],
        "num_world_queries": world_queries[1],
        "world_ffn_dim": world_ffn[0],
        "action_ffn_dim": action_ffn[0],
        "attention_inner_dim": world_query[0],
        "num_layers": len(layer_indices),
    }


def _validate_causal_construction_config(
    construction_config: dict,
    weight_contract: dict,
) -> bool:
    """Ensure config.yaml will reconstruct the state_dict that was inspected."""

    framework = construction_config.get("framework") or {}
    _expect(
        str(framework.get("name", "")),
        "QwenWorldActionMoT",
        "config.yaml framework.name",
    )
    mot = framework.get("world_action_mot") or {}
    _expect(
        str(mot.get("architecture", "legacy")).lower(),
        "causal_dino_mot",
        "config.yaml framework.world_action_mot.architecture",
    )
    for field in (
        "world_hidden_size",
        "action_hidden_size",
        "world_ffn_dim",
        "action_ffn_dim",
        "num_layers",
    ):
        configured = int(
            mot.get(field, _CAUSAL_MOT_CONSTRUCTION_DEFAULTS[field])
        )
        _expect(
            configured,
            int(weight_contract[field]),
            f"config.yaml framework.world_action_mot.{field} vs checkpoint",
        )
    _expect(
        bool(mot.get("layerwise_planner_coupling", False)),
        bool(weight_contract["layerwise_planner_coupling"]),
        "config.yaml framework.world_action_mot.layerwise_planner_coupling vs checkpoint",
    )
    attention_inner_dim = (
        int(
            mot.get(
                "num_attention_heads",
                _CAUSAL_MOT_CONSTRUCTION_DEFAULTS["num_attention_heads"],
            )
        )
        * int(
            mot.get(
                "attention_head_dim",
                _CAUSAL_MOT_CONSTRUCTION_DEFAULTS["attention_head_dim"],
            )
        )
    )
    _expect(
        attention_inner_dim,
        int(weight_contract["attention_inner_dim"]),
        "config.yaml causal-MoT attention inner dimension vs checkpoint",
    )
    action = framework.get("action_model") or {}
    dino = framework.get("dino") or {}
    _expect(
        int(action.get("action_dim", 14)),
        int(weight_contract["action_dim"]),
        "config.yaml framework.action_model.action_dim vs checkpoint",
    )
    _expect(
        int(action.get("state_dim", 0) or 0),
        int(weight_contract["state_dim"]),
        "config.yaml framework.action_model.state_dim vs checkpoint",
    )
    planner = framework.get("planner") or {}
    configured_action_queries = int(planner.get("num_action_queries", 16))
    _expect(
        configured_action_queries,
        int(weight_contract["num_action_queries"]),
        "config.yaml planner.num_action_queries vs checkpoint",
    )
    _expect(
        int(action.get("action_horizon", 16)),
        configured_action_queries,
        "config.yaml action_horizon vs action planner queries",
    )
    _expect(
        int(planner.get("num_world_queries", 16)),
        int(weight_contract["num_world_queries"]),
        "config.yaml planner.num_world_queries vs checkpoint",
    )
    _expect(
        int(dino.get("embed_dim", 768)),
        int(weight_contract["world_dim"]),
        "config.yaml framework.dino.embed_dim vs checkpoint",
    )
    if dino.get("action_dino_pool", None) is not None:
        raise ValueError(
            "config.yaml framework.dino.action_dino_pool is obsolete; use "
            "current_dino_pool so full-resolution current DINO enters the "
            "FastWAM-style clean world prefix"
        )
    image_size = list(dino.get("image_size", [384, 320]))
    if len(image_size) != 2:
        raise ValueError("config.yaml framework.dino.image_size must be [H,W]")
    patch_size = int(dino.get("patch_size", 16))
    if (
        patch_size <= 0
        or int(image_size[0]) % patch_size
        or int(image_size[1]) % patch_size
    ):
        raise ValueError(
            "config.yaml DINO image size must be divisible by patch_size"
        )
    rows = int(image_size[0]) // patch_size
    columns = int(image_size[1]) // patch_size
    future_pool = int(dino.get("dino_pool", 1))
    if (
        future_pool <= 0
        or rows % future_pool
        or columns % future_pool
    ):
        raise ValueError(
            "config.yaml framework.dino.dino_pool must divide the DINO grid"
        )
    _expect(
        (rows // future_pool, columns // future_pool),
        (
            int(mot.get("world_grid_height", 12)),
            int(mot.get("world_grid_width", 10)),
        ),
        "config.yaml pooled future DINO grid vs physical world grid",
    )
    configured_current_pool = dino.get("current_dino_pool", None)
    if configured_current_pool is None:
        return False
    current_pool = int(configured_current_pool)
    if (
        current_pool <= 0
        or current_pool >= future_pool
        or rows % current_pool
        or columns % current_pool
    ):
        raise ValueError(
            "config.yaml framework.dino.current_dino_pool must be positive, "
            "divide the DINO grid, and be smaller than dino_pool"
        )
    return True


def _resolved_mot_inference_semantics(config: dict) -> dict:
    """Resolve shape-neutral fields that can silently change inference."""

    framework = config.get("framework") or {}
    qwenvl = framework.get("qwenvl") or {}
    mem_vision = qwenvl.get("mem_vision_encoder") or {}
    planner = framework.get("planner") or {}
    text = planner.get("text_supervision") or {}
    text_history = text.get("history") or {}
    mot = framework.get("world_action_mot") or {}
    action = framework.get("action_model") or {}
    dino = framework.get("dino") or {}
    data = ((config.get("datasets") or {}).get("vla_data") or {})
    annotation_history = (
        (data.get("text_annotations") or {}).get("history") or {}
    )
    annotation_event = (
        (data.get("text_annotations") or {}).get("event_memory") or {}
    )
    return {
        "base_vlm": str(qwenvl.get("base_vlm", "")),
        "attn_implementation": str(
            qwenvl.get("attn_implementation", "flash_attention_2")
        ),
        "enable_thinking": bool(qwenvl.get("enable_thinking", False)),
        "truncate_vlm_layers": int(
            qwenvl.get("truncate_vlm_layers", 0) or 0
        ),
        "mem_vision_enabled": bool(mem_vision.get("enabled", False)),
        "mem_vision_num_frames": int(mem_vision.get("num_frames", 6)),
        "mem_vision_stride": int(
            mem_vision.get("spacetime_layer_stride", 4)
        ),
        "mem_vision_time_embed_base": float(
            mem_vision.get("time_embed_base", 100.0)
        ),
        "mem_match_history_resolution": bool(
            mem_vision.get("match_current_to_history_resolution", False)
        ),
        "mem_output_token_policy": str(
            mem_vision.get("output_token_policy", "disabled")
        ).strip().lower(),
        "num_world_queries": int(planner.get("num_world_queries", 16)),
        "num_action_queries": int(planner.get("num_action_queries", 16)),
        "text_enabled": bool(text.get("enabled", False)),
        "text_mode": str(text.get("mode", "legacy_full_text")).lower(),
        "text_keep_token": str(text.get("keep_token", "<KEEP>")),
        "text_update_token": str(text.get("update_token", "<UPDATE>")),
        "text_prompt_template": str(
            text.get(
                "prompt_template",
                "{instruction}\n"
                "Report the current subtask and the completed subtask.",
            )
        ),
        "text_max_new_tokens": int(text.get("max_new_tokens", 64)),
        "text_do_sample": bool(text.get("do_sample", False)),
        "text_update_response_template": str(
            text.get("update_response_template", "")
        ),
        "text_history_enabled": bool(text_history.get("enabled", False)),
        "text_history_image_size": tuple(
            text_history.get("history_image_size", [])
        ),
        "text_history_frame_offsets": tuple(
            int(value)
            for value in annotation_history.get("frame_offsets", [])
        ),
        "text_history_memory_offset": int(
            annotation_history.get("memory_offset", 0)
        ),
        "event_memory_enabled": bool(annotation_event.get("enabled", False)),
        "event_semantic_offset": int(
            annotation_event.get("semantic_offset", -10)
        ),
        "event_replan_interval": int(
            annotation_event.get("replan_interval", 10)
        ),
        "event_empty_memory": str(
            annotation_event.get("empty_memory", "None.")
        ),
        "event_empty_cached_subtask": str(
            annotation_event.get("empty_cached_subtask", "None.")
        ),
        "event_semantic_memory_field": str(
            annotation_event.get("semantic_memory_field", "semantic_memory")
        ),
        "event_cached_subtask_field": str(
            annotation_event.get(
                "cached_subtask_field", "cached_current_subtask"
            )
        ),
        "event_cache_valid_field": str(
            annotation_event.get("cache_valid_field", "semantic_cache_valid")
        ),
        "architecture": str(mot.get("architecture", "legacy")).lower(),
        "interaction_mode": str(mot.get("interaction_mode", "base")).lower(),
        "world_attention_mask_mode": str(
            mot.get("world_attention_mask_mode", "first_frame_causal")
        ).lower(),
        "layerwise_planner_coupling": bool(
            mot.get("layerwise_planner_coupling", False)
        ),
        "world_grid_height": int(mot.get("world_grid_height", 12)),
        "world_grid_width": int(mot.get("world_grid_width", 10)),
        "world_infer_shift": float(mot.get("world_infer_shift", 5.0)),
        "action_infer_shift": float(mot.get("action_infer_shift", 5.0)),
        "world_num_train_timesteps": int(
            mot.get("world_num_train_timesteps", 1000)
        ),
        "action_num_train_timesteps": int(
            mot.get("action_num_train_timesteps", 1000)
        ),
        # The current QwenWorldActionMoT merged default is ten.
        "num_inference_timesteps": int(
            mot.get("num_inference_timesteps", 10)
        ),
        "action_prediction_type": str(
            mot.get("action_prediction_type", "velocity")
        ).lower(),
        "action_velocity_target": str(
            mot.get("action_velocity_target", "noise_minus_clean")
        ).lower(),
        "action_horizon": int(action.get("action_horizon", 16)),
        "action_dim": int(action.get("action_dim", 14)),
        "state_dim": int(action.get("state_dim", 0) or 0),
        "dino_image_size": tuple(dino.get("image_size", [384, 320])),
        "dino_patch_size": int(dino.get("patch_size", 16)),
        "dino_embed_dim": int(dino.get("embed_dim", 768)),
        "dino_pool": int(dino.get("dino_pool", 2)),
        "current_dino_pool": (
            None
            if dino.get("current_dino_pool", None) is None
            else int(dino.get("current_dino_pool"))
        ),
        "dino_stats_path": dino.get("stats_path"),
        "include_state": bool(data.get("include_state", False)),
        "obs_image_size": tuple(data.get("obs_image_size", [])),
    }


def _validate_construction_inference_semantics(
    construction_config: dict,
    contract_config: dict,
) -> None:
    construction = _resolved_mot_inference_semantics(construction_config)
    contract = _resolved_mot_inference_semantics(contract_config)
    mismatches = {
        key: {
            "config.yaml": construction[key],
            "contract": contract[key],
        }
        for key in contract
        if construction[key] != contract[key]
    }
    if mismatches:
        raise ValueError(
            "config.yaml would construct inference semantics that differ from "
            f"the saved full training contract: {mismatches}"
        )


def _validate_released_50k_contract(
    *,
    run_dir: Path,
    framework: dict,
    mot: dict,
    weight_contract: dict,
    text_planning_enabled: bool,
) -> dict | None:
    expected = _RELEASED_50K_CONTRACTS.get(run_dir.name)
    if expected is None:
        return None
    _expect(
        str(mot.get("interaction_mode", "")).lower(),
        "joint",
        f"{run_dir.name} interaction_mode",
    )
    _expect(
        weight_contract["version"],
        "legacy_shared_context_state_world_v1",
        f"{run_dir.name} physical checkpoint generation",
    )
    _expect(
        text_planning_enabled,
        bool(expected["text"]),
        f"{run_dir.name} text supervision",
    )
    _expect(
        str(mot.get("action_prediction_type", "velocity")).lower(),
        "velocity",
        f"{run_dir.name} action prediction type",
    )
    _expect(
        str(
            mot.get(
                "action_velocity_target",
                "noise_minus_clean",
            )
        ).lower(),
        "noise_minus_clean",
        f"{run_dir.name} action velocity target",
    )
    _expect(
        int(mot.get("repeated_diffusion_steps", 1)),
        1,
        f"{run_dir.name} repeated diffusion steps",
    )
    base_vlm = str((framework.get("qwenvl") or {}).get("base_vlm", "")).lower()
    planner = (
        "rynnbrain"
        if "rynnbrain" in base_vlm
        else "qwen3"
        if "qwen3" in base_vlm
        else "unknown"
    )
    _expect(
        planner,
        expected["planner"],
        f"{run_dir.name} planner family",
    )
    return {
        "run": run_dir.name,
        "planner": planner,
        "text": text_planning_enabled,
        "physical": weight_contract["version"],
    }


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

    with accessed_config.open("r", encoding="utf-8") as handle:
        construction_config = yaml.safe_load(handle)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with stats_path.open("r", encoding="utf-8") as handle:
        all_stats = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Checkpoint config is not a mapping: {config_path}")
    if not isinstance(construction_config, dict):
        raise TypeError(
            f"Checkpoint construction config is not a mapping: {accessed_config}"
        )
    if not isinstance(all_stats, dict):
        raise TypeError(f"Checkpoint dataset statistics are not a mapping: {stats_path}")

    framework = config.get("framework") or {}
    action = framework.get("action_model") or {}
    data = ((config.get("datasets") or {}).get("vla_data") or {})
    framework_name = str(framework.get("name", ""))
    mot_action_recipe = None
    mot_weight_contract = None
    text_planning_enabled = False
    event_memory_enabled = False
    event_memory_contract = None
    released_50k_contract = None
    if framework_name not in {"QwenGR00T", "QwenWorldActionMoT"}:
        raise ValueError(
            "RoboDojo checkpoint framework.name="
            f"{framework_name!r}; expected QwenGR00T or QwenWorldActionMoT."
        )
    _expect(int(action.get("action_dim", -1)), 14, "framework.action_model.action_dim")
    action_horizon = int(action.get("action_horizon", -1))
    if framework_name == "QwenWorldActionMoT":
        if action_horizon <= 0:
            raise ValueError(
                "framework.action_model.action_horizon must be positive, "
                f"got {action_horizon}"
            )
    else:
        _expect(action_horizon, 16, "framework.action_model.action_horizon")
    if framework_name == "QwenWorldActionMoT":
        _validate_construction_inference_semantics(
            construction_config,
            config,
        )
        mot = framework.get("world_action_mot") or {}
        _expect(
            bool(framework.get("enable_world_action_mot", False)),
            True,
            "framework.enable_world_action_mot",
        )
        if str(mot.get("architecture", "legacy")).lower() == "causal_dino_mot":
            action_precision_mode = str(
                mot.get("action_precision_mode", "inherit")
            ).strip().lower()
            if action_precision_mode not in {"inherit", "fp32_shell"}:
                raise ValueError(
                    "framework.world_action_mot.action_precision_mode must be "
                    "'inherit' or 'fp32_shell', got "
                    f"{action_precision_mode!r}"
                )
            construction_mot = (
                (construction_config.get("framework") or {}).get(
                    "world_action_mot"
                )
                or {}
            )
            _expect(
                str(
                    construction_mot.get(
                        "action_precision_mode",
                        "inherit",
                    )
                ).strip().lower(),
                action_precision_mode,
                "config.yaml framework.world_action_mot.action_precision_mode",
            )
            shapes = _checkpoint_tensor_shapes(checkpoint)
            mot_weight_contract = _infer_causal_mot_weight_contract(shapes)
            multires_world_input = _validate_causal_construction_config(
                construction_config,
                mot_weight_contract,
            )
            if multires_world_input:
                mot_weight_contract["multires_world_input"] = True
                mot_weight_contract["version"] = (
                    "layerwise_query_only_multires_world_v3"
                    if mot_weight_contract["layerwise_planner_coupling"]
                    else "legacy_planner_multires_world_v3"
                )
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
            layerwise_coupling = bool(
                mot.get("layerwise_planner_coupling", False)
            )
            base_vlm = str((framework.get("qwenvl") or {}).get("base_vlm", ""))
            layerwise_depth = 24 if "rynnbrain" in base_vlm.lower() else 28
            if layerwise_coupling:
                profile_fields = (
                    "world_hidden_size",
                    "action_hidden_size",
                    "world_ffn_dim",
                    "action_ffn_dim",
                    "num_layers",
                    "num_attention_heads",
                    "attention_head_dim",
                )
                layerwise_profiles = {
                    "base": (640, 1024, 2560, 4096, layerwise_depth, 16, 64),
                    # Large-capacity 24-layer Rynn recipe. Wider FFNs keep its
                    # Action/World parameter counts aligned with the released
                    # 30-layer non-layerwise model.
                    "base_L": (
                        512,
                        1024,
                        4352,
                        6656,
                        layerwise_depth,
                        24,
                        128,
                    ),
                }
                configured_profile = tuple(
                    int(mot.get(field, -1))
                    for field in profile_fields
                )
                matched_profile = next(
                    (
                        name
                        for name, values in layerwise_profiles.items()
                        if configured_profile == values
                    ),
                    None,
                )
                if matched_profile is None:
                    known = {
                        name: dict(zip(profile_fields, values, strict=True))
                        for name, values in layerwise_profiles.items()
                    }
                    raise ValueError(
                        "Unsupported layer-wise causal-MoT architecture: "
                        f"{dict(zip(profile_fields, configured_profile, strict=True))}; "
                        f"supported profiles={known}"
                    )
                architecture_contract = tuple(
                    zip(
                        profile_fields,
                        layerwise_profiles[matched_profile],
                        strict=True,
                    )
                )
                mot_weight_contract["architecture_profile"] = matched_profile
            else:
                architecture_contract = (
                    # Released 50k/80k causal-MoT checkpoints.
                    ("world_hidden_size", 512),
                    ("action_hidden_size", 1024),
                    ("world_ffn_dim", 2048),
                    ("action_ffn_dim", 4096),
                    ("num_layers", 30),
                    ("num_attention_heads", 24),
                    ("attention_head_dim", 128),
                )
                mot_weight_contract["architecture_profile"] = "released_legacy"
            for field, expected in (
                *architecture_contract,
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
            if layerwise_coupling:
                _expect(
                    int((framework.get("qwenvl") or {}).get("truncate_vlm_layers", -1)),
                    layerwise_depth,
                    "framework.qwenvl.truncate_vlm_layers",
                )
                _expect(
                    int(mot.get("num_inference_timesteps", -1)),
                    10,
                    "framework.world_action_mot.num_inference_timesteps",
                )
            else:
                legacy_inference_steps = int(
                    mot.get("num_inference_timesteps", 10)
                )
                if legacy_inference_steps not in {10, 20}:
                    raise ValueError(
                        "Released causal-MoT checkpoints support saved inference "
                        "steps 10 or 20; RoboDojo evaluation explicitly overrides "
                        f"to 10, got saved={legacy_inference_steps}"
                    )
            _expect(
                bool(mot_weight_contract["layerwise_planner_coupling"]),
                layerwise_coupling,
                "checkpoint layerwise context layout",
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
            # Missing fields identify legacy 50k/80k causal-MoT checkpoints;
            # preserve their scheduler-velocity/repeat=1 contract.
            action_prediction_type = str(
                mot.get("action_prediction_type", "velocity")
            ).lower()
            if action_prediction_type not in {"velocity", "jit_x"}:
                raise ValueError(
                    "framework.world_action_mot.action_prediction_type must be "
                    f"'velocity' or 'jit_x', got {action_prediction_type!r}"
                )
            action_velocity_target = str(
                mot.get(
                    "action_velocity_target",
                    "noise_minus_clean",
                )
            ).lower()
            if action_velocity_target not in {
                "clean_minus_noise",
                "noise_minus_clean",
            }:
                raise ValueError(
                    "framework.world_action_mot.action_velocity_target must be "
                    "'clean_minus_noise' or 'noise_minus_clean', got "
                    f"{action_velocity_target!r}"
                )
            repeated_diffusion_steps = int(
                mot.get("repeated_diffusion_steps", 1)
            )
            if repeated_diffusion_steps <= 0:
                raise ValueError(
                    "framework.world_action_mot.repeated_diffusion_steps must "
                    "be positive"
                )
            jit_t_eps = float(mot.get("jit_t_eps", 0.05))
            if jit_t_eps <= 0:
                raise ValueError(
                    "framework.world_action_mot.jit_t_eps must be positive"
                )
            mot_action_recipe = {
                "prediction_type": action_prediction_type,
                "velocity_target": action_velocity_target,
                "loss": "velocity_mse",
                "repeated_diffusion_steps": repeated_diffusion_steps,
                "jit_t_eps": jit_t_eps,
                "world_loss_weight": float(
                    mot.get("world_loss_weight", 1.0)
                ),
                "saved_num_inference_steps": int(
                    mot.get("num_inference_timesteps", 10)
                ),
            }
            planner = framework.get("planner") or {}
            text_cfg = planner.get("text_supervision") or {}
            text_planning_enabled = bool(text_cfg.get("enabled", False))
            text_loss_weight = float(mot.get("text_loss_weight", 0.0))
            if text_planning_enabled != (text_loss_weight > 0.0):
                raise ValueError(
                    "RoboDojo text checkpoint contract requires "
                    "planner.text_supervision.enabled exactly when "
                    "world_action_mot.text_loss_weight > 0"
                )
            annotations = data.get("text_annotations") or {}
            if text_planning_enabled and not bool(annotations.get("enabled", False)):
                raise ValueError(
                    "RoboDojo text checkpoint requires "
                    "datasets.vla_data.text_annotations.enabled=true"
                )
            event_cfg = annotations.get("event_memory") or {}
            event_enabled = bool(event_cfg.get("enabled", False))
            event_mode = str(
                text_cfg.get("mode", "legacy_full_text")
            ).lower() == "event_driven_memory_ntp"
            if event_enabled != event_mode:
                raise ValueError(
                    "Event-memory checkpoint requires matching planner mode and "
                    "dataset event_memory.enabled"
                )
            if event_enabled:
                if bool((text_cfg.get("history") or {}).get("enabled", False)):
                    raise ValueError("Event-memory checkpoint must not use RGB history")
                if int(event_cfg.get("semantic_offset", 0)) != -10 or int(
                    event_cfg.get("replan_interval", 0)
                ) != 10:
                    raise ValueError(
                        "RoboDojo event-memory checkpoint requires t-10 labels "
                        "and 10-step action replanning"
                    )
                if int(text_cfg.get("max_new_tokens", 0)) < 66:
                    raise ValueError(
                        "Event-memory max_new_tokens is below the audited 66-token maximum"
                    )
                event_memory_contract = {
                    "semantic_offset": int(event_cfg["semantic_offset"]),
                    "replan_interval": int(event_cfg["replan_interval"]),
                    "keep_token": str(text_cfg.get("keep_token", "<KEEP>")),
                    "update_token": str(
                        text_cfg.get("update_token", "<UPDATE>")
                    ),
                    "max_new_tokens": int(text_cfg["max_new_tokens"]),
                    "rgb_history": False,
                }
            event_memory_enabled = event_enabled
            released_50k_contract = _validate_released_50k_contract(
                run_dir=run_dir,
                framework=framework,
                mot=mot,
                weight_contract=mot_weight_contract,
                text_planning_enabled=text_planning_enabled,
            )
        else:
            _expect(
                mot.get("attention_pattern"),
                "alternating_condition_joint",
                "framework.world_action_mot.attention_pattern",
            )
        expected_state_dim = (
            int(mot_weight_contract["state_dim"])
            if mot_weight_contract is not None
            else 14
        )
        _expect(
            int(action.get("state_dim", 0) or 0),
            expected_state_dim,
            "framework.action_model.state_dim",
        )
        _expect(
            bool(data.get("include_state", False)),
            expected_state_dim > 0,
            "datasets.vla_data.include_state",
        )
        _expect(bool(data.get("online_dino", False)), True, "datasets.vla_data.online_dino")
        _expect(bool(data.get("decode_future_video", False)), True, "datasets.vla_data.decode_future_video")
    else:
        # Preserve the historical baseline/WAM checkpoint contract verbatim.
        _expect(int(action.get("state_dim", -1)), 14, "framework.action_model.state_dim")
        _expect(bool(data.get("include_state", False)), True, "datasets.vla_data.include_state")
    _expect(
        int(data.get("action_horizon", action_horizon)),
        action_horizon,
        "datasets.vla_data.action_horizon vs framework action_horizon",
    )
    data_mix = data.get("data_mix")
    # ``robodojo_v21_language_optional`` was used by runs where the auxiliary
    # planning annotations were optional.  It has the same RoboDojo physical
    # ABI (views, state/action order, normalization and horizon) as the other
    # v2.1 mixtures, so it is valid for policy deployment.
    allowed_data_mixes = {
        "robodojo_v21",
        "robodojo_v21_language",
        "robodojo_v21_language_optional",
    }
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
    if bool(data.get("include_state", False)):
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
            expected_size=action_horizon * 14,
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
        "state_action_normalization": (
            "state+action fastwam_zscore via new_embodiment statistics"
            if bool(data.get("include_state", False))
            else "state disabled; action fastwam_zscore via new_embodiment statistics"
        ),
        "image": "head+left_wrist+right_wrist -> FastWAM 320x384 composite",
        "action_chunk": [action_horizon, 14],
        "use_correlated_noise": uses_correlated_noise,
        "correlation_shape": correlation_shape,
        "mot_action_recipe": mot_action_recipe,
        "mot_weight_contract": mot_weight_contract,
        "text_planning_enabled": text_planning_enabled,
        "event_memory_enabled": event_memory_enabled,
        "event_memory_contract": event_memory_contract,
        "released_50k_contract": released_50k_contract,
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
