"""Frozen contract for the high-score RoboDojo RynnBrain causal-MoT run.

The original 50k recipe was later replaced by a layer-wise, state-free 80k
recipe under the same short YAML names. This module makes the
released recipe an explicit compatibility profile so future refactors cannot
silently reinterpret it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROBODOJO_RYNN50K_PROFILE = "robodojo_rynnbrain11_causal_dino_mot_50k_v1"
ROBODOJO_RYNN50K_BASE_H25_PROFILE = (
    "robodojo_rynnbrain11_causal_dino_mot_base_h25_50k_v1"
)
ROBODOJO_RYNN50K_BASE_H25_CURRENT_DINO_FULLRES_PROFILE = (
    "robodojo_rynnbrain11_causal_dino_mot_base_h25_current_dino_fullres_50k_v1"
)
ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_PROFILE = (
    "robodojo_rynnbrain11_causal_dino_mot_base_text_h25_eventmem_ntp_nohist_fp32_50k_v1"
)
ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_BF16_PROFILE = (
    "robodojo_rynnbrain11_causal_dino_mot_base_text_h25_eventmem_ntp_nohist_bf16_50k_v1"
)
ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_BF16_CURRENT_DINO_FULLRES_PROFILE = (
    "robodojo_rynnbrain11_causal_dino_mot_base_text_h25_eventmem_ntp_nohist_bf16_current_dino_fullres_50k_v1"
)
ROBODOJO_RYNN50K_LEGACY_BASE_TEXT_H25_MEM_PROFILE = (
    "robodojo_rynnbrain11_causal_dino_mot_base_text_h25_mem_currenttoken_fullres_50k"
)
ROBODOJO_RYNN50K_BASE_HISTORY_H25_MEM_PROFILE = (
    "robodojo_rynnbrain11_causal_dino_mot_base_history_h25_mem_currenttoken_fullres_50k"
)
ROBODOJO_RYNN50K_WEIGHT_CONTRACT = "legacy_shared_context_state_world_v1"

# Canonical JSON hash (sorted keys, compact separators) of the statistics next
# to the preserved high-score run.  This catches accidental v1 -> v2 dataset
# alias changes before the first optimizer step.
ROBODOJO_RYNN50K_STATS_SHA256 = (
    "9aa0f70cc17882f8a51e87874ae6a307944ddd09700e65a86760a3604076cb75"
)

_MISSING = object()


def _get_child(value: Any, key: str, default: Any = _MISSING) -> Any:
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            result = getter(key, _MISSING)
        except TypeError:
            result = _MISSING
        if result is not _MISSING:
            return result
    if isinstance(value, dict) and key in value:
        return value[key]
    if hasattr(value, key):
        return getattr(value, key)
    if default is not _MISSING:
        return default
    raise KeyError(key)


def _select(config: Any, path: str, default: Any = _MISSING) -> Any:
    value = config
    for key in path.split("."):
        try:
            value = _get_child(value, key)
        except KeyError:
            if default is not _MISSING:
                return default
            raise KeyError(path) from None
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "items"):
        try:
            return {str(key): _plain(item) for key, item in value.items()}
        except TypeError:
            pass
    if isinstance(value, (list, tuple)) or (
        hasattr(value, "__iter__") and not isinstance(value, (str, bytes))
    ):
        return [_plain(item) for item in value]
    return value


# These baseline fields define the physical contract inherited by the H25
# recipes. Interaction mode is checked separately so legacy checkpoints remain
# loadable, while every maintained H25 recipe requires base masking.
#
# ``trainer.is_resume`` and ``trainer.resume_state_path`` are deliberately not
# part of this contract. They describe how a particular process restores an
# already-defined training run, and therefore legitimately differ between an
# initial launch, a resumed launch, and checkpoint evaluation. In particular,
# they must not affect validation or the config fingerprint stored with weights.
_EXPECTED = {
    "run_root_dir": "/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robodojo",
    "seed": 42,
    "version_id": "0.21",
    "framework.name": "QwenWorldActionMoT",
    "framework.enable_world_action_mot": True,
    "framework.reproduction_profile": ROBODOJO_RYNN50K_PROFILE,
    "framework.qwenvl.base_vlm": "/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/rynnbrain1.1-2B/",
    "framework.qwenvl.attn_implementation": "sdpa",
    "framework.qwenvl.require_attn_implementation": True,
    "framework.qwenvl.enable_gradient_checkpointing": True,
    "framework.qwenvl.enable_thinking": False,
    "framework.qwenvl.vl_hidden_dim": 2048,
    "framework.qwenvl.truncate_vlm_layers": 0,
    "framework.planner.num_world_queries": 16,
    "framework.planner.num_action_queries": 16,
    "framework.planner.world_placeholder_token": "<WORLD_PLAN>",
    "framework.planner.action_placeholder_token": "<ACTION_PLAN>",
    "framework.planner.text_supervision.enabled": False,
    "framework.world_action_mot.architecture": "causal_dino_mot",
    "framework.world_action_mot.world_attention_mask_mode": "first_frame_causal",
    "framework.world_action_mot.world_hidden_size": 512,
    "framework.world_action_mot.action_hidden_size": 1024,
    "framework.world_action_mot.world_ffn_dim": 2048,
    "framework.world_action_mot.action_ffn_dim": 4096,
    "framework.world_action_mot.num_layers": 30,
    "framework.world_action_mot.num_attention_heads": 24,
    "framework.world_action_mot.attention_head_dim": 128,
    "framework.world_action_mot.layerwise_planner_coupling": False,
    "framework.world_action_mot.norm_eps": 1.0e-6,
    "framework.world_action_mot.time_frequency_dim": 256,
    "framework.world_action_mot.world_grid_height": 12,
    "framework.world_action_mot.world_grid_width": 10,
    "framework.world_action_mot.max_world_tokens": 120,
    "framework.world_action_mot.enable_gradient_checkpointing": True,
    "framework.world_action_mot.world_train_shift": 5.0,
    "framework.world_action_mot.world_infer_shift": 5.0,
    "framework.world_action_mot.world_num_train_timesteps": 1000,
    "framework.world_action_mot.action_train_shift": 5.0,
    "framework.world_action_mot.action_infer_shift": 5.0,
    "framework.world_action_mot.action_num_train_timesteps": 1000,
    "framework.world_action_mot.num_inference_timesteps": 20,
    "framework.world_action_mot.action_prediction_type": "velocity",
    "framework.world_action_mot.action_velocity_target": "noise_minus_clean",
    "framework.world_action_mot.jit_t_eps": 0.05,
    "framework.world_action_mot.repeated_diffusion_steps": 1,
    "framework.world_action_mot.action_loss_weight": 1.0,
    "framework.world_action_mot.world_loss_weight": 1.0,
    "framework.world_action_mot.text_loss_weight": 0.0,
    "framework.dino.name": "dinov3_vitb16",
    "framework.dino.model_size": "base",
    "framework.dino.hf_model_id": "/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/DINO-B/",
    "framework.dino.repo_or_dir": "facebookresearch/dinov3",
    "framework.dino.weights": "/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/DINO-B/",
    "framework.dino.loader": "hf",
    "framework.dino.image_size": [384, 320],
    "framework.dino.patch_size": 16,
    "framework.dino.embed_dim": 768,
    "framework.dino.stats_path": None,
    "framework.dino.load_live_backbone": True,
    "framework.dino.force_online": True,
    "framework.dino.dino_pool": 2,
    "framework.dino.current_dino_pool": None,
    "framework.action_model.action_dim": 14,
    "framework.action_model.state_dim": 14,
    "framework.action_model.action_horizon": 16,
    "datasets.vla_data.dataset_py": "jointflow",
    "datasets.vla_data.data_root_dir": "/horizon-bucket/robot_lab/users/sen.wang-labs/RoboDojo",
    "datasets.vla_data.data_mix": "robodojo_v21_language_optional",
    "datasets.vla_data.lerobot_version": "v2.1",
    "datasets.vla_data.image_layout": "tri_view_composite",
    "datasets.vla_data.composite_source_view_keys": [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ],
    "datasets.vla_data.composite_view_key": "video.tri_view_composite",
    "datasets.vla_data.include_state": True,
    "datasets.vla_data.action_type": "abs_qpos",
    "datasets.vla_data.action_mode": "abs",
    "datasets.vla_data.action_horizon": 16,
    "datasets.vla_data.world_model.future_stride": 16,
    "datasets.vla_data.future_valid_requires_full_stride": True,
    "datasets.vla_data.decode_future_video": True,
    "datasets.vla_data.online_dino": True,
    "datasets.vla_data.dino_target_latents": False,
    "datasets.vla_data.optional_text_annotations.enabled": False,
    "datasets.vla_data.optional_text_annotations.require_columns": False,
    "datasets.vla_data.text_annotations.enabled": False,
    "datasets.vla_data.sequential_step_sampling": False,
    "datasets.vla_data.balance_dataset_weights": False,
    "datasets.vla_data.balance_trajectory_weights": False,
    "datasets.vla_data.per_device_batch_size": 12,
    "datasets.vla_data.load_all_data_for_training": True,
    "datasets.vla_data.obs_image_size": [320, 384],
    "datasets.vla_data.video_backend": "pyav",
    "datasets.vla_data.num_workers": 8,
    "datasets.vla_data.prefetch_factor": 2,
    "datasets.vla_data.pin_memory": True,
    "datasets.vla_data.persistent_workers": False,
    "trainer.seed_before_model_init": True,
    "trainer.expected_global_batch_size": 768,
    "trainer.max_train_steps": 50000,
    "trainer.num_warmup_steps": 2000,
    "trainer.save_interval": 10000,
    "trainer.eval_interval": 1000,
    "trainer.action_eval_enabled": True,
    "trainer.world_validation.enabled": False,
    "trainer.pretrained_checkpoint": None,
    "trainer.save_full_training_state": True,
    "trainer.learning_rate.base": 1.0e-5,
    "trainer.learning_rate.qwen_vl_interface": 1.0e-5,
    "trainer.learning_rate.action_plan_queries": 1.0e-4,
    "trainer.learning_rate.world_plan_queries": 1.0e-4,
    "trainer.learning_rate.action_model": 1.0e-4,
    "trainer.lr_scheduler_type": "cosine_with_min_lr",
    "trainer.scheduler_specific_kwargs.min_lr": 5.0e-7,
    "trainer.freeze_modules": "",
    "trainer.loss_scale.vla": 1.0,
    "trainer.max_grad_norm": 1.0,
    "trainer.weight_decay": 0.0,
    "trainer.logging_frequency": 200,
    "trainer.log_grad_norms": False,
    "trainer.log_global_grad_norm": True,
    "trainer.deepspeed_skip_unused_param_anchors": True,
    "trainer.gradient_clipping": 1.0,
    "trainer.gradient_accumulation_steps": 1,
    "trainer.optimizer.name": "AdamW",
    "trainer.optimizer.betas": [0.9, 0.95],
    "trainer.optimizer.eps": 1.0e-8,
    "trainer.optimizer.weight_decay": 1.0e-8,
}

_BASE_H25_EXPECTED = {
    **_EXPECTED,
    "framework.reproduction_profile": ROBODOJO_RYNN50K_BASE_H25_PROFILE,
    "framework.planner.num_action_queries": 25,
    "framework.action_model.action_horizon": 25,
    "datasets.vla_data.action_horizon": 25,
}

_BASE_H25_CURRENT_DINO_FULLRES_EXPECTED = {
    **_BASE_H25_EXPECTED,
    "framework.reproduction_profile": (
        ROBODOJO_RYNN50K_BASE_H25_CURRENT_DINO_FULLRES_PROFILE
    ),
    # Keep the future denoising target on the historical 12x10 pool=2 grid,
    # while the clean current prefix uses the full 24x20 DINO patch grid.
    "framework.dino.current_dino_pool": 1,
}

_LEGACY_BASE_TEXT_H25_MEM_EXPECTED = {
    **_EXPECTED,
    "framework.reproduction_profile": ROBODOJO_RYNN50K_LEGACY_BASE_TEXT_H25_MEM_PROFILE,
    # The complete Rynn tower uses the stable SDPA backend; every fourth MEM
    # block performs explicit temporal-then-spatial SDPA with the same Q/K/V.
    "framework.qwenvl.attn_implementation": "sdpa",
    "framework.qwenvl.mem_vision_encoder.enabled": True,
    "framework.qwenvl.mem_vision_encoder.num_frames": 6,
    "framework.qwenvl.mem_vision_encoder.spacetime_layer_stride": 4,
    "framework.qwenvl.mem_vision_encoder.time_embed_base": 100.0,
    "framework.qwenvl.mem_vision_encoder.match_current_to_history_resolution": True,
    "framework.qwenvl.mem_vision_encoder.output_token_policy": "current_only",
    "framework.planner.num_action_queries": 25,
    "framework.planner.text_supervision.enabled": True,
    "framework.planner.text_supervision.subtask_field": "subtask_text",
    "framework.planner.text_supervision.completed_subtask_field": "completed_subtask_text",
    "framework.planner.text_supervision.prompt_template": (
        "Task instruction: {instruction}\n"
        "Previous Finished Task List: {finished_task_list}\n"
        "Using the history observations and current observation, report the "
        "current subtask and update the Finished Task List. Preserve every "
        "previously finished task and append only tasks that are visibly completed."
    ),
    "framework.planner.text_supervision.response_template": (
        "Current subtask: {subtask_text}\n"
        "Finished Task List: {completed_subtask_text}"
    ),
    "framework.planner.text_supervision.max_new_tokens": 96,
    "framework.planner.text_supervision.do_sample": False,
    "framework.planner.text_supervision.history.enabled": True,
    "framework.planner.text_supervision.history.image_field": "planner_history_images",
    "framework.planner.text_supervision.history.finished_task_list_field": "finished_task_list",
    "framework.planner.text_supervision.history.finished_task_list_prefix": "Finished Task List:",
    "framework.planner.text_supervision.history.empty_finished_task_list": "None",
    # [width, height]: keep every sparse-history/current composite at the
    # native RoboDojo observation resolution before the MEM vision tower.
    "framework.planner.text_supervision.history.history_image_size": [320, 384],
    "framework.world_action_mot.text_loss_weight": 0.005,
    "framework.action_model.action_horizon": 25,
    "datasets.vla_data.data_mix": "robodojo_v21_language",
    "datasets.vla_data.action_horizon": 25,
    "datasets.vla_data.text_annotations.enabled": True,
    "datasets.vla_data.text_annotations.fields.subtask_text": "subtask_text",
    "datasets.vla_data.text_annotations.fields.completed_subtask_text": "complete_text",
    "datasets.vla_data.text_annotations.history.enabled": True,
    "datasets.vla_data.text_annotations.history.image_field": "planner_history_images",
    "datasets.vla_data.text_annotations.history.frame_offsets": [
        -100,
        -80,
        -60,
        -40,
        -20,
    ],
    "datasets.vla_data.text_annotations.history.memory_offset": -100,
    "datasets.vla_data.text_annotations.history.finished_task_list_source_field": "complete_text",
    "datasets.vla_data.text_annotations.history.finished_task_list_field": "finished_task_list",
    "datasets.vla_data.text_annotations.history.empty_finished_task_list": "None",
    "trainer.action_eval_enabled": False,
}

# Strict text-plan ablation: preserve the full text+MEM recipe and its exact
# dataset/history inputs, but remove autoregressive text planning and NTP loss.
_BASE_HISTORY_H25_MEM_EXPECTED = {
    **_LEGACY_BASE_TEXT_H25_MEM_EXPECTED,
    "framework.reproduction_profile": ROBODOJO_RYNN50K_BASE_HISTORY_H25_MEM_PROFILE,
    "framework.planner.text_supervision.enabled": False,
    "framework.world_action_mot.text_loss_weight": 0.0,
}


_BASE_TEXT_H25_MEM_EXPECTED = {
    **_BASE_H25_EXPECTED,
    "framework.reproduction_profile": ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_PROFILE,
    "framework.planner.text_supervision.enabled": True,
    "framework.planner.text_supervision.mode": "event_driven_memory_ntp",
    "framework.planner.text_supervision.keep_token": "<KEEP>",
    "framework.planner.text_supervision.update_token": "<UPDATE>",
    "framework.planner.text_supervision.subtask_field": "subtask_text",
    "framework.planner.text_supervision.prompt_template": (
        "Task instruction: {instruction}\n"
        "Semantic Memory: {semantic_memory}\n"
        "Cached Current Subtask: {cached_current_subtask}\n"
        "Decide whether the cached subtask is still valid. Reply with <KEEP> "
        "only when it remains valid. Otherwise begin with <UPDATE>, then provide "
        "only `Memory Add:` and `Current Subtask:`."
    ),
    "framework.planner.text_supervision.update_response_template": (
        "{update_token}\n"
        "Memory Add: {memory_add}\n"
        "Current Subtask: {subtask_text}"
    ),
    "framework.planner.text_supervision.max_new_tokens": 96,
    "framework.planner.text_supervision.do_sample": False,
    "framework.planner.text_supervision.scheduled_sampling.enabled": True,
    "framework.planner.text_supervision.scheduled_sampling.points": [
        [0, 0.0],
        [30000, 0.0],
        [40000, 0.2],
        [50000, 0.5],
    ],
    "framework.planner.text_supervision.history.enabled": False,
    "framework.world_action_mot.action_precision_mode": "fp32_shell",
    "framework.world_action_mot.text_loss_weight": 0.005,
    "datasets.vla_data.data_mix": "robodojo_v21_language",
    "datasets.vla_data.text_annotations.enabled": True,
    "datasets.vla_data.text_annotations.fields.subtask_text": "subtask_text",
    "datasets.vla_data.text_annotations.fields.completed_subtask_text": "complete_text",
    "datasets.vla_data.text_annotations.history.enabled": False,
    "datasets.vla_data.text_annotations.event_memory.enabled": True,
    "datasets.vla_data.text_annotations.event_memory.semantic_offset": -10,
    "datasets.vla_data.text_annotations.event_memory.replan_interval": 10,
    "datasets.vla_data.text_annotations.event_memory.replan_phase": 0,
    "datasets.vla_data.text_annotations.event_memory.normalization": "n1",
    "datasets.vla_data.text_annotations.event_memory.empty_memory": "None.",
    "datasets.vla_data.text_annotations.event_memory.empty_cached_subtask": "None.",
    "datasets.vla_data.text_annotations.event_memory.semantic_memory_field": "semantic_memory",
    "datasets.vla_data.text_annotations.event_memory.cached_subtask_field": "cached_current_subtask",
    "datasets.vla_data.text_annotations.event_memory.decision_field": "semantic_decision",
    "datasets.vla_data.text_annotations.event_memory.memory_add_field": "memory_add",
    "datasets.vla_data.text_annotations.event_memory.cache_valid_field": "semantic_cache_valid",
    "datasets.vla_data.text_annotations.event_memory.index_cache_name": "semantic_index_v1.npz",
    "datasets.vla_data.text_annotations.event_memory.expected_phase_update_ratio": 0.064,
    "datasets.vla_data.text_annotations.event_memory.expected_phase_update_tolerance": 0.005,
    "datasets.vla_data.text_annotations.event_memory.sampler.per_device_batch_size": 6,
    "datasets.vla_data.text_annotations.event_memory.sampler.update_per_batch": 2,
    "datasets.vla_data.text_annotations.event_memory.sampler.hard_keep_per_batch": 2,
    "datasets.vla_data.text_annotations.event_memory.sampler.random_keep_per_batch": 2,
    "datasets.vla_data.text_annotations.event_memory.sampler.seed": 42,
    "datasets.vla_data.text_annotations.event_memory.sampler.num_workers": 4,
    "datasets.vla_data.text_annotations.event_memory.sampler.prefetch_factor": 2,
    "datasets.vla_data.text_annotations.event_memory.sampler.pin_memory": True,
    "datasets.vla_data.text_annotations.event_memory.sampler.persistent_workers": False,
    "trainer.action_eval_enabled": False,
}

# Exact BF16 counterpart of the event-memory text recipe.  ``inherit`` makes
# the action shell follow the launcher/model dtype (BF16 in the maintained
# DeepSpeed and RoboDojo policy-server entrypoints) instead of restoring the
# explicit FP32 action boundary.
_BASE_TEXT_H25_MEM_BF16_EXPECTED = {
    **_BASE_TEXT_H25_MEM_EXPECTED,
    "framework.reproduction_profile": (
        ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_BF16_PROFILE
    ),
    "framework.world_action_mot.action_precision_mode": "inherit",
}

# Strict dense-current counterpart of the BF16 event-memory recipe.  Only the
# clean current prefix changes to the full 24x20 grid; the future denoising
# target remains pool=2 on the historical 12x10 physical world grid.
_BASE_TEXT_H25_MEM_BF16_CURRENT_DINO_FULLRES_EXPECTED = {
    **_BASE_TEXT_H25_MEM_BF16_EXPECTED,
    "framework.reproduction_profile": (
        ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_BF16_CURRENT_DINO_FULLRES_PROFILE
    ),
    "framework.dino.current_dino_pool": 1,
}

# Fields added later as explicit pins.  The preserved config.full.yaml omitted
# them only because the historical implementation supplied these defaults.
_HISTORICAL_DEFAULTS = {
    "framework.reproduction_profile": ROBODOJO_RYNN50K_PROFILE,
    "framework.qwenvl.truncate_vlm_layers": 0,
    "framework.world_action_mot.layerwise_planner_coupling": False,
    "framework.world_action_mot.action_prediction_type": "velocity",
    "framework.world_action_mot.action_velocity_target": "noise_minus_clean",
    "framework.world_action_mot.jit_t_eps": 0.05,
    "framework.world_action_mot.repeated_diffusion_steps": 1,
    "framework.dino.current_dino_pool": None,
    "datasets.vla_data.text_annotations.enabled": False,
    "trainer.action_eval_enabled": True,
}


def historical_contract_payload(config: Any, *, include_mode: bool = True) -> dict[str, Any]:
    """Normalize a current recipe or the preserved historical config."""

    payload = {}
    for path, expected in _EXPECTED.items():
        default = _HISTORICAL_DEFAULTS.get(path, _MISSING)
        payload[path] = _plain(_select(config, path, default))
    if include_mode:
        payload["framework.world_action_mot.interaction_mode"] = _plain(
            _select(config, "framework.world_action_mot.interaction_mode")
        )
    return payload


def config_fingerprint(config: Any, *, include_mode: bool = True) -> str:
    payload = historical_contract_payload(config, include_mode=include_mode)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_contract(
    config: Any,
    *,
    expected: dict[str, Any],
    profile: str,
    expected_mode: str | None,
) -> dict[str, Any]:
    mismatches = {}
    for path, expected_value in expected.items():
        try:
            actual = _plain(_select(config, path))
        except KeyError:
            actual = "<MISSING>"
        if actual != expected_value:
            mismatches[path] = {"actual": actual, "expected": expected_value}

    mode = str(
        _select(config, "framework.world_action_mot.interaction_mode", "")
    ).lower()
    if mode not in {"base", "joint"}:
        mismatches["framework.world_action_mot.interaction_mode"] = {
            "actual": mode,
            "expected": "base or joint",
        }
    if expected_mode is not None and mode != str(expected_mode).lower():
        mismatches["framework.world_action_mot.interaction_mode"] = {
            "actual": mode,
            "expected": str(expected_mode).lower(),
        }

    if mismatches:
        details = "; ".join(
            f"{path}={values['actual']!r} (expected {values['expected']!r})"
            for path, values in sorted(mismatches.items())
        )
        raise ValueError(
            f"Invalid {profile} reproduction contract: {details}"
        )

    return {
        "profile": profile,
        "mode": mode,
        "weight_contract": ROBODOJO_RYNN50K_WEIGHT_CONTRACT,
        "fingerprint": config_fingerprint(config),
        "global_batch_size": 768,
        "max_train_steps": 50000,
    }


def validate_config(config: Any, *, expected_mode: str | None = None) -> dict[str, Any]:
    """Fail fast unless a config is the frozen released-50k contract."""

    return _validate_contract(
        config,
        expected=_EXPECTED,
        profile=ROBODOJO_RYNN50K_PROFILE,
        expected_mode=expected_mode,
    )


def validate_base_h25_config(config: Any) -> dict[str, Any]:
    """Validate the base-mask 25-action variant of the frozen 50k recipe."""

    return _validate_contract(
        config,
        expected=_BASE_H25_EXPECTED,
        profile=ROBODOJO_RYNN50K_BASE_H25_PROFILE,
        expected_mode="base",
    )


def validate_base_h25_current_dino_fullres_config(
    config: Any,
) -> dict[str, Any]:
    """Validate base H25 with dense current and pooled future DINO tokens."""

    return _validate_contract(
        config,
        expected=_BASE_H25_CURRENT_DINO_FULLRES_EXPECTED,
        profile=ROBODOJO_RYNN50K_BASE_H25_CURRENT_DINO_FULLRES_PROFILE,
        expected_mode="base",
    )


def validate_base_text_h25_mem_config(config: Any) -> dict[str, Any]:
    """Validate best-H25 plus explicit FP32/event NTP opt-ins without RGB history."""

    return _validate_contract(
        config,
        expected=_BASE_TEXT_H25_MEM_EXPECTED,
        profile=ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_PROFILE,
        expected_mode="base",
    )


def validate_base_text_h25_mem_bf16_config(config: Any) -> dict[str, Any]:
    """Validate the event-memory text recipe with an inherited BF16 action shell."""

    return _validate_contract(
        config,
        expected=_BASE_TEXT_H25_MEM_BF16_EXPECTED,
        profile=ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_BF16_PROFILE,
        expected_mode="base",
    )


def validate_base_text_h25_mem_bf16_current_dino_fullres_config(
    config: Any,
) -> dict[str, Any]:
    """Validate BF16 event memory with dense current and pooled future DINO."""

    return _validate_contract(
        config,
        expected=_BASE_TEXT_H25_MEM_BF16_CURRENT_DINO_FULLRES_EXPECTED,
        profile=(
            ROBODOJO_RYNN50K_BASE_TEXT_H25_MEM_BF16_CURRENT_DINO_FULLRES_PROFILE
        ),
        expected_mode="base",
    )


def validate_base_history_h25_mem_config(config: Any) -> dict[str, Any]:
    """Validate the six-frame MEM ablation with no text plan or text loss."""

    return _validate_contract(
        config,
        expected=_BASE_HISTORY_H25_MEM_EXPECTED,
        profile=ROBODOJO_RYNN50K_BASE_HISTORY_H25_MEM_PROFILE,
        expected_mode="base",
    )


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_dataset_statistics(path: str | Path) -> str:
    """Require the exact v1 state/action statistics used by the source run."""

    stats_path = Path(path)
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Released-50k dataset statistics are missing: {stats_path}"
        )
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    actual = _canonical_json_sha256(payload)
    if actual != ROBODOJO_RYNN50K_STATS_SHA256:
        embodiment = payload.get("new_embodiment", {}) if isinstance(payload, dict) else {}
        raise ValueError(
            "RoboDojo released-50k dataset statistics changed: "
            f"actual_sha256={actual}, expected_sha256={ROBODOJO_RYNN50K_STATS_SHA256}, "
            f"num_transitions={embodiment.get('num_transitions')}, "
            f"num_trajectories={embodiment.get('num_trajectories')}. "
            "Check that robodojo_v21_language_optional still resolves to "
            "RoboDojo_lerobot_v21_language_v1 and that the dataset was not modified."
        )
    return actual
