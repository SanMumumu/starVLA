"""Checkpoint-aware FastWAM-composite adapter between RoboDojo and StarVLA.

This module deliberately lives in the uploaded StarVLA repository.  The
RoboDojo checkout supplies the simulator and XPolicyLab transport only; its
possibly stale ``policy/starVLA`` directory is never imported.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


# Train/deploy ABI from examples/RoboDojo/train_files/data_registry/data_config.py.
_EXPECTED_STATE_KEYS = [
    "state.left_joints",
    "state.left_gripper",
    "state.right_joints",
    "state.right_gripper",
]
_EXPECTED_ACTION_KEYS = [
    "action.left_joints",
    "action.left_gripper",
    "action.right_joints",
    "action.right_gripper",
]
_EXPECTED_SOURCE_VIEW_KEYS = [
    "video.cam_high",
    "video.cam_left_wrist",
    "video.cam_right_wrist",
]


def _load_fastwam_image_module():
    """Load the lightweight compositor without importing dataloader.__init__.

    The RoboDojo client image intentionally does not carry StarVLA's training
    environment. Importing ``starVLA.dataloader.fastwam_image`` normally first
    executes ``starVLA.dataloader.__init__`` and pulls in training-only packages
    such as Accelerate. Loading this single dependency-light file directly
    keeps the client limited to Torch/TorchVision/PIL, all provided by the
    RoboDojo Isaac environment.
    """

    module_path = Path(__file__).resolve().parents[3] / "deployment/fastwam_image.py"
    spec = importlib.util.spec_from_file_location("_robodojo_fastwam_image", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load FastWAM compositor from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _semantic_memory_items(value: Any, empty_value: str = "None.") -> list[str]:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip(" \t.;")
    empty = re.sub(r"\s+", " ", str(empty_value)).strip(" \t.;").lower()
    if not normalized or normalized.lower() in {empty, "none", "nothing", "n/a"}:
        return []
    pieces = re.split(
        r"(?:\r?\n|\s*[;|]\s*|(?<=[.!?])\s+)", str(value).strip()
    )
    return [
        re.sub(r"\s+", " ", piece).strip(" \t.;")
        for piece in pieces
        if re.sub(r"\s+", " ", piece).strip(" \t.;")
    ]


def _append_semantic_memory(previous: Any, delta: Any, empty_value: str) -> str:
    merged: list[str] = []
    seen = set()
    for value in (previous, delta):
        for item in _semantic_memory_items(value, empty_value):
            key = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
    return empty_value if not merged else ". ".join(merged) + "."


def _decode_image(image: Any) -> np.ndarray:
    """Convert RoboDojo RGB/compressed observations to contiguous HWC uint8."""

    if isinstance(image, (bytes, bytearray, memoryview)):
        image = np.frombuffer(bytes(image), dtype=np.uint8)
    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode compressed RoboDojo image bytes.")
        image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got shape {image.shape}.")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got shape {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return np.ascontiguousarray(image)


def _extract_camera(observation: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    vision = observation.get("vision", {})
    for name in names:
        if name not in vision:
            continue
        camera = vision[name]
        if isinstance(camera, dict):
            for key in ("color", "rgb", "colors"):
                if key in camera:
                    return _decode_image(camera[key])
        else:
            return _decode_image(camera)
    raise KeyError(f"Missing RoboDojo camera; tried {list(names)}; available={list(vision)}")


def _instruction(observation: dict[str, Any], fallback: str) -> str:
    value = observation.get("task_instruction")
    if value is None:
        value = observation.get("instruction", observation.get("instructions"))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is not None and hasattr(value, "item"):
        value = value.item()
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _validate_checkpoint_selection(
    expected_checkpoint: Any,
    metadata: dict[str, Any],
) -> None:
    if not expected_checkpoint:
        return
    server_checkpoint = metadata.get("ckpt_path")
    if not server_checkpoint:
        raise RuntimeError(
            "StarVLA server metadata omitted ckpt_path; cannot verify that "
            "client and server selected the same checkpoint."
        )
    expected_path = Path(str(expected_checkpoint)).expanduser().resolve()
    server_path = Path(str(server_checkpoint)).expanduser().resolve()
    if server_path != expected_path:
        raise RuntimeError(
            "RoboDojo client/server checkpoint mismatch: "
            f"client={expected_path}, server={server_path}."
        )


def _validate_mot_runtime_contract(metadata: dict[str, Any]) -> None:
    """Validate the recipe actually instantiated by a causal-MoT server."""

    if metadata.get("framework_name") != "QwenWorldActionMoT":
        return

    planner_mask_contract = metadata.get("planner_query_mask_contract")
    if planner_mask_contract != "positional_suffix_v2":
        raise RuntimeError(
            "Loaded QwenWorldActionMoT server is running stale planner-mask code: "
            "expected planner_query_mask_contract='positional_suffix_v2', "
            f"got {planner_mask_contract!r}. Update and restart every StarVLA "
            "policy server before evaluation."
        )

    version = metadata.get("mot_contract_version")
    interaction_mode = str(metadata.get("mot_interaction_mode", "")).lower()
    if interaction_mode not in {"base", "joint"}:
        raise RuntimeError(
            "Loaded causal-MoT server has an invalid interaction mode: "
            f"{interaction_mode!r}."
        )

    actual = {
        "layerwise": _as_bool(
            metadata.get("mot_layerwise_planner_coupling", False)
        ),
        "prediction_type": str(
            metadata.get("mot_action_prediction_type", "")
        ).lower(),
        "velocity_target": str(
            metadata.get("mot_action_velocity_target", "")
        ).lower(),
        "multires_world": _as_bool(
            metadata.get("mot_multires_world_input", False)
        ),
    }
    if actual["velocity_target"] not in {
        "clean_minus_noise",
        "noise_minus_clean",
    }:
        raise RuntimeError(
            "Loaded causal-MoT server has an invalid action velocity target: "
            f"{actual['velocity_target']!r}."
        )
    saved_steps = int(metadata.get("mot_num_inference_timesteps", 0) or 0)

    if version == "legacy_shared_context_state_world_v1":
        if actual["prediction_type"] == "jit_x":
            # jit_x converts its clean-action prediction to scheduler velocity.
            expected = {
                "layerwise": False,
                "prediction_type": "jit_x",
                "multires_world": False,
            }
        else:
            expected = {
                "layerwise": False,
                "prediction_type": "velocity",
                "velocity_target": "noise_minus_clean",
                "multires_world": False,
            }
        if saved_steps not in {10, 20}:
            raise RuntimeError(
                "Released causal-MoT checkpoints must save 10 or 20 inference "
                f"steps, got {saved_steps}."
            )
    elif version == "layerwise_query_only_world_v2":
        expected = {
            "layerwise": True,
            "prediction_type": "velocity",
            "multires_world": False,
        }
        if saved_steps != 10:
            raise RuntimeError(
                "Layer-wise causal-MoT checkpoints must train/save with exactly "
                f"10 inference steps, got {saved_steps}."
            )
    elif version == "layerwise_query_only_multires_world_v3":
        expected = {
            "layerwise": True,
            "prediction_type": "velocity",
            "multires_world": True,
        }
        if saved_steps != 10:
            raise RuntimeError(
                "Multi-resolution causal-MoT checkpoints must "
                f"train/save with exactly 10 inference steps, got {saved_steps}."
            )
    elif version == "legacy_planner_multires_world_v3":
        expected = {
            "layerwise": False,
            "prediction_type": "velocity",
            "velocity_target": "noise_minus_clean",
            "multires_world": True,
        }
        if saved_steps not in {10, 20}:
            raise RuntimeError(
                "Legacy-planner multi-resolution causal-MoT checkpoints "
                f"must save 10 or 20 inference steps, got {saved_steps}."
            )
    else:
        raise RuntimeError(
            "Loaded QwenWorldActionMoT server did not publish a supported "
            f"checkpoint contract version: {version!r}."
        )

    mismatches = {
        key: {"loaded": actual[key], "expected": expected_value}
        for key, expected_value in expected.items()
        if actual[key] != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "Loaded causal-MoT train/inference recipe mismatch: "
            f"{mismatches}."
        )
    if actual["multires_world"]:
        current_tokens = int(
            metadata.get("mot_current_dino_tokens", 0) or 0
        )
        future_tokens = int(
            metadata.get("mot_future_dino_tokens", 0) or 0
        )
        if (
            min(current_tokens, future_tokens) <= 0
            or current_tokens <= future_tokens
        ):
            raise RuntimeError(
                "Multi-resolution causal-MoT requires the clean current token "
                "count to exceed the future denoising token count, got "
                f"current={current_tokens}, future={future_tokens}."
            )


class Model(ModelTemplate):
    """XPolicyLab policy facade backed by the active StarVLA websocket server."""

    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg.get("action_type", "joint"))
        if self.action_type != "joint":
            raise ValueError("The RoboDojo StarVLA adapter requires action_type='joint'.")
        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        if not self.env_cfg_type:
            raise ValueError("The RoboDojo StarVLA adapter requires env_cfg_type.")

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )
        expected_action_dim = int(self.model_cfg.get("expected_action_dim", 14))
        # ARX X5 joint observations have a fixed 14-D proprioceptive ABI. This
        # is consulted only for historical checkpoints whose metadata says they
        # consume state.
        self.state_dim = 14
        if self.action_dim != expected_action_dim:
            raise ValueError(
                f"RoboDojo {self.env_cfg_type} has action_dim={self.action_dim}; "
                f"checkpoint adapter expects {expected_action_dim}."
            )

        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
        fastwam_image = _load_fastwam_image_module()
        FASTWAM_COMPOSITE_LAYOUT = fastwam_image.FASTWAM_COMPOSITE_LAYOUT
        FASTWAM_COMPOSITE_SIZE = fastwam_image.FASTWAM_COMPOSITE_SIZE
        FASTWAM_COMPOSITE_VIEW_KEY = fastwam_image.FASTWAM_COMPOSITE_VIEW_KEY
        TRI_VIEW_COMPOSITE_LAYOUT = fastwam_image.TRI_VIEW_COMPOSITE_LAYOUT
        TRI_VIEW_COMPOSITE_VIEW_KEY = fastwam_image.TRI_VIEW_COMPOSITE_VIEW_KEY
        build_robotwin_composite = fastwam_image.build_robotwin_composite

        self._build_composite = build_robotwin_composite
        self.image_size = tuple(int(v) for v in self.model_cfg.get("image_size", FASTWAM_COMPOSITE_SIZE))
        if self.image_size != FASTWAM_COMPOSITE_SIZE:
            raise ValueError(
                f"RoboDojo FastWAM composite must be {FASTWAM_COMPOSITE_SIZE} (width,height), "
                f"got {self.image_size}."
            )

        self.client = WebsocketClientPolicy(
            str(self.model_cfg.get("starvla_server_host", "127.0.0.1")),
            int(self.model_cfg.get("starvla_server_port", 5694)),
        )
        metadata = self.client.get_server_metadata()
        if not isinstance(metadata, dict):
            raise TypeError(f"Invalid StarVLA server metadata: {metadata!r}")
        _validate_checkpoint_selection(
            self.model_cfg.get("expected_checkpoint_path"),
            metadata,
        )
        _validate_mot_runtime_contract(metadata)

        expected_layout = str(self.model_cfg.get("expected_image_layout", FASTWAM_COMPOSITE_LAYOUT))
        expected_key = str(self.model_cfg.get("expected_composite_view_key", FASTWAM_COMPOSITE_VIEW_KEY))
        equivalent_composite_contracts = {
            (FASTWAM_COMPOSITE_LAYOUT, FASTWAM_COMPOSITE_VIEW_KEY),
            (TRI_VIEW_COMPOSITE_LAYOUT, TRI_VIEW_COMPOSITE_VIEW_KEY),
        }
        expected_composite_contract = (expected_layout, expected_key)
        checkpoint_composite_contract = (
            metadata.get("image_layout"),
            metadata.get("composite_view_key"),
        )
        if (
            checkpoint_composite_contract != expected_composite_contract
            and not (
                checkpoint_composite_contract in equivalent_composite_contracts
                and expected_composite_contract in equivalent_composite_contracts
            )
        ):
            raise RuntimeError(
                "Checkpoint composite layout/key="
                f"{checkpoint_composite_contract!r}; "
                f"expected {expected_composite_contract!r}."
            )
        if list(metadata.get("composite_source_view_keys") or []) != _EXPECTED_SOURCE_VIEW_KEYS:
            raise RuntimeError(
                "Checkpoint composite camera order is not [head,left_wrist,right_wrist]: "
                f"{metadata.get('composite_source_view_keys')!r}."
            )

        self.action_chunk_size = int(metadata["action_chunk_size"])
        expected_chunk = int(self.model_cfg.get("expected_action_chunk_size", 16))
        if self.action_chunk_size != expected_chunk:
            raise RuntimeError(
                f"Checkpoint chunk={self.action_chunk_size}; RoboDojo experiment requires {expected_chunk}."
            )
        self.replan_interval = int(self.model_cfg.get("replan_interval", 12))
        if not 1 <= self.replan_interval <= self.action_chunk_size:
            raise ValueError(
                "RoboDojo replan_interval must be in "
                f"[1,{self.action_chunk_size}], got {self.replan_interval}."
            )
        self.rtc_enabled = _as_bool(self.model_cfg.get("rtc_enabled", False))
        self.rtc_overlap = self.action_chunk_size - self.replan_interval
        self.rtc_execution_horizon = int(
            self.model_cfg.get(
                "rtc_execution_horizon",
                max(self.rtc_overlap, 1),
            )
        )
        self.rtc_inference_delay = int(
            self.model_cfg.get("rtc_inference_delay", 1)
        )
        self.rtc_max_guidance_weight = float(
            self.model_cfg.get("rtc_max_guidance_weight", 20.0)
        )
        self.rtc_prefix_attention_schedule = str(
            self.model_cfg.get("rtc_prefix_attention_schedule", "linear")
        ).strip().lower()
        self.rtc_debug_max_replans = int(
            self.model_cfg.get("rtc_debug_max_replans", 8)
        )
        if self.rtc_debug_max_replans < 0:
            raise ValueError("rtc_debug_max_replans must be non-negative")
        if self.rtc_enabled:
            if not _as_bool(metadata.get("mot_rtc_supported", False)):
                raise RuntimeError(
                    "Selected checkpoint/server does not support causal-MoT RTC guidance"
                )
            if self.rtc_overlap <= 0:
                raise ValueError(
                    "RTC requires replan_interval < action_chunk_size so the "
                    "previous chunk has an unexecuted tail"
                )
            if not 1 <= self.rtc_execution_horizon <= self.rtc_overlap:
                raise ValueError(
                    "rtc_execution_horizon must be in "
                    f"[1,{self.rtc_overlap}] for chunk={self.action_chunk_size}, "
                    f"replan={self.replan_interval}; got {self.rtc_execution_horizon}"
                )
            if not 0 <= self.rtc_inference_delay <= self.rtc_execution_horizon:
                raise ValueError(
                    "rtc_inference_delay must be in "
                    f"[0,{self.rtc_execution_horizon}], got {self.rtc_inference_delay}"
                )
            if self.rtc_max_guidance_weight <= 0:
                raise ValueError("rtc_max_guidance_weight must be positive")
            if self.rtc_prefix_attention_schedule not in {
                "exp",
                "linear",
                "ones",
                "zeros",
            }:
                raise ValueError(
                    "rtc_prefix_attention_schedule must be exp, linear, ones, "
                    f"or zeros, got {self.rtc_prefix_attention_schedule!r}"
                )
        if "expects_state" not in metadata:
            raise RuntimeError(
                "Selected checkpoint server did not declare its state-input contract."
            )
        self.expects_state = _as_bool(metadata["expects_state"])
        actual_action_keys = metadata.get("action_keys")
        if actual_action_keys is None or list(actual_action_keys) != _EXPECTED_ACTION_KEYS:
            raise RuntimeError(
                "Checkpoint action_keys="
                f"{actual_action_keys!r}; expected {_EXPECTED_ACTION_KEYS!r}."
            )
        if self.expects_state:
            actual_state_keys = metadata.get("state_keys")
            if actual_state_keys is None or list(actual_state_keys) != _EXPECTED_STATE_KEYS:
                raise RuntimeError(
                    "Checkpoint state_keys="
                    f"{actual_state_keys!r}; expected {_EXPECTED_STATE_KEYS!r}."
                )

        self.unnorm_key = str(self.model_cfg.get("unnorm_key", "new_embodiment"))
        available = list(metadata.get("available_unnorm_keys") or [])
        if self.unnorm_key not in available:
            raise RuntimeError(f"unnorm_key={self.unnorm_key!r} not available; checkpoint has {available}.")
        self.use_ddim = _as_bool(self.model_cfg.get("use_ddim", True))
        self.num_ddim_steps = int(self.model_cfg.get("num_ddim_steps", 10))
        if not self.use_ddim or self.num_ddim_steps != 10:
            raise ValueError(
                "RoboDojo causal-MoT evaluation requires exactly 10 flow "
                f"steps, got use_ddim={self.use_ddim}, "
                f"num_ddim_steps={self.num_ddim_steps}."
            )
        self.text_planning_enabled = _as_bool(
            metadata.get("text_planning_enabled", False)
        )
        self.event_memory_enabled = _as_bool(
            metadata.get("event_memory_enabled", False)
        )
        if self.event_memory_enabled and not self.text_planning_enabled:
            raise RuntimeError(
                "Event-memory checkpoint must also declare text_planning_enabled"
            )
        self.event_semantic_fields = dict(
            metadata.get("event_semantic_fields") or {}
        )
        required_event_fields = {
            "memory",
            "cached_subtask",
            "decision",
            "memory_add",
            "cache_valid",
        }
        if self.event_memory_enabled and set(self.event_semantic_fields) != required_event_fields:
            raise RuntimeError(
                "Event-memory server published an invalid semantic-field ABI: "
                f"{self.event_semantic_fields!r}"
            )
        self.event_empty_memory = str(
            metadata.get("event_empty_memory", "None.")
        ).strip()
        self.event_empty_cached_subtask = str(
            metadata.get("event_empty_cached_subtask", "None.")
        ).strip()
        self.event_semantic_offset = int(
            metadata.get("event_semantic_offset", 0)
        )
        self.event_replan_interval = int(
            metadata.get("event_replan_interval", 0)
        )
        if self.event_memory_enabled and (
            not self.event_empty_memory or not self.event_empty_cached_subtask
        ):
            raise RuntimeError("Event-memory empty sentinels must be non-empty")
        if self.event_memory_enabled and (
            self.event_replan_interval != self.replan_interval
            or self.event_semantic_offset != -self.replan_interval
        ):
            raise RuntimeError(
                "Event-memory train/eval cadence mismatch: checkpoint uses "
                f"semantic_offset={self.event_semantic_offset}, "
                f"replan_interval={self.event_replan_interval}, while the "
                f"client uses replan_interval={self.replan_interval}."
            )
        self.text_replan_chunks = int(
            self.model_cfg.get("text_replan_chunks", 4)
        )
        self.log_planner_text = _as_bool(
            self.model_cfg.get("log_planner_text", False)
        )
        if not self.event_memory_enabled and self.text_replan_chunks <= 0:
            raise ValueError(
                "RoboDojo text_replan_chunks must be positive, got "
                f"{self.text_replan_chunks}."
            )
        if (
            self.text_planning_enabled
            and not self.event_memory_enabled
            and not _as_bool(
            metadata.get("planner_text_cache_supported", False)
            )
        ):
            raise RuntimeError(
                "Selected text-planning checkpoint/server does not support "
                "low-frequency cached planner text."
            )
        self.text_history_enabled = _as_bool(
            metadata.get("text_history_enabled", False)
        )
        self.text_history_frame_offsets = tuple(
            int(offset)
            for offset in metadata.get("text_history_frame_offsets", [])
        )
        if self.text_history_enabled:
            if not self.text_history_frame_offsets:
                raise RuntimeError(
                    "History-aware checkpoint omitted text_history_frame_offsets"
                )
            if (
                any(offset >= 0 for offset in self.text_history_frame_offsets)
                or tuple(sorted(self.text_history_frame_offsets))
                != self.text_history_frame_offsets
                or len(set(self.text_history_frame_offsets))
                != len(self.text_history_frame_offsets)
            ):
                raise RuntimeError(
                    "History-aware checkpoint published invalid past-frame offsets: "
                    f"{self.text_history_frame_offsets}"
                )
        self.text_history_memory_offset = int(
            metadata.get("text_history_memory_offset", 0)
        )
        if self.text_history_enabled and self.text_planning_enabled:
            expected_memory_offset = -(
                self.replan_interval * self.text_replan_chunks
            )
            if self.text_history_memory_offset != expected_memory_offset:
                raise RuntimeError(
                    "RoboDojo text refresh schedule does not match training "
                    "Finished Task List memory: "
                    f"replan={self.replan_interval} * "
                    f"text_replan_chunks={self.text_replan_chunks} expects "
                    f"memory_offset={expected_memory_offset}, checkpoint has "
                    f"{self.text_history_memory_offset}"
                )
        self.text_history_image_field = str(
            metadata.get(
                "text_history_image_field",
                "planner_history_images",
            )
        )
        self.text_history_image_size = tuple(
            int(value)
            for value in metadata.get("text_history_image_size", [])
        )
        if self.text_history_enabled and (
            len(self.text_history_image_size) != 2
            or min(self.text_history_image_size) <= 0
        ):
            raise RuntimeError(
                "History-aware checkpoint published invalid [width,height] "
                f"text_history_image_size={self.text_history_image_size}"
            )
        self.finished_task_list_field = str(
            metadata.get("finished_task_list_field", "finished_task_list")
        )
        self.empty_finished_task_list = str(
            metadata.get("empty_finished_task_list", "None")
        ).strip()
        if (
            self.text_history_enabled
            and self.text_planning_enabled
            and not self.empty_finished_task_list
        ):
            raise RuntimeError(
                "History-aware checkpoint requires an empty Finished Task List sentinel"
            )
        self.default_instruction = str(self.model_cfg.get("task_name") or "follow the instruction")
        self.obs_by_env: dict[int, dict[str, Any]] = {}
        self.action_chunks_by_env: dict[int, np.ndarray] = {}
        self.normalized_action_chunks_by_env: dict[int, np.ndarray] = {}
        self.rtc_debug_replans_by_env: dict[int, int] = {}
        self.planner_text_by_env: dict[int, str] = {}
        self.semantic_memory_by_env: dict[int, str] = {}
        self.cached_current_subtask_by_env: dict[int, str] = {}
        self.planner_observation_history_by_env: dict[
            int,
            dict[int, np.ndarray],
        ] = {}
        self.finished_task_list_by_env: dict[int, str] = {}
        self.planner_input_finished_task_list_by_env: dict[int, str] = {}
        self.step_by_env: dict[int, int] = {}
        self._latest_env_idx_list = [0]
        checkpoint_path = Path(str(metadata.get("ckpt_path", "")))
        checkpoint_run = (
            checkpoint_path.parents[1].name
            if len(checkpoint_path.parents) >= 2
            else checkpoint_path.name
        )
        rtc_summary = "disabled"
        if self.rtc_enabled:
            rtc_summary = (
                f"overlap{self.rtc_overlap}/h{self.rtc_execution_horizon}"
                f"/delay{self.rtc_inference_delay}/"
                f"{self.rtc_prefix_attention_schedule}"
                f"/w{self.rtc_max_guidance_weight:g}"
                f"/debug{self.rtc_debug_max_replans}"
            )
        print(
            "[starVLA][RoboDojo] contract OK: "
            f"chunk={self.action_chunk_size}, replan={self.replan_interval}, "
            f"rtc={rtc_summary}, "
            f"text_planning={self.text_planning_enabled}, "
            f"event_memory={self.event_memory_enabled}, "
            f"text_history={self.text_history_enabled}, "
            f"text_replan_chunks={self.text_replan_chunks}, "
            f"eval_flow_steps={self.num_ddim_steps}, "
            f"saved_flow_steps={metadata.get('mot_num_inference_timesteps')}, "
            f"mot_contract={metadata.get('mot_contract_version')}, "
            f"learned_query_mask={metadata.get('planner_query_mask_contract')}, "
            f"qwen35_conv={metadata.get('qwen35_causal_conv1d_backend')}, "
            f"qwen35_fla={metadata.get('qwen35_fla_backend')}, "
            "qwen35_attn="
            f"{metadata.get('qwen_attn_implementation')}/"
            f"{metadata.get('qwen35_attn_implementation_source')}, "
            "dino_tokens="
            f"current{metadata.get('mot_current_dino_tokens')}/"
            f"future{metadata.get('mot_future_dino_tokens')}/"
            f"action{self.action_chunk_size}, "
            f"checkpoint={checkpoint_run}, "
            f"state={'14D/zscore' if self.expects_state else 'disabled'}, "
            "image=FastWAM-320x384"
        )

    def _convert_obs(self, observation: dict[str, Any]) -> dict[str, Any]:
        source_images = [
            _extract_camera(observation, ("cam_head", "cam_high", "head_camera")),
            _extract_camera(observation, ("cam_left_wrist", "left_camera")),
            _extract_camera(observation, ("cam_right_wrist", "right_camera")),
        ]
        composite = self._build_composite(source_images)
        if composite.size != self.image_size:
            raise ValueError(f"Composite size {composite.size} != expected {self.image_size}.")

        converted = {
            "lang": _instruction(observation, self.default_instruction),
            "image": [np.asarray(composite, dtype=np.uint8).copy()],
        }
        if self.expects_state:
            state = pack_robot_state(
                observation,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            ).astype(np.float32)
            if state.ndim == 2 and state.shape[0] == 1:
                state = state[0]
            if state.ndim != 1 or state.shape[0] != self.state_dim:
                raise ValueError(
                    f"Expected {self.state_dim}-D "
                    "[left_arm,left_gripper,right_arm,right_gripper] state, "
                    f"got {state.shape}."
                )
            # Raw state is intentional: PolicyServerWrapper applies the exact
            # training-time z-score transform exactly once.
            converted["state"] = state
        return converted

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError("update_obs_batch received an empty observation list.")
        self._latest_env_idx_list = []
        for obs in obs_list:
            env_idx = int(obs.get("env_idx", 0))
            self._latest_env_idx_list.append(env_idx)
            converted = self._convert_obs(obs)
            self.obs_by_env[env_idx] = converted
            if getattr(self, "text_history_enabled", False):
                self._record_planner_observation(
                    env_idx,
                    self.step_by_env.get(env_idx, 0),
                    converted["image"][0],
                )

    def _record_planner_observation(
        self,
        env_idx: int,
        step: int,
        image: Any,
    ) -> None:
        """Keep only the image window required by the loaded checkpoint."""

        histories = getattr(
            self,
            "planner_observation_history_by_env",
            None,
        )
        if histories is None:
            histories = {}
            self.planner_observation_history_by_env = histories
        history = histories.setdefault(int(env_idx), {})
        history[int(step)] = np.asarray(image, dtype=np.uint8).copy()
        offsets = tuple(getattr(self, "text_history_frame_offsets", ()) or ())
        if offsets:
            oldest_needed = int(step) + min(offsets)
            for old_step in list(history):
                if old_step < oldest_needed:
                    del history[old_step]

    def _planner_history_images(self, env_idx: int, step: int) -> np.ndarray:
        histories = getattr(
            self,
            "planner_observation_history_by_env",
            {},
        )
        history = histories.get(int(env_idx), {})
        if not history:
            raise RuntimeError(
                f"No planner observation history recorded for env {env_idx}"
            )
        available = sorted(history)
        selected = []
        for offset in self.text_history_frame_offsets:
            target = max(int(step) + int(offset), 0)
            if target in history:
                selected_step = target
            else:
                earlier = [candidate for candidate in available if candidate <= target]
                selected_step = earlier[-1] if earlier else available[0]
            image = history[selected_step]
            target_size = tuple(
                getattr(self, "text_history_image_size", ()) or ()
            )
            if target_size and (
                image.shape[1] != target_size[0]
                or image.shape[0] != target_size[1]
            ):
                image = cv2.resize(
                    image,
                    target_size,
                    interpolation=cv2.INTER_AREA,
                )
            selected.append(image)
        return np.stack(selected, axis=0)

    def _should_refresh_planner_text(self, env_idx: int, step: int) -> bool:
        if not getattr(self, "text_planning_enabled", False):
            return False
        if getattr(self, "event_memory_enabled", False):
            # Event checkpoints make a fresh KEEP/UPDATE decision at every
            # action replan; they have no fixed low-frequency planner refresh.
            return False
        if env_idx not in self.planner_text_by_env:
            return True
        action_replan_index = step // self.replan_interval
        return action_replan_index % self.text_replan_chunks == 0

    def _infer_chunks(
        self,
        env_idx_list: list[int],
        *,
        refresh_text_envs: list[int] | set[int] | tuple[int, ...] | None = None,
    ) -> dict[int, np.ndarray]:
        if not env_idx_list:
            return {}
        missing = [env_idx for env_idx in env_idx_list if env_idx not in self.obs_by_env]
        if missing:
            raise RuntimeError(
                "update_obs must be called before get_action; "
                f"missing envs={missing}."
            )
        refresh_set = set(refresh_text_envs or ())
        prompt_memory_by_env: dict[int, str] = {}
        examples = []
        for env_idx in env_idx_list:
            example = dict(self.obs_by_env[env_idx])
            if getattr(self, "event_memory_enabled", False):
                fields = self.event_semantic_fields
                cache_valid = env_idx in self.cached_current_subtask_by_env
                example[fields["memory"]] = self.semantic_memory_by_env.get(
                    env_idx, self.event_empty_memory
                )
                example[fields["cached_subtask"]] = (
                    self.cached_current_subtask_by_env.get(
                        env_idx, self.event_empty_cached_subtask
                    )
                )
                example[fields["cache_valid"]] = bool(cache_valid)
            if getattr(self, "text_history_enabled", False):
                step = self.step_by_env.get(env_idx, 0)
                example[self.text_history_image_field] = (
                    self._planner_history_images(env_idx, step)
                )
                if getattr(self, "text_planning_enabled", False):
                    is_refresh = (
                        env_idx in refresh_set
                        or env_idx not in self.planner_text_by_env
                    )
                    if is_refresh:
                        prompt_memory = self.finished_task_list_by_env.get(
                            env_idx,
                            self.empty_finished_task_list,
                        )
                    else:
                        prompt_memory = (
                            self.planner_input_finished_task_list_by_env.get(
                                env_idx,
                                self.finished_task_list_by_env.get(
                                    env_idx,
                                    self.empty_finished_task_list,
                                ),
                            )
                        )
                    prompt_memory_by_env[env_idx] = str(prompt_memory)
                    example[self.finished_task_list_field] = str(prompt_memory)
            examples.append(example)
        payload = {
            "examples": examples,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
            "unnorm_key": self.unnorm_key,
        }
        if getattr(self, "rtc_enabled", False):
            previous = np.zeros(
                (
                    len(env_idx_list),
                    self.action_chunk_size,
                    self.action_dim,
                ),
                dtype=np.float32,
            )
            prefix_lengths = np.zeros(len(env_idx_list), dtype=np.int64)
            normalized_cache = getattr(
                self,
                "normalized_action_chunks_by_env",
                {},
            )
            for position, env_idx in enumerate(env_idx_list):
                old_chunk = normalized_cache.get(env_idx)
                if old_chunk is None:
                    continue
                old_chunk = np.asarray(old_chunk, dtype=np.float32)
                expected_old = (self.action_chunk_size, self.action_dim)
                if old_chunk.shape != expected_old:
                    raise ValueError(
                        f"Cached normalized RTC chunk for env {env_idx} has "
                        f"shape {old_chunk.shape}, expected {expected_old}"
                    )
                tail = old_chunk[self.replan_interval :]
                length = min(len(tail), self.rtc_execution_horizon)
                previous[position, :length] = tail[:length]
                prefix_lengths[position] = length
            payload.update(
                prev_action_chunk_normalized=previous,
                rtc_prefix_lengths=prefix_lengths,
                inference_delay=self.rtc_inference_delay,
                execution_horizon=self.rtc_execution_horizon,
                prefix_attention_schedule=self.rtc_prefix_attention_schedule,
                max_guidance_weight=self.rtc_max_guidance_weight,
            )
        if (
            getattr(self, "text_planning_enabled", False)
            and not getattr(self, "event_memory_enabled", False)
        ):
            payload["cached_planner_texts"] = [
                None
                if env_idx in refresh_set
                else self.planner_text_by_env.get(env_idx)
                for env_idx in env_idx_list
            ]
        response = self.client.predict_action(payload)
        if not response.get("ok", False):
            raise RuntimeError(f"StarVLA inference failed: {response.get('error', response)}")
        response_data = response["data"]
        chunks = np.asarray(response_data["actions"], dtype=np.float32)
        expected = (len(env_idx_list), self.action_chunk_size, self.action_dim)
        if chunks.shape != expected:
            raise ValueError(
                f"Expected unnormalized action chunks {expected}, got {chunks.shape}."
            )
        if getattr(self, "rtc_enabled", False):
            raw_normalized = response_data.get("normalized_actions")
            if raw_normalized is None:
                raise ValueError(
                    "RTC requires normalized_actions in the policy-server response"
                )
            normalized_chunks = np.asarray(raw_normalized, dtype=np.float32)
            if normalized_chunks.shape != expected:
                raise ValueError(
                    "RTC requires normalized_actions aligned with actions; "
                    f"expected {expected}, got {normalized_chunks.shape}"
                )
            normalized_cache = getattr(
                self,
                "normalized_action_chunks_by_env",
                None,
            )
            if normalized_cache is None:
                normalized_cache = {}
                self.normalized_action_chunks_by_env = normalized_cache
            debug_counts = getattr(self, "rtc_debug_replans_by_env", None)
            if debug_counts is None:
                debug_counts = {}
                self.rtc_debug_replans_by_env = debug_counts
            debug_limit = int(getattr(self, "rtc_debug_max_replans", 0))
            for position, env_idx in enumerate(env_idx_list):
                old_chunk = normalized_cache.get(env_idx)
                prefix_length = int(prefix_lengths[position])
                debug_count = debug_counts.get(env_idx, 0)
                if (
                    old_chunk is not None
                    and prefix_length > 0
                    and debug_count < debug_limit
                ):
                    old_chunk = np.asarray(old_chunk, dtype=np.float32)
                    new_chunk = normalized_chunks[position]
                    old_tail = old_chunk[
                        self.replan_interval : self.replan_interval + prefix_length
                    ]
                    previous_action = old_chunk[self.replan_interval - 1]

                    def _rmse(delta: np.ndarray) -> float:
                        return float(np.sqrt(np.mean(np.square(delta))))

                    planned_step_rmse = _rmse(old_tail[0] - previous_action)
                    new_switch_rmse = _rmse(new_chunk[0] - previous_action)
                    rtc_target_rmse = _rmse(new_chunk[0] - old_tail[0])
                    prefix_rmse = _rmse(new_chunk[:prefix_length] - old_tail)
                    print(
                        "[starVLA][RoboDojo][RTC] "
                        f"env={env_idx} replan={debug_count + 1} "
                        f"planned_step_rmse={planned_step_rmse:.6f} "
                        f"new_switch_rmse={new_switch_rmse:.6f} "
                        f"rtc_target_rmse={rtc_target_rmse:.6f} "
                        f"prefix_rmse={prefix_rmse:.6f}",
                        flush=True,
                    )
                    debug_counts[env_idx] = debug_count + 1
                normalized_cache[env_idx] = normalized_chunks[position].copy()

        planner_texts = response_data.get("planner_text")
        if getattr(self, "event_memory_enabled", False):
            decisions = response_data.get("semantic_decision")
            memory_adds = response_data.get("semantic_memory_add")
            current_subtasks = response_data.get("semantic_current_subtask")
            if not all(
                isinstance(values, (list, tuple))
                and len(values) == len(env_idx_list)
                for values in (decisions, memory_adds, current_subtasks)
            ):
                raise RuntimeError(
                    "Event-memory response fields must be aligned lists: "
                    f"decisions={decisions!r}, memory_adds={memory_adds!r}, "
                    f"current_subtasks={current_subtasks!r}"
                )
            if planner_texts is not None and isinstance(planner_texts, str):
                planner_texts = [planner_texts]
            for position, env_idx in enumerate(env_idx_list):
                decision = str(decisions[position]).strip().upper()
                if decision == "KEEP":
                    if (
                        memory_adds[position] is not None
                        or current_subtasks[position] is not None
                    ):
                        raise RuntimeError(
                            "KEEP must not carry a memory delta or replacement subtask"
                        )
                elif decision == "UPDATE":
                    memory_add = str(memory_adds[position] or "").strip()
                    current_subtask = str(
                        current_subtasks[position] or ""
                    ).strip()
                    if not memory_add or not current_subtask:
                        raise RuntimeError(
                            "UPDATE must carry non-empty Memory Add and Current Subtask"
                        )
                    previous = self.semantic_memory_by_env.get(
                        env_idx, self.event_empty_memory
                    )
                    self.semantic_memory_by_env[env_idx] = (
                        _append_semantic_memory(
                            previous, memory_add, self.event_empty_memory
                        )
                    )
                    self.cached_current_subtask_by_env[env_idx] = current_subtask
                else:
                    raise RuntimeError(
                        f"Unknown semantic decision for env {env_idx}: {decision!r}"
                    )
                if planner_texts is not None:
                    if len(planner_texts) != len(env_idx_list):
                        raise RuntimeError(
                            "Event planner_text must align with the request batch"
                        )
                    self.planner_text_by_env[env_idx] = str(
                        planner_texts[position]
                    ).strip()
                if getattr(self, "log_planner_text", False):
                    print(
                        "[starVLA][RoboDojo] "
                        f"env={env_idx} semantic_decision={decision} "
                        f"memory={self.semantic_memory_by_env.get(env_idx, self.event_empty_memory)!r} "
                        f"subtask={self.cached_current_subtask_by_env.get(env_idx, self.event_empty_cached_subtask)!r}",
                        flush=True,
                    )
        elif getattr(self, "text_planning_enabled", False):
            if planner_texts is None:
                raise RuntimeError(
                    "Text-planning checkpoint omitted planner_text from its response"
                )
            if isinstance(planner_texts, str):
                planner_texts = [planner_texts]
            if len(planner_texts) != len(env_idx_list):
                raise RuntimeError(
                    "planner_text response must align with the request batch; "
                    f"got {len(planner_texts)} for {len(env_idx_list)} environments"
                )
            refreshed = response_data.get("planner_text_refreshed")
            if refreshed is None:
                refreshed = [
                    env_idx not in self.planner_text_by_env
                    for env_idx in env_idx_list
                ]
            if len(refreshed) != len(env_idx_list):
                raise RuntimeError(
                    "planner_text_refreshed response must align with request batch"
                )
            planner_finished = response_data.get(
                "planner_finished_task_list"
            )
            if getattr(self, "text_history_enabled", False):
                if planner_finished is None:
                    raise RuntimeError(
                        "History-aware checkpoint omitted planner_finished_task_list"
                    )
                if isinstance(planner_finished, str):
                    planner_finished = [planner_finished]
                if len(planner_finished) != len(env_idx_list):
                    raise RuntimeError(
                        "planner_finished_task_list must align with the request batch"
                    )
            for position, (
                env_idx,
                planner_text,
                was_refreshed,
            ) in enumerate(
                zip(env_idx_list, planner_texts, refreshed)
            ):
                text = str(planner_text).strip()
                if not text:
                    raise RuntimeError(
                        f"Text planner returned an empty response for env {env_idx}"
                    )
                self.planner_text_by_env[env_idx] = text
                if (
                    getattr(self, "text_history_enabled", False)
                    and _as_bool(was_refreshed)
                ):
                    finished = str(planner_finished[position]).strip()
                    if not finished:
                        raise RuntimeError(
                            "Text planner returned an empty Finished Task List "
                            f"for env {env_idx}"
                        )
                    self.planner_input_finished_task_list_by_env[env_idx] = (
                        prompt_memory_by_env[env_idx]
                    )
                    self.finished_task_list_by_env[env_idx] = finished
                if getattr(self, "log_planner_text", False) and _as_bool(
                    was_refreshed
                ):
                    print(f"[starVLA][RoboDojo][env={env_idx}] planner_text={text!r}")
        return {
            env_idx: np.asarray(chunks[position], dtype=np.float32)
            for position, env_idx in enumerate(env_idx_list)
        }

    def _infer_chunk(self, env_idx: int) -> np.ndarray:
        step = self.step_by_env.get(env_idx, 0)
        refresh = (
            [env_idx]
            if self._should_refresh_planner_text(env_idx, step)
            else []
        )
        return self._infer_chunks(
            [env_idx],
            refresh_text_envs=refresh,
        )[env_idx]

    def _next_action(self, env_idx: int) -> np.ndarray:
        step = self.step_by_env.get(env_idx, 0)
        chunk_offset = step % self.replan_interval
        chunk = self.action_chunks_by_env.get(env_idx)
        if chunk is None or chunk_offset == 0:
            chunk = self._infer_chunk(env_idx)
            self.action_chunks_by_env[env_idx] = chunk
        action = np.asarray(chunk[chunk_offset], dtype=np.float32)
        self.step_by_env[env_idx] = step + 1
        return action

    def get_action(self):
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        env_idx_list = [
            int(env_idx)
            for env_idx in (env_idx_list or self._latest_env_idx_list)
        ]
        replan_envs = []
        refresh_text_envs = []
        for env_idx in env_idx_list:
            step = self.step_by_env.get(env_idx, 0)
            if (
                env_idx not in self.action_chunks_by_env
                or step % self.replan_interval == 0
            ):
                replan_envs.append(env_idx)
                if self._should_refresh_planner_text(env_idx, step):
                    refresh_text_envs.append(env_idx)
        if replan_envs:
            self.action_chunks_by_env.update(
                self._infer_chunks(
                    replan_envs,
                    refresh_text_envs=refresh_text_envs,
                )
            )

        actions = []
        for env_idx in env_idx_list:
            step = self.step_by_env.get(env_idx, 0)
            chunk_offset = step % self.replan_interval
            action = np.asarray(
                self.action_chunks_by_env[env_idx][chunk_offset],
                dtype=np.float32,
            )
            self.step_by_env[env_idx] = step + 1
            actions.append(
                [
                    unpack_robot_state(
                        action,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                ]
            )
        return actions

    def reset(self):
        self.obs_by_env.clear()
        self.action_chunks_by_env.clear()
        getattr(self, "normalized_action_chunks_by_env", {}).clear()
        getattr(self, "rtc_debug_replans_by_env", {}).clear()
        self.planner_text_by_env.clear()
        getattr(self, "semantic_memory_by_env", {}).clear()
        getattr(self, "cached_current_subtask_by_env", {}).clear()
        self.planner_observation_history_by_env.clear()
        self.finished_task_list_by_env.clear()
        self.planner_input_finished_task_list_by_env.clear()
        self.step_by_env.clear()
        self._latest_env_idx_list = [0]


__all__ = ["Model"]
