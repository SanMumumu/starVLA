"""State-aware FastWAM-composite adapter between RoboDojo and StarVLA.

This module deliberately lives in the uploaded StarVLA repository.  The
RoboDojo checkout supplies the simulator and XPolicyLab transport only; its
possibly stale ``policy/starVLA`` directory is never imported.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
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
        self.state_dim = int(self.model_cfg.get("expected_state_dim", 14))
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

        expected_layout = str(self.model_cfg.get("expected_image_layout", FASTWAM_COMPOSITE_LAYOUT))
        if metadata.get("image_layout") != expected_layout:
            raise RuntimeError(
                f"Checkpoint image_layout={metadata.get('image_layout')!r}; expected {expected_layout!r}."
            )
        expected_key = str(self.model_cfg.get("expected_composite_view_key", FASTWAM_COMPOSITE_VIEW_KEY))
        if metadata.get("composite_view_key") != expected_key:
            raise RuntimeError(
                f"Checkpoint composite_view_key={metadata.get('composite_view_key')!r}; expected {expected_key!r}."
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
        if not _as_bool(self.model_cfg.get("include_state", True)):
            raise RuntimeError("RoboDojo was trained with state; deploy include_state must be true.")
        if "expects_state" not in metadata or not _as_bool(metadata["expects_state"]):
            raise RuntimeError(
                "Selected checkpoint does not declare include_state=true through config.full.yaml."
            )
        for key, expected in (
            ("state_keys", _EXPECTED_STATE_KEYS),
            ("action_keys", _EXPECTED_ACTION_KEYS),
        ):
            actual = metadata.get(key)
            if actual is None or list(actual) != expected:
                raise RuntimeError(f"Checkpoint {key}={actual!r}; expected {expected!r}.")

        self.unnorm_key = str(self.model_cfg.get("unnorm_key", "new_embodiment"))
        available = list(metadata.get("available_unnorm_keys") or [])
        if self.unnorm_key not in available:
            raise RuntimeError(f"unnorm_key={self.unnorm_key!r} not available; checkpoint has {available}.")
        self.use_ddim = _as_bool(self.model_cfg.get("use_ddim", True))
        self.num_ddim_steps = int(self.model_cfg.get("num_ddim_steps", 10))
        self.default_instruction = str(self.model_cfg.get("task_name") or "follow the instruction")
        self.obs_by_env: dict[int, dict[str, Any]] = {}
        self.action_chunks_by_env: dict[int, np.ndarray] = {}
        self.step_by_env: dict[int, int] = {}
        self._latest_env_idx_list = [0]
        print(
            "[starVLA][RoboDojo] contract OK: "
            f"chunk={self.action_chunk_size}, state=14D/zscore, image=FastWAM-320x384, metadata={metadata}"
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
                f"Expected {self.state_dim}-D [left_arm,left_gripper,right_arm,right_gripper] state, "
                f"got {state.shape}."
            )
        return {
            "lang": _instruction(observation, self.default_instruction),
            "image": [np.asarray(composite, dtype=np.uint8).copy()],
            # Raw state is intentional: PolicyServerWrapper applies the exact
            # training-time FastWAM z-score transform exactly once.
            "state": state,
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError("update_obs_batch received an empty observation list.")
        self._latest_env_idx_list = []
        for obs in obs_list:
            env_idx = int(obs.get("env_idx", 0))
            self._latest_env_idx_list.append(env_idx)
            self.obs_by_env[env_idx] = self._convert_obs(obs)

    def _infer_chunk(self, env_idx: int) -> np.ndarray:
        if env_idx not in self.obs_by_env:
            raise RuntimeError("update_obs must be called before get_action.")
        response = self.client.predict_action(
            {
                "examples": [self.obs_by_env[env_idx]],
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
                "unnorm_key": self.unnorm_key,
            }
        )
        if not response.get("ok", False):
            raise RuntimeError(f"StarVLA inference failed: {response.get('error', response)}")
        chunk = np.asarray(response["data"]["actions"][0], dtype=np.float32)
        expected = (self.action_chunk_size, self.action_dim)
        if chunk.shape != expected:
            raise ValueError(f"Expected unnormalized action chunk {expected}, got {chunk.shape}.")
        return chunk

    def _next_action(self, env_idx: int) -> np.ndarray:
        step = self.step_by_env.get(env_idx, 0)
        chunk = self.action_chunks_by_env.get(env_idx)
        if chunk is None or step % self.action_chunk_size == 0:
            chunk = self._infer_chunk(env_idx)
            self.action_chunks_by_env[env_idx] = chunk
        action = np.asarray(chunk[step % self.action_chunk_size], dtype=np.float32)
        self.step_by_env[env_idx] = step + 1
        return action

    def get_action(self):
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        env_idx_list = env_idx_list or self._latest_env_idx_list
        return [
            [
                unpack_robot_state(
                    self._next_action(int(env_idx)),
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
            ]
            for env_idx in env_idx_list
        ]

    def reset(self):
        self.obs_by_env.clear()
        self.action_chunks_by_env.clear()
        self.step_by_env.clear()
        self._latest_env_idx_list = [0]


__all__ = ["Model"]
