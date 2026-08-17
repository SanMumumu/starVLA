"""Dual-query LeRobot dataset wrappers.

The implementation composes the standard single/mixture datasets, registry,
and collator without modifying the upstream LeRobot data path. It supports
both precomputed DINO targets and the online feature path.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from starVLA.dataloader.action_correlation import compute_action_noise_matrix
from starVLA.dataloader.fastwam_image import (
    FASTWAM_COMPOSITE_LAYOUT,
    FASTWAM_COMPOSITE_SIZE,
    FASTWAM_COMPOSITE_VIEW_KEY,
    TRI_VIEW_COMPOSITE_LAYOUT,
    TRI_VIEW_COMPOSITE_VIEW_KEY,
    build_robotwin_composite,
)
from starVLA.dataloader.libero_image import (
    LIBERO_COMPOSITE_LAYOUT,
    LIBERO_COMPOSITE_SIZE,
    LIBERO_COMPOSITE_VIEW_KEY,
    build_libero_composite,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.registry import EmbodimentTag, ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.jointflow.mix_registry import resolve_data_mix
from starVLA.dataloader.jointflow.event_memory import (
    SemanticBoundaryBatchSampler,
    build_or_load_semantic_index,
    event_memory_from_trajectory,
)
from starVLA.dataloader.jointflow.text_history import (
    text_history_frame_offsets,
    text_history_memory_from_trajectory,
)
from starVLA.dataloader.lerobot_datasets import collate_fn


######### // code // ##########
def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_COMPOSITE_LAYOUTS = {
    FASTWAM_COMPOSITE_LAYOUT,
    TRI_VIEW_COMPOSITE_LAYOUT,
    LIBERO_COMPOSITE_LAYOUT,
}


def _composite_contract(image_layout: str):
    """Return ``(size, view_key, builder)`` for a configured layout."""

    if image_layout == LIBERO_COMPOSITE_LAYOUT:
        return (
            LIBERO_COMPOSITE_SIZE,
            LIBERO_COMPOSITE_VIEW_KEY,
            build_libero_composite,
        )
    if image_layout == TRI_VIEW_COMPOSITE_LAYOUT:
        return (
            FASTWAM_COMPOSITE_SIZE,
            TRI_VIEW_COMPOSITE_VIEW_KEY,
            build_robotwin_composite,
        )
    if image_layout == FASTWAM_COMPOSITE_LAYOUT:
        return (
            FASTWAM_COMPOSITE_SIZE,
            FASTWAM_COMPOSITE_VIEW_KEY,
            build_robotwin_composite,
        )
    raise ValueError(f"Unknown composite image layout: {image_layout!r}")


def _text_value(value: Any) -> str:
    """Normalize one parquet text cell without inventing a label."""

    if value is None:
        return ""
    try:
        if bool(np.asarray(value).ndim == 0 and np.asarray(value).dtype.kind == "f" and np.isnan(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def text_annotation_from_row(row, config) -> dict[str, str]:
    """Read required current/completed-subtask labels from one parquet row."""

    fields = _cfg_get(
        config,
        "fields",
        {"subtask_text": "subtask_text", "completed_subtask_text": "complete_text"},
    )
    fields = dict(fields) if hasattr(fields, "items") else {}
    if not fields:
        raise ValueError("text_annotations.fields must contain at least one mapping")
    result = {
        output_name: _text_value(row.get(str(column_name), ""))
        for output_name, column_name in fields.items()
    }
    missing = sorted(name for name, value in result.items() if not value)
    if missing:
        raise ValueError(
            "Text-supervised RoboDojo data requires every configured annotation "
            f"to be non-empty; missing={missing}"
        )
    return result


def _safe_view_name(key: str) -> str:
    return str(key).replace("/", "__").replace(".", "_")


def _configured_dino_target_view_keys(data_cfg) -> list[str] | None:
    """Return the explicitly configured latent-target views, preserving order."""
    raw = _cfg_get(data_cfg, "dino_target_view_keys", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    keys = [str(key) for key in raw]
    if not keys:
        raise ValueError("dino_target_view_keys must contain at least one view when configured.")
    if len(set(keys)) != len(keys):
        raise ValueError(f"dino_target_view_keys contains duplicate views: {keys}")
    return keys


def _view_original_key(dataset: LeRobotSingleDataset, video_key: str) -> str:
    subkey = video_key.replace("video.", "", 1)
    meta = dataset.lerobot_modality_meta.video[subkey]
    return meta.original_key or subkey


def _append_state_norm_if_needed(base_transforms, state_keys: list[str], state_norm_modes: dict | None):
    if not state_keys:
        return base_transforms
    if not isinstance(base_transforms, ComposedModalityTransform):
        base_transforms = ComposedModalityTransform(transforms=[base_transforms])

    existing = []
    for transform in base_transforms.transforms:
        existing.extend(list(getattr(transform, "apply_to", []) or []))
    if any(key in existing for key in state_keys):
        return base_transforms

    modes = state_norm_modes or {}
    if not modes:
        modes = {key: ("binary" if "gripper" in key else "q99") for key in state_keys}
    base_transforms.transforms.append(StateActionToTensor(apply_to=state_keys))
    base_transforms.transforms.append(StateActionTransform(apply_to=state_keys, normalization_modes=modes))
    return base_transforms


def _drop_video_transforms(base_transforms):
    if not isinstance(base_transforms, ComposedModalityTransform):
        return base_transforms

    kept = []
    for transform in base_transforms.transforms:
        apply_to = list(getattr(transform, "apply_to", []) or [])
        if not apply_to:
            kept.append(transform)
            continue

        non_video_keys = [key for key in apply_to if not str(key).startswith("video.")]
        if not non_video_keys:
            continue

        if len(non_video_keys) != len(apply_to):
            transform = copy.deepcopy(transform)
            transform.apply_to = non_video_keys
        kept.append(transform)

    base_transforms.transforms = kept
    return base_transforms


######### // code // ##########


######### // code // ##########
class JointLiberoDataset(LeRobotSingleDataset):
    def __init__(
        self,
        *args,
        online_dino: bool = True,
        dino_feature_dir: str = "latents",
        dino_episode_cache_size: int = 4,
        dino_target_latents: bool = False,
        **kwargs,
    ):
        self.online_dino = bool(online_dino)
        self.dino_feature_dir_name = dino_feature_dir
        #######
        self._dino_target_latents = bool(dino_target_latents)
        #######
        self._dino_episode_cache_size = max(int(dino_episode_cache_size), 0)
        self._dino_layout = None
        self._dino_memmap = None
        self._dino_index = None
        self._dino_stats = None
        self._episode_latent_cache = {}
        self._last_trajectory_id = None
        self._last_base_index = None
        self._last_video_frames = None
        data_cfg = kwargs.get("data_cfg")
        self._action_pack_dtype = np.dtype(str(_cfg_get(data_cfg, "action_pack_dtype", "float32")))
        self._text_config = _cfg_get(data_cfg, "text_annotations", {})
        self._text_history_offsets = text_history_frame_offsets(self._text_config)
        if self._action_pack_dtype not in {np.dtype("float16"), np.dtype("float32")}:
            raise ValueError(f"action_pack_dtype must be float16 or float32, got {self._action_pack_dtype}")
        super().__init__(*args, **kwargs)

        text_enabled = bool(_cfg_get(self._text_config, "enabled", False))
        if text_enabled:
            fields = dict(_cfg_get(self._text_config, "fields", {}))
            if not fields:
                raise ValueError(
                    "text_annotations.enabled=true requires non-empty fields mapping"
                )
            available = set(self.lerobot_info_meta.get("features", {}))
            missing = sorted(str(column) for column in fields.values() if str(column) not in available)
            if missing:
                raise ValueError(
                    "Text supervision requires parquet columns declared in meta/info.json; "
                    f"missing={missing}, dataset={self.dataset_path}"
                )
            history = _cfg_get(self._text_config, "history", {})
            event_memory = _cfg_get(self._text_config, "event_memory", {})
            if bool(_cfg_get(event_memory, "enabled", False)):
                if bool(_cfg_get(history, "enabled", False)):
                    raise ValueError(
                        "Event-driven semantic memory and planner RGB history are "
                        "mutually exclusive in the no-history recipe"
                    )
                semantic_offset = int(
                    _cfg_get(event_memory, "semantic_offset", -10)
                )
                replan_interval = int(
                    _cfg_get(event_memory, "replan_interval", 10)
                )
                replan_phase = int(_cfg_get(event_memory, "replan_phase", 0))
                if semantic_offset >= 0 or abs(semantic_offset) != replan_interval:
                    raise ValueError(
                        "event_memory requires semantic_offset=-replan_interval, "
                        f"got offset={semantic_offset}, interval={replan_interval}"
                    )
                if not 0 <= replan_phase < replan_interval:
                    raise ValueError(
                        "event_memory.replan_phase must lie in [0, interval), "
                        f"got phase={replan_phase}, interval={replan_interval}"
                    )
            if bool(_cfg_get(history, "enabled", False)):
                fields = dict(_cfg_get(self._text_config, "fields", {}))
                memory_source = str(
                    _cfg_get(
                        history,
                        "finished_task_list_source_field",
                        fields.get("completed_subtask_text", "complete_text"),
                    )
                )
                if memory_source not in available:
                    raise ValueError(
                        "Finished Task List history requires its source parquet "
                        f"column in meta/info.json; missing={memory_source!r}, "
                        f"dataset={self.dataset_path}"
                    )
                memory_offset = int(_cfg_get(history, "memory_offset", 0))
                if memory_offset >= 0:
                    raise ValueError(
                        "text_annotations.history.memory_offset must be negative, "
                        f"got {memory_offset}"
                    )
                image_layout = str(
                    _cfg_get(data_cfg, "image_layout", "separate_views")
                ).lower()
                if image_layout not in _COMPOSITE_LAYOUTS:
                    raise ValueError(
                        "Planner image history currently requires a composite image "
                        f"layout, got {image_layout!r}"
                    )

    @property
    def dino_dir(self) -> Path:
        return self.dataset_path / self.dino_feature_dir_name

    def _load_dino_store(self):
        if self._dino_layout is not None:
            return

        lingbot_index_path = self.dino_dir / "dino_v3_index.json"
        lingbot_stats_path = self.dino_dir / "dino_v3_stats.json"
        if lingbot_index_path.exists() and lingbot_stats_path.exists():
            with open(lingbot_index_path, "r", encoding="utf-8") as f:
                self._dino_index = json.load(f)
            with open(lingbot_stats_path, "r", encoding="utf-8") as f:
                self._dino_stats = json.load(f)
            self._dino_layout = "lingbot_episode"
            return

        index_path = self.dino_dir / "index.json"
        stats_path = self.dino_dir / "dino_v3_stats.json"
        mmap_path = self.dino_dir / "features.float16.mmap"
        if index_path.exists() and stats_path.exists() and mmap_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                self._dino_index = json.load(f)
            with open(stats_path, "r", encoding="utf-8") as f:
                self._dino_stats = json.load(f)
            shape = tuple(int(x) for x in self._dino_index["shape"])
            self._dino_memmap = np.memmap(mmap_path, mode="r", dtype=np.float16, shape=shape)
            self._dino_layout = "memmap"
            return

        if self.dino_feature_dir_name != "latents":
            fallback_root = self.dataset_path / "latents"
            fallback_index = fallback_root / "dino_v3_index.json"
            fallback_stats = fallback_root / "dino_v3_stats.json"
            if fallback_index.exists() and fallback_stats.exists():
                self.dino_feature_dir_name = "latents"
                with open(fallback_index, "r", encoding="utf-8") as f:
                    self._dino_index = json.load(f)
                with open(fallback_stats, "r", encoding="utf-8") as f:
                    self._dino_stats = json.load(f)
                self._dino_layout = "lingbot_episode"
                return

        raise FileNotFoundError(
            "Missing DINOv3 precomputed features. Expected Lingbot-style files under "
            f"{self.dino_dir}/dino_v3_index.json plus latents/chunk-*/<view>/episode_*.pth, "
            "or legacy memmap files index.json, dino_v3_stats.json, features.float16.mmap."
        )

    def _original_view_keys(self) -> list[str]:
        self._load_dino_store()
        if self._dino_index and "original_view_keys" in self._dino_index:
            return [str(k) for k in self._dino_index["original_view_keys"]]
        return [_view_original_key(self, key) for key in self.modality_keys["video"]]

    def _episode_latent_path(self, trajectory_id: int) -> Path:
        traj_pos = int(self.get_trajectory_index(trajectory_id))
        length = int(self.trajectory_lengths[traj_pos])
        chunk_index = self.get_episode_chunk(int(trajectory_id))
        return self.dino_dir / f"chunk-{chunk_index:03d}" / "{view}" / f"episode_{int(trajectory_id):06d}_0_{length}.pth"

    def _load_episode_view_latent(self, trajectory_id: int, view_idx: int) -> torch.Tensor:
        self._load_dino_store()
        cache_key = (int(trajectory_id), int(view_idx))
        if cache_key in self._episode_latent_cache:
            return self._episode_latent_cache[cache_key]

        original_view_key = self._original_view_keys()[view_idx]
        templated = self._episode_latent_path(int(trajectory_id))
        path = Path(str(templated).replace("{view}", original_view_key))
        if not path.exists():
            view_dir = path.parent
            matches = sorted(view_dir.glob(f"episode_{int(trajectory_id):06d}_0_*.pth"))
            if not matches:
                raise FileNotFoundError(f"Missing DINOv3 latent episode file: {path}")
            path = matches[0]

        # Episode files contain hundreds of bf16 frames while each sample consumes one.
        # mmap avoids eagerly reading the whole tensor from shared storage; fall back for
        # older PyTorch/filesystems that do not support mmap or weights_only.
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except (TypeError, RuntimeError, ValueError):
            try:
                obj = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                obj = torch.load(path, map_location="cpu")
        latent = obj["latent"] if isinstance(obj, dict) else obj
        if not torch.is_tensor(latent):
            latent = torch.as_tensor(latent)

        if latent.ndim == 2:
            if isinstance(obj, dict):
                num_frames = int(obj.get("latent_num_frames", obj.get("video_num_frames", 0)))
                num_tokens = int(obj.get("num_tokens", 0))
                if num_tokens <= 0:
                    height = int(obj.get("latent_height", 0))
                    width = int(obj.get("latent_width", 0))
                    num_tokens = height * width if height and width else 0
            else:
                num_frames = int(self.trajectory_lengths[int(self.get_trajectory_index(trajectory_id))])
                num_tokens = int(self._dino_index.get("num_tokens", 0))
            if num_frames <= 0 or num_tokens <= 0:
                raise ValueError(f"Cannot infer latent shape for {path}; got {tuple(latent.shape)}")
            latent = latent.reshape(num_frames, num_tokens, latent.shape[-1])
        elif latent.ndim != 3:
            raise ValueError(f"Expected latent [T,N,D] or flattened [T*N,D] in {path}, got {tuple(latent.shape)}")

        if self._dino_episode_cache_size > 0:
            self._episode_latent_cache[cache_key] = latent
            while len(self._episode_latent_cache) > self._dino_episode_cache_size:
                self._episode_latent_cache.pop(next(iter(self._episode_latent_cache)))
        return latent

    def _absolute_frame_index(self, trajectory_id: int, frame_index: int) -> int:
        self._load_dino_store()
        starts = self._dino_index.get("trajectory_start_indices", None)
        if starts is None:
            starts = (np.cumsum(self.trajectory_lengths) - self.trajectory_lengths).tolist()
        traj_pos = int(self.get_trajectory_index(trajectory_id))
        length = int(self.trajectory_lengths[traj_pos])
        frame_index = int(np.clip(frame_index, 0, length - 1))
        return int(starts[traj_pos]) + frame_index

    def _dino_target_view_indices(self) -> "list[int] | None":
        """Resolve target views against the latent store, preferring stable camera names.

        ``dino_target_view_keys`` is the canonical interface (for example
        ``["video.cam_high"]``). ``dino_target_latent_views`` remains as a legacy
        index-based alias. Reading only selected views avoids loading an entire wrist-camera
        episode latent that the WAM target never consumes.
        """
        canonical_keys = list(self.modality_keys.get("video", []))
        original_keys = self._original_view_keys()
        requested_keys = _configured_dino_target_view_keys(self.data_cfg)
        legacy_views = _cfg_get(self.data_cfg, "dino_target_latent_views", None)

        if requested_keys is not None:
            view_ids = []
            for requested in requested_keys:
                if requested in canonical_keys:
                    canonical_idx = canonical_keys.index(requested)
                    original_key = _view_original_key(self, requested)
                    if original_key in original_keys:
                        view_idx = original_keys.index(original_key)
                    elif len(original_keys) == len(canonical_keys):
                        # Older stores do not always record canonical-to-original names,
                        # but preserve the dataset camera order.
                        view_idx = canonical_idx
                    else:
                        raise ValueError(
                            f"Cannot map DINO target view {requested!r} to latent store views {original_keys}."
                        )
                elif requested in original_keys:
                    view_idx = original_keys.index(requested)
                else:
                    raise ValueError(
                        f"Unknown DINO target view {requested!r}; dataset views={canonical_keys}, "
                        f"latent store views={original_keys}."
                    )
                view_ids.append(int(view_idx))

            if legacy_views is not None:
                legacy_ids = [int(view) for view in legacy_views]
                if legacy_ids != view_ids:
                    raise ValueError(
                        "dino_target_view_keys and legacy dino_target_latent_views disagree: "
                        f"resolved names={view_ids}, indices={legacy_ids}."
                    )
            return view_ids

        if legacy_views is None:
            return None
        view_ids = [int(view) for view in legacy_views]
        invalid = [view for view in view_ids if view < 0 or view >= len(original_keys)]
        if invalid:
            raise ValueError(f"Invalid DINO target view indices {invalid}; latent store has {len(original_keys)} views.")
        return view_ids

    def _dino_target_view_keys(self) -> list[str]:
        requested = _configured_dino_target_view_keys(self.data_cfg)
        if requested is not None:
            return requested
        canonical_keys = list(self.modality_keys.get("video", []))
        view_ids = self._dino_target_view_indices()
        if view_ids is None:
            return canonical_keys
        return [
            canonical_keys[idx] if idx < len(canonical_keys) else self._original_view_keys()[idx] for idx in view_ids
        ]

    def _read_dino(self, trajectory_id: int, frame_index: int) -> np.ndarray:
        self._load_dino_store()
        if self._dino_layout == "lingbot_episode":
            traj_pos = int(self.get_trajectory_index(trajectory_id))
            length = int(self.trajectory_lengths[traj_pos])
            frame_index = int(np.clip(frame_index, 0, length - 1))
            view_ids = self._dino_target_view_indices()
            if view_ids is None:
                view_ids = list(range(len(self._original_view_keys())))
            per_view = []
            for view_idx in view_ids:
                latent = self._load_episode_view_latent(int(trajectory_id), view_idx)
                safe_idx = int(np.clip(frame_index, 0, latent.shape[0] - 1))
                # Convert only the selected frame, not the entire bf16 episode tensor.
                per_view.append(latent[safe_idx].float().numpy())
            z = np.stack(per_view, axis=0).astype(np.float32)
        else:
            abs_idx = self._absolute_frame_index(trajectory_id, frame_index)
            z = np.asarray(self._dino_memmap[abs_idx], dtype=np.float32)
            view_ids = self._dino_target_view_indices()
            if view_ids is not None:
                z = z[view_ids]
        mean = np.asarray(self._dino_stats["mean"], dtype=np.float32)
        std = np.asarray(self._dino_stats["std"], dtype=np.float32)
        std = np.maximum(std, 1e-6)
        return ((z - mean) / std).astype(np.float32)

    def _future_dino_stride(self) -> int:
        action_horizon = int(
            _cfg_get(self.data_cfg, "action_horizon", _cfg_get(self.data_cfg, "future_action_window_size", 8))
        )
        world_model_cfg = _cfg_get(self.data_cfg, "world_model", None)
        stride = _cfg_get(world_model_cfg, "future_stride", None)
        if stride is None:
            stride = _cfg_get(self.data_cfg, "future_dino_stride", action_horizon)
        return max(int(stride), 0)

    def _future_dino_index(self, trajectory_id: int, base_index: int) -> tuple[int, int, int]:
        traj_pos = int(self.get_trajectory_index(trajectory_id))
        length = int(self.trajectory_lengths[traj_pos])
        requested_stride = self._future_dino_stride()
        remaining_steps = max(length - 1 - int(base_index), 0)
        actual_stride = min(requested_stride, remaining_steps)
        return int(base_index) + actual_stride, actual_stride, requested_stride

    def get_step_data(self, trajectory_id: int, base_index: int, decode_video: bool = True) -> dict:
        self._last_trajectory_id = int(trajectory_id)
        self._last_base_index = int(base_index)
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)

        if self.online_dino and decode_video:
            self._last_video_frames = {}
            for key in self.modality_keys.get("video", []):
                self._last_video_frames[key] = self.get_data_by_modality(trajectory_id, "video", key, base_index)

        for modality in ("state", "action", "language"):
            for key in self.modality_keys.get(modality, []):
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return self._apply_action_mode(data)

    def read_action_only(self, trajectory_id: int, base_index: int) -> np.ndarray:
        """Read one normalized action chunk without video or latent decoding.

        Correlated-noise estimation uses the same state/action transforms as
        ``__getitem__`` while avoiding expensive distributed PyAV reads.
        """
        raw = self.get_step_data(trajectory_id, base_index, decode_video=False)
        data = self.transforms(raw)
        action = [_to_numpy(data[k]) for k in self.modality_keys["action"]]
        return np.concatenate(action, axis=1).astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        trajectory_id, base_index = self.all_steps[index]
        raw_data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(raw_data)
        return self._pack_sample(data)

    def _pack_sample(self, data: dict) -> dict:
        if self._last_trajectory_id is None or self._last_base_index is None:
            raise RuntimeError("JointLiberoDataset._pack_sample requires get_step_data to be called first.")
        trajectory_id = int(self._last_trajectory_id)
        base_index = int(self._last_base_index)
        action = []
        for action_key in self.modality_keys["action"]:
            action.append(_to_numpy(data[action_key]))
        # Standard LeRobotSingleDataset packs normalized action labels as
        # float16. WAM ablations can opt into the same label quantization while
        # other JointFlow experiments retain their historical float32 default.
        action = np.concatenate(action, axis=1).astype(self._action_pack_dtype, copy=False)

        future_index, future_steps, future_stride = self._future_dino_index(trajectory_id, base_index)
        require_full_future = bool(_cfg_get(self.data_cfg, "future_valid_requires_full_stride", False))
        future_valid = future_steps == future_stride if require_full_future else future_steps > 0
        sample = {
            "action": action,
            "lang": data[self.modality_keys["language"][0]][0],
            "robot_tag": self.tag,
            "dataset_name": self.dataset_name,
            "trajectory_id": np.int64(trajectory_id),
            "base_index": np.int64(base_index),
            "future_index": np.int64(future_index),
            "future_valid": bool(future_valid),
            "future_valid_steps": np.int64(future_steps),
            "future_stride": np.int64(future_stride),
        }

        if bool(_cfg_get(self._text_config, "enabled", False)):
            row = self.curr_traj_data.iloc[int(self._last_base_index)]
            sample.update(text_annotation_from_row(row, self._text_config))
            sample.update(
                text_history_memory_from_trajectory(
                    self.curr_traj_data,
                    base_index,
                    self._text_config,
                )
            )
            sample.update(
                event_memory_from_trajectory(
                    self.curr_traj_data,
                    base_index,
                    self._text_config,
                )
            )

        if self.online_dino:
            source_view_keys = list(self.modality_keys.get("video", []))
            if self._last_video_frames is None:
                raise RuntimeError("JointLiberoDataset(online) requires get_step_data to decode video first.")
            decode_future = bool(_cfg_get(self.data_cfg, "decode_future_video", True))
            image_layout = str(_cfg_get(self.data_cfg, "image_layout", "separate_views")).lower()
            current_views, future_views = [], []
            video_offsets = [
                int(offset) for offset in self.delta_indices[source_view_keys[0]]
            ]
            for key in source_view_keys[1:]:
                key_offsets = [int(offset) for offset in self.delta_indices[key]]
                if key_offsets != video_offsets:
                    raise ValueError(
                        "All video views must share planner/history delta indices; "
                        f"{source_view_keys[0]}={video_offsets}, {key}={key_offsets}"
                    )
            if 0 not in video_offsets:
                raise ValueError(
                    f"Decoded RGB offsets must contain the current frame, got {video_offsets}"
                )
            current_position = video_offsets.index(0)
            future_position = (
                video_offsets.index(future_stride)
                if decode_future and future_stride in video_offsets
                else None
            )
            for key in source_view_keys:
                frames = np.asarray(self._last_video_frames[key])  # [T,H,W,C]
                current_views.append(frames[current_position])
                if decode_future:
                    if future_position is None:
                        raise ValueError(
                            "Decoded future video offset is missing from modality "
                            f"indices: future_stride={future_stride}, offsets={video_offsets}"
                        )
                    future_views.append(frames[future_position])

            if image_layout in _COMPOSITE_LAYOUTS:
                expected_source_keys = list(_cfg_get(self.data_cfg, "composite_source_view_keys", []))
                if expected_source_keys and source_view_keys != expected_source_keys:
                    raise ValueError(
                        "Composite camera order mismatch: "
                        f"dataset={source_view_keys}, configured={expected_source_keys}"
                    )
                composite_size, default_view_key, composite_builder = (
                    _composite_contract(image_layout)
                )
                configured_size = tuple(
                    int(value)
                    for value in _cfg_get(
                        self.data_cfg, "obs_image_size", composite_size
                    )
                )
                if configured_size != composite_size:
                    raise ValueError(
                        f"{image_layout} must be configured as {composite_size} "
                        "(width,height), "
                        f"got {configured_size}"
                    )
                composite_view_key = str(
                    _cfg_get(
                        self.data_cfg,
                        "composite_view_key",
                        default_view_key,
                    )
                )
                img0 = [
                    np.asarray(composite_builder(current_views), dtype=np.uint8)
                ]
                img1 = (
                    [
                        np.asarray(
                            composite_builder(future_views), dtype=np.uint8
                        )
                    ]
                    if decode_future
                    else []
                )
                view_keys = [composite_view_key]
                if self._text_history_offsets:
                    history_cfg = _cfg_get(self._text_config, "history", {})
                    history_image_field = str(
                        _cfg_get(
                            history_cfg,
                            "image_field",
                            "planner_history_images",
                        )
                    )
                    history_images = []
                    for offset in self._text_history_offsets:
                        if offset not in video_offsets:
                            raise ValueError(
                                f"Planner history offset {offset} missing from video offsets {video_offsets}"
                            )
                        position = video_offsets.index(offset)
                        history_views = [
                            np.asarray(self._last_video_frames[key])[position]
                            for key in source_view_keys
                        ]
                        history_images.append(
                            np.asarray(
                                composite_builder(history_views),
                                dtype=np.uint8,
                            )
                        )
                    sample[history_image_field] = np.stack(
                        history_images,
                        axis=0,
                    )
            else:
                img0, img1 = [], []
                target_size = _cfg_get(self.data_cfg, "obs_image_size", None)
                target_size = tuple(int(v) for v in target_size) if target_size else None
                for current_array in current_views:
                    current = Image.fromarray(current_array)
                    if target_size and current.size != target_size:
                        current = current.resize(target_size)
                    img0.append(np.asarray(current))
                for future_array in future_views:
                    future = Image.fromarray(future_array)
                    if target_size and future.size != target_size:
                        future = future.resize(target_size)
                    img1.append(np.asarray(future))
                view_keys = source_view_keys
            sample["image_0"] = np.stack(img0, axis=0)
            if img1:
                sample["image_1"] = np.stack(img1, axis=0)
            sample["image_view_keys"] = view_keys
            sample["dino_view_keys"] = view_keys
            if image_layout in _COMPOSITE_LAYOUTS:
                sample["dino_target_view_keys"] = view_keys
            #######
            if self._dino_target_latents:
                target_view_keys = self._dino_target_view_keys()
                if bool(_cfg_get(self.data_cfg, "load_current_dino_target", True)):
                    sample["dino_0"] = self._read_dino(trajectory_id, base_index)
                sample["dino_1"] = self._read_dino(trajectory_id, future_index)
                # Keep image and latent metadata separate: current policy RGB may have
                # three cameras while the future DINO target intentionally has only one.
                sample["dino_target_view_keys"] = target_view_keys
                sample["dino_view_keys"] = target_view_keys
            #######
        else:
            sample["dino_0"] = self._read_dino(trajectory_id, base_index)
            sample["dino_1"] = self._read_dino(trajectory_id, future_index)
            sample["dino_target_view_keys"] = self._dino_target_view_keys()
            sample["dino_view_keys"] = sample["dino_target_view_keys"]

        if self.data_cfg is not None and _cfg_get(self.data_cfg, "include_state", True) not in ["False", False]:
            state = []
            for state_key in self.modality_keys.get("state", []):
                if state_key in data:
                    state.append(_to_numpy(data[state_key]))
            if state:
                sample["state"] = np.concatenate(state, axis=1).astype(np.float32)
        #######
        if self._episode_latent_cache:
            self._episode_latent_cache.clear()
        #######
        return sample


######### // code // ##########


######### // code // ##########
class JointFastWAMRobotWinDataset(JointLiberoDataset):
    """JointFlow sample ABI over FastWAM's aggregate RoboTwin release.

    The aggregate release has a single global statistics file instead of the
    per-task metadata layout consumed by ``JointLiberoDataset``.  This adapter
    preserves FastWAM's release-order z-score contract and direct frame index,
    while reusing JointFlow's current/future composite packing required by
    ``QwenWorldActionMoT``.
    """

    _STAT_NAMES = ("min", "max", "mean", "std", "q01", "q99")

    def __init__(self, *args, **kwargs) -> None:
        data_cfg = kwargs.get("data_cfg")
        configured_stats = _cfg_get(data_cfg, "fastwam_dataset_stats_path", None)
        self._fastwam_stats_path = Path(str(configured_stats)) if configured_stats else None
        self._fastwam_expected_fps = float(
            _cfg_get(data_cfg, "fastwam_expected_fps", 50.0)
        )
        self._fastwam_val_fraction = float(
            _cfg_get(data_cfg, "fastwam_val_fraction", 0.0)
        )
        self._fastwam_split = str(
            _cfg_get(data_cfg, "fastwam_split", "all")
        ).lower()
        self._fastwam_split_seed = int(
            _cfg_get(data_cfg, "fastwam_split_seed", 42)
        )
        self._fastwam_domain = str(
            _cfg_get(data_cfg, "fastwam_domain", "all")
        ).lower()
        if not 0.0 <= self._fastwam_val_fraction < 1.0:
            raise ValueError(
                "fastwam_val_fraction must be in [0,1), got "
                f"{self._fastwam_val_fraction}"
            )
        if self._fastwam_split not in {"all", "train", "val", "validation"}:
            raise ValueError(f"Unsupported fastwam_split={self._fastwam_split!r}")
        if self._fastwam_domain not in {"all", "clean", "random", "randomized"}:
            raise ValueError(f"Unsupported fastwam_domain={self._fastwam_domain!r}")
        super().__init__(*args, **kwargs)
        self._trajectory_index_by_id = {
            int(trajectory_id): index
            for index, trajectory_id in enumerate(self.trajectory_ids)
        }

    def _get_metadata(self, embodiment_tag: EmbodimentTag):
        from starVLA.dataloader.fastwam_robotwin_dataset import (
            build_fastwam_dataset_metadata,
        )

        stats_path = self._fastwam_stats_path
        if stats_path is not None and not stats_path.is_absolute():
            stats_path = self.dataset_path / stats_path
        return build_fastwam_dataset_metadata(
            self.dataset_path,
            embodiment_tag,
            stats_path=stats_path,
            expected_fps=self._fastwam_expected_fps,
        )

    def _get_lerobot_modality_meta(self):
        from starVLA.dataloader.fastwam_robotwin_dataset import (
            _build_modality_metadata,
        )

        return _build_modality_metadata()

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        from starVLA.dataloader.fastwam_robotwin_dataset import _episode_domain

        episodes_path = self.dataset_path / "meta" / "episodes.jsonl"
        with episodes_path.open("r", encoding="utf-8") as handle:
            episodes = [json.loads(line) for line in handle if line.strip()]

        wanted_domain = (
            "randomized"
            if self._fastwam_domain == "random"
            else self._fastwam_domain
        )
        if wanted_domain != "all":
            classified = [
                (episode, _episode_domain(episode)) for episode in episodes
            ]
            unknown = [
                episode.get("episode_index")
                for episode, domain in classified
                if domain is None
            ]
            if unknown:
                raise ValueError(
                    "fastwam_domain filtering requires Clean/Randomized "
                    "provenance in meta/episodes.jsonl; "
                    f"{len(unknown)} episodes are unclassified, first={unknown[:5]}"
                )
            episodes = [
                episode
                for episode, domain in classified
                if domain == wanted_domain
            ]

        if not episodes:
            raise ValueError(
                "No FastWAM episodes remain after "
                f"fastwam_domain={self._fastwam_domain}"
            )

        if self._fastwam_split != "all" and self._fastwam_val_fraction > 0:
            split_index = int(
                len(episodes) * (1.0 - self._fastwam_val_fraction)
            )
            order = list(range(len(episodes)))
            rng = np.random.default_rng(self._fastwam_split_seed)
            rng.shuffle(order)
            chosen = (
                order[:split_index]
                if self._fastwam_split == "train"
                else order[split_index:]
            )
            episodes = [episodes[index] for index in chosen]

        trajectory_ids = np.asarray(
            [int(episode["episode_index"]) for episode in episodes],
            dtype=np.int64,
        )
        trajectory_lengths = np.asarray(
            [int(episode["length"]) for episode in episodes],
            dtype=np.int64,
        )
        return trajectory_ids, trajectory_lengths

    def _get_all_steps(self):
        from starVLA.dataloader.fastwam_robotwin_dataset import LazyEpisodeSteps

        return LazyEpisodeSteps(self.trajectory_ids, self.trajectory_lengths)

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)
        trajectory_id = int(self._last_trajectory_id)
        base_index = int(self._last_base_index)
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = int(self.trajectory_lengths[trajectory_index])
        action_key = self.modality_keys["action"][0]
        action_offsets = np.asarray(self.delta_indices[action_key], dtype=np.int64)
        sample["action_is_pad"] = (
            base_index + action_offsets >= trajectory_length
        )
        return sample

    def save_dataset_statistics(
        self,
        save_path: Path | str,
        format: str = "json",  # noqa: A002 - public dataset API
    ) -> None:
        """Save statistics in the exact release-order deployment contract."""

        if format.lower() != "json":
            raise ValueError(f"Unsupported statistics format: {format}")

        def combine(modality: str) -> dict[str, list]:
            stats = getattr(self.metadata.statistics, modality)
            keys = [
                key.split(".", 1)[1]
                for key in self.modality_keys[modality]
            ]
            return {
                stat_name: np.concatenate(
                    [
                        np.asarray(
                            getattr(stats[key], stat_name),
                            dtype=np.float64,
                        )
                        for key in keys
                    ]
                ).tolist()
                for stat_name in self._STAT_NAMES
            }

        action_stats = combine("action")
        action_stats["mask"] = [True] * sum(
            int(np.prod(self.metadata.modalities.action[key].shape))
            for key in [
                name.split(".", 1)[1]
                for name in self.modality_keys["action"]
            ]
        )
        payload = {
            self.tag: {
                "action": action_stats,
                "state": combine("state"),
                "num_transitions": len(self),
                "num_trajectories": len(self.trajectory_ids),
            }
        }
        path = Path(save_path)
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"FastWAM JointFlow statistics saved to: {path}")

    def get_trajectory_index(self, trajectory_id: int) -> int:
        try:
            return self._trajectory_index_by_id[int(trajectory_id)]
        except AttributeError:
            # Initialization queries trajectories before this lookup exists.
            return int(super().get_trajectory_index(trajectory_id))
        except KeyError as exc:
            raise ValueError(f"Unknown trajectory id: {trajectory_id}") from exc

    def get_trajectory_data(self, trajectory_id: int):
        trajectory_id = int(trajectory_id)
        data = super().get_trajectory_data(trajectory_id)
        self.curr_traj_id = trajectory_id
        self.curr_traj_data = data
        return data


######### // code // ##########
def _make_joint_single_dataset(
    dataset_path: Path, robot_type: str, data_cfg, online_dino: bool | None = None
) -> JointLiberoDataset:
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = copy.deepcopy(data_config.modality_config())

    action_horizon = int(_cfg_get(data_cfg, "action_horizon", _cfg_get(data_cfg, "future_action_window_size", 8)))
    if online_dino is None:
        raw = _cfg_get(data_cfg, "online_dino", True)
        online_dino = True if (isinstance(raw, str) and raw.strip().lower() == "auto") else bool(raw)
    world_model_cfg = _cfg_get(data_cfg, "world_model", None)
    future_stride = _cfg_get(world_model_cfg, "future_stride", None)
    if future_stride is None:
        future_stride = _cfg_get(data_cfg, "future_dino_stride", action_horizon)
    future_stride = max(int(future_stride), 1)
    if "video" in modality_config:
        decode_future_video = bool(_cfg_get(data_cfg, "decode_future_video", True))
        text_config = _cfg_get(data_cfg, "text_annotations", {})
        history_offsets = text_history_frame_offsets(text_config)
        base_video_delta = (
            ([0, future_stride] if decode_future_video else [0])
            if online_dino
            else [0, 1]
        )
        video_delta = list(
            dict.fromkeys(
                [
                    *history_offsets,
                    *base_video_delta,
                ]
            )
        )
        modality_config["video"] = ModalityConfig(
            delta_indices=video_delta,
            modality_keys=modality_config["video"].modality_keys,
        )
    if "language" in modality_config:
        modality_config["language"] = ModalityConfig(
            delta_indices=[0], modality_keys=modality_config["language"].modality_keys
        )
    if "state" in modality_config:
        modality_config["state"] = ModalityConfig(
            delta_indices=[0], modality_keys=modality_config["state"].modality_keys
        )
    if "action" in modality_config:
        modality_config["action"] = ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=modality_config["action"].modality_keys,
        )

    transforms = _drop_video_transforms(data_config.transform())
    state_norm_modes = _cfg_get(data_cfg, "state_norm_modes", None)
    state_config = modality_config.get("state", ModalityConfig(delta_indices=[], modality_keys=[]))
    transforms = _append_state_norm_if_needed(transforms, state_config.modality_keys, state_norm_modes)

    embodiment_tag = getattr(data_config, "embodiment_tag", None) or EmbodimentTag.NEW_EMBODIMENT
    if hasattr(data_config, "make_joint_dataset"):
        return data_config.make_joint_dataset(
            dataset_path=dataset_path,
            modality_configs=modality_config,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=_cfg_get(data_cfg, "video_backend", "torchvision_av"),
            delete_pause_frame=bool(_cfg_get(data_cfg, "delete_pause_frame", False)),
            data_cfg=data_cfg,
            online_dino=online_dino,
            dino_feature_dir=_cfg_get(data_cfg, "dino_feature_dir", "latents"),
            dino_episode_cache_size=int(_cfg_get(data_cfg, "dino_episode_cache_size", 2)),
            dino_target_latents=bool(_cfg_get(data_cfg, "dino_target_latents", False)),
        )
    return JointLiberoDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=_cfg_get(data_cfg, "video_backend", "torchvision_av"),
        delete_pause_frame=bool(_cfg_get(data_cfg, "delete_pause_frame", False)),
        data_cfg=data_cfg,
        online_dino=online_dino,
        dino_feature_dir=_cfg_get(data_cfg, "dino_feature_dir", "latents"),
        dino_episode_cache_size=int(_cfg_get(data_cfg, "dino_episode_cache_size", 2)),
        dino_target_latents=bool(_cfg_get(data_cfg, "dino_target_latents", False)),
    )


######### // code // ##########
def _dataset_offline_latents_dim(dataset_path: Path, feature_dir: str) -> int | None:
    root = Path(dataset_path) / str(feature_dir)
    stats_path = root / "dino_v3_stats.json"
    if not stats_path.exists():
        return None
    has_lingbot = (root / "dino_v3_index.json").exists()
    has_memmap = (root / "index.json").exists() and (root / "features.float16.mmap").exists()
    if not (has_lingbot or has_memmap):
        return None
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        mean = stats.get("mean", [])
        std = stats.get("std", [])
        if not mean or len(mean) != len(std):
            return None
        return int(len(mean))
    except Exception:
        return None


def _inspect_precomputed_dino_store(
    dataset_path: Path,
    feature_dir: str,
    target_view_keys: list[str],
    expected_dim: int,
    expected_tokens: int,
) -> list[str]:
    """Check store metadata and lightweight first/last-file sentinels without loading latent tensors."""
    root = Path(dataset_path) / str(feature_dir)
    dim = _dataset_offline_latents_dim(dataset_path, feature_dir)
    if dim is None:
        return [f"missing/invalid {feature_dir} DINO store"]

    problems = []
    if dim != expected_dim:
        problems.append(f"stats dim {dim} != configured DINO dim {expected_dim}")

    # Legacy memmap stores have no per-view episode files. Their shape and stats
    # are validated above and their selected view axis is checked when sampled.
    index_path = root / "dino_v3_index.json"
    if not index_path.exists():
        return problems
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    except Exception as exc:
        return problems + [f"cannot read dino_v3_index.json: {exc}"]

    layout = index.get("layout", None)
    if layout is not None and str(layout) != "lingbot_episode":
        problems.append(f"unsupported index layout {layout!r}")
    declared_root = index.get("root", None)
    if declared_root is not None and Path(str(declared_root)).name != Path(feature_dir).name:
        problems.append(f"index root {declared_root!r} != configured feature dir {feature_dir!r}")
    feature_type = index.get("feature_type", None)
    if feature_type is not None and str(feature_type) != "dinov3_patch_tokens":
        problems.append(f"feature_type {feature_type!r} is not dinov3_patch_tokens")
    if index.get("feature_dim", None) is not None and int(index["feature_dim"]) != expected_dim:
        problems.append(f"index feature_dim {index['feature_dim']} != {expected_dim}")
    if index.get("num_tokens", None) is not None and int(index["num_tokens"]) != expected_tokens:
        problems.append(f"index num_tokens {index['num_tokens']} != {expected_tokens}")

    view_keys = [str(key) for key in index.get("view_keys", [])]
    original_view_keys = [str(key) for key in index.get("original_view_keys", [])]
    if not view_keys or len(view_keys) != len(original_view_keys):
        problems.append(
            "index must contain aligned non-empty view_keys and original_view_keys "
            f"(got {len(view_keys)} and {len(original_view_keys)})"
        )
        return problems

    trajectory_ids = [int(value) for value in index.get("trajectory_ids", [])]
    sentinel_ids = list(dict.fromkeys(trajectory_ids[:1] + trajectory_ids[-1:]))
    for target_view in target_view_keys:
        if target_view not in view_keys:
            problems.append(f"target view {target_view!r} missing from index view_keys={view_keys}")
            continue
        original_view = original_view_keys[view_keys.index(target_view)]
        if sentinel_ids:
            for trajectory_id in sentinel_ids:
                pattern = f"chunk-*/{original_view}/episode_{trajectory_id:06d}_0_*.pth"
                if next(root.glob(pattern), None) is None:
                    problems.append(f"missing target-view sentinel {pattern}")
        elif next(root.glob(f"chunk-*/{original_view}/episode_*.pth"), None) is None:
            problems.append(f"no episode files for target view {target_view!r} ({original_view})")
    return problems


def _validate_precomputed_dino_targets(cfg, mixture_spec) -> None:
    """Fail before training unless every configured dataset has the requested latent store.

    Validation work is sharded across distributed ranks so a 100-dataset RoboTwin mix
    performs roughly two metadata reads per rank rather than 100 reads on every rank.
    Individual episode files remain checked by ``_load_episode_view_latent`` when sampled.
    """
    vla_cfg = cfg.datasets.vla_data
    if not bool(_cfg_get(vla_cfg, "require_precomputed_dino_targets", False)):
        return
    if not bool(_cfg_get(vla_cfg, "dino_target_latents", False)):
        raise ValueError("require_precomputed_dino_targets=true requires dino_target_latents=true.")

    target_view_keys = _configured_dino_target_view_keys(vla_cfg)
    if target_view_keys is None:
        raise ValueError(
            "require_precomputed_dino_targets=true requires explicit dino_target_view_keys; "
            "index-only target selection is not strict enough."
        )

    framework_cfg = getattr(cfg, "framework", None)
    dino_cfg = getattr(framework_cfg, "dino", None) if framework_cfg is not None else None
    if dino_cfg is None:
        raise ValueError("Strict precomputed DINO targets require framework.dino configuration.")
    from starVLA.model.framework.VLM4A.jointflow.dino_v3 import dino_num_patches, resolve_dino_spec

    dino_spec = resolve_dino_spec(dino_cfg)
    expected_dim = int(dino_spec["embed_dim"])
    expected_tokens = dino_num_patches(dino_spec["image_size"], dino_spec["patch_size"])
    feature_dir = str(_cfg_get(vla_cfg, "dino_feature_dir", "latents"))
    root = Path(vla_cfg.data_root_dir)

    # Check the requested canonical camera name against every embodiment in the mix.
    view_problems = []
    for robot_type in sorted({str(entry[2]) for entry in mixture_spec}):
        data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        video_cfg = data_config.modality_config().get("video")
        available = list(video_cfg.modality_keys) if video_cfg is not None else []
        missing = [key for key in target_view_keys if key not in available]
        if missing:
            view_problems.append(f"robot_type={robot_type}: missing target views {missing}; available={available}")
    if view_problems:
        raise ValueError("Invalid precomputed DINO target view configuration: " + "; ".join(view_problems))

    entries = []
    seen = set()
    for data_name, _weight, _robot_type in mixture_spec:
        data_name = str(data_name)
        if data_name not in seen:
            seen.add(data_name)
            entries.append(data_name)

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    world_size = torch.distributed.get_world_size() if distributed else 1
    local_problems = []
    for index, data_name in enumerate(entries):
        if index % world_size != rank:
            continue
        try:
            store_problems = _inspect_precomputed_dino_store(
                root / data_name,
                feature_dir,
                target_view_keys,
                expected_dim,
                expected_tokens,
            )
        except Exception as exc:
            # Every rank must still reach all_gather_object; turn malformed metadata
            # or filesystem errors into validation results instead of deadlocking peers.
            store_problems = [f"inspection error ({type(exc).__name__}): {exc}"]
        local_problems.extend(f"{data_name}: {problem}" for problem in store_problems)

    if distributed:
        gathered: list[list[str] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local_problems)
        problems = sorted(problem for rank_problems in gathered for problem in (rank_problems or []))
    else:
        problems = local_problems
    if problems:
        shown = "; ".join(problems[:8]) + (" ..." if len(problems) > 8 else "")
        raise RuntimeError(
            "Strict precomputed DINO target validation failed; online fallback is disabled. "
            f"Expected tokens={expected_tokens}, dim={expected_dim}, views={target_view_keys}. {shown}"
        )
    if rank == 0:
        print(
            f"[jointflow] strict precomputed DINO targets verified: {len(entries)} stores, "
            f"dir={feature_dir}, tokens={expected_tokens}, dim={expected_dim}, views={target_view_keys}",
            flush=True,
        )


def _resolve_online_dino(cfg, mixture_spec) -> bool:
    vla_cfg = cfg.datasets.vla_data
    raw = _cfg_get(vla_cfg, "online_dino", True)
    if not (isinstance(raw, str) and raw.strip().lower() == "auto"):
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)

    expected_dim = None
    framework_cfg = getattr(cfg, "framework", None)
    dino_cfg = getattr(framework_cfg, "dino", None) if framework_cfg is not None else None
    if dino_cfg is not None:
        try:
            from starVLA.model.framework.VLM4A.jointflow.dino_v3 import resolve_dino_spec

            expected_dim = int(resolve_dino_spec(dino_cfg)["embed_dim"])
        except Exception:
            expected_dim = None

    feature_dir = _cfg_get(vla_cfg, "dino_feature_dir", "latents")
    root = Path(vla_cfg.data_root_dir)
    problems = []
    for data_name, _, _ in mixture_spec:
        dim = _dataset_offline_latents_dim(root / str(data_name), feature_dir)
        if dim is None:
            problems.append(f"{data_name}: no latents")
        elif expected_dim is not None and dim != expected_dim:
            problems.append(f"{data_name}: latents dim {dim} != expected {expected_dim}")
    online = bool(problems)
    if online:
        shown = "; ".join(problems[:3]) + (" ..." if len(problems) > 3 else "")
        print(f"[jointflow] online_dino=auto -> ONLINE ({shown})", flush=True)
    else:
        print(
            f"[jointflow] online_dino=auto -> OFFLINE latents "
            f"({len(mixture_spec)} datasets, dim={expected_dim if expected_dim is not None else 'unchecked'})",
            flush=True,
        )
    return online


######### // code // ##########


def _validate_io_shortcuts(cfg) -> None:
    """Reject I/O shortcuts unless the configured WAM graph makes omitted fields unreachable."""
    vla_cfg = cfg.datasets.vla_data
    decode_future_video = bool(_cfg_get(vla_cfg, "decode_future_video", True))
    load_current_dino = bool(_cfg_get(vla_cfg, "load_current_dino_target", True))
    if decode_future_video and load_current_dino:
        return

    framework_cfg = getattr(cfg, "framework", None)
    wam_cfg = getattr(framework_cfg, "wam", None) if framework_cfg is not None else None
    wam_enabled = wam_cfg is not None and bool(wam_cfg.get("enabled", False))
    tasks_cfg = getattr(framework_cfg, "tasks", None) if framework_cfg is not None else None
    weights = tasks_cfg.get("weights", {}) if tasks_cfg is not None else {}

    if not decode_future_video:
        if not wam_enabled or float(weights.get("idm", 0.0)) > 0.0:
            raise ValueError("decode_future_video=false is only valid for WAM training without the IDM task.")
        if not bool(_cfg_get(vla_cfg, "dino_target_latents", False)):
            raise ValueError("decode_future_video=false requires dino_target_latents=true for future supervision.")

    if not load_current_dino:
        world_target = str(wam_cfg.get("world_target", "absolute")).lower() if wam_enabled else ""
        if not wam_enabled or world_target != "absolute":
            raise ValueError("load_current_dino_target=false requires WAM world_target=absolute.")


def build_joint_dataset(cfg, mode: str = "train"):
    _validate_io_shortcuts(cfg)
    vla_cfg = cfg.datasets.vla_data
    mixture_spec = resolve_data_mix(vla_cfg.data_mix)
    _validate_precomputed_dino_targets(cfg, mixture_spec)
    online_dino = _resolve_online_dino(cfg, mixture_spec)
    dataset_mixture = []
    seen = set()
    for data_name, weight, robot_type in mixture_spec:
        key = (data_name, robot_type)
        if key in seen:
            continue
        seen.add(key)
        dataset_path = Path(vla_cfg.data_root_dir) / data_name
        dataset_mixture.append(
            (_make_joint_single_dataset(dataset_path, robot_type, vla_cfg, online_dino=online_dino), weight)
        )

    if bool(_cfg_get(vla_cfg, "fastwam_direct_frame_sampling", False)):
        if len(dataset_mixture) != 1:
            raise ValueError(
                "fastwam_direct_frame_sampling requires exactly one JointFlow "
                f"dataset entry, got {len(dataset_mixture)}"
            )
        return dataset_mixture[0][0]

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=vla_cfg.get("balance_dataset_weights", False),
        balance_trajectory_weights=vla_cfg.get("balance_trajectory_weights", False),
        seed=int(getattr(cfg, "seed", 42)),
        data_cfg=vla_cfg,
    )


def build_joint_dataloader(cfg, mode: str = "train") -> DataLoader:
    dataset = build_joint_dataset(cfg, mode=mode)
    workers = int(cfg.datasets.vla_data.get("num_workers", 16))
    prefetch_factor = int(cfg.datasets.vla_data.get("prefetch_factor", 2)) if workers > 0 else None
    persistent_workers = bool(cfg.datasets.vla_data.get("persistent_workers", workers > 0)) if workers > 0 else False
    pin_memory = bool(cfg.datasets.vla_data.get("pin_memory", True))
    loader_kwargs = {}
    if bool(_cfg_get(cfg.datasets.vla_data, "fastwam_direct_frame_sampling", False)):
        from starVLA.dataloader.fastwam_robotwin_dataset import FastWAMEpochSampler

        loader_kwargs["sampler"] = FastWAMEpochSampler(
            dataset,
            seed=int(
                _cfg_get(
                    cfg.datasets.vla_data,
                    "fastwam_split_seed",
                    getattr(cfg, "seed", 42),
                )
            ),
        )
    return DataLoader(
        dataset,
        batch_size=int(cfg.datasets.vla_data.per_device_batch_size),
        collate_fn=collate_fn,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        **loader_kwargs,
    )


def build_event_memory_dataloader(
    cfg,
    *,
    output_dir: str | Path | None = None,
) -> DataLoader | None:
    """Build the opt-in semantic NTP stream used by event-memory training.

    The physical dataloader remains untouched.  This second stream decodes only
    the current RGB observation and uses an exact 2/2/2 boundary sampler.  A
    single dataset is required because sampler indices address one episode-local
    frame table directly; the RoboDojo text recipe intentionally satisfies this
    contract.
    """

    vla_cfg = cfg.datasets.vla_data
    text_cfg = _cfg_get(vla_cfg, "text_annotations", {})
    event_cfg = _cfg_get(text_cfg, "event_memory", {})
    if not bool(_cfg_get(event_cfg, "enabled", False)):
        return None

    mixture_spec = resolve_data_mix(vla_cfg.data_mix)
    unique_entries: list[tuple[str, str]] = []
    seen = set()
    for data_name, _weight, robot_type in mixture_spec:
        key = (str(data_name), str(robot_type))
        if key not in seen:
            seen.add(key)
            unique_entries.append(key)
    if len(unique_entries) != 1:
        raise ValueError(
            "Event-memory semantic sampling currently requires exactly one "
            f"dataset/embodiment entry, got {unique_entries}"
        )

    # Do not mutate the physical recipe.  The semantic stream has no future
    # world target and needs only the current observation plus parquet labels.
    if OmegaConf.is_config(vla_cfg):
        semantic_payload = OmegaConf.to_container(vla_cfg, resolve=True)
    elif callable(getattr(vla_cfg, "to_dict", None)):
        semantic_payload = vla_cfg.to_dict(resolve=True)
    elif hasattr(vla_cfg, "items"):
        semantic_payload = dict(vla_cfg.items())
    else:
        raise TypeError(
            "datasets.vla_data must be an OmegaConf/config mapping for the "
            "event-memory semantic stream"
        )
    semantic_data_cfg = OmegaConf.create(semantic_payload)
    semantic_data_cfg.online_dino = True
    semantic_data_cfg.decode_future_video = False
    semantic_data_cfg.action_horizon = 1
    semantic_data_cfg.future_action_window_size = 1
    semantic_data_cfg.load_current_dino_target = False
    semantic_data_cfg.dino_target_latents = False
    semantic_data_cfg.require_precomputed_dino_targets = False

    data_name, robot_type = unique_entries[0]
    dataset = _make_joint_single_dataset(
        Path(semantic_data_cfg.data_root_dir) / data_name,
        robot_type,
        semantic_data_cfg,
        online_dino=True,
    )

    cache_root = Path(output_dir or getattr(cfg, "output_dir", "."))
    index_name = str(_cfg_get(event_cfg, "index_cache_name", "semantic_index_v1.npz"))
    pools = build_or_load_semantic_index(
        dataset,
        semantic_data_cfg.text_annotations,
        cache_root / index_name,
    )
    sampler_cfg = _cfg_get(event_cfg, "sampler", {})
    sampler = SemanticBoundaryBatchSampler(
        pools,
        seed=int(_cfg_get(sampler_cfg, "seed", getattr(cfg, "seed", 42))),
        update_per_batch=int(_cfg_get(sampler_cfg, "update_per_batch", 2)),
        hard_keep_per_batch=int(_cfg_get(sampler_cfg, "hard_keep_per_batch", 2)),
        random_keep_per_batch=int(_cfg_get(sampler_cfg, "random_keep_per_batch", 2)),
    )
    expected_batch = int(_cfg_get(sampler_cfg, "per_device_batch_size", 6))
    if sampler.batch_size != expected_batch:
        raise ValueError(
            "Semantic sampler category counts must sum to per_device_batch_size: "
            f"counts={sampler.batch_size}, configured={expected_batch}"
        )

    workers = int(_cfg_get(sampler_cfg, "num_workers", 4))
    loader_kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(_cfg_get(sampler_cfg, "pin_memory", True)),
        "persistent_workers": bool(
            _cfg_get(sampler_cfg, "persistent_workers", workers > 0)
        )
        if workers > 0
        else False,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(
            _cfg_get(sampler_cfg, "prefetch_factor", 2)
        )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        **loader_kwargs,
    )


######### // code // ##########


#######
def compute_action_correlation_cholesky(
    mixture_dataset: LeRobotMixtureDataset,
    num_samples: int = 20000,
    beta: float = 0.5,
    seed: int = 0,
    matrix_type: str = "covariance",
) -> np.ndarray:
    """Estimate the correlated-noise Cholesky from action chunks.

    Sampling follows the mixture's dataset and trajectory weights, then draws
    a step uniformly inside the selected trajectory. Dual-query datasets use
    ``read_action_only``; generic datasets fall back to ``__getitem__``. Bad
    samples are skipped independently so one damaged trajectory cannot abort
    startup.
    """
    target = int(num_samples)
    if target < 2:
        raise ValueError(f"num_samples must be at least 2, got {target}.")
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"correlation beta must be in [0, 1], got {beta}.")

    rng = np.random.default_rng(seed)
    datasets = list(getattr(mixture_dataset, "datasets", None) or [mixture_dataset])
    dataset_probs = getattr(mixture_dataset, "dataset_sampling_weights", None)
    if dataset_probs is None or len(dataset_probs) != len(datasets):
        dataset_probs = np.ones(len(datasets), dtype=np.float64)
    dataset_probs = np.asarray(dataset_probs, dtype=np.float64)
    dataset_probs = dataset_probs / dataset_probs.sum()
    mixture_trajectory_probs = getattr(mixture_dataset, "trajectory_sampling_weights", None)
    rows: list[np.ndarray] = []
    tries = 0
    rejected = 0
    last_error: Exception | None = None
    max_tries = max(target * 4, 16)
    while len(rows) < target and tries < max_tries:
        tries += 1
        dataset_index = int(rng.choice(len(datasets), p=dataset_probs))
        d = datasets[dataset_index]
        try:
            if hasattr(d, "read_action_only"):
                data_cfg = getattr(d, "data_cfg", None)
                direct_frames = (
                    bool(data_cfg.get("fastwam_direct_frame_sampling", False)) if data_cfg is not None else False
                )
                if direct_frames:
                    # Match FastWAM training's global-frame distribution: draw
                    # uniformly from selected frames, then resolve episode/step.
                    traj_id, base_index = d.all_steps[int(rng.integers(0, len(d)))]
                    a = d.read_action_only(int(traj_id), int(base_index))
                else:
                    num_trajectories = len(d.trajectory_ids)
                    if mixture_trajectory_probs is not None and dataset_index < len(mixture_trajectory_probs):
                        trajectory_probs = np.asarray(mixture_trajectory_probs[dataset_index], dtype=np.float64)
                    else:
                        trajectory_probs = np.ones(num_trajectories, dtype=np.float64)
                    trajectory_probs = trajectory_probs / trajectory_probs.sum()
                    trajectory_index = int(rng.choice(num_trajectories, p=trajectory_probs))
                    traj_id = int(d.trajectory_ids[trajectory_index])
                    base_index = int(rng.integers(0, int(d.trajectory_lengths[trajectory_index])))
                    a = d.read_action_only(int(traj_id), int(base_index))
            else:
                a = mixture_dataset[int(rng.integers(0, len(mixture_dataset)))].get("action")
        except Exception as exc:
            rejected += 1
            last_error = exc
            continue
        if a is None:
            continue
        a = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
        flattened = a.reshape(-1).astype(np.float32)
        if not np.isfinite(flattened).all():
            rejected += 1
            last_error = ValueError("sample action contains NaN or infinite values")
            continue
        if rows and flattened.shape != rows[0].shape:
            rejected += 1
            last_error = ValueError(
                f"inconsistent flattened action shape {flattened.shape}; expected {rows[0].shape}"
            )
            continue
        rows.append(flattened)
    if len(rows) != target:
        detail = f"; last_error={type(last_error).__name__}: {last_error}" if last_error is not None else ""
        raise RuntimeError(
            "compute_action_correlation_cholesky could not collect the requested action sample count: "
            f"collected={len(rows)}, requested={target}, tries={tries}, rejected={rejected}{detail}"
        )
    X = np.stack(rows, axis=0)  # [M, flat]
    flat = X.shape[1]
    action_matrix = compute_action_noise_matrix(X, matrix_type=matrix_type)
    Sigma_reg = beta * action_matrix + (1.0 - beta) * np.eye(flat)
    L = np.linalg.cholesky(Sigma_reg + 1e-6 * np.eye(flat))
    from starVLA.dataloader.action_correlation import validate_action_correlation_cholesky

    return validate_action_correlation_cholesky(L, expected_size=flat)


#######


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/jointflow/configs/jointflow_libero.yaml")
    parser.add_argument("--num_batches", type=int, default=1)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    loader = build_joint_dataloader(cfg)
    for idx, batch in enumerate(loader):
        sample = batch[0]
        info = {
            "action": np.asarray(sample["action"]).shape,
            "state": np.asarray(sample.get("state", [])).shape,
            "future_valid": sample.get("future_valid"),
            "future_stride": int(sample.get("future_stride", -1)),
            "lang": sample["lang"][:80],
        }
        if "image_0" in sample:
            info["image_0"] = np.asarray(sample["image_0"]).shape
            info["image_1"] = np.asarray(sample["image_1"]).shape
            info["dino_view_keys"] = sample.get("dino_view_keys")
        else:
            info["dino_0"] = np.asarray(sample["dino_0"]).shape
            info["dino_1"] = np.asarray(sample["dino_1"]).shape
        print(info)
        if idx + 1 >= args.num_batches:
            break
