"""FastWAM RoboTwin LeRobot v2.1 adapter.

This module intentionally preserves the data contract used by the original
``starvla_qwengroot_robotwin_fastwam`` run: a single 384x320 composite image,
14-D proprioception/action vectors in release order, padded action masks, the
official seeded episode split, and without-replacement frame sampling.  The
Co-Flow sample ABI is one H16 action chunk paired with one t+16 future image.
Those details are part of the checkpoint ABI, not cosmetic preprocessing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

from starVLA.dataloader.fastwam_image import build_robotwin_composite
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    LeRobotModalityMetadata,
)

_VECTOR_SLICES = {
    "left_joints": slice(0, 6),
    "left_gripper": slice(6, 7),
    "right_joints": slice(7, 13),
    "right_gripper": slice(13, 14),
}
_VIDEO_KEYS = {
    "cam_high": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}
_STAT_NAMES = ("min", "max", "mean", "std", "q01", "q99")
_COMPOSITE_VIEW_KEY = "video.robotwin_composite"


def _episode_domain(episode: dict) -> str | None:
    """Infer Clean/Randomized provenance from FastWAM's source metadata."""

    provenance_keys = (
        "raw_file_name",
        "source_path",
        "source_file",
        "dataset_path",
        "relative_path",
        "setting",
        "domain",
    )
    values = [str(episode[key]) for key in provenance_keys if episode.get(key) is not None]
    normalized = "/" + "/".join(values).replace("\\", "/").lower().strip("/") + "/"
    randomized = any(marker in normalized for marker in ("/randomized/", "/random/", "_randomized", "-randomized"))
    clean = any(marker in normalized for marker in ("/clean/", "_clean", "-clean"))
    if randomized and not clean:
        return "randomized"
    if clean and not randomized:
        return "clean"
    return None


class FastWAMEpochSampler(Sampler[int]):
    """Seeded, without-replacement permutation over selected global frames."""

    def __init__(self, data_source, seed: int = 42) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


class LazyEpisodeSteps(Sequence[tuple[int, int]]):
    """Map a global frame index to ``(episode_id, frame_id)`` in O(log episodes)."""

    def __init__(self, episode_ids: np.ndarray, episode_lengths: np.ndarray) -> None:
        self._episode_ids = np.asarray(episode_ids, dtype=np.int64)
        lengths = np.asarray(episode_lengths, dtype=np.int64)
        if len(self._episode_ids) != len(lengths) or np.any(lengths <= 0):
            raise ValueError("Episode ids and positive lengths must have matching shapes")
        self._ends = np.cumsum(lengths, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._ends[-1]) if len(self._ends) else 0

    def __getitem__(self, index: int | slice) -> tuple[int, int] | list[tuple[int, int]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_pos = int(np.searchsorted(self._ends, index, side="right"))
        episode_start = 0 if episode_pos == 0 else int(self._ends[episode_pos - 1])
        return int(self._episode_ids[episode_pos]), int(index - episode_start)


def _vector_field_metadata(original_key: str) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "start": span.start,
            "end": span.stop,
            "absolute": True,
            "dtype": "float32",
            "original_key": original_key,
        }
        for name, span in _VECTOR_SLICES.items()
    }


def _build_modality_metadata() -> LeRobotModalityMetadata:
    return LeRobotModalityMetadata.model_validate(
        {
            "state": _vector_field_metadata("observation.state"),
            "action": _vector_field_metadata("action"),
            "video": {name: {"original_key": original} for name, original in _VIDEO_KEYS.items()},
            "annotation": {
                "human.action.task_description": {"original_key": "task_index"},
            },
        }
    )


def _get_video_shape(info: dict, original_key: str) -> tuple[int, int, int, float]:
    feature = info["features"][original_key]
    if feature.get("dtype") != "video":
        raise ValueError(f"Expected {original_key} to have dtype='video', got {feature.get('dtype')!r}")
    shape = tuple(int(value) for value in feature.get("shape", []))
    if len(shape) != 3:
        raise ValueError(f"Expected {original_key} to have a three-dimensional video shape, got {shape}")
    video_info = feature.get("info") or feature.get("video_info") or {}
    metadata_dims = {
        "height": video_info.get("video.height"),
        "width": video_info.get("video.width"),
        "channels": video_info.get("video.channels"),
    }
    names = feature.get("names")
    if names is None:
        metadata_hwc = tuple(
            int(metadata_dims[key]) for key in ("height", "width", "channels") if metadata_dims[key] is not None
        )
        names = (
            ["height", "width", "channels"]
            if len(metadata_hwc) == 3 and shape == metadata_hwc
            else [
                "channels",
                "height",
                "width",
            ]
        )
    names = [str(name).lower() for name in names]
    if len(names) != len(shape) or len(set(names)) != len(names):
        raise ValueError(f"Invalid {original_key} dimension names {names!r} for shape {shape}")
    named_shape = dict(zip(names, shape, strict=True))
    aliases = {
        "height": ("height",),
        "width": ("width",),
        "channels": ("channels", "channel", "rgb"),
    }
    dimensions = {}
    for dimension, candidates in aliases.items():
        named = next((named_shape[name] for name in candidates if name in named_shape), None)
        metadata = metadata_dims[dimension]
        metadata = int(metadata) if metadata is not None else None
        if named is None and metadata is None:
            raise ValueError(f"{original_key} does not describe its {dimension} dimension: {feature}")
        if named is not None and metadata is not None and named != metadata:
            raise ValueError(
                f"{original_key} {dimension} disagrees between shape/names ({named}) and metadata ({metadata})"
            )
        dimensions[dimension] = named if named is not None else metadata

    height = dimensions["height"]
    width = dimensions["width"]
    channels = dimensions["channels"]
    fps = float(video_info.get("video.fps", info["fps"]))
    return width, height, channels, fps


def _extract_global_stats(raw_stats: dict, modality: str, span: slice) -> dict[str, list[float]]:
    try:
        source = raw_stats[modality]["default"]
    except KeyError as exc:
        raise KeyError(f"FastWAM dataset_stats.json is missing {modality}.default") from exc

    result = {}
    for stat_name in _STAT_NAMES:
        source_name = f"global_{stat_name}"
        if source_name not in source:
            raise KeyError(f"FastWAM dataset_stats.json is missing {modality}.default.{source_name}")
        values = np.asarray(source[source_name], dtype=np.float64)
        if values.shape != (14,):
            raise ValueError(f"Expected {source_name} to have shape (14,), got {values.shape}")
        result[stat_name] = values[span].tolist()
    return result


def build_fastwam_dataset_metadata(
    dataset_root: Path,
    embodiment_tag: EmbodimentTag,
    stats_path: Path | None = None,
    expected_fps: float = 50.0,
) -> DatasetMetadata:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"FastWAM RoboTwin info.json not found: {info_path}")
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)

    actual_fps = float(info["fps"])
    if not np.isclose(actual_fps, expected_fps):
        raise ValueError(f"Expected FastWAM RoboTwin data at {expected_fps:g} FPS, got {actual_fps:g}")

    stats_path = stats_path or dataset_root / "dataset_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"FastWAM dataset_stats.json not found: {stats_path}")
    with stats_path.open("r", encoding="utf-8") as handle:
        raw_stats = json.load(handle)

    statistics = {
        modality: {name: _extract_global_stats(raw_stats, modality, span) for name, span in _VECTOR_SLICES.items()}
        for modality in ("state", "action")
    }
    state_action_modalities = {
        name: {
            "absolute": True,
            "rotation_type": None,
            "shape": [span.stop - span.start],
            "continuous": True,
        }
        for name, span in _VECTOR_SLICES.items()
    }
    video_modalities = {}
    for name, original_key in _VIDEO_KEYS.items():
        width, height, channels, fps = _get_video_shape(info, original_key)
        if not np.isclose(fps, expected_fps):
            raise ValueError(
                f"Expected {original_key} at the same {expected_fps:g} FPS as state/action, got {fps:g}"
            )
        video_modalities[name] = {
            "resolution": [width, height],
            "channels": channels,
            "fps": fps,
        }

    return DatasetMetadata.model_validate(
        {
            "statistics": statistics,
            "modalities": {
                "video": video_modalities,
                "state": state_action_modalities,
                "action": state_action_modalities,
            },
            "embodiment_tag": embodiment_tag,
        }
    )


class FastWAMRobotWinDataset(LeRobotSingleDataset):
    """Read FastWAM's release using its checkpoint-compatible sample ABI."""

    def __init__(self, *args, **kwargs) -> None:
        data_cfg = kwargs.get("data_cfg")
        self._fastwam_stats_path: Path | None = None
        self._fastwam_expected_fps = 50.0
        self._fastwam_val_fraction = 0.0
        self._fastwam_split = "all"
        self._fastwam_split_seed = 42
        self._fastwam_domain = "all"
        self._fastwam_wam_targets = False
        self._fastwam_wam_target = "composite"
        self._fastwam_future_stride = 32
        # Independent opt-in contract for Physically-Aligned Action--World
        # Co-Flow.  Old WAM keeps its existing [t,t+stride] fields unchanged.
        self._fastwam_action_world_coflow_targets = False
        if data_cfg is not None:
            configured_stats = data_cfg.get("fastwam_dataset_stats_path")
            if configured_stats:
                self._fastwam_stats_path = Path(str(configured_stats))
            self._fastwam_expected_fps = float(data_cfg.get("fastwam_expected_fps", 50.0))
            self._fastwam_val_fraction = float(data_cfg.get("fastwam_val_fraction", 0.0))
            self._fastwam_split = str(data_cfg.get("fastwam_split", "all")).lower()
            self._fastwam_split_seed = int(data_cfg.get("fastwam_split_seed", 42))
            self._fastwam_domain = str(data_cfg.get("fastwam_domain", "all")).lower()
            self._fastwam_wam_targets = bool(data_cfg.get("fastwam_wam_targets", False))
            self._fastwam_wam_target = str(data_cfg.get("fastwam_wam_target", "composite")).lower()
            self._fastwam_future_stride = int(data_cfg.get("fastwam_future_stride", 32))
            self._fastwam_action_world_coflow_targets = bool(
                data_cfg.get("fastwam_action_world_coflow_targets", False)
            )
            if "fastwam_coflow_future_strides" in data_cfg:
                raise ValueError(
                    "fastwam_coflow_future_strides was removed; single-bridge Co-Flow "
                    "always loads exactly the t+16 target"
                )
        if not 0.0 <= self._fastwam_val_fraction < 1.0:
            raise ValueError(f"fastwam_val_fraction must be in [0,1), got {self._fastwam_val_fraction}")
        if self._fastwam_split not in {"all", "train", "val", "validation"}:
            raise ValueError(f"Unsupported fastwam_split={self._fastwam_split!r}")
        if self._fastwam_domain not in {"all", "clean", "random", "randomized"}:
            raise ValueError(f"Unsupported fastwam_domain={self._fastwam_domain!r}")
        # stride=0 is the current-frame DINO reconstruction ablation: image_1 == image_0.
        # Negative strides are invalid.
        if self._fastwam_future_stride < 0:
            raise ValueError(f"fastwam_future_stride must be >= 0, got {self._fastwam_future_stride}")
        if self._fastwam_wam_targets and self._fastwam_wam_target != "composite":
            raise ValueError(
                f"FastWAM WAM only supports fastwam_wam_target='composite', got {self._fastwam_wam_target!r}"
            )
        if self._fastwam_wam_targets and self._fastwam_action_world_coflow_targets:
            raise ValueError(
                "fastwam_wam_targets and fastwam_action_world_coflow_targets are independent sample ABIs; "
                "enable exactly one"
            )
        super().__init__(*args, **kwargs)
        if self._fastwam_action_world_coflow_targets:
            action_key = self.modality_keys["action"][0]
            action_offsets = np.asarray(self.delta_indices[action_key], dtype=np.int64)
            if tuple(action_offsets.tolist()) != tuple(range(16)):
                raise ValueError(
                    "single-bridge Co-Flow requires one contiguous H16 action chunk; "
                    f"got offsets={action_offsets.tolist()}"
                )
        self._trajectory_index_by_id = {
            int(trajectory_id): index for index, trajectory_id in enumerate(self.trajectory_ids)
        }

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        stats_path = self._fastwam_stats_path
        if stats_path is not None and not stats_path.is_absolute():
            stats_path = self.dataset_path / stats_path
        return build_fastwam_dataset_metadata(
            self.dataset_path,
            embodiment_tag,
            stats_path=stats_path,
            expected_fps=self._fastwam_expected_fps,
        )

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        return _build_modality_metadata()

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        episodes_path = self.dataset_path / "meta" / "episodes.jsonl"
        with episodes_path.open("r", encoding="utf-8") as handle:
            episodes = [json.loads(line) for line in handle if line.strip()]

        wanted_domain = "randomized" if self._fastwam_domain == "random" else self._fastwam_domain
        if wanted_domain != "all":
            classified = [(episode, _episode_domain(episode)) for episode in episodes]
            unknown = [episode.get("episode_index") for episode, domain in classified if domain is None]
            if unknown:
                raise ValueError(
                    "fastwam_domain filtering requires Clean/Randomized provenance in meta/episodes.jsonl "
                    f"(for example raw_file_name); {len(unknown)} episodes are unclassified, first={unknown[:5]}"
                )
            episodes = [episode for episode, domain in classified if domain == wanted_domain]
            expected = self.data_cfg.get("fastwam_expected_domain_episodes") if self.data_cfg is not None else None
            if expected is not None and len(episodes) != int(expected):
                raise ValueError(
                    f"fastwam_domain={wanted_domain} selected {len(episodes)} episodes, expected {int(expected)}"
                )

        if not episodes:
            raise ValueError(f"No FastWAM episodes remain after fastwam_domain={self._fastwam_domain}")

        if self._fastwam_split != "all" and self._fastwam_val_fraction > 0:
            split_index = int(len(episodes) * (1.0 - self._fastwam_val_fraction))
            order = list(range(len(episodes)))
            rng = np.random.default_rng(self._fastwam_split_seed)
            rng.shuffle(order)
            chosen = order[:split_index] if self._fastwam_split == "train" else order[split_index:]
            episodes = [episodes[index] for index in chosen]

        trajectory_ids = np.asarray([int(episode["episode_index"]) for episode in episodes], dtype=np.int64)
        trajectory_lengths = np.asarray([int(episode["length"]) for episode in episodes], dtype=np.int64)
        return trajectory_ids, trajectory_lengths

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        indices = super()._get_delta_indices()
        if self._fastwam_wam_targets:
            future_indices = np.asarray([0, self._fastwam_future_stride], dtype=np.int64)
            for video_key in self.modality_keys["video"]:
                indices[video_key] = future_indices.copy()
        elif self._fastwam_action_world_coflow_targets:
            # All three camera streams and the action chunk use the release's
            # same 50 Hz integer index.  Base video loading clamps
            # out-of-episode indices; the explicit validity masks below make
            # those padded targets loss-inert.
            future_indices = np.asarray([0, 16], dtype=np.int64)
            for video_key in self.modality_keys["video"]:
                indices[video_key] = future_indices.copy()
        return indices

    def _get_all_steps(self) -> LazyEpisodeSteps:
        return LazyEpisodeSteps(self.trajectory_ids, self.trajectory_lengths)

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = super().get_step_data(trajectory_id, base_index)
        trajectory_index = self.get_trajectory_index(int(trajectory_id))
        trajectory_length = int(self.trajectory_lengths[trajectory_index])
        action_key = self.modality_keys["action"][0]
        action_offsets = np.asarray(self.delta_indices[action_key], dtype=np.int64)
        data["action_is_pad"] = base_index + action_offsets >= trajectory_length
        if self._fastwam_wam_targets:
            data["future_valid"] = np.float32(base_index + self._fastwam_future_stride < trajectory_length)
        elif self._fastwam_action_world_coflow_targets:
            data["future_valid_16"] = np.float32(base_index + 16 < trajectory_length)
        return data

    @staticmethod
    def _as_numpy(value) -> np.ndarray:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _pack_sample(self, data: dict) -> dict:
        current_views = [data[key][0] for key in self.modality_keys["video"]]
        composite = build_robotwin_composite(current_views)
        action = np.concatenate([self._as_numpy(data[key]) for key in self.modality_keys["action"]], axis=1).astype(
            np.float32
        )
        sample = {
            "action": action,
            "action_is_pad": np.asarray(data["action_is_pad"], dtype=np.bool_),
            "image": [composite],
            "lang": data[self.modality_keys["language"][0]][0],
            "robot_tag": self.tag,
        }

        if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
            sample["state"] = np.concatenate(
                [self._as_numpy(data[key]) for key in self.modality_keys.get("state", [])], axis=1
            ).astype(np.float32)

        if self._fastwam_wam_targets:
            future_views = [data[key][1] for key in self.modality_keys["video"]]
            future_composite = build_robotwin_composite(future_views)
            sample.update(
                {
                    "image_0": [composite],
                    "image_1": [future_composite],
                    "future_valid": np.float32(data["future_valid"]),
                    "dino_target_view_keys": [_COMPOSITE_VIEW_KEY],
                    "dino_view_keys": [_COMPOSITE_VIEW_KEY],
                }
            )
        elif self._fastwam_action_world_coflow_targets:
            future_views = [data[key][1] for key in self.modality_keys["video"]]
            sample.update(
                {
                    "image_0": [composite],
                    "image_16": [build_robotwin_composite(future_views)],
                    "future_valid_16": np.float32(data["future_valid_16"]),
                    "coflow_view_keys": [_COMPOSITE_VIEW_KEY],
                }
            )
        return sample

    def save_dataset_statistics(
        self,
        save_path: Path | str,
        format: str = "json",  # noqa: A002 - preserved public dataset API
    ) -> None:
        """Save this direct dataset's already-global FastWAM statistics."""

        if format.lower() != "json":
            raise ValueError(f"Unsupported statistics format: {format}")

        def combine(modality: str) -> dict[str, list]:
            stats = getattr(self.metadata.statistics, modality)
            keys = [key.split(".", 1)[1] for key in self.modality_keys[modality]]
            result = {}
            for stat_name in _STAT_NAMES:
                result[stat_name] = np.concatenate(
                    [np.asarray(getattr(stats[key], stat_name), dtype=np.float64) for key in keys]
                ).tolist()
            return result

        action_stats = combine("action")
        action_stats["mask"] = [True] * sum(span.stop - span.start for span in _VECTOR_SLICES.values())
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
        print(f"FastWAM dataset statistics saved to: {path}")

    def get_trajectory_index(self, trajectory_id: int) -> int:
        try:
            return self._trajectory_index_by_id[int(trajectory_id)]
        except KeyError as exc:
            raise ValueError(f"Unknown trajectory id: {trajectory_id}") from exc

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        trajectory_id = int(trajectory_id)
        data = super().get_trajectory_data(trajectory_id)
        self.curr_traj_id = trajectory_id
        self.curr_traj_data = data
        return data
