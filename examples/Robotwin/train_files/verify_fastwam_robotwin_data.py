"""Validate FastWAM's global RoboTwin release before a distributed launch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _video_feature_dimensions(feature: dict) -> tuple[int, int, int]:
    """Read ``(height, width, channels)`` from CHW or HWC LeRobot metadata."""

    if feature.get("dtype") != "video":
        raise ValueError(f"Expected dtype='video', got {feature.get('dtype')!r}")
    shape = tuple(int(value) for value in feature.get("shape", []))
    if len(shape) != 3:
        raise ValueError(f"Expected a three-dimensional video shape, got {shape}")

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
        if len(metadata_hwc) == 3 and shape == metadata_hwc:
            names = ["height", "width", "channels"]
        else:
            names = ["channels", "height", "width"]
    names = [str(name).lower() for name in names]
    if len(names) != len(shape) or len(set(names)) != len(names):
        raise ValueError(f"Invalid video dimension names {names!r} for shape {shape}")
    named_shape = dict(zip(names, shape, strict=True))

    aliases = {
        "height": ("height",),
        "width": ("width",),
        # LeRobot v2.1 uses ``rgb`` for the three-channel axis in HWC metadata.
        "channels": ("channels", "channel", "rgb"),
    }
    dimensions = {}
    for dimension, candidates in aliases.items():
        named = next((named_shape[name] for name in candidates if name in named_shape), None)
        metadata = metadata_dims[dimension]
        metadata = int(metadata) if metadata is not None else None
        if named is None and metadata is None:
            raise ValueError(f"Video feature does not describe its {dimension} dimension: {feature}")
        if named is not None and metadata is not None and named != metadata:
            raise ValueError(
                f"Video feature {dimension} disagrees between shape/names ({named}) and metadata ({metadata}): "
                f"{feature}"
            )
        dimensions[dimension] = named if named is not None else metadata
    return dimensions["height"], dimensions["width"], dimensions["channels"]


def _episode_domain(episode: dict) -> str | None:
    keys = ("raw_file_name", "source_path", "source_file", "dataset_path", "relative_path", "setting", "domain")
    value = "/".join(str(episode[key]) for key in keys if episode.get(key) is not None)
    value = "/" + value.replace("\\", "/").lower().strip("/") + "/"
    randomized = any(marker in value for marker in ("/randomized/", "/random/", "_randomized", "-randomized"))
    clean = any(marker in value for marker in ("/clean/", "_clean", "-clean"))
    if randomized and not clean:
        return "randomized"
    if clean and not randomized:
        return "clean"
    return None


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_tasks_jsonl(path: Path, expected_total: int) -> int:
    """Validate the LeRobot task table without retaining all rows in memory.

    ``total_tasks`` counts rows in LeRobot's task table.  It is not the number
    of RoboTwin benchmark task classes and may be much larger than 50 when the
    converted release stores many distinct instruction entries.
    """

    seen = np.zeros(int(expected_total), dtype=np.bool_)
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            task = json.loads(line)
            task_index = int(task["task_index"])
            if not 0 <= task_index < expected_total:
                raise ValueError(
                    f"FastWAM tasks.jsonl row {line_number} has task_index={task_index}, "
                    f"outside [0, {expected_total})"
                )
            if seen[task_index]:
                raise ValueError(f"FastWAM tasks.jsonl contains duplicate task_index={task_index} at row {line_number}")
            seen[task_index] = True
            count += 1
    return count


def verify_fastwam_robotwin_data(
    data_root: Path,
    stats_path: Path,
    expected_fps: float = 50.0,
    expected_episodes: int = 27_500,
    expected_frames: int = 6_075_103,
    expected_tasks: int = 0,
) -> dict:
    info_path = data_root / "meta" / "info.json"
    episodes_path = data_root / "meta" / "episodes.jsonl"
    tasks_path = data_root / "meta" / "tasks.jsonl"
    for path in (info_path, episodes_path, tasks_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    episodes = _read_jsonl(episodes_path)
    task_count = _validate_tasks_jsonl(tasks_path, int(info["total_tasks"]))

    if str(info.get("codebase_version")) != "v2.1":
        raise ValueError(f"Expected LeRobot codebase_version v2.1, got {info.get('codebase_version')!r}")
    if not np.isclose(float(info["fps"]), expected_fps):
        raise ValueError(f"Expected {expected_fps:g} FPS, info.json reports {info['fps']}")
    if expected_episodes and int(info["total_episodes"]) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes, info.json reports {info['total_episodes']}")
    if expected_frames and int(info["total_frames"]) != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, info.json reports {info['total_frames']}")
    if expected_tasks and int(info["total_tasks"]) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} tasks, info.json reports {info['total_tasks']}")
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(f"episodes.jsonl has {len(episodes)} rows, expected {info['total_episodes']}")
    if task_count != int(info["total_tasks"]):
        raise ValueError(f"tasks.jsonl has {task_count} rows, expected {info['total_tasks']}")
    episode_ids = [int(episode["episode_index"]) for episode in episodes]
    if episode_ids != list(range(len(episodes))):
        raise ValueError("FastWAM episodes.jsonl must contain contiguous episode_index values starting at zero")
    episode_frames = sum(int(episode["length"]) for episode in episodes)
    if episode_frames != int(info["total_frames"]):
        raise ValueError(f"Episode lengths sum to {episode_frames}, expected {info['total_frames']}")

    features = info["features"]
    for key in ("observation.state", "action"):
        if features[key]["shape"] != [14]:
            raise ValueError(f"Expected info.json feature {key} to have shape [14], got {features[key]['shape']}")
    for key in _VIDEO_KEYS:
        if key not in features:
            raise ValueError(f"info.json is missing required video feature {key}")
        feature = features[key]
        try:
            actual_shape = _video_feature_dimensions(feature)
        except ValueError as exc:
            raise ValueError(f"Invalid info.json feature {key}: {exc}") from exc
        if actual_shape != (480, 640, 3):
            raise ValueError(f"Expected info.json feature {key} to describe 480x640x3 RGB, got {feature}")
        video_info = feature.get("info") or feature.get("video_info") or {}
        feature_fps = float(video_info.get("video.fps", info["fps"]))
        if not np.isclose(feature_fps, expected_fps):
            raise ValueError(f"Expected {key} video.fps={expected_fps:g}, got {feature_fps:g}")

    with stats_path.open("r", encoding="utf-8") as handle:
        raw_stats = json.load(handle)
    for modality in ("state", "action"):
        try:
            modality_stats = raw_stats[modality]["default"]
        except KeyError as exc:
            raise ValueError(f"dataset_stats.json is missing {modality}.default") from exc
        for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
            values = np.asarray(modality_stats.get(f"global_{stat_name}"), dtype=np.float64)
            if values.shape != (14,) or not np.isfinite(values).all():
                raise ValueError(
                    f"Expected finite {modality}.default.global_{stat_name} with shape (14,), got {values.shape}"
                )

    sample_positions = sorted({0, len(episodes) // 2, len(episodes) - 1})
    for position in sample_positions:
        episode = episodes[position]
        episode_index = int(episode["episode_index"])
        episode_chunk = episode_index // int(info["chunks_size"])
        format_args = {"episode_index": episode_index, "episode_chunk": episode_chunk}
        parquet_path = data_root / info["data_path"].format(**format_args)
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)
        frame = pd.read_parquet(
            parquet_path,
            columns=[
                "observation.state",
                "action",
                "task_index",
                "timestamp",
                "episode_index",
                "frame_index",
                "index",
            ],
        )
        if len(frame) != int(episode["length"]):
            raise ValueError(f"{parquet_path} has {len(frame)} rows, expected {episode['length']}")
        for key in ("observation.state", "action"):
            if np.asarray(frame[key].iloc[0]).shape != (14,):
                raise ValueError(f"{parquet_path}:{key} is not 14-D")
        task_indices = frame["task_index"].to_numpy(dtype=np.int64, copy=False)
        if len(task_indices) and (task_indices.min() < 0 or task_indices.max() >= task_count):
            raise ValueError(
                f"{parquet_path}: task_index range [{task_indices.min()}, {task_indices.max()}] "
                f"falls outside tasks.jsonl with {task_count} rows"
            )
        frame_indices = frame["frame_index"].to_numpy(dtype=np.int64, copy=False)
        expected_frame_indices = np.arange(len(frame), dtype=np.int64)
        if not np.array_equal(frame_indices, expected_frame_indices):
            raise ValueError(f"{parquet_path}: frame_index is not contiguous [0, episode_length)")
        episode_indices = frame["episode_index"].to_numpy(dtype=np.int64, copy=False)
        if not bool((episode_indices == episode_index).all()):
            raise ValueError(f"{parquet_path}: episode_index column disagrees with episode_{episode_index:06d}")
        expected_timestamps = expected_frame_indices.astype(np.float64) / float(expected_fps)
        timestamps = frame["timestamp"].to_numpy(dtype=np.float64, copy=False)
        if not np.allclose(timestamps, expected_timestamps, rtol=0.0, atol=1.0e-4):
            raise ValueError(
                f"{parquet_path}: timestamps are not frame_index/{expected_fps:g}; "
                f"max_error={float(np.max(np.abs(timestamps - expected_timestamps))):.6g}"
            )
        global_start = sum(int(item["length"]) for item in episodes[:position])
        expected_global_indices = global_start + expected_frame_indices
        global_indices = frame["index"].to_numpy(dtype=np.int64, copy=False)
        if not np.array_equal(global_indices, expected_global_indices):
            raise ValueError(f"{parquet_path}: global index column is inconsistent with episode lengths/order")

        for video_key in _VIDEO_KEYS:
            video_path = data_root / info["video_path"].format(video_key=video_key, **format_args)
            if not video_path.is_file():
                raise FileNotFoundError(video_path)

    domains = [_episode_domain(episode) for episode in episodes]
    domain_counts = {
        "clean": sum(domain == "clean" for domain in domains),
        "randomized": sum(domain == "randomized" for domain in domains),
        "unknown": sum(domain is None for domain in domains),
    }

    summary = {
        "data_root": str(data_root),
        "stats_path": str(stats_path),
        "codebase_version": str(info["codebase_version"]),
        "fps": float(info["fps"]),
        "episodes": len(episodes),
        "frames": episode_frames,
        "tasks": task_count,
        "sampled_episodes": [int(episodes[position]["episode_index"]) for position in sample_positions],
        "domain_counts": domain_counts,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=27_500)
    parser.add_argument("--expected-frames", type=int, default=6_075_103)
    parser.add_argument(
        "--expected-tasks",
        type=int,
        default=0,
        help="Optional exact LeRobot tasks.jsonl row count; 0 checks metadata consistency without assuming 50 classes",
    )
    args = parser.parse_args()

    with args.config_yaml.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_config = config["datasets"]["vla_data"]
    action_horizon = int(config["framework"]["action_model"]["action_horizon"])
    data_mix = data_config.get("data_mix")
    valid_layouts = {
        (32, "robotwin_fastwam"),
        (16, "robotwin_fastwam_h16"),
    }
    if (action_horizon, data_mix) not in valid_layouts:
        raise ValueError(
            "FastWAM config must use (H32,robotwin_fastwam) or "
            f"(H16,robotwin_fastwam_h16), got H={action_horizon}, data_mix={data_mix!r}"
        )
    expected_config = {
        "fastwam_expected_fps": 50,
        "fastwam_val_fraction": 0.01,
        "fastwam_split": "train",
        "fastwam_split_seed": 42,
        "fastwam_direct_frame_sampling": True,
        "action_mode": "abs",
        "obs_image_size": [320, 384],
        "video_backend": "pyav",
        "balance_dataset_weights": False,
        "balance_trajectory_weights": False,
        "load_all_data_for_training": True,
    }
    for key, expected in expected_config.items():
        if data_config.get(key) != expected:
            raise ValueError(f"Expected datasets.vla_data.{key}={expected!r}, got {data_config.get(key)!r}")
    if "include_state" not in data_config or not isinstance(data_config["include_state"], bool):
        raise ValueError(
            "datasets.vla_data.include_state must be an explicit YAML boolean so the checkpoint records its "
            "inference ABI"
        )
    if bool(data_config.get("fastwam_action_world_coflow_targets", False)):
        if (action_horizon, data_mix) != (16, "robotwin_fastwam_h16"):
            raise ValueError(
                "Action--World Co-Flow is one H16 bridge and requires "
                f"data_mix=robotwin_fastwam_h16, got H={action_horizon}, data_mix={data_mix!r}"
            )
        if "fastwam_coflow_future_strides" in data_config:
            raise ValueError(
                "fastwam_coflow_future_strides was removed; Co-Flow always uses t+16"
            )

    data_root = Path(data_config["data_root_dir"])
    stats_path = Path(data_config.get("fastwam_dataset_stats_path", data_root / "dataset_stats.json"))
    summary = verify_fastwam_robotwin_data(
        data_root,
        stats_path,
        expected_fps=float(data_config.get("fastwam_expected_fps", 50)),
        expected_episodes=int(data_config.get("fastwam_expected_episodes", args.expected_episodes)),
        expected_frames=int(data_config.get("fastwam_expected_frames", args.expected_frames)),
        expected_tasks=int(data_config.get("fastwam_expected_tasks", args.expected_tasks)),
    )
    domain = str(data_config.get("fastwam_domain", "all")).lower()
    if domain in {"clean", "random", "randomized"}:
        wanted = "randomized" if domain == "random" else domain
        counts = summary["domain_counts"]
        if counts["unknown"]:
            raise ValueError(
                f"fastwam_domain={wanted} cannot be verified: {counts['unknown']} episodes lack source provenance"
            )
        expected_domain = data_config.get("fastwam_expected_domain_episodes")
        if expected_domain is not None and counts[wanted] != int(expected_domain):
            raise ValueError(f"fastwam_domain={wanted} has {counts[wanted]} episodes, expected {int(expected_domain)}")
    summary["include_state"] = data_config["include_state"]
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
