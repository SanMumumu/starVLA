"""Dependency-light configuration and label helpers for planner text memory."""

from __future__ import annotations

from typing import Any

import numpy as np


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        array = np.asarray(value)
        if bool(
            array.ndim == 0
            and array.dtype.kind == "f"
            and np.isnan(value)
        ):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def text_history_frame_offsets(config) -> list[int]:
    """Return opt-in planner-history offsets without changing no-text video I/O."""

    history = _cfg_get(config, "history", {})
    if not bool(_cfg_get(history, "enabled", False)):
        return []
    if not bool(_cfg_get(config, "enabled", False)):
        raise ValueError(
            "text_annotations.history.enabled=true requires "
            "text_annotations.enabled=true"
        )
    raw_offsets = _cfg_get(history, "frame_offsets", [])
    if isinstance(raw_offsets, (int, float)):
        raw_offsets = [raw_offsets]
    offsets = [int(offset) for offset in raw_offsets]
    if not offsets:
        raise ValueError(
            "text_annotations.history.enabled=true requires non-empty frame_offsets"
        )
    if any(offset >= 0 for offset in offsets):
        raise ValueError(
            "text_annotations.history.frame_offsets must contain only past "
            f"(negative) offsets, got {offsets}"
        )
    if len(set(offsets)) != len(offsets):
        raise ValueError(
            f"text_annotations.history.frame_offsets contains duplicates: {offsets}"
        )
    if offsets != sorted(offsets):
        raise ValueError(
            "text_annotations.history.frame_offsets must be ordered oldest-to-newest, "
            f"got {offsets}"
        )
    return offsets


def text_history_memory_from_trajectory(
    trajectory,
    base_index: int,
    config,
) -> dict[str, str]:
    """Read the Finished Task List from the preceding in-episode refresh."""

    history = _cfg_get(config, "history", {})
    if not bool(_cfg_get(history, "enabled", False)):
        return {}
    if not bool(_cfg_get(config, "enabled", False)):
        raise ValueError(
            "text_annotations.history.enabled=true requires "
            "text_annotations.enabled=true"
        )
    memory_offset = int(_cfg_get(history, "memory_offset", 0))
    if memory_offset >= 0:
        raise ValueError(
            "text_annotations.history.memory_offset must be negative, got "
            f"{memory_offset}"
        )
    output_field = str(
        _cfg_get(history, "finished_task_list_field", "finished_task_list")
    )
    fields = dict(_cfg_get(config, "fields", {}))
    default_source = fields.get("completed_subtask_text", "complete_text")
    source_field = str(
        _cfg_get(history, "finished_task_list_source_field", default_source)
    )
    memory_index = int(
        np.clip(int(base_index) + memory_offset, 0, len(trajectory) - 1)
    )
    memory = _text_value(trajectory.iloc[memory_index].get(source_field, ""))
    if not memory:
        memory = str(
            _cfg_get(history, "empty_finished_task_list", "None")
        ).strip()
    if not memory:
        raise ValueError(
            "text history requires a non-empty empty_finished_task_list sentinel"
        )
    return {output_field: memory}


__all__ = [
    "text_history_frame_offsets",
    "text_history_memory_from_trajectory",
]
