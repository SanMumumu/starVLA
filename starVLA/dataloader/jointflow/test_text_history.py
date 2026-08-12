from __future__ import annotations

import pandas as pd
import pytest

from starVLA.dataloader.jointflow.text_history import (
    text_history_frame_offsets,
    text_history_memory_from_trajectory,
)


def test_text_history_is_strictly_opt_in() -> None:
    assert text_history_frame_offsets({"enabled": False}) == []
    assert text_history_frame_offsets(
        {
            "enabled": True,
            "history": {"enabled": False},
        }
    ) == []


def test_text_history_offsets_must_be_ordered_past_frames() -> None:
    config = {
        "enabled": True,
        "history": {
            "enabled": True,
            "frame_offsets": [-36, -24, -12],
        },
    }
    assert text_history_frame_offsets(config) == [-36, -24, -12]
    with pytest.raises(ValueError, match="text_annotations.enabled=true"):
        text_history_frame_offsets(
            {
                "enabled": False,
                "history": {
                    "enabled": True,
                    "frame_offsets": [-12],
                },
            }
        )
    with pytest.raises(ValueError, match="past"):
        text_history_frame_offsets(
            {
                "enabled": True,
                "history": {
                    "enabled": True,
                    "frame_offsets": [-12, 0],
                }
            }
        )
    with pytest.raises(ValueError, match="oldest-to-newest"):
        text_history_frame_offsets(
            {
                "enabled": True,
                "history": {
                    "enabled": True,
                    "frame_offsets": [-12, -24],
                }
            }
        )


def test_finished_task_memory_stays_inside_the_episode() -> None:
    trajectory = pd.DataFrame(
        {
            "complete_text": [
                "None",
                "Place A.",
                "Place A.",
                "Place A. Place B.",
            ]
        }
    )
    config = {
        "enabled": True,
        "fields": {
            "completed_subtask_text": "complete_text",
        },
        "history": {
            "enabled": True,
            "memory_offset": -2,
            "finished_task_list_field": "finished_task_list",
            "empty_finished_task_list": "None",
        },
    }

    assert text_history_memory_from_trajectory(
        trajectory,
        3,
        config,
    ) == {"finished_task_list": "Place A."}
    # A negative offset at episode start clamps to row zero, never to the
    # preceding episode.
    assert text_history_memory_from_trajectory(
        trajectory,
        0,
        config,
    ) == {"finished_task_list": "None"}
