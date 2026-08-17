"""Deployment-safe event-memory recorder for live RoboDojo rollouts.

The simulator client imports this module from the uploaded StarVLA package.
It deliberately has no dependency on ``Third_github``, RoboDojo datasets, or
training-only packages; live RGB frames and semantic state are supplied by the
repository-owned RoboDojo adapter.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

THIRD_VIEW_KEY = "video.cam_high"

# Pastel palette aligned with the reference annotation figure.
COLORS = {
    "bg": (252, 252, 253),
    "task_fill": (214, 239, 214),
    "task_label": (76, 145, 76),
    "timeline_label": (90, 90, 95),
    "decision_fill": (235, 235, 240),
    "decision_label": (110, 110, 120),
    "keep_fill": (178, 225, 190),
    "keep_text": (34, 110, 58),
    "update_fill": (235, 195, 200),
    "update_text": (125, 48, 58),
    "subtask_fill": (198, 220, 248),
    "subtask_label": (55, 105, 175),
    "memory_fill": (255, 228, 196),
    "memory_label": (175, 110, 55),
    "border": (170, 170, 178),
    "text": (25, 25, 30),
    "muted": (95, 95, 105),
}

PAD_X = 8
PAD_Y = 6


def _font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ("DejaVuSansMono.ttf", "LiberationMono-Regular.ttf")
        if mono
        else ("DejaVuSans.ttf", "LiberationSans-Regular.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = str(text or "").strip().split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    radius: int = 10,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _line_height(font) -> int:
    try:
        bbox = font.getbbox("Ag")
        return bbox[3] - bbox[1] + 4
    except AttributeError:
        return int(getattr(font, "size", 14)) + 5


def _draw_box_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: list[str],
    *,
    font,
    fill: tuple[int, int, int],
    max_lines: int,
) -> None:
    line_h = _line_height(font)
    x0, y0, x1, _y1 = box
    text_y = y0 + PAD_Y
    for line in lines[:max_lines]:
        draw.text((x0 + PAD_X, text_y), line, fill=fill, font=font)
        text_y += line_h


def _span_box_height(lines: list[str], font, *, max_lines: int, min_h: int) -> int:
    line_h = _line_height(font)
    used = min(len(lines), max_lines)
    return max(min_h, PAD_Y * 2 + used * line_h)


def _decision(record: dict[str, Any]) -> str:
    return str(record.get("pred_decision", "")).strip().upper()


def _update_segment_ranges(records: list[dict[str, Any]]) -> list[tuple[int, int]]:
    update_indices = [i for i, record in enumerate(records) if _decision(record) == "UPDATE"]
    if not update_indices:
        return [(0, len(records) - 1)]
    segments: list[tuple[int, int]] = []
    for seg_i, start_idx in enumerate(update_indices):
        end_idx = (
            update_indices[seg_i + 1] - 1
            if seg_i + 1 < len(update_indices)
            else len(records) - 1
        )
        segments.append((start_idx, end_idx))
    return segments


def effective_max_frames(
    records: list[dict[str, Any]],
    max_frames: int,
    *,
    keeps_per_update: int = 2,
) -> int:
    num_updates = sum(1 for record in records if _decision(record) == "UPDATE")
    target = num_updates * (1 + keeps_per_update)
    return min(len(records), max(max_frames, target))


def _evenly_pick(indices: list[int], count: int) -> list[int]:
    if count <= 0 or not indices:
        return []
    if count >= len(indices):
        return list(indices)
    picks = np.linspace(0, len(indices) - 1, num=count, dtype=int)
    return sorted({indices[int(i)] for i in picks})


def select_keyframe_indices(
    records: list[dict[str, Any]],
    *,
    max_frames: int = 9,
    keeps_per_update: int = 2,
) -> list[int]:
    """Return indices showing UPDATE boundaries and representative KEEP frames."""

    budget = effective_max_frames(records, max_frames, keeps_per_update=keeps_per_update)
    if len(records) <= budget:
        return list(range(len(records)))

    update_indices = [i for i, record in enumerate(records) if _decision(record) == "UPDATE"]
    if not update_indices:
        picks = np.linspace(0, len(records) - 1, num=budget, dtype=int)
        return [int(i) for i in picks]

    segments = _update_segment_ranges(records)
    selected: set[int] = {start_idx for start_idx, _end_idx in segments}

    keep_slots = budget - len(selected)
    if keep_slots <= 0:
        return sorted(selected)[:budget]

    segment_keep_pools: list[list[int]] = []
    for start_idx, end_idx in segments:
        pool = [
            i
            for i in range(start_idx + 1, end_idx + 1)
            if _decision(records[i]) == "KEEP"
        ]
        segment_keep_pools.append(pool)

    per_segment_quota = [
        min(len(pool), keeps_per_update) if pool else 0 for pool in segment_keep_pools
    ]
    quota_sum = sum(per_segment_quota)
    if quota_sum > keep_slots:
        per_segment_quota = [min(len(pool), 1) if pool else 0 for pool in segment_keep_pools]
        quota_sum = sum(per_segment_quota)

    extras = keep_slots - quota_sum
    if extras > 0:
        order = sorted(
            range(len(segment_keep_pools)),
            key=lambda i: len(segment_keep_pools[i]),
            reverse=True,
        )
        for seg_i in order:
            if extras <= 0:
                break
            room = len(segment_keep_pools[seg_i]) - per_segment_quota[seg_i]
            add = min(room, extras)
            per_segment_quota[seg_i] += add
            extras -= add

    for pool, quota in zip(segment_keep_pools, per_segment_quota):
        selected.update(_evenly_pick(pool, quota))

    if len(selected) < budget:
        remaining_keep = [
            i
            for i, record in enumerate(records)
            if i not in selected and _decision(record) == "KEEP"
        ]
        selected.update(_evenly_pick(remaining_keep, budget - len(selected)))

    if len(selected) > budget:
        update_set = set(update_indices)
        keep_selected = sorted(i for i in selected if i not in update_set)
        need_keep = max(0, budget - len(update_set))
        selected = update_set | set(_evenly_pick(keep_selected, need_keep))

    return sorted(selected)


def select_keyframes(
    records: list[dict[str, Any]],
    *,
    max_frames: int = 9,
    keeps_per_update: int = 2,
) -> list[dict[str, Any]]:
    """Pick keyframes that show UPDATE boundaries plus multiple KEEP frames between them."""

    return [
        records[index]
        for index in select_keyframe_indices(
            records,
            max_frames=max_frames,
            keeps_per_update=keeps_per_update,
        )
    ]


def get_third_view_image(dataset, trajectory_id: int, frame_idx: int) -> Image.Image:
    dataset.get_step_data(int(trajectory_id), int(frame_idx), decode_video=True)
    frames = np.asarray(dataset._last_video_frames[THIRD_VIEW_KEY])
    offsets = [int(offset) for offset in dataset.delta_indices[THIRD_VIEW_KEY]]
    position = offsets.index(0)
    arr = np.asarray(frames[position], dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _column_x(content_left: int, col_w: int, gap: int, index: int) -> int:
    return content_left + index * (col_w + gap)


def _group_spans(records: list[dict[str, Any]], key: str) -> list[tuple[int, int, str]]:
    if not records:
        return []
    spans: list[tuple[int, int, str]] = []
    start = 0
    value = str(records[0].get(key, ""))
    for idx in range(1, len(records)):
        current = str(records[idx].get(key, ""))
        if current != value:
            spans.append((start, idx - 1, value))
            start = idx
            value = current
    spans.append((start, len(records) - 1, value))
    return spans


def render_eventmem_figure(
    *,
    records: list[dict[str, Any]],
    images: list[Image.Image],
    instruction: str,
    task_name: str,
    episode: int,
    mode: str,
    checkpoint_label: str = "eventmem",
) -> Image.Image:
    if len(records) != len(images):
        raise ValueError(f"records ({len(records)}) and images ({len(images)}) length mismatch")

    n = len(records)
    label_w = 124
    margin = 16
    gap = 12
    col_w = 192
    thumb_h = 148
    row_gap = 10

    content_w = n * col_w + (n - 1) * gap
    canvas_w = margin * 2 + label_w + content_w

    font_label = _font(16, mono=True)
    font_title = _font(18, mono=True)
    font_body = _font(15, mono=True)
    font_small = _font(13, mono=True)
    font_badge = _font(15, mono=True)

    label_line_h = _line_height(font_label)
    title_line_h = _line_height(font_title)
    body_line_h = _line_height(font_body)

    title = instruction.strip() or task_name
    title_lines = _wrap(ImageDraw.Draw(Image.new("RGB", (1, 1))), title, font_title, content_w - PAD_X * 2)[:2]

    subtask_spans = _group_spans(records, "running_subtask_after")
    memory_spans = _group_spans(records, "running_memory_after")

    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    subtask_wrapped: list[list[str]] = []
    for span_start, span_end, subtask in subtask_spans:
        span_w = (span_end - span_start + 1) * col_w + (span_end - span_start) * gap - 4
        text = subtask if subtask and subtask != "None." else "(empty)"
        subtask_wrapped.append(_wrap(dummy, text, font_body, span_w - PAD_X * 2)[:4])
    memory_wrapped: list[list[str]] = []
    for span_start, span_end, memory in memory_spans:
        span_w = (span_end - span_start + 1) * col_w + (span_end - span_start) * gap - 4
        text = memory if memory and memory != "None." else "None."
        memory_wrapped.append(_wrap(dummy, text, font_body, span_w - PAD_X * 2)[:5])

    task_h = max(52, PAD_Y * 2 + len(title_lines) * title_line_h)
    timeline_h = thumb_h + 22
    decision_h = 40
    subtask_h = max(
        64,
        max(
            (_span_box_height(lines, font_body, max_lines=4, min_h=0) for lines in subtask_wrapped),
            default=64,
        ),
    )
    memory_h = max(
        72,
        max(
            (_span_box_height(lines, font_body, max_lines=5, min_h=0) for lines in memory_wrapped),
            default=72,
        ),
    )
    canvas_h = margin * 2 + task_h + timeline_h + decision_h + subtask_h + memory_h + row_gap * 4 + 18

    canvas = Image.new("RGB", (canvas_w, canvas_h), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)

    content_left = margin + label_w
    y = margin

    def draw_row_label(text: str, row_y: int, row_h: int, *, fill: tuple[int, int, int], text_color) -> None:
        box = (margin, row_y, margin + label_w - 8, row_y + row_h)
        _rounded_rect(draw, box, fill=fill, radius=8)
        lines = _wrap(draw, text, font_label, label_w - 16)
        text_y = row_y + max(PAD_Y, (row_h - len(lines) * label_line_h) // 2)
        for line in lines:
            draw.text((margin + PAD_X, text_y), line, fill=text_color, font=font_label)
            text_y += label_line_h

    # --- Task ---
    draw_row_label("Task", y, task_h, fill=COLORS["task_fill"], text_color=COLORS["task_label"])
    task_box = (content_left, y, content_left + content_w, y + task_h)
    _rounded_rect(draw, task_box, fill=COLORS["task_fill"], outline=COLORS["border"], radius=10)
    for i, line in enumerate(title_lines):
        draw.text(
            (content_left + PAD_X, y + PAD_Y + i * title_line_h),
            line,
            fill=COLORS["text"],
            font=font_title,
        )
    y += task_h + row_gap

    # --- Timeline ---
    draw_row_label("Video\nTimeline", y, timeline_h, fill=(245, 245, 247), text_color=COLORS["timeline_label"])
    for idx, (record, image) in enumerate(zip(records, images)):
        x = _column_x(content_left, col_w, gap, idx)
        thumb = image.copy()
        thumb.thumbnail((col_w, thumb_h), Image.BILINEAR)
        paste_x = x + (col_w - thumb.width) // 2
        paste_y = y + 2
        canvas.paste(thumb, (paste_x, paste_y))
        frame = int(record["frame"])
        tag = f"f{frame}"
        tag_w = draw.textlength(tag, font=font_small) + 10
        tag_box = (paste_x, paste_y + thumb.height - 20, paste_x + tag_w, paste_y + thumb.height - 2)
        draw.rectangle(tag_box, fill=(20, 20, 24))
        draw.text((tag_box[0] + 5, tag_box[1] + 1), tag, fill=(245, 245, 245), font=font_small)
    y += timeline_h + row_gap

    # --- Decision ---
    draw_row_label("Pred\nKEEP/\nUPDATE", y, decision_h, fill=COLORS["decision_fill"], text_color=COLORS["decision_label"])
    badge_pad_x = 12
    badge_pad_y = 5
    for idx, record in enumerate(records):
        decision = str(record.get("pred_decision", "")).upper()
        x = _column_x(content_left, col_w, gap, idx)
        fill = COLORS["update_fill"] if decision == "UPDATE" else COLORS["keep_fill"]
        text_color = COLORS["update_text"] if decision == "UPDATE" else COLORS["keep_text"]
        badge = decision.title()
        text_w = draw.textlength(badge, font=font_badge)
        box = (
            x + (col_w - text_w) // 2 - badge_pad_x,
            y + 4,
            x + (col_w + text_w) // 2 + badge_pad_x,
            y + decision_h - 4,
        )
        _rounded_rect(draw, box, fill=fill, outline=COLORS["border"], radius=8)
        draw.text((box[0] + badge_pad_x, box[1] + badge_pad_y), badge, fill=text_color, font=font_badge)
    y += decision_h + row_gap

    # --- Subtask ---
    draw_row_label("Subtask", y, subtask_h, fill=COLORS["subtask_fill"], text_color=COLORS["subtask_label"])
    for (span_start, span_end, _subtask), lines in zip(subtask_spans, subtask_wrapped):
        x0 = _column_x(content_left, col_w, gap, span_start)
        x1 = _column_x(content_left, col_w, gap, span_end) + col_w
        box = (x0 + 2, y + 2, x1 - 2, y + subtask_h - 2)
        _rounded_rect(draw, box, fill=COLORS["subtask_fill"], outline=COLORS["border"], radius=10)
        _draw_box_text(draw, box, lines, font=font_body, fill=COLORS["text"], max_lines=4)
    y += subtask_h + row_gap

    # --- Memory ---
    draw_row_label("Semantic\nMemory", y, memory_h, fill=COLORS["memory_fill"], text_color=COLORS["memory_label"])
    for (span_start, span_end, _memory), lines in zip(memory_spans, memory_wrapped):
        x0 = _column_x(content_left, col_w, gap, span_start)
        x1 = _column_x(content_left, col_w, gap, span_end) + col_w
        box = (x0 + 2, y + 2, x1 - 2, y + memory_h - 2)
        _rounded_rect(draw, box, fill=COLORS["memory_fill"], outline=COLORS["border"], radius=10)
        _draw_box_text(draw, box, lines, font=font_body, fill=COLORS["text"], max_lines=5)

    footer = f"{task_name} | ep={episode} | mode={mode} | ckpt={checkpoint_label}"
    draw.text((margin, y + memory_h + 6), footer, fill=COLORS["muted"], font=font_small)
    return canvas.crop((0, 0, canvas_w, y + memory_h + margin + 18))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


class RolloutEventMemoryRecorder:
    """Collect simulator replans and write one event-memory figure per episode.

    This recorder is deliberately dataset-free.  RoboDojo supplies the live
    third-person image, while the StarVLA adapter supplies the semantic state
    immediately after each KEEP/UPDATE decision.
    """

    def __init__(
        self,
        output_root: str | Path,
        *,
        task_name: str,
        checkpoint_label: str = "eventmem",
        num_envs: int = 1,
        max_frames: int = 9,
        mode: str = "simulator_rollout",
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.task_name = str(task_name)
        self.checkpoint_label = str(checkpoint_label)
        self.num_envs = int(num_envs)
        self.max_frames = int(max_frames)
        self.mode = str(mode)
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")
        if self.max_frames < 1:
            raise ValueError(f"max_frames must be positive, got {self.max_frames}")
        existing_episodes: list[int] = []
        for path in (self.output_root / self.task_name).glob(
            "episode_*_env*"
        ):
            try:
                existing_episodes.append(int(path.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
        self.group_index = (
            max(existing_episodes) // self.num_envs + 1
            if existing_episodes
            else 0
        )
        self.records_by_env: dict[int, list[dict[str, Any]]] = {}
        self.images_by_env: dict[int, list[Image.Image]] = {}
        self.instruction_by_env: dict[int, str] = {}

    def _episode_index(self, env_idx: int) -> int:
        return self.group_index * self.num_envs + int(env_idx)

    def _episode_dir(self, env_idx: int) -> Path:
        episode = self._episode_index(env_idx)
        return (
            self.output_root
            / self.task_name
            / f"episode_{episode:07d}_env{int(env_idx):02d}"
        )

    def capture(
        self,
        *,
        env_idx: int,
        frame: int,
        image: Image.Image | np.ndarray,
        instruction: str,
        decision: str,
        running_subtask_after: str,
        running_memory_after: str,
        memory_add: str | None = None,
        planner_text: str | None = None,
    ) -> None:
        env_idx = int(env_idx)
        decision = str(decision).strip().upper()
        if decision not in {"KEEP", "UPDATE"}:
            raise ValueError(f"decision must be KEEP or UPDATE, got {decision!r}")
        if isinstance(image, Image.Image):
            thumbnail = image.convert("RGB").copy()
        else:
            array = np.asarray(image, dtype=np.uint8)
            if array.ndim != 3 or array.shape[-1] not in (3, 4):
                raise ValueError(f"rollout image must be HWC RGB/RGBA, got {array.shape}")
            thumbnail = Image.fromarray(array[..., :3], mode="RGB")
        # The renderer displays 192x148 thumbnails.  Keeping only a 2x source
        # bounds memory for long vectorized episodes without reducing quality.
        thumbnail.thumbnail((384, 296), Image.BILINEAR)

        record = {
            "frame": int(frame),
            "pred_decision": decision,
            "memory_add": memory_add,
            "planner_text": planner_text,
            "running_subtask_after": str(running_subtask_after),
            "running_memory_after": str(running_memory_after),
        }
        self.records_by_env.setdefault(env_idx, []).append(record)
        self.images_by_env.setdefault(env_idx, []).append(thumbnail)
        self.instruction_by_env[env_idx] = str(instruction)
        self._write_records(env_idx, status="running", result=None)

    def _result_for_env(
        self,
        result: dict[str, Any] | None,
        env_idx: int,
    ) -> Any:
        if not isinstance(result, dict):
            return None
        values = result.get("success_by_env")
        if not isinstance(values, dict):
            return None
        return values.get(env_idx, values.get(str(env_idx)))

    def _write_records(
        self,
        env_idx: int,
        *,
        status: str,
        result: dict[str, Any] | None,
    ) -> Path:
        episode = self._episode_index(env_idx)
        payload = {
            "task_name": self.task_name,
            "episode": episode,
            "env_idx": int(env_idx),
            "mode": self.mode,
            "checkpoint": self.checkpoint_label,
            "instruction": self.instruction_by_env.get(env_idx, self.task_name),
            "status": status,
            "success": self._result_for_env(result, env_idx),
            "records": self.records_by_env.get(env_idx, []),
        }
        path = self._episode_dir(env_idx) / "records.json"
        _atomic_json(path, payload)
        return path

    def finish_episode_group(
        self,
        result: dict[str, Any] | None = None,
        *,
        status: str = "complete",
    ) -> list[Path]:
        """Render all active vector environments, then advance the group id."""

        written: list[Path] = []
        for env_idx in sorted(self.records_by_env):
            records = self.records_by_env[env_idx]
            images = self.images_by_env[env_idx]
            indices = select_keyframe_indices(records, max_frames=self.max_frames)
            selected_records = [records[index] for index in indices]
            selected_images = [images[index] for index in indices]
            episode = self._episode_index(env_idx)
            figure = render_eventmem_figure(
                records=selected_records,
                images=selected_images,
                instruction=self.instruction_by_env.get(env_idx, self.task_name),
                task_name=self.task_name,
                episode=episode,
                mode=self.mode,
                checkpoint_label=self.checkpoint_label,
            )
            figure_path = self._episode_dir(env_idx) / "eventmem.png"
            _atomic_png(figure_path, figure)
            self._write_records(env_idx, status=status, result=result)
            written.append(figure_path)

        if self.records_by_env:
            self.group_index += 1
        self.records_by_env.clear()
        self.images_by_env.clear()
        self.instruction_by_env.clear()
        return written

    def finish_interrupted_group(self) -> list[Path]:
        return self.finish_episode_group(status="interrupted")


__all__ = [
    "RolloutEventMemoryRecorder",
    "effective_max_frames",
    "render_eventmem_figure",
    "select_keyframe_indices",
    "select_keyframes",
]
