"""PR1: Offline DINOv3 feature precompute into each dataset directory.

复用:
- JointFlow DINOv3Backbone
- starVLA.dataloader.gr00t_lerobot registry + LeRobotSingleDataset helpers

说明:
默认输出采用 LingbotVA 风格，全部写到 <dataset_path>/latents/：
- latents/chunk-000/<original_video_key>/episode_000000_0_214.pth
- latents/dino_v3_index.json
- latents/dino_v3_stats.json
每个 episode/view 一个 .pth，训练时读取 raw DINO patch token 后再用 stats 做 norm。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import os
import queue
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP, EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.video import get_all_frames
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    DatasetModalities,
    DatasetStatistics,
    DatasetStatisticalValues,
    StateActionMetadata,
    VideoMetadata,
    LeRobotModalityMetadata,
)
from starVLA.jointflow.modules.dino_v3 import DINOv3Backbone
from starVLA.jointflow.data.mix_registry import resolve_data_mix


######### // code // ##########
# 中文注释：在线统计每个 DINO channel 的 mean/std。
# 输入批量 feature [M,N_v,384]，把 frame/view/token 都并入样本维。
class RunningChannelStats:
    def __init__(self, channels: int):
        self.count = 0
        self.sum = np.zeros(channels, dtype=np.float64)
        self.sumsq = np.zeros(channels, dtype=np.float64)

    def update(self, feats: np.ndarray) -> None:
        flat = feats.reshape(-1, feats.shape[-1]).astype(np.float64)
        self.count += flat.shape[0]
        self.sum += flat.sum(axis=0)
        self.sumsq += (flat * flat).sum(axis=0)

    def to_json(self) -> dict:
        mean = self.sum / max(self.count, 1)
        var = self.sumsq / max(self.count, 1) - mean * mean
        std = np.sqrt(np.maximum(var, 1e-12))
        return {"mean": mean.tolist(), "std": std.tolist(), "count": int(self.count)}
######### // code // ##########


######### // code // ##########
# 中文注释：autocast 精度选择。离线 latent 最终存成 bf16，所以用 bf16 跑 DINO
# 几乎不损失精度，却能在 H20 上把吞吐拉满（fp32 仅作兜底）。
_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


def _amp_ctx(amp_dtype):
    if amp_dtype is None or not torch.cuda.is_available():
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=amp_dtype)


def _enable_fast_matmul() -> None:
    # 中文注释：H20 上放开 TF32 / cudnn benchmark，纯推理无副作用。
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _atomic_torch_save(obj, out_path: Path) -> None:
    # 中文注释：先写临时文件再原子 rename，崩溃/抢占都不会留下半截 .pth，resume 才安全。
    tmp = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
    torch.save(obj, tmp)
    tmp.replace(out_path)


def _atomic_json_dump(obj, out_path: Path) -> None:
    tmp = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(out_path)


######### // code // ##########
# 中文注释：分片级进度 / ETA。以“帧 × 视角”为最小工作单位统计整 shard 的剩余时间，
# 每隔 print_every_sec 秒打印一次（被跳过的帧按已完成计入，避免 ETA 抖动）。
class ShardProgress:
    def __init__(self, total_frames: int, total_datasets: int, shard_tag: str, print_every_sec: float = 15.0):
        self.total_frames = int(max(total_frames, 0))
        self.total_datasets = int(total_datasets)
        self.shard_tag = shard_tag
        self.print_every_sec = float(print_every_sec)
        self.done_frames = 0
        self.done_datasets = 0
        self.start = time.perf_counter()
        self._last_print = 0.0

    def add_frames(self, n: int, force: bool = False) -> None:
        self.done_frames += int(n)
        now = time.perf_counter()
        if force or (now - self._last_print) >= self.print_every_sec:
            self._last_print = now
            self.print_eta()

    def finish_dataset(self) -> None:
        self.done_datasets += 1
        self.print_eta()

    def print_eta(self) -> None:
        elapsed = time.perf_counter() - self.start
        rate = self.done_frames / max(elapsed, 1e-6)
        remaining = max(self.total_frames - self.done_frames, 0)
        eta = remaining / max(rate, 1e-6)
        pct = 100.0 * self.done_frames / max(self.total_frames, 1)
        tqdm.write(
            f"[eta]{self.shard_tag} frames {self.done_frames}/{self.total_frames} ({pct:5.1f}%) "
            f"datasets {self.done_datasets}/{self.total_datasets} "
            f"| {rate:7.1f} frame/s | elapsed {_fmt_hms(elapsed)} | eta {_fmt_hms(eta)}"
        )
######### // code // ##########


######### // code // ##########
# 中文注释：构造只取单帧 video 的 LeRobotSingleDataset，用于逐帧预计算。
class _VideoOnlyLeRobotSingleDataset(LeRobotSingleDataset):
    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        modality_meta_path = self.dataset_path / "meta/modality.json"
        info_path = self.dataset_path / "meta/info.json"
        with open(modality_meta_path, "r", encoding="utf-8") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        with open(info_path, "r", encoding="utf-8") as f:
            le_info = json.load(f)

        video_meta = {}
        for new_key, field_meta in le_modality_meta.video.items():
            original_key = field_meta.original_key or new_key
            le_video_meta = le_info["features"][original_key]
            names = le_video_meta["names"]
            shape = le_video_meta["shape"]
            height = shape[names.index("height")]
            width = shape[names.index("width")]
            try:
                channels = shape[names.index("channel")]
            except ValueError:
                channels = 3
            fps = (
                le_video_meta.get("video_info", {}).get("video.fps")
                or le_video_meta.get("info", {}).get("video.fps")
                or le_info.get("fps", 30)
            )
            video_meta[new_key] = VideoMetadata(resolution=(width, height), channels=channels, fps=fps)

        def _dummy_stats(dim: int) -> DatasetStatisticalValues:
            zeros = [0.0] * dim
            ones = [1.0] * dim
            return DatasetStatisticalValues(max=ones, min=zeros, mean=zeros, std=ones, q01=zeros, q99=ones)

        state_modalities = {}
        state_stats = {}
        for key, meta in le_modality_meta.state.items():
            dim = int(meta.end - meta.start)
            state_modalities[key] = StateActionMetadata(
                absolute=meta.absolute,
                rotation_type=meta.rotation_type,
                shape=(dim,),
                continuous=True,
            )
            state_stats[key] = _dummy_stats(dim)

        action_modalities = {}
        action_stats = {}
        for key, meta in le_modality_meta.action.items():
            dim = int(meta.end - meta.start)
            action_modalities[key] = StateActionMetadata(
                absolute=meta.absolute,
                rotation_type=meta.rotation_type,
                shape=(dim,),
                continuous=True,
            )
            action_stats[key] = _dummy_stats(dim)

        return DatasetMetadata(
            statistics=DatasetStatistics(state=state_stats, action=action_stats),
            modalities=DatasetModalities(video=video_meta, state=state_modalities, action=action_modalities),
            embodiment_tag=embodiment_tag,
        )


def _make_raw_video_dataset(dataset_path: Path, robot_type: str, data_cfg) -> LeRobotSingleDataset:
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = copy.deepcopy(data_config.modality_config())
    if "video" in modality_config:
        modality_config["video"] = ModalityConfig(delta_indices=[0], modality_keys=modality_config["video"].modality_keys)
    keep = {"video": modality_config["video"]}
    return _VideoOnlyLeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=keep,
        transforms=None,
        embodiment_tag=getattr(data_config, "embodiment_tag", None) or EmbodimentTag.NEW_EMBODIMENT,
        video_backend=data_cfg.get("video_backend", "torchvision_av"),
        delete_pause_frame=bool(data_cfg.get("delete_pause_frame", False)),
        data_cfg=data_cfg,
    )


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    if frame.ndim == 4:
        frame = frame[0]
    return Image.fromarray(frame.astype(np.uint8)).convert("RGB")


def _view_original_key(dataset: LeRobotSingleDataset, video_key: str) -> str:
    subkey = video_key.replace("video.", "", 1)
    meta = dataset.lerobot_modality_meta.video[subkey]
    return meta.original_key or subkey


def _episode_latent_path(
    dataset: LeRobotSingleDataset,
    latents_root: Path,
    original_view_key: str,
    trajectory_id: int,
    length: int,
) -> Path:
    chunk_index = dataset.get_episode_chunk(int(trajectory_id))
    return (
        latents_root
        / f"chunk-{chunk_index:03d}"
        / original_view_key
        / f"episode_{int(trajectory_id):06d}_0_{int(length)}.pth"
    )


def _episode_exists_for_all_views(
    dataset: LeRobotSingleDataset,
    latents_root: Path,
    original_view_keys: list[str],
) -> bool:
    for trajectory_id, length in zip(dataset.trajectory_ids, dataset.trajectory_lengths):
        for original_key in original_view_keys:
            if not _episode_latent_path(dataset, latents_root, original_key, int(trajectory_id), int(length)).exists():
                return False
    return True


def _align_episode_frames(frames: np.ndarray, length: int) -> np.ndarray:
    if frames.shape[0] == length:
        return frames
    if frames.shape[0] > length:
        return frames[:length]
    if frames.shape[0] <= 0:
        raise ValueError("Cannot pad empty video frames.")
    pad = np.repeat(frames[-1:], length - frames.shape[0], axis=0)
    return np.concatenate([frames, pad], axis=0)


def _decode_all_frames_pyav(video_path: str, thread_type: str, thread_count: int) -> list:
    # 中文注释：用 PyAV 解码整段视频。RoboTwin 是 AV1 编码，默认多线程 frame-threading
    # 每个线程都分配一份解码缓冲，在“每节点 8 进程 + 预取”下极易触发 [Errno 12] ENOMEM。
    # thread_type="NONE" + thread_count=1 走单线程软解，显存/内存占用最小，是 AV1 的稳妥兜底。
    import av

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        try:
            stream.thread_type = thread_type
            stream.codec_context.thread_count = int(thread_count)
        except Exception:
            pass
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    finally:
        container.close()


def _get_all_frames_with_closed_reader(
    video_path: str,
    video_backend: str,
    video_backend_kwargs: dict,
) -> np.ndarray:
    if video_backend in {"torchvision_av", "pyav"}:
        import cv2

        last_exc: Exception | None = None
        # 1) 有界多线程（cap=4，避免 thread_count=0 在多核机上开满线程把 AV1 缓冲撑爆 ENOMEM）；
        # 2) 仍失败则退到单线程低内存软解，专治 AV1 ENOMEM。
        for thread_type, thread_count in (("FRAME", 4), ("NONE", 1)):
            try:
                frames = _decode_all_frames_pyav(video_path, thread_type, thread_count)
                if frames:
                    return np.asarray(frames)
            except Exception as exc:
                last_exc = exc
                tqdm.write(f"[warning] PyAV thread_type={thread_type} failed on {video_path}: {exc}")

        # OpenCV 兜底（对 AV1 多半无效，但保留以兼容 H.264 等）。
        cap = cv2.VideoCapture(video_path)
        cv_frames = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                cv_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception as exc:
            last_exc = exc
        finally:
            cap.release()
        if cv_frames:
            return np.asarray(cv_frames)
        raise RuntimeError(f"Unable to decode any frame from {video_path}") from last_exc
    return get_all_frames(
        video_path,
        video_backend=video_backend,
        video_backend_kwargs=video_backend_kwargs,
    )


def _load_episode_video_frames(
    dataset: LeRobotSingleDataset,
    trajectory_id: int,
    view_key: str,
    length: int,
    curr_traj_data=None,
) -> np.ndarray:
    subkey = view_key.replace("video.", "", 1)
    video_path = dataset.get_video_path(int(trajectory_id), subkey)
    if video_path.exists():
        frames = _get_all_frames_with_closed_reader(
            video_path.as_posix(),
            dataset.video_backend,
            dataset.video_backend_kwargs,
        )
        return _align_episode_frames(frames, length)

    # 兼容 image-only parquet 数据集；RoboTwin/LIBERO 正常会走上面的 mp4 分支。
    # 注意：预取线程不写共享的 dataset.curr_traj_data，按需显式传入本 trajectory 的数据。
    if curr_traj_data is None:
        curr_traj_data = dataset.get_trajectory_data(int(trajectory_id))
    original_key = _view_original_key(dataset, view_key)
    if curr_traj_data is not None and original_key in curr_traj_data.columns:
        from io import BytesIO

        out = []
        for entry in curr_traj_data[original_key].tolist():
            if isinstance(entry, np.ndarray):
                out.append(entry)
            elif isinstance(entry, Image.Image):
                out.append(np.array(entry.convert("RGB")))
            elif isinstance(entry, dict) and entry.get("bytes", None) is not None:
                out.append(np.array(Image.open(BytesIO(entry["bytes"])).convert("RGB")))
            elif isinstance(entry, dict) and entry.get("path", None) is not None:
                img_path = Path(entry["path"])
                if not img_path.is_absolute():
                    img_path = dataset.dataset_path / img_path
                out.append(np.array(Image.open(img_path).convert("RGB")))
            else:
                raise TypeError(f"Unsupported image entry type for {original_key}: {type(entry)}")
        return _align_episode_frames(np.asarray(out), length)

    raise FileNotFoundError(f"Cannot find video frames for {trajectory_id=} {view_key=} at {video_path}")


@torch.inference_mode()
def _encode_frames(
    dino: DINOv3Backbone,
    frames: np.ndarray,
    batch_size: int,
    amp_dtype,
) -> torch.Tensor:
    # 中文注释：把一个 episode 的所有帧按 batch_size 切块跑 DINO（autocast 提速），
    # 返回 [num_frames, n_tokens, d_dino] 的 fp32 CPU 张量。
    feats = []
    for start in range(0, len(frames), batch_size):
        chunk = frames[start : start + batch_size]
        images = [_frame_to_pil(frame) for frame in chunk]
        tensor = dino.prepare_dino_input([[img] for img in images])
        with _amp_ctx(amp_dtype):
            out = dino(tensor)
        feats.append(out.detach().float().cpu())
    return torch.cat(feats, dim=0)


def _safe_put(out_queue: "queue.Queue", item, stop_event: threading.Event) -> bool:
    # 中文注释：带超时的 put，消费者出错时 stop_event 置位 → 生产者不会卡死在满队列上。
    while not stop_event.is_set():
        try:
            out_queue.put(item, timeout=1.0)
            return True
        except queue.Full:
            continue
    return False


def _shard_frame_producer(
    dataset: LeRobotSingleDataset,
    work_items: list[tuple],
    latents_root: Path,
    overwrite: bool,
    n_tokens: int,
    d_dino: int,
    out_queue: "queue.Queue",
    stop_event: threading.Event,
) -> None:
    # 中文注释：后台线程预解码视频帧，与主线程的 GPU 推理重叠，喂满显卡。
    # 队列有界 → 自带反压，不会把所有 episode 都解码进内存。
    for trajectory_id, length, view_key, original_view_key in work_items:
        if stop_event.is_set():
            return
        out_path = _episode_latent_path(dataset, latents_root, original_view_key, trajectory_id, length)
        payload = None
        if out_path.exists() and not overwrite:
            try:
                obj = torch.load(out_path, map_location="cpu")
                latent = obj["latent"] if isinstance(obj, dict) else obj
                latent_np = latent.float().numpy()
                if latent_np.ndim == 2:
                    latent_np = latent_np.reshape(length, n_tokens, d_dino)
                payload = ("skip", trajectory_id, length, view_key, original_view_key, out_path, latent_np)
            except Exception as exc:
                # 中文注释：上一轮崩溃可能留下半截/损坏的 .pth；不要让 resume 卡死在它上面，
                # 直接当作缺失重算（配合下面的原子写入，正常情况下不会出现半截文件）。
                tqdm.write(f"[warning] corrupt latent {out_path}: {exc}; recomputing.")
        if payload is None:
            try:
                frames = _load_episode_video_frames(dataset, trajectory_id, view_key, length)
                payload = ("frames", trajectory_id, length, view_key, original_view_key, out_path, frames)
            except Exception as exc:
                payload = ("error", trajectory_id, length, view_key, original_view_key, out_path, exc)
        if not _safe_put(out_queue, payload, stop_event):
            return
    _safe_put(out_queue, None, stop_event)


@torch.inference_mode()
def precompute_single_dataset(
    dataset: LeRobotSingleDataset,
    dino: DINOv3Backbone,
    output_dir_name: str = "latents",
    batch_size: int = 64,
    overwrite: bool = False,
    amp_dtype=torch.bfloat16,
    prefetch: int = 3,
    progress: "ShardProgress | None" = None,
    dataset_frames: int = 0,
) -> None:
    latents_root = dataset.dataset_path / output_dir_name
    latents_root.mkdir(parents=True, exist_ok=True)
    index_path = latents_root / "dino_v3_index.json"
    stats_path = latents_root / "dino_v3_stats.json"

    view_keys = list(dataset.modality_keys["video"])
    original_view_keys = [_view_original_key(dataset, key) for key in view_keys]
    if (
        index_path.exists()
        and stats_path.exists()
        and _episode_exists_for_all_views(dataset, latents_root, original_view_keys)
        and not overwrite
    ):
        print(f"[skip] DINOv3 Lingbot-style latents already exist: {latents_root}", flush=True)
        if progress is not None:
            progress.add_frames(dataset_frames, force=True)
        return

    first_traj = int(dataset.trajectory_ids[0])
    dataset.curr_traj_data = dataset.get_trajectory_data(first_traj)
    dataset.curr_traj_id = first_traj
    first_frame = dataset.get_video(first_traj, view_keys[0], 0)
    first_image = _frame_to_pil(first_frame)
    video_height, video_width = int(np.asarray(first_image).shape[0]), int(np.asarray(first_image).shape[1])
    first_tensor = dino.prepare_dino_input([[first_image]])
    with _amp_ctx(amp_dtype):
        first_feat = dino(first_tensor)
    first_feat = first_feat.detach().float().cpu().numpy()
    n_tokens, d_dino = int(first_feat.shape[1]), int(first_feat.shape[2])
    grid_size = int(n_tokens**0.5)
    latent_height = grid_size if grid_size * grid_size == n_tokens else n_tokens
    latent_width = grid_size if grid_size * grid_size == n_tokens else 1

    stats = RunningChannelStats(d_dino)

    # 中文注释：展平成 (traj, view) 工作项；后台线程预取帧，主线程跑 GPU。
    work_items: list[tuple] = []
    for trajectory_id, length in zip(dataset.trajectory_ids, dataset.trajectory_lengths):
        for view_key, original_view_key in zip(view_keys, original_view_keys):
            work_items.append((int(trajectory_id), int(length), view_key, original_view_key))

    out_queue: "queue.Queue" = queue.Queue(maxsize=max(int(prefetch), 1))
    stop_event = threading.Event()
    producer = threading.Thread(
        target=_shard_frame_producer,
        args=(dataset, work_items, latents_root, overwrite, n_tokens, d_dino, out_queue, stop_event),
        daemon=True,
    )
    producer.start()

    bar = tqdm(total=len(work_items), desc=f"precompute {dataset.dataset_name}")
    try:
        while True:
            item = out_queue.get()
            if item is None:
                break
            kind, trajectory_id, length, view_key, original_view_key, out_path, payload = item
            if kind == "error":
                raise RuntimeError(f"frame load failed traj={trajectory_id} view={view_key}") from payload
            if kind == "skip":
                stats.update(payload)
                bar.update(1)
                if progress is not None:
                    progress.add_frames(length)
                continue

            feats_t = _encode_frames(dino, payload, batch_size, amp_dtype).reshape(length, n_tokens, d_dino)
            stats.update(feats_t.float().numpy())
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_torch_save(
                {
                    "latent": feats_t.reshape(length * n_tokens, d_dino).to(torch.bfloat16).cpu(),
                    "latent_num_frames": length,
                    "latent_height": latent_height,
                    "latent_width": latent_width,
                    "video_num_frames": length,
                    "video_height": video_height,
                    "video_width": video_width,
                    "frame_ids": torch.arange(length, dtype=torch.long),
                    "start_frame": 0,
                    "end_frame": length,
                    "feature_type": "dinov3_patch_tokens",
                    "feature_dim": d_dino,
                    "num_tokens": n_tokens,
                    "view_key": view_key,
                    "original_view_key": original_view_key,
                },
                out_path,
            )
            del feats_t, payload
            bar.update(1)
            if progress is not None:
                progress.add_frames(length)
    finally:
        # 中文注释：置位 stop 并排空队列，解除生产者可能卡在满队列上的 put，避免 join 死锁。
        stop_event.set()
        try:
            while True:
                out_queue.get_nowait()
        except queue.Empty:
            pass
        bar.close()
        producer.join(timeout=60)
    gc.collect()

    # 中文注释：index 先写、stats 最后写，且都用原子替换。stats.json 作为“数据集已完成”的
    # 标记 → 只有所有 episode .pth 落盘后它才出现，find stats.json 计数才可靠。
    _atomic_json_dump(
        {
            "layout": "lingbot_episode",
            "dtype": "bfloat16",
            "feature_type": "dinov3_patch_tokens",
            "feature_dim": d_dino,
            "num_tokens": n_tokens,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "view_keys": view_keys,
            "original_view_keys": original_view_keys,
            "trajectory_ids": [int(x) for x in dataset.trajectory_ids.tolist()],
            "trajectory_lengths": [int(x) for x in dataset.trajectory_lengths.tolist()],
            "root": output_dir_name,
        },
        index_path,
    )
    _atomic_json_dump(stats.to_json(), stats_path)
    print(f"[done] wrote Lingbot-style DINOv3 latents to {latents_root}", flush=True)


def _dataset_completeness(dataset: LeRobotSingleDataset, latents_root: Path) -> dict:
    # 中文注释：仅靠文件是否存在判断完整性，不解码视频。stats.json 是“整库完成”标记，
    # episode .pth 逐个核对，缺一不可。
    index_ok = (latents_root / "dino_v3_index.json").exists()
    stats_ok = (latents_root / "dino_v3_stats.json").exists()
    view_keys = list(dataset.modality_keys["video"])
    original_view_keys = [_view_original_key(dataset, key) for key in view_keys]
    total = 0
    missing = 0
    for trajectory_id, length in zip(dataset.trajectory_ids, dataset.trajectory_lengths):
        for original_view_key in original_view_keys:
            total += 1
            if not _episode_latent_path(dataset, latents_root, original_view_key, int(trajectory_id), int(length)).exists():
                missing += 1
    return {
        "index": index_ok,
        "stats": stats_ok,
        "episodes_total": total,
        "episodes_missing": missing,
        "complete": bool(index_ok and stats_ok and missing == 0),
    }


def run_completeness_check(vla_cfg, output_dir_name: str, mixture: list) -> bool:
    complete: list[str] = []
    incomplete: list[tuple[str, str]] = []
    missing_dir: list[str] = []
    for data_name, _, robot_type in mixture:
        ds_path = Path(vla_cfg.data_root_dir) / data_name
        if not ds_path.exists():
            missing_dir.append(data_name)
            continue
        try:
            dataset = _make_raw_video_dataset(ds_path, robot_type, vla_cfg)
        except Exception as exc:  # pragma: no cover - broken dataset metadata
            incomplete.append((data_name, f"build_failed: {exc}"))
            continue
        info = _dataset_completeness(dataset, dataset.dataset_path / output_dir_name)
        if info["complete"]:
            complete.append(data_name)
        else:
            incomplete.append(
                (
                    data_name,
                    f"index={info['index']} stats={info['stats']} "
                    f"missing_eps={info['episodes_missing']}/{info['episodes_total']}",
                )
            )

    print(
        f"[check] complete={len(complete)}/{len(mixture)} "
        f"incomplete={len(incomplete)} missing_dataset_dir={len(missing_dir)}",
        flush=True,
    )
    for data_name, why in incomplete:
        print(f"[check][INCOMPLETE] {data_name} :: {why}", flush=True)
    for data_name in missing_dir:
        print(f"[check][NO_DATASET_DIR] {data_name}", flush=True)
    all_ok = not incomplete and not missing_dir
    if all_ok:
        print("[check] ALL datasets complete; latents are ready for training.", flush=True)
    else:
        print(
            "[check] NOT complete. Re-run the precompute job with the same config/shards; "
            "resume skips finished episodes and fills only the gaps.",
            flush=True,
        )
    return all_ok


def _preflight_gpu_check() -> None:
    # 中文注释：启动即自检 GPU 能否真正跑 kernel。常见坑：docker 里的 PyTorch 没有为当前 GPU
    # 架构编译 kernel（如 RTX 5090=Blackwell sm_120，旧 torch 只到 sm_90）→ 跑起来才报
    # "no kernel image is available for execution on the device"，且每个 dataset 都崩。
    # 这里用一次小 matmul 提前暴露，给出可操作的建议，而不是崩 72 次。
    if not torch.cuda.is_available():
        print("[env] CUDA not available; running on CPU (will be very slow).", flush=True)
        return
    name = torch.cuda.get_device_name()
    cap = torch.cuda.get_device_capability()
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        arch_list = []
    print(
        f"[env] torch={torch.__version__} cuda={torch.version.cuda} gpu={name} "
        f"capability=sm_{cap[0]}{cap[1]} torch_arch_list={arch_list}",
        flush=True,
    )
    try:
        x = torch.randn(8, 8, device="cuda")
        _ = (x @ x).sum().item()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise SystemExit(
            f"[env][FATAL] GPU kernel smoke test failed: {exc}\n"
            f"  -> The installed PyTorch ({torch.__version__}, archs {arch_list}) has no CUDA kernels "
            f"for this GPU ({name}, sm_{cap[0]}{cap[1]}).\n"
            f"  -> Fix: run on a GPU whose sm is in torch_arch_list (e.g. H20=sm_90 via queue "
            f"project-h20-robot-lab-tcloud-bj), OR use a docker image whose torch matches this arch "
            f"(RTX 5090 = sm_120 needs CUDA 12.8 + torch>=2.7)."
        )


def _claim_path(latents_root: Path) -> Path:
    return latents_root / ".dino_precompute_claim"


def _try_claim(latents_root: Path, stale_seconds: float) -> bool:
    # 中文注释：用 mkdir 的原子性抢占一个 dataset。被占用则看 claim 目录的 mtime：
    # 超过 stale_seconds（默认 30min，远大于单库处理耗时）说明占有者已崩溃 → 抢过来。
    # 这样“只要还有 ≥1 个 worker 在跑”，所有库（含崩溃遗留）最终都会被处理完。
    latents_root.mkdir(parents=True, exist_ok=True)
    claim = _claim_path(latents_root)
    try:
        claim.mkdir(exist_ok=False)
    except FileExistsError:
        try:
            age = time.time() - claim.stat().st_mtime
        except FileNotFoundError:
            age = 0.0
        if age <= stale_seconds:
            return False
        tqdm.write(f"[claim] steal stale claim (age={age:.0f}s>{stale_seconds:.0f}s) {claim}")
        shutil.rmtree(claim, ignore_errors=True)
        try:
            claim.mkdir(exist_ok=False)
        except Exception:
            return False
    try:
        (claim / "owner").write_text(f"pid={os.getpid()} ts={time.time():.0f}")
    except Exception:
        pass
    return True


def _release_claim(latents_root: Path) -> None:
    shutil.rmtree(_claim_path(latents_root), ignore_errors=True)


def run_dynamic(dino, vla_cfg, output_dir_name: str, mixture: list, args, amp_dtype, shard_tag: str) -> None:
    # 中文注释：动态认领模式。不再静态 idx%num_shards 分片，而是每个进程扫全量、抢未完成的库来做。
    # 对“worker 没起齐 / rank 撞车 / 个别 shard 崩了”都鲁棒：谁在跑谁就把剩下的全包了。
    data_root = Path(vla_cfg.data_root_dir)
    n = len(mixture)
    rot = args.shard_id % n if n else 0
    order = list(range(rot, n)) + list(range(0, rot))  # 按 shard_id 轮转，降低初始撞车
    failed: set[int] = set()
    cache: dict = {}  # idx -> built dataset，跨 pass 复用，避免每轮重建 metadata 砸 bucket
    stale = float(args.claim_stale_seconds)
    print(f"[dino_precompute]{shard_tag} dynamic mode over {n} datasets (claim_stale={stale:.0f}s)", flush=True)

    def _get_dataset(idx: int):
        if idx not in cache:
            data_name, _, robot_type = mixture[idx]
            cache[idx] = _make_raw_video_dataset(data_root / data_name, robot_type, vla_cfg)
        return cache[idx]

    while True:
        did_work = False
        incomplete_open = 0  # 仍未完成、且不在本进程 failed 名单里的库数
        for i in order:
            if i in failed:
                continue
            data_name = mixture[i][0]
            if not (data_root / data_name).exists():
                tqdm.write(f"[dynamic] dataset dir missing: {data_name}")
                failed.add(i)
                continue
            try:
                dataset = _get_dataset(i)
            except Exception as exc:
                tqdm.write(f"[dynamic] build failed {data_name}: {exc}")
                failed.add(i)
                continue
            latents_root = dataset.dataset_path / output_dir_name
            if _dataset_completeness(dataset, latents_root)["complete"]:
                continue
            incomplete_open += 1
            if not _try_claim(latents_root, stale):
                continue  # 别的 worker 正在做
            did_work = True
            try:
                precompute_single_dataset(
                    dataset=dataset,
                    dino=dino,
                    output_dir_name=output_dir_name,
                    batch_size=args.batch_size,
                    overwrite=args.overwrite,
                    amp_dtype=amp_dtype,
                    prefetch=args.prefetch,
                    progress=None,
                    dataset_frames=0,
                )
            except Exception as exc:
                tqdm.write(f"[dynamic] dataset {data_name} failed: {exc}")
                failed.add(i)
            finally:
                _release_claim(latents_root)

        # 收敛判断。incomplete_open = 未完成且不在本进程 failed 名单里的库数（含被他人占用的）。
        # - incomplete_open==0：剩下的要么全完成了、要么全在本进程 failed 名单 → 本进程帮不上忙，退出。
        # - incomplete_open>0 但本轮没抢到（全被别人占着）→ sleep 后再扫；占有者若崩溃，其 claim 过期后
        #   会被本进程偷过来重做 → 只要还有进程在跑，最终一定收敛到全部完成。
        if incomplete_open == 0:
            print(f"[dino_precompute]{shard_tag} dynamic mode: nothing left for this worker to claim; exit.", flush=True)
            return
        if not did_work:
            tqdm.write(f"[dynamic] {incomplete_open} datasets owned by others; sleep {args.claim_idle_sleep:.0f}s then rescan.")
            time.sleep(float(args.claim_idle_sleep))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/jointflow/configs/jointflow_libero.yaml")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1, help="Dataset-level shard count for multi-process precompute.")
    parser.add_argument("--shard_id", type=int, default=0, help="This process shard id in [0, num_shards).")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bf16",
        choices=sorted(_AMP_DTYPES.keys()),
        help="DINO autocast precision. bf16 matches the bf16 latent storage and maximizes H20 throughput.",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=3,
        help="Number of episodes to decode ahead on a background thread so the GPU stays fed.",
    )
    parser.add_argument(
        "--eta_interval_sec",
        type=float,
        default=15.0,
        help="Minimum seconds between shard-level ETA prints.",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Do not compute; scan the whole data_mix and report which datasets are missing/incomplete latents.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Dynamic claim-based work distribution: every process sweeps the full mix and atomically claims "
        "incomplete datasets. Robust to missing workers / colliding ranks / crashed shards (any running worker "
        "finishes everything). Ignores --num_shards/--shard_id for assignment (shard_id only rotates scan order).",
    )
    parser.add_argument(
        "--claim_stale_seconds",
        type=float,
        default=1800.0,
        help="A dataset claim older than this is considered crashed and may be stolen (dynamic mode).",
    )
    parser.add_argument(
        "--claim_idle_sleep",
        type=float,
        default=30.0,
        help="When all incomplete datasets are claimed by others, wait this long before rescanning (dynamic mode).",
    )
    parser.add_argument(
        "--config_override",
        action="append",
        default=[],
        help="OmegaConf dotlist override, e.g. datasets.vla_data.data_root_dir=/path.",
    )
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError(f"--num_shards must be >= 1, got {args.num_shards}")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"--shard_id must be in [0, {args.num_shards}), got {args.shard_id}")

    amp_dtype = _AMP_DTYPES[args.amp_dtype]
    _enable_fast_matmul()

    cfg = OmegaConf.load(args.config_yaml)
    if args.config_override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.config_override))

    vla_cfg = cfg.datasets.vla_data
    output_dir_name = vla_cfg.get("dino_feature_dir", "latents")
    mixture = list(resolve_data_mix(vla_cfg.data_mix))

    # 中文注释：--check_only 只做完整性体检（不加载 DINO、不跑 GPU），扫描整个 data_mix，
    # 列出哪些 dataset 缺 index/stats 或缺 episode，最后给出 complete X/N。用于确认 latent 完整。
    if args.check_only:
        all_ok = run_completeness_check(vla_cfg, output_dir_name, mixture)
        # 非零退出码方便脚本里 `while ! ...; do 重跑; done` 循环补齐。
        raise SystemExit(0 if all_ok else 2)

    # 中文注释：先做 GPU 自检（架构不匹配立刻报错退出，避免每个 dataset 崩一遍）。
    _preflight_gpu_check()

    # 中文注释：经 resolve_dino_spec 统一解析（支持 framework.dino.model_size: vitl16 等），
    # 预计算 latents 的维度与在线/framework 完全一致 → online_dino=auto 可直接命中离线路径。
    from starVLA.jointflow.modules.dino_v3 import resolve_dino_spec

    spec = resolve_dino_spec(cfg.framework.dino)
    dino = DINOv3Backbone(**spec)
    dino = dino.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()

    shard_tag = f"[shard {args.shard_id}/{args.num_shards}]"

    # 中文注释：动态认领模式（推荐用于多机）。对 worker 没起齐 / rank 撞车 / 个别 shard 崩溃都鲁棒。
    if args.dynamic:
        run_dynamic(dino, vla_cfg, output_dir_name, mixture, args, amp_dtype, shard_tag)
        return

    assigned = [(idx, entry) for idx, entry in enumerate(mixture) if idx % args.num_shards == args.shard_id]
    print(
        f"[dino_precompute] config={args.config_yaml} data_mix={vla_cfg.data_mix} "
        f"datasets={len(mixture)} shard={args.shard_id}/{args.num_shards} assigned={len(assigned)} "
        f"amp={args.amp_dtype} batch_size={args.batch_size} prefetch={args.prefetch}",
        flush=True,
    )
    if not assigned:
        print("[dino_precompute] no datasets assigned to this shard; exit.")
        return
    print(
        f"[dino_precompute] config={args.config_yaml} data_mix={vla_cfg.data_mix} "
        f"datasets={len(mixture)} shard={args.shard_id}/{args.num_shards} assigned={len(assigned)} "
        f"amp={args.amp_dtype} batch_size={args.batch_size} prefetch={args.prefetch}",
        flush=True,
    )
    if not assigned:
        print("[dino_precompute] no datasets assigned to this shard; exit.")
        return

    # 中文注释：先构建每个 dataset（仅读 metadata），统计本 shard 的总帧数 = Σ length × 视角数，
    # 用于打印整 shard 的 ETA；dataset 对象复用，避免二次构建。
    built: list[tuple] = []
    total_frames = 0
    for idx, (data_name, _, robot_type) in assigned:
        dataset = _make_raw_video_dataset(Path(vla_cfg.data_root_dir) / data_name, robot_type, vla_cfg)
        n_views = len(dataset.modality_keys["video"])
        n_frames = int(sum(int(x) for x in dataset.trajectory_lengths)) * n_views
        built.append((idx, data_name, dataset, n_frames))
        total_frames += n_frames

    progress = ShardProgress(total_frames, len(built), shard_tag, print_every_sec=float(args.eta_interval_sec))
    print(f"[dino_precompute]{shard_tag} total work frames={total_frames} across {len(built)} datasets", flush=True)

    for idx, data_name, dataset, n_frames in built:
        print(
            f"[dino_precompute]{shard_tag} dataset_index={idx} dataset={data_name} frames={n_frames}",
            flush=True,
        )
        precompute_single_dataset(
            dataset=dataset,
            dino=dino,
            output_dir_name=output_dir_name,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            amp_dtype=amp_dtype,
            prefetch=args.prefetch,
            progress=progress,
            dataset_frames=n_frames,
        )
        progress.finish_dataset()


if __name__ == "__main__":
    main()
######### // code // ##########
