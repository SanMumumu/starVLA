"""PR1: JointFlow LeRobot dataset wrapper.

复用:
- starVLA.dataloader.gr00t_lerobot.datasets.LeRobotSingleDataset / LeRobotMixtureDataset
- starVLA.dataloader.gr00t_lerobot.registry registries
- starVLA.dataloader.lerobot_datasets.collate_fn

说明:
本文件通过子类化/组合读取 DINOv3 离线特征，不修改 gr00t_lerobot 源码。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP, EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.lerobot_datasets import collate_fn
from starVLA.dataloader.jointflow.mix_registry import resolve_data_mix


######### // code // ##########
# 中文注释：把 torch.Tensor / numpy / list 统一转成 numpy，便于 pack sample。
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


def _safe_view_name(key: str) -> str:
    return str(key).replace("/", "__").replace(".", "_")


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
# 中文注释：JointFlow 数据集子类。
# 在线模式（online_dino=True，默认）：返回原始图像 image_0=s_t / image_1=s_{t+stride}（[V,H,W,C] uint8），
#   DINO 在模型 forward 内部在线提特征（见 framework）。视频解码在 worker 里完成，不在 worker 跑 DINO。
# 离线模式（online_dino=False）：保留旧逻辑，读取离线 DINOv3 特征 dino_0/dino_1 [V,N_v,D]（已按 stats 标准化）。
# 两种模式都返回 action [H_a,7]，state [1,state_dim]，lang str。
class JointLiberoDataset(LeRobotSingleDataset):
    def __init__(
        self,
        *args,
        online_dino: bool = True,
        dino_feature_dir: str = "latents",
        # 中文注释：≥视角数才能让同一样本 dino_0→dino_1 全部命中（3 视角时 2 会把 view0 挤掉→重读）。
        # 缓存现已在 _pack_sample 末尾逐样本清空（防 per-dataset×worker 常驻膨胀→节点 OOM），此值只是样本内瞬态上限。
        dino_episode_cache_size: int = 4,
        dino_target_latents: bool = False,
        **kwargs,
    ):
        self.online_dino = bool(online_dino)
        self.dino_feature_dir_name = dino_feature_dir
        #######
        # 中文注释：hybrid 开关——在线模式(出 raw 图给 Qwen 原生视觉)的同时,额外读预存 DINO latent 当
        # 世界模型 target(dino_0/dino_1)。这样 WAM 既有 raw 图喂 policy、又用上预存 latent,
        # 不用每步在线跑 DINO backbone(省算力 + 不必加载大 backbone 权重)。需 latents/ store 存在。
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
        super().__init__(*args, **kwargs)

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

        obj = torch.load(path, map_location="cpu")
        latent = obj["latent"] if isinstance(obj, dict) else obj
        if not torch.is_tensor(latent):
            latent = torch.as_tensor(latent)
        latent = latent.float()

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
            raise ValueError(
                f"Expected latent [T,N,D] or flattened [T*N,D] in {path}, got {tuple(latent.shape)}"
            )

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
        """中文注释：latent target 只读部分视角（config: datasets.vla_data.dino_target_latent_views，如 [0]=cam_high）。
        None=全读（兼容旧行为）。WAM 世界模型 target 只用 view 0（QwenGR00T._wam_dino_target 取 z[:,0]），
        读全部视角纯属浪费 bucket I/O 与 worker 内存（每视角=整个 episode 的 fp32 latent）。"""
        views = _cfg_get(self.data_cfg, "dino_target_latent_views", None)
        if views is None:
            return None
        return [int(v) for v in views]

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
                per_view.append(latent[safe_idx].numpy())
            z = np.stack(per_view, axis=0).astype(np.float32)
        else:
            abs_idx = self._absolute_frame_index(trajectory_id, frame_index)
            z = np.asarray(self._dino_memmap[abs_idx], dtype=np.float32)
        mean = np.asarray(self._dino_stats["mean"], dtype=np.float32)
        std = np.asarray(self._dino_stats["std"], dtype=np.float32)
        std = np.maximum(std, 1e-6)
        return ((z - mean) / std).astype(np.float32)

    def _future_dino_stride(self) -> int:
        action_horizon = int(_cfg_get(self.data_cfg, "action_horizon", _cfg_get(self.data_cfg, "future_action_window_size", 8)))
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

        # 中文注释：在线模式解码视频帧（video delta_indices=[0, stride] → 每个 view 返回 2 帧：
        # 当前帧 s_t 和未来帧 s_{t+stride}）。原始帧直接暂存到 self，不放进 data，
        # 这样 transforms（已 drop video）不会碰它们，避免缺 key 报错 / 重复处理。
        # DINO 不在 worker 里跑，只解码 RGB → 模型 forward 在 GPU 上在线提特征。
        # decode_video=False：只取 state/action（如启动估协方差），跳过 PyAV 解码。
        if self.online_dino and decode_video:
            self._last_video_frames = {}
            for key in self.modality_keys.get("video", []):
                self._last_video_frames[key] = self.get_data_by_modality(trajectory_id, "video", key, base_index)

        # 离线模式不解码视频（直接读离线 latent），避免 PyAV/torchvision worker OOM。
        for modality in ("state", "action", "language"):
            for key in self.modality_keys.get(modality, []):
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return self._apply_action_mode(data)

    def read_action_only(self, trajectory_id: int, base_index: int) -> np.ndarray:
        """返回单样本归一化 action [H, action_dim]，与 __getitem__ 完全同 transform，但**不解码视频/不读 latent**。
        中文注释：供启动时估 correlated-noise 协方差用——只读 parquet 里的 state/action（走同一 get_step_data +
        transforms 路径，只是 decode_video=False），避免逐样本 PyAV 解码在多机 bucket I/O 下挂起/超时。"""
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
        action = np.concatenate(action, axis=1).astype(np.float32)

        future_index, future_steps, future_stride = self._future_dino_index(trajectory_id, base_index)
        sample = {
            "action": action,
            "lang": data[self.modality_keys["language"][0]][0],
            "robot_tag": self.tag,
            "dataset_name": self.dataset_name,
            "trajectory_id": np.int64(trajectory_id),
            "base_index": np.int64(base_index),
            "future_index": np.int64(future_index),
            "future_valid": bool(future_steps > 0),
            "future_valid_steps": np.int64(future_steps),
            "future_stride": np.int64(future_stride),
        }

        if self.online_dino:
            # 中文注释：在线模式返回原始图像。image_0=当前帧（每个 view 第 0 帧），
            # image_1=未来帧（每个 view 第 1 帧，即 s_{t+stride}；边界处 get_video 已 clip 到末帧）。
            # 形状 [V,H,W,C] uint8；view 顺序 = self.modality_keys["video"]（LIBERO: primary 在前）。
            view_keys = list(self.modality_keys.get("video", []))
            if self._last_video_frames is None:
                raise RuntimeError("JointLiberoDataset(online) requires get_step_data to decode video first.")
            img0, img1 = [], []
            for key in view_keys:
                frames = np.asarray(self._last_video_frames[key])  # [T,H,W,C], T>=2
                img0.append(frames[0])
                img1.append(frames[min(1, frames.shape[0] - 1)])
            sample["image_0"] = np.stack(img0, axis=0)
            sample["image_1"] = np.stack(img1, axis=0)
            sample["dino_view_keys"] = view_keys
            #######
            # 中文注释：hybrid——在线出 raw 图的同时,再读预存的 DINO latent（已按 store stats 标准化）当 target。
            # dino_0=当前帧(delta 用)、dino_1=未来帧。wam 优先用它当世界模型 target,省掉在线 DINO 抽取。
            if self._dino_target_latents:
                sample["dino_0"] = self._read_dino(trajectory_id, base_index)
                sample["dino_1"] = self._read_dino(trajectory_id, future_index)
            #######
        else:
            # 中文注释：离线模式读取预计算 + 标准化后的 DINO 特征。
            sample["dino_0"] = self._read_dino(trajectory_id, base_index)
            sample["dino_1"] = self._read_dino(trajectory_id, future_index)
            sample["dino_view_keys"] = (
                list(self._dino_index.get("view_keys", self.modality_keys["video"]))
                if self._dino_index
                else list(self.modality_keys["video"])
            )

        if self.data_cfg is not None and _cfg_get(self.data_cfg, "include_state", True) not in ["False", False]:
            state = []
            for state_key in self.modality_keys.get("state", []):
                if state_key in data:
                    state.append(_to_numpy(data[state_key]))
            if state:
                sample["state"] = np.concatenate(state, axis=1).astype(np.float32)
        #######
        # 中文注释：episode latent 缓存只为**同一样本内** dino_0→dino_1 复用（同 episode 同 view 文件不二读）。
        # 随机采样下跨样本命中≈0，但缓存是 per-dataset 的：mixture 有 50-100 个数据集对象 × 每个 2-4 条
        # × 整集 fp32 latent（~百 MB/条）× 每节点几十个 persistent worker → 常驻内存线性膨胀直至节点 OOM
        # （表现为某 rank 被 SIGKILL、其余 rank 报 ncclRemoteError）。∴ 样本打包完立即清空。
        if self._episode_latent_cache:
            self._episode_latent_cache.clear()
        #######
        return sample
######### // code // ##########


######### // code // ##########
# 中文注释：构造 JointFlow 专用 dataset，不经 starVLA.dataloader.__init__.build_dataloader。
# 它临时覆盖 observation_indices：video=[0, future_stride]，language=[0]，state=[0]，action=[0..H-1]。
# 在线模式下 video 取 [0, future_stride]：第 0 帧=s_t，第 1 帧=s_{t+stride}（与未来动作 / 世界模型目标对齐）。
def _make_joint_single_dataset(
    dataset_path: Path, robot_type: str, data_cfg, online_dino: bool | None = None
) -> JointLiberoDataset:
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = copy.deepcopy(data_config.modality_config())

    action_horizon = int(_cfg_get(data_cfg, "action_horizon", _cfg_get(data_cfg, "future_action_window_size", 8)))
    # 中文注释：online_dino 由 build_joint_dataset 在"混合级"统一解析后传入（支持 "auto"：
    # 有 latent 用 latent、没有就在线）。直接调用本函数未传时退回 data_cfg 的 true/false
    # （"auto" 字符串兜底按在线处理，避免 bool("auto")=True 的歧义恰好也指在线）。
    if online_dino is None:
        raw = _cfg_get(data_cfg, "online_dino", True)
        online_dino = True if (isinstance(raw, str) and raw.strip().lower() == "auto") else bool(raw)
    world_model_cfg = _cfg_get(data_cfg, "world_model", None)
    future_stride = _cfg_get(world_model_cfg, "future_stride", None)
    if future_stride is None:
        future_stride = _cfg_get(data_cfg, "future_dino_stride", action_horizon)
    future_stride = max(int(future_stride), 1)
    if "video" in modality_config:
        # 中文注释：在线取未来帧用 stride；离线不解码视频，delta 取 [0,1] 仅占位无影响。
        video_delta = [0, future_stride] if online_dino else [0, 1]
        modality_config["video"] = ModalityConfig(delta_indices=video_delta, modality_keys=modality_config["video"].modality_keys)
    if "language" in modality_config:
        modality_config["language"] = ModalityConfig(delta_indices=[0], modality_keys=modality_config["language"].modality_keys)
    if "state" in modality_config:
        modality_config["state"] = ModalityConfig(delta_indices=[0], modality_keys=modality_config["state"].modality_keys)
    if "action" in modality_config:
        modality_config["action"] = ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=modality_config["action"].modality_keys,
        )

    transforms = _drop_video_transforms(data_config.transform())
    state_norm_modes = _cfg_get(data_cfg, "state_norm_modes", None)
    transforms = _append_state_norm_if_needed(transforms, modality_config.get("state", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys, state_norm_modes)

    embodiment_tag = getattr(data_config, "embodiment_tag", None) or EmbodimentTag.NEW_EMBODIMENT
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
# 中文注释：online_dino="auto" 的混合级解析（"有 latent 就 latent，没有就在线"）。
# 规则：混合内"所有"数据集都有离线 latents 且统计维度与 framework.dino（经
# resolve_dino_spec，含 model_size）期望的 embed_dim 一致 → 离线；任一缺失/维度
# 不符 → 整个混合在线。必须整混合统一：framework._stack_field 以首样本字段为准，
# 同一 batch 混出 dino_0 与 image_0 两种字段会崩。
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
        return int(len(stats.get("mean", [])))
    except Exception:
        return None


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


def build_joint_dataset(cfg, mode: str = "train") -> LeRobotMixtureDataset:
    vla_cfg = cfg.datasets.vla_data
    mixture_spec = resolve_data_mix(vla_cfg.data_mix)
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
    return DataLoader(
        dataset,
        batch_size=int(cfg.datasets.vla_data.per_device_batch_size),
        collate_fn=collate_fn,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
######### // code // ##########


#######
# 中文注释：E1.3 correlated noise——从数据集采样归一化动作 chunk，估计 flat(=horizon·action_dim) 的协方差 Σ，
# 返回正则化协方差 βΣ+(1−β)I 的下三角 Cholesky（[flat,flat]）。训练启动时算一次，注入 action head。
def compute_action_correlation_cholesky(
    mixture_dataset: LeRobotMixtureDataset, num_samples: int = 20000, beta: float = 0.5, seed: int = 0
) -> np.ndarray:
    """估动作协方差的 Cholesky（correlated-noise 用）。

    中文注释：**只读 action，不解码视频**——按各子数据集步数比例随机取样，走子数据集的 `read_action_only`
    （同 __getitem__ 的 transform，但跳过 PyAV 视频解码 + latent 读取）。这样在多机 bucket I/O 下 rank0 估计
    只碰 parquet，避免逐样本视频解码挂起/超时（见 train_starvla 启动 barrier）。子数据集若无 `read_action_only`
    （非 jointflow）则回退到全量 __getitem__。逐样本 try/except 跳过坏样本，单个坏轨迹不再拖垮启动。
    """
    rng = np.random.default_rng(seed)
    datasets = list(getattr(mixture_dataset, "datasets", None) or [mixture_dataset])
    lengths = np.array([max(int(len(d)), 1) for d in datasets], dtype=np.float64)
    probs = lengths / lengths.sum()
    target = int(num_samples)
    rows: list[np.ndarray] = []
    tries = 0
    max_tries = max(target * 4, 16)
    while len(rows) < target and tries < max_tries:
        tries += 1
        d = datasets[int(rng.choice(len(datasets), p=probs))]
        try:
            if hasattr(d, "read_action_only"):
                traj_id, base_index = d.all_steps[int(rng.integers(0, len(d)))]
                a = d.read_action_only(int(traj_id), int(base_index))
            else:  # 回退：非 jointflow 数据集，走全量 __getitem__
                a = mixture_dataset[int(rng.integers(0, len(mixture_dataset)))].get("action")
        except Exception:
            continue  # 坏样本/坏轨迹直接跳过，不拖垮启动
        if a is None:
            continue
        a = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
        rows.append(a.reshape(-1).astype(np.float32))
    if not rows:
        raise RuntimeError("compute_action_correlation_cholesky: 采不到 action，检查数据集。")
    X = np.stack(rows, axis=0)  # [M, flat]
    flat = X.shape[1]
    Sigma = np.cov(X, rowvar=False).reshape(flat, flat).astype(np.float64)
    Sigma_reg = float(beta) * Sigma + (1.0 - float(beta)) * np.eye(flat)
    # 数值稳健：对角加微小抖动后 Cholesky
    L = np.linalg.cholesky(Sigma_reg + 1e-6 * np.eye(flat))
    return L.astype(np.float32)
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
        if "image_0" in sample:  # 在线模式
            info["image_0"] = np.asarray(sample["image_0"]).shape
            info["image_1"] = np.asarray(sample["image_1"]).shape
            info["dino_view_keys"] = sample.get("dino_view_keys")
        else:  # 离线模式
            info["dino_0"] = np.asarray(sample["dino_0"]).shape
            info["dino_1"] = np.asarray(sample["dino_1"]).shape
        print(info)
        if idx + 1 >= args.num_batches:
            break
