"""JointFlow policy-server wrapper (state normalization + live DINO at eval).

所属：JointFlow LIBERO eval（server 端）。
复用（全部 import，不改源）：
- deployment.model_server.policy_wrapper.PolicyServerWrapper
  （加载 framework + 动作反归一化 + handshake metadata）
- deployment.model_server.policy_norm_processor 的 _resolve_robot_type /
  _infer_key_dims / _build_dataset_metadata（从 dataset_statistics.json 重建 metadata）
- starVLA.jointflow.data.joint_dataset._append_state_norm_if_needed（与训练同款的
  state 归一化 transform 组装）
- starVLA.jointflow.modules.dino_v3.DINOv3Backbone（eval 现场提特征）

为何要子类化而不是直接用 PolicyServerWrapper：
1) 必须先 import jointflow framework 触发 @FRAMEWORK_REGISTRY.register("QwenJointFlow")，
   否则 from_pretrained→build_framework 找不到该 framework（其 auto-import 只扫
   starVLA/model/framework/，不扫 jointflow）。
2) 模型用 state.inject_mode=token 训练（注入的是**归一化后**的 proprio），而 LIBERO
   env 给的是**原始** state；需在 server 侧用训练同款 transform 归一化后再喂模型，
   否则 StateEncoder 收到原始/零值，train/eval 失配，成功率下降（用户已选"归一化真实 state"）。
3) eval 是 live DINO 第一次被调用：DINOv3 权重可能 gated/离线，需支持用 env 覆盖
   本地权重路径并刷新 DINO 归一化统计。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# 中文注释：import 即注册 QwenJointFlow，必须在 from_pretrained 之前发生。
import starVLA.jointflow.framework.qwen_joint_flow  # noqa: F401
from starVLA.jointflow.data.joint_dataset import _append_state_norm_if_needed
from starVLA.jointflow.data.mix_registry import register_jointflow_mixtures
from starVLA.jointflow.modules.dino_v3 import DINOv3Backbone

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.policy_norm_processor import (
    _build_dataset_metadata,
    _infer_key_dims,
    _resolve_robot_type,
)

from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.model.framework.share_tools import read_mode_config

logger = logging.getLogger(__name__)


######### // code // ##########
# 中文注释：JointFlow 专用 policy-server wrapper。
# __init__ 在父类加载 framework + 动作反归一化之后，额外做三件事：
#   (a) 按 env 覆盖重建 live DINO（离线/gated 权重场景）；
#   (b) 若 run_dir 下有打包好的 dino_v3_stats.json，刷新 framework 的 DINO 归一化统计；
#   (c) 构建与训练同款的 state 归一化 transform。
# predict_action 在转发给父类前，先把每个 example 里的原始 state 归一化。
class JointFlowPolicyServerWrapper(PolicyServerWrapper):
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
    ) -> None:
        register_jointflow_mixtures()
        super().__init__(ckpt_path=ckpt_path, device=device, use_bf16=use_bf16, unnorm_key=unnorm_key)
        self._run_dir = Path(self._ckpt_path).parents[1]

        self._state_keys: List[str] = []
        self._state_key_dims: Dict[str, int] = {}
        self._state_normalizer: Optional[ComposedModalityTransform] = None
        self._state_total_dim: int = 0

        self._maybe_rebuild_live_dino()
        self._maybe_refresh_dino_stats()
        self._build_state_normalizer()

    # ------------------------------------------------------------------
    # (a) live DINO rebuild
    # ------------------------------------------------------------------
    def _maybe_rebuild_live_dino(self) -> None:
        repo = os.environ.get("JOINTFLOW_DINO_REPO_OR_DIR", "").strip()
        weights = os.environ.get("JOINTFLOW_DINO_WEIGHTS", "").strip()
        loader = os.environ.get("JOINTFLOW_DINO_LOADER", "").strip()
        if not (repo or weights or loader):
            return  # 中文注释：无 env 覆盖时，交给 framework 在 predict_action 里 lazy 加载。

        fw = self._framework
        dino_cfg = fw.config.framework.dino
        logger.info(
            "JointFlow eval: rebuilding live DINO (repo=%s weights=%s loader=%s)",
            repo or "<cfg>", weights or "<cfg>", loader or "<cfg>",
        )
        fw.dino = DINOv3Backbone(
            name=dino_cfg.get("name", "dinov3_vits16"),
            hf_model_id=dino_cfg.get("hf_model_id", "facebook/dinov3-vits16-pretrain-lvd1689m"),
            repo_or_dir=repo or dino_cfg.get("repo_or_dir", "facebookresearch/dinov3"),
            weights=(weights or dino_cfg.get("weights", None)) or None,
            loader=loader or dino_cfg.get("loader", "auto"),
            image_size=int(dino_cfg.get("image_size", 224)),
            patch_size=int(dino_cfg.get("patch_size", 16)),
            embed_dim=int(dino_cfg.get("embed_dim", 384)),
        ).to(fw.device)

    # ------------------------------------------------------------------
    # (b) refresh DINO normalization stats from bundled file
    # ------------------------------------------------------------------
    def _maybe_refresh_dino_stats(self) -> None:
        bundled = self._run_dir / "dino_v3_stats.json"
        if not bundled.exists():
            return  # 中文注释：没有打包统计时，framework __init__ 已尝试从数据根加载。
        try:
            self._framework._load_dino_stats(str(bundled))
            logger.info("JointFlow eval: refreshed DINO stats from %s", bundled)
        except Exception as exc:  # pragma: no cover
            logger.warning("JointFlow eval: failed to refresh DINO stats from %s: %s", bundled, exc)

    # ------------------------------------------------------------------
    # (c) state normalizer (match training-time normalization)
    # ------------------------------------------------------------------
    def _build_state_normalizer(self) -> None:
        cfg, norm_stats = read_mode_config(self._ckpt_path)

        inject_mode = ((cfg.get("framework") or {}).get("state") or {}).get("inject_mode", "token")
        if inject_mode != "token":
            logger.info("JointFlow eval: state.inject_mode=%s -> no state normalization.", inject_mode)
            return

        unnorm_key = self._default_unnorm_key
        if unnorm_key is None:
            if len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats.keys()))
            else:
                logger.warning(
                    "JointFlow eval: multiple unnorm_keys %s and no default; state will NOT be "
                    "normalized (StateEncoder may see raw values).", list(norm_stats.keys()),
                )
                return

        robot_type = _resolve_robot_type(cfg, unnorm_key=unnorm_key)
        data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        state_keys = list(getattr(data_config, "state_keys", []))
        if not state_keys:
            logger.info("JointFlow eval: data_config has no state_keys -> no state normalization.")
            return
        action_keys = list(data_config.action_keys)

        stats_for_key = norm_stats[unnorm_key]
        state_key_dims = _infer_key_dims(data_config, stats_for_key, state_keys, "state")
        action_key_dims = _infer_key_dims(data_config, stats_for_key, action_keys, "action")
        ds_meta = _build_dataset_metadata(
            stats_for_key=stats_for_key,
            embodiment_tag=data_config.embodiment_tag,
            action_keys=action_keys,
            state_keys=state_keys,
            action_key_dims=action_key_dims,
            state_key_dims=state_key_dims,
        )

        # 中文注释：state 归一化模式必须与训练一致。训练侧 joint_dataset 里
        # _append_state_norm_if_needed 默认 gripper->binary、其余->q99；这里复用同函数，
        # 若 cfg 显式给了 state_norm_modes 也透传。
        try:
            state_norm_modes = ((cfg.get("datasets") or {}).get("vla_data") or {}).get("state_norm_modes", None)
        except Exception:
            state_norm_modes = None

        transform = ComposedModalityTransform(transforms=[])
        transform = _append_state_norm_if_needed(transform, state_keys, state_norm_modes)
        transform.set_metadata(ds_meta)
        transform.eval()

        self._state_normalizer = transform
        self._state_keys = state_keys
        self._state_key_dims = state_key_dims
        self._state_total_dim = sum(int(state_key_dims.get(k, 1)) for k in state_keys)
        logger.info(
            "JointFlow eval: state normalizer ready (keys=%s, total_dim=%d, modes from training).",
            state_keys, self._state_total_dim,
        )

    def _normalize_state(self, raw_state) -> np.ndarray:
        # 中文注释：原始 state -> 切成各 state.<sub> 子键 -> transform.apply 归一化 -> 拼回。
        # 输入可为 [D] 或 [T,D]；输出 [T,D]（与训练 state 形状一致，StateEncoder 取 t=0）。
        arr = np.asarray(raw_state, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        total = self._state_total_dim
        if arr.shape[-1] < total:
            pad = np.zeros((*arr.shape[:-1], total - arr.shape[-1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=-1)
        elif arr.shape[-1] > total:
            arr = arr[..., :total]

        data: Dict[str, np.ndarray] = {}
        cursor = 0
        for key in self._state_keys:
            dim = int(self._state_key_dims.get(key, 1))
            data[key] = np.ascontiguousarray(arr[..., cursor:cursor + dim], dtype=np.float32)
            cursor += dim

        out = self._state_normalizer.apply(data)
        parts: List[np.ndarray] = []
        for key in self._state_keys:
            v = out[key]
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            parts.append(np.asarray(v, dtype=np.float32))
        return np.concatenate(parts, axis=-1)

    @staticmethod
    def _to_pil_images(imgs) -> List[Image.Image]:
        # 中文注释：client 经 msgpack 发来的是 np.uint8（msgpack 不能序列化 PIL）；
        # live DINO 的 prepare_dino_input 需要 PIL，这里在 server 端统一转换。
        seq = imgs if isinstance(imgs, (list, tuple)) else [imgs]
        out = []
        for im in seq:
            if isinstance(im, Image.Image):
                out.append(im.convert("RGB"))
            else:
                out.append(Image.fromarray(np.asarray(im).astype(np.uint8)).convert("RGB"))
        return out

    def _prepare_examples(self, examples: List[dict]) -> List[dict]:
        # 中文注释：转发给 framework 前预处理每个 example：
        #   (1) image: np -> PIL（live DINO 需要）；(2) state: 原始 -> 训练同款归一化。
        out = []
        for ex in examples:
            if not isinstance(ex, dict):
                out.append(ex)
                continue
            new = dict(ex)
            if new.get("image", None) is not None:
                new["image"] = self._to_pil_images(new["image"])
            if self._state_normalizer is not None and new.get("state", None) is not None:
                new["state"] = self._normalize_state(new["state"])
            out.append(new)
        return out

    # ------------------------------------------------------------------
    # override: np->PIL images + state normalization before base un-norm path
    # ------------------------------------------------------------------
    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        examples = self._prepare_examples(examples)
        return super().predict_action(examples=examples, unnorm_key=unnorm_key, **kwargs)
######### // code // ##########
