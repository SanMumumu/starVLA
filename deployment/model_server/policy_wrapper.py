# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config

from deployment.model_server.policy_norm_processor import (
    PolicyNormProcessor,
    _build_dataset_metadata,
    _infer_key_dims,
    _resolve_robot_type,
)


class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
        dino_stats_path: Optional[str] = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path)
        #######
        # 中文注释：E1.3 correlated noise——Cholesky buffer persistent=False 不进 ckpt，评测时从 ckpt 上两级目录
        # 的 action_correlation_cholesky.npy 读出并注入；缺文件或非 jointflow 模型时安全跳过（不影响其它实验）。
        chol_path = Path(self._ckpt_path).parents[1] / "action_correlation_cholesky.npy"
        if chol_path.exists() and hasattr(framework, "set_action_correlation"):
            framework.set_action_correlation(np.load(str(chol_path)))
            logging.info("PolicyServerWrapper: injected correlated-noise Cholesky from %s", chol_path)
        #######
        #######
        # 中文注释：jointflow 训练用离线 DINO 特征（按 per-suite stats 标准化），eval 在线提取必须用同一份 stats。
        # 不加载则在线特征是原始尺度→视觉条件失效→SR≈0。stats 随 suite 不同，由 --dino_stats_path 指定。
        if dino_stats_path and hasattr(framework, "set_dino_stats"):
            framework.set_dino_stats(str(dino_stats_path))
            logging.info("PolicyServerWrapper: loaded online-DINO stats from %s", dino_stats_path)
        #######
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        # Co-located metadata.
        model_cfg, _ = read_mode_config(self._ckpt_path)
        self._model_cfg = model_cfg
        self._state_normalizer: Optional[ComposedModalityTransform] = None
        self._state_keys: List[str] = []
        self._state_key_dims: Dict[str, int] = {}
        self._state_total_dim = 0
        self._expects_state = self._config_expects_state(model_cfg)

        # action_chunk_size = future_action_window_size + 1 (matches old client).
        action_model_cfg = model_cfg["framework"]["action_model"]

        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                f"PolicyServerWrapper: no action_horizon or future_action_window_size found in model config for {self._ckpt_path}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        _, _ns = read_mode_config(self._ckpt_path)
        self._available_unnorm_keys: List[str] = list(_ns.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

        if self._expects_state:
            self._build_state_normalizer(model_cfg, _ns)

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "none", "no"}
        return bool(value)

    @classmethod
    def _config_expects_state(cls, model_cfg: dict) -> bool:
        vla_cfg = ((model_cfg.get("datasets") or {}).get("vla_data") or {})
        if cls._truthy(vla_cfg.get("include_state", False)):
            return True

        state_cfg = ((model_cfg.get("framework") or {}).get("state") or {})
        return str(state_cfg.get("inject_mode", "none")) == "token"

    def _build_state_normalizer(self, model_cfg: dict, norm_stats: dict) -> None:
        unnorm_key = self._default_unnorm_key
        if unnorm_key is None:
            if len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats.keys()))
            else:
                logging.warning(
                    "PolicyServerWrapper: model expects state but multiple unnorm keys exist (%s); "
                    "state will be forwarded without normalization until unnorm_key is specified.",
                    list(norm_stats.keys()),
                )
                return

        robot_type = _resolve_robot_type(model_cfg, unnorm_key=unnorm_key)
        data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        state_keys = list(getattr(data_config, "state_keys", []))
        if not state_keys:
            logging.info("PolicyServerWrapper: model expects state but data_config has no state keys.")
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

        transform = data_config.transform()
        if not isinstance(transform, ComposedModalityTransform):
            transform = ComposedModalityTransform(transforms=[transform])
        transform.set_metadata(ds_meta)
        transform.eval()

        self._state_normalizer = transform
        self._state_keys = state_keys
        self._state_key_dims = state_key_dims
        self._state_total_dim = sum(int(state_key_dims.get(key, 1)) for key in state_keys)
        logging.info(
            "PolicyServerWrapper: state normalizer ready (robot_type=%s, unnorm_key=%s, keys=%s, total_dim=%d)",
            robot_type,
            unnorm_key,
            state_keys,
            self._state_total_dim,
        )

    def _normalize_state(self, raw_state: Any) -> np.ndarray:
        arr = np.asarray(raw_state, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        total = self._state_total_dim
        if total <= 0:
            return arr
        if arr.shape[-1] < total:
            pad = np.zeros((*arr.shape[:-1], total - arr.shape[-1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=-1)
        elif arr.shape[-1] > total:
            arr = arr[..., :total]

        data: Dict[str, np.ndarray] = {}
        cursor = 0
        for key in self._state_keys:
            dim = int(self._state_key_dims.get(key, 1))
            data[key] = np.ascontiguousarray(arr[..., cursor : cursor + dim], dtype=np.float32)
            cursor += dim

        out = self._state_normalizer.apply(data)
        parts: List[np.ndarray] = []
        for key in self._state_keys:
            value = out[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            parts.append(np.asarray(value, dtype=np.float32))
        return np.concatenate(parts, axis=-1)

    def _prepare_examples(self, examples: List[dict]) -> List[dict]:
        prepared_examples = []
        for example in examples:
            if not isinstance(example, dict):
                prepared_examples.append(example)
                continue
            prepared = dict(example)
            if not self._expects_state:
                prepared.pop("state", None)
            elif self._state_normalizer is not None and prepared.get("state") is not None:
                prepared["state"] = self._normalize_state(prepared["state"])
            prepared_examples.append(prepared)
        return prepared_examples

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._ckpt_path, unnorm_key=unnorm_key
            )
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
            "expects_state": self._expects_state,
        }
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
        return base

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized.
        """
        examples = self._prepare_examples(examples)
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        out = self._framework.predict_action(examples=examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        return {"actions": unnorm}
