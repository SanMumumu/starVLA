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
  - chunk-cache scheduling (the execution horizon may be shorter than the
    model's predicted action chunk)

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

from starVLA.dataloader.action_correlation import validate_action_correlation_cholesky
from deployment.model_server.checkpoint_contract import (
    load_checkpoint_contract_config,
    resolve_config_expects_state,
)
from deployment.model_server.policy_norm_processor import (
    PolicyNormProcessor,
    _build_dataset_metadata,
    _infer_key_dims,
    _resolve_robot_type,
)
from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config


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

        # Read both snapshots before model placement. ``config.yaml`` remains
        # authoritative for construction; ``config.full.yaml`` supplies strict
        # runtime ABI fields such as state/image/correlated-noise settings.
        model_cfg, norm_stats = read_mode_config(self._ckpt_path)
        contract_cfg, contract_cfg_path = load_checkpoint_contract_config(
            self._ckpt_path,
            accessed_config=model_cfg,
        )

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path)

        # Correlated-noise is part of the checkpoint's inference distribution,
        # not an optional optimization artifact.  If the run declares it, fail
        # before serving rather than silently falling back to iid noise.
        contract_action_cfg = ((contract_cfg.get("framework") or {}).get("action_model") or {})
        accessed_action_cfg = ((model_cfg.get("framework") or {}).get("action_model") or {})
        action_contract = {**accessed_action_cfg, **contract_action_cfg}
        uses_correlated_noise = bool(action_contract.get("use_correlated_noise", False))
        chol_path = Path(self._ckpt_path).parents[1] / "action_correlation_cholesky.npy"
        if uses_correlated_noise:
            if not chol_path.is_file():
                raise FileNotFoundError(
                    "Checkpoint declares use_correlated_noise=true but the required run artifact is missing: "
                    f"{chol_path}. Sync action_correlation_cholesky.npy together with the checkpoint."
                )
            if not hasattr(framework, "set_action_correlation"):
                raise RuntimeError(
                    "Checkpoint declares use_correlated_noise=true but the loaded framework cannot inject its "
                    "Cholesky factor."
                )
            horizon = int(action_contract.get("action_horizon", 0))
            action_dim = int(action_contract.get("action_dim", 0))
            expected_size = horizon * action_dim
            if expected_size <= 0:
                raise ValueError(
                    "Invalid correlated-noise checkpoint contract: "
                    f"action_horizon={horizon}, action_dim={action_dim}."
                )
            try:
                cholesky = validate_action_correlation_cholesky(
                    np.load(chol_path, allow_pickle=False),
                    expected_size=expected_size,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid correlated-noise artifact {chol_path}: {exc}") from exc
            framework.set_action_correlation(cholesky)
            logging.info("PolicyServerWrapper: injected correlated-noise Cholesky from %s", chol_path)

        #######
        if dino_stats_path and hasattr(framework, "set_dino_stats"):
            framework.set_dino_stats(str(dino_stats_path))
            logging.info("PolicyServerWrapper: loaded online-DINO stats from %s", dino_stats_path)
        #######
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        # Co-located metadata.
        self._model_cfg = model_cfg
        self._contract_cfg_path = str(contract_cfg_path)
        contract_framework_cfg = (contract_cfg.get("framework") or {})
        accessed_framework_cfg = (model_cfg.get("framework") or {})
        self._framework_name = str(
            contract_framework_cfg.get("name", accessed_framework_cfg.get("name", ""))
        )
        self._wam_runtime_metadata = self._build_wam_runtime_metadata(
            framework,
            contract_cfg,
        )
        contract_vla_cfg = (contract_cfg.get("datasets") or {}).get("vla_data") or {}
        self._image_layout = str(contract_vla_cfg.get("image_layout", "separate_views"))
        self._composite_view_key = contract_vla_cfg.get("composite_view_key")
        self._composite_source_view_keys = list(contract_vla_cfg.get("composite_source_view_keys", []) or [])
        self._state_normalizer: Optional[ComposedModalityTransform] = None
        self._state_keys: List[str] = []
        self._state_key_dims: Dict[str, int] = {}
        self._state_total_dim = 0
        self._expects_state, self._state_contract_source = resolve_config_expects_state(contract_cfg)
        logging.info(
            "PolicyServerWrapper: expects_state=%s (%s; config=%s)",
            self._expects_state,
            self._state_contract_source,
            self._contract_cfg_path,
        )
        self._sync_framework_state_contract(framework)

        # action_chunk_size = future_action_window_size + 1 (matches old client).
        action_model_cfg = model_cfg["framework"]["action_model"]

        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                "PolicyServerWrapper: no action_horizon or future_action_window_size found "
                f"in model config for {self._ckpt_path}"
            )
        if self._framework_name == "QwenActionWorldCoFlow" and self._action_chunk_size != 16:
            raise ValueError(
                "PolicyServerWrapper: QwenActionWorldCoFlow is a single H16 bridge, "
                f"got action_chunk_size={self._action_chunk_size}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        self._available_unnorm_keys: List[str] = list(norm_stats.keys())

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
            self._build_state_normalizer(contract_cfg, norm_stats)

    @classmethod
    def _config_expects_state(cls, model_cfg: dict) -> bool:
        """Compatibility shim for callers/tests that only have a config dict."""

        return resolve_config_expects_state(model_cfg)[0]

    @staticmethod
    def _build_wam_runtime_metadata(framework: baseframework, contract_cfg: dict) -> Dict[str, Any]:
        """Describe the loaded WAM path, including the checkpoint's actual gates."""

        framework_cfg = (contract_cfg.get("framework") or {})
        trainer_cfg = (contract_cfg.get("trainer") or {})
        wam_cfg = (framework_cfg.get("wam") or {})
        saved_guidance = (wam_cfg.get("guidance") or {})
        runtime_guidance = getattr(framework, "wam_guidance", None)
        if not hasattr(runtime_guidance, "get"):
            runtime_guidance = saved_guidance

        wam_enabled = bool(getattr(framework, "wam_enabled", wam_cfg.get("enabled", False)))
        guidance_enabled = bool(runtime_guidance.get("enabled", False)) if wam_enabled else False
        phase = str(trainer_cfg.get("wam_two_stage_phase", "") or "").lower() or None
        recipe = str(
            trainer_cfg.get("wam_two_stage_recipe", "legacy_v1") or "legacy_v1"
        ).lower()
        pretraining_aligned_queries = bool(
            getattr(
                framework,
                "wam_pretraining_aligned_queries",
                runtime_guidance.get("pretraining_aligned_queries", False),
            )
        )
        action_query_bank = getattr(framework, "action_queries", None)
        action_query_count = (
            int(getattr(action_query_bank, "action_horizon"))
            if action_query_bank is not None
            and getattr(action_query_bank, "action_horizon", None) is not None
            else None
        )
        future_query_bank = getattr(framework, "future_dino_queries", None)
        future_query_capacity = (
            int(getattr(future_query_bank, "max_queries"))
            if future_query_bank is not None
            and getattr(future_query_bank, "max_queries", None) is not None
            else None
        )
        future_query_count = int(getattr(framework, "wam_n_flow", 0) or 0)
        qwen_interface = getattr(framework, "qwen_vl_interface", None)
        qwen_attn_implementation = (
            str(getattr(qwen_interface, "attn_implementation", "")).lower()
            if qwen_interface is not None
            else ""
        )
        if guidance_enabled and recipe in {
            "isolated_queries_v4",
            "causal_action_world_queries_v1",
        }:
            expected_action_queries = int(getattr(framework, "action_horizon", 0))
            if not pretraining_aligned_queries or action_query_count != expected_action_queries:
                raise RuntimeError(
                    f"{recipe} checkpoint did not restore its pretraining-aligned "
                    "ACT query bank: "
                    f"enabled={pretraining_aligned_queries}, count={action_query_count}, "
                    f"expected={expected_action_queries}"
                )
            if future_query_capacity is None:
                raise RuntimeError(
                    f"{recipe} checkpoint did not restore its FUTURE query bank"
                )
            if recipe == "causal_action_world_queries_v1" and not bool(
                runtime_guidance.get("future_query_through_qwen", False)
            ):
                raise RuntimeError(
                    "causal_action_world_queries_v1 checkpoint did not restore FUTURE-query Qwen injection"
                )
            if recipe == "causal_action_world_queries_v1":
                if action_query_count != 32 or int(getattr(framework, "action_horizon", 0)) != 32:
                    raise RuntimeError(
                        "causal_action_world_queries_v1 server requires ACT32/H32, got "
                        f"queries={action_query_count}, horizon={getattr(framework, 'action_horizon', None)}"
                    )
                if future_query_count != 64 or future_query_capacity != 64:
                    raise RuntimeError(
                        "causal_action_world_queries_v1 server requires FUTURE64, got "
                        f"active={future_query_count}, capacity={future_query_capacity}"
                    )
                if bool(runtime_guidance.get("separate_world_backbone_pass", False)):
                    raise RuntimeError(
                        "causal_action_world_queries_v1 checkpoint requests the obsolete two-pass Qwen path"
                    )
                if bool(runtime_guidance.get("detach_action_query_in_world_pass", False)):
                    raise RuntimeError(
                        "causal_action_world_queries_v1 checkpoint unexpectedly detaches ACT query from world loss"
                    )
                if qwen_attn_implementation != "flash_attention_2":
                    raise RuntimeError(
                        "causal_action_world_queries_v1 server requires active flash_attention_2, got "
                        f"{qwen_attn_implementation!r}"
                    )
        metadata: Dict[str, Any] = {
            "wam_enabled": wam_enabled,
            "wam_two_stage_phase": phase,
            # Standalone query-conditioned checkpoints reuse the same proven
            # causal-query recipe without declaring a warmup/gate phase.
            "wam_two_stage_recipe": recipe if guidance_enabled else None,
            "wam_guidance_enabled": guidance_enabled,
            "wam_action_world_bypass": (
                bool(runtime_guidance.get("action_world_bypass", False))
                if guidance_enabled
                else None
            ),
            "wam_world_to_action_enabled": (
                bool(runtime_guidance.get("world_to_action_enabled", True))
                if guidance_enabled
                else None
            ),
            "wam_bridge_source": (
                str(runtime_guidance.get("bridge_source", "")).lower()
                if guidance_enabled
                else None
            ),
            "wam_world_eval_mode": (
                str(runtime_guidance.get("world_eval_mode", "")).lower()
                if guidance_enabled
                else None
            ),
            "wam_detach_action_backbone": (
                bool(runtime_guidance.get("detach_action_backbone", False))
                if guidance_enabled
                else None
            ),
            "wam_detach_world_backbone": (
                bool(runtime_guidance.get("detach_world_backbone", False))
                if guidance_enabled
                else None
            ),
            "wam_baseline_action_context": (
                bool(runtime_guidance.get("baseline_action_context", False))
                if guidance_enabled
                else None
            ),
            "wam_pretraining_aligned_queries": (
                pretraining_aligned_queries if guidance_enabled else None
            ),
            "wam_future_query_through_qwen": (
                bool(runtime_guidance.get("future_query_through_qwen", False))
                if guidance_enabled
                else None
            ),
            "qwen_attn_implementation": qwen_attn_implementation or None,
            "wam_query_attention_pattern": (
                "causal_act_then_future"
                if guidance_enabled and recipe == "causal_action_world_queries_v1"
                else None
            ),
            "wam_single_qwen_forward": (
                not bool(runtime_guidance.get("separate_world_backbone_pass", False))
                if guidance_enabled and recipe == "causal_action_world_queries_v1"
                else None
            ),
            "wam_queries_are_final_suffix": (
                True
                if guidance_enabled and recipe == "causal_action_world_queries_v1"
                else None
            ),
            "wam_separate_world_backbone_pass": (
                bool(runtime_guidance.get("separate_world_backbone_pass", False))
                if guidance_enabled
                else None
            ),
            "wam_detach_action_query_in_world_pass": (
                bool(runtime_guidance.get("detach_action_query_in_world_pass", False))
                if guidance_enabled
                else None
            ),
            "wam_action_query_count": (
                action_query_count if guidance_enabled else None
            ),
            "wam_future_query_capacity": (
                future_query_capacity if guidance_enabled else None
            ),
            "wam_future_query_count": (
                future_query_count if guidance_enabled else None
            ),
            "wam_gate_openness": None,
            "wam_gate_signed_mean": None,
            "wam_gate_max_openness": None,
        }

        gate_reader = getattr(framework, "_wam_world_gate_metrics", None)
        mode = str(runtime_guidance.get("mode", "")).lower() if guidance_enabled else ""
        has_gated_path = bool(
            metadata["wam_world_to_action_enabled"]
            and mode in {"dual_xattn", "dual_xattn_adaln"}
        )
        if guidance_enabled and has_gated_path and callable(gate_reader):
            gate_metrics = gate_reader()

            def scalar(name: str) -> float:
                value = gate_metrics[name]
                if torch.is_tensor(value):
                    value = value.detach().float().cpu().item()
                return float(value)

            metadata.update(
                {
                    "wam_gate_openness": scalar("world_gate_openness"),
                    "wam_gate_signed_mean": scalar("world_gate_signed_mean"),
                    "wam_gate_max_openness": scalar("world_gate_max_openness"),
                }
            )
        elif phase == "gate_ft":
            raise RuntimeError(
                "Loaded gate_ft checkpoint does not expose the gated WAM action path; "
                "refusing to start an unverifiable policy server"
            )
        return metadata

    def _sync_framework_state_contract(self, framework: baseframework) -> None:
        """Make the loaded model consume exactly the checkpoint-declared state ABI.

        Model construction intentionally uses the compact accessed-only config,
        while the complete checkpoint contract may come from ``config.full.yaml``.
        WAM checks ``framework.config`` again during every forward, so an old
        compact config that omitted ``include_state`` must be synchronized after
        construction. This changes only runtime input routing, not model shape.
        """

        source = self._state_contract_source
        if not source.startswith("datasets.vla_data.include_state"):
            return
        uses_action_state = getattr(framework, "_uses_action_state", None)
        if not callable(uses_action_state) or bool(uses_action_state()) == self._expects_state:
            return

        datasets_cfg = getattr(getattr(framework, "config", None), "datasets", None)
        vla_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
        if vla_cfg is None:
            raise RuntimeError(
                "PolicyServerWrapper: checkpoint declares a FastWAM state ABI, but the loaded framework has no "
                "datasets.vla_data config to enforce it"
            )
        try:
            vla_cfg["include_state"] = self._expects_state
        except (TypeError, KeyError):
            setattr(vla_cfg, "include_state", self._expects_state)
        if bool(uses_action_state()) != self._expects_state:
            raise RuntimeError("PolicyServerWrapper: failed to synchronize the framework's include_state contract")
        logging.info(
            "PolicyServerWrapper: synchronized framework include_state=%s from %s",
            self._expects_state,
            self._contract_cfg_path,
        )

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
        if arr.shape[-1] != total:
            raise ValueError(
                f"PolicyServerWrapper: checkpoint expects a {total}-D state, got shape {arr.shape}. "
                "Refusing to pad or truncate state because that would break train/inference consistency."
            )

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
            else:
                if prepared.get("state") is None:
                    raise ValueError(
                        "PolicyServerWrapper: this checkpoint was trained with include_state=true, "
                        "but the inference request did not provide state."
                    )
                if self._state_normalizer is None:
                    raise RuntimeError(
                        "PolicyServerWrapper: this checkpoint requires state, but its training-time "
                        "state normalizer could not be constructed."
                    )
                prepared["state"] = self._normalize_state(prepared["state"])
            prepared_examples.append(prepared)
        return prepared_examples

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(self._ckpt_path, unnorm_key=unnorm_key)
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
            "state_contract_source": self._state_contract_source,
            "contract_config": self._contract_cfg_path,
            "framework_name": self._framework_name,
            "coflow_supported_inference_modes": (
                ["policy", "diagonal"] if self._framework_name == "QwenActionWorldCoFlow" else []
            ),
            "coflow_bridge_horizon": (
                16 if self._framework_name == "QwenActionWorldCoFlow" else None
            ),
            "image_layout": self._image_layout,
            "composite_view_key": self._composite_view_key,
            "composite_source_view_keys": self._composite_source_view_keys,
        }
        base.update(self._wam_runtime_metadata)
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
