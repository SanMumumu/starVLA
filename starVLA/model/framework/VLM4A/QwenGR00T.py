# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""

import sys
from pathlib import Path
#######
# 中文注释：JointFlow 分支需要在局部禁用 autocast，并为 CED unused-parameter anchor 遍历模块参数。
from contextlib import nullcontext
import json
#######

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
#######
# 中文注释：JointFlow 迁移到 QwenGR00T 后会新增若干可训练子模块，统一使用 nn.Module 类型标注。
from torch import nn
#######
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images
#######
# 中文注释：复用 JointFlow 已验证的 DINO token、query token、visual flow head 和 CED probe 小模块；
# 这些模块只在 framework.jointflow.enabled=true 时实例化，默认不影响原生 QwenGR00T。
from starVLA.model.framework.VLM4A.jointflow.attention_mask import build_block_causal_mask
from starVLA.model.framework.VLM4A.jointflow.ced_probe import CedProbeMLP, RepaProjector
from starVLA.model.framework.VLM4A.jointflow.dino_v3 import DINOv3Backbone, resolve_dino_spec
from starVLA.model.framework.VLM4A.jointflow.joint_modules import (
    ActionContextEncoder,
    ActionQueryTokenBank,
    DinoProjector,
    FutureDinoQueryTokenBank,
)
from starVLA.model.framework.VLM4A.jointflow.visual_dino_flow_head import VisualFlowMatchingHead
#######


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenGR00T
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenGR00TDefaultConfig:
    """QwenGR00T framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier ---
    name: str = "QwenGR00T"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (used for cross-attention alignment)
            "vl_hidden_dim": 2048,
        }
    )

    # # === DINO encoder (optional multi-view spatial tokens) === Dino is not used in this QwenGR00T version, we can add it later when we want to use it
    # dino: dict = field(default_factory=lambda: {
    #     # DINO backbone variant: "dinov2_vits14" | "dinov2_vitb14" | ...
    #     "dino_backbone": "dinov2_vits14",
    # })

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            # DiT model size: "DiT-B" | "DiT-L" | "DiT-XL"
            "action_model_type": "DiT-B",
            # Hidden dim for action model (auto-aligned at runtime)
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            # Whether to add positional embeddings in the action head
            "add_pos_embed": True,
            "max_seq_len": 1024,
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 8,
            # Repeat factor for flow-matching loss (more noise samples per batch)
            "repeated_diffusion_steps": 8,
            # Beta distribution params for noise schedule
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # Inference denoising steps
            "num_inference_timesteps": 4,
            # Number of vision tokens fed to action head
            "num_target_vision_tokens": 32,
            # === DiT Transformer sub-config ===
            "diffusion_model_cfg": {
                # Cross-attention dim (aligned to VLM hidden_size at runtime)
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    #######
    # 中文注释：JointFlow 迁移开关与默认配置。enabled 默认关闭，保证旧 QwenGR00T 配置、
    # checkpoint、训练和推理路径不实例化任何新增模块，也不改变原有 action_loss 语义。
    jointflow: dict = field(default_factory=lambda: {"enabled": False})

    # 中文注释：JointFlow future-DINO 条件分支配置；启用后用于在线/离线 DINO latent 处理。
    dino: dict = field(
        default_factory=lambda: {
            "name": "dinov3_vits16",
            "hf_model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "repo_or_dir": "facebookresearch/dinov3",
            "weights": None,
            "loader": "auto",
            "image_size": 224,
            "patch_size": 16,
            "embed_dim": 384,
            "future_view_keys": ["video.primary_image", "primary_image", "agentview"],
            "stats_path": None,
            "load_live_backbone": False,
            "dino_pool": 1,
        }
    )

    # 中文注释：state 与原生 gr00t 完全一致——走 action head 内部 state_encoder（state_dim 在 action_model 里），
    # 由 datasets.vla_data.include_state 控制是否真喂（原生 LIBERO recipe 默认 false，即声明但惰性、实际不用）。
    # 不再有 jointflow 特有的 state-token 机制。

    # 中文注释：JointFlow 多任务采样与 block attention mask 配置。
    tasks: dict = field(
        default_factory=lambda: {
            "weights": {"policy": 1.0},
            "hybrid_mask": True,
            "attention_mask_neg_value": -1.0e4,
        }
    )

    # 中文注释：future DINO flow-matching head 配置，fdm/passive/effect 任务使用。
    visual_model: dict = field(
        default_factory=lambda: {
            "d_dino": 384,
            "n_query": 196,
            "max_image_queries": 256,
            "hidden_size": 768,
            "cross_attention_dim": 2048,
            "num_attention_heads": 12,
            "attention_head_dim": 64,
            "num_layers": 8,
            "dropout": 0.1,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "patch_weighting": "none",
        }
    )

    # 中文注释：CED 默认关闭；打开后构造 no-op action / delta effect / ident / align 相关模块和 loss。
    ced: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "tau": 1.0,
            "eps_gate": 1.0e-3,
            "probe_hidden": 512,
            "probe_val_mod": 10,
        }
    )

    # 中文注释：CED loss 权重。lam_i 控制 loss_ident，lam_a 控制 loss_align。
    losses: dict = field(
        default_factory=lambda: {
            "lam_i": 0.1,
            "lam_a": 0.0,
            "teacher": "delta",
            "effect_delta_prob": 1.0,
        }
    )
    #######

    # # === Training precision flag === This is unnecessary, unused parameter
    # reduce_in_full_precision: bool = True


@FRAMEWORK_REGISTRY.register("QwenGR00T")
class Qwen_GR00T(baseframework):
    """
    Multimodal vision-language-action model (GR00T variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Flow-matching (DiT) diffusion head for continuous action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenGR00TDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
        #######
        # 中文注释：JointFlow 分支显式开关。默认 false，保持原生 QwenGR00T 完全不变。
        # state 与原生 gr00t 一致——走 action head 的 state_encoder（state_dim 由 action_model 配置，
        # include_state=false 时惰性不用），不再强制 state_dim=0、不再用 jointflow state-token。
        self.jointflow_enabled = bool(self.config.framework.get("jointflow", {}).get("enabled", False))
        if self.jointflow_enabled:
            self.config.framework.visual_model.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size
        #######

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        #######
        # 中文注释：JointFlow 相关模块只在开关打开时构造，避免旧 checkpoint strict load 出现新增参数。
        if self.jointflow_enabled:
            self._init_jointflow_modules()
        #######

    #######
    # 中文注释：初始化 JointFlow 迁移模块。这里复用原生 Qwen3-VL 的 language_model 作为 joint sequence
    # backbone，复用原生 GR00T action_model 作为 policy/idm 动作流头，只新增 DINO/FDM/CED 必需模块。
    def _init_jointflow_modules(self) -> None:
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        dino_cfg = self.config.framework.dino
        visual_cfg = self.config.framework.visual_model
        action_cfg = self.config.framework.action_model

        self.action_dim = int(action_cfg.action_dim)
        self._jointflow_dino_spec = resolve_dino_spec(dino_cfg)
        self.d_dino = int(self._jointflow_dino_spec["embed_dim"])
        self.config.framework.visual_model.d_dino = self.d_dino

        self.dino_proj = DinoProjector(d_dino=self.d_dino, hidden_size=hidden_size)
        self.act_ctx = ActionContextEncoder(action_dim=self.action_dim, hidden_size=hidden_size)
        #######
        # 中文注释：action query token 数（conditioning 容量）与 chunk 长度解耦——n_action_query 默认=action_horizon
        # （不改默认行为）；调大只增加喂给 action DiT cross-attn 的 cond token 数（DiT 是 cross-attn，context 长度可变），
        # 不改 chunk/数据/eval。E3.x 用它做 sweep（8/16/32/64/128）。
        n_action_query = int(action_cfg.get("n_action_query", self.action_horizon))
        self.action_queries = ActionQueryTokenBank(action_horizon=n_action_query, hidden_size=hidden_size)
        #######
        self.future_dino_queries = FutureDinoQueryTokenBank(
            max_queries=int(visual_cfg.get("max_image_queries", visual_cfg.get("n_query", 196))),
            hidden_size=hidden_size,
        )
        self.visual_head = VisualFlowMatchingHead(self.config)

        ced_cfg = self.config.framework.get("ced", {})
        self.ced_enabled = bool(ced_cfg.get("enabled", False))
        self.probe_mlp = None
        self.repa_proj = None
        #######
        # 中文注释：REPA 对齐——repa_layer 取动作专家 DiT 的早期层（默认 4，≈12 层的 28%，对齐 REPA 配方）。
        self.repa_layer = int(ced_cfg.get("repa_layer", 4))
        #######
        if self.ced_enabled:
            self.probe_mlp = CedProbeMLP(
                d_in=self.d_dino,
                hidden=int(ced_cfg.get("probe_hidden", 512)),
                d_out=self.action_horizon * self.action_dim,
            )
            #######
            # 中文注释：REPA 投影头输入维 = 动作专家 DiT 的 inner_dim；把 DiT 早期层池化态投到 Δ 空间，
            # 与 detach 的 teacher 做 cosine（对齐 DiT 早期层，而非原先的 Qwen query 特征）。
            dit_inner = int(self.action_model.model.inner_dim)
            self.repa_proj = RepaProjector(
                d_in=dit_inner,
                d_out=self.d_dino,
                hidden=int(ced_cfg.get("repa_proj_hidden", 2048)),
            )
            #######

        self.dino = None
        if bool(dino_cfg.get("load_live_backbone", False)):
            self.dino = DINOv3Backbone(**self._jointflow_dino_spec)
        self._null_action_table: dict[str, torch.Tensor] = {}
        self.register_buffer("_dino_mean", torch.zeros(self.d_dino), persistent=False)
        self.register_buffer("_dino_std", torch.ones(self.d_dino), persistent=False)
        self._load_jointflow_dino_stats(dino_cfg.get("stats_path", None))
    #######

    #######
    # 中文注释：以下方法是从 QwenJointFlow 迁移到 QwenGR00T 的核心训练逻辑。
    # 迁移时替换掉独立 text-only Qwen2 wrapper，改用原生 Qwen3-VL/Qwen-VL 的 language_model，
    # 从而保留 QwenGR00T 的模型注册、checkpoint、trainer、policy server 等工程结构。
    def _jointflow_language_model(self):
        model_root = self.qwen_vl_interface.model.model
        return getattr(model_root, "language_model", model_root)

    def _jointflow_embed_tokens(self):
        language_model = self._jointflow_language_model()
        #######
        # 中文注释：不能把 get_input_embeddings() 放进 getattr 默认值；
        # Python 会提前求值，部分 Qwen wrapper/fake smoke 对象没有该方法时会误报。
        if hasattr(language_model, "embed_tokens"):
            return language_model.embed_tokens
        return self.qwen_vl_interface.model.get_input_embeddings()
        #######

    @staticmethod
    def _zero_grad_anchor_for_modules(modules: list[nn.Module | None], ref: torch.Tensor) -> torch.Tensor:
        anchor = ref.new_zeros(())
        seen: set[int] = set()
        for module in modules:
            if module is None:
                continue
            for param in module.parameters(recurse=True):
                if not param.requires_grad or param.numel() == 0:
                    continue
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                anchor = anchor + param.reshape(-1)[0].to(dtype=anchor.dtype) * 0.0
        return anchor

    def _unused_jointflow_param_anchor(self, task: str, ref: torch.Tensor) -> torch.Tensor:
        modules: list[nn.Module | None] = []
        if task in {"policy", "idm"}:
            modules.extend([self.visual_head, self.future_dino_queries, self.act_ctx, self.probe_mlp, self.repa_proj])
        elif task == "fdm":
            modules.extend([self.action_model, self.action_queries, self.probe_mlp, self.repa_proj])
        elif task == "passive":
            modules.extend([self.action_model, self.action_queries, self.act_ctx, self.probe_mlp, self.repa_proj])
        elif task == "effect":
            pass
        else:
            raise ValueError(f"Unsupported JointFlow task `{task}`")
        return self._zero_grad_anchor_for_modules(modules, ref)

    def _all_jointflow_trainable_anchor(self, ref: torch.Tensor) -> torch.Tensor:
        modules = [
            self.qwen_vl_interface,
            self.dino_proj,
            self.act_ctx,
            self.action_queries,
            self.future_dino_queries,
            self.action_model,
            self.visual_head,
            self.probe_mlp,
            self.repa_proj,
        ]
        return self._zero_grad_anchor_for_modules(modules, ref)

    @staticmethod
    def _jointflow_module_dtype(module: nn.Module, fallback: torch.dtype = torch.float32) -> torch.dtype:
        #######
        # 中文注释：DeepSpeed bf16 会把 head 参数转成 bf16；head 输入必须跟随参数 dtype，
        # 否则禁用 autocast 的局部 fp32 分支会在 Linear 上触发 Float/BFloat16 mismatch。
        for param in module.parameters(recurse=True):
            return param.dtype
        return fallback
        #######

    def _load_jointflow_dino_stats(self, stats_path: str | None) -> None:
        if not stats_path:
            return
        path = Path(stats_path)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32).clamp_min(1e-6)
        if mean.numel() != self.d_dino or std.numel() != self.d_dino:
            logger.warning(
                f"JointFlow DINO stats dim {mean.numel()} != embed_dim {self.d_dino}; using identity stats."
            )
            return
        self._dino_mean.copy_(mean)
        self._dino_std.copy_(std)

    def _format_jointflow_instruction(self, instruction: str) -> str:
        data_cfg = getattr(getattr(self.config, "datasets", None), "vla_data", None)
        prompt = data_cfg.get("CoT_prompt", None) if data_cfg is not None and hasattr(data_cfg, "get") else None
        return prompt.replace("{instruction}", instruction) if prompt else instruction

    def _embed_jointflow_text(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        max_text_length = int(self.config.framework.qwenvl.get("max_text_length", 256))
        #######
        # 中文注释：JointFlow 的 block mask 用 text_valid_lens 表示“前 N 个文本 token 有效”，
        # 因此这里必须使用右 padding；原生 Qwen3-VL 推理常用 left padding，不能直接沿用。
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        try:
            encoded = tokenizer(
                [self._format_jointflow_instruction(text) for text in instructions],
                padding=True,
                truncation=True,
                max_length=max_text_length,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = old_padding_side
        #######
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device, dtype=torch.bool)
        embeds = self._jointflow_embed_tokens()(input_ids)
        valid_lens = attention_mask.long().sum(dim=1)
        return embeds, valid_lens

    def _run_jointflow_backbone(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask_4d: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        language_model = self._jointflow_language_model()
        outputs = language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_4d.to(dtype=inputs_embeds.dtype),
            position_ids=position_ids,
            use_cache=False,
        )
        return outputs.last_hidden_state

    def _stack_jointflow_field(self, examples: List[dict], key: str, required: bool = True) -> torch.Tensor | None:
        vals = [ex.get(key, None) for ex in examples]
        if vals[0] is None:
            if required:
                raise KeyError(f"Missing required field `{key}` in JointFlow batch.")
            return None
        arrays = []
        for val in vals:
            if torch.is_tensor(val):
                arrays.append(val.detach().cpu().numpy())
            else:
                arrays.append(np.asarray(val))
        return torch.tensor(np.stack(arrays), device=self.device, dtype=torch.float32)

    def _get_jointflow_view_keys(self, examples: List[dict], num_views: int) -> list[str]:
        keys = examples[0].get("dino_view_keys") or examples[0].get("view_keys")
        if keys:
            return [str(k) for k in keys]
        return [f"view_{i}" for i in range(num_views)]

    def _select_jointflow_future_dino(self, dino_1: torch.Tensor, examples: List[dict]) -> torch.Tensor:
        if dino_1.ndim == 3:
            return dino_1
        if dino_1.ndim != 4:
            raise ValueError(f"Expected dino_1 [B,V,N,D] or [B,N,D], got {tuple(dino_1.shape)}")
        view_keys = self._get_jointflow_view_keys(examples, dino_1.shape[1])
        preferred = list(self.config.framework.dino.get("future_view_keys", []))
        chosen = 0
        for name in preferred:
            if name in view_keys:
                chosen = view_keys.index(name)
                break
        return dino_1[:, chosen]

    def _flatten_jointflow_dino(self, dino: torch.Tensor) -> torch.Tensor:
        if dino.ndim == 4:
            bsz, views, n_tokens, dim = dino.shape
            dino = dino.reshape(bsz, views * n_tokens, dim)
        elif dino.ndim != 3:
            raise ValueError(f"Expected DINO features [B,V,N,D] or [B,N,D], got {tuple(dino.shape)}")
        pool = int(self.config.framework.dino.get("dino_pool", 1))
        if pool > 1:
            n_tokens = dino.shape[1]
            side = int(n_tokens**0.5)
            if side * side == n_tokens and side % pool == 0:
                dino = dino.view(dino.shape[0], side, side, dino.shape[-1])
                dino = dino.view(dino.shape[0], side // pool, pool, side // pool, pool, dino.shape[-1]).mean(dim=(2, 4))
                dino = dino.reshape(dino.shape[0], -1, dino.shape[-1])
        return dino

    def _ensure_jointflow_dino(self) -> DINOv3Backbone:
        if self.dino is None:
            self.dino = DINOv3Backbone(**self._jointflow_dino_spec).to(self.device)
        return self.dino

    def _collect_jointflow_images(self, examples: List[dict], keys: list[str]):
        batch_views = []
        for ex in examples:
            val = None
            for key in keys:
                cand = ex.get(key, None)
                if cand is not None:
                    val = cand
                    break
            if val is None:
                return None
            if isinstance(val, Image.Image):
                views = [val]
            elif isinstance(val, np.ndarray):
                if val.ndim == 4:
                    views = [val[i] for i in range(val.shape[0])]
                elif val.ndim == 3:
                    views = [val]
                else:
                    raise ValueError(f"Unsupported image array ndim={val.ndim} for keys={keys}")
            elif isinstance(val, (list, tuple)):
                views = [to_pil_preserve(img) for img in val]
            else:
                views = [to_pil_preserve(val)]
            batch_views.append(views)
        return batch_views

    def _run_jointflow_dino_on_images(
        self,
        examples: List[dict],
        keys: list[str],
        required: bool,
    ) -> torch.Tensor | None:
        batch_views = self._collect_jointflow_images(examples, keys)
        if batch_views is None:
            if required:
                raise KeyError(f"Online DINO needs one of {keys} in the batch.")
            return None
        dino = self._ensure_jointflow_dino()
        n_views = len(batch_views[0])
        flat = [img for views in batch_views for img in views]
        device_type = self.device.type
        autocast_ctx = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in {"cuda", "cpu"}
            else nullcontext()
        )
        with autocast_ctx:
            tensor = dino.preprocess_batch(flat)
            feats = dino(tensor)
        feats = feats.view(len(batch_views), n_views, feats.shape[1], feats.shape[2]).float()
        return (feats - self._dino_mean.to(feats.device, feats.dtype)) / self._dino_std.to(feats.device, feats.dtype)

    def _jointflow_examples_to_batch(
        self,
        examples: List[dict],
        require_future_dino: bool,
        require_action: bool,
    ) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        dino_0 = self._stack_jointflow_field(examples, "dino_0", required=False)
        if dino_0 is None:
            dino_0 = self._run_jointflow_dino_on_images(examples, ["image_0", "image"], required=True)
        dino_1 = self._stack_jointflow_field(examples, "dino_1", required=False)
        if dino_1 is None and require_future_dino:
            dino_1 = self._run_jointflow_dino_on_images(examples, ["image_1"], required=True)
        batch = {
            "examples": examples,
            "instructions": [str(ex.get("lang", "")) for ex in examples],
            "dino_0": dino_0,
            "dino_1": dino_1,
            "action": self._stack_jointflow_field(examples, "action", required=require_action),
            "state": self._stack_jointflow_field(examples, "state", required=False),
            "future_valid": self._stack_jointflow_field(examples, "future_valid", required=False),
        }
        if batch["future_valid"] is None and require_future_dino:
            batch["future_valid"] = torch.ones(len(examples), device=self.device, dtype=torch.float32)
        return batch

    def set_null_action_table(self, table: dict) -> None:
        self._null_action_table = {
            str(name): torch.as_tensor(np.asarray(vec), dtype=torch.float32) for name, vec in table.items()
        }

    #######
    # 中文注释：E1.3 correlated noise——把 trainer 算好的 Σ-Cholesky 透传给 action head。
    def set_action_correlation(self, chol) -> None:
        self.action_model.set_action_correlation(chol)
    #######

    def make_null_action(self, action: torch.Tensor, examples: List[dict]) -> torch.Tensor:
        bsz, horizon, dim = action.shape
        bases = []
        for ex in examples:
            name = str(ex.get("dataset_name", ""))
            if name not in self._null_action_table:
                raise KeyError(
                    f"[CED] dataset `{name}` missing from null-action table "
                    f"(have: {sorted(self._null_action_table)})."
                )
            base = self._null_action_table[name]
            if base.shape[0] != dim:
                raise ValueError(f"[CED] null-action dim mismatch for `{name}`: table {base.shape[0]} vs action {dim}")
            bases.append(base)
        bases = torch.stack(bases).to(device=action.device, dtype=action.dtype)
        nan_mask = torch.isnan(bases).unsqueeze(1).expand(bsz, horizon, dim)
        null = bases.unsqueeze(1).expand(bsz, horizon, dim).clone()
        first = action[:, :1, :].expand(bsz, horizon, dim)
        null[nan_mask] = first[nan_mask]
        return null

    @staticmethod
    def _jointflow_change_weights(z_t: torch.Tensor, z_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            w = (z_h.float() - z_t.float()).norm(dim=-1)
            w = w / w.mean(dim=1, keepdim=True).clamp_min(1e-8)
            clamped = w.clamp(0.1, 10.0)
            clamp_frac = (w != clamped).float().mean()
        return clamped, clamp_frac

    @staticmethod
    def _jointflow_fdm_split_metrics(per_patch: torch.Tensor, z_t: torch.Tensor, z_h: torch.Tensor) -> dict:
        with torch.no_grad():
            change = (z_h.float() - z_t.float()).norm(dim=-1)
            k = max(1, int(round(0.1 * change.shape[1])))
            thresh = change.topk(k, dim=1).values[:, -1:]
            dyn_mask = change >= thresh
            out = {"loss/fdm_dynamic": per_patch[dyn_mask].mean().detach()}
            out["loss/fdm_static"] = (
                per_patch[~dyn_mask].mean().detach() if bool((~dyn_mask).any()) else per_patch.detach().mean() * 0.0
            )
        return out

    def _jointflow_future_valid_mask(self, batch: dict, cond: torch.Tensor) -> torch.Tensor:
        valid = batch.get("future_valid", None)
        if valid is None:
            return torch.ones(cond.shape[0], device=cond.device, dtype=torch.bool)
        valid = valid.to(device=cond.device)
        return valid.reshape(valid.shape[0], -1)[:, 0] > 0.5

    def _assemble_jointflow_sequence(self, task: str, batch: dict):
        examples = batch["examples"]
        text_embeds, text_valid_lens = self._embed_jointflow_text(batch["instructions"])
        target_dtype = text_embeds.dtype

        blocks: list[torch.Tensor] = [text_embeds]
        block_sizes: list[int] = [text_embeds.shape[1]]
        #######
        # 中文注释：序列里不放 state token（state 与源一致走 action head）。块顺序：text → img0 → [action_ctx|img1] → query。
        #######

        #######
        # 中文注释：DINO projector 在 bf16 server 下参数 dtype 会变化，输入按参数 dtype 投影后再对齐 Qwen。
        dino_proj_dtype = self._jointflow_module_dtype(self.dino_proj, fallback=target_dtype)
        dino0 = self._flatten_jointflow_dino(batch["dino_0"]).to(text_embeds.device, dtype=dino_proj_dtype)
        img0_tokens = self.dino_proj(dino0).to(dtype=target_dtype)
        #######
        blocks.append(img0_tokens)
        block_sizes.append(img0_tokens.shape[1])

        action = batch["action"]
        if task == "fdm":
            if action is None:
                raise KeyError("fdm requires `action` context.")
            #######
            # 中文注释：FDM action context 编码器同样按自身参数 dtype 接收 action，兼容 bf16 推理/训练包装。
            act_ctx_dtype = self._jointflow_module_dtype(self.act_ctx, fallback=target_dtype)
            action_ctx = self.act_ctx(
                action[:, -self.action_horizon :, : self.action_dim].to(
                    device=text_embeds.device,
                    dtype=act_ctx_dtype,
                )
            ).to(dtype=target_dtype)
            #######
            blocks.append(action_ctx)
            block_sizes.append(action_ctx.shape[1])

        if task == "idm":
            if batch["dino_1"] is None:
                raise KeyError("idm requires `dino_1`.")
            dino1_ctx = self._flatten_jointflow_dino(batch["dino_1"]).to(text_embeds.device, dtype=dino_proj_dtype)
            img1_tokens = self.dino_proj(dino1_ctx).to(dtype=target_dtype)
            blocks.append(img1_tokens)
            block_sizes.append(img1_tokens.shape[1])

        query_start = sum(block_sizes)
        if task in {"policy", "idm"}:
            query = self.action_queries(text_embeds.shape[0], device=text_embeds.device).to(dtype=target_dtype)
        elif task in {"fdm", "passive"}:
            if batch["dino_1"] is None:
                raise KeyError(f"{task} requires `dino_1`.")
            target = self._select_jointflow_future_dino(batch["dino_1"], examples)
            query = self.future_dino_queries(
                text_embeds.shape[0],
                n_query=int(target.shape[1]),
                device=text_embeds.device,
            ).to(dtype=target_dtype)
        else:
            raise ValueError(f"Unsupported JointFlow task `{task}`")

        blocks.append(query)
        block_sizes.append(query.shape[1])
        query_slice = slice(query_start, query_start + query.shape[1])

        inputs_embeds = torch.cat(blocks, dim=1)
        position_ids = (
            torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            .unsqueeze(0)
            .expand(inputs_embeds.shape[0], -1)
        )
        attn4d = build_block_causal_mask(
            block_sizes=block_sizes,
            text_valid_lens=text_valid_lens,
            dtype=torch.float32,
            device=inputs_embeds.device,
            hybrid=bool(self.config.framework.tasks.get("hybrid_mask", True)),
            neg_value=float(self.config.framework.tasks.get("attention_mask_neg_value", -1.0e4)),
        )
        return inputs_embeds, attn4d, position_ids, query_slice

    def _run_jointflow_path(self, task: str, batch: dict) -> torch.Tensor:
        inputs_embeds, attn4d, position_ids, query_slice = self._assemble_jointflow_sequence(task, batch)
        hidden = self._run_jointflow_backbone(
            inputs_embeds=inputs_embeds,
            attention_mask_4d=attn4d,
            position_ids=position_ids,
        )
        return hidden[:, query_slice].float()

    def _jointflow_forward(self, examples: List[dict], task: str = "policy") -> dict:
        if task == "effect":
            return self._jointflow_effect_forward(examples)
        batch = self._jointflow_examples_to_batch(
            examples,
            require_future_dino=task in {"fdm", "idm", "passive"},
            require_action=task in {"policy", "fdm", "idm"},
        )
        cond = self._run_jointflow_path(task, batch)
        extras: dict = {}
        if task in {"policy", "idm"}:
            actions = batch["action"]
            if actions is None:
                raise KeyError(f"{task} requires `action` labels.")
            target = actions[:, -self.action_horizon :, : self.action_dim].float()
            if task == "idm":
                valid_mask = self._jointflow_future_valid_mask(batch, cond)
                if not bool(valid_mask.any()):
                    loss = cond.sum() * 0.0
                    return {f"{task}_loss": loss + self._all_jointflow_trainable_anchor(loss)}
                cond = cond[valid_mask]
                target = target[valid_mask]
            device_type = cond.device.type
            ac = (
                torch.autocast(device_type=device_type, enabled=False)
                if device_type in {"cuda", "cpu"}
                else nullcontext()
            )
            with ac:
                head_dtype = self._jointflow_module_dtype(self.action_model)
                loss = self.action_model(
                    cond.to(dtype=head_dtype),
                    target.to(dtype=head_dtype),
                    state=batch.get("state"),  # 中文注释：与源一致，state 透传 action head；include_state=false 时为 None（惰性）
                    encoder_attention_mask=None,
                )
        else:
            target = self._select_jointflow_future_dino(batch["dino_1"], batch["examples"]).to(
                cond.device,
                dtype=torch.float32,
            )
            valid_mask = self._jointflow_future_valid_mask(batch, cond)
            if not bool(valid_mask.any()):
                loss = cond.sum() * 0.0
                return {f"{task}_loss": loss + self._all_jointflow_trainable_anchor(loss)}
            cond = cond[valid_mask]
            target = target[valid_mask]
            if str(self.config.framework.visual_model.get("patch_weighting", "none")) == "change":
                z_t = self._select_jointflow_future_dino(batch["dino_0"], batch["examples"]).to(
                    target.device,
                    torch.float32,
                )
                z_t = z_t[valid_mask]
                weights, clamp_frac = self._jointflow_change_weights(z_t, target)
                loss, _, per_patch = self.visual_head(cond, target, weights=weights, return_pred=True)
                extras.update(self._jointflow_fdm_split_metrics(per_patch, z_t, target))
                extras["stat/w_clamp_frac"] = clamp_frac.detach()
            else:
                loss = self.visual_head(cond, target)
        out = {f"{task}_loss": loss + self._unused_jointflow_param_anchor(task, loss)}
        out.update(extras)
        return out

    def _jointflow_effect_forward(self, examples: List[dict]) -> dict:
        if not self.ced_enabled:
            raise ValueError("task=effect requires framework.ced.enabled=true")
        batch = self._jointflow_examples_to_batch(examples, require_future_dino=True, require_action=True)
        fw = self.config.framework
        ced_cfg = fw.ced
        losses_cfg = fw.get("losses", {})
        lam_i = float(losses_cfg.get("lam_i", 0.1))
        lam_a = float(losses_cfg.get("lam_a", 0.0))
        teacher = str(losses_cfg.get("teacher", "delta"))
        delta_prob = float(losses_cfg.get("effect_delta_prob", 1.0))
        tau = float(ced_cfg.get("tau", 1.0))
        eps_gate = float(ced_cfg.get("eps_gate", 1.0e-3))
        val_mod = int(ced_cfg.get("probe_val_mod", 10))

        h_act = self._run_jointflow_path("policy", batch)
        target_action = batch["action"][:, -self.action_horizon :, : self.action_dim].float()
        device_type = h_act.device.type
        ac = torch.autocast(device_type=device_type, enabled=False) if device_type in {"cuda", "cpu"} else nullcontext()
        with ac:
            head_dtype = self._jointflow_module_dtype(self.action_model)
            #######
            # 中文注释：CED 用 REPA 对齐——loss_act 这次前向顺便取回 DiT 第 repa_layer 层隐藏态做 align 学生特征，
            # 复用同一次 DiT 前向，不额外多跑。
            loss_act, repa_feat = self.action_model(
                h_act.to(dtype=head_dtype),
                target_action.to(dtype=head_dtype),
                state=batch.get("state"),  # 中文注释：与源一致；include_state=false 时为 None（惰性）
                encoder_attention_mask=None,
                return_repa_features=True,
                repa_layer=self.repa_layer,
            )
            #######
        extras: dict = {"loss/act": loss_act.detach(), "loss_act": loss_act.detach()}

        valid_mask = self._jointflow_future_valid_mask(batch, h_act)
        if not bool(valid_mask.any()):
            total = loss_act
            out = {"effect_loss": total + self._all_jointflow_trainable_anchor(total)}
            out.update(extras)
            return out

        cond_pos = self._run_jointflow_path("fdm", batch)
        batch_nul = dict(batch)
        batch_nul["action"] = self.make_null_action(batch["action"], batch["examples"])
        cond_nul = self._run_jointflow_path("fdm", batch_nul)

        z_h = self._select_jointflow_future_dino(batch["dino_1"], batch["examples"]).to(cond_pos.device, torch.float32)
        z_t = self._select_jointflow_future_dino(batch["dino_0"], batch["examples"]).to(cond_pos.device, torch.float32)
        cond_pos_v = cond_pos[valid_mask]
        cond_nul_v = cond_nul[valid_mask]
        z_h_v = z_h[valid_mask]
        z_t_v = z_t[valid_mask]
        #######
        # 中文注释：REPA 学生特征取 valid 样本的 DiT 早期层隐藏态（取代原 Qwen policy query 特征 h_act_v）。
        repa_feat_v = repa_feat[valid_mask]
        #######
        a_v = target_action[valid_mask]
        examples_v = [ex for ex, keep in zip(batch["examples"], valid_mask.tolist()) if keep]

        weights = None
        if str(fw.visual_model.get("patch_weighting", "none")) == "change":
            weights, clamp_frac = self._jointflow_change_weights(z_t_v, z_h_v)
            extras["stat/w_clamp_frac"] = clamp_frac.detach()
        loss_fdm, _, per_patch = self.visual_head(cond_pos_v, z_h_v, weights=weights, return_pred=True)
        extras["loss/fdm"] = loss_fdm.detach()
        extras["loss_fdm"] = loss_fdm.detach()
        extras.update(self._jointflow_fdm_split_metrics(per_patch, z_t_v, z_h_v))

        do_delta = delta_prob >= 1.0 or bool(torch.rand(()).item() < delta_prob)
        if do_delta:
            eps = torch.randn_like(z_h_v)
            was_training = self.visual_head.training
            self.visual_head.eval()
            _, v_pos, _ = self.visual_head(cond_pos_v, z_h_v, noise=eps, t=0.0, return_pred=True)
            _, v_nul, _ = self.visual_head(cond_nul_v, z_h_v, noise=eps, t=0.0, return_pred=True)
            if was_training:
                self.visual_head.train()
            delta = v_pos.float() - v_nul.float()
            pool_w = torch.softmax(delta.norm(dim=-1) / max(tau, 1e-6), dim=1)
            d_vec = (delta * pool_w.unsqueeze(-1)).sum(dim=1)
            extras["stat/delta_norm_mean"] = d_vec.detach().norm(dim=-1).mean()

            a_flat = a_v.reshape(a_v.shape[0], -1)
            #######
            # 中文注释：CED probe/proj 在 bf16 eval 或手动 bf16 smoke 中参数 dtype 会变化；
            # 输入按各自模块 dtype 前向，loss/余弦计算再转 fp32，兼顾兼容性和数值日志。
            probe_dtype = self._jointflow_module_dtype(self.probe_mlp, fallback=d_vec.dtype)
            probe_pred = self.probe_mlp(d_vec.to(dtype=probe_dtype)).float()
            #######
            is_val = torch.tensor(
                [int(ex.get("trajectory_id", -1)) % val_mod == 0 for ex in examples_v],
                device=d_vec.device,
                dtype=torch.bool,
            )
            if bool((~is_val).any()):
                loss_ident = ((probe_pred[~is_val] - a_flat[~is_val]) ** 2).mean()
            else:
                loss_ident = self._zero_grad_anchor_for_modules([self.probe_mlp], loss_fdm)
            if bool(is_val.any()):
                with torch.no_grad():
                    extras["metric/probe_mse_val"] = ((probe_pred[is_val] - a_flat[is_val]) ** 2).mean()
            extras["loss/ident"] = loss_ident.detach()
            extras["loss_ident"] = loss_ident.detach()

            if teacher == "delta":
                t_vec = d_vec.detach()
            elif teacher == "pooled_future":
                pw = torch.softmax(v_pos.float().norm(dim=-1) / max(tau, 1e-6), dim=1)
                t_vec = (v_pos.float() * pw.unsqueeze(-1)).sum(dim=1).detach()
            elif teacher == "delta_shuffled":
                t_vec = d_vec[torch.randperm(d_vec.shape[0], device=d_vec.device)].detach()
            #######
            # 中文注释（实验 E2.2/E2.3 的对照 teacher，只新增 teacher 选项、均 detach，不改 CED 机制/不加 trick）：
            #   future_dino → 原始未来 DINO 特征 z_h 的范数 softmax 池化（"用 future DINO 与 DiT 对齐"的对照）
            #   delta_dino  → 原始 (未来−当前) DINO 特征 (z_h−z_t) 的范数 softmax 池化（"用 delta DINO 对齐"的对照）
            # 三者池化方式与 teacher=delta 完全一致，保证 E2.1/E2.2/E2.3 仅 teacher 不同、其余严格同构。
            elif teacher == "future_dino":
                pw = torch.softmax(z_h_v.float().norm(dim=-1) / max(tau, 1e-6), dim=1)
                t_vec = (z_h_v.float() * pw.unsqueeze(-1)).sum(dim=1).detach()
            elif teacher == "delta_dino":
                z_diff = z_h_v.float() - z_t_v.float()
                pw = torch.softmax(z_diff.norm(dim=-1) / max(tau, 1e-6), dim=1)
                t_vec = (z_diff * pw.unsqueeze(-1)).sum(dim=1).detach()
            #######
            else:
                raise ValueError(f"Unsupported CED teacher `{teacher}`")
            gate = (t_vec.norm(dim=-1) > eps_gate).float()
            #######
            # 中文注释：REPA 对齐学生 = DiT 早期层（repa_layer）在 valid 样本上按 token 池化 → repa_proj 投到 Δ 空间，
            # 再 fp32 与 detach 的 teacher 算 cosine。这样 align 真正作用在动作专家 DiT 的早期表征上。
            proj_dtype = self._jointflow_module_dtype(self.repa_proj, fallback=repa_feat_v.dtype)
            proj = self.repa_proj(repa_feat_v.mean(dim=1).to(dtype=proj_dtype)).float()
            #######
            cos = torch.nn.functional.cosine_similarity(proj.float(), t_vec, dim=-1)
            loss_align = (gate * (1.0 - cos)).mean()
            extras["loss/align"] = loss_align.detach()
            extras["loss_align"] = loss_align.detach()
            extras["stat/gate_on_frac"] = gate.detach().mean()
        else:
            loss_ident = self._zero_grad_anchor_for_modules([self.probe_mlp, self.repa_proj], loss_fdm)
            loss_align = loss_fdm.new_zeros(())
            extras["loss_ident"] = loss_ident.detach()
            extras["loss_align"] = loss_align.detach()

        total = loss_act + loss_fdm + lam_i * loss_ident + lam_a * loss_align
        out = {"effect_loss": total + self._unused_jointflow_param_anchor("effect", total)}
        out.update(extras)
        return out

    @torch.inference_mode()
    def _jointflow_predict_action(self, examples: List[dict], **kwargs) -> dict:
        batch = self._jointflow_examples_to_batch(examples, require_future_dino=False, require_action=False)
        cond = self._run_jointflow_path("policy", batch)
        #######
        # 中文注释：评估服务器可将 action head 参数转成 bf16；predict_action 没有 trainer autocast，
        # 因此 policy 条件 token 进入 action head 前必须跟随 action head 参数 dtype。
        head_dtype = self._jointflow_module_dtype(self.action_model, fallback=cond.dtype)
        pred_actions = self.action_model.predict_action(
            cond.to(dtype=head_dtype),
            state=batch.get("state"),  # 中文注释：与源一致；include_state=false 时为 None（惰性）
            encoder_attention_mask=None,
        )
        #######
        return {"normalized_actions": pred_actions.float().detach().cpu().numpy()}
    #######

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """ """
        #######
        # 中文注释：JointFlow-style 训练只在显式启用时接管 forward；原生 QwenGR00T 保持 action_loss 路径。
        if self.jointflow_enabled:
            return self._jointflow_forward(examples, task=str(kwargs.get("task", "policy")))
        #######
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated,
                encoder_attention_mask=backbone_attention_mask,
            )  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        #######
        # 中文注释：评估/部署阶段只走 JointFlow 的 policy 路径，禁止触发 fdm/idm/effect 辅助任务。
        if self.jointflow_enabled:
            return self._jointflow_predict_action(examples=examples, **kwargs)
        #######
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden, state, encoder_attention_mask=backbone_attention_mask
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
