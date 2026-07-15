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
# 中文注释：JointFlow 分支需要在局部禁用 autocast，并为 unused-parameter anchor 遍历模块参数。
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
# 中文注释：复用 JointFlow 已验证的 DINO token、query token、visual flow head 模块；
# 这些模块只在 framework.jointflow.enabled=true 时实例化，默认不影响原生 QwenGR00T。
from starVLA.model.framework.VLM4A.jointflow.attention_mask import build_block_causal_mask
from starVLA.model.framework.VLM4A.jointflow.dino_v3 import (
    DINOv3Backbone,
    dino_num_patches,
    dino_patch_grid,
    resolve_dino_spec,
)
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
            # DiT model size: "DiT-B" | "DiT-LAWAM" | "DiT-L"
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
            # Action output parameterization. Keep velocity for old checkpoints;
            # set jit_x to predict clean actions and train with JiT's v-loss.
            "prediction_type": "velocity",
            "jit_t_eps": 5e-2,
            # Keep legacy for old checkpoints. ``gr00t`` uses
            # t=(1-Beta(alpha,beta))*noise_s, as in Isaac-GR00T/LaWAM.
            "flow_time_sampling": "legacy",
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

    # 中文注释：future DINO flow-matching head 配置，fdm/passive 任务使用。
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
        self._validate_joint_e2e_contract()
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
        #######
        # 中文注释：WAM 分支开关（参考 LaWAM）：qwenvl-gr00t 原生视觉 policy + DINO 只做监督头。
        # 与 jointflow 互斥——jointflow 把 DINO 当 policy 视觉；wam 用 Qwen 原生视觉，DINO 仅作 future 监督 target，推理不碰。
        self.wam_enabled = bool(self.config.framework.get("wam", {}).get("enabled", False))
        if self.wam_enabled:
            self.config.framework.visual_model.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size
        # Multi-task branches normally attach every unused parameter to the loss with a zero-valued
        # autograd edge.  The trainer may disable those anchors after DeepSpeed ZeRO-2 is initialized:
        # ZeRO-2 fills missing partition gradients with zeros before AdamW, preserving optimizer semantics
        # without reducing a dense zero gradient for every inactive head.
        self._unused_param_anchors_enabled = True
        #######

        #######
        # 中文注释：动作头骨干选择——默认 GR00T DiT-B；`framework.action_model.backbone: wan` 时换成
        # Wan-初始化动作头（Wan DiTBlock 架构 + Wan 视频 DiT 骨干截断/插值初始化，参考 FastWAM）。
        # 对外接口与 FlowmatchingActionHead 完全一致（forward/predict_action/set_action_correlation），
        # 所以 wam/native 调用处都不用改。
        #######
        # 中文注释：World→Action guidance 的 M5/M6/M6+ 需要 action DiT 内建 world 子模块（world_attn / world_to_temb），
        # 必须在构造 action head 之前把开关写进 diffusion_model_cfg。M4(alternate)/concat-family 不需要新参数。
        self._inject_guidance_dit_flags()
        #######
        action_backbone = str(self.config.framework.action_model.get("backbone", "gr00t")).lower()
        if action_backbone == "wan":
            from starVLA.model.modules.action_model.wan_action_head import WanFlowMatchingActionHead

            self.action_model = WanFlowMatchingActionHead(self.config)
        else:
            self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        #######

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
        if self.wam_enabled:
            self._init_wam_modules()
        #######

    #######
    def _validate_joint_e2e_contract(self) -> None:
        """Fail before loading large backbones if joint E2E semantics are broken."""

        framework = self.config.framework
        tasks = framework.get("tasks", {})
        weights = tasks.get("weights", {}) if hasattr(tasks, "get") else {}
        active = [str(name) for name, weight in weights.items() if float(weight) > 0.0]
        if "joint_e2e" not in active:
            return
        if active != ["joint_e2e"]:
            raise ValueError(
                "joint_e2e is an exclusive task because each of its batches already computes "
                f"action_loss + world_loss; active tasks={active}."
            )

        wam = framework.get("wam", {})
        guidance = wam.get("guidance", {}) if hasattr(wam, "get") else {}
        problems = []
        if not bool(wam.get("enabled", False)):
            problems.append("framework.wam.enabled must be true")
        if not bool(guidance.get("enabled", False)):
            problems.append("framework.wam.guidance.enabled must be true")
        if str(guidance.get("prompt_mode", "")).lower() != "dual_query":
            problems.append("guidance.prompt_mode must be dual_query")
        if not bool(guidance.get("exclude_post_query_context", False)):
            problems.append("guidance.exclude_post_query_context must be true")
        if str(guidance.get("bridge_source", "")).lower() != "predicted":
            problems.append("guidance.bridge_source must be predicted")
        if bool(guidance.get("detach_world", True)):
            problems.append("guidance.detach_world must be false")
        if str(guidance.get("signal", "")).lower() not in {"z_pred", "delta_z_pred"}:
            problems.append("guidance.signal must be z_pred or delta_z_pred")
        if float(wam.get("dino_loss_weight", 0.0)) <= 0.0:
            problems.append("framework.wam.dino_loss_weight must be positive")
        action_cfg = framework.get("action_model", {})
        if bool(action_cfg.get("use_correlated_noise", False)):
            problems.append("framework.action_model.use_correlated_noise must be false for the E2E experiment")

        ramp = guidance.get("action_world_gradient_ramp", {})
        if hasattr(ramp, "get") and bool(ramp.get("enabled", False)):
            start = int(ramp.get("start_step", 0))
            end = int(ramp.get("end_step", 0))
            start_scale = float(ramp.get("start_scale", 0.0))
            end_scale = float(ramp.get("end_scale", 1.0))
            if start < 0 or end < start:
                problems.append(f"gradient ramp requires 0 <= start_step <= end_step, got {start} -> {end}")
            if start_scale < 0.0 or end_scale < 0.0:
                problems.append(
                    f"gradient ramp scales must be non-negative, got {start_scale} -> {end_scale}"
                )
        if problems:
            raise ValueError("Invalid joint_e2e configuration: " + "; ".join(problems))

    #######
    # 中文注释：在构造 action head 前，把 guidance 的 M5/M6/M6+ 开关注入 action_model.diffusion_model_cfg，
    # 让 DiT 在 __init__ 时建好 world_attn(每 cross block) / world_to_temb(AdaLN)。默认关 → 不注入任何键。
    def _inject_guidance_dit_flags(self) -> None:
        from omegaconf import OmegaConf

        wam = self.config.framework.get("wam", {})
        g = wam.get("guidance", {}) if hasattr(wam, "get") else {}
        if not bool(g.get("enabled", False)):
            return
        mode = str(g.get("mode", "none")).lower()
        if mode not in ("dual_xattn", "adaln", "dual_xattn_adaln"):
            return  # M4 alternate 复用 attn1，concat-family 走 memory 拼接，都不需要新 DiT 参数
        dcfg = self.config.framework.action_model.diffusion_model_cfg
        # 训练时 self.config 被 AccessTrackedConfig(config_tracker) 包了一层，嵌套节点也是包装对象，
        # 而 OmegaConf.set_struct 的 monkey-patch 没覆盖 → 直接调会 AttributeError(_set_flag)。
        # 先 unwrap 成原生 OmegaConf（底层同一引用，改键照样对 self.config 可见）。
        if hasattr(dcfg, "unwrap"):
            dcfg = dcfg.unwrap()
        OmegaConf.set_struct(dcfg, False)
        if mode in ("dual_xattn", "dual_xattn_adaln"):
            dcfg.world_cross_attention = True
            dcfg.world_gate_init = float(g.get("gate_init", 0.0))
        if mode in ("adaln", "dual_xattn_adaln"):
            dcfg.world_adaln = True
        # world_cross_attention_dim / world_global_dim 默认 = cross_attention_dim（= d_model），DiT 内部兜底。

    #######

    def _uses_action_state(self) -> bool:
        datasets_cfg = getattr(self.config, "datasets", None)
        vla_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
        if vla_cfg is None:
            return False
        return vla_cfg.get("include_state", False) not in ["False", "false", False, 0, None]

    #######
    # 中文注释：初始化 JointFlow 迁移模块。这里复用原生 Qwen3-VL 的 language_model 作为 joint sequence
    # backbone，复用原生 GR00T action_model 作为 policy/idm 动作流头，只新增 DINO/FDM 必需模块。
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

        # Frozen DINO is a target encoder, not a trainable/checkpointed model
        # component.  Keep it outside nn.Module registration so checkpoints do
        # not silently grow by the full DINO-L state dict.
        object.__setattr__(self, "_dino_teacher", None)
        if bool(dino_cfg.get("load_live_backbone", False)):
            self._set_dino_teacher(DINOv3Backbone(**self._jointflow_dino_spec))
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

    def set_unused_param_anchors_enabled(self, enabled: bool) -> None:
        """Enable conservative unused-parameter anchors for non-ZeRO distributed backends."""
        self._unused_param_anchors_enabled = bool(enabled)

    def _zero_grad_anchor_for_modules(self, modules: list[nn.Module | None], ref: torch.Tensor) -> torch.Tensor:
        anchor = ref.new_zeros(())
        if not self._unused_param_anchors_enabled:
            return anchor
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
            modules.extend([self.visual_head, self.future_dino_queries, self.act_ctx])
        elif task == "fdm":
            modules.extend([self.action_model, self.action_queries])
        elif task == "passive":
            modules.extend([self.action_model, self.action_queries, self.act_ctx])
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
            logger.warning(f"JointFlow DINO stats dim {mean.numel()} != embed_dim {self.d_dino}; using identity stats.")
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
            rows, columns = dino_patch_grid(
                self._jointflow_dino_spec["image_size"], self._jointflow_dino_spec["patch_size"]
            )
            if n_tokens == rows * columns and rows % pool == 0 and columns % pool == 0:
                dino = dino.view(dino.shape[0], rows, columns, dino.shape[-1])
                dino = dino.view(dino.shape[0], rows // pool, pool, columns // pool, pool, dino.shape[-1]).mean(
                    dim=(2, 4)
                )
                dino = dino.reshape(dino.shape[0], -1, dino.shape[-1])
        return dino

    def _set_dino_teacher(self, teacher: DINOv3Backbone | None) -> None:
        if teacher is not None:
            teacher.requires_grad_(False)
            teacher.eval()
        object.__setattr__(self, "_dino_teacher", teacher)

    def _ensure_jointflow_dino(self) -> DINOv3Backbone:
        teacher = getattr(self, "_dino_teacher", None)
        if teacher is None:
            teacher = DINOv3Backbone(**self._jointflow_dino_spec)
            self._set_dino_teacher(teacher)
        teacher = teacher.to(self.device).eval()
        self._set_dino_teacher(teacher)
        return teacher

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
            torch.autocast(device_type=device_type, enabled=False) if device_type in {"cuda", "cpu"} else nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
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

    #######
    # 中文注释：E1.3 correlated noise——把 trainer 算好的 Σ-Cholesky 透传给 action head。
    def set_action_correlation(self, chol) -> None:
        self.action_model.set_action_correlation(chol)

    #######

    #######
    # 中文注释：评测时设置在线 DINO 的 per-suite 归一化 stats（dino_v3_stats.json）。训练用的是离线精算特征
    # （已按各 suite 的 stats 标准化），而 eval 走在线提取——必须用同一份 stats 归一化，否则 dino_proj 收到
    # 原始尺度特征→视觉条件失效→SR≈0。每个 suite 的 stats 不同，评测前按 suite 注入对应文件。
    def set_dino_stats(self, stats_path: str) -> None:
        self._load_jointflow_dino_stats(stats_path)

    #######

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
                    state=batch.get("state") if self._uses_action_state() else None,
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
            state=batch.get("state") if self._uses_action_state() else None,
            encoder_attention_mask=None,
        )
        #######
        return {"normalized_actions": pred_actions.float().detach().cpu().numpy()}

    #######

    #######
    # 中文注释：WAM 模块初始化（qwenvl-gr00t 原生视觉 policy + DINO 只做监督头，参考 LaWAM 占位 token 方案）。
    # 占位 token：act/flow query 共用一个特殊 token，按出现顺序前 n_act 为 act、其后 n_flow 为 flow（同 LaWAM build_placeholder_masks）。
    def _init_wam_modules(self) -> None:
        dino_cfg = self.config.framework.dino
        visual_cfg = self.config.framework.visual_model
        action_cfg = self.config.framework.action_model
        wam_cfg = self.config.framework.get("wam", {})
        self.action_dim = int(action_cfg.action_dim)
        self.wam_n_act = int(action_cfg.get("n_action_query", self.action_horizon))
        self.wam_n_flow = int(visual_cfg.get("n_flow_query", 8))
        self.wam_dino_loss_weight = float(wam_cfg.get("dino_loss_weight", 1.0))
        #######
        # 中文注释：fdm 预测 delta-DINO 开关（默认 False=预测绝对未来 DINO）。
        # 开启后 fdm target = DINO(image_1) - DINO(image_0)，让监督头学「动作引起的特征变化」而非整张未来特征，
        # 与 idm/世界模型差分思路一致；只影响 fdm 范式（passive 仍预测绝对未来 DINO）。
        self.wam_fdm_delta = bool(wam_cfg.get("fdm_delta_dino", False))
        #######
        #######
        # 中文注释：world_model_no_language——fdm/passive 的 prompt 去掉语言指令(消融:显式动作条件 vs 语言对未来动态建模的作用)。
        # 默认 False(带语言)；policy/idm 不受影响,永远带语言。
        self.wam_world_model_no_language = bool(wam_cfg.get("world_model_no_language", False))
        #######
        # 占位 token 注册 + 词表 resize（参考 LaWAM configure_latent_world_processor）。
        self.wam_ph = str(wam_cfg.get("placeholder_token", "<ACT_PH>"))
        tok = self.qwen_vl_interface.processor.tokenizer
        tok.add_special_tokens({"additional_special_tokens": [self.wam_ph]})
        self.wam_ph_id = int(tok.convert_tokens_to_ids(self.wam_ph))
        self.qwen_vl_interface.model.resize_token_embeddings(len(tok))
        # DINO 仅作监督 target：复用 jointflow 的 DINO 提取(在线) + 未来帧 flow 头；推理完全不用。
        self._jointflow_dino_spec = resolve_dino_spec(dino_cfg)
        self.d_dino = int(self._jointflow_dino_spec["embed_dim"])
        self.config.framework.visual_model.d_dino = self.d_dino
        self.register_buffer("_dino_mean", torch.zeros(self.d_dino), persistent=False)
        self.register_buffer("_dino_std", torch.ones(self.d_dino), persistent=False)
        self._load_jointflow_dino_stats(dino_cfg.get("stats_path", None))
        object.__setattr__(self, "_dino_teacher", None)  # 仅训练算 target；不注册、不保存进 checkpoint
        if bool(dino_cfg.get("load_live_backbone", False)):
            self._set_dino_teacher(DINOv3Backbone(**self._jointflow_dino_spec))
        self.wam_visual_head = VisualFlowMatchingHead(self.config)
        # End-to-end predicted-future guidance unrolls the visual flow head
        # several times before the action loss.  Checkpoint its DiT blocks so
        # this graph remains practical at the FastWAM 480-token resolution.
        if bool(wam_cfg.get("visual_gradient_checkpointing", False)):
            visual_dit = getattr(self.wam_visual_head, "model", None)
            if visual_dit is not None and hasattr(visual_dit, "gradient_checkpointing"):
                visual_dit.gradient_checkpointing = True
        # WAM keeps the baseline batch/repeat objective. Checkpoint only the
        # action-DiT blocks to avoid materializing activations for the effective
        # batch (micro-batch x repeated_diffusion_steps); parameters, precision,
        # noise samples and loss are unchanged.
        if bool(wam_cfg.get("action_gradient_checkpointing", True)):
            action_dit = getattr(self.action_model, "model", None)
            if action_dit is not None and hasattr(action_dit, "gradient_checkpointing"):
                action_dit.gradient_checkpointing = True
        # 中文注释：fdm（前向动力学）范式需要动作上下文——把 act_ctx(动作) 拼到 flow-query 作为 DINO 头的 cross-attn 条件。
        self.wam_act_ctx = ActionContextEncoder(
            action_dim=self.action_dim, hidden_size=int(self.qwen_vl_interface.model.config.hidden_size)
        )
        #######
        # 中文注释：World→Action guidance（M0–M6+ 大计划）的配置与子模块初始化。默认全关。
        self._init_wam_guidance(wam_cfg)
        #######

    #######
    # 中文注释：World→Action guidance 配置解析（plan §3）。所有键默认值在此集中，默认 enabled=False →
    # guidance 子模块不构造、forward 不进 guided 分支，旧 WAM/原生 QwenGR00T 行为/ckpt 完全不变。
    @staticmethod
    def _wam_guidance_defaults() -> dict:
        return {
            "enabled": False,
            # 注入结构 mode：concat-family(无需改 action head/DiT) = none|concat|sa_fusion|qformer；
            # 需改 DiT(后续 P6-P8) = alternate_xattn|dual_xattn|adaln|dual_xattn_adaln。
            "mode": "none",
            # world 信号 signal：none|h_future|z_pred|delta_z_pred|z_oracle|delta_z_oracle。
            "signal": "none",
            # prompt：action_only(原 WAM 单 query) | dual_query(act+future 占位同一 forward)。
            "prompt_mode": "action_only",
            # 原生 causal 顺序是 ACT→FUTURE；FUTURE 可读取动作意图，ACT 不能直接读取 FUTURE。
            # add_generation_prompt 产生的后缀 token 能同时读取两组 query；新 E2E 必须将其从
            # action/world raw context memory 排除。默认 false 只用于兼容旧 checkpoint 的原始语义。
            "exclude_post_query_context": False,
            # bridge：predicted(用 world head 预测) | oracle(用 GT DINO) | scheduled(按 oracle_ratio 混)。
            "bridge_source": "predicted",
            "detach_world": True,  # True=action loss 不回传 world head（Stage1/2）；False=e2e(Stage3)
            "oracle_ratio": 1.0,  # scheduled 时逐样本取 oracle 的概率（1→全 oracle, 0→全 predicted）
            "n_world_tokens": 16,  # M3 Q-Former 压缩后的 token 数
            "qformer_layers": 2,
            "fusion_layers": 2,  # M2 CompactSAFusion 层数
            "fusion_heads": 8,
            "qformer_heads": 8,
            "pooler_heads": 8,
            "gate_init": 0.0,  # M5 world_gate 初值（tanh(0)=0 → 平滑从 baseline 起步），DiT 阶段用
            "world_dropout": 0.0,
            "include_context_in_world_memory": True,  # action memory 是否保留 Qwen 图文 context
            # End-to-end mode keeps predicted future in the forward path from
            # step 0, but can linearly ramp only the action-loss gradient that
            # enters the world predictor. Disabled defaults preserve old runs.
            "action_world_gradient_ramp": {
                "enabled": False,
                "start_step": 0,
                "end_step": 0,
                "start_scale": 1.0,
                "end_scale": 1.0,
            },
            # 因果消融(plan §11，eval 时生效)：correct|off|zero|shuffled|wrong_task|gt
            "world_eval_mode": "correct",
        }

    def _init_wam_guidance(self, wam_cfg) -> None:
        g_user = dict(wam_cfg.get("guidance", {})) if hasattr(wam_cfg, "get") else {}
        g = self._wam_guidance_defaults()
        g.update({k: v for k, v in g_user.items() if v is not None})
        self.wam_guidance = g
        self.wam_guidance_enabled = bool(g["enabled"])
        # world_target(absolute|delta)：guidance 下优先；否则回落已有 fdm_delta_dino。delta → world head 学差分。
        wt = str(wam_cfg.get("world_target", "")).lower() if hasattr(wam_cfg, "get") else ""
        if wt in ("absolute", "delta"):
            self.wam_fdm_delta = wt == "delta"
        if not self.wam_guidance_enabled:
            return
        mode = str(g["mode"]).lower()
        signal = str(g["signal"]).lower()
        # 信号是否走 DINO 潜变量空间（d_dino）：z_pred/delta_z_pred/z_oracle/delta_z_oracle；否则 h_future 走 d_model。
        self._wam_signal_is_latent = signal in ("z_pred", "delta_z_pred", "z_oracle", "delta_z_oracle")
        self._wam_signal_is_delta = signal in ("delta_z_pred", "delta_z_oracle")
        self._wam_signal_is_oracle = signal in ("z_oracle", "delta_z_oracle")
        if self._wam_signal_is_delta:
            self.wam_fdm_delta = True  # 信号是差分 → world head 必须训练成预测差分
        d_model = int(self.qwen_vl_interface.model.config.hidden_size)
        d_dino = int(self.d_dino)
        world_in_dim = d_dino if self._wam_signal_is_latent else d_model
        from starVLA.model.framework.VLM4A.wam_guidance import (
            CompactSAFusion,
            WorldQFormer,
            WorldTokenAdapter,
            WorldTokenPooler,
        )

        drop = float(g["world_dropout"])
        # 子模块按 mode/signal 条件实例化（未用到的不建，省参数 + 避免 DDP unused-param；guided 分支另有 anchor 兜底）。
        self.world_adapter = None
        self.world_fusion = None
        self.world_qformer = None
        self.world_pooler = None
        if signal != "none":
            if mode == "qformer":
                # M3：压缩空间 DINO tokens → n_world_tokens（不需要 adapter）。
                self.world_qformer = WorldQFormer(
                    in_dim=world_in_dim,
                    out_dim=d_model,
                    n_query=int(g["n_world_tokens"]),
                    num_layers=int(g["qformer_layers"]),
                    num_heads=int(g["qformer_heads"]),
                    dropout=drop,
                )
            elif mode == "sa_fusion":
                # M2：对 [h_act; h_future] 轻量 SA（都在 d_model，不需要 adapter）。
                self.world_fusion = CompactSAFusion(
                    dim=d_model,
                    num_layers=int(g["fusion_layers"]),
                    num_heads=int(g["fusion_heads"]),
                    dropout=drop,
                )
            else:
                # concat / adaln / dual_xattn / dual_xattn_adaln / alternate_xattn：投影 world 信号到 d_model。
                self.world_adapter = WorldTokenAdapter(in_dim=world_in_dim, out_dim=d_model, dropout=drop)
            if mode in ("adaln", "dual_xattn_adaln"):
                self.world_pooler = WorldTokenPooler(dim=d_model, out_dim=d_model, num_heads=int(g["pooler_heads"]))
        # world head 预测时的默认 token 数（无 GT 兜底，如 live eval）：H/P * W/P。
        self._wam_world_n = dino_num_patches(
            self._jointflow_dino_spec["image_size"], self._jointflow_dino_spec["patch_size"]
        )
        # dual_query：注册独立 future 占位 token（与 act 占位 <ACT_PH> 区分），便于一个 forward 取两组 query。
        self.wam_future_ph = None
        self.wam_future_ph_id = None
        if str(g["prompt_mode"]).lower() == "dual_query":
            tok = self.qwen_vl_interface.processor.tokenizer
            self.wam_future_ph = "<FUTURE_PH>"
            tok.add_special_tokens({"additional_special_tokens": [self.wam_future_ph]})
            self.wam_future_ph_id = int(tok.convert_tokens_to_ids(self.wam_future_ph))
            self.qwen_vl_interface.model.resize_token_embeddings(len(tok))
        logger.info(
            "WAM guidance ON: mode=%s signal=%s prompt=%s exclude_post_query_context=%s "
            "bridge=%s detach_world=%s "
            "(adapter=%s qformer=%s fusion=%s pooler=%s)",
            mode,
            signal,
            g["prompt_mode"],
            bool(g["exclude_post_query_context"]),
            g["bridge_source"],
            g["detach_world"],
            self.world_adapter is not None,
            self.world_qformer is not None,
            self.world_fusion is not None,
            self.world_pooler is not None,
        )

    #######

    @staticmethod
    def _wam_view_list(v):
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return list(v)
        v = np.asarray(v)
        return [v[i] for i in range(v.shape[0])] if v.ndim == 4 else [v]

    # 中文注释：取 (当前所有视角 list, future_main) PIL——当前帧用 image_0(在线 raw [V,H,W,C])/eval 用 image；
    # 未来帧用 image_1（idm 用，取第 0 视角）。支持任意相机数：LIBERO 2 路(agentview+wrist)、RoboTwin 3 路(head+left+right)。
    def _wam_views(self, ex: dict):
        cur = self._wam_view_list(ex.get("image_0", ex.get("image")))
        views = [to_pil_preserve(v) for v in cur]
        fut = self._wam_view_list(ex.get("image_1"))
        future_main = to_pil_preserve(fut[0]) if fut else None
        # Native LeRobot samples are resized to 224x224 in _pack_sample, but the JointFlow/WAM
        # loader returns raw RoboTwin frames (480x640). Honor the same obs_image_size here so
        # training and deployment use identical resolution and Qwen does not tokenize raw frames.
        target_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if target_size:
            target_size = tuple(int(v) for v in target_size)
            views = [view if view.size == target_size else view.resize(target_size) for view in views]
            if future_main is not None and future_main.size != target_size:
                future_main = future_main.resize(target_size)
        return views, future_main

    # 中文注释：task-aware Qwen 原生视觉 prompt（每个任务只放它自己需要的 query 占位）：
    #   action 任务(policy/idm)：[图(idm 含未来帧), 文本, act 占位×n_act]   —— 只放 action query；
    #   visual 任务(passive/fdm)：[图,            文本, flow 占位×n_flow]   —— 只放 future query。
    # 这样 action/future query 不再互相出现在对方任务里：flow query 不会被迫 attend act query（passive 更干净），
    # action head 也不会吃到 flow 占位。只有 idm 把未来帧也当输入（逆动力学：当前+未来→动作）。
    # 返回 processor 输入 + 占位 mask（= 本任务唯一一组占位 = query 位置；用于取 query hidden + 在 action memory 屏蔽 raw 占位）。
    def _build_wam_inputs(self, examples: List[dict], task: str = "policy"):
        task = str(task)
        is_action = task in ("policy", "idm")
        n_ph = self.wam_n_act if is_action else self.wam_n_flow
        ph_str = " ".join([self.wam_ph] * n_ph)
        include_future = task == "idm"
        #######
        # 中文注释：world_model_no_language=true 时,visual 任务(fdm/passive)的 prompt **去掉语言指令**(只 image+flow 占位),
        # 用来消融「世界模型分支是否需要语言」。policy/idm(动作任务)**永远带语言**(动作必须条件在指令上)。
        drop_lang = (not is_action) and bool(getattr(self, "wam_world_model_no_language", False))
        #######
        proc = self.qwen_vl_interface.processor
        cot = self.config.datasets.vla_data.get("CoT_prompt", "{instruction}")
        messages = []
        for ex in examples:
            views, future_main = self._wam_views(ex)
            text = "" if drop_lang else str(cot).replace("{instruction}", str(ex.get("lang", "")))
            # 中文注释：所有当前视角（LIBERO 2 / RoboTwin 3）先放，idm 再加未来帧，最后文本 + 单组 query 占位——
            # 保证 query 占位（native causal）能 attend 到全部图像。
            imgs = list(views)
            if include_future and future_main is not None:
                imgs.append(future_main)
            content = [{"type": "image", "image": im} for im in imgs]
            content.append({"type": "text", "text": f"{text}\n{ph_str}"})
            messages.append([{"role": "user", "content": content}])
        old = proc.tokenizer.padding_side
        proc.tokenizer.padding_side = "left"
        try:
            inputs = proc.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            proc.tokenizer.padding_side = old
        inputs = inputs.to(self.qwen_vl_interface.model.device)
        ph_mask = inputs["input_ids"] == self.wam_ph_id  # 本任务唯一占位组 = query 位置
        return inputs, ph_mask

    def _wam_qwen_hidden(self, inputs) -> torch.Tensor:
        """Run the multimodal backbone without the unused language-model head."""
        full_model = self.qwen_vl_interface.model
        backbone = getattr(full_model, "model", None)
        if backbone is not None and backbone is not full_model:
            outputs = backbone(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
            hidden = getattr(outputs, "last_hidden_state", None)
            return hidden if hidden is not None else outputs[0]

        # Compatibility fallback for VLM wrappers that do not expose their
        # backbone. This preserves the previous behavior.
        outputs = full_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("WAM requires backbone hidden states, but the configured VLM did not return them")
        return hidden_states[-1]

    def _wam_backbone(self, examples: List[dict], task: str = "policy"):
        inputs, ph_mask = self._build_wam_inputs(examples, task=task)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self._wam_qwen_hidden(inputs)
        bsz = len(examples)
        d = hidden.shape[-1]
        n_q = self.wam_n_act if str(task) in ("policy", "idm") else self.wam_n_flow
        h_query = hidden[ph_mask].view(bsz, n_q, d)
        #######
        # 中文注释：返回本任务 query hidden（act 或 flow）+ 完整隐藏序列 + padding mask + 占位 mask。
        # 占位 mask 供 action memory 屏蔽 raw hidden 里的占位 token（只保留图文 hidden + 显式 query）。
        attn = inputs.get("attention_mask", None)
        return h_query, hidden, attn, ph_mask
        #######

    #######
    # 中文注释：wam action head 的 cross-attn memory = act-query（显式保留 metaquery，创新点1）⊕ 图文 hidden。
    # mask = [act-query 全 1] ⊕ [padding mask AND-NOT 占位]——即屏蔽掉 raw hidden 里的占位 token（task-aware 后只会是 act 占位），
    # action DiT 只看「显式 h_act + 非占位图文上下文」，与 baseline action_model(完整 last_hidden + mask) 同口径但去掉占位污染。
    def _wam_action_memory(self, h_act: torch.Tensor, hidden: torch.Tensor, attn, ph_mask):
        mem = torch.cat([h_act, hidden.to(h_act.dtype)], dim=1)
        non_ph = ~ph_mask.to(torch.bool)
        keep = (attn.to(torch.bool) & non_ph) if attn is not None else non_ph
        act_ones = torch.ones(h_act.shape[0], h_act.shape[1], device=mem.device, dtype=torch.bool)
        mem_mask = torch.cat([act_ones, keep.to(mem.device)], dim=1)
        return mem, mem_mask

    #######

    def _wam_action_state_and_mask(self, examples: List[dict]):
        state = self._stack_jointflow_field(examples, "state", required=False) if self._uses_action_state() else None
        action_is_pad = self._stack_jointflow_field(examples, "action_is_pad", required=False)
        if action_is_pad is not None:
            action_is_pad = action_is_pad[:, -self.action_horizon :].to(dtype=torch.bool)
        return state, action_is_pad

    @staticmethod
    def _repeat_wam_batch(value: torch.Tensor | None, repeats: int):
        if value is None:
            return None
        return value.repeat(repeats, *([1] * (value.ndim - 1)))

    def _wam_action_loss(
        self,
        mem: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor | None,
        mem_mask: torch.Tensor | None,
        action_is_pad: torch.Tensor | None,
        *,
        world_embs: torch.Tensor | None = None,
        world_attention_mask: torch.Tensor | None = None,
        world_global: torch.Tensor | None = None,
        guidance_mode: str | None = None,
    ) -> torch.Tensor:
        """Call the WAM action head with native QwenGR00T loss semantics."""

        repeats = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
        if repeats <= 0:
            raise ValueError(f"repeated_diffusion_steps must be positive, got {repeats}")
        kwargs = {
            "encoder_attention_mask": self._repeat_wam_batch(mem_mask, repeats),
            "action_is_pad": self._repeat_wam_batch(action_is_pad, repeats),
        }
        if world_embs is not None:
            kwargs["world_embs"] = self._repeat_wam_batch(world_embs, repeats)
            kwargs["world_attention_mask"] = self._repeat_wam_batch(world_attention_mask, repeats)
        if world_global is not None:
            kwargs["world_global"] = self._repeat_wam_batch(world_global, repeats)
        if guidance_mode is not None:
            kwargs["guidance_mode"] = guidance_mode
        return self.action_model(
            self._repeat_wam_batch(mem, repeats),
            self._repeat_wam_batch(actions, repeats),
            self._repeat_wam_batch(state, repeats),
            **kwargs,
        )

    def _wam_visual_loss(self, cond: torch.Tensor, target: torch.Tensor, examples: List[dict]) -> torch.Tensor:
        """Ignore episode-tail future targets while preserving loss scale."""

        valid = self._stack_jointflow_field(examples, "future_valid", required=False)
        weights = None
        if valid is not None:
            valid = valid.reshape(valid.shape[0], -1)[:, 0].clamp_(0.0, 1.0)
            normalizer = valid.new_tensor(float(valid.shape[0])) / valid.sum().clamp_min(1.0)
            weights = (valid * normalizer)[:, None].expand(-1, target.shape[1])
        return self.wam_visual_head(cond, target, weights=weights)

    #######
    # 中文注释：wam 多任务每步只跑一个 task，未用到的子模块（DINO 头/act_ctx 或 action head）拿不到梯度 →
    # DeepSpeed/DDP 的 unused-parameter 会卡死/报错（同 jointflow 的坑 [[jointflow-multigpu-ddp]]）。
    # 给每步「未用模块」加零梯度 anchor 覆盖：policy/idm 未用 visual_head+act_ctx；passive 未用 action_model+act_ctx；fdm 未用 action_model。
    # 这样单任务（如 exp2/exp3 只 policy）也能多卡训练不挂。
    def _wam_unused_anchor(self, task: str, ref: torch.Tensor) -> torch.Tensor:
        if task in ("policy", "idm"):
            modules = [self.wam_visual_head, self.wam_act_ctx]
        elif task == "passive":
            modules = [self.action_model, self.wam_act_ctx]
        elif task == "fdm":
            modules = [self.action_model]
        else:
            modules = []
        return self._zero_grad_anchor_for_modules(modules, ref)

    #######

    #######
    # 中文注释：WAM 世界模型 target 取法——优先用数据集预存的 DINO latent（dino_target_latents 开时数据集会
    # 在线出 raw 图的同时附带 dino_0/dino_1）。普通配置缺失时仍可在线抽取；严格预计算配置则立即报错，
    # 防止集群 latent 路径/字段配置错后悄悄加载 DINO backbone，改变速度、显存和监督来源。
    def _wam_dino_target(self, examples: List[dict], precomp_key: str, online_keys: list[str]) -> torch.Tensor:
        datasets_cfg = getattr(self.config, "datasets", None)
        vla_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
        strict_precomputed = (
            bool(vla_cfg.get("require_precomputed_dino_targets", False)) if vla_cfg is not None else False
        )
        configured_views_raw = vla_cfg.get("dino_target_view_keys", []) if vla_cfg is not None else []
        configured_views = (
            [str(configured_views_raw)]
            if isinstance(configured_views_raw, str)
            else [str(key) for key in configured_views_raw]
        )
        z = self._stack_jointflow_field(examples, precomp_key, required=False)
        if z is None:
            if strict_precomputed:
                raise RuntimeError(
                    f"Missing required precomputed `{precomp_key}` DINO target. "
                    "Online DINO fallback is disabled by require_precomputed_dino_targets=true."
                )
            z = self._run_jointflow_dino_on_images(examples, online_keys, required=True)
        if z.ndim == 4:
            sample_views = examples[0].get("dino_target_view_keys") or examples[0].get("dino_view_keys")
            sample_views = [str(key) for key in sample_views] if sample_views else []
            if sample_views and len(sample_views) != z.shape[1]:
                raise ValueError(
                    f"DINO target view metadata {sample_views} does not match tensor shape {tuple(z.shape)}."
                )
            if strict_precomputed and configured_views:
                if sample_views != configured_views:
                    raise ValueError(
                        f"Precomputed DINO target views {sample_views} != configured views {configured_views}."
                    )
                if z.shape[1] != len(configured_views):
                    raise ValueError(
                        f"Precomputed DINO target has {z.shape[1]} views; expected {len(configured_views)}."
                    )
            preferred = list(self.config.framework.dino.get("future_view_keys", []))
            chosen = 0
            if sample_views:
                matched = next((key for key in preferred if key in sample_views), None)
                if matched is not None:
                    chosen = sample_views.index(matched)
                elif strict_precomputed and preferred:
                    raise ValueError(
                        f"None of framework.dino.future_view_keys={preferred} is present in target views {sample_views}."
                    )
            z = z[:, chosen]
        elif z.ndim != 3:
            raise ValueError(f"Expected DINO target [B,V,N,D] or [B,N,D], got {tuple(z.shape)}")
        expected_tokens = dino_num_patches(
            self._jointflow_dino_spec["image_size"], self._jointflow_dino_spec["patch_size"]
        )
        if strict_precomputed and z.shape[1] != expected_tokens:
            raise ValueError(
                f"Precomputed DINO target has {z.shape[1]} tokens; expected one image/composite grid "
                f"with {expected_tokens} tokens."
            )
        if z.shape[-1] != self.d_dino:
            raise ValueError(
                f"DINO target dim {z.shape[-1]} != configured encoder dim {self.d_dino} "
                f"({self._jointflow_dino_spec['name']})."
            )
        return z

    #######

    # 中文注释：WAM 四范式前向（每步一种，由 trainer 按 tasks.weights 广播采样）：
    #   policy: 当前图 → act-query → action（→ "action_loss"，兼容无 tasks 的 else 分支）
    #   idm   : 当前图+未来图 → act-query → action（逆动力学，→ "idm_loss"）
    #   passive: 当前图 → flow-query → 预测未来 DINO（→ "passive_loss"）
    #   fdm   : 当前图 + 动作上下文 → flow-query → 预测未来 DINO（前向动力学，→ "fdm_loss"）
    # 两组 query 依托不同范式：act-query 服务 policy/idm（预测 action）；flow-query 服务 passive/fdm（预测 DINO）。
    def _wam_forward(self, examples: List[dict], task: str = "policy", **kwargs) -> dict:
        task = str(task)
        #######
        # 中文注释：World→Action guidance 开启时走 guided 路径（dual-query + world 注入）。默认关→原 WAM。
        if getattr(self, "wam_guidance_enabled", False):
            return self._wam_guided_forward(examples, task=task, **kwargs)
        #######
        h_query, hidden, attn, ph_mask = self._wam_backbone(examples, task=task)
        if task in ("policy", "idm"):
            actions = self._stack_jointflow_field(examples, "action", required=True)
            actions = actions[:, -self.action_horizon :, : self.action_dim].float()
            state, action_is_pad = self._wam_action_state_and_mask(examples)
            #######
            # 中文注释：action 条件 = act-query ⊕ 图文 hidden（屏蔽 raw 占位，见 _wam_action_memory）。
            mem, mem_mask = self._wam_action_memory(h_query, hidden, attn, ph_mask)
            hd = self._jointflow_module_dtype(self.action_model, fallback=mem.dtype)
            loss = self._wam_action_loss(
                mem.to(hd),
                actions.to(hd),
                state.to(hd) if state is not None else None,
                mem_mask,
                action_is_pad,
            )
            #######
            return {("action_loss" if task == "policy" else "idm_loss"): loss + self._wam_unused_anchor(task, loss)}
        # passive / fdm：flow-query → 预测未来帧 DINO；fdm 额外把 act_ctx(动作) 拼进 cross-attn 条件。
        h_flow = h_query
        z_gt = self._wam_dino_target(examples, "dino_1", ["image_1"])
        cond = h_flow
        if task == "fdm":
            actions = self._stack_jointflow_field(examples, "action", required=True)
            actions = actions[:, -self.action_horizon :, : self.action_dim]
            ad = self._jointflow_module_dtype(self.wam_act_ctx, fallback=h_flow.dtype)
            actx = self.wam_act_ctx(actions.to(ad)).to(h_flow.dtype)
            cond = torch.cat([h_flow, actx], dim=1)
        #######
        # 中文注释：delta-DINO target（未来 - 当前 DINO 差分）——**对 fdm 和 passive 都生效**（由 fdm_delta_dino 控）。
        # 这样 fdm(有 act_ctx 动作条件) 与 passive(无动作条件) 用**同一监督 target**，单变量只差「动作条件」，
        # 干净判断「动作能否塑造更好表征」（用户意图：判断动作时 target 不应改变）。
        # 历史名 fdm_delta_dino,语义其实是「世界模型 target 用差分」,对两个 visual 范式通用。
        # 残差幅值远小于 N(0,1) 噪声→按通道(B,N 维)标准化到≈单位方差;DINO 头纯监督、推理不用→无需反归一化。
        if self.wam_fdm_delta:
            z_0 = self._wam_dino_target(examples, "dino_0", ["image_0", "image"])
            z_gt = z_gt - z_0.to(z_gt.device, z_gt.dtype)
            z_gt = z_gt / (z_gt.std(dim=(0, 1), keepdim=True) + 1e-6)
        #######
        vh = self._jointflow_module_dtype(self.wam_visual_head, fallback=cond.dtype)
        #######
        # 中文注释：拆出 raw（视觉头原始 MSE）与 weighted（× dino_loss_weight，= 实际反传的）两份，
        # 供 trainer 分别记录 loss_<task>_raw / loss_<task>_weighted。raw 仅作日志（detach，不进反传）；
        # backward 只用 weighted（key 以 _loss 结尾才会被 total_loss = sum(*_loss) 计入，raw 以 _loss_raw 结尾不计入）。
        raw_dino = self._wam_visual_loss(cond.to(vh), z_gt, examples)
        loss = self.wam_dino_loss_weight * raw_dino
        return {
            f"{task}_loss": loss + self._wam_unused_anchor(task, loss),
            f"{task}_loss_raw": raw_dino.detach(),
        }
        #######

    # 中文注释：WAM 推理 = 只走原生视觉 policy（act-query→action head 采样）；DINO/监督头完全不碰。
    @torch.inference_mode()
    def _wam_predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        #######
        # 中文注释：World→Action guidance 开启时走 guided 推理（dual-query + world 注入 + 可选因果消融）。
        if getattr(self, "wam_guidance_enabled", False):
            return self._wam_guided_predict_action(examples, **kwargs)
        #######
        #######
        # 中文注释：推理 action 条件与训练 policy 一致 = act-query ⊕ 图文 hidden（屏蔽 raw 占位）；走 policy prompt（仅 act 占位）。
        h_act, hidden, attn, ph_mask = self._wam_backbone(examples, task="policy")
        mem, mem_mask = self._wam_action_memory(h_act, hidden, attn, ph_mask)
        head_dtype = self._jointflow_module_dtype(self.action_model, fallback=mem.dtype)
        state, _ = self._wam_action_state_and_mask(examples)
        pred = self.action_model.predict_action(
            mem.to(head_dtype),
            state=state.to(head_dtype) if state is not None else None,
            encoder_attention_mask=mem_mask,
        )
        #######
        return {"normalized_actions": pred.float().detach().cpu().numpy()}

    #######

    #######################################################################
    # 中文注释：World→Action guidance（M0–M6+ 大计划）的 guided 路径。仅在
    # framework.wam.guidance.enabled=true 时由 _wam_forward/_wam_predict_action 路由进来。
    # 与原 WAM 的差别：用 dual-query prompt（act 占位 + future 占位同一 forward），把 future-query
    # hidden（或 world head 预测的未来 DINO 潜变量）作为 world 信号注入 action 条件，保留原始
    # action 条件不删（plan §2.1）。当前支持 concat-family 注入：none(M0-Q)/concat(M1.x)/
    # sa_fusion(M2)/qformer(M3)——均走 action memory 拼接，复用现有 action head 接口，不改 DiT。
    # alternate/dual_xattn/adaln/dual_xattn_adaln(M4/M5/M6/M6+) 需 DiT 改造，留任务 #14(P6-P8)。
    #######################################################################

    @staticmethod
    def _wam_validate_dual_query_layout(
        act_mask: torch.Tensor,
        future_mask: torch.Tensor,
        expected_act: int,
        expected_future: int,
    ) -> None:
        """Validate the token layout that gives ACT -> FUTURE causal direction."""

        if act_mask.ndim != 2 or future_mask.shape != act_mask.shape:
            raise ValueError(
                "Dual-query masks must have the same [B,T] shape; "
                f"got ACT={tuple(act_mask.shape)}, FUTURE={tuple(future_mask.shape)}"
            )
        act_mask = act_mask.to(dtype=torch.bool)
        future_mask = future_mask.to(dtype=torch.bool)
        act_counts = act_mask.sum(dim=1)
        future_counts = future_mask.sum(dim=1)
        if not bool((act_counts == int(expected_act)).all()) or not bool(
            (future_counts == int(expected_future)).all()
        ):
            raise ValueError(
                "Unexpected dual-query placeholder counts: "
                f"ACT={act_counts.tolist()} (expected {expected_act}), "
                f"FUTURE={future_counts.tolist()} (expected {expected_future})"
            )
        positions = torch.arange(act_mask.shape[1], device=act_mask.device).unsqueeze(0)
        last_act = positions.masked_fill(~act_mask, -1).max(dim=1).values
        first_future = positions.masked_fill(~future_mask, act_mask.shape[1]).min(dim=1).values
        if not bool((last_act < first_future).all()):
            raise ValueError(
                "Dual-query prompt must place every ACT placeholder before every FUTURE placeholder "
                "to preserve ACT->FUTURE causal conditioning without a direct FUTURE->ACT path"
            )

    @staticmethod
    def _wam_query_suffix_exclusion_mask(act_mask: torch.Tensor, future_mask: torch.Tensor) -> torch.Tensor:
        """Exclude both query groups and every later template token from raw context memory."""

        act_mask = act_mask.to(dtype=torch.bool)
        future_mask = future_mask.to(dtype=torch.bool)
        if not bool(act_mask.any(dim=1).all()) or not bool(future_mask.any(dim=1).all()):
            raise ValueError("Every dual-query sample must contain ACT and FUTURE placeholders")
        query_mask = act_mask | future_mask
        return query_mask.to(dtype=torch.int64).cumsum(dim=1) > 0

    def _build_wam_guided_inputs(self, examples: List[dict], task: str = "policy"):
        """dual-query prompt：指令后接 act 占位×n_act + future 占位×n_flow（同一 forward）。

        中文注释：与 _build_wam_inputs 同构，但**每个任务**都同时放两组 query 占位。原生 causal 顺序
        使 h_future 可读取 h_act 的动作意图，而 h_act 不可读取 future query；future→action 只经显式 gate 返回。
        返回 inputs + act_mask + future_mask + 原始 2D padding mask + raw-context 排除 mask。
        """
        task = str(task)
        is_action = task in ("policy", "idm", "joint_e2e")
        include_future = task == "idm"
        act_str = " ".join([self.wam_ph] * self.wam_n_act)
        fut_str = " ".join([self.wam_future_ph] * self.wam_n_flow)
        ph_str = f"{act_str} {fut_str}"
        drop_lang = (not is_action) and bool(getattr(self, "wam_world_model_no_language", False))
        proc = self.qwen_vl_interface.processor
        cot = self.config.datasets.vla_data.get("CoT_prompt", "{instruction}")
        messages = []
        for ex in examples:
            views, future_main = self._wam_views(ex)
            text = "" if drop_lang else str(cot).replace("{instruction}", str(ex.get("lang", "")))
            imgs = list(views)
            if include_future and future_main is not None:
                imgs.append(future_main)
            content = [{"type": "image", "image": im} for im in imgs]
            content.append({"type": "text", "text": f"{text}\n{ph_str}"})
            messages.append([{"role": "user", "content": content}])
        old = proc.tokenizer.padding_side
        proc.tokenizer.padding_side = "left"
        try:
            inputs = proc.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            proc.tokenizer.padding_side = old
        inputs = inputs.to(self.qwen_vl_interface.model.device)
        act_mask = inputs["input_ids"] == self.wam_ph_id
        fut_mask = inputs["input_ids"] == self.wam_future_ph_id
        self._wam_validate_dual_query_layout(
            act_mask,
            fut_mask,
            expected_act=self.wam_n_act,
            expected_future=self.wam_n_flow,
        )
        attention_2d = inputs.get("attention_mask", None)
        if attention_2d is None:
            attention_2d = torch.ones_like(inputs["input_ids"], dtype=torch.bool)
        else:
            attention_2d = attention_2d.to(dtype=torch.bool)

        if bool(self.wam_guidance.get("exclude_post_query_context", False)):
            # Tokens emitted by add_generation_prompt occur after both query
            # groups and can attend to both.  They must not re-enter either the
            # action memory or the gated world memory as an implicit bridge.
            context_exclusion_mask = self._wam_query_suffix_exclusion_mask(act_mask, fut_mask)
        else:
            # Historical behavior for old guided checkpoints/config snapshots.
            context_exclusion_mask = act_mask | fut_mask
        return inputs, act_mask, fut_mask, attention_2d, context_exclusion_mask

    def _wam_guided_backbone(self, examples: List[dict], task: str = "policy"):
        """Run Qwen and return both query groups plus raw-context validity/exclusion masks."""
        inputs, act_mask, fut_mask, attention_2d, context_exclusion_mask = self._build_wam_guided_inputs(
            examples, task=task
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self._wam_qwen_hidden(inputs)
        bsz = len(examples)
        d = hidden.shape[-1]
        h_act = hidden[act_mask].view(bsz, self.wam_n_act, d)
        h_future = hidden[fut_mask].view(bsz, self.wam_n_flow, d)
        # Downstream DiT memories use the original 2D validity mask. New E2E
        # configs make context_exclusion_mask remove every post-query
        # chat-template token, closing the only non-gate FUTURE->action bridge.
        return h_act, h_future, hidden, attention_2d, context_exclusion_mask

    def _wam_world_target(self, examples: List[dict]) -> torch.Tensor:
        """GT 未来 world 监督 target：absolute=DINO(image_1)；delta=标准化(DINO(img1)-DINO(img0))。

        中文注释：passive/fdm 的监督 target 与 oracle bridge 的 GT 必须**同一函数**产出，否则 oracle 与
        world head 学到的空间不一致（delta 还要按 batch 标准化到≈单位方差，与 _wam_forward 老路一致）。
        """
        z1 = self._wam_dino_target(examples, "dino_1", ["image_1"])
        if not self.wam_fdm_delta:
            return z1
        z0 = self._wam_dino_target(examples, "dino_0", ["image_0", "image"])
        z = z1 - z0.to(z1.device, z1.dtype)
        return z / (z.std(dim=(0, 1), keepdim=True) + 1e-6)

    def _wam_action_world_grad_scale(self, global_step: int) -> float:
        """Scale only the action-loss gradient entering the world branch."""

        ramp = self.wam_guidance.get("action_world_gradient_ramp", {})
        if not hasattr(ramp, "get") or not bool(ramp.get("enabled", False)):
            return 1.0
        from starVLA.model.framework.VLM4A.wam_guidance import linear_gradient_ramp

        return linear_gradient_ramp(
            global_step=global_step,
            start_step=ramp.get("start_step", 0),
            end_step=ramp.get("end_step", 0),
            start_scale=ramp.get("start_scale", 0.0),
            end_scale=ramp.get("end_scale", 1.0),
        )

    def _wam_world_gate_metrics(self) -> dict[str, torch.Tensor]:
        """Return the effective strength of every gated world cross-attention.

        The residual multiplier used by the DiT is ``tanh(world_gate)``.  Log
        statistics of that effective multiplier (rather than the unbounded raw
        parameter) so ``openness=0`` means fully closed and values approaching
        one mean that the world residual is being admitted at full scale.
        """

        action_dit = getattr(self.action_model, "model", None)
        blocks = getattr(action_dit, "transformer_blocks", ())
        gates = [getattr(block, "world_gate", None) for block in blocks]
        gates = [gate for gate in gates if gate is not None]
        if not gates:
            raise RuntimeError(
                "WAM dual_xattn is active but the action DiT exposes no world_gate parameters; "
                "cannot verify or log the gated world branch."
            )
        with torch.no_grad():
            effective = torch.cat([torch.tanh(gate.detach().float()).reshape(-1) for gate in gates])
            return {
                "world_gate_openness": effective.abs().mean(),
                "world_gate_signed_mean": effective.mean(),
                "world_gate_max_openness": effective.abs().max(),
            }

    def _build_world_signal(
        self,
        h_future: torch.Tensor,
        examples: List[dict],
        eval_mode: str = "correct",
        action_world_grad_scale: float = 1.0,
    ):
        """产出注入 action 的 world tokens [B,N_w,D_model]（或 None=不注入）。

        中文注释：按 guidance.signal 取信号——h_future(直接用 future-query hidden)；z_pred/delta_z_pred
        (world head 预测未来 DINO 潜变量)；z_oracle/delta_z_oracle(直接用 GT)。bridge_source 决定
        predicted/oracle/scheduled 混合；detach_world 决定 action loss 是否回传 world head。最后用
        adapter(concat/dual/adaln) 或 qformer(M3) 投到 d_model。eval_mode 实现因果消融(off/zero/shuffled)。
        """
        from starVLA.model.framework.VLM4A.wam_guidance import (
            mix_world_tokens,
            scale_gradient,
            shuffle_along_batch,
        )

        g = self.wam_guidance
        signal = str(g["signal"]).lower()
        mode = str(g["mode"]).lower()
        em = str(eval_mode or "correct").lower()
        if signal == "none" or em == "off":
            return None

        if not self._wam_signal_is_latent:
            # M1.1/M2：world 信号 = future-query hidden（d_model）。
            base = h_future
        else:
            bridge = str(g["bridge_source"]).lower()
            # oracle（GT DINO/Δ）：仅训练时 dino_1 在 batch 才有；live eval 没有未来帧 → None。
            oracle = None
            want_oracle = self._wam_signal_is_oracle or bridge in ("oracle", "scheduled") or em == "gt"
            if want_oracle:
                try:
                    oracle = self._wam_world_target(examples)
                except Exception:
                    datasets_cfg = getattr(self.config, "datasets", None)
                    vla_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
                    strict_precomputed = (
                        bool(vla_cfg.get("require_precomputed_dino_targets", False)) if vla_cfg is not None else False
                    )
                    # Missing future GT is expected during live evaluation, where the
                    # causal predicted world signal is used. During training, strict
                    # precomputed mode must expose every latent/view/shape error.
                    if self.training and strict_precomputed:
                        raise
                    oracle = None
            # predicted：world head 以 h_future 为条件，flow 采样未来 DINO 潜变量。
            predicted = None
            need_pred = (
                (not self._wam_signal_is_oracle)
                and em != "gt"
                and (bridge in ("predicted", "scheduled") or oracle is None)
            )
            if need_pred:
                n_v = oracle.shape[1] if oracle is not None else int(self._wam_world_n)
                if bool(g["detach_world"]):
                    # detach_world：action loss 不回传 world head（Stage1/2）。直接 no_grad 出预测，
                    # 不建无用反传图（省显存）；world head 由 passive/fdm 步监督训练。
                    with torch.no_grad():
                        predicted = self.wam_visual_head.predict_latent(h_future, n=n_v)
                else:
                    # e2e（Stage3）：保留梯度，action loss 经 predict_latent 回传到 world head + h_future。
                    predicted = self.wam_visual_head.predict_latent(h_future, n=n_v)
            if self._wam_signal_is_oracle or em == "gt":
                base = oracle
            elif bridge == "oracle":
                base = oracle if oracle is not None else predicted
            elif bridge == "scheduled":
                base = mix_world_tokens(predicted, oracle, float(g["oracle_ratio"]))
            else:
                base = predicted if predicted is not None else oracle
            if base is None:
                return None
        base = base.to(h_future.dtype)
        # 因果消融（plan §11）：zero=置零；shuffled/wrong_task=batch 内错排（拿别人的未来）。
        if em == "zero":
            base = torch.zeros_like(base)
        elif em in ("shuffled", "wrong_task"):
            base = shuffle_along_batch(base)
        # 投影到 action cross-attn 维度。
        if mode == "qformer" and self.world_qformer is not None:
            qd = self._jointflow_module_dtype(self.world_qformer, fallback=base.dtype)
            world = self.world_qformer(base.to(qd)).to(h_future.dtype)
        elif self.world_adapter is not None:
            ad = self._jointflow_module_dtype(self.world_adapter, fallback=base.dtype)
            world = self.world_adapter(base.to(ad)).to(h_future.dtype)
        else:
            world = base
        # Forward is always the predicted future.  This identity operation only
        # scales action-loss gradients through the complete world path
        # (adapter + visual predictor + future-query representation).
        return scale_gradient(world, action_world_grad_scale)

    def _build_guided_action_memory(
        self,
        h_act,
        h_future,
        hidden,
        attn,
        context_exclusion_mask,
        world_tokens,
    ):
        """组装 action DiT 的 cross-attn memory（concat-family）。

        中文注释（plan §4.2）：
          none(M0-Q)   memory = [h_act ; qwen_context]（world 不注入，仅 dual-query 影响 Qwen 表征）。
          concat/qformer memory = [h_act ; world_tokens ; qwen_context]。
          sa_fusion(M2)  memory = [SA([h_act;h_future]) ; qwen_context]。
        qwen_context = 原始图文 hidden（屏蔽两组占位）。保留原始 action 条件不删。
        """
        g = self.wam_guidance
        mode = str(g["mode"]).lower()
        dtype = h_act.dtype
        blocks, masks = [], []
        if mode == "sa_fusion" and self.world_fusion is not None:
            fd = self._jointflow_module_dtype(self.world_fusion, fallback=dtype)
            fused = self.world_fusion(torch.cat([h_act, h_future], dim=1).to(fd)).to(dtype)
            blocks.append(fused)
            masks.append(torch.ones(fused.shape[:2], dtype=torch.bool, device=fused.device))
        else:
            blocks.append(h_act)
            masks.append(torch.ones(h_act.shape[:2], dtype=torch.bool, device=h_act.device))
            if mode in ("concat", "qformer") and world_tokens is not None:
                wt = world_tokens.to(dtype)
                blocks.append(wt)
                masks.append(torch.ones(wt.shape[:2], dtype=torch.bool, device=wt.device))
        if bool(g["include_context_in_world_memory"]):
            context_positions = ~context_exclusion_mask.to(torch.bool)
            keep = (attn.to(torch.bool) & context_positions) if attn is not None else context_positions
            blocks.append(hidden.to(dtype))
            masks.append(keep.to(hidden.device))
        return torch.cat(blocks, dim=1), torch.cat(masks, dim=1)

    #######
    # 中文注释：World memory（M4 alternate / M5,M6+ dual cross-attn 用）：M_w=[world_tokens ; qwen_context]
    # （plan §4.5，include_context_in_world_memory 控制是否带 context）。供 action DiT 的 world cross-attn 读。
    def _build_world_memory(self, world_tokens, hidden, attn, context_exclusion_mask):
        g = self.wam_guidance
        dtype = world_tokens.dtype
        blocks = [world_tokens]
        masks = [torch.ones(world_tokens.shape[:2], dtype=torch.bool, device=world_tokens.device)]
        if bool(g["include_context_in_world_memory"]):
            context_positions = ~context_exclusion_mask.to(torch.bool)
            keep = (attn.to(torch.bool) & context_positions) if attn is not None else context_positions
            blocks.append(hidden.to(dtype))
            masks.append(keep.to(hidden.device))
        return torch.cat(blocks, dim=1), torch.cat(masks, dim=1)

    # 中文注释：统一组装 guided action head 的输入——返回 (action_mem, action_mask, world_embs, world_mask, world_global)。
    #   concat-family(none/concat/sa_fusion/qformer)：world 进 action memory 拼接，world_embs/global=None。
    #   DIT-family(alternate/dual/adaln/dual_adaln)：action_mem=[h_act;ctx]（baseline），world 经 cross-attn(world_embs)
    #     和/或 AdaLN(world_global=pooler(world_tokens)) 注入。
    _CONCAT_MODES = ("none", "concat", "sa_fusion", "qformer")
    _DIT_XATTN_MODES = ("alternate_xattn", "dual_xattn", "dual_xattn_adaln")
    _DIT_ADALN_MODES = ("adaln", "dual_xattn_adaln")

    def _assemble_guided_inputs(
        self,
        mode,
        h_act,
        h_future,
        hidden,
        attn,
        context_exclusion_mask,
        world_tokens,
    ):
        mem, mem_mask = self._build_guided_action_memory(
            h_act,
            h_future,
            hidden,
            attn,
            context_exclusion_mask,
            world_tokens,
        )
        world_embs, world_mask, world_global = None, None, None
        if mode in self._DIT_XATTN_MODES and world_tokens is not None:
            world_embs, world_mask = self._build_world_memory(
                world_tokens,
                hidden,
                attn,
                context_exclusion_mask,
            )
        if mode in self._DIT_ADALN_MODES and world_tokens is not None and self.world_pooler is not None:
            pd = self._jointflow_module_dtype(self.world_pooler, fallback=world_tokens.dtype)
            world_global = self.world_pooler(world_tokens.to(pd)).to(h_act.dtype)
        return mem, mem_mask, world_embs, world_mask, world_global

    #######

    def _wam_guided_unused_anchor(self, task: str, ref: torch.Tensor) -> torch.Tensor:
        """guided 单任务/优化步：给本步**结构上不会用到**的子模块加零梯度 anchor（DDP/DeepSpeed 全覆盖）。"""
        g = self.wam_guidance
        mode = str(g["mode"]).lower()
        signal = str(g["signal"]).lower()
        all_mods = {
            "action": self.action_model,
            "visual": self.wam_visual_head,
            "act_ctx": self.wam_act_ctx,
            "adapter": getattr(self, "world_adapter", None),
            "fusion": getattr(self, "world_fusion", None),
            "qformer": getattr(self, "world_qformer", None),
            "pooler": getattr(self, "world_pooler", None),
        }
        used: set[str] = set()
        if task in ("policy", "idm", "joint_e2e"):
            used.add("action")
            if signal != "none":
                if mode == "qformer":
                    used.add("qformer")
                elif mode == "sa_fusion":
                    used.add("fusion")
                else:
                    used.add("adapter")
                if mode in ("adaln", "dual_xattn_adaln"):
                    used.add("pooler")
                # world head 仅在 e2e(detach_world=false) 时由 action loss 真正受梯度；detach_world=true 下
                # predict_latent 在 no_grad 里跑、输出 detach → world head 本步无梯度 → 需 anchor 兜底
                # （它由 passive/fdm 步监督）。oracle 信号根本不跑 world head。
                if self._wam_signal_is_latent and not self._wam_signal_is_oracle and not bool(g["detach_world"]):
                    used.add("visual")
        else:  # passive / fdm
            used.add("visual")
            if task == "fdm":
                used.add("act_ctx")
        unused = [m for k, m in all_mods.items() if k not in used]
        return self._zero_grad_anchor_for_modules(unused, ref)

    def _wam_guided_forward(self, examples: List[dict], task: str = "policy", **kwargs) -> dict:
        task = str(task)
        mode = str(self.wam_guidance["mode"]).lower()
        h_act, h_future, hidden, attn, context_exclusion_mask = self._wam_guided_backbone(examples, task=task)
        if task == "joint_e2e":
            # One causal batch, one Qwen forward, two objectives.  The action
            # branch never sees GT future features; GT is used only as the
            # auxiliary world-prediction target.
            bridge = str(self.wam_guidance["bridge_source"]).lower()
            if bridge != "predicted" or bool(self.wam_guidance["detach_world"]):
                raise ValueError(
                    "joint_e2e requires guidance.bridge_source=predicted and detach_world=false; "
                    f"got bridge_source={bridge!r}, detach_world={self.wam_guidance['detach_world']!r}."
                )

            z_gt = self._wam_world_target(examples)
            vh = self._jointflow_module_dtype(self.wam_visual_head, fallback=h_future.dtype)
            raw_world_loss = self._wam_visual_loss(h_future.to(vh), z_gt, examples)
            world_loss = self.wam_dino_loss_weight * raw_world_loss

            grad_scale = self._wam_action_world_grad_scale(int(kwargs.get("global_step", 0)))
            world_tokens = self._build_world_signal(
                h_future,
                examples,
                action_world_grad_scale=grad_scale,
            )
            mem, mem_mask, w_embs, w_mask, w_global = self._assemble_guided_inputs(
                mode, h_act, h_future, hidden, attn, context_exclusion_mask, world_tokens
            )
            actions = self._stack_jointflow_field(examples, "action", required=True)
            actions = actions[:, -self.action_horizon :, : self.action_dim].float()
            state, action_is_pad = self._wam_action_state_and_mask(examples)
            hd = self._jointflow_module_dtype(self.action_model, fallback=mem.dtype)
            action_loss = self._wam_action_loss(
                mem.to(hd),
                actions.to(hd),
                state.to(hd) if state is not None else None,
                mem_mask,
                action_is_pad,
                world_embs=(w_embs.to(hd) if w_embs is not None else None),
                world_attention_mask=w_mask,
                world_global=(w_global.to(hd) if w_global is not None else None),
                guidance_mode=mode,
            )
            action_loss = action_loss + self._wam_guided_unused_anchor(task, action_loss)
            output = {
                "action_loss": action_loss,
                "world_loss": world_loss,
                "world_loss_raw": raw_world_loss.detach(),
                "action_world_grad_scale": action_loss.detach().new_tensor(grad_scale),
            }
            output.update(self._wam_world_gate_metrics())
            return output
        if task in ("policy", "idm"):
            actions = self._stack_jointflow_field(examples, "action", required=True)
            actions = actions[:, -self.action_horizon :, : self.action_dim].float()
            state, action_is_pad = self._wam_action_state_and_mask(examples)
            world_tokens = self._build_world_signal(h_future, examples)
            mem, mem_mask, w_embs, w_mask, w_global = self._assemble_guided_inputs(
                mode, h_act, h_future, hidden, attn, context_exclusion_mask, world_tokens
            )
            hd = self._jointflow_module_dtype(self.action_model, fallback=mem.dtype)
            loss = self._wam_action_loss(
                mem.to(hd),
                actions.to(hd),
                state.to(hd) if state is not None else None,
                mem_mask,
                action_is_pad,
                world_embs=(w_embs.to(hd) if w_embs is not None else None),
                world_attention_mask=w_mask,
                world_global=(w_global.to(hd) if w_global is not None else None),
                guidance_mode=mode,
            )
            key = "action_loss" if task == "policy" else "idm_loss"
            return {key: loss + self._wam_guided_unused_anchor(task, loss)}
        # passive / fdm：world head 监督（dual-query 的 h_future 为条件，与 policy 一致）。
        z_gt = self._wam_world_target(examples)
        cond = h_future
        if task == "fdm":
            a = self._stack_jointflow_field(examples, "action", required=True)
            a = a[:, -self.action_horizon :, : self.action_dim]
            ad = self._jointflow_module_dtype(self.wam_act_ctx, fallback=h_future.dtype)
            actx = self.wam_act_ctx(a.to(ad)).to(h_future.dtype)
            cond = torch.cat([h_future, actx], dim=1)
        vh = self._jointflow_module_dtype(self.wam_visual_head, fallback=cond.dtype)
        raw = self._wam_visual_loss(cond.to(vh), z_gt, examples)
        loss = self.wam_dino_loss_weight * raw
        return {
            f"{task}_loss": loss + self._wam_guided_unused_anchor(task, loss),
            f"{task}_loss_raw": raw.detach(),
        }

    @torch.inference_mode()
    def _wam_guided_predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        mode = str(self.wam_guidance["mode"]).lower()
        h_act, h_future, hidden, attn, context_exclusion_mask = self._wam_guided_backbone(
            examples, task="policy"
        )
        eval_mode = str(self.wam_guidance.get("world_eval_mode", "correct")).lower()
        world_tokens = self._build_world_signal(h_future, examples, eval_mode=eval_mode)
        mem, mem_mask, w_embs, w_mask, w_global = self._assemble_guided_inputs(
            mode, h_act, h_future, hidden, attn, context_exclusion_mask, world_tokens
        )
        hd = self._jointflow_module_dtype(self.action_model, fallback=mem.dtype)
        state, _ = self._wam_action_state_and_mask(examples)
        pred = self.action_model.predict_action(
            mem.to(hd),
            state=state.to(hd) if state is not None else None,
            encoder_attention_mask=mem_mask,
            world_embs=(w_embs.to(hd) if w_embs is not None else None),
            world_attention_mask=w_mask,
            world_global=(w_global.to(hd) if w_global is not None else None),
            guidance_mode=mode,
        )
        return {"normalized_actions": pred.float().detach().cpu().numpy()}

    #######################################################################

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """ """
        #######
        # 中文注释：WAM 优先接管（原生视觉 policy + DINO 监督头）；其次 JointFlow；都关则原生 action_loss。
        if self.wam_enabled:
            return self._wam_forward(examples, **kwargs)
        # 中文注释：JointFlow-style 训练只在显式启用时接管 forward；原生 QwenGR00T 保持 action_loss 路径。
        if self.jointflow_enabled:
            return self._jointflow_forward(examples, task=str(kwargs.get("task", "policy")))
        #######
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        action_is_pad = (
            [example["action_is_pad"] for example in examples]
            if all("action_is_pad" in example for example in examples)
            else None
        )

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
            action_is_pad_target = None
            if action_is_pad is not None:
                action_is_pad_target = torch.as_tensor(
                    np.asarray(action_is_pad), device=last_hidden.device, dtype=torch.bool
                )[:, -self.action_horizon :]

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            action_is_pad_repeated = (
                action_is_pad_target.repeat(repeated_diffusion_steps, 1) if action_is_pad_target is not None else None
            )
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
                last_hidden_repeated,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=backbone_attention_mask,
                action_is_pad=action_is_pad_repeated,
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
        # 中文注释：WAM 推理只走原生视觉 policy（不碰 DINO）；其次 JointFlow。
        if self.wam_enabled:
            return self._wam_predict_action(examples=examples, **kwargs)
        # 中文注释：评估/部署阶段只走 JointFlow 的 policy 路径，禁止触发 fdm/idm 辅助任务。
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
