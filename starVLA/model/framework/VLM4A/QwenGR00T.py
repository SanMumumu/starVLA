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
from contextlib import contextmanager, nullcontext
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
from torch import nn

#######
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def _qwen_enable_thinking(config) -> bool:
    framework = getattr(config, "framework", None) if config is not None else None
    qwenvl = framework.get("qwenvl", {}) if framework is not None else {}
    return bool(qwenvl.get("enable_thinking", False))

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images

#######
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
            # DiT model size: "DiT-B" | "DiT-M" | "DiT-L"
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
            # t=(1-Beta(alpha,beta))*noise_s for the project GR00T schedule.
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
    jointflow: dict = field(default_factory=lambda: {"enabled": False})

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
            # Ignore any precomputed sample fields and always encode the
            # source images with the frozen online DINO teacher.
            "force_online": False,
        }
    )


    tasks: dict = field(
        default_factory=lambda: {
            "weights": {"policy": 1.0},
            "hybrid_mask": True,
            "attention_mask_neg_value": -1.0e4,
        }
    )

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
        self._validate_causal_query_contract()
        self._validate_wam_two_stage_contract()
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        wam_recipe = str(
            self.config.trainer.get("wam_two_stage_recipe", "legacy_v1")
            if getattr(self.config, "trainer", None) is not None
            else "legacy_v1"
        ).lower()
        if wam_recipe == "causal_action_world_queries_v1":
            actual_attn = str(
                getattr(
                    self.qwen_vl_interface,
                    "attn_implementation",
                    self.config.framework.qwenvl.get("attn_implementation", ""),
                )
            ).lower()
            if actual_attn != "flash_attention_2":
                raise RuntimeError(
                    "causal_action_world_queries_v1 requires an active FlashAttention-2 Qwen backbone; "
                    f"loaded implementation={actual_attn!r}"
                )
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
        #######
        self.jointflow_enabled = bool(self.config.framework.get("jointflow", {}).get("enabled", False))
        if self.jointflow_enabled:
            self.config.framework.visual_model.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size
        #######
        #######
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
        #######
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
        if self.jointflow_enabled:
            self._init_jointflow_modules()
        #######
        #######
        if self.wam_enabled:
            recipe = str(
                self.config.trainer.get("wam_two_stage_recipe", "legacy_v1")
                if getattr(self.config, "trainer", None) is not None
                else "legacy_v1"
            ).lower()
            if recipe in {
                "baseline_preserving_v3",
                "isolated_queries_v4",
                "causal_action_world_queries_v1",
            }:
                # Auxiliary token rows, predictor, conditioners and adapters
                # need random initialization, but must not advance the policy
                # RNG beyond native baseline model construction.  All modules
                # are still on CPU at this point; runtime CUDA RNG is isolated
                # separately around every auxiliary forward.
                with torch.random.fork_rng(devices=[], enabled=True):
                    self._init_wam_modules()
            else:
                self._init_wam_modules()
        #######

    #######
    def _validate_joint_e2e_contract(self) -> None:
        """Fail before loading large backbones if a joint WAM objective is broken.

        ``joint_e2e`` preserves the original action->world gradient experiment.
        ``joint_detached`` computes the same action + world objectives on every
        batch, but requires the policy path to treat the predicted future as a
        detached condition.  Keeping distinct task names prevents a config
        merge from silently changing the gradient contract.
        """

        framework = self.config.framework
        tasks = framework.get("tasks", {})
        weights = tasks.get("weights", {}) if hasattr(tasks, "get") else {}
        active = [str(name) for name, weight in weights.items() if float(weight) > 0.0]
        joint_tasks = [name for name in active if name in {"joint_e2e", "joint_detached"}]
        if not joint_tasks:
            return
        if len(joint_tasks) != 1 or len(active) != 1:
            raise ValueError(
                "joint_e2e/joint_detached is exclusive because each batch already computes "
                f"action_loss + world_loss; active tasks={active}."
            )
        joint_task = joint_tasks[0]

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
        detach_world = bool(guidance.get("detach_world", True))
        if joint_task == "joint_e2e" and detach_world:
            problems.append("joint_e2e requires guidance.detach_world=false")
        if joint_task == "joint_detached" and not detach_world:
            problems.append("joint_detached requires guidance.detach_world=true")
        world_to_action_enabled = bool(
            guidance.get("world_to_action_enabled", True)
        )
        if joint_task == "joint_detached" and world_to_action_enabled and not bool(
            guidance.get("detached_prediction_eval_mode", False)
        ):
            problems.append(
                "joint_detached requires guidance.detached_prediction_eval_mode=true "
                "so policy training samples the world head with inference-time dropout semantics"
            )
        if not world_to_action_enabled and not bool(
            guidance.get("action_world_bypass", False)
        ):
            problems.append(
                "guidance.world_to_action_enabled=false requires "
                "guidance.action_world_bypass=true"
            )
        signal = str(guidance.get("signal", "")).lower()
        valid_signals = {"z_pred", "delta_z_pred"}
        if not world_to_action_enabled:
            valid_signals.add("none")
        if signal not in valid_signals:
            problems.append(
                "guidance.signal must be z_pred/delta_z_pred when world->action is enabled, "
                "or none when it is disabled"
            )
        if float(wam.get("dino_loss_weight", 0.0)) <= 0.0:
            problems.append("framework.wam.dino_loss_weight must be positive")
        action_cfg = framework.get("action_model", {})
        if bool(guidance.get("world_condition_on_state", False)):
            datasets_cfg = getattr(self.config, "datasets", None)
            vla_cfg = datasets_cfg.get("vla_data", {}) if datasets_cfg is not None else {}
            if int(action_cfg.get("state_dim", 0)) <= 0:
                problems.append("world_condition_on_state requires action_model.state_dim > 0")
            if not bool(vla_cfg.get("include_state", False)):
                problems.append("world_condition_on_state requires datasets.vla_data.include_state=true")
        if bool(action_cfg.get("use_correlated_noise", False)):
            problems.append("framework.action_model.use_correlated_noise must be false for the E2E experiment")

        ramp = guidance.get("action_world_gradient_ramp", {})
        ramp_enabled = hasattr(ramp, "get") and bool(ramp.get("enabled", False))
        if joint_task == "joint_detached" and ramp_enabled:
            problems.append("joint_detached forbids action_world_gradient_ramp; the action->world scale is fixed at zero")
        if ramp_enabled:
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
            raise ValueError(f"Invalid {joint_task} configuration: " + "; ".join(problems))

    def _validate_causal_query_contract(self) -> None:
        """Fail fast for standalone causal action/world query training."""

        framework = self.config.framework
        wam = framework.get("wam", {})
        guidance = wam.get("guidance", {}) if hasattr(wam, "get") else {}
        if not bool(guidance.get("causal_query_suffix", False)) or bool(
            guidance.get("world_to_action_enabled", True)
        ):
            return

        trainer = self.config.trainer
        datasets_cfg = getattr(self.config, "datasets", None)
        vla_cfg = datasets_cfg.get("vla_data", {}) if datasets_cfg is not None else {}
        action_cfg = framework.get("action_model", {})
        visual_cfg = framework.get("visual_model", {})
        dino_cfg = framework.get("dino", {})
        weights = framework.get("tasks", {}).get("weights", {})
        active = [str(name) for name, weight in weights.items() if float(weight) > 0.0]
        problems: list[str] = []

        expected_true = {
            "guidance.action_world_bypass": guidance.get("action_world_bypass", False),
            "guidance.pretraining_aligned_queries": guidance.get("pretraining_aligned_queries", False),
            "guidance.future_query_through_qwen": guidance.get("future_query_through_qwen", False),
            "framework.dino.load_live_backbone": dino_cfg.get("load_live_backbone", False),
            "framework.dino.force_online": dino_cfg.get("force_online", False),
        }
        for name, value in expected_true.items():
            if not bool(value):
                problems.append(f"{name} must be true")
        expected_false = {
            "guidance.world_condition_on_state": guidance.get("world_condition_on_state", False),
            "guidance.include_context_in_action_memory": guidance.get("include_context_in_action_memory", True),
            "guidance.include_context_in_world_memory": guidance.get("include_context_in_world_memory", True),
            "datasets.vla_data.include_state": vla_cfg.get("include_state", False),
        }
        for name, value in expected_false.items():
            if bool(value):
                problems.append(f"{name} must be false")
        if active != ["joint_detached"]:
            problems.append(f"only joint_detached may be active, got {active}")
        if str(guidance.get("mode", "")).lower() != "none":
            problems.append("guidance.mode must be none")
        if str(guidance.get("signal", "")).lower() != "none":
            problems.append("guidance.signal must be none")
        if int(action_cfg.get("state_dim", -1)) != 0:
            problems.append("action_model.state_dim must be 0")
        if int(action_cfg.get("action_horizon", 0)) != 32 or int(
            action_cfg.get("n_action_query", 0)
        ) != 32:
            problems.append("action_model requires action_horizon=n_action_query=32")
        if int(visual_cfg.get("n_flow_query", 0)) != 64:
            problems.append("visual_model.n_flow_query must be 64")
        if str(visual_cfg.get("patch_weighting", "none")).lower() != "none":
            problems.append(
                "visual_model.patch_weighting must be none so current DINO is used only by the explicit ablation"
            )
        if str(dino_cfg.get("model_size", "")).lower() not in {"base", "b", "vitb16"}:
            problems.append("framework.dino.model_size must select DINO-B")
        if not str(dino_cfg.get("weights", "")).rstrip("/").endswith("/DINO-B"):
            problems.append("framework.dino.weights must point to the DINO-B snapshot")
        if int(vla_cfg.get("per_device_batch_size", 0)) != 16:
            problems.append("datasets.vla_data.per_device_batch_size must be 16")
        if int(trainer.get("expected_global_batch_size", 0)) != 1024:
            problems.append("trainer.expected_global_batch_size must be 1024")
        if int(trainer.get("gradient_accumulation_steps", 0)) != 1:
            problems.append("trainer.gradient_accumulation_steps must be 1")
        if int(trainer.get("max_train_steps", 0)) != 80000:
            problems.append("trainer.max_train_steps must be 80000")
        if trainer.get("wam_two_stage_phase", None):
            problems.append("standalone causal query training forbids wam_two_stage_phase")
        if problems:
            raise ValueError("Invalid standalone causal query configuration: " + "; ".join(problems))

    def _validate_wam_two_stage_contract(self) -> None:
        """Fail early when the strict 80k warmup -> 20k gate-FT recipe drifts.

        This contract is opt-in through ``trainer.wam_two_stage_phase`` and
        therefore leaves historical configs/checkpoints untouched.
        """

        trainer = getattr(self.config, "trainer", None)
        phase = str(trainer.get("wam_two_stage_phase", "") if trainer is not None else "").lower()
        if not phase:
            return
        if phase not in {"predictor_warmup", "gate_ft"}:
            raise ValueError(
                "trainer.wam_two_stage_phase must be predictor_warmup or gate_ft, "
                f"got {phase!r}"
            )
        recipe = str(trainer.get("wam_two_stage_recipe", "legacy_v1") or "legacy_v1").lower()
        if recipe not in {
            "legacy_v1",
            "policy_first_v2",
            "baseline_preserving_v3",
            "isolated_queries_v4",
            "causal_action_world_queries_v1",
        }:
            raise ValueError(
                "trainer.wam_two_stage_recipe must be legacy_v1, policy_first_v2, "
                "baseline_preserving_v3, isolated_queries_v4, or causal_action_world_queries_v1, "
                f"got {recipe!r}"
            )

        framework = self.config.framework
        wam = framework.get("wam", {})
        guidance = wam.get("guidance", {})
        text_supervision = wam.get("text_supervision", {})
        weights = framework.get("tasks", {}).get("weights", {})
        active = [str(name) for name, weight in weights.items() if float(weight) > 0.0]
        action_cfg = framework.get("action_model", {})
        visual_cfg = framework.get("visual_model", {})
        dino_cfg = framework.get("dino", {})
        datasets_cfg = getattr(self.config, "datasets", None)
        vla_cfg = datasets_cfg.get("vla_data", {}) if datasets_cfg is not None else {}
        world_val = trainer.get("world_validation", {})
        problems: list[str] = []

        loss_weights = framework.get("tasks", {}).get("loss_weights", {})
        text_enabled = bool(text_supervision.get("enabled", False))
        if text_enabled:
            required_loss_names = {"action", "world", "text"}
            actual_loss_names = set(loss_weights.keys()) if hasattr(loss_weights, "keys") else set()
            if actual_loss_names != required_loss_names:
                problems.append(
                    "text-supervised WAM requires exactly tasks.loss_weights={action,world,text}"
                )
            for name in required_loss_names:
                if float(loss_weights.get(name, 0.0)) <= 0.0:
                    problems.append(f"tasks.loss_weights.{name} must be positive")
            if float(loss_weights.get("world", -1.0)) != float(wam.get("dino_loss_weight", 1.0)):
                problems.append(
                    "tasks.loss_weights.world must equal framework.wam.dino_loss_weight"
                )
            if phase != "predictor_warmup":
                problems.append("text supervision is supported only during predictor_warmup")
            if not bool(text_supervision.get("allow_missing_annotations", False)):
                problems.append(
                    "mixed annotated/unannotated RoboDojo training requires allow_missing_annotations=true"
                )
            optional_text = vla_cfg.get("optional_text_annotations", {})
            if not bool(optional_text.get("enabled", False)):
                problems.append(
                    "text supervision requires datasets.vla_data.optional_text_annotations.enabled=true"
                )
            for key in ("subtask_field", "completed_subtask_field", "response_template"):
                if not str(text_supervision.get(key, "")).strip():
                    problems.append(f"framework.wam.text_supervision.{key} must be configured")
        elif phase == "gate_ft" and hasattr(loss_weights, "get") and float(
            loss_weights.get("text", 0.0)
        ) != 0.0:
            problems.append("gate_ft freezes Qwen and therefore requires tasks.loss_weights.text=0")

        causal_query_recipe = recipe == "causal_action_world_queries_v1"
        if causal_query_recipe:
            if str(guidance.get("prompt_mode", "")).lower() != "dual_query":
                problems.append("causal_action_world_queries_v1 requires guidance.prompt_mode=dual_query")
            if not bool(guidance.get("exclude_post_query_context", False)):
                problems.append(
                    "causal_action_world_queries_v1 requires guidance.exclude_post_query_context=true"
                )
            if int(action_cfg.get("action_horizon", 0)) != 32:
                problems.append("causal_action_world_queries_v1 requires action_model.action_horizon=32")
            if int(action_cfg.get("n_action_query", 0)) != 32:
                problems.append("causal_action_world_queries_v1 requires action_model.n_action_query=32")
            if int(visual_cfg.get("n_flow_query", 0)) != 64:
                problems.append("causal_action_world_queries_v1 requires visual_model.n_flow_query=64")
            if str(guidance.get("mode", "")).lower() != "dual_xattn":
                problems.append("causal_action_world_queries_v1 requires guidance.mode=dual_xattn")
            if str(guidance.get("signal", "")).lower() != "z_pred":
                problems.append("causal_action_world_queries_v1 requires guidance.signal=z_pred")
            if not bool(guidance.get("world_to_action_enabled", False)):
                problems.append(
                    "causal_action_world_queries_v1 two-stage training requires world_to_action_enabled=true"
                )
            if str(visual_cfg.get("patch_weighting", "none")).lower() != "none":
                problems.append("causal_action_world_queries_v1 requires visual_model.patch_weighting=none")
            if str(dino_cfg.get("model_size", "")).lower() not in {"base", "b", "vitb16"}:
                problems.append("causal_action_world_queries_v1 requires online DINO-B")
            if not str(dino_cfg.get("weights", "")).rstrip("/").endswith("/DINO-B"):
                problems.append("causal_action_world_queries_v1 requires the DINO-B snapshot")
            if not bool(dino_cfg.get("load_live_backbone", False)) or not bool(
                dino_cfg.get("force_online", False)
            ):
                problems.append(
                    "causal_action_world_queries_v1 requires load_live_backbone=true and force_online=true"
                )
            if int(vla_cfg.get("per_device_batch_size", 0)) != 16:
                problems.append("causal_action_world_queries_v1 requires per_device_batch_size=16")
            if int(trainer.get("expected_global_batch_size", 0)) != 1024:
                problems.append("causal_action_world_queries_v1 requires expected_global_batch_size=1024")
            if int(trainer.get("gradient_accumulation_steps", 0)) != 1:
                problems.append("causal_action_world_queries_v1 requires gradient_accumulation_steps=1")

        if bool(action_cfg.get("use_correlated_noise", False)):
            problems.append("action_model.use_correlated_noise must be false")
        if any(str(key).startswith("correlation_") for key in action_cfg.keys()):
            problems.append("action_model must not contain correlation_* keys")
        if causal_query_recipe:
            if bool(vla_cfg.get("include_state", False)):
                problems.append(
                    "causal_action_world_queries_v1 requires datasets.vla_data.include_state=false"
                )
            if int(action_cfg.get("state_dim", -1)) != 0:
                problems.append(
                    "causal_action_world_queries_v1 requires action_model.state_dim=0"
                )
            if bool(guidance.get("world_condition_on_state", False)):
                problems.append(
                    "causal_action_world_queries_v1 requires guidance.world_condition_on_state=false"
                )
            if bool(guidance.get("include_context_in_action_memory", True)):
                problems.append(
                    "causal_action_world_queries_v1 requires include_context_in_action_memory=false"
                )
            if bool(guidance.get("include_context_in_world_memory", True)):
                problems.append(
                    "causal_action_world_queries_v1 requires include_context_in_world_memory=false"
                )
            if not bool(guidance.get("causal_query_suffix", False)):
                problems.append(
                    "causal_action_world_queries_v1 requires guidance.causal_query_suffix=true"
                )
        else:
            if not bool(vla_cfg.get("include_state", False)):
                problems.append("datasets.vla_data.include_state must be true")
            if not bool(guidance.get("world_condition_on_state", False)):
                problems.append("guidance.world_condition_on_state must be true")
        if str(guidance.get("bridge_source", "")).lower() != "predicted":
            problems.append("guidance.bridge_source must be predicted")
        if not bool(guidance.get("detach_world", False)):
            problems.append("guidance.detach_world must be true")
        if not bool(guidance.get("detached_prediction_eval_mode", False)):
            problems.append("guidance.detached_prediction_eval_mode must be true")
        if not causal_query_recipe and not bool(guidance.get("include_context_in_world_memory", False)):
            problems.append("guidance.include_context_in_world_memory must be true")
        if bool(world_val.get("enabled", False)) and int(world_val.get("interval", 0)) != 5000:
            problems.append("enabled trainer.world_validation must run every 5000 steps")

        bypass = bool(guidance.get("action_world_bypass", False))
        if phase == "predictor_warmup":
            if active != ["joint_detached"]:
                problems.append(f"predictor_warmup requires only joint_detached, got {active}")
            if not bypass:
                problems.append("predictor_warmup requires guidance.action_world_bypass=true")
            detach_action_backbone = bool(guidance.get("detach_action_backbone", False))
            detach_world_backbone = bool(guidance.get("detach_world_backbone", False))
            if recipe in {"policy_first_v2", "baseline_preserving_v3", "isolated_queries_v4"}:
                if detach_action_backbone:
                    problems.append(
                        f"{recipe} predictor_warmup requires "
                        "guidance.detach_action_backbone=false so action loss updates Qwen"
                    )
                if not detach_world_backbone:
                    problems.append(
                        f"{recipe} predictor_warmup requires "
                        "guidance.detach_world_backbone=true so world loss cannot update Qwen"
                    )
                if int(trainer.get("num_warmup_steps", 0)) != 2000:
                    problems.append(
                        f"{recipe} predictor_warmup requires num_warmup_steps=2000 "
                        "to match the IID baseline"
                    )
                if recipe == "baseline_preserving_v3" and not bool(
                    guidance.get("baseline_action_context", False)
                ):
                    problems.append(
                        "baseline_preserving_v3 requires guidance.baseline_action_context=true "
                        "so policy training/inference reuse the native baseline Qwen context"
                    )
                if recipe == "isolated_queries_v4":
                    if not bool(guidance.get("pretraining_aligned_queries", False)):
                        problems.append(
                            "isolated_queries_v4 requires guidance.pretraining_aligned_queries=true"
                        )
                    if bool(guidance.get("baseline_action_context", False)):
                        problems.append(
                            "isolated_queries_v4 requires baseline_action_context=false so the "
                            "explicit action query conditions Action DiT"
                        )
                    if not bool(guidance.get("freeze_world_to_action_in_warmup", False)):
                        problems.append(
                            "isolated_queries_v4 predictor_warmup requires "
                            "freeze_world_to_action_in_warmup=true"
                        )
                    clipping_cfg = trainer.get("independent_gradient_clipping", {})
                    if not hasattr(clipping_cfg, "get") or not bool(
                        clipping_cfg.get("enabled", False)
                    ):
                        problems.append(
                            "isolated_queries_v4 predictor_warmup requires independent gradient clipping"
                        )
                    else:
                        for branch in ("action", "world"):
                            if float(clipping_cfg.get(branch, 0.0)) <= 0.0:
                                problems.append(
                                    "isolated_queries_v4 independent_gradient_clipping."
                                    f"{branch} must be positive"
                                )
                    lr_cfg = trainer.get("learning_rate", {})
                    for name in (
                        "qwen_vl_interface",
                        "action_model",
                        "action_queries",
                        "wam_visual_head",
                        "wam_state_ctx",
                        "future_dino_queries",
                    ):
                        if float(lr_cfg.get(name, 0.0)) <= 0.0:
                            problems.append(
                                f"isolated_queries_v4 warmup requires positive learning_rate.{name}"
                            )
            elif recipe == "causal_action_world_queries_v1":
                if detach_action_backbone:
                    problems.append(
                        "causal_action_world_queries_v1 predictor_warmup requires "
                        "guidance.detach_action_backbone=false"
                    )
                if detach_world_backbone:
                    problems.append(
                        "causal_action_world_queries_v1 predictor_warmup requires "
                        "guidance.detach_world_backbone=false so world loss can shape Qwen"
                    )
                if int(trainer.get("num_warmup_steps", 0)) != 2000:
                    problems.append(
                        "causal_action_world_queries_v1 predictor_warmup requires "
                        "num_warmup_steps=2000 to match the IID baseline"
                    )
                if not bool(guidance.get("pretraining_aligned_queries", False)):
                    problems.append(
                        "causal_action_world_queries_v1 requires guidance.pretraining_aligned_queries=true"
                    )
                if bool(guidance.get("baseline_action_context", False)):
                    problems.append(
                        "causal_action_world_queries_v1 requires baseline_action_context=false so ACT queries "
                        "remain on the policy path"
                    )
                if not bool(guidance.get("future_query_through_qwen", False)):
                    problems.append(
                        "causal_action_world_queries_v1 requires guidance.future_query_through_qwen=true"
                    )
                if bool(guidance.get("separate_world_backbone_pass", False)):
                    problems.append(
                        "causal_action_world_queries_v1 is a one-forward recipe and forbids "
                        "guidance.separate_world_backbone_pass"
                    )
                if bool(guidance.get("detach_action_query_in_world_pass", False)):
                    problems.append(
                        "causal_action_world_queries_v1 uses one causal dual-query pass and forbids "
                        "the obsolete guidance.detach_action_query_in_world_pass"
                    )
                if str(framework.get("qwenvl", {}).get("attn_implementation", "")).lower() != "flash_attention_2":
                    problems.append(
                        "causal_action_world_queries_v1 requires "
                        "framework.qwenvl.attn_implementation=flash_attention_2"
                    )
                if not bool(framework.get("qwenvl", {}).get("require_attn_implementation", False)):
                    problems.append(
                        "causal_action_world_queries_v1 requires "
                        "framework.qwenvl.require_attn_implementation=true"
                    )
                if not bool(guidance.get("freeze_world_to_action_in_warmup", False)):
                    problems.append(
                        "causal_action_world_queries_v1 predictor_warmup requires "
                        "freeze_world_to_action_in_warmup=true"
                    )
                clipping_cfg = trainer.get("independent_gradient_clipping", {})
                if hasattr(clipping_cfg, "get") and bool(clipping_cfg.get("enabled", False)):
                    problems.append(
                        "causal_action_world_queries_v1 shares Qwen between action/world losses and therefore "
                        "forbids independent_gradient_clipping; use one positive global gradient_clipping"
                    )
                if float(trainer.get("gradient_clipping", 0.0) or 0.0) <= 0.0:
                    problems.append(
                        "causal_action_world_queries_v1 requires positive trainer.gradient_clipping"
                    )
                lr_cfg = trainer.get("learning_rate", {})
                for name in (
                    "qwen_vl_interface",
                    "action_model",
                    "action_queries",
                    "wam_visual_head",
                    "future_dino_queries",
                ):
                    if float(lr_cfg.get(name, 0.0)) <= 0.0:
                        problems.append(
                            f"causal_action_world_queries_v1 warmup requires positive learning_rate.{name}"
                        )
            elif not detach_action_backbone:
                # Preserve construction of already-trained strict Stage-1
                # checkpoints whose saved config predates policy_first_v2.
                problems.append(
                    "legacy_v1 predictor_warmup requires guidance.detach_action_backbone=true"
                )
            if int(trainer.get("max_train_steps", 0)) != 80000:
                problems.append("predictor_warmup requires max_train_steps=80000")
            if trainer.get("pretrained_checkpoint", None):
                problems.append("predictor_warmup must start without pretrained_checkpoint")
        else:
            if active != ["policy"]:
                problems.append(f"gate_ft requires only policy, got {active}")
            if bypass:
                problems.append("gate_ft requires guidance.action_world_bypass=false")
            if not bool(guidance.get("world_to_action_enabled", True)):
                problems.append("gate_ft requires guidance.world_to_action_enabled=true")
            if int(trainer.get("max_train_steps", 0)) != 20000:
                problems.append("gate_ft requires max_train_steps=20000")
            if not trainer.get("pretrained_checkpoint", None):
                problems.append("gate_ft requires the 80k warmup pretrained_checkpoint")
            if not bool(trainer.get("reset_world_gates_after_pretrained_load", False)):
                problems.append("gate_ft requires reset_world_gates_after_pretrained_load=true")
            if recipe in {"policy_first_v2", "baseline_preserving_v3", "isolated_queries_v4"} and not bool(
                guidance.get("detach_world_backbone", False)
            ):
                problems.append(
                    f"{recipe} gate_ft must preserve guidance.detach_world_backbone=true "
                    "from its Stage-1 parent contract"
                )
            if recipe == "baseline_preserving_v3" and not bool(
                guidance.get("baseline_action_context", False)
            ):
                problems.append(
                    "baseline_preserving_v3 gate_ft requires guidance.baseline_action_context=true"
                )
            if recipe == "isolated_queries_v4":
                if not bool(guidance.get("pretraining_aligned_queries", False)):
                    problems.append(
                        "isolated_queries_v4 requires guidance.pretraining_aligned_queries=true"
                    )
                if bool(guidance.get("baseline_action_context", False)):
                    problems.append(
                        "isolated_queries_v4 gate_ft requires baseline_action_context=false"
                    )
                if bool(guidance.get("freeze_world_to_action_in_warmup", False)):
                    problems.append(
                        "isolated_queries_v4 gate_ft must unfreeze the world-to-action path"
                    )
                lr_cfg = trainer.get("learning_rate", {})
                required_lrs = {
                    "world_gates": 1.0e-4,
                    "world_to_action_adapters": 1.0e-4,
                    "action_model": 1.0e-5,
                    "action_queries": 1.0e-5,
                }
                for name, expected in required_lrs.items():
                    actual = float(lr_cfg.get(name, 0.0))
                    if actual != expected:
                        problems.append(
                            f"isolated_queries_v4 gate_ft requires learning_rate.{name}="
                            f"{expected:g}, got {actual:g}"
                        )
            if recipe == "causal_action_world_queries_v1":
                if bool(guidance.get("detach_action_backbone", False)):
                    problems.append(
                        "causal_action_world_queries_v1 gate_ft requires "
                        "guidance.detach_action_backbone=false"
                    )
                if bool(guidance.get("detach_world_backbone", False)):
                    problems.append(
                        "causal_action_world_queries_v1 gate_ft must preserve its Stage-1 "
                        "detach_world_backbone=false provenance"
                    )
                if not bool(guidance.get("pretraining_aligned_queries", False)):
                    problems.append(
                        "causal_action_world_queries_v1 requires guidance.pretraining_aligned_queries=true"
                    )
                if bool(guidance.get("baseline_action_context", False)):
                    problems.append(
                        "causal_action_world_queries_v1 gate_ft requires baseline_action_context=false"
                    )
                if not bool(guidance.get("future_query_through_qwen", False)):
                    problems.append(
                        "causal_action_world_queries_v1 requires guidance.future_query_through_qwen=true"
                    )
                if bool(guidance.get("separate_world_backbone_pass", False)):
                    problems.append(
                        "causal_action_world_queries_v1 is a one-forward recipe and forbids "
                        "guidance.separate_world_backbone_pass"
                    )
                if bool(guidance.get("detach_action_query_in_world_pass", False)):
                    problems.append(
                        "causal_action_world_queries_v1 uses one causal dual-query pass and forbids "
                        "the obsolete guidance.detach_action_query_in_world_pass"
                    )
                if str(framework.get("qwenvl", {}).get("attn_implementation", "")).lower() != "flash_attention_2":
                    problems.append(
                        "causal_action_world_queries_v1 requires "
                        "framework.qwenvl.attn_implementation=flash_attention_2"
                    )
                if not bool(framework.get("qwenvl", {}).get("require_attn_implementation", False)):
                    problems.append(
                        "causal_action_world_queries_v1 requires "
                        "framework.qwenvl.require_attn_implementation=true"
                    )
                if bool(guidance.get("freeze_world_to_action_in_warmup", False)):
                    problems.append(
                        "causal_action_world_queries_v1 gate_ft must unfreeze the world-to-action path"
                    )
                if int(trainer.get("num_warmup_steps", 0)) != 500:
                    problems.append(
                        "causal_action_world_queries_v1 gate_ft requires num_warmup_steps=500"
                    )
                lr_cfg = trainer.get("learning_rate", {})
                required_lrs = {
                    "world_gates": 1.0e-4,
                    "world_to_action_adapters": 1.0e-4,
                    "action_model": 1.0e-5,
                    "action_queries": 1.0e-5,
                }
                for name, expected in required_lrs.items():
                    actual = float(lr_cfg.get(name, 0.0))
                    if actual != expected:
                        problems.append(
                            f"causal_action_world_queries_v1 gate_ft requires learning_rate.{name}="
                            f"{expected:g}, got {actual:g}"
                        )
            frozen = {
                item.strip()
                for item in str(trainer.get("freeze_modules", "")).split(",")
                if item.strip()
            }
            required_frozen = {
                "qwen_vl_interface",
                "wam_visual_head",
                "wam_act_ctx",
            }
            if not causal_query_recipe:
                required_frozen.add("wam_state_ctx")
            if recipe in {"isolated_queries_v4", "causal_action_world_queries_v1"}:
                required_frozen.add("future_dino_queries")
            if not required_frozen.issubset(frozen):
                problems.append(
                    "gate_ft freeze_modules must include " + ",".join(sorted(required_frozen))
                )

        if problems:
            raise ValueError(f"Invalid WAM two-stage {phase} configuration: " + "; ".join(problems))

    #######
    def _inject_guidance_dit_flags(self) -> None:
        from omegaconf import OmegaConf

        wam = self.config.framework.get("wam", {})
        g = wam.get("guidance", {}) if hasattr(wam, "get") else {}
        if not bool(g.get("enabled", False)):
            return
        if not bool(g.get("world_to_action_enabled", True)):
            return
        mode = str(g.get("mode", "none")).lower()
        if mode not in ("dual_xattn", "adaln", "dual_xattn_adaln"):
            return
        dcfg = self.config.framework.action_model.diffusion_model_cfg
        if hasattr(dcfg, "unwrap"):
            dcfg = dcfg.unwrap()
        OmegaConf.set_struct(dcfg, False)
        if mode in ("dual_xattn", "dual_xattn_adaln"):
            dcfg.world_cross_attention = True
            dcfg.world_gate_init = float(g.get("gate_init", 0.0))
        if mode in ("adaln", "dual_xattn_adaln"):
            dcfg.world_adaln = True

    #######

    def _uses_action_state(self) -> bool:
        datasets_cfg = getattr(self.config, "datasets", None)
        vla_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
        if vla_cfg is None:
            return False
        return vla_cfg.get("include_state", False) not in ["False", "false", False, 0, None]

    #######
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
    def _jointflow_language_model(self):
        model_root = self.qwen_vl_interface.model.model
        return getattr(model_root, "language_model", model_root)

    def _jointflow_embed_tokens(self):
        language_model = self._jointflow_language_model()
        #######
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
    def set_action_correlation(self, chol) -> None:
        self.action_model.set_action_correlation(chol)

    #######

    #######
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
        #######

        #######
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
    def _init_wam_modules(self) -> None:
        dino_cfg = self.config.framework.dino
        visual_cfg = self.config.framework.visual_model
        action_cfg = self.config.framework.action_model
        wam_cfg = self.config.framework.get("wam", {})
        self.action_dim = int(action_cfg.action_dim)
        self.wam_n_act = int(action_cfg.get("n_action_query", self.action_horizon))
        self.wam_n_flow = int(visual_cfg.get("n_flow_query", 8))
        task_loss_weights = self.config.framework.get("tasks", {}).get("loss_weights", {})
        self.wam_action_loss_weight = float(task_loss_weights.get("action", 1.0))
        self.wam_dino_loss_weight = float(
            task_loss_weights.get("world", wam_cfg.get("dino_loss_weight", 1.0))
        )
        self.wam_text_loss_weight = float(task_loss_weights.get("text", 0.0))
        self.wam_text_supervision = dict(wam_cfg.get("text_supervision", {}))
        #######
        self.wam_fdm_delta = bool(wam_cfg.get("fdm_delta_dino", False))
        #######
        #######
        self.wam_world_model_no_language = bool(wam_cfg.get("world_model_no_language", False))
        #######
        self.wam_ph = str(wam_cfg.get("placeholder_token", "<ACT_PH>"))
        tok = self.qwen_vl_interface.processor.tokenizer
        tok.add_special_tokens({"additional_special_tokens": [self.wam_ph]})
        self.wam_ph_id = int(tok.convert_tokens_to_ids(self.wam_ph))
        self.qwen_vl_interface.model.resize_token_embeddings(len(tok))
        self._jointflow_dino_spec = resolve_dino_spec(dino_cfg)
        self.d_dino = int(self._jointflow_dino_spec["embed_dim"])
        self.config.framework.visual_model.d_dino = self.d_dino
        self.register_buffer("_dino_mean", torch.zeros(self.d_dino), persistent=False)
        self.register_buffer("_dino_std", torch.ones(self.d_dino), persistent=False)
        self._load_jointflow_dino_stats(dino_cfg.get("stats_path", None))
        object.__setattr__(self, "_dino_teacher", None)
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
        self.wam_act_ctx = ActionContextEncoder(
            action_dim=self.action_dim, hidden_size=int(self.qwen_vl_interface.model.config.hidden_size)
        )
        #######
        self._init_wam_guidance(wam_cfg)
        #######

    #######
    @staticmethod
    def _wam_guidance_defaults() -> dict:
        return {
            "enabled": False,
            "mode": "none",
            "signal": "none",
            "prompt_mode": "action_only",
            "exclude_post_query_context": False,
            "bridge_source": "predicted",
            "detach_world": True,
            # Opt-in because historical detached runs sampled the visual head
            # in train mode.  New joint_detached runs use eval-mode sampling
            # for the action condition, exactly matching live inference, then
            # immediately restore train mode for the independent world loss.
            "detached_prediction_eval_mode": False,
            # Strict predictor warmup: compute action + world objectives on
            # the same batch, but keep the policy path identical to baseline
            # by omitting every predicted-world injection.  Stage-2 disables
            # this switch and trains the adapter/gates from predicted future.
            "action_world_bypass": False,
            # Architecture-level ablation: train the native action predictor
            # and the future-DINO predictor, but do not instantiate any
            # world->action adapter, action-DiT cross-attention, or gate.
            # True is the backward-compatible default for old WAM configs.
            "world_to_action_enabled": True,
            # Legacy Stage-1 routing can detach the action/context states from
            # Qwen. New policy_first_v2 runs leave this false and instead use
            # detach_world_backbone below, keeping the policy objective in
            # charge of the shared representation.
            "detach_action_backbone": False,
            # Policy-first Stage-1 routing: the auxiliary world loss consumes
            # a detached Qwen future-query condition.  The visual predictor
            # and state conditioner still receive gradients, while Qwen is
            # optimized only by the primary action objective.  Opt-in so old
            # checkpoints retain their original gradient contract.
            "detach_world_backbone": False,
            # Opt-in strict baseline contract.  The policy branch uses the
            # same build_qwenvl_inputs -> full Qwen hidden sequence as native
            # QwenGR00T; the dual-query pass exists only for future prediction.
            # Old checkpoints keep the historical [ACT query; context] memory.
            "baseline_action_context": False,
            # Project-native action/world query banks. ACTION replaces its
            # causal Qwen input embeddings; WORLD can specialize causal Qwen
            # states while preserving explicit gradient ownership.
            "pretraining_aligned_queries": False,
            # Run both learned query banks through one standard causal Qwen
            # forward. Their order is selected by action_query_last below.
            "future_query_through_qwen": False,
            # Put both query groups at the literal end of the causal Qwen
            # sequence. This is independent of their configured order.
            "causal_query_suffix": False,
            # New action-conditioning layout: WORLD/visual queries precede
            # ACTION queries, so causal ACTION hidden states can attend to the
            # visual-query states. False preserves old checkpoint ordering.
            "action_query_last": False,
            # Deprecated experimental two-pass switches. They remain in the
            # defaults only so saved development configs can still be loaded;
            # the causal one-pass recipe explicitly forbids them.
            "separate_world_backbone_pass": False,
            "detach_action_query_in_world_pass": False,
            # Stage-1 strict isolation freezes every parameter that exists
            # solely for world->action injection (adapter, cross-attention,
            # normalization and gates).  Stage 2 reconstructs the same model
            # with this false, loads Stage-1 weights, and trains that path.
            "freeze_world_to_action_in_warmup": False,
            "oracle_ratio": 1.0,
            "n_world_tokens": 16,
            "qformer_layers": 2,
            "fusion_layers": 2,
            "fusion_heads": 8,
            "qformer_heads": 8,
            "pooler_heads": 8,
            "gate_init": 0.0,
            # Opt-in for two-stage policy/passive gate runs. joint_e2e logs
            # these metrics unconditionally because the gate is part of its
            # core training contract.
            "log_gate_openness": False,
            "world_dropout": 0.0,
            # Explicit proprio conditioning for the visual world predictor.
            # Default false preserves old checkpoint module/state-dict ABI.
            "world_condition_on_state": False,
            # causal action/world query world conditioning uses WORLD-index Qwen states only.
            # This optional ablation appends projected online current-frame
            # DINO tokens; it never adds proprio state.
            "concat_current_dino": False,
            # Backward compatibility: historically the world-memory flag also
            # controlled whether action memory kept the Qwen image/language
            # context.  A missing action-memory override therefore inherits the
            # legacy world-memory value, preserving every existing YAML/ckpt.
            "include_context_in_action_memory": None,
            "include_context_in_world_memory": True,
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
            "world_eval_mode": "correct",
        }

    def _init_wam_guidance(self, wam_cfg) -> None:
        g_user = dict(wam_cfg.get("guidance", {})) if hasattr(wam_cfg, "get") else {}
        g = self._wam_guidance_defaults()
        g.update({k: v for k, v in g_user.items() if v is not None})
        if g["include_context_in_action_memory"] is None:
            g["include_context_in_action_memory"] = bool(g["include_context_in_world_memory"])
        self.wam_guidance = g
        self.wam_guidance_enabled = bool(g["enabled"])
        self.wam_state_ctx = None
        self.wam_current_dino_proj = None
        self.wam_pretraining_aligned_queries = False
        self.wam_future_query_through_qwen = False
        wt = str(wam_cfg.get("world_target", "")).lower() if hasattr(wam_cfg, "get") else ""
        if wt in ("absolute", "delta"):
            self.wam_fdm_delta = wt == "delta"
        if not self.wam_guidance_enabled:
            return
        mode = str(g["mode"]).lower()
        signal = str(g["signal"]).lower()
        self._wam_signal_is_latent = signal in ("z_pred", "delta_z_pred", "z_oracle", "delta_z_oracle")
        self._wam_signal_is_delta = signal in ("delta_z_pred", "delta_z_oracle")
        self._wam_signal_is_oracle = signal in ("z_oracle", "delta_z_oracle")
        if self._wam_signal_is_delta:
            self.wam_fdm_delta = True
        d_model = int(self.qwen_vl_interface.model.config.hidden_size)
        d_dino = int(self.d_dino)
        self.wam_pretraining_aligned_queries = bool(
            g.get("pretraining_aligned_queries", False)
        )
        self.wam_future_query_through_qwen = bool(
            g.get("future_query_through_qwen", False)
        )
        if self.wam_future_query_through_qwen and not self.wam_pretraining_aligned_queries:
            raise ValueError(
                "guidance.future_query_through_qwen requires pretraining_aligned_queries=true"
            )
        if self.wam_pretraining_aligned_queries:
            # Reuse the exact module classes and public parameter names from
            # project query architecture. Saved checkpoints retain the older
            # post-Qwen WORLD path unless the one-pass flag is explicit.
            if not hasattr(self, "action_queries"):
                self.action_queries = ActionQueryTokenBank(
                    action_horizon=self.wam_n_act,
                    hidden_size=d_model,
                )
            if not hasattr(self, "future_dino_queries"):
                visual_cfg = self.config.framework.visual_model
                self.future_dino_queries = FutureDinoQueryTokenBank(
                    max_queries=max(
                        self.wam_n_flow,
                        int(visual_cfg.get("max_image_queries", self.wam_n_flow)),
                    ),
                    hidden_size=d_model,
                )
        if bool(g.get("world_condition_on_state", False)):
            state_dim = int(self.config.framework.action_model.get("state_dim", 0))
            if state_dim <= 0:
                raise ValueError("guidance.world_condition_on_state requires action_model.state_dim > 0")
            self.wam_state_ctx = ActionContextEncoder(
                action_dim=state_dim,
                hidden_size=d_model,
            )
        if bool(g.get("concat_current_dino", False)):
            self.wam_current_dino_proj = DinoProjector(
                d_dino=d_dino,
                hidden_size=d_model,
                dropout=float(g["world_dropout"]),
            )
        world_in_dim = d_dino if self._wam_signal_is_latent else d_model
        from starVLA.model.framework.VLM4A.wam_guidance import (
            CompactSAFusion,
            WorldQFormer,
            WorldTokenAdapter,
            WorldTokenPooler,
        )

        drop = float(g["world_dropout"])
        self.world_adapter = None
        self.world_fusion = None
        self.world_qformer = None
        self.world_pooler = None
        world_to_action_enabled = bool(g.get("world_to_action_enabled", True))
        if signal != "none" and world_to_action_enabled:
            if mode == "qformer":
                self.world_qformer = WorldQFormer(
                    in_dim=world_in_dim,
                    out_dim=d_model,
                    n_query=int(g["n_world_tokens"]),
                    num_layers=int(g["qformer_layers"]),
                    num_heads=int(g["qformer_heads"]),
                    dropout=drop,
                )
            elif mode == "sa_fusion":
                self.world_fusion = CompactSAFusion(
                    dim=d_model,
                    num_layers=int(g["fusion_layers"]),
                    num_heads=int(g["fusion_heads"]),
                    dropout=drop,
                )
            else:
                self.world_adapter = WorldTokenAdapter(in_dim=world_in_dim, out_dim=d_model, dropout=drop)
            if mode in ("adaln", "dual_xattn_adaln"):
                self.world_pooler = WorldTokenPooler(dim=d_model, out_dim=d_model, num_heads=int(g["pooler_heads"]))
        self._wam_world_n = dino_num_patches(
            self._jointflow_dino_spec["image_size"], self._jointflow_dino_spec["patch_size"]
        )
        self.wam_future_ph = None
        self.wam_future_ph_id = None
        if str(g["prompt_mode"]).lower() == "dual_query":
            tok = self.qwen_vl_interface.processor.tokenizer
            self.wam_future_ph = "<FUTURE_PH>"
            tok.add_special_tokens({"additional_special_tokens": [self.wam_future_ph]})
            self.wam_future_ph_id = int(tok.convert_tokens_to_ids(self.wam_future_ph))
            self.qwen_vl_interface.model.resize_token_embeddings(len(tok))
        if bool(g.get("freeze_world_to_action_in_warmup", False)):
            frozen = self._set_wam_world_to_action_trainable(False)
            logger.info(
                "WAM strict warmup froze %d world-to-action parameters",
                frozen,
            )
        logger.info(
            "WAM guidance ON: mode=%s signal=%s prompt=%s exclude_post_query_context=%s "
            "bridge=%s detach_world=%s detach_action_backbone=%s detach_world_backbone=%s "
            "baseline_action_context=%s pretraining_queries=%s future_query_through_qwen=%s "
            "separate_world_pass=%s detach_act_query_in_world=%s world_to_action=%s "
            "detached_prediction_eval=%s "
            "action_context=%s world_context=%s world_state=%s "
            "world_current_dino=%s (adapter=%s qformer=%s fusion=%s pooler=%s)",
            mode,
            signal,
            g["prompt_mode"],
            bool(g["exclude_post_query_context"]),
            g["bridge_source"],
            g["detach_world"],
            bool(g["detach_action_backbone"]),
            bool(g["detach_world_backbone"]),
            bool(g["baseline_action_context"]),
            self.wam_pretraining_aligned_queries,
            self.wam_future_query_through_qwen,
            bool(g["separate_world_backbone_pass"]),
            bool(g["detach_action_query_in_world_pass"]),
            world_to_action_enabled,
            bool(g["detached_prediction_eval_mode"]),
            bool(g["include_context_in_action_memory"]),
            bool(g["include_context_in_world_memory"]),
            self.wam_state_ctx is not None,
            self.wam_current_dino_proj is not None,
            self.world_adapter is not None,
            self.world_qformer is not None,
            self.world_fusion is not None,
            self.world_pooler is not None,
        )

    @staticmethod
    def _is_wam_world_to_action_parameter(name: str) -> bool:
        """Return whether ``name`` belongs only to explicit world injection."""

        name = str(name)
        if name.startswith(("world_adapter.", "world_fusion.", "world_qformer.", "world_pooler.")):
            return True
        return any(
            marker in name
            for marker in (
                ".world_attn.",
                ".world_norm.",
                ".world_to_temb.",
            )
        ) or name.endswith(".world_gate")

    def _set_wam_world_to_action_trainable(self, trainable: bool) -> int:
        """Set trainability of only adapter/gate/cross-attention parameters."""

        count = 0
        for name, parameter in self.named_parameters():
            if self._is_wam_world_to_action_parameter(name):
                parameter.requires_grad_(bool(trainable))
                count += parameter.numel()
        return count

    def _wam_action_query_condition(self, h_act: torch.Tensor) -> torch.Tensor:
        """Return ACT states whose input tokens came from ``action_queries``.

        The actual query replacement happens before Qwen in
        ``_wam_query_embedding_override``.  Keeping this named identity helper
        documents the action-owned boundary at each downstream call site.
        """

        return h_act

    def _wam_future_query_condition(self, h_future: torch.Tensor) -> torch.Tensor:
        """Add the independently trainable FUTURE query bank, if enabled."""

        bank = getattr(self, "future_dino_queries", None)
        if not self.wam_pretraining_aligned_queries or bank is None:
            return h_future
        if bool(getattr(self, "wam_future_query_through_qwen", False)):
            # causal one-pass already substituted FUTURE input embeddings before Qwen. A
            # second post-Qwen addition would train/infer under different query
            # semantics and count the same parameter twice.
            return h_future
        query_dtype = self._jointflow_module_dtype(bank, fallback=h_future.dtype)
        query = bank(
            h_future.shape[0],
            n_query=h_future.shape[1],
            device=h_future.device,
        ).to(dtype=query_dtype)
        if query.shape != h_future.shape:
            raise ValueError(
                "FUTURE query bank shape "
                f"{tuple(query.shape)} != Qwen FUTURE states {tuple(h_future.shape)}"
            )
        return h_future + query.to(dtype=h_future.dtype)

    @contextmanager
    def _wam_query_embedding_override(
        self,
        act_mask: torch.Tensor,
        future_mask: torch.Tensor,
        *,
        detach_action_query: bool = False,
    ):
        """Replace selected placeholders with pretraining query-bank embeddings.

        Qwen still receives ``input_ids`` (required for Qwen-VL image-token
        scattering), while a temporary embedding hook substitutes ACT and,
        for causal one-pass, FUTURE positions. Standard causal attention lets
        the configured later query group read the earlier group. The detach
        argument remains solely for loading the abandoned experimental
        two-pass path. In Stage 2, gradients still reach ACT queries through
        frozen Qwen.
        """

        if not self.wam_pretraining_aligned_queries:
            yield
            return
        action_bank = getattr(self, "action_queries", None)
        if action_bank is None:
            raise RuntimeError(
                "pretraining_aligned_queries requires action_queries"
            )
        batch_size = int(act_mask.shape[0])
        action_query = action_bank(batch_size, device=act_mask.device)
        if detach_action_query:
            action_query = action_query.detach()
        future_query = None
        if bool(getattr(self, "wam_future_query_through_qwen", False)):
            future_bank = getattr(self, "future_dino_queries", None)
            if future_bank is None:
                raise RuntimeError(
                    "future_query_through_qwen requires future_dino_queries"
                )
            future_query = future_bank(
                batch_size,
                n_query=int(future_mask.sum(dim=1)[0].item()),
                device=future_mask.device,
            )
        embedding = self.qwen_vl_interface.model.get_input_embeddings()
        matched_calls = 0

        def replace_queries(_module, _args, output):
            nonlocal matched_calls
            if not torch.is_tensor(output) or output.ndim != 3:
                return output
            if tuple(output.shape[:2]) != tuple(act_mask.shape):
                return output
            if int(output.shape[-1]) != int(action_query.shape[-1]):
                raise ValueError(
                    "Qwen embedding/query hidden-size mismatch: "
                    f"embedding={tuple(output.shape)}, ACT={tuple(action_query.shape)}"
                )
            replaced = output.clone()
            replaced[act_mask.to(device=output.device)] = action_query.to(
                device=output.device,
                dtype=output.dtype,
            ).reshape(-1, output.shape[-1])
            if future_query is not None:
                if int(output.shape[-1]) != int(future_query.shape[-1]):
                    raise ValueError(
                        "Qwen embedding/FUTURE-query hidden-size mismatch: "
                        f"embedding={tuple(output.shape)}, FUTURE={tuple(future_query.shape)}"
                    )
                expected_future = int(future_query.shape[0] * future_query.shape[1])
                actual_future = int(future_mask.sum().item())
                if actual_future != expected_future:
                    raise ValueError(
                        "FUTURE query mask/query-bank count mismatch: "
                        f"mask={actual_future}, bank={expected_future}"
                    )
                replaced[future_mask.to(device=output.device)] = future_query.to(
                    device=output.device,
                    dtype=output.dtype,
                ).reshape(-1, output.shape[-1])
            matched_calls += 1
            return replaced

        handle = embedding.register_forward_hook(replace_queries)
        try:
            yield
        finally:
            handle.remove()
        if matched_calls != 1:
            raise RuntimeError(
                "Expected exactly one Qwen input-embedding call for causal ACT/FUTURE "
                f"replacement, observed {matched_calls}. Refusing to train with an inactive "
                "or multiply-applied query bank."
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

    def _wam_text_target(self, example: dict) -> tuple[str, str] | None:
        """Return the prompt/answer pair for one annotated RoboDojo sample."""

        cfg = self.wam_text_supervision
        if not bool(cfg.get("enabled", False)):
            return None
        if example.get("text_annotation_available", None) is False:
            return None
        subtask_field = str(cfg.get("subtask_field", "subtask_text"))
        completed_field = str(cfg.get("completed_subtask_field", "completed_subtask_text"))
        subtask = str(example.get(subtask_field, "") or "").strip()
        completed = str(example.get(completed_field, "") or "").strip()
        if not subtask and not completed:
            return None
        prompt_template = str(
            cfg.get(
                "prompt_template",
                "{instruction}\nReport the current subtask and the completed subtask.",
            )
        )
        response_template = str(
            cfg.get(
                "response_template",
                "Current subtask: {subtask_text}\nCompleted subtask: {completed_subtask_text}",
            )
        )
        values = {
            "instruction": str(example.get("lang", "")),
            "subtask_text": subtask,
            "completed_subtask_text": completed,
        }
        return prompt_template.format(**values), response_template.format(**values)

    @staticmethod
    def _assistant_only_labels(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Mask image/user/padding tokens, retaining only assistant targets."""

        labels = input_ids.clone()
        valid = attention_mask.to(dtype=torch.bool)
        labels[~valid] = -100
        full_lengths = valid.sum(dim=1).to(dtype=torch.long)
        width = int(labels.shape[1])
        for row in range(int(labels.shape[0])):
            if int(prompt_lengths[row]) >= int(full_lengths[row]):
                raise ValueError(
                    "Assistant-only text supervision requires at least one target token; "
                    f"row={row} prompt_length={int(prompt_lengths[row])} "
                    f"full_length={int(full_lengths[row])}"
                )
            start = width - int(full_lengths[row])
            answer_start = start + int(prompt_lengths[row])
            labels[row, : min(max(answer_start, 0), width)] = -100
        return labels

    def _wam_optional_text_loss(
        self,
        examples: List[dict],
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Compute assistant-only LM loss for annotated rows; mask missing rows."""

        selected: list[tuple[dict, str, str]] = []
        for example in examples:
            target = self._wam_text_target(example)
            if target is not None:
                selected.append((example, target[0], target[1]))
        if not selected:
            zero = reference.new_zeros(())
            return zero, zero, 0

        user_messages = []
        full_messages = []
        for example, prompt, response in selected:
            views, _future = self._wam_views(example)
            user_content = [{"type": "image", "image": image} for image in views]
            user_content.append({"type": "text", "text": prompt})
            user = {"role": "user", "content": user_content}
            user_messages.append([user])
            full_messages.append(
                [user, {"role": "assistant", "content": [{"type": "text", "text": response}]}]
            )

        processor = self.qwen_vl_interface.processor
        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            full_inputs = processor.apply_chat_template(
                full_messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                enable_thinking=_qwen_enable_thinking(getattr(self, "config", None)),
                return_dict=True,
                return_tensors="pt",
            )
            prompt_inputs = processor.apply_chat_template(
                user_messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                enable_thinking=_qwen_enable_thinking(getattr(self, "config", None)),
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            processor.tokenizer.padding_side = old_padding_side

        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        full_inputs["labels"] = self._assistant_only_labels(
            full_inputs["input_ids"],
            full_inputs["attention_mask"],
            prompt_lengths,
        )
        if not bool((full_inputs["labels"] != -100).any()):
            raise RuntimeError("Text-supervised batch produced no assistant target tokens")
        full_inputs = full_inputs.to(self.qwen_vl_interface.model.device)
        outputs = self.qwen_vl_interface(
            **full_inputs,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        raw_loss = outputs.loss
        return self.wam_text_loss_weight * raw_loss, raw_loss.detach(), len(selected)

    def _build_wam_inputs(self, examples: List[dict], task: str = "policy"):
        task = str(task)
        is_action = task in ("policy", "idm")
        n_ph = self.wam_n_act if is_action else self.wam_n_flow
        ph_str = " ".join([self.wam_ph] * n_ph)
        include_future = task == "idm"
        #######
        drop_lang = (not is_action) and bool(getattr(self, "wam_world_model_no_language", False))
        #######
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
                enable_thinking=_qwen_enable_thinking(getattr(self, "config", None)),
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            proc.tokenizer.padding_side = old
        inputs = inputs.to(self.qwen_vl_interface.model.device)
        ph_mask = inputs["input_ids"] == self.wam_ph_id
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
        attn = inputs.get("attention_mask", None)
        return h_query, hidden, attn, ph_mask
        #######

    #######
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

    def _native_qwen_action_context(
        self,
        examples: List[dict],
        *,
        resize_to_training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the exact Qwen context consumed by the native baseline.

        This helper is shared by ordinary QwenGR00T and the opt-in
        baseline-preserving WAM recipe.  Keeping one implementation prevents
        prompt, resize, mask, or Qwen-forward drift between the two paths.
        """

        batch_images = []
        for example in examples:
            if "image" in example:
                batch_images.append(example["image"])
            else:
                views, _future = self._wam_views(example)
                batch_images.append(views)
        instructions = [example["lang"] for example in examples]
        if resize_to_training:
            batch_images = [to_pil_preserve(images) for images in batch_images]
            train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
            if train_obs_image_size:
                batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
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
            last_hidden = qwenvl_outputs.hidden_states[-1]
        return last_hidden, backbone_attention_mask

    def _native_action_loss_from_context(
        self,
        examples: List[dict],
        last_hidden: torch.Tensor,
        backbone_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the native baseline action objective to a native Qwen context."""

        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        action_is_pad = (
            [example["action_is_pad"] for example in examples]
            if all("action_is_pad" in example for example in examples)
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            action_tensor = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )
            actions_target = action_tensor[:, -self.action_horizon :, :]
            action_is_pad_target = None
            if action_is_pad is not None:
                action_is_pad_target = torch.as_tensor(
                    np.asarray(action_is_pad), device=last_hidden.device, dtype=torch.bool
                )[:, -self.action_horizon :]

            repeats = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            if repeats <= 0:
                raise ValueError(f"repeated_diffusion_steps must be positive, got {repeats}")
            actions_target = actions_target.repeat(repeats, 1, 1)
            last_hidden = last_hidden.repeat(repeats, 1, 1)
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeats, 1)
            action_is_pad_target = (
                action_is_pad_target.repeat(repeats, 1)
                if action_is_pad_target is not None
                else None
            )

            state_repeated = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state_tensor.repeat(repeats, 1, 1)

            return self.action_model(
                last_hidden,
                actions_target,
                state_repeated,
                encoder_attention_mask=backbone_attention_mask,
                action_is_pad=action_is_pad_target,
            )

    def _native_predict_action_from_context(
        self,
        examples: List[dict],
        last_hidden: torch.Tensor,
        backbone_attention_mask: torch.Tensor | None,
    ) -> dict:
        """Apply the native baseline action sampler to a native Qwen context."""

        state = [example["state"] for example in examples] if "state" in examples[0] else None
        state_tensor = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden,
                state_tensor,
                encoder_attention_mask=backbone_attention_mask,
            )
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

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

    def _wam_visual_loss(
        self,
        cond: torch.Tensor,
        target: torch.Tensor,
        examples: List[dict],
        *,
        return_details: bool = False,
    ):
        """Compute the WAM loss with valid-future and spatial-change weights.

        Historical WAM YAMLs set ``visual_model.patch_weighting: change``, but
        the WAM path previously ignored it (only the older JointFlow path used
        the setting).  That made static/background patches dominate the 480
        token objective.  Apply it here while normalizing every sample back to
        mean weight one so the configured world-loss scale remains meaningful.
        ``change_balanced`` retains a full static-scene floor and is the safer
        default for structured absolute-future prediction.
        """

        valid = self._stack_jointflow_field(examples, "future_valid", required=False)
        weights = None
        valid_patch_weights = None
        if valid is not None:
            valid = valid.reshape(valid.shape[0], -1)[:, 0].clamp_(0.0, 1.0)
            normalizer = valid.new_tensor(float(valid.shape[0])) / valid.sum().clamp_min(1.0)
            valid_patch_weights = (valid * normalizer)[:, None].expand(-1, target.shape[1])

        visual_cfg = getattr(
            getattr(getattr(self, "config", None), "framework", None),
            "visual_model",
            None,
        )
        weighting = str(visual_cfg.get("patch_weighting", "none") if visual_cfg is not None else "none").lower()
        if weighting not in {"none", "change", "change_balanced"}:
            raise ValueError(
                "visual_model.patch_weighting must be none, change, or change_balanced; "
                f"got {weighting!r}"
            )
        if weighting != "none":
            z0 = self._wam_dino_target(examples, "dino_0", ["image_0", "image"])
            # Change weights always use absolute current/future DINO tokens,
            # even when the model target itself is a delta representation.
            z1 = target
            if bool(getattr(self, "wam_fdm_delta", False)):
                z1 = self._wam_dino_target(examples, "dino_1", ["image_1"])
            change = (z1.float() - z0.to(z1.device, torch.float32)).norm(dim=-1)
            relative = change / change.mean(dim=1, keepdim=True).clamp_min(1.0e-8)
            if weighting == "change":
                patch_weights = relative.clamp(0.1, 10.0)
            else:
                strength = float(visual_cfg.get("change_weight_strength", 1.0))
                if strength < 0.0:
                    raise ValueError(
                        f"visual_model.change_weight_strength must be non-negative, got {strength}"
                    )
                patch_weights = 1.0 + strength * relative.clamp(0.1, 10.0)
            patch_weights = patch_weights / patch_weights.mean(dim=1, keepdim=True).clamp_min(1.0e-8)
            weights = patch_weights.to(device=target.device)

        if valid_patch_weights is not None:
            weights = valid_patch_weights if weights is None else weights * valid_patch_weights.to(weights.device)
        if return_details:
            return self.wam_visual_head(
                cond,
                target,
                weights=weights,
                return_details=True,
            )
        return self.wam_visual_head(cond, target, weights=weights)

    #######
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
    def _wam_dino_target(self, examples: List[dict], precomp_key: str, online_keys: list[str]) -> torch.Tensor:
        datasets_cfg = getattr(self.config, "datasets", None)
        vla_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
        dino_cfg = self.config.framework.dino
        force_online = bool(dino_cfg.get("force_online", False))
        strict_precomputed = (
            bool(vla_cfg.get("require_precomputed_dino_targets", False)) if vla_cfg is not None else False
        )
        if force_online and strict_precomputed:
            raise ValueError(
                "framework.dino.force_online=true conflicts with "
                "datasets.vla_data.require_precomputed_dino_targets=true"
            )
        configured_views_raw = vla_cfg.get("dino_target_view_keys", []) if vla_cfg is not None else []
        configured_views = (
            [str(configured_views_raw)]
            if isinstance(configured_views_raw, str)
            else [str(key) for key in configured_views_raw]
        )
        z = None if force_online else self._stack_jointflow_field(examples, precomp_key, required=False)
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
        if z.shape[1] != expected_tokens:
            raise ValueError(
                f"DINO target has {z.shape[1]} tokens; expected one image/composite grid "
                f"with {expected_tokens} tokens."
            )
        if z.shape[-1] != self.d_dino:
            raise ValueError(
                f"DINO target dim {z.shape[-1]} != configured encoder dim {self.d_dino} "
                f"({self._jointflow_dino_spec['name']})."
            )
        finite = torch.isfinite(z).all()
        if z.device.type == "cuda" and hasattr(torch, "_assert_async"):
            # Avoid a full GPU synchronization on every training batch while
            # retaining a fail-fast target-integrity assertion.
            torch._assert_async(finite, f"DINO target {precomp_key!r} contains NaN or infinite values")
        elif not bool(finite):
            raise ValueError(f"DINO target {precomp_key!r} contains NaN or infinite values")
        return z

    #######

    def _wam_forward(self, examples: List[dict], task: str = "policy", **kwargs) -> dict:
        task = str(task)
        #######
        if getattr(self, "wam_guidance_enabled", False):
            return self._wam_guided_forward(examples, task=task, **kwargs)
        #######
        h_query, hidden, attn, ph_mask = self._wam_backbone(examples, task=task)
        if task in ("policy", "idm"):
            actions = self._stack_jointflow_field(examples, "action", required=True)
            actions = actions[:, -self.action_horizon :, : self.action_dim].float()
            state, action_is_pad = self._wam_action_state_and_mask(examples)
            #######
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
        if self.wam_fdm_delta:
            z_0 = self._wam_dino_target(examples, "dino_0", ["image_0", "image"])
            z_gt = z_gt - z_0.to(z_gt.device, z_gt.dtype)
            z_gt = z_gt / (z_gt.std(dim=(0, 1), keepdim=True) + 1e-6)
        #######
        vh = self._jointflow_module_dtype(self.wam_visual_head, fallback=cond.dtype)
        #######
        raw_dino = self._wam_visual_loss(cond.to(vh), z_gt, examples)
        loss = self.wam_dino_loss_weight * raw_dino
        return {
            f"{task}_loss": loss + self._wam_unused_anchor(task, loss),
            f"{task}_loss_raw": raw_dino.detach(),
        }
        #######

    @torch.inference_mode()
    def _wam_predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        #######
        if getattr(self, "wam_guidance_enabled", False):
            return self._wam_guided_predict_action(examples, **kwargs)
        #######
        #######
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
    #######################################################################

    @staticmethod
    def _wam_validate_dual_query_layout(
        act_mask: torch.Tensor,
        future_mask: torch.Tensor,
        expected_act: int,
        expected_future: int,
        action_query_last: bool = False,
    ) -> None:
        """Validate the configured causal order of visual and action queries."""

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
        if action_query_last:
            last_future = positions.masked_fill(~future_mask, -1).max(dim=1).values
            first_act = positions.masked_fill(~act_mask, act_mask.shape[1]).min(dim=1).values
            if not bool((last_future < first_act).all()):
                raise ValueError(
                    "Dual-query prompt must place every FUTURE/WORLD placeholder before every "
                    "ACT placeholder so the final ACTION queries can attend to visual queries"
                )
        else:
            last_act = positions.masked_fill(~act_mask, -1).max(dim=1).values
            first_future = positions.masked_fill(~future_mask, act_mask.shape[1]).min(dim=1).values
            if not bool((last_act < first_future).all()):
                raise ValueError(
                    "Legacy dual-query prompt must place every ACT placeholder before every "
                    "FUTURE placeholder (ACT->FUTURE)"
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

    @staticmethod
    def _wam_append_causal_query_suffix(
        inputs,
        *,
        act_token_id: int,
        future_token_id: int,
        n_act: int,
        n_future: int,
        action_query_last: bool = False,
    ):
        """Physically append both query groups after the complete chat context.

        Building placeholders inside the user-message text leaves chat-template
        terminators after FUTURE.  That is causally harmless, but it weakens the
        sequence ABI and makes it easier for a later memory builder to leak a
        post-query token.  causal one-pass therefore lets the processor finish all visual,
        language, and chat framing first, then appends the two learned-query
        blocks as the literal final suffix.  Only a standard 2D padding mask is
        produced, which keeps the native FlashAttention-2 path available.
        """

        input_ids = inputs.get("input_ids", None)
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError(
                "Causal query suffix requires processor input_ids shaped [B,T], got "
                f"{None if input_ids is None else tuple(input_ids.shape)}"
            )
        if int(n_act) <= 0 or int(n_future) <= 0:
            raise ValueError(
                f"Causal query counts must be positive, got ACT={n_act}, FUTURE={n_future}"
            )
        if bool(
            ((input_ids == int(act_token_id)) | (input_ids == int(future_token_id))).any()
        ):
            raise ValueError(
                "Reserved ACT/FUTURE placeholder token appeared inside the chat context; "
                "queries must be introduced only by the causal suffix builder"
            )

        batch, context_len = input_ids.shape
        act_suffix = input_ids.new_full((batch, int(n_act)), int(act_token_id))
        future_suffix = input_ids.new_full((batch, int(n_future)), int(future_token_id))
        suffix_parts = (
            [future_suffix, act_suffix]
            if action_query_last
            else [act_suffix, future_suffix]
        )
        suffix = torch.cat(suffix_parts, dim=1)
        inputs["input_ids"] = torch.cat([input_ids, suffix], dim=1)

        attention = inputs.get("attention_mask", None)
        if attention is None:
            attention = torch.ones_like(input_ids, dtype=torch.long)
        if not torch.is_tensor(attention) or tuple(attention.shape) != tuple(input_ids.shape):
            raise ValueError(
                "FlashAttention-2 causal query suffix requires a 2D padding mask aligned "
                f"with input_ids; ids={tuple(input_ids.shape)}, mask="
                f"{None if attention is None else tuple(attention.shape)}"
            )
        suffix_valid = torch.ones(
            (batch, int(n_act) + int(n_future)),
            dtype=attention.dtype,
            device=attention.device,
        )
        attention_2d = torch.cat([attention, suffix_valid], dim=1)
        inputs["attention_mask"] = attention_2d

        # Let Qwen recompute multimodal RoPE positions for the longer sequence.
        # Qwen processors normally omit these keys, but stale precomputed values
        # would otherwise retain the pre-suffix length.
        inputs.pop("position_ids", None)
        inputs.pop("cache_position", None)

        total_len = context_len + int(n_act) + int(n_future)
        act_mask = torch.zeros((batch, total_len), dtype=torch.bool, device=input_ids.device)
        future_mask = torch.zeros_like(act_mask)
        if action_query_last:
            future_mask[:, context_len : context_len + int(n_future)] = True
            act_mask[:, context_len + int(n_future) :] = True
        else:
            act_mask[:, context_len : context_len + int(n_act)] = True
            future_mask[:, context_len + int(n_act) :] = True
        return act_mask, future_mask, attention_2d.to(dtype=torch.bool)

    def _build_wam_guided_inputs(self, examples: List[dict], task: str = "policy"):
        """Build the configured dual-query suffix for one native causal Qwen pass.

        New configs use ``context -> FUTURE/WORLD -> ACT`` so ACTION states can
        read visual-query states. The processor's 2D padding mask is extended
        only with valid suffix positions, preserving native FlashAttention-2.
        """
        task = str(task)
        is_action = task in ("policy", "idm", "joint_e2e", "joint_detached")
        include_future = task == "idm"
        recipe = str(
            self.config.trainer.get("wam_two_stage_recipe", "legacy_v1")
            if getattr(self.config, "trainer", None) is not None
            else "legacy_v1"
        ).lower()
        physical_causal_suffix = bool(
            self.wam_guidance.get("causal_query_suffix", False)
        ) or recipe == "causal_action_world_queries_v1"
        action_query_last = bool(
            self.wam_guidance.get("action_query_last", False)
        )
        act_str = " ".join([self.wam_ph] * self.wam_n_act)
        fut_str = " ".join([self.wam_future_ph] * self.wam_n_flow)
        ph_str = (
            f"{fut_str} {act_str}"
            if action_query_last
            else f"{act_str} {fut_str}"
        )
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
            prompt_text = text if physical_causal_suffix else f"{text}\n{ph_str}"
            content.append({"type": "text", "text": prompt_text})
            messages.append([{"role": "user", "content": content}])
        old = proc.tokenizer.padding_side
        proc.tokenizer.padding_side = "left"
        try:
            inputs = proc.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                enable_thinking=_qwen_enable_thinking(getattr(self, "config", None)),
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            proc.tokenizer.padding_side = old
        inputs = inputs.to(self.qwen_vl_interface.model.device)
        if physical_causal_suffix:
            act_mask, fut_mask, attention_2d = self._wam_append_causal_query_suffix(
                inputs,
                act_token_id=self.wam_ph_id,
                future_token_id=self.wam_future_ph_id,
                n_act=self.wam_n_act,
                n_future=self.wam_n_flow,
                action_query_last=action_query_last,
            )
        else:
            # Preserve the exact prompt ABI of existing v1-v4 checkpoints.
            act_mask = inputs["input_ids"] == self.wam_ph_id
            fut_mask = inputs["input_ids"] == self.wam_future_ph_id
            attention_2d = inputs.get("attention_mask", None)
            if attention_2d is None:
                attention_2d = torch.ones_like(inputs["input_ids"], dtype=torch.bool)
            else:
                attention_2d = attention_2d.to(dtype=torch.bool)
        self._wam_validate_dual_query_layout(
            act_mask,
            fut_mask,
            expected_act=self.wam_n_act,
            expected_future=self.wam_n_flow,
            action_query_last=action_query_last,
        )
        if attention_2d.ndim != 2:
            raise ValueError(
                "WAM causal Qwen path requires a standard 2D padding mask for "
                f"FlashAttention-2, got {tuple(attention_2d.shape)}"
            )

        if bool(self.wam_guidance.get("exclude_post_query_context", False)):
            # For causal one-pass this is exactly the two suffix blocks. For legacy prompts
            # it also removes any chat-template tokens emitted after queries.
            context_exclusion_mask = self._wam_query_suffix_exclusion_mask(act_mask, fut_mask)
        else:
            # Historical behavior for old guided checkpoints/config snapshots.
            context_exclusion_mask = act_mask | fut_mask
        return inputs, act_mask, fut_mask, attention_2d, context_exclusion_mask

    def _wam_guided_backbone(
        self,
        examples: List[dict],
        task: str = "policy",
        *,
        detach_action_query: bool = False,
    ):
        """Run Qwen and return both query groups plus raw-context validity/exclusion masks."""
        inputs, act_mask, fut_mask, attention_2d, context_exclusion_mask = self._build_wam_guided_inputs(
            examples, task=task
        )
        with self._wam_query_embedding_override(
            act_mask,
            fut_mask,
            detach_action_query=detach_action_query,
        ):
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

    def _wam_visual_condition(
        self,
        h_future: torch.Tensor,
        examples: List[dict],
        *,
        task: str,
    ) -> torch.Tensor:
        """Build the one canonical condition sequence for world prediction.

        By default this is exactly the hidden states at the WORLD query
        indices.  The optional current-DINO ablation appends projected online
        patch tokens.  Proprio is appended only for legacy configs that
        explicitly opt into ``world_condition_on_state``.
        """

        h_future = self._wam_future_query_condition(h_future)
        cond_parts = [h_future]
        current_dino_proj = getattr(self, "wam_current_dino_proj", None)
        if current_dino_proj is not None:
            current_dino = self._wam_dino_target(
                examples,
                "dino_0",
                ["image_0", "image"],
            )
            projector_dtype = self._jointflow_module_dtype(
                current_dino_proj,
                fallback=h_future.dtype,
            )
            current_tokens = current_dino_proj(
                current_dino.to(projector_dtype)
            ).to(h_future.dtype)
            cond_parts.append(current_tokens)
        state_encoder = getattr(self, "wam_state_ctx", None)
        if state_encoder is not None:
            state = self._stack_jointflow_field(examples, "state", required=True)
            if state.ndim == 2:
                state = state[:, None, :]
            if state.ndim != 3:
                raise ValueError(f"World state condition must be [B,T,D], got {tuple(state.shape)}")
            state_dtype = self._jointflow_module_dtype(state_encoder, fallback=h_future.dtype)
            state_tokens = state_encoder(state.to(state_dtype)).to(h_future.dtype)
            cond_parts.append(state_tokens)

        if str(task) == "fdm":
            actions = self._stack_jointflow_field(examples, "action", required=True)
            actions = actions[:, -self.action_horizon :, : self.action_dim]
            action_dtype = self._jointflow_module_dtype(self.wam_act_ctx, fallback=h_future.dtype)
            action_tokens = self.wam_act_ctx(actions.to(action_dtype)).to(h_future.dtype)
            cond_parts.append(action_tokens)
        return torch.cat(cond_parts, dim=1) if len(cond_parts) > 1 else h_future

    def _wam_world_training_condition(
        self,
        h_future: torch.Tensor,
        examples: List[dict],
        *,
        task: str,
    ) -> torch.Tensor:
        """Build the world-loss condition with an explicit Qwen gradient ABI.

        ``detach_world`` controls gradients from *action loss* through a
        predicted future.  It does not isolate the auxiliary reconstruction
        loss from Qwen. ``detach_world_backbone`` is the complementary
        policy-first switch: only the Qwen future-query tensor is detached,
        while the visual head and the optional state conditioner remain fully
        trainable under ``world_loss``.
        """

        detach_backbone = bool(
            self.wam_guidance.get("detach_world_backbone", False)
        )
        qwen_condition = h_future.detach() if detach_backbone else h_future
        return self._wam_visual_condition(qwen_condition, examples, task=task)

    def _wam_world_target(self, examples: List[dict]) -> torch.Tensor:
        """Build the canonical future-world target for supervision and oracle input.

        Absolute mode uses ``DINO(image_1)``; delta mode standardizes
        ``DINO(image_1) - DINO(image_0)`` to approximately unit variance. Both
        passive/FDM supervision and the oracle bridge must use this exact path.
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

    def _wam_world_to_action_metrics(self, ref: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return structural zeros for the no-world-to-action baseline."""

        if not bool(self.wam_guidance.get("world_to_action_enabled", True)):
            zero = ref.detach().new_zeros(())
            return {
                "world_gate_openness": zero,
                "world_gate_signed_mean": zero,
                "world_gate_max_openness": zero,
            }
        return self._wam_world_gate_metrics()

    def _wam_maybe_logged_gate_metrics(self) -> dict[str, torch.Tensor]:
        """Expose gate state for configured two-stage policy/passive runs."""

        if not bool(self.wam_guidance.get("log_gate_openness", False)):
            return {}
        return self._wam_world_gate_metrics()

    def _build_world_signal(
        self,
        h_future: torch.Tensor,
        examples: List[dict],
        eval_mode: str = "correct",
        action_world_grad_scale: float = 1.0,
    ):
        """Produce world tokens ``[B, N_w, D_model]`` for action conditioning.

        ``guidance.signal`` selects future-query, predicted-latent, or oracle
        latent features. ``bridge_source`` selects predicted, oracle, or
        scheduled input; ``detach_world`` controls action-to-world gradients.
        The configured adapter or Q-Former projects the result, while
        ``eval_mode`` implements off, zero, and shuffled causal ablations.
        """
        from starVLA.model.framework.VLM4A.wam_guidance import (
            mix_world_tokens,
            scale_gradient,
            shuffle_along_batch,
        )

        g = self.wam_guidance
        if not bool(g.get("world_to_action_enabled", True)):
            raise RuntimeError(
                "world signal requested while guidance.world_to_action_enabled=false"
            )
        signal = str(g["signal"]).lower()
        mode = str(g["mode"]).lower()
        em = str(eval_mode or "correct").lower()
        if signal == "none" or em == "off":
            return None

        if not self._wam_signal_is_latent:
            base = h_future
        else:
            bridge = str(g["bridge_source"]).lower()
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
            predicted = None
            need_pred = (
                (not self._wam_signal_is_oracle)
                and em != "gt"
                and (bridge in ("predicted", "scheduled") or oracle is None)
            )
            if need_pred:
                n_v = oracle.shape[1] if oracle is not None else int(self._wam_world_n)
                condition_builder = getattr(self, "_wam_visual_condition", None)
                if bool(g["detach_world"]):
                    with torch.no_grad():
                        prediction_cond = (
                            condition_builder(h_future, examples, task="policy")
                            if callable(condition_builder)
                            else h_future
                        )
                        use_eval_mode = bool(g.get("detached_prediction_eval_mode", False))
                        was_training = bool(self.wam_visual_head.training)
                        if use_eval_mode and was_training:
                            self.wam_visual_head.eval()
                        try:
                            predicted = self.wam_visual_head.predict_latent(prediction_cond, n=n_v)
                        finally:
                            if use_eval_mode and was_training:
                                self.wam_visual_head.train()
                else:
                    prediction_cond = (
                        condition_builder(h_future, examples, task="policy")
                        if callable(condition_builder)
                        else h_future
                    )
                    predicted = self.wam_visual_head.predict_latent(prediction_cond, n=n_v)
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
        if em == "zero":
            base = torch.zeros_like(base)
        elif em in ("shuffled", "wrong_task"):
            base = shuffle_along_batch(base)
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
        """Assemble cross-attention memory for concat-family action heads.

        M0-Q uses ``[h_act; qwen_context]``; concat/Q-Former inserts world
        tokens; compact self-attention uses ``[SA(h_act, h_future);
        qwen_context]``. Query placeholders are excluded from raw context, and
        the original action condition is never removed.
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
        # The fallback also protects legacy in-memory harnesses/models whose
        # guidance dict predates the explicit action-memory key.
        include_action_context = g.get("include_context_in_action_memory", None)
        if include_action_context is None:
            include_action_context = g["include_context_in_world_memory"]
        if bool(include_action_context):
            context_positions = ~context_exclusion_mask.to(torch.bool)
            keep = (attn.to(torch.bool) & context_positions) if attn is not None else context_positions
            blocks.append(hidden.to(dtype))
            masks.append(keep.to(hidden.device))
        return torch.cat(blocks, dim=1), torch.cat(masks, dim=1)

    #######
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

    def _wam_action_loss_from_guided_context(
        self,
        examples: List[dict],
        *,
        mode: str,
        h_act: torch.Tensor,
        h_future: torch.Tensor,
        hidden: torch.Tensor,
        attn: torch.Tensor | None,
        context_exclusion_mask: torch.Tensor,
        world_tokens: torch.Tensor | None,
        detach_action_backbone: bool,
    ) -> torch.Tensor:
        """Compute the action objective from one dual-query Qwen pass.

        Keeping this in one helper lets causal one-pass evaluate the complete action graph
        before entering the forked auxiliary world RNG domain. It also keeps
        the legacy one-pass branch byte-for-byte on the same assembly/loss ABI.
        """

        action_h_act = h_act.detach() if detach_action_backbone else h_act
        action_h_future = h_future.detach() if detach_action_backbone else h_future
        action_hidden = hidden.detach() if detach_action_backbone else hidden
        mem, mem_mask, world_embs, world_mask, world_global = self._assemble_guided_inputs(
            mode,
            action_h_act,
            action_h_future,
            action_hidden,
            attn,
            context_exclusion_mask,
            world_tokens,
        )
        actions = self._stack_jointflow_field(examples, "action", required=True)
        actions = actions[:, -self.action_horizon :, : self.action_dim].float()
        state, action_is_pad = self._wam_action_state_and_mask(examples)
        head_dtype = self._jointflow_module_dtype(self.action_model, fallback=mem.dtype)
        return self._wam_action_loss(
            mem.to(head_dtype),
            actions.to(head_dtype),
            state.to(head_dtype) if state is not None else None,
            mem_mask,
            action_is_pad,
            world_embs=(world_embs.to(head_dtype) if world_embs is not None else None),
            world_attention_mask=world_mask,
            world_global=(world_global.to(head_dtype) if world_global is not None else None),
            guidance_mode=mode,
        )

    def _wam_uses_native_action_context(self) -> bool:
        """Whether policy conditioning must be byte-for-byte baseline shaped."""

        return bool(self.wam_guidance.get("baseline_action_context", False))

    @staticmethod
    def _wam_auxiliary_rng_context(reference: torch.Tensor):
        """Keep auxiliary future sampling from perturbing baseline policy RNG."""

        devices = []
        if reference.device.type == "cuda":
            devices = [
                reference.device.index
                if reference.device.index is not None
                else torch.cuda.current_device()
            ]
        return torch.random.fork_rng(devices=devices, enabled=True)

    def _assemble_native_guided_inputs(
        self,
        mode: str,
        native_hidden: torch.Tensor,
        native_attention_mask: torch.Tensor | None,
        world_tokens: torch.Tensor | None,
    ):
        """Add only gated world memory around the native baseline context.

        The primary DiT encoder memory stays exactly ``last_hidden`` from
        ``build_qwenvl_inputs``.  No ACT query token or dual-query suffix is
        inserted into that memory.  Stage 2 can therefore change the policy
        only through the explicitly zero-initialized world residual.
        """

        mem = native_hidden
        mem_mask = (
            native_attention_mask.to(device=mem.device, dtype=torch.bool)
            if native_attention_mask is not None
            else torch.ones(mem.shape[:2], device=mem.device, dtype=torch.bool)
        )
        world_embs, world_mask, world_global = None, None, None
        if mode in self._DIT_XATTN_MODES and world_tokens is not None:
            no_exclusions = torch.zeros_like(mem_mask, dtype=torch.bool)
            world_embs, world_mask = self._build_world_memory(
                world_tokens,
                native_hidden,
                mem_mask,
                no_exclusions,
            )
        if mode in self._DIT_ADALN_MODES and world_tokens is not None and self.world_pooler is not None:
            pd = self._jointflow_module_dtype(self.world_pooler, fallback=world_tokens.dtype)
            world_global = self.world_pooler(world_tokens.to(pd)).to(native_hidden.dtype)
        return mem, mem_mask, world_embs, world_mask, world_global

    #######

    def _wam_guided_unused_anchor(self, task: str, ref: torch.Tensor) -> torch.Tensor:
        """Anchor structurally unused modules with zero gradients for DDP/DeepSpeed."""
        g = self.wam_guidance
        mode = str(g["mode"]).lower()
        signal = str(g["signal"]).lower()
        all_mods = {
            "action": self.action_model,
            "action_query": getattr(self, "action_queries", None),
            "visual": self.wam_visual_head,
            "world_query": getattr(self, "future_dino_queries", None),
            "act_ctx": self.wam_act_ctx,
            "state_ctx": getattr(self, "wam_state_ctx", None),
            "current_dino": getattr(self, "wam_current_dino_proj", None),
            "adapter": getattr(self, "world_adapter", None),
            "fusion": getattr(self, "world_fusion", None),
            "qformer": getattr(self, "world_qformer", None),
            "pooler": getattr(self, "world_pooler", None),
        }
        used: set[str] = set()
        action_world_bypass = bool(g.get("action_world_bypass", False)) and task in {
            "joint_e2e",
            "joint_detached",
        }
        if task in ("policy", "idm", "joint_e2e", "joint_detached"):
            used.add("action")
            if self.wam_pretraining_aligned_queries:
                used.add("action_query")
            if task in ("joint_e2e", "joint_detached") and self.wam_state_ctx is not None:
                used.add("state_ctx")
            if task in ("joint_e2e", "joint_detached") and self.wam_current_dino_proj is not None:
                used.add("current_dino")
            if task in ("joint_e2e", "joint_detached") and self.wam_pretraining_aligned_queries:
                used.add("world_query")
            if signal != "none" and not action_world_bypass:
                if mode == "qformer":
                    used.add("qformer")
                elif mode == "sa_fusion":
                    used.add("fusion")
                else:
                    used.add("adapter")
                if mode in ("adaln", "dual_xattn_adaln"):
                    used.add("pooler")
                if self._wam_signal_is_latent and not self._wam_signal_is_oracle and not bool(g["detach_world"]):
                    used.add("visual")
        else:  # passive / fdm
            used.add("visual")
            if self.wam_pretraining_aligned_queries:
                used.add("world_query")
            if self.wam_state_ctx is not None:
                used.add("state_ctx")
            if self.wam_current_dino_proj is not None:
                used.add("current_dino")
            if task == "fdm":
                used.add("act_ctx")
        unused = [m for k, m in all_mods.items() if k not in used]
        return self._zero_grad_anchor_for_modules(unused, ref)

    def _wam_action_world_bypass_anchor(self, task: str, ref: torch.Tensor) -> torch.Tensor:
        """Cover action-DiT world-only parameters while Stage-1 bypasses them."""

        if task not in {"joint_e2e", "joint_detached"} or not bool(
            self.wam_guidance.get("action_world_bypass", False)
        ):
            return ref.new_zeros(())
        if not self._unused_param_anchors_enabled:
            return ref.new_zeros(())

        action_dit = getattr(self.action_model, "model", None)
        blocks = getattr(action_dit, "transformer_blocks", ())
        world_modules: list[nn.Module | None] = []
        gates: list[torch.Tensor] = []
        for block in blocks:
            world_modules.extend(
                [
                    getattr(block, "world_attn", None),
                    getattr(block, "world_to_temb", None),
                ]
            )
            gate = getattr(block, "world_gate", None)
            if gate is not None:
                gates.append(gate)
        anchor = self._zero_grad_anchor_for_modules(world_modules, ref)
        for gate in gates:
            if gate.requires_grad and gate.numel() > 0:
                anchor = anchor + gate.reshape(-1)[0].to(dtype=anchor.dtype) * 0.0
        return anchor

    def _wam_guided_forward(self, examples: List[dict], task: str = "policy", **kwargs) -> dict:
        task = str(task)
        mode = str(self.wam_guidance["mode"]).lower()
        use_native_action = self._wam_uses_native_action_context() and task in {
            "policy",
            "joint_e2e",
            "joint_detached",
        }
        native_hidden, native_attn, native_joint_action_loss = None, None, None
        query_joint_action_loss = None
        if use_native_action:
            native_hidden, native_attn = self._native_qwen_action_context(
                examples,
                resize_to_training=False,
            )
            # Compute action first, through the exact native baseline path.
            # The auxiliary future branch below runs in a forked RNG domain,
            # so it cannot shift policy noise/dropout in later optimizer steps.
            if task in {"joint_e2e", "joint_detached"}:
                native_joint_action_loss = self._native_action_loss_from_context(
                    examples,
                    native_hidden,
                    native_attn,
                )

        auxiliary_rng = (
            self._wam_auxiliary_rng_context(native_hidden)
            if use_native_action
            else nullcontext()
        )
        with auxiliary_rng:
            detach_world_backbone = bool(
                self.wam_guidance.get("detach_world_backbone", False)
            )
            future_backbone_context = (
                torch.no_grad()
                if use_native_action and detach_world_backbone
                else nullcontext()
            )
            with future_backbone_context:
                h_act, h_future, hidden, attn, context_exclusion_mask = self._wam_guided_backbone(
                    examples,
                    task=task,
                )
            # The learned ACT bank is action-owned. v4 adds FUTURE after Qwen;
            # causal one-pass already injected it at the Qwen input embedding boundary.
            action_query_condition = getattr(
                self,
                "_wam_action_query_condition",
                None,
            )
            if callable(action_query_condition):
                h_act = action_query_condition(h_act)

            if task in ("joint_e2e", "joint_detached"):
                bridge = str(self.wam_guidance["bridge_source"]).lower()
                detach_world = bool(self.wam_guidance["detach_world"])
                expected_detach = task == "joint_detached"
                if bridge != "predicted" or detach_world != expected_detach:
                    raise ValueError(
                        f"{task} requires guidance.bridge_source=predicted and "
                        f"detach_world={str(expected_detach).lower()}; got "
                        f"bridge_source={bridge!r}, detach_world={detach_world!r}."
                    )

                predictor_grad_scale = (
                    0.0
                    if detach_world
                    else self._wam_action_world_grad_scale(int(kwargs.get("global_step", 0)))
                )
                action_world_bypass = bool(self.wam_guidance.get("action_world_bypass", False))
                detach_action_backbone = bool(
                    self.wam_guidance.get("detach_action_backbone", False)
                )
                separate_world_pass = bool(
                    self.wam_guidance.get("separate_world_backbone_pass", False)
                )
                causal_shared_query_pass = bool(
                    self.wam_guidance.get("future_query_through_qwen", False)
                ) and not separate_world_pass
                if separate_world_pass:
                    if use_native_action:
                        raise RuntimeError(
                            "separate_world_backbone_pass requires the explicit ACT-query action path"
                        )
                    if not action_world_bypass:
                        raise RuntimeError(
                            "separate_world_backbone_pass is a Stage-1 warmup contract and requires "
                            "action_world_bypass=true"
                        )
                if causal_shared_query_pass and not action_world_bypass:
                    raise RuntimeError(
                        "Joint causal action/world-query warmup requires action_world_bypass=true; gate FT uses "
                        "the policy task and enables predicted-world injection there"
                    )
                if separate_world_pass or causal_shared_query_pass:
                    # Finish stochastic action diffusion first. causal one-pass reuses the
                    # same causal Qwen hidden states for both losses. With
                    # action_query_last, ACTION reads the preceding visual query
                    # states while Action DiT is called only here.
                    # The fork below isolates world-head flow noise and never
                    # reruns Qwen.
                    query_joint_action_loss = self._wam_action_loss_from_guided_context(
                        examples,
                        mode=mode,
                        h_act=h_act,
                        h_future=h_future,
                        hidden=hidden,
                        attn=attn,
                        context_exclusion_mask=context_exclusion_mask,
                        world_tokens=None,
                        detach_action_backbone=detach_action_backbone,
                    )

                world_h_future = h_future
                world_rng = (
                    self._wam_auxiliary_rng_context(h_future)
                    if separate_world_pass or causal_shared_query_pass
                    else nullcontext()
                )
                with world_rng:
                    if separate_world_pass:
                        (
                            _world_h_act,
                            world_h_future,
                            _world_hidden,
                            _world_attn,
                            _world_excluded,
                        ) = self._wam_guided_backbone(
                            examples,
                            task=task,
                            detach_action_query=bool(
                                self.wam_guidance.get(
                                    "detach_action_query_in_world_pass",
                                    False,
                                )
                            ),
                        )
                    z_gt = self._wam_world_target(examples)
                    world_cond = self._wam_world_training_condition(
                        world_h_future,
                        examples,
                        task=task,
                    )
                    vh = self._jointflow_module_dtype(
                        self.wam_visual_head,
                        fallback=world_h_future.dtype,
                    )
                    raw_world_loss, world_details = self._wam_visual_loss(
                        world_cond.to(vh),
                        z_gt,
                        examples,
                        return_details=True,
                    )
                    world_loss = self.wam_dino_loss_weight * raw_world_loss

                signal_grad_scale = 1.0 if detach_world else predictor_grad_scale
                world_tokens = None
                if not action_world_bypass:
                    world_tokens = self._build_world_signal(
                        h_future,
                        examples,
                        action_world_grad_scale=signal_grad_scale,
                    )
            elif task in ("policy", "idm"):
                world_tokens = self._build_world_signal(h_future, examples)
            else:
                z_gt = self._wam_world_target(examples)
                cond = self._wam_visual_condition(h_future, examples, task=task)
                vh = self._jointflow_module_dtype(self.wam_visual_head, fallback=cond.dtype)
                raw = self._wam_visual_loss(cond.to(vh), z_gt, examples)

        if task in ("joint_e2e", "joint_detached"):
            if use_native_action:
                raw_action_loss = native_joint_action_loss
            elif query_joint_action_loss is not None:
                raw_action_loss = query_joint_action_loss
            else:
                # Historical guided action memory remains unchanged for every
                # legacy checkpoint/config that does not opt into v3.
                raw_action_loss = self._wam_action_loss_from_guided_context(
                    examples,
                    mode=mode,
                    h_act=h_act,
                    h_future=h_future,
                    hidden=hidden,
                    attn=attn,
                    context_exclusion_mask=context_exclusion_mask,
                    world_tokens=world_tokens,
                    detach_action_backbone=detach_action_backbone,
                )
            action_loss = (
                float(getattr(self, "wam_action_loss_weight", 1.0)) * raw_action_loss
                + self._wam_guided_unused_anchor(task, raw_action_loss)
                + self._wam_action_world_bypass_anchor(task, raw_action_loss)
            )
            output = {
                "action_loss": action_loss,
                "action_loss_raw": raw_action_loss.detach(),
                "world_loss": world_loss,
                "world_loss_raw": raw_world_loss.detach(),
                "action_world_grad_scale": action_loss.detach().new_tensor(predictor_grad_scale),
                "action_world_bypassed": action_loss.detach().new_tensor(float(action_world_bypass)),
                "action_backbone_detached": action_loss.detach().new_tensor(
                    float(detach_action_backbone)
                ),
                "world_backbone_detached": action_loss.detach().new_tensor(
                    float(detach_world_backbone)
                ),
                "baseline_action_context": action_loss.detach().new_tensor(
                    float(use_native_action)
                ),
                "world_flow_loss_raw": world_details["flow_loss_raw"],
                "world_clean_loss_raw": world_details["clean_loss_raw"],
                "world_cosine_loss_raw": world_details["cosine_loss_raw"],
            }
            if bool(getattr(self, "wam_text_supervision", {}).get("enabled", False)):
                text_loss, raw_text_loss, annotated_samples = self._wam_optional_text_loss(
                    examples,
                    action_loss,
                )
                output["text_loss"] = text_loss
                output["text_loss_raw"] = raw_text_loss
                output["text_annotated_samples"] = action_loss.detach().new_tensor(
                    float(annotated_samples)
                )
            output.update(self._wam_world_to_action_metrics(action_loss))
            return output
        if task in ("policy", "idm"):
            actions = self._stack_jointflow_field(examples, "action", required=True)
            actions = actions[:, -self.action_horizon :, : self.action_dim].float()
            state, action_is_pad = self._wam_action_state_and_mask(examples)
            if use_native_action:
                mem, mem_mask, w_embs, w_mask, w_global = self._assemble_native_guided_inputs(
                    mode,
                    native_hidden,
                    native_attn,
                    world_tokens,
                )
            else:
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
            loss = float(getattr(self, "wam_action_loss_weight", 1.0)) * loss
            key = "action_loss" if task == "policy" else "idm_loss"
            output = {
                key: loss + self._wam_guided_unused_anchor(task, loss),
                "baseline_action_context": loss.detach().new_tensor(float(use_native_action)),
            }
            output.update(self._wam_maybe_logged_gate_metrics())
            return output
        loss = self.wam_dino_loss_weight * raw
        output = {
            f"{task}_loss": loss + self._wam_guided_unused_anchor(task, loss),
            f"{task}_loss_raw": raw.detach(),
        }
        # Log on passive/fdm steps too. This is a detached read of the same
        # gate parameter, so it keeps a continuous W&B curve without changing
        # either loss or gradient flow.
        output.update(self._wam_maybe_logged_gate_metrics())
        return output

    @torch.inference_mode()
    def evaluate_wam_world_prediction(
        self,
        examples: List[dict],
        *,
        task: str = "joint_detached",
        seed: int = 0,
        num_samples: int = 1,
        num_inference_timesteps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return reducible held-out metrics for the sampled future DINO.

        This evaluates the quantity used by policy inference, not the training
        velocity loss: sample a future latent from IID Gaussian noise, compare
        it directly with the held-out t+stride DINO tokens, and also report a
        no-change (copy-current) baseline.  All values are sums/counts so the
        trainer can all-reduce them exactly across data-parallel workers.
        """

        if not isinstance(examples, list):
            examples = [examples]
        if num_samples <= 0:
            raise ValueError(f"world validation num_samples must be positive, got {num_samples}")

        _h_act, h_future, _hidden, _attn, _excluded = self._wam_guided_backbone(
            examples, task=str(task)
        )
        target = self._wam_world_target(examples)
        cond = self._wam_visual_condition(h_future, examples, task=str(task))

        valid = self._stack_jointflow_field(examples, "future_valid", required=False)
        if valid is None:
            valid_mask = torch.ones(target.shape[0], device=target.device, dtype=torch.bool)
        else:
            valid_mask = valid.reshape(valid.shape[0], -1)[:, 0] > 0.5

        device = cond.device
        result = torch.zeros(6, device=device, dtype=torch.float64)
        if not bool(valid_mask.any()):
            return {
                "squared_error_sum": result[0],
                "element_count": result[1],
                "cosine_sum": result[2],
                "token_count": result[3],
                "copy_squared_error_sum": result[4],
                "valid_sample_count": result[5],
            }

        target_valid = target[valid_mask].float()
        if bool(getattr(self, "wam_fdm_delta", False)):
            copy_prediction = torch.zeros_like(target_valid)
        else:
            current = self._wam_dino_target(examples, "dino_0", ["image_0", "image"])
            copy_prediction = current[valid_mask].float()
        copy_sse = (copy_prediction - target_valid).square().sum(dtype=torch.float64)

        head_dtype = self._jointflow_module_dtype(self.wam_visual_head, fallback=cond.dtype)
        for sample_index in range(int(num_samples)):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed) + sample_index)
            prediction = self.wam_visual_head.predict_latent(
                cond.to(head_dtype),
                n=int(target.shape[1]),
                generator=generator,
                num_inference_timesteps=num_inference_timesteps,
            )
            prediction_valid = prediction[valid_mask].float()
            result[0] += (prediction_valid - target_valid).square().sum(dtype=torch.float64)
            result[1] += float(target_valid.numel())
            result[2] += torch.nn.functional.cosine_similarity(
                prediction_valid, target_valid, dim=-1, eps=1.0e-8
            ).sum(dtype=torch.float64)
            result[3] += float(target_valid.shape[0] * target_valid.shape[1])
            result[4] += copy_sse
            result[5] += float(target_valid.shape[0])

        return {
            "squared_error_sum": result[0],
            "element_count": result[1],
            "cosine_sum": result[2],
            "token_count": result[3],
            "copy_squared_error_sum": result[4],
            "valid_sample_count": result[5],
        }

    @torch.inference_mode()
    def _wam_guided_predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        mode = str(self.wam_guidance["mode"]).lower()
        use_native_action = self._wam_uses_native_action_context()
        if use_native_action:
            native_hidden, native_attn = self._native_qwen_action_context(
                examples,
                resize_to_training=True,
            )
            if bool(self.wam_guidance.get("action_world_bypass", False)):
                # Stage-1 evaluation is the native baseline policy, including
                # its prompt, context mask and action sampler.  No auxiliary
                # future pass is needed when the explicit residual is closed.
                return self._native_predict_action_from_context(
                    examples,
                    native_hidden,
                    native_attn,
                )
            with self._wam_auxiliary_rng_context(native_hidden):
                _h_act, h_future, _hidden, _attn, _excluded = self._wam_guided_backbone(
                    examples,
                    task="policy",
                )
                eval_mode = str(self.wam_guidance.get("world_eval_mode", "correct")).lower()
                world_tokens = self._build_world_signal(h_future, examples, eval_mode=eval_mode)
            mem, mem_mask, w_embs, w_mask, w_global = self._assemble_native_guided_inputs(
                mode,
                native_hidden,
                native_attn,
                world_tokens,
            )
        else:
            h_act, h_future, hidden, attn, context_exclusion_mask = self._wam_guided_backbone(
                examples, task="policy"
            )
            action_query_condition = getattr(
                self,
                "_wam_action_query_condition",
                None,
            )
            if callable(action_query_condition):
                h_act = action_query_condition(h_act)
            if bool(self.wam_guidance.get("action_world_bypass", False)):
                world_tokens = None
            else:
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
        if self.wam_enabled:
            return self._wam_forward(examples, **kwargs)
        if self.jointflow_enabled:
            return self._jointflow_forward(examples, task=str(kwargs.get("task", "policy")))
        #######
        last_hidden, backbone_attention_mask = self._native_qwen_action_context(
            examples,
            resize_to_training=False,
        )
        action_loss = self._native_action_loss_from_context(
            examples,
            last_hidden,
            backbone_attention_mask,
        )
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
        if self.wam_enabled:
            return self._wam_predict_action(examples=examples, **kwargs)
        if self.jointflow_enabled:
            return self._jointflow_predict_action(examples=examples, **kwargs)
        #######
        if type(examples) is not list:
            examples = [examples]
        last_hidden, backbone_attention_mask = self._native_qwen_action_context(
            examples,
            resize_to_training=True,
        )
        return self._native_predict_action_from_context(
            examples,
            last_hidden,
            backbone_attention_mask,
        )


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
