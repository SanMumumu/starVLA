"""Opt-in Qwen3-VL Physically-Aligned Action--World Co-Flow framework."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.action_world_coflow import (
    ActionWorldCoFlowModel,
    FrozenQwenMultiLayerVisionLatent,
)
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenActionWorldCoFlowDefaultConfig(QwenGR00TDefaultConfig):
    """Defaults are isolated from ``QwenGR00T`` and are opt-in by name+flag."""

    name: str = "QwenActionWorldCoFlow"
    enable_action_world_coflow: bool = False
    action_model: dict = field(
        default_factory=lambda: {
            "action_dim": 14,
            "state_dim": 14,
            "action_horizon": 16,
            "repeated_diffusion_steps": 1,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "prediction_type": "jit_x",
            "jit_t_eps": 5.0e-2,
            "flow_time_sampling": "gr00t",
            "use_correlated_noise": False,
        }
    )
    action_world_coflow: dict = field(
        default_factory=lambda: {
            "freeze_qwen_vision_encoder": True,
            "world_feature_layer_strategy": "evenly_spaced",
            "world_feature_num_layers": 4,
            "world_feature_layers": None,
            "world_layer_fusion_type": "fixed_mean",
            "world_feature_normalization": "layernorm",
            "world_spatial_pool_type": "adaptive_avg_pool2d",
            "world_token_grid": [4, 8],
            "num_world_tokens": 32,
            "world_bridge_type": "qantara_brownian_bridge",
            "bridge_noise_scale": 1.0,
            "world_prediction_type": "qantara_x_delta",
            "action_timestep_sampling": "starvla_gr00t",
            "world_timestep_sampling": "qantara_monotone",
            "action_loss_weight": 1.0,
            "world_loss_weight": 0.1,
            "noise_plane_sampling": {
                "policy_ratio": 0.35,
                "forward_ratio": 0.20,
                "inverse_ratio": 0.10,
                "joint_ratio": 0.15,
                "diagonal_ratio": 0.20,
            },
            "action_inference_steps": 10,
            "world_inference_steps": 10,
            "default_inference_mode": "policy",
            "hidden_size": 1024,
            "num_layers": 12,
            "num_attention_heads": 16,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "enable_gradient_checkpointing": True,
            "log_attention_statistics": True,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenActionWorldCoFlow")
class QwenActionWorldCoFlow(baseframework):
    """Independent framework; old QwenGR00T/WAM construction is untouched."""

    def __init__(self, config: Optional[dict] = None, **_kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenActionWorldCoFlowDefaultConfig, config)
        framework_cfg = self.config.framework
        if not bool(framework_cfg.get("enable_action_world_coflow", False)):
            raise ValueError(
                "QwenActionWorldCoFlow requires framework.enable_action_world_coflow=true; "
                "use framework.name=QwenGR00T for every legacy path"
            )
        if bool(framework_cfg.get("wam", {}).get("enabled", False)) or bool(
            framework_cfg.get("jointflow", {}).get("enabled", False)
        ):
            raise ValueError("Action--World Co-Flow is an independent model type and cannot enable WAM/jointflow")

        coflow_cfg = framework_cfg.action_world_coflow
        action_horizon = int(framework_cfg.action_model.get("action_horizon", 16))
        if action_horizon != 16:
            raise ValueError(
                "RoboTwin Action--World Co-Flow is a single H16 bridge; "
                f"got action_horizon={action_horizon}"
            )
        removed_multibridge_keys = {
            "segment_boundaries",
            "future_strides",
            "z32_loss_weight",
            "intermediate_state_source",
            "predicted_z16_detach",
            "predicted_action_prefix_detach",
            "z16_teacher_ratio_start",
            "z16_teacher_ratio_end",
            "z16_teacher_decay_steps",
        }
        configured_removed = sorted(key for key in removed_multibridge_keys if key in coflow_cfg)
        if configured_removed:
            raise ValueError(
                "Removed multi-bridge Co-Flow fields are not accepted by the single-bridge model: "
                f"{configured_removed}"
            )
        datasets_cfg = getattr(self.config, "datasets", None)
        vla_data_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
        if vla_data_cfg is not None:
            if not bool(vla_data_cfg.get("fastwam_action_world_coflow_targets", False)):
                raise ValueError(
                    "QwenActionWorldCoFlow requires "
                    "datasets.vla_data.fastwam_action_world_coflow_targets=true"
                )
            if bool(vla_data_cfg.get("fastwam_wam_targets", False)):
                raise ValueError("Co-Flow and legacy fastwam_wam_targets sample ABIs cannot be enabled together")
            if "fastwam_coflow_future_strides" in vla_data_cfg:
                raise ValueError(
                    "datasets.vla_data.fastwam_coflow_future_strides was removed; "
                    "single-bridge Co-Flow always uses the t+16 target"
                )
            if str(vla_data_cfg.get("data_mix", "")) != "robotwin_fastwam_h16":
                raise ValueError("single-bridge Co-Flow requires data_mix=robotwin_fastwam_h16")

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        hf_model = self.qwen_vl_interface.model
        generation_head = getattr(hf_model, "lm_head", None)
        input_embeddings = hf_model.get_input_embeddings()
        if generation_head is not None:
            # Co-Flow consumes the base model's context states and never
            # computes vocabulary logits.  Freeze every generation-head-only
            # parameter so it cannot become an unused DeepSpeed parameter;
            # tied Qwen checkpoints keep the shared input-embedding weight
            # trainable because that exact Parameter object is used below.
            for parameter in generation_head.parameters():
                if parameter is not input_embeddings.weight:
                    parameter.requires_grad_(False)
        qwen_backbone = getattr(hf_model, "model", None)
        visual = getattr(qwen_backbone, "visual", None)
        if visual is None or not hasattr(visual, "blocks"):
            raise TypeError(
                "QwenActionWorldCoFlow currently requires HuggingFace Qwen3-VL with model.visual.blocks"
            )
        self.freeze_qwen_vision_encoder = bool(coflow_cfg.get("freeze_qwen_vision_encoder", True))
        if self.freeze_qwen_vision_encoder:
            for parameter in visual.parameters():
                parameter.requires_grad_(False)
            visual.eval()

        explicit_layers = coflow_cfg.get("world_feature_layers", None)
        explicit_layers = list(explicit_layers) if explicit_layers is not None else None
        fixed_weights = coflow_cfg.get("fixed_layer_weights", None)
        fixed_weights = list(fixed_weights) if fixed_weights is not None else None
        self.world_target_extractor = FrozenQwenMultiLayerVisionLatent(
            vision_depth=len(visual.blocks),
            strategy=str(coflow_cfg.get("world_feature_layer_strategy", "evenly_spaced")),
            num_layers=int(coflow_cfg.get("world_feature_num_layers", 4)),
            explicit_layers=explicit_layers,
            fusion_type=str(coflow_cfg.get("world_layer_fusion_type", "fixed_mean")),
            fixed_layer_weights=fixed_weights,
            normalization=str(coflow_cfg.get("world_feature_normalization", "layernorm")),
            token_grid=tuple(int(value) for value in coflow_cfg.get("world_token_grid", [4, 8])),
            spatial_pool_type=str(coflow_cfg.get("world_spatial_pool_type", "adaptive_avg_pool2d")),
        )
        configured_world_tokens = int(coflow_cfg.get("num_world_tokens", 32))
        if configured_world_tokens != self.world_target_extractor.num_world_tokens:
            raise ValueError(
                f"num_world_tokens={configured_world_tokens} disagrees with extractor grid "
                f"({self.world_target_extractor.num_world_tokens})"
            )

        context_dim = int(hf_model.config.text_config.hidden_size)
        world_dim = int(visual.config.hidden_size)
        # Keep the conventional ``action_model`` name for LR grouping and
        # deployment contracts, but it is the joint co-flow model, not GR00T's
        # separate Action DiT.
        self.action_model = ActionWorldCoFlowModel(
            context_dim=context_dim,
            world_dim=world_dim,
            action_config=framework_cfg.action_model,
            coflow_config=coflow_cfg,
        )
        self.action_horizon = self.action_model.action_horizon
        self.action_dim = self.action_model.action_dim

    def _qwen_backbone_model(self):
        # Resolve through the owning Qwen wrapper every time instead of
        # registering the same module under a second attribute/state-dict path.
        return self.qwen_vl_interface.model.model

    def _visual(self):
        return self._qwen_backbone_model().visual

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_qwen_vision_encoder:
            # ``nn.Module.train`` recurses into frozen children; restore a
            # deterministic teacher even while the language backbone trains.
            self._visual().eval()
        return self

    def _uses_action_state(self) -> bool:
        datasets_cfg = getattr(self.config, "datasets", None)
        vla_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
        return bool(vla_cfg.get("include_state", False)) if vla_cfg is not None else False

    def set_action_correlation(self, chol: torch.Tensor) -> None:
        self.action_model.set_action_correlation(chol)

    def _resize_batch_images(self, images):
        target_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        converted = [[to_pil_preserve(image) for image in sample] for sample in images]
        return resize_images(converted, target_size=tuple(target_size)) if target_size else converted

    def _autocast_context(self):
        try:
            device = next(self.qwen_vl_interface.parameters()).device
        except StopIteration:
            return nullcontext()
        if device.type != "cuda":
            return nullcontext()
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def _encode_current_context(
        self,
        images,
        instructions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = self._resize_batch_images(images)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
        grid_thw = qwen_inputs.get("image_grid_thw")
        if grid_thw is None or grid_thw.shape[0] != len(images):
            raise ValueError(
                "Action--World Co-Flow requires exactly one composite image per sample; "
                f"batch={len(images)}, image_grid_thw={getattr(grid_thw, 'shape', None)}"
            )
        with self.world_target_extractor.capture(self._visual()) as captured:
            with self._autocast_context():
                outputs = self._qwen_backbone_model()(
                    **qwen_inputs,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
        context = outputs.last_hidden_state
        context_valid = qwen_inputs.get("attention_mask")
        context_valid = (
            torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
            if context_valid is None
            else context_valid.to(device=context.device, dtype=torch.bool)
        )
        z0 = self.world_target_extractor.pool_captured(
            captured,
            grid_thw,
            int(self._visual().config.spatial_merge_size),
        )
        return context, context_valid, z0

    def _encode_future_target(self, examples: List[dict]) -> torch.Tensor:
        required = ["image_16", "future_valid_16"]
        missing = [key for key in required if not all(key in example for example in examples)]
        if missing:
            raise KeyError(
                "Action--World Co-Flow training requires the single t+16 FastWAM target; "
                f"missing={missing}. Set fastwam_action_world_coflow_targets=true."
            )
        forbidden = [
            key
            for key in ("image_32", "future_valid_32", "coflow_future_strides")
            if any(key in example for example in examples)
        ]
        if forbidden:
            raise ValueError(
                "Single-bridge Co-Flow received removed multi-bridge fields: "
                f"{forbidden}"
            )
        batch = len(examples)
        resized = self._resize_batch_images([example["image_16"] for example in examples])
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=resized,
            instructions=[""] * batch,
        )
        grid_thw = inputs.get("image_grid_thw")
        if grid_thw is None or grid_thw.shape[0] != batch:
            raise ValueError(
                "future Qwen encoding requires exactly one t+16 composite per sample; "
                f"expected={batch}, got={getattr(grid_thw, 'shape', None)}"
            )
        # The world target is a frozen teacher with an explicit detach boundary.
        with torch.no_grad():
            with self._autocast_context():
                target = self.world_target_extractor.encode(
                    self._visual(),
                    inputs["pixel_values"].to(dtype=next(self._visual().parameters()).dtype),
                    grid_thw,
                )
        return target.detach()

    def _stack_state(self, examples: List[dict], device, dtype) -> torch.Tensor | None:
        if not self._uses_action_state():
            return None
        if not all("state" in example for example in examples):
            raise KeyError("checkpoint/data contract requires state, but at least one example has no state")
        return torch.as_tensor(
            np.stack([np.asarray(example["state"]) for example in examples]),
            device=device,
            dtype=dtype,
        )

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        if not isinstance(examples, list) or not examples:
            raise ValueError("QwenActionWorldCoFlow.forward expects a non-empty list of examples")
        current_images = [example["image"] for example in examples]
        instructions = [str(example["lang"]) for example in examples]
        context, context_valid, z0 = self._encode_current_context(current_images, instructions)
        z16_target = self._encode_future_target(examples)
        model_dtype = next(self.action_model.parameters()).dtype
        context = context.to(dtype=model_dtype)
        z0 = z0.to(device=context.device, dtype=model_dtype)
        actions = torch.as_tensor(
            np.stack([np.asarray(example["action"]) for example in examples]),
            device=context.device,
            dtype=model_dtype,
        )[:, : self.action_horizon, : self.action_dim]
        if not all("action_is_pad" in example for example in examples):
            raise KeyError(
                "Action--World Co-Flow training requires action_is_pad for every sample so padded "
                "episode-tail actions are excluded from both loss and attention keys"
            )
        action_is_pad = torch.as_tensor(
            np.stack([np.asarray(example["action_is_pad"]) for example in examples]),
            device=context.device,
            dtype=torch.bool,
        )[:, : self.action_horizon]
        state = self._stack_state(examples, context.device, model_dtype)
        future_valid_16 = torch.as_tensor(
            [example["future_valid_16"] for example in examples], device=context.device
        )
        return self.action_model.forward_train(
            context=context,
            context_valid=context_valid,
            z0=z0,
            z16_target=z16_target.to(device=context.device, dtype=model_dtype),
            actions=actions,
            state=state,
            action_is_pad=action_is_pad,
            future_valid_16=future_valid_16,
            global_step=int(kwargs.get("global_step", 0)),
        )

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("QwenActionWorldCoFlow.predict_action expects at least one example")
        # There is deliberately no code path accepting image_16 or a
        # future latent here: deployment can only provide current observation.
        current_images = [example["image"] for example in examples]
        instructions = [str(example["lang"]) for example in examples]
        context, context_valid, z0 = self._encode_current_context(current_images, instructions)
        model_dtype = next(self.action_model.parameters()).dtype
        context = context.to(dtype=model_dtype)
        z0 = z0.to(device=context.device, dtype=model_dtype)
        state = self._stack_state(examples, context.device, model_dtype)
        actions, _z16_prediction = self.action_model.sample_actions(
            context=context,
            context_valid=context_valid,
            z0=z0,
            state=state,
            inference_mode=kwargs.get("coflow_inference_mode"),
            output_horizon=kwargs.get("coflow_inference_horizon"),
            inference_seed=kwargs.get("coflow_inference_seed"),
        )
        return {"normalized_actions": actions.float().cpu().numpy()}
