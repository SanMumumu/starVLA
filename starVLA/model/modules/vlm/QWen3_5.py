# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2026].
# Design and Merged by [Jinhui YE / HKUST University] in [2026].

import importlib
import os
from typing import Optional

import torch
import torch.nn.functional as F
from starVLA.model.tools import has_flash_attn  # unified flash-attn detection (GPU / NPU)
from starVLA.model.modules.vlm.rynn_mem_encoder import (
    apply_rynn_mem_encoder_patch,
)
from starVLA.training.trainer_utils import initialize_overwatch
import transformers
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 248056
VIDEO_TOKEN_INDEX = 248057
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 248077  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    248077 + 2047
)  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean value, got {value!r}"
    )


def _resolve_qwen35_attn_implementation(configured: str) -> tuple[str, str]:
    """Resolve an optional inference-only attention backend override."""

    configured = str(configured).strip().lower()
    override = os.getenv("STARVLA_QWEN35_ATTN_IMPLEMENTATION", "").strip().lower()
    if override:
        allowed = {"flash_attention_2", "sdpa", "eager"}
        if override not in allowed:
            raise ValueError(
                "STARVLA_QWEN35_ATTN_IMPLEMENTATION must be one of "
                f"{sorted(allowed)}, got {override!r}"
            )
        return override, "environment"
    return configured, "config"


def _apply_causal_activation(
    hidden_states: torch.Tensor,
    activation: str | None,
) -> torch.Tensor:
    normalized = str(activation or "silu").lower()
    if normalized in {"silu", "swish"}:
        return F.silu(hidden_states)
    if normalized == "relu":
        return F.relu(hidden_states)
    raise ValueError(
        "Qwen3.5 safe causal-conv fallback supports silu/swish/relu, "
        f"got activation={activation!r}"
    )


def _safe_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    seq_idx=None,
    **_kwargs,
) -> torch.Tensor:
    """Torch equivalent of Qwen3.5's full-sequence causal depthwise conv."""

    if seq_idx is not None:
        raise RuntimeError(
            "Qwen3.5 safe causal-conv fallback does not support packed seq_idx"
        )
    if x.ndim != 3 or weight.ndim != 2 or weight.shape[0] != x.shape[1]:
        raise ValueError(
            "Qwen3.5 causal-conv expects x=[B,C,T], weight=[C,K], got "
            f"x={tuple(x.shape)}, weight={tuple(weight.shape)}"
        )
    sequence_length = int(x.shape[-1])
    output = F.conv1d(
        x,
        weight.unsqueeze(1),
        bias,
        padding=int(weight.shape[-1]) - 1,
        groups=int(x.shape[1]),
    )[..., :sequence_length]
    return _apply_causal_activation(output, activation).to(dtype=x.dtype)


def _safe_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    **_kwargs,
) -> torch.Tensor:
    """Torch equivalent of Qwen3.5's cached causal-conv update."""

    if x.ndim != 3 or conv_state.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "Qwen3.5 cached causal-conv expects x/state rank 3 and weight rank 2"
        )
    sequence_length = int(x.shape[-1])
    state_length = int(conv_state.shape[-1])
    combined = torch.cat([conv_state, x], dim=-1).to(dtype=weight.dtype)
    conv_state.copy_(combined[..., -state_length:])
    output = F.conv1d(
        combined,
        weight.unsqueeze(1),
        bias,
        groups=int(x.shape[1]),
    )[..., -sequence_length:]
    return _apply_causal_activation(output, activation).to(dtype=x.dtype)


def _qwen35_modeling_module(model_or_class):
    module_name = getattr(model_or_class, "__module__", None)
    if module_name is None:
        module_name = model_or_class.__class__.__module__
    return importlib.import_module(module_name)


def _prepare_safe_qwen35_kernels(
    model_cls,
    *,
    safe_causal_conv: bool,
    safe_fla: bool,
):
    """Select reference kernels before model construction when possible."""

    modeling_module = _qwen35_modeling_module(model_cls)
    if safe_causal_conv:
        if hasattr(modeling_module, "causal_conv1d_fn"):
            modeling_module.causal_conv1d_fn = _safe_causal_conv1d_fn
        if hasattr(modeling_module, "causal_conv1d_update"):
            modeling_module.causal_conv1d_update = _safe_causal_conv1d_update
    if safe_fla:
        for name in (
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
            "FusedRMSNormGated",
        ):
            if hasattr(modeling_module, name):
                setattr(modeling_module, name, None)
        if hasattr(modeling_module, "is_fast_path_available"):
            modeling_module.is_fast_path_available = False
    return modeling_module


def _install_safe_qwen35_causal_conv(
    model: nn.Module,
    modeling_module=None,
) -> int:
    """Replace only causal-conv CUDA extensions; retain FA2 and FLA."""

    modeling_module = (
        _qwen35_modeling_module(model)
        if modeling_module is None
        else modeling_module
    )
    # Newer Transformers calls module globals, while 5.2 stores the functions
    # on every GatedDeltaNet instance. Cover both without touching attention or
    # gated-delta-rule kernels.
    if hasattr(modeling_module, "causal_conv1d_fn"):
        modeling_module.causal_conv1d_fn = _safe_causal_conv1d_fn
    if hasattr(modeling_module, "causal_conv1d_update"):
        modeling_module.causal_conv1d_update = _safe_causal_conv1d_update

    replaced = 0
    for module in model.modules():
        if not hasattr(module, "causal_conv1d_fn"):
            continue
        module.causal_conv1d_fn = _safe_causal_conv1d_fn
        module.causal_conv1d_update = _safe_causal_conv1d_update
        replaced += 1
    return replaced


def _install_safe_qwen35_fla(
    model: nn.Module,
    modeling_module=None,
) -> int:
    """Use HF reference GatedDeltaNet/RMSNorm while retaining FA2."""

    modeling_module = (
        _qwen35_modeling_module(model)
        if modeling_module is None
        else modeling_module
    )
    chunk_reference = getattr(
        modeling_module,
        "torch_chunk_gated_delta_rule",
        None,
    )
    recurrent_reference = getattr(
        modeling_module,
        "torch_recurrent_gated_delta_rule",
        None,
    )
    norm_reference = getattr(
        modeling_module,
        "Qwen3_5RMSNormGated",
        None,
    )
    if not all(
        callable(reference)
        for reference in (
            chunk_reference,
            recurrent_reference,
            norm_reference,
        )
    ):
        raise RuntimeError(
            "The active Transformers Qwen3.5 implementation does not expose "
            "the reference GatedDeltaNet kernels required by "
            "STARVLA_QWEN35_DISABLE_FLA=1"
        )

    replaced = 0
    for module in model.modules():
        if not hasattr(module, "chunk_gated_delta_rule"):
            continue
        module.chunk_gated_delta_rule = chunk_reference
        module.recurrent_gated_delta_rule = recurrent_reference
        current_norm = getattr(module, "norm", None)
        if current_norm is not None and not isinstance(
            current_norm,
            norm_reference,
        ):
            weight = getattr(current_norm, "weight", None)
            if not torch.is_tensor(weight):
                raise RuntimeError(
                    "Cannot replace Qwen3.5 fused gated RMSNorm without weight"
                )
            safe_norm = norm_reference(
                int(module.head_v_dim),
                eps=float(module.layer_norm_epsilon),
            ).to(device=weight.device, dtype=weight.dtype)
            with torch.no_grad():
                safe_norm.weight.copy_(weight)
            module.norm = safe_norm
        replaced += 1
    return replaced


class _QWen3_5_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3.5-VL (Qwen3_5ForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3.5-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3.5-VL-4B-Instruct")
        attn_implementation, attn_source = _resolve_qwen35_attn_implementation(
            qwenvl_config.get("attn_implementation", "sdpa")
        )
        require_attn_implementation = bool(
            qwenvl_config.get("require_attn_implementation", False)
        )
        if attn_implementation == "flash_attention_2":
            if not has_flash_attn():
                if require_attn_implementation:
                    raise RuntimeError(
                        "Qwen3.5 was configured with required flash_attention_2, but flash-attn "
                        "is unavailable."
                    )
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"
        print(
            f"[Qwen3.5] attn_implementation={attn_implementation} "
            f"(source={attn_source})",
            flush=True,
        )

        model_cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
        if model_cls is None:
            raise RuntimeError(
                "RynnBrain1.1 requires Qwen3.5 support from transformers>=5.2.0; "
                f"the active version is {transformers.__version__}. Upgrade the cluster image/environment "
                "before loading this checkpoint."
            )

        use_safe_causal_conv = _env_flag(
            "STARVLA_QWEN35_DISABLE_CAUSAL_CONV1D",
            default=False,
        )
        use_safe_fla = _env_flag(
            "STARVLA_QWEN35_DISABLE_FLA",
            default=False,
        )
        modeling_module = _prepare_safe_qwen35_kernels(
            model_cls,
            safe_causal_conv=use_safe_causal_conv,
            safe_fla=use_safe_fla,
        )
        model = model_cls.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        )
        safe_causal_conv_layers = (
            _install_safe_qwen35_causal_conv(
                model,
                modeling_module=modeling_module,
            )
            if use_safe_causal_conv
            else 0
        )
        safe_fla_layers = (
            _install_safe_qwen35_fla(
                model,
                modeling_module=modeling_module,
            )
            if use_safe_fla
            else 0
        )
        self.mem_vision_encoder = apply_rynn_mem_encoder_patch(
            model,
            qwenvl_config.get("mem_vision_encoder", {}),
        )
        if self.mem_vision_encoder["enabled"]:
            print(
                "[Qwen3.5] Rynn MEM vision encoder ENABLED "
                f"(frames={self.mem_vision_encoder['num_frames']}, "
                f"stride={self.mem_vision_encoder['spacetime_layer_stride']}, "
                f"blocks={self.mem_vision_encoder['patched_indices']}, "
                "added_parameters=0)",
                flush=True,
            )
        self.causal_conv1d_backend = (
            "torch_safe" if use_safe_causal_conv else "auto"
        )
        self.fla_backend = "torch_safe" if use_safe_fla else "auto"
        if use_safe_causal_conv:
            print(
                "[Qwen3.5] causal_conv1d=torch_safe "
                f"(layers={safe_causal_conv_layers})",
                flush=True,
            )
        if use_safe_fla:
            print(
                "[Qwen3.5] gated_delta_net=torch_safe "
                f"(layers={safe_fla_layers}; FA2 unchanged)",
                flush=True,
            )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        if bool(qwenvl_config.get("enable_gradient_checkpointing", False)):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            print("[Qwen3.5] gradient_checkpointing ENABLED (use_reentrant=False)", flush=True)

        self.model = model
        self.processor = processor
        self.config = config
        self.attn_implementation = str(attn_implementation)
        self.attn_implementation_source = str(attn_source)

        if str(getattr(self.model.config, "model_type", "")) != "qwen3_5":
            raise ValueError(
                f"Qwen3.5 interface received model_type={self.model.config.model_type!r}"
            )
        # Align the shared StarVLA interface with Qwen2.5/Qwen3.
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3.5-VL Instruct format: https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            enable_thinking=bool(
                self.config.framework.qwenvl.get("enable_thinking", False)
            ),
            return_dict=True,
            return_tensors="pt",
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    logger.warning(
                        "No action token found in sequence; please check action-tokenized tokenizer in "
                        "starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md"
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct"
    qwen_vl = _QWen3_5_VL_Interface(cfg)
    pass
