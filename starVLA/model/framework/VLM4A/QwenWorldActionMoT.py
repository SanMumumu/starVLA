"""Qwen planner plus an independent dual-stream World--Action MoT.

This framework is deliberately opt-in.  It neither subclasses nor constructs
the existing staged action/world implementations.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.jointflow.dino_v3 import (
    DINOv3Backbone,
    dino_patch_grid,
    resolve_dino_spec,
)
from starVLA.model.framework.VLM4A.world_action_mot import (
    CausalDINOActionMoT,
    WorldActionMoT,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _qwen_enable_thinking(config) -> bool:
    framework = getattr(config, "framework", None) if config is not None else None
    qwenvl = framework.get("qwenvl", {}) if framework is not None else {}
    return bool(qwenvl.get("enable_thinking", False))


@dataclass
class QwenWorldActionMoTDefaultConfig:
    name: str = "QwenWorldActionMoT"
    enable_world_action_mot: bool = False
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-2B-Instruct",
            "attn_implementation": "flash_attention_2",
            "require_attn_implementation": True,
            "enable_gradient_checkpointing": True,
            "vl_hidden_dim": 2048,
        }
    )
    dino: dict = field(
        default_factory=lambda: {
            "name": "dinov3_vitb16",
            "model_size": "base",
            "hf_model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "repo_or_dir": "facebookresearch/dinov3",
            "weights": None,
            "loader": "auto",
            "image_size": [384, 320],
            "patch_size": 16,
            "embed_dim": 768,
            "stats_path": None,
            "load_live_backbone": False,
            "force_online": True,
            "dino_pool": 2,
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_dim": 14,
            "state_dim": 0,
            "action_horizon": 16,
        }
    )
    planner: dict = field(
        default_factory=lambda: {
            "num_world_queries": 16,
            "num_action_queries": 16,
            "world_placeholder_token": "<WORLD_PLAN>",
            "action_placeholder_token": "<ACTION_PLAN>",
            "text_supervision": {
                "enabled": False,
                "allow_missing_annotations": True,
                "subtask_field": "subtask_text",
                "completed_subtask_field": "completed_subtask_text",
                "prompt_template": (
                    "{instruction}\nReport the current subtask and the completed subtask."
                ),
                "response_template": (
                    "Current subtask: {subtask_text}\n"
                    "Completed subtask: {completed_subtask_text}"
                ),
                "max_new_tokens": 64,
                "do_sample": False,
            },
        }
    )
    world_action_mot: dict = field(
        default_factory=lambda: {
            "hidden_size": 1024,
            "num_layers": 12,
            "num_attention_heads": 16,
            "attention_pattern": "alternating_condition_joint",
            "action_mlp_ratio": 4.0,
            "world_mlp_ratio": 4.0,
            "dropout": 0.1,
            "time_frequency_dim": 256,
            "max_world_tokens": 512,
            "enable_gradient_checkpointing": True,
            "num_inference_timesteps": 10,
            "repeated_diffusion_steps": 1,
            "action_prediction_type": "velocity",
            "jit_t_eps": 0.05,
            "flow_time_sampling": "uniform",
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "action_loss_weight": 1.0,
            "world_loss_weight": 0.1,
            "text_loss_weight": 0.0,
        }
    )


class LearnedQueryBank(nn.Module):
    def __init__(self, count: int, hidden_size: int) -> None:
        super().__init__()
        if count <= 0:
            raise ValueError("planner query count must be positive")
        self.count = int(count)
        self.embedding = nn.Parameter(torch.empty(1, self.count, int(hidden_size)))
        nn.init.normal_(self.embedding, std=0.02)

    def forward(self, batch_size: int, *, device) -> torch.Tensor:
        return self.embedding.to(device=device).expand(int(batch_size), -1, -1)


@FRAMEWORK_REGISTRY.register("QwenWorldActionMoT")
class QwenWorldActionMoT(baseframework):
    """VLM long-range planner and short-range dual-expert physical model."""

    def __init__(self, config: Optional[dict] = None, **_kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenWorldActionMoTDefaultConfig, config)
        framework = self.config.framework
        if not bool(framework.get("enable_world_action_mot", False)):
            raise ValueError(
                "QwenWorldActionMoT requires framework.enable_world_action_mot=true"
            )
        incompatible = []
        if bool(framework.get("wam", {}).get("enabled", False)):
            incompatible.append("wam")
        if bool(framework.get("jointflow", {}).get("enabled", False)):
            incompatible.append("jointflow")
        if bool(framework.get("enable_action_world_coflow", False)):
            incompatible.append("action_world_coflow")
        if incompatible:
            raise ValueError(
                "World--Action MoT is a standalone framework; disable " + ", ".join(incompatible)
            )

        data_cfg = self.config.datasets.vla_data
        if str(data_cfg.get("dataset_py", "")) != "jointflow":
            raise ValueError("QwenWorldActionMoT currently requires dataset_py=jointflow")
        if not bool(data_cfg.get("online_dino", False)):
            raise ValueError("QwenWorldActionMoT requires online_dino=true")
        self.include_state = bool(data_cfg.get("include_state", False))
        configured_state_dim = int(framework.action_model.get("state_dim", 0) or 0)
        if self.include_state != (configured_state_dim > 0):
            raise ValueError(
                "QwenWorldActionMoT state contract requires include_state=true exactly when "
                f"action_model.state_dim > 0, got include_state={self.include_state} "
                f"and state_dim={configured_state_dim}"
            )
        if not bool(data_cfg.get("decode_future_video", True)):
            raise ValueError("QwenWorldActionMoT requires decode_future_video=true")

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        full_model = self.qwen_vl_interface.model
        text_config = getattr(full_model.config, "text_config", full_model.config)
        planner_dim = int(getattr(text_config, "hidden_size"))
        planner_cfg = framework.planner
        self.num_world_queries = int(planner_cfg.get("num_world_queries", 16))
        self.num_action_queries = int(planner_cfg.get("num_action_queries", 16))
        self.action_horizon = int(framework.action_model.get("action_horizon", 16))
        self.action_dim = int(framework.action_model.get("action_dim", 14))
        if self.num_action_queries != self.action_horizon:
            raise ValueError(
                "ACTION-PLAN queries must align one-to-one with the action chunk: "
                f"queries={self.num_action_queries}, horizon={self.action_horizon}"
            )

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        self.world_placeholder = str(
            planner_cfg.get("world_placeholder_token", "<WORLD_PLAN>")
        )
        self.action_placeholder = str(
            planner_cfg.get("action_placeholder_token", "<ACTION_PLAN>")
        )
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    self.world_placeholder,
                    self.action_placeholder,
                ]
            }
        )
        self.world_placeholder_id = int(
            tokenizer.convert_tokens_to_ids(self.world_placeholder)
        )
        self.action_placeholder_id = int(
            tokenizer.convert_tokens_to_ids(self.action_placeholder)
        )
        full_model.resize_token_embeddings(len(tokenizer))
        self.world_plan_queries = LearnedQueryBank(self.num_world_queries, planner_dim)
        self.action_plan_queries = LearnedQueryBank(self.num_action_queries, planner_dim)

        self._dino_spec = resolve_dino_spec(framework.dino)
        self.dino_dim = int(self._dino_spec["embed_dim"])
        self.dino_pool = int(framework.dino.get("dino_pool", 1))
        rows, columns = dino_patch_grid(
            self._dino_spec["image_size"], self._dino_spec["patch_size"]
        )
        if self.dino_pool <= 0 or rows % self.dino_pool or columns % self.dino_pool:
            raise ValueError(
                f"dino_pool={self.dino_pool} must divide DINO grid {(rows, columns)}"
            )
        pooled_tokens = (rows // self.dino_pool) * (columns // self.dino_pool)
        max_world_tokens = int(framework.world_action_mot.get("max_world_tokens", 512))
        if pooled_tokens > max_world_tokens:
            raise ValueError(
                f"pooled DINO tokens={pooled_tokens} exceeds max_world_tokens={max_world_tokens}"
            )
        self.register_buffer("_dino_mean", torch.zeros(self.dino_dim), persistent=False)
        self.register_buffer("_dino_std", torch.ones(self.dino_dim), persistent=False)
        self._load_dino_stats(framework.dino.get("stats_path", None))
        object.__setattr__(self, "_dino_teacher", None)
        if bool(framework.dino.get("load_live_backbone", False)):
            self._set_dino_teacher(DINOv3Backbone(**self._dino_spec))

        # Keep the familiar action_model attribute so existing optimizer and
        # deployment plumbing can discover the complete physical model.  The
        # new implementation is opt-in, preserving all existing checkpoints.
        physical_architecture = str(
            framework.world_action_mot.get("architecture", "legacy")
        ).lower()
        if physical_architecture == "causal_dino_mot":
            physical_model_cls = CausalDINOActionMoT
            configured_world_tokens = int(
                framework.world_action_mot.get("world_grid_height", 12)
            ) * int(framework.world_action_mot.get("world_grid_width", 10))
            if pooled_tokens != configured_world_tokens:
                raise ValueError(
                    "pooled online-DINO token count must equal the configured "
                    "physical world grid: "
                    f"pooled={pooled_tokens}, configured={configured_world_tokens}"
                )
        elif physical_architecture == "legacy":
            physical_model_cls = WorldActionMoT
        else:
            raise ValueError(
                "framework.world_action_mot.architecture must be "
                "'legacy' or 'causal_dino_mot', got "
                f"{physical_architecture!r}"
            )
        self.action_model = physical_model_cls(
            planner_dim=planner_dim,
            world_dim=self.dino_dim,
            action_config=framework.action_model,
            mot_config=framework.world_action_mot,
        )
        if physical_architecture == "causal_dino_mot":
            planner_dtype = next(full_model.parameters()).dtype
            self.action_model.to(dtype=planner_dtype)
        self.text_supervision = dict(planner_cfg.get("text_supervision", {}))
        self.text_loss_weight = float(
            framework.world_action_mot.get("text_loss_weight", 0.0)
        )
        self.text_planning_enabled = bool(
            self.text_supervision.get("enabled", False)
        )
        if self.text_planning_enabled != (self.text_loss_weight > 0):
            raise ValueError(
                "Unified text planning requires planner.text_supervision.enabled and "
                "world_action_mot.text_loss_weight > 0 to be enabled or disabled together. "
                "Teacher-forcing text into WORLD/ACTION queries without NTP supervision would "
                "break train/inference consistency."
            )
        annotations_enabled = bool(
            data_cfg.get("optional_text_annotations", {}).get("enabled", False)
        )
        if self.text_planning_enabled and not annotations_enabled:
            raise ValueError(
                "Unified text planning requires "
                "datasets.vla_data.optional_text_annotations.enabled=true"
            )

        # The vocabulary head is needed only by unified AR text planning.
        # Otherwise exclude it from optimizer/ZeRO state.
        if not self.text_planning_enabled:
            language_head = getattr(full_model, "lm_head", None)
            input_embeddings = full_model.get_input_embeddings()
            if language_head is not None:
                for parameter in language_head.parameters():
                    if parameter is not input_embeddings.weight:
                        parameter.requires_grad_(False)

    @property
    def device(self):
        return next(self.parameters()).device

    def _uses_action_state(self) -> bool:
        return self.include_state

    def set_dino_stats(self, stats_path: str) -> None:
        self._load_dino_stats(stats_path)

    def _qwen_backbone(self):
        full_model = self.qwen_vl_interface.model
        backbone = getattr(full_model, "model", None)
        return backbone if backbone is not None and backbone is not full_model else full_model

    @staticmethod
    def _view_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        array = np.asarray(value)
        return [array[index] for index in range(array.shape[0])] if array.ndim == 4 else [array]

    def _current_views(self, example: dict) -> list[Image.Image]:
        views = self._view_list(example.get("image_0", example.get("image")))
        if not views:
            raise KeyError("World--Action MoT requires current image_0/image")
        images = [to_pil_preserve(view) for view in views]
        target_size = self.config.datasets.vla_data.get("obs_image_size", None)
        if target_size:
            size = tuple(int(value) for value in target_size)
            images = [image if image.size == size else image.resize(size) for image in images]
        return images

    def _future_views(self, example: dict) -> list[Image.Image]:
        views = self._view_list(example.get("image_1"))
        if not views:
            raise KeyError("World--Action MoT training requires decoded future image_1")
        return [to_pil_preserve(view) for view in views]

    def _planner_user_message(self, example: dict, prompt: str) -> dict:
        content = [
            {"type": "image", "image": image}
            for image in self._current_views(example)
        ]
        content.append({"type": "text", "text": str(prompt)})
        return {"role": "user", "content": content}

    def _physical_prompt(self, example: dict) -> str:
        prompt_template = self.config.datasets.vla_data.get(
            "CoT_prompt", "{instruction}"
        )
        return str(prompt_template).replace(
            "{instruction}", str(example.get("lang", ""))
        )

    def _text_prompt(self, example: dict) -> str:
        values = {
            "instruction": str(example.get("lang", "")),
            # Targets are deliberately unavailable to the prompt.  Keeping the
            # keys permits older templates to format, but prevents label leak.
            "subtask_text": "",
            "completed_subtask_text": "",
        }
        return str(self.text_supervision["prompt_template"]).format(**values)

    @staticmethod
    def _append_query_suffix(inputs, world_token_id: int, action_token_id: int, n_world: int, n_action: int):
        input_ids = inputs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("planner requires processor input_ids [B,T]")
        if bool(
            ((input_ids == int(world_token_id)) | (input_ids == int(action_token_id))).any()
        ):
            raise ValueError("planner placeholder token leaked into the user prompt")
        batch, context_length = input_ids.shape
        world_suffix = input_ids.new_full((batch, n_world), int(world_token_id))
        action_suffix = input_ids.new_full((batch, n_action), int(action_token_id))
        inputs["input_ids"] = torch.cat([input_ids, world_suffix, action_suffix], dim=1)
        attention = inputs.get("attention_mask", torch.ones_like(input_ids))
        suffix_attention = torch.ones(
            (batch, n_world + n_action),
            device=attention.device,
            dtype=attention.dtype,
        )
        inputs["attention_mask"] = torch.cat([attention, suffix_attention], dim=1)
        inputs.pop("position_ids", None)
        inputs.pop("cache_position", None)
        world_mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
        action_mask = torch.zeros_like(world_mask)
        world_mask[:, context_length : context_length + n_world] = True
        action_mask[:, context_length + n_world :] = True
        return world_mask, action_mask

    @contextmanager
    def _replace_query_embeddings(
        self, world_mask: torch.Tensor, action_mask: torch.Tensor
    ):
        embedding = self.qwen_vl_interface.model.get_input_embeddings()
        world_queries = self.world_plan_queries(
            world_mask.shape[0], device=world_mask.device
        )
        action_queries = self.action_plan_queries(
            action_mask.shape[0], device=action_mask.device
        )
        matched = 0

        def hook(_module, _args, output):
            nonlocal matched
            if not torch.is_tensor(output) or output.ndim != 3:
                return output
            if tuple(output.shape[:2]) != tuple(world_mask.shape):
                return output
            replaced = output.clone()
            replaced[world_mask.to(output.device)] = world_queries.to(
                device=output.device, dtype=output.dtype
            ).reshape(-1, output.shape[-1])
            replaced[action_mask.to(output.device)] = action_queries.to(
                device=output.device, dtype=output.dtype
            ).reshape(-1, output.shape[-1])
            matched += 1
            return replaced

        handle = embedding.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
        if matched != 1:
            raise RuntimeError(
                f"planner query embedding replacement matched {matched} calls; expected exactly one"
            )

    def _run_planner_backbone(
        self,
        inputs,
        world_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        with self._replace_query_embeddings(world_mask, action_mask):
            device_type = self.device.type
            autocast = (
                torch.autocast(device_type, dtype=torch.bfloat16)
                if device_type == "cuda"
                else nullcontext()
            )
            with autocast:
                outputs = self._qwen_backbone()(
                    **inputs,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        return hidden

    def _extract_plans(
        self,
        hidden: torch.Tensor,
        world_mask: torch.Tensor,
        action_mask: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_size = hidden.shape[-1]
        world_plan = hidden[world_mask].reshape(
            batch_size, self.num_world_queries, hidden_size
        )
        action_plan = hidden[action_mask].reshape(
            batch_size, self.num_action_queries, hidden_size
        )
        return action_plan, world_plan

    def _planner_hidden(self, examples: List[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """No-text planner used by all existing checkpoints and YAMLs."""

        processor = self.qwen_vl_interface.processor
        messages = [
            [self._planner_user_message(example, self._physical_prompt(example))]
            for example in examples
        ]
        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            processor.tokenizer.padding_side = old_padding_side
        inputs = inputs.to(self.device)
        world_mask, action_mask = self._append_query_suffix(
            inputs,
            self.world_placeholder_id,
            self.action_placeholder_id,
            self.num_world_queries,
            self.num_action_queries,
        )
        hidden = self._run_planner_backbone(inputs, world_mask, action_mask)
        return self._extract_plans(
            hidden, world_mask, action_mask, batch_size=len(examples)
        )

    def _set_dino_teacher(self, teacher: DINOv3Backbone | None) -> None:
        if teacher is not None:
            teacher.requires_grad_(False)
            teacher.eval()
        object.__setattr__(self, "_dino_teacher", teacher)

    def _ensure_dino(self) -> DINOv3Backbone:
        teacher = getattr(self, "_dino_teacher", None)
        if teacher is None:
            teacher = DINOv3Backbone(**self._dino_spec)
        teacher = teacher.to(self.device).eval()
        self._set_dino_teacher(teacher)
        return teacher

    def _load_dino_stats(self, stats_path: str | None) -> None:
        if not stats_path:
            return
        path = Path(stats_path)
        if not path.exists():
            raise FileNotFoundError(f"Configured DINO stats do not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
        std = torch.as_tensor(stats["std"], dtype=torch.float32).clamp_min(1.0e-6)
        if mean.numel() != self.dino_dim or std.numel() != self.dino_dim:
            raise ValueError(
                f"DINO stats dim={mean.numel()} does not match encoder dim={self.dino_dim}"
            )
        self._dino_mean.copy_(mean)
        self._dino_std.copy_(std)

    def _pool_dino(self, features: torch.Tensor) -> torch.Tensor:
        if self.dino_pool == 1:
            return features
        rows, columns = dino_patch_grid(
            self._dino_spec["image_size"], self._dino_spec["patch_size"]
        )
        batch, _, dim = features.shape
        features = features.reshape(batch, rows, columns, dim)
        pool = self.dino_pool
        features = features.reshape(
            batch, rows // pool, pool, columns // pool, pool, dim
        ).mean(dim=(2, 4))
        return features.reshape(batch, -1, dim)

    @torch.no_grad()
    def _encode_dino(self, batch_views: list[list[Image.Image]]) -> torch.Tensor:
        teacher = self._ensure_dino()
        view_counts = {len(views) for views in batch_views}
        if len(view_counts) != 1:
            raise ValueError(f"all examples must have the same view count, got {view_counts}")
        views_per_example = next(iter(view_counts))
        flat = [image for views in batch_views for image in views]
        tensor = teacher.preprocess_batch(flat)
        features = teacher(tensor).float()
        features = self._pool_dino(features)
        features = features.reshape(
            len(batch_views), views_per_example * features.shape[1], features.shape[2]
        )
        return (features - self._dino_mean) / self._dino_std

    def _stack_actions(self, examples: List[dict], dtype: torch.dtype) -> torch.Tensor:
        values = np.stack([np.asarray(example["action"]) for example in examples])
        actions = torch.as_tensor(values, device=self.device, dtype=dtype)
        if actions.shape[1] < self.action_horizon or actions.shape[2] < self.action_dim:
            raise ValueError(
                f"action batch {tuple(actions.shape)} cannot satisfy "
                f"H={self.action_horizon}, D={self.action_dim}"
            )
        return actions[:, : self.action_horizon, : self.action_dim]

    def _stack_state(
        self, examples: List[dict], dtype: torch.dtype
    ) -> torch.Tensor | None:
        if not self.include_state:
            return None
        if not all("state" in example for example in examples):
            raise KeyError(
                "QwenWorldActionMoT requires normalized current state for every example"
            )
        values = np.stack([np.asarray(example["state"]) for example in examples])
        state = torch.as_tensor(values, device=self.device, dtype=dtype)
        if state.ndim == 2:
            state = state[:, None, :]
        state_dim = int(self.config.framework.action_model.state_dim)
        if state.ndim != 3 or state.shape[-1] != state_dim:
            raise ValueError(
                f"state batch must have shape [B,N,{state_dim}], got {tuple(state.shape)}"
            )
        # The joint dataset is configured at delta_indices=[0], so this is the
        # normalized current proprioception rather than a state trajectory.
        return state[:, :1]

    def _action_pad_mask(self, examples: List[dict]) -> torch.Tensor | None:
        if not all("action_is_pad" in example for example in examples):
            return None
        values = np.stack([np.asarray(example["action_is_pad"]) for example in examples])
        return torch.as_tensor(values, device=self.device, dtype=torch.bool)[
            :, : self.action_horizon
        ]

    def _text_target(self, example: dict) -> str | None:
        """Return only the supervised response; prompts never contain targets."""

        cfg = self.text_supervision
        if not self.text_planning_enabled:
            return None
        if example.get("text_annotation_available", None) is False:
            return None
        subtask = str(example.get(str(cfg.get("subtask_field", "subtask_text")), "") or "").strip()
        completed = str(
            example.get(
                str(cfg.get("completed_subtask_field", "completed_subtask_text")), ""
            )
            or ""
        ).strip()
        if not subtask and not completed:
            return None
        values = {
            "instruction": str(example.get("lang", "")),
            "subtask_text": subtask,
            "completed_subtask_text": completed,
        }
        return str(cfg["response_template"]).format(**values)

    @staticmethod
    def _assistant_labels(input_ids, attention_mask, prompt_lengths):
        labels = input_ids.clone()
        valid = attention_mask.to(dtype=torch.bool)
        labels[~valid] = -100
        full_lengths = valid.sum(dim=1)
        width = labels.shape[1]
        for row in range(labels.shape[0]):
            start = width - int(full_lengths[row])
            labels[row, : start + int(prompt_lengths[row])] = -100
        return labels

    @staticmethod
    def _validate_prompt_prefix(full, prompts) -> None:
        for row in range(full["input_ids"].shape[0]):
            full_tokens = full["input_ids"][row][
                full["attention_mask"][row].to(dtype=torch.bool)
            ]
            prompt_tokens = prompts["input_ids"][row][
                prompts["attention_mask"][row].to(dtype=torch.bool)
            ]
            if (
                prompt_tokens.numel() > full_tokens.numel()
                or not torch.equal(
                    full_tokens[: prompt_tokens.numel()], prompt_tokens
                )
            ):
                raise ValueError(
                    "assistant training prompt is not an exact causal prefix of "
                    "the text->EOS sequence"
                )

    def _truncate_assistant_at_eos(self, inputs, prompt_lengths):
        """Remove chat-template tokens after the assistant EOS boundary."""

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        full_model = getattr(self.qwen_vl_interface, "model", None)
        eos_ids = self._token_id_set(getattr(tokenizer, "eos_token_id", None))
        eos_ids.update(
            self._token_id_set(
                getattr(
                    getattr(full_model, "generation_config", None),
                    "eos_token_id",
                    None,
                )
            )
        )
        if not eos_ids:
            raise ValueError("unified text planner tokenizer has no EOS token")
        pad_id = getattr(tokenizer, "pad_token_id", None)
        pad_id = min(eos_ids) if pad_id is None else int(pad_id)

        input_ids = inputs["input_ids"]
        attention = inputs["attention_mask"].to(dtype=torch.bool)
        rows = []
        for row in range(input_ids.shape[0]):
            tokens = input_ids[row][attention[row]]
            prompt_length = int(prompt_lengths[row])
            if prompt_length > tokens.numel():
                raise ValueError(
                    f"assistant prompt length {prompt_length} exceeds sequence "
                    f"length {tokens.numel()}"
                )
            eos_position = None
            for index in range(prompt_length, tokens.numel()):
                if int(tokens[index]) in eos_ids:
                    eos_position = index
                    break
            if eos_position is None:
                raise ValueError(
                    "assistant chat template did not terminate its response with EOS"
                )
            rows.append(tokens[: eos_position + 1])

        width = max(tokens.numel() for tokens in rows)
        rebuilt_ids = input_ids.new_full((len(rows), width), pad_id)
        rebuilt_attention = inputs["attention_mask"].new_zeros((len(rows), width))
        for row, tokens in enumerate(rows):
            rebuilt_ids[row, width - tokens.numel() :] = tokens
            rebuilt_attention[row, width - tokens.numel() :] = 1
        inputs["input_ids"] = rebuilt_ids
        inputs["attention_mask"] = rebuilt_attention
        for stale_key in ("position_ids", "cache_position", "token_type_ids"):
            inputs.pop(stale_key, None)
        return inputs

    def _next_token_loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NTP only at supervised response/EOS positions.

        Selecting hidden states before the vocabulary projection avoids
        materializing [batch, full_sequence, vocab] logits.  The shift exactly
        matches Hugging Face causal-LM loss: hidden[t] predicts labels[t + 1].
        """

        shifted_labels = labels[:, 1:]
        supervised = shifted_labels.ne(-100)
        output_head = self.qwen_vl_interface.model.get_output_embeddings()
        if output_head is None:
            raise RuntimeError(
                "Unified text planning requires a causal-LM output embedding/head"
            )
        if not bool(supervised.any()):
            # Keep an untied Qwen3.5 LM head in the distributed graph even for
            # a batch containing only unannotated trajectories.
            anchor = next(output_head.parameters()).reshape(-1)[0] * 0.0
            return hidden.sum() * 0.0 + anchor
        prediction_hidden = hidden[:, :-1][supervised]
        targets = shifted_labels[supervised]
        logits = output_head(prediction_hidden)
        return F.cross_entropy(logits.float(), targets)

    def _teacher_forced_planner_hidden(
        self,
        examples: List[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """One causal training pass: context -> text -> EOS -> WORLD -> ACTION."""

        if not self.text_planning_enabled:
            action_plan, world_plan = self._planner_hidden(examples)
            return action_plan, world_plan, action_plan.new_zeros(()), 0

        responses = [self._text_target(example) for example in examples]
        if (
            not bool(self.text_supervision.get("allow_missing_annotations", True))
            and any(response is None for response in responses)
        ):
            missing = [index for index, response in enumerate(responses) if response is None]
            raise ValueError(
                "Unified text planning requires annotations for every row when "
                "allow_missing_annotations=false; missing batch rows="
                f"{missing}"
            )
        users, conversations = [], []
        for example, response in zip(examples, responses):
            user = self._planner_user_message(example, self._text_prompt(example))
            users.append([user])
            # An unannotated row retains the same physical training path.  Its
            # empty assistant turn contributes no NTP labels, while the chat
            # terminator still gives WORLD/ACTION a stable causal boundary.
            conversations.append(
                [
                    user,
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "" if response is None else response,
                            }
                        ],
                    },
                ]
            )
        processor = self.qwen_vl_interface.processor
        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            full = processor.apply_chat_template(
                conversations,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
            prompts = processor.apply_chat_template(
                users,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            processor.tokenizer.padding_side = old_padding_side

        self._validate_prompt_prefix(full, prompts)
        prompt_lengths = prompts["attention_mask"].sum(dim=1)
        full = self._truncate_assistant_at_eos(full, prompt_lengths)
        labels = self._assistant_labels(
            full["input_ids"], full["attention_mask"], prompt_lengths
        )
        for row, response in enumerate(responses):
            if response is None:
                labels[row].fill_(-100)

        full = full.to(self.device)
        labels = labels.to(self.device)
        world_mask, action_mask = self._append_query_suffix(
            full,
            self.world_placeholder_id,
            self.action_placeholder_id,
            self.num_world_queries,
            self.num_action_queries,
        )
        query_labels = labels.new_full(
            (labels.shape[0], self.num_world_queries + self.num_action_queries),
            -100,
        )
        labels = torch.cat([labels, query_labels], dim=1)
        hidden = self._run_planner_backbone(full, world_mask, action_mask)
        action_plan, world_plan = self._extract_plans(
            hidden, world_mask, action_mask, batch_size=len(examples)
        )
        return (
            action_plan,
            world_plan,
            self._next_token_loss(hidden, labels),
            sum(response is not None for response in responses),
        )

    @staticmethod
    def _token_id_set(value) -> set[int]:
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            return {int(token_id) for token_id in value}
        return {int(value)}

    def _generated_sequence_inputs(
        self,
        inputs,
        generated_ids: torch.Tensor,
    ) -> tuple[object, list[str]]:
        """Trim generation padding and left-pad complete context->text->EOS rows."""

        prompt_ids = inputs["input_ids"]
        prompt_attention = inputs["attention_mask"].to(dtype=torch.bool)
        prompt_width = prompt_ids.shape[1]
        if generated_ids.ndim != 2 or generated_ids.shape[0] != prompt_ids.shape[0]:
            raise ValueError(
                "planner generation must return [B,T] token ids, got "
                f"{tuple(generated_ids.shape)}"
            )
        if generated_ids.shape[1] < prompt_width:
            raise ValueError("decoder-only planner generation omitted the input prompt")

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        full_model = self.qwen_vl_interface.model
        eos_ids = self._token_id_set(getattr(tokenizer, "eos_token_id", None))
        eos_ids.update(
            self._token_id_set(
                getattr(getattr(full_model, "generation_config", None), "eos_token_id", None)
            )
        )
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            if not eos_ids:
                raise ValueError("text planner tokenizer defines neither pad nor EOS token")
            pad_id = min(eos_ids)
        pad_id = int(pad_id)

        sequences = []
        decoded = []
        for row in range(prompt_ids.shape[0]):
            prompt = prompt_ids[row][prompt_attention[row]]
            completion = generated_ids[row, prompt_width:]
            end = completion.numel()
            found_eos = False
            for index, token_id in enumerate(completion.tolist()):
                if int(token_id) in eos_ids:
                    end = index + 1
                    found_eos = True
                    break
            if not found_eos and pad_id not in eos_ids:
                while end > 0 and int(completion[end - 1]) == pad_id:
                    end -= 1
            completion = completion[:end]
            if not found_eos:
                if not eos_ids:
                    raise ValueError(
                        "text planner generation reached its token limit but no EOS id is defined"
                    )
                completion = torch.cat(
                    [completion, completion.new_tensor([min(eos_ids)])], dim=0
                )
            sequences.append(torch.cat([prompt, completion], dim=0))
            decoded.append(
                tokenizer.decode(completion.tolist(), skip_special_tokens=True).strip()
            )

        width = max(sequence.numel() for sequence in sequences)
        rebuilt_ids = prompt_ids.new_full((len(sequences), width), pad_id)
        rebuilt_attention = inputs["attention_mask"].new_zeros(
            (len(sequences), width)
        )
        for row, sequence in enumerate(sequences):
            rebuilt_ids[row, width - sequence.numel() :] = sequence
            rebuilt_attention[row, width - sequence.numel() :] = 1
        inputs["input_ids"] = rebuilt_ids
        inputs["attention_mask"] = rebuilt_attention
        for stale_key in ("position_ids", "cache_position", "token_type_ids"):
            inputs.pop(stale_key, None)
        return inputs, decoded

    def _generated_planner_hidden(
        self,
        examples: List[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Inference: context -> AR text through EOS -> WORLD -> ACTION."""

        processor = self.qwen_vl_interface.processor
        messages = [
            [self._planner_user_message(example, self._text_prompt(example))]
            for example in examples
        ]
        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                enable_thinking=_qwen_enable_thinking(self.config),
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            processor.tokenizer.padding_side = old_padding_side
        inputs = inputs.to(self.device)

        generation_kwargs = {
            "max_new_tokens": int(self.text_supervision.get("max_new_tokens", 64)),
            "do_sample": bool(self.text_supervision.get("do_sample", False)),
            "use_cache": True,
        }
        if generation_kwargs["max_new_tokens"] <= 0:
            raise ValueError("text_supervision.max_new_tokens must be positive")
        if generation_kwargs["do_sample"]:
            generation_kwargs["temperature"] = float(
                self.text_supervision.get("temperature", 1.0)
            )
        device_type = self.device.type
        autocast = (
            torch.autocast(device_type, dtype=torch.bfloat16)
            if device_type == "cuda"
            else nullcontext()
        )
        with autocast:
            generated = self.qwen_vl_interface.model.generate(
                **inputs,
                **generation_kwargs,
            )
        generated_ids = getattr(generated, "sequences", generated)
        inputs, planner_text = self._generated_sequence_inputs(inputs, generated_ids)
        world_mask, action_mask = self._append_query_suffix(
            inputs,
            self.world_placeholder_id,
            self.action_placeholder_id,
            self.num_world_queries,
            self.num_action_queries,
        )
        hidden = self._run_planner_backbone(inputs, world_mask, action_mask)
        action_plan, world_plan = self._extract_plans(
            hidden, world_mask, action_mask, batch_size=len(examples)
        )
        return action_plan, world_plan, planner_text

    def forward(self, examples: List[dict] = None, **_kwargs) -> dict:
        if not isinstance(examples, list) or not examples:
            raise ValueError("QwenWorldActionMoT.forward expects a non-empty example list")
        (
            action_plan,
            world_plan,
            text_loss,
            annotated,
        ) = self._teacher_forced_planner_hidden(examples)
        current_world = self._encode_dino([self._current_views(example) for example in examples])
        target_world = self._encode_dino([self._future_views(example) for example in examples])
        model_dtype = next(self.action_model.parameters()).dtype
        action_plan = action_plan.to(dtype=model_dtype)
        world_plan = world_plan.to(dtype=model_dtype)
        current_world = current_world.to(device=self.device, dtype=model_dtype)
        target_world = target_world.to(device=self.device, dtype=model_dtype)
        target_action = self._stack_actions(examples, model_dtype)
        current_state = self._stack_state(examples, model_dtype)
        future_valid = torch.as_tensor(
            [example.get("future_valid", True) for example in examples],
            device=self.device,
            dtype=torch.float32,
        )
        physical = self.action_model.forward_train(
            action_plan=action_plan,
            world_plan=world_plan,
            current_world=current_world,
            target_action=target_action,
            target_world=target_world,
            action_is_pad=self._action_pad_mask(examples),
            future_valid=future_valid,
            current_state=current_state,
        )
        total = physical["loss"] + self.text_loss_weight * text_loss
        return {
            "action_loss": total,
            "mot_action_loss_raw": physical["action_loss_raw"],
            "mot_world_loss_raw": physical["world_loss_raw"],
            "mot_text_loss_raw": text_loss.detach(),
            "mot_text_annotated_count": total.new_tensor(float(annotated)),
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("QwenWorldActionMoT.predict_action expects at least one example")
        planner_text = None
        if self.text_planning_enabled:
            action_plan, world_plan, planner_text = self._generated_planner_hidden(
                examples
            )
        else:
            action_plan, world_plan = self._planner_hidden(examples)
        current_world = self._encode_dino([self._current_views(example) for example in examples])
        model_dtype = next(self.action_model.parameters()).dtype
        current_state = self._stack_state(examples, model_dtype)
        action, _future_world = self.action_model.sample(
            action_plan=action_plan.to(dtype=model_dtype),
            world_plan=world_plan.to(dtype=model_dtype),
            current_world=current_world.to(device=self.device, dtype=model_dtype),
            current_state=current_state,
            seed=kwargs.get("mot_inference_seed", None),
        )
        output = {"normalized_actions": action.float().cpu().numpy()}
        if planner_text is not None:
            output["planner_text"] = planner_text
        return output
