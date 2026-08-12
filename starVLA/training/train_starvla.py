# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
#######
import random
import re
#######
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    setup_optimizer_and_scheduler,
    normalize_dotlist_args,
    build_task_grad_groups,
    group_grad_sqnorm,
    group_grad_local_sqnorm,
)

# Constructed only after the experiment YAML is loaded.  DeepSpeed's AIDI
# config leaves gradient_accumulation_steps="auto"; initializing here would
# resolve it to Accelerator's default (1) before the YAML value is known.
accelerator: Accelerator | None = None

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def configure_torch_runtime() -> None:
    """Configure optional CUDA speed knobs controlled by launch env vars."""
    allow_tf32 = os.getenv("STARVLA_ALLOW_TF32", "0").strip().lower() in {"1", "true", "yes", "on"}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    if allow_tf32:
        precision = os.getenv("STARVLA_FLOAT32_MATMUL_PRECISION", "high")
        torch.set_float32_matmul_precision(precision)
        logger.info("STARVLA torch runtime: TF32 enabled, float32_matmul_precision=%s", precision)


def build_training_accelerator(cfg) -> Accelerator:
    """Build Accelerate/DeepSpeed with the accumulation value from this run."""

    accumulation_steps = int(cfg.trainer.get("gradient_accumulation_steps", 1))
    if accumulation_steps <= 0:
        raise ValueError(
            "trainer.gradient_accumulation_steps must be positive, "
            f"got {accumulation_steps}."
        )
    independent_clipping = cfg.trainer.get("independent_gradient_clipping", {})
    independent_clipping_enabled = bool(
        hasattr(independent_clipping, "get")
        and independent_clipping.get("enabled", False)
    )
    # Accelerate otherwise resolves DeepSpeed's ``gradient_clipping=auto`` to
    # 1.0 inside _prepare_deepspeed.  The strict branch clipper operates on
    # ZeRO-2's reduced partitions immediately before engine.step(), so global
    # clipping must be explicitly disabled rather than left as ``None``.
    clipping = (
        0.0
        if independent_clipping_enabled
        else cfg.trainer.get("gradient_clipping", None)
    )
    plugin = DeepSpeedPlugin(
        gradient_accumulation_steps=accumulation_steps,
        gradient_clipping=float(clipping) if clipping is not None else None,
    )
    # DeepSpeed ZeRO-2 partitions gradients and explicitly rejects
    # ``engine.no_sync()``. Accelerate normally enters no_sync on non-boundary
    # micro-batches, which crashes before the first E2E forward when
    # gradient_accumulation_steps > 1. Keep the same numerical accumulation,
    # but synchronize each micro-batch so DeepSpeed can own accumulation and
    # optimizer-boundary handling without the unsupported context.
    zero_no_sync_guard = accumulation_steps > 1
    accumulation_plugin = GradientAccumulationPlugin(
        num_steps=accumulation_steps,
        sync_each_batch=zero_no_sync_guard,
    )
    result = Accelerator(
        deepspeed_plugin=plugin,
        gradient_accumulation_plugin=accumulation_plugin,
    )
    if int(result.gradient_accumulation_steps) != accumulation_steps:
        raise RuntimeError(
            "Accelerate ignored trainer.gradient_accumulation_steps: "
            f"requested {accumulation_steps}, active {result.gradient_accumulation_steps}."
        )
    deepspeed_accumulation = plugin.get_value("gradient_accumulation_steps")
    if deepspeed_accumulation != "auto" and int(deepspeed_accumulation) != accumulation_steps:
        raise RuntimeError(
            "DeepSpeed accumulation disagrees with the training YAML: "
            f"requested {accumulation_steps}, active {deepspeed_accumulation}."
        )
    accumulation_kwargs = result.gradient_state.plugin_kwargs
    if zero_no_sync_guard and not bool(accumulation_kwargs.get("sync_each_batch", False)):
        raise RuntimeError(
            "DeepSpeed ZeRO accumulation requires GradientAccumulationPlugin(sync_each_batch=True) "
            "to avoid the unsupported engine.no_sync() path."
        )
    result.print(result.state)
    result.print(
        f"[train] gradient_accumulation_steps={result.gradient_accumulation_steps} "
        f"deepspeed={deepspeed_accumulation} "
        f"zero_no_sync_guard={str(zero_no_sync_guard).lower()} "
        f"independent_gradient_clipping={str(independent_clipping_enabled).lower()}"
    )
    return result


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(
    cfg, accelerator, output_dir
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """Prepare physical, optional semantic-NTP, and optional validation data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    event_cfg = cfg.datasets.vla_data.get("text_annotations", {}).get(
        "event_memory", {}
    )
    semantic_train_dataloader = None
    if bool(event_cfg.get("enabled", False)):
        from starVLA.dataloader.jointflow.joint_dataset import (
            build_event_memory_dataloader,
        )

        logger.info(
            "Creating event-memory semantic NTP stream (phase-aligned 2/2/2 sampler)"
        )
        semantic_train_dataloader = build_event_memory_dataloader(
            cfg, output_dir=output_dir
        )

    world_validation = cfg.trainer.get("world_validation", {})
    vla_val_dataloader = None
    if hasattr(world_validation, "get") and bool(world_validation.get("enabled", False)):
        val_fraction = float(cfg.datasets.vla_data.get("fastwam_val_fraction", 0.0))
        if val_fraction <= 0.0:
            raise ValueError(
                "trainer.world_validation.enabled=true requires "
                "datasets.vla_data.fastwam_val_fraction > 0"
            )
        raw_cfg = cfg.to_dict(resolve=True) if isinstance(cfg, AccessTrackedConfig) else OmegaConf.to_container(
            cfg, resolve=True
        )
        val_cfg = OmegaConf.create(raw_cfg)
        val_cfg.datasets.vla_data.fastwam_split = "val"
        val_cfg.datasets.vla_data.per_device_batch_size = int(
            world_validation.get("per_device_batch_size", 2)
        )
        if val_cfg.datasets.vla_data.per_device_batch_size <= 0:
            raise ValueError("world_validation.per_device_batch_size must be positive")
        if world_validation.get("num_workers", None) is not None:
            val_cfg.datasets.vla_data.num_workers = int(world_validation.get("num_workers"))
        logger.info(
            "Creating fixed FastWAM world-validation split: fraction=%s seed=%s batch/device=%s",
            val_fraction,
            val_cfg.datasets.vla_data.fastwam_split_seed,
            val_cfg.datasets.vla_data.per_device_batch_size,
        )
        vla_val_dataloader = build_dataloader(
            cfg=val_cfg,
            dataset_py=val_cfg.datasets.vla_data.dataset_py,
            save_statistics=False,
        )

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader, semantic_train_dataloader, vla_val_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    _COFLOW_DASHBOARD_KEYS = frozenset(
        {
            "train/task",
            "train/action_loss_raw",
            "train/z16_loss_raw",
            "train/world_loss_raw",
            "train/action_loss_weighted",
            "train/world_loss_weighted",
            "train/world_loss_weight",
            "train/loss_total",
            "train/learning_rate",
            "train/grad_norm",
        }
    )

    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        semantic_train_dataloader=None,
        vla_val_dataloader=None,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.semantic_train_dataloader = semantic_train_dataloader
        self.semantic_batch_sampler = (
            semantic_train_dataloader.batch_sampler
            if semantic_train_dataloader is not None
            else None
        )
        self.vla_val_dataloader = vla_val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        independent_clip_cfg = cfg.trainer.get("independent_gradient_clipping", {})
        self._independent_gradient_clipping = bool(
            hasattr(independent_clip_cfg, "get")
            and independent_clip_cfg.get("enabled", False)
        )
        self._independent_clip_limits = {
            branch: float(independent_clip_cfg.get(branch, 0.0))
            for branch in ("action", "world")
        } if self._independent_gradient_clipping else {}
        self._last_independent_clip_metrics: dict[str, float] = {}

        self.completed_steps = 0
        self._pending_full_state_path: str | None = None
        self.total_batch_size = self._calculate_total_batch_size()
        #######
        self._jointflow_rng = random.Random(int(getattr(cfg, "seed", 42)))
        self._jointflow_tasks = None
        self._jointflow_weights = None
        self._jointflow_cur_task = None
        self._jointflow_resample = True
        framework_cfg = getattr(cfg, "framework", None)
        jointflow_cfg = getattr(framework_cfg, "jointflow", None) if framework_cfg is not None else None
        tasks_cfg = getattr(framework_cfg, "tasks", None) if framework_cfg is not None else None
        #######
        wam_cfg = getattr(framework_cfg, "wam", None) if framework_cfg is not None else None
        _multitask_on = (jointflow_cfg is not None and bool(jointflow_cfg.get("enabled", False))) or (
            wam_cfg is not None and bool(wam_cfg.get("enabled", False))
        )
        if (
            _multitask_on
            and tasks_cfg is not None
            and hasattr(tasks_cfg, "get")
        ):
        #######
            weights = tasks_cfg.get("weights", None)
            if weights:
                self._jointflow_tasks = [str(k) for k, v in weights.items() if float(v) > 0]
                self._jointflow_weights = [float(weights[k]) for k in self._jointflow_tasks]
                if not self._jointflow_tasks:
                    raise ValueError("framework.tasks.weights must contain at least one positive task weight.")
                # Joint tasks already compute action + world losses in the
                # same forward pass.  Sampling it together with policy/passive
                # would silently reintroduce alternating single-objective
                # updates and bypass the configured action->world ramp on the
                # policy-only steps.  Treat this as a startup-time invariant,
                # so a bad merge/config can never run for hours unnoticed.
                joint_tasks = {
                    task for task in self._jointflow_tasks if task in {"joint_e2e", "joint_detached"}
                }
                if joint_tasks and len(self._jointflow_tasks) != 1:
                    raise ValueError(
                        "framework.tasks.weights: joint_e2e/joint_detached is exclusive because it optimizes "
                        "action_loss + world_loss in every batch; positive companion tasks are not allowed. "
                        f"Active tasks: {self._jointflow_tasks}."
                    )
                if self.accelerator.is_main_process:
                    logger.info(
                        "Active QwenGR00T task sampler: tasks=%s weights=%s",
                        self._jointflow_tasks,
                        self._jointflow_weights,
                    )
                    if self._jointflow_tasks in (["joint_e2e"], ["joint_detached"]):
                        logger.info(
                            "Verified %s objective: every optimizer step uses one batch for "
                            "action_loss + weighted world_loss; no policy/passive alternation.",
                            self._jointflow_tasks[0],
                        )
        #######

        #######
        _gflag = getattr(cfg.trainer, "log_grad_norms", None)
        if _gflag is None:
            _gflag = getattr(cfg.trainer, "log_grad_norm_per_head", False)
        self._log_grad_norms = bool(_gflag)
        # Cheap scalar norm for the main experiment panel.  Under DeepSpeed
        # this reads the engine's already-reduced norm and adds no collective;
        # per-head decomposition remains controlled separately above.
        self._log_global_grad_norm = bool(
            getattr(cfg.trainer, "log_global_grad_norm", False)
        )
        self._grad_groups = None
        self._task_head = {}
        self._task_logname = {}
        self._using_deepspeed = "DEEPSPEED" in str(getattr(accelerator, "distributed_type", "")).upper()
        self._last_clip_norm = None
        self._grad_warned = False
        shared_vlm_diag_cfg = cfg.trainer.get(
            "shared_vlm_gradient_diagnostics", {}
        )
        self._shared_vlm_gradient_diagnostics_enabled = bool(
            hasattr(shared_vlm_diag_cfg, "get")
            and shared_vlm_diag_cfg.get("enabled", False)
        )
        self._shared_vlm_gradient_diagnostics_interval = int(
            shared_vlm_diag_cfg.get(
                "interval",
                cfg.trainer.logging_frequency,
            )
            if hasattr(shared_vlm_diag_cfg, "get")
            else cfg.trainer.logging_frequency
        )
        if self._shared_vlm_gradient_diagnostics_enabled:
            logging_frequency = int(cfg.trainer.logging_frequency)
            if self._shared_vlm_gradient_diagnostics_interval <= 0:
                raise ValueError(
                    "trainer.shared_vlm_gradient_diagnostics.interval must be positive"
                )
            if (
                self._shared_vlm_gradient_diagnostics_interval
                % logging_frequency
                != 0
            ):
                raise ValueError(
                    "trainer.shared_vlm_gradient_diagnostics.interval must be "
                    "a multiple of trainer.logging_frequency so every diagnostic "
                    "is emitted to W&B"
                )
            if int(cfg.trainer.get("gradient_accumulation_steps", 1)) != 1:
                raise ValueError(
                    "shared-VLM interface gradient diagnostics currently require "
                    "trainer.gradient_accumulation_steps=1 so the measured query "
                    "gradient is exactly the optimizer step's full local batch"
                )
        #######

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)
        self._validate_joint_world_optimizer_contract()
        self._validate_wam_two_stage_runtime_contract()
        self._validate_independent_gradient_clipping_contract(distributed=False)

        self._prepare_distributed_components()
        self._validate_independent_gradient_clipping_contract(distributed=True)

        # The historical loop intentionally steps the raw Transformers
        # scheduler itself instead of wrapping it with Accelerator.  Register
        # that scheduler as a checkpointable object so save_state/load_state
        # also preserves its exact step and per-group LR without changing the
        # established stepping semantics.
        self.accelerator.register_for_checkpointing(self.lr_scheduler)
        if self.semantic_batch_sampler is not None:
            self.accelerator.register_for_checkpointing(
                self.semantic_batch_sampler
            )

        # Full optimizer/scheduler state can only be restored after Accelerate
        # has registered and wrapped every training object.
        self._restore_pending_full_training_state()

        self._validate_runtime_batch_contract()
        self._configure_unused_param_anchors()

        #######
        if self._log_grad_norms:
            try:
                base_model = self.accelerator.unwrap_model(self.model)
                self._grad_groups, self._task_head, self._task_logname = build_task_grad_groups(base_model)
                if self.accelerator.is_main_process:
                    logger.info(
                        "Grad-norm groups: %s (deepspeed=%s)",
                        {k: len(v) for k, v in self._grad_groups.items()},
                        self._using_deepspeed,
                    )
            except Exception as e:
                logger.warning("build_task_grad_groups failed (%s); disabling per-head grad logging.", e)
                self._log_grad_norms = False
                self._grad_groups = None
        #######
        if (
            self._shared_vlm_gradient_diagnostics_enabled
            and self.accelerator.is_main_process
        ):
            logger.info(
                "Shared-VLM interface action/world gradient norms: interval=%d "
                "(weighted physical objectives; text loss excluded)",
                self._shared_vlm_gradient_diagnostics_interval,
            )

        self._init_wandb()

    def _prepare_distributed_components(self) -> None:
        """Initialize DeepSpeed from the train loader, then shard validation.

        Accelerate resolves DeepSpeed's ``train_micro_batch_size_per_gpu=auto``
        from every dataloader passed to one ``prepare`` call.  Its default
        ``is_train_batch_min=True`` therefore chose the 2-sample world-val
        loader instead of the 12-sample train loader.  Prepare the engine with
        the train loader alone; validation still needs distributed sharding,
        but must not participate in the engine's training-batch ABI.
        """

        prepared = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        self.model, self.optimizer, self.vla_train_dataloader = prepared
        if getattr(self, "semantic_train_dataloader", None) is not None:
            self.semantic_train_dataloader = self.accelerator.prepare_data_loader(
                self.semantic_train_dataloader
            )
        if self.vla_val_dataloader is not None:
            self.vla_val_dataloader = self.accelerator.prepare_data_loader(
                self.vla_val_dataloader
            )

    def _validate_joint_world_optimizer_contract(self) -> None:
        """Fail before distributed launch if the joint world head cannot update."""

        if self._jointflow_tasks not in (["joint_e2e"], ["joint_detached"]):
            return
        visual_head = getattr(self.model, "wam_visual_head", None)
        if not isinstance(visual_head, torch.nn.Module):
            raise RuntimeError(
                f"{self._jointflow_tasks[0]} requires a trainable wam_visual_head"
            )
        visual_params = [parameter for parameter in visual_head.parameters() if parameter.requires_grad]
        if not visual_params:
            raise RuntimeError(
                f"{self._jointflow_tasks[0]} has no trainable wam_visual_head parameters; "
                "check trainer.freeze_modules"
            )
        world_params = list(visual_params)
        state_context = getattr(self.model, "wam_state_ctx", None)
        if isinstance(state_context, torch.nn.Module):
            state_params = [parameter for parameter in state_context.parameters() if parameter.requires_grad]
            if not state_params:
                raise RuntimeError(
                    "world_condition_on_state is enabled but wam_state_ctx has no trainable parameters"
                )
            world_params.extend(state_params)
        current_dino_projector = getattr(self.model, "wam_current_dino_proj", None)
        if isinstance(current_dino_projector, torch.nn.Module):
            projector_params = [
                parameter
                for parameter in current_dino_projector.parameters()
                if parameter.requires_grad
            ]
            if not projector_params:
                raise RuntimeError(
                    "concat_current_dino is enabled but wam_current_dino_proj has no trainable parameters"
                )
            world_params.extend(projector_params)
        config = getattr(self, "config", None)
        trainer_cfg = getattr(config, "trainer", None) if config is not None else None
        recipe = str(
            trainer_cfg.get("wam_two_stage_recipe", "legacy_v1")
            if trainer_cfg is not None
            else "legacy_v1"
        ).lower()
        if recipe in {"isolated_queries_v4", "causal_action_world_queries_v1"}:
            world_queries = getattr(self.model, "future_dino_queries", None)
            if not isinstance(world_queries, torch.nn.Module):
                raise RuntimeError(
                    f"{recipe} requires an explicit future_dino_queries module"
                )
            query_params = [
                parameter
                for parameter in world_queries.parameters()
                if parameter.requires_grad
            ]
            if not query_params:
                raise RuntimeError(
                    f"{recipe} warmup requires trainable world-query parameters"
                )
            world_params.extend(query_params)

        # Transformers schedulers set the *current* optimizer LR to zero at
        # scheduler construction when linear/cosine warmup starts at step 0.
        # The configured, update-capable LR is retained in ``initial_lr``.
        # Validate that base LR here; checking only ``lr`` rejects every valid
        # warmup run before its first optimizer step.
        optimizer_lrs: dict[int, tuple[str, float, float]] = {}
        for index, group in enumerate(self.optimizer.param_groups):
            name = str(group.get("name", index))
            current_lr = float(group.get("lr", 0.0))
            initial_lr = float(group.get("initial_lr", current_lr))
            for parameter in group["params"]:
                optimizer_lrs[id(parameter)] = (name, initial_lr, current_lr)
        missing = [parameter for parameter in world_params if id(parameter) not in optimizer_lrs]
        if missing:
            raise RuntimeError(
                f"{len(missing)} trainable world-predictor parameters are absent from the optimizer"
            )
        groups = {optimizer_lrs[id(parameter)] for parameter in world_params}
        bad_groups = sorted(
            (name, initial_lr)
            for name, initial_lr, _current_lr in groups
            if initial_lr <= 0.0
        )
        if bad_groups:
            raise RuntimeError(
                "World-predictor optimizer LR must be positive "
                f"(scheduler initial/base LR), got {bad_groups}"
            )
        if recipe == "causal_action_world_queries_v1":
            action_modules = {
                "qwen_vl_interface": getattr(self.model, "qwen_vl_interface", None),
                "action_queries": getattr(self.model, "action_queries", None),
                "action_model": getattr(self.model, "action_model", None),
            }
            action_params: list[torch.nn.Parameter] = []
            seen: set[int] = set()
            empty_modules: list[str] = []
            for module_name, module in action_modules.items():
                params = (
                    [parameter for parameter in module.parameters() if parameter.requires_grad]
                    if isinstance(module, torch.nn.Module)
                    else []
                )
                if not params:
                    empty_modules.append(module_name)
                for parameter in params:
                    if id(parameter) not in seen:
                        seen.add(id(parameter))
                        action_params.append(parameter)
            if empty_modules:
                raise RuntimeError(
                    "causal_action_world_queries_v1 requires trainable action-path modules: "
                    + ",".join(empty_modules)
                )
            missing_action = [
                parameter for parameter in action_params if id(parameter) not in optimizer_lrs
            ]
            if missing_action:
                raise RuntimeError(
                    f"{len(missing_action)} trainable causal action-path parameters are absent "
                    "from the optimizer"
                )
            action_groups = {optimizer_lrs[id(parameter)] for parameter in action_params}
            bad_action_groups = sorted(
                (name, initial_lr)
                for name, initial_lr, _current_lr in action_groups
                if initial_lr <= 0.0
            )
            if bad_action_groups:
                raise RuntimeError(
                    "Causal shared-query action-path optimizer LR must be positive, got "
                    f"{bad_action_groups}"
                )
            if self.accelerator.is_main_process:
                logger.info(
                    "Verified causal action optimizer: params=%d "
                    "groups=(name, initial_lr, current_lr)=%s",
                    sum(parameter.numel() for parameter in action_params),
                    sorted(action_groups),
                )
        if self.accelerator.is_main_process:
            logger.info(
                "Verified joint world optimizer: task=%s params=%d "
                "groups=(name, initial_lr, current_lr)=%s",
                self._jointflow_tasks[0],
                sum(parameter.numel() for parameter in world_params),
                sorted(groups),
            )

    def _validate_wam_two_stage_runtime_contract(self) -> None:
        """Verify actual requires-grad state after checkpoint load and freezing."""

        phase = str(self.config.trainer.get("wam_two_stage_phase", "")).lower()
        if not phase:
            return
        recipe = str(
            self.config.trainer.get("wam_two_stage_recipe", "legacy_v1")
            or "legacy_v1"
        ).lower()
        guidance = self.config.framework.get("wam", {}).get("guidance", {})
        world_to_action_enabled = bool(guidance.get("world_to_action_enabled", True))

        def trainable(module_name: str) -> list[torch.nn.Parameter]:
            module = getattr(self.model, module_name, None)
            if not isinstance(module, torch.nn.Module):
                return []
            return [parameter for parameter in module.parameters() if parameter.requires_grad]

        def validate_optimizer_coverage(
            module_names: tuple[str, ...],
            *,
            label: str,
        ) -> None:
            """Prove every live parameter is assigned a positive base LR.

            A scheduler with warmup legitimately exposes ``lr=0`` before its
            first step, so the update-capable contract is ``initial_lr > 0``.
            This catches both a stale freeze selector and a YAML LR group that
            silently omitted Action DiT/query parameters.
            """

            optimizer_groups: dict[int, tuple[str, float, float]] = {}
            for index, group in enumerate(self.optimizer.param_groups):
                name = str(group.get("name", index))
                current_lr = float(group.get("lr", 0.0))
                initial_lr = float(group.get("initial_lr", current_lr))
                for parameter in group["params"]:
                    optimizer_groups[id(parameter)] = (name, initial_lr, current_lr)

            parameters: list[torch.nn.Parameter] = []
            seen: set[int] = set()
            empty: list[str] = []
            for module_name in module_names:
                selected = trainable(module_name)
                if not selected:
                    empty.append(module_name)
                for parameter in selected:
                    if id(parameter) not in seen:
                        seen.add(id(parameter))
                        parameters.append(parameter)
            if empty:
                raise RuntimeError(
                    f"{label} requires trainable modules missing at runtime: {empty}"
                )
            missing = [parameter for parameter in parameters if id(parameter) not in optimizer_groups]
            if missing:
                raise RuntimeError(
                    f"{label} has {len(missing)} trainable parameters absent from the optimizer"
                )
            groups = {optimizer_groups[id(parameter)] for parameter in parameters}
            bad = sorted(
                (name, initial_lr)
                for name, initial_lr, _current_lr in groups
                if initial_lr <= 0.0
            )
            if bad:
                raise RuntimeError(
                    f"{label} optimizer groups must have positive initial/base LR, got {bad}"
                )
            if self.accelerator.is_main_process:
                logger.info(
                    "Verified %s optimizer coverage: params=%d groups=%s",
                    label,
                    sum(parameter.numel() for parameter in parameters),
                    sorted(groups),
                )

        gates = [
            parameter
            for name, parameter in self.model.named_parameters()
            if name.rsplit(".", 1)[-1] == "world_gate"
        ]
        action_dit = getattr(getattr(self.model, "action_model", None), "model", None)
        blocks = getattr(action_dit, "transformer_blocks", ())
        world_action_modules = [
            module
            for block in blocks
            for module in (
                getattr(block, "world_attn", None),
                getattr(block, "world_norm", None),
                getattr(block, "world_to_temb", None),
            )
            if isinstance(module, torch.nn.Module)
        ]
        if isinstance(getattr(action_dit, "world_to_temb", None), torch.nn.Module):
            world_action_modules.append(action_dit.world_to_temb)
        world_action_modules.extend(
            module
            for module_name in (
                "world_adapter",
                "world_fusion",
                "world_qformer",
                "world_pooler",
            )
            if isinstance((module := getattr(self.model, module_name, None)), torch.nn.Module)
        )
        if world_to_action_enabled:
            if not gates:
                raise RuntimeError(f"WAM two-stage {phase} exposes no world_gate parameters")
            max_gate = max(float(torch.tanh(gate.detach().float()).abs().max()) for gate in gates)
            if max_gate > 1.0e-7:
                raise RuntimeError(
                    f"WAM two-stage {phase} must start with closed world gates, max openness={max_gate:.6g}"
                )
        else:
            if phase != "predictor_warmup":
                raise RuntimeError(
                    "world_to_action_enabled=false is only valid for predictor_warmup"
                )
            if gates or world_action_modules:
                raise RuntimeError(
                    "No-world-to-action baseline instantiated forbidden gate/adapter/attention modules: "
                    f"gates={len(gates)} modules={len(world_action_modules)}"
                )
            max_gate = 0.0

        if phase == "predictor_warmup":
            if not trainable("wam_visual_head"):
                raise RuntimeError("predictor_warmup requires a trainable visual head")
            if bool(guidance.get("world_condition_on_state", False)) and not trainable(
                "wam_state_ctx"
            ):
                raise RuntimeError(
                    "state-conditioned predictor_warmup requires a trainable state conditioner"
                )
            if recipe in {
                "policy_first_v2",
                "baseline_preserving_v3",
                "isolated_queries_v4",
                "causal_action_world_queries_v1",
            }:
                if not trainable("qwen_vl_interface"):
                    raise RuntimeError(
                        f"{recipe} predictor_warmup requires trainable Qwen so action loss can update it"
                    )
                if not trainable("action_model"):
                    raise RuntimeError(
                        f"{recipe} predictor_warmup requires a trainable action model"
                    )
            if recipe in {"isolated_queries_v4", "causal_action_world_queries_v1"}:
                if not trainable("action_queries"):
                    raise RuntimeError(
                        f"{recipe} warmup requires a trainable action query bank"
                    )
                if not trainable("future_dino_queries"):
                    raise RuntimeError(
                        f"{recipe} warmup requires a trainable world query bank"
                    )
                leaking_world_to_action = sum(
                    parameter.numel()
                    for module in world_action_modules
                    for parameter in module.parameters()
                    if parameter.requires_grad
                ) + sum(parameter.numel() for parameter in gates if parameter.requires_grad)
                if leaking_world_to_action:
                    raise RuntimeError(
                        f"{recipe} warmup must physically freeze adapter/gate/"
                        f"world-attention parameters; trainable={leaking_world_to_action}"
                    )
        elif phase == "gate_ft":
            unexpectedly_trainable = {
                name: len(trainable(name))
                for name in (
                    "qwen_vl_interface",
                    "wam_visual_head",
                    "wam_state_ctx",
                    "wam_act_ctx",
                    (
                        "future_dino_queries"
                        if recipe in {"isolated_queries_v4", "causal_action_world_queries_v1"}
                        else ""
                    ),
                )
                if name
                if trainable(name)
            }
            if unexpectedly_trainable:
                raise RuntimeError(
                    "gate_ft world/Qwen modules were not frozen: " + str(unexpectedly_trainable)
                )
            if not trainable("action_model") or not trainable("world_adapter"):
                raise RuntimeError("gate_ft requires trainable action_model and world_adapter")
            if recipe in {"isolated_queries_v4", "causal_action_world_queries_v1"} and not trainable(
                "action_queries"
            ):
                raise RuntimeError(
                    f"{recipe} gate_ft requires a trainable action query bank"
                )
        else:
            raise RuntimeError(f"Unsupported wam_two_stage_phase={phase!r}")

        if recipe == "causal_action_world_queries_v1":
            if phase == "predictor_warmup":
                warmup_modules = [
                    "qwen_vl_interface",
                    "action_queries",
                    "action_model",
                    "future_dino_queries",
                    "wam_visual_head",
                ]
                if bool(guidance.get("world_condition_on_state", False)):
                    warmup_modules.append("wam_state_ctx")
                validate_optimizer_coverage(
                    tuple(warmup_modules),
                    label="causal action/world query warmup",
                )
            else:
                validate_optimizer_coverage(
                    ("action_queries", "action_model", "world_adapter"),
                    label="causal action/world query gate FT",
                )

        if self.accelerator.is_main_process:
            logger.info(
                "Verified WAM two-stage runtime: phase=%s recipe=%s "
                "world_to_action=%s gates=%d max_initial_openness=%.3g",
                phase,
                recipe,
                world_to_action_enabled,
                len(gates),
                max_gate,
            )

    def _validate_independent_gradient_clipping_contract(
        self,
        *,
        distributed: bool,
    ) -> None:
        """Validate strict action/world ownership and the ZeRO-2 clip ABI."""

        if not self._independent_gradient_clipping:
            return
        bad_limits = {
            branch: limit
            for branch, limit in self._independent_clip_limits.items()
            if limit <= 0.0
        }
        if bad_limits:
            raise RuntimeError(
                "independent_gradient_clipping limits must be positive: "
                f"{bad_limits}"
            )
        global_clip = self.config.trainer.get("gradient_clipping", None)
        if global_clip is not None and float(global_clip) != 0.0:
            raise RuntimeError(
                "independent_gradient_clipping requires trainer.gradient_clipping "
                "to be null/0 so DeepSpeed cannot apply a second global clip"
            )

        groups = list(self.optimizer.param_groups)
        partitions = {str(group.get("gradient_partition", "")) for group in groups}
        invalid = sorted(partitions - {"action", "world"})
        if invalid:
            raise RuntimeError(
                "Every strict optimizer group must belong exclusively to action or world; "
                f"invalid gradient_partition values={invalid}"
            )
        missing = {"action", "world"} - partitions
        if missing:
            raise RuntimeError(
                "Strict warmup optimizer is missing gradient partitions: "
                f"{sorted(missing)}"
            )
        if not distributed:
            return

        if not self._using_deepspeed:
            if self.accelerator.is_main_process:
                logger.info(
                    "Verified independent clipping without DeepSpeed: groups=%s limits=%s",
                    {branch: sum(g.get("gradient_partition") == branch for g in groups) for branch in partitions},
                    self._independent_clip_limits,
                )
            return

        stage_fn = getattr(self.model, "zero_optimization_stage", None)
        zero_stage = int(stage_fn()) if callable(stage_fn) else -1
        if zero_stage != 2:
            raise RuntimeError(
                "independent_gradient_clipping currently requires DeepSpeed ZeRO-2, "
                f"got stage={zero_stage}"
            )
        zero_optimizer = getattr(self.model, "optimizer", None)
        if zero_optimizer is None or not hasattr(zero_optimizer, "averaged_gradients"):
            raise RuntimeError(
                "DeepSpeed optimizer does not expose ZeRO-2 averaged_gradients; "
                "cannot clip action/world partitions independently"
            )
        if bool(getattr(zero_optimizer, "cpu_offload", False)):
            raise RuntimeError(
                "independent_gradient_clipping does not support optimizer CPU offload"
            )
        if float(getattr(zero_optimizer, "clip_grad", -1.0)) != 0.0:
            raise RuntimeError(
                "DeepSpeed global gradient clipping must resolve to 0 for strict branch clipping; "
                f"active={getattr(zero_optimizer, 'clip_grad', None)}"
            )
        inner_groups = list(getattr(zero_optimizer.optimizer, "param_groups", ()))
        inner_partitions = {
            str(group.get("gradient_partition", "")) for group in inner_groups
        }
        if inner_partitions != partitions:
            raise RuntimeError(
                "DeepSpeed did not preserve optimizer gradient_partition metadata: "
                f"before={sorted(partitions)}, after={sorted(inner_partitions)}"
            )
        if self.accelerator.is_main_process:
            logger.info(
                "Verified ZeRO-2 independent clipping: groups=%s limits=%s global_clip=0",
                {branch: sum(g.get("gradient_partition") == branch for g in inner_groups) for branch in partitions},
                self._independent_clip_limits,
            )

    @staticmethod
    def _runtime_int(obj, name: str):
        value = getattr(obj, name, None)
        if value is None:
            return None
        value = value() if callable(value) else value
        return int(value)

    def _validate_runtime_batch_contract(self) -> None:
        """Fail immediately if Accelerate/DeepSpeed changed the YAML batch ABI."""

        expected_accumulation = int(self.config.trainer.gradient_accumulation_steps)
        active_accumulation = int(self.accelerator.gradient_accumulation_steps)
        if active_accumulation != expected_accumulation:
            raise RuntimeError(
                "Accelerate runtime accumulation differs from the training YAML: "
                f"yaml={expected_accumulation}, runtime={active_accumulation}."
            )

        expected_micro_batch = int(self.config.datasets.vla_data.per_device_batch_size)
        expected_global_batch = expected_micro_batch * int(self.accelerator.num_processes) * expected_accumulation
        declared_global_batch = self.config.trainer.get("expected_global_batch_size", None)
        if declared_global_batch is not None and expected_global_batch != int(declared_global_batch):
            raise RuntimeError(
                "Runtime topology violates trainer.expected_global_batch_size: "
                f"micro={expected_micro_batch} x world={self.accelerator.num_processes} x "
                f"accumulation={expected_accumulation} = {expected_global_batch}, "
                f"declared={int(declared_global_batch)}."
            )
        engine_values = {}
        if self._using_deepspeed:
            for name, expected in (
                ("gradient_accumulation_steps", expected_accumulation),
                ("train_micro_batch_size_per_gpu", expected_micro_batch),
                ("train_batch_size", expected_global_batch),
            ):
                active = self._runtime_int(self.model, name)
                if active is None:
                    raise RuntimeError(f"DeepSpeed engine does not expose {name}; cannot verify the batch contract.")
                engine_values[name] = active
                if active != expected:
                    raise RuntimeError(
                        f"DeepSpeed {name} differs from the training YAML/runtime topology: "
                        f"expected={expected}, active={active}."
                    )

        if self.accelerator.is_main_process:
            logger.info(
                "Verified runtime batch contract: micro=%d x world=%d x accumulation=%d = global=%d; deepspeed=%s",
                expected_micro_batch,
                self.accelerator.num_processes,
                expected_accumulation,
                expected_global_batch,
                engine_values or "disabled",
            )

    def _configure_unused_param_anchors(self) -> None:
        """Skip dense zero-gradient communication when ZeRO-2 can preserve the same optimizer update."""
        requested = bool(self.config.trainer.get("deepspeed_skip_unused_param_anchors", False))
        base_model = self.accelerator.unwrap_model(self.model)
        setter = getattr(base_model, "set_unused_param_anchors_enabled", None)
        if not requested or setter is None:
            return

        engine = self.model
        stage_fn = getattr(engine, "zero_optimization_stage", None)
        ignore_fn = getattr(engine, "zero_ignore_unused_parameters", None)
        try:
            zero_stage = int(stage_fn()) if callable(stage_fn) else -1
            ignore_unused = bool(ignore_fn()) if callable(ignore_fn) else False
        except (TypeError, ValueError):
            zero_stage = -1
            ignore_unused = False

        safe_to_skip = self._using_deepspeed and zero_stage == 2 and ignore_unused
        setter(not safe_to_skip)
        if self.accelerator.is_main_process:
            if safe_to_skip:
                logger.info(
                    "DeepSpeed ZeRO-2 unused-parameter optimization enabled: zero anchors and inactive-head "
                    "gradient communication are skipped; ZeRO supplies zero optimizer gradients."
                )
            else:
                logger.warning(
                    "trainer.deepspeed_skip_unused_param_anchors=true requires DeepSpeed ZeRO-2 with "
                    "ignore_unused_parameters=true; retaining conservative zero anchors."
                )

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = bool(getattr(self.config.trainer, "is_resume", False))
        reset_world_gates = bool(
            getattr(self.config.trainer, "reset_world_gates_after_pretrained_load", False)
        )
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            explicit_state = getattr(self.config.trainer, "resume_state_path", None)
            if explicit_state:
                state_path, state_step = self._validate_full_state_checkpoint(explicit_state)
            else:
                state_path, state_step = self._get_latest_full_state_checkpoint(self.checkpoint_dir)
            if state_path:
                self._pending_full_state_path = state_path
                self.resume_from_checkpoint = state_path
                self.completed_steps = state_step
                logger.info(
                    "Preparing full optimizer/scheduler resume from %s at step %d",
                    state_path,
                    state_step,
                )
                if reset_world_gates:
                    logger.info(
                        "Ignoring reset_world_gates_after_pretrained_load during exact full-state resume; "
                        "the restored gate values are preserved"
                    )
                return

            # Backward-compatible fallback for historical weight-only runs.
            # This cannot restore Adam moments; the scheduler is reconstructed
            # deterministically below and the limitation is made explicit.
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    "Legacy weight-only resume from %s at step %d; optimizer moments are unavailable",
                    self.resume_from_checkpoint,
                    self.completed_steps,
                )
                if reset_world_gates:
                    logger.info(
                        "Ignoring reset_world_gates_after_pretrained_load during legacy in-run resume; "
                        "the saved Stage-2 gate values are preserved"
                    )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            pretrained_strict = bool(
                getattr(self.config.trainer, "pretrained_strict", False)
            )
            self.model = self.load_pretrained_backbones(
                self.model,
                pretrained_checkpoint,
                reload_modules=reload_modules,
                strict=pretrained_strict,
            )
            if reset_world_gates:
                if reload_modules:
                    raise ValueError(
                        "reset_world_gates_after_pretrained_load requires a full pretrained load; "
                        "trainer.reload_modules must be empty"
                    )
                gate_summary = self._reset_world_gate_parameters(self.model, value=0.0)
                logger.info(
                    "Reset %d oracle-trained world gates after loading %s "
                    "(mean openness %.6f -> %.6f)",
                    gate_summary["count"],
                    pretrained_checkpoint,
                    gate_summary["before_openness"],
                    gate_summary["after_openness"],
                )
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            if reset_world_gates:
                raise ValueError(
                    "reset_world_gates_after_pretrained_load=true requires trainer.pretrained_checkpoint"
                )
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self._pending_full_state_path is not None:
            return
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    @staticmethod
    def _reset_world_gate_parameters(model, value: float = 0.0) -> dict[str, float | int]:
        """Reset only M5/M6 world residual gates after a weight-only load."""

        gates = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.rsplit(".", 1)[-1] == "world_gate"
        ]
        if not gates:
            raise RuntimeError(
                "reset_world_gates_after_pretrained_load=true but the model exposes no world_gate parameters"
            )
        before = torch.cat(
            [torch.tanh(parameter.detach().float()).abs().reshape(-1) for _, parameter in gates]
        ).mean()
        with torch.no_grad():
            for _, parameter in gates:
                parameter.fill_(float(value))
        after = torch.cat(
            [torch.tanh(parameter.detach().float()).abs().reshape(-1) for _, parameter in gates]
        ).mean()
        return {
            "count": len(gates),
            "before_openness": float(before),
            "after_openness": float(after),
        }

    @staticmethod
    def _full_state_metadata_path(state_path: str | Path) -> Path:
        return Path(state_path) / "trainer_state.json"

    @classmethod
    def _validate_full_state_checkpoint(cls, state_path: str | Path) -> tuple[str | None, int]:
        state_path = Path(state_path)
        success_path = state_path / "_SUCCESS"
        metadata_path = cls._full_state_metadata_path(state_path)
        if not state_path.is_dir() or not success_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Incomplete full training-state checkpoint: {state_path}; "
                "expected trainer_state.json and _SUCCESS"
            )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        step = int(metadata["steps"])
        match = re.fullmatch(r"steps_(\d+)_state", state_path.name)
        if match is not None and int(match.group(1)) != step:
            raise ValueError(
                f"Full-state checkpoint step mismatch: directory={state_path.name}, metadata={step}"
            )
        return str(state_path), step

    @classmethod
    def _get_latest_full_state_checkpoint(cls, checkpoint_dir: str | Path) -> tuple[str | None, int]:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            return None, 0
        candidates: list[tuple[int, str]] = []
        for state_path in checkpoint_dir.iterdir():
            if not state_path.is_dir() or re.fullmatch(r"steps_(\d+)_state", state_path.name) is None:
                continue
            try:
                validated_path, step = cls._validate_full_state_checkpoint(state_path)
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            candidates.append((step, validated_path))
        if not candidates:
            return None, 0
        step, state_path = max(candidates, key=lambda item: item[0])
        return state_path, step

    def _restore_pending_full_training_state(self) -> None:
        state_path = self._pending_full_state_path
        if state_path is None:
            return
        metadata_path = self._full_state_metadata_path(state_path)
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        saved_world_size = int(metadata.get("world_size", self.accelerator.num_processes))
        if saved_world_size != int(self.accelerator.num_processes):
            raise RuntimeError(
                "DeepSpeed full-state resume requires the saved data-parallel world size: "
                f"saved={saved_world_size}, current={self.accelerator.num_processes}"
            )
        saved_accumulation = int(
            metadata.get(
                "gradient_accumulation_steps",
                self.accelerator.gradient_accumulation_steps,
            )
        )
        if saved_accumulation != int(self.accelerator.gradient_accumulation_steps):
            raise RuntimeError(
                "Full-state resume requires the saved gradient accumulation setting: "
                f"saved={saved_accumulation}, current={self.accelerator.gradient_accumulation_steps}"
            )
        self.accelerator.load_state(input_dir=state_path)
        extras_path = Path(state_path) / "trainer_extras.pt"
        if extras_path.is_file():
            extras = torch.load(extras_path, map_location="cpu", weights_only=False)
            rng_state = extras.get("jointflow_rng_state")
            if rng_state is not None:
                self._jointflow_rng.setstate(rng_state)
        logger.info(
            "Restored full training state from %s at step %d; current LR=%s",
            state_path,
            self.completed_steps,
            self.lr_scheduler.get_last_lr(),
        )

    def _save_checkpoint(self):
        """Save current training state."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        # Do not let non-main ranks enter DeepSpeed's collective state save
        # while rank 0 is still materializing the portable weight checkpoint.
        self.accelerator.wait_for_everyone()
        if bool(getattr(self.config.trainer, "save_full_training_state", False)):
            full_state_path = checkpoint_path + "_state"
            # `_SUCCESS` is the commit marker.  Remove it before rewriting an
            # existing step so a preemption during save can never make a
            # partial optimizer shard look resumable.
            if self.accelerator.is_main_process:
                (Path(full_state_path) / "_SUCCESS").unlink(missing_ok=True)
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(output_dir=full_state_path)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                metadata = {
                    "steps": int(self.completed_steps),
                    "world_size": int(self.accelerator.num_processes),
                    "gradient_accumulation_steps": int(self.accelerator.gradient_accumulation_steps),
                }
                with self._full_state_metadata_path(full_state_path).open("w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, ensure_ascii=True, indent=2)
                torch.save(
                    {"jointflow_rng_state": self._jointflow_rng.getstate()},
                    Path(full_state_path) / "trainer_extras.pt",
                )
                (Path(full_state_path) / "_SUCCESS").touch()
                logger.info("Saved resumable optimizer/scheduler state at %s", full_state_path)

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        has_validation = any(str(key).startswith("val/") for key in metrics)
        if (
            (self.completed_steps % self.config.trainer.logging_frequency == 0 or has_validation)
            and dist.get_rank() == 0
        ):
            last_lrs = self.lr_scheduler.get_last_lr()
            if metrics.get("train/task") == "action_world_coflow":
                # Keep the permanent Co-Flow dashboard intentionally small.
                # Audit ratios remain model outputs covered by regressions but
                # aren't experiment-result time series.
                metrics = self._filter_coflow_dashboard_metrics(metrics)
                # The joint co-flow transformer lives under action_model; its
                # LR is the single policy-primary curve requested for the main
                # dashboard.  Qwen/base group details remain in saved config.
                primary_index = next(
                    (
                        index
                        for index, group in enumerate(self.optimizer.param_groups)
                        if group.get("name") == "action_model"
                    ),
                    0,
                )
                metrics["train/learning_rate"] = (
                    last_lrs[primary_index]
                    if primary_index < len(last_lrs)
                    else last_lrs[-1]
                )
            else:
                for i, group in enumerate(self.optimizer.param_groups):
                    group_name = group.get("name", str(i))
                    metrics[f"learning_rate/{group_name}"] = (
                        last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
                    )
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    @classmethod
    def _filter_coflow_dashboard_metrics(cls, metrics: dict) -> dict:
        """Pure helper used by contract tests and external dashboard tooling."""

        return {
            key: value
            for key, value in metrics.items()
            if key in cls._COFLOW_DASHBOARD_KEYS
        }

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)
        if getattr(self, "semantic_train_dataloader", None) is not None:
            # The sampler epoch is checkpointed independently.  Seed the loop
            # counter from the restored value so the first exhaustion after a
            # resume advances e.g. 17 -> 18 instead of silently resetting to 1.
            self.semantic_epoch_count = int(
                getattr(self.semantic_batch_sampler, "epoch", 0)
            )
            self.semantic_iter = iter(self.semantic_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def _get_next_semantic_batch(self):
        if getattr(self, "semantic_train_dataloader", None) is None:
            return None
        try:
            return next(self.semantic_iter)
        except StopIteration:
            if not hasattr(self, "semantic_epoch_count"):
                self.semantic_epoch_count = 0
            self.semantic_epoch_count += 1
            if self.semantic_batch_sampler is not None:
                self.semantic_batch_sampler.set_epoch(self.semantic_epoch_count)
            self.semantic_iter = iter(self.semantic_train_dataloader)
            return next(self.semantic_iter)

    #######
    def _sample_jointflow_task(self) -> str | None:
        if not self._jointflow_tasks:
            return None
        if not self._jointflow_resample and self._jointflow_cur_task is not None:
            return self._jointflow_cur_task
        idx = self._jointflow_rng.choices(range(len(self._jointflow_tasks)), weights=self._jointflow_weights, k=1)[0]
        if dist.is_available() and dist.is_initialized():
            idx_t = torch.tensor([idx], device=self.accelerator.device, dtype=torch.long)
            dist.broadcast(idx_t, src=0)
            idx = int(idx_t.item())
        self._jointflow_cur_task = self._jointflow_tasks[idx]
        self._jointflow_resample = False
        return self._jointflow_cur_task

    _TASK_LOSS_KEY = {
        "policy": "action_loss",
        "idm": "idm_loss",
        "fdm": "fdm_loss",
        "passive": "passive_loss",
        "joint_e2e": "action_loss",
        "joint_detached": "action_loss",
    }
    _TASK_LOGNAME = {
        "policy": "policy",
        "idm": "inverse",
        "fdm": "fdm",
        "passive": "passive",
        "joint_e2e": "joint_e2e",
        "joint_detached": "joint_detached",
    }

    @classmethod
    def _build_loss_metrics(cls, output_dict: dict, task: str, total_loss: torch.Tensor) -> dict:
        """Build raw, weighted, and total metrics for the sampled task only.

        Inactive tasks do not receive synthetic zero-valued metrics. Raw values
        are pre-weight visual losses; weighted values are the actual backward
        objectives. Policy and inverse losses have no internal multiplier.
        """
        if task in {"joint_e2e", "joint_detached"}:
            action = output_dict["action_loss"]
            action_raw = output_dict.get("action_loss_raw", action)
            world = output_dict["world_loss"]
            world_raw = output_dict.get("world_loss_raw", world)
            ratio = world.detach().abs() / action.detach().abs().clamp_min(1.0e-12)
            metrics = {
                "train/task": task,
                "train/loss_total": float(total_loss.detach()),
                "train/loss_policy": float(action.detach()),
                "train/loss_policy_weighted": float(action.detach()),
                "train/loss_policy_raw": float(action_raw.detach()),
                "train/loss_world": float(world.detach()),
                "train/loss_world_weighted": float(world.detach()),
                "train/loss_world_raw": float(world_raw.detach()),
                "train/world_to_policy_loss_ratio": float(ratio),
                "train/action_world_grad_scale": float(output_dict["action_world_grad_scale"].detach()),
                "train/action_world_bypassed": float(
                    output_dict.get("action_world_bypassed", action.new_zeros(())).detach()
                ),
                "train/action_backbone_detached": float(
                    output_dict.get("action_backbone_detached", action.new_zeros(())).detach()
                ),
                "train/world_backbone_detached": float(
                    output_dict.get("world_backbone_detached", action.new_zeros(())).detach()
                ),
                "train/baseline_action_context": float(
                    output_dict.get("baseline_action_context", action.new_zeros(())).detach()
                ),
                # Effective residual multiplier is tanh(raw_gate): 0=closed,
                # 1=full-magnitude world cross-attention.  Mean absolute value
                # is the clearest single "open degree"; signed/max expose
                # cancellation and one-layer outliers without logging 8 curves.
                "train/world_gate_openness": float(output_dict["world_gate_openness"].detach()),
                "train/world_gate_signed_mean": float(output_dict["world_gate_signed_mean"].detach()),
                "train/world_gate_max_openness": float(output_dict["world_gate_max_openness"].detach()),
            }
            for source_key, log_key in (
                ("world_flow_loss_raw", "train/world_flow_loss_raw"),
                ("world_clean_loss_raw", "train/world_clean_loss_raw"),
                ("world_cosine_loss_raw", "train/world_cosine_loss_raw"),
            ):
                value = output_dict.get(source_key)
                if value is not None:
                    metrics[log_key] = float(value.detach())
            text_loss = output_dict.get("text_loss")
            if text_loss is not None:
                metrics["train/loss_text"] = float(text_loss.detach())
                metrics["train/loss_text_weighted"] = float(text_loss.detach())
                metrics["train/loss_text_raw"] = float(
                    output_dict.get("text_loss_raw", text_loss).detach()
                )
                metrics["train/text_samples"] = float(
                    output_dict.get(
                        "text_samples",
                        text_loss.new_zeros(()),
                    ).detach()
                )
            return metrics
        logname = cls._TASK_LOGNAME.get(task, task)
        loss_key = cls._TASK_LOSS_KEY.get(task, f"{task}_loss")
        weighted = output_dict.get(loss_key)
        if weighted is None:
            for k, v in output_dict.items():
                if torch.is_tensor(v) and k.endswith("_loss"):
                    weighted = v
                    break
        raw = output_dict.get(f"{task}_loss_raw", weighted)
        m = {"train/task": task, "train/loss_total": float(total_loss.detach())}
        m[f"train/loss_{logname}"] = float(weighted.detach())
        m[f"train/loss_{logname}_weighted"] = float(weighted.detach())
        m[f"train/loss_{logname}_raw"] = float(raw.detach())
        # Two-stage gate runs use policy/passive task sampling rather than the
        # joint_e2e task. Forward the same detached effective-gate statistics
        # whenever the model explicitly exposes them.
        for key in (
            "world_gate_openness",
            "world_gate_signed_mean",
            "world_gate_max_openness",
        ):
            value = output_dict.get(key)
            if value is not None:
                m[f"train/{key}"] = float(value.detach())
        return m

    @staticmethod
    def _build_native_loss_metrics(output_dict: dict, total_loss: torch.Tensor) -> dict:
        """Expose weighted training objectives plus explicitly named raw losses."""

        if "mot_action_loss_raw" in output_dict:
            action_raw = output_dict["mot_action_loss_raw"].detach()
            world_raw = output_dict["mot_world_loss_raw"].detach()
            text_raw = output_dict["mot_text_loss_raw"].detach()
            action_weighted = output_dict.get(
                "mot_action_loss_weighted",
                action_raw,
            ).detach()
            world_weighted = output_dict.get(
                "mot_world_loss_weighted",
                world_raw,
            ).detach()
            text_weighted = output_dict.get(
                "mot_text_loss_weighted",
                text_raw,
            ).detach()
            metrics = {
                "train/task": "world_action_mot",
                "train/loss_total": float(total_loss.detach()),
                # Un-suffixed curves are the actual weighted objectives used
                # by backward. Explicit raw curves remain available for loss
                # scale/debug audits.
                "train/action_loss": float(action_weighted),
                "train/world_loss": float(world_weighted),
                "train/text_loss": float(text_weighted),
                "train/action_loss_raw": float(action_raw),
                "train/world_loss_raw": float(world_raw),
                "train/text_loss_raw": float(text_raw),
                "train/action_loss_weighted": float(action_weighted),
                "train/world_loss_weighted": float(world_weighted),
                "train/text_loss_weighted": float(text_weighted),
                "train/text_sample_count": float(
                    output_dict["mot_text_sample_count"].detach()
                ),
            }
            for source_key, metric_key in (
                ("mot_world_loss_weight", "train/world_loss_weight"),
                ("mot_text_loss_weight", "train/text_loss_weight"),
                ("mot_text_decision_loss", "train/text_decision_loss"),
                (
                    "mot_text_update_body_loss",
                    "train/text_update_body_loss",
                ),
                (
                    "mot_text_decision_accuracy",
                    "train/text_decision_accuracy",
                ),
                ("mot_text_update_count", "train/text_update_count"),
                (
                    "mot_text_scheduled_probability",
                    "train/text_scheduled_probability",
                ),
                (
                    "mot_text_scheduled_count",
                    "train/text_scheduled_count",
                ),
                (
                    "mot_text_scheduled_update_count",
                    "train/text_scheduled_update_count",
                ),
            ):
                value = output_dict.get(source_key)
                if value is not None:
                    metrics[metric_key] = float(value.detach())
            return metrics
        if "coflow_action_loss_raw" not in output_dict:
            value = float(total_loss.detach())
            return {
                "train/task": "action",
                "train/loss_total": value,
                "train/loss_action": value,
                "train/loss_action_raw": value,
                "train/loss_action_weighted": value,
            }
        # ``action_loss`` remains the trainer's single optimization ABI and is
        # the weighted action+world total for Co-Flow.  Do not label that total
        # as policy loss in W&B.
        metrics = {
            "train/task": "action_world_coflow",
            "train/loss_total": float(total_loss.detach()),
            "train/action_loss_raw": float(output_dict["coflow_action_loss_raw"].detach()),
            "train/z16_loss_raw": float(output_dict["coflow_z16_loss_raw"].detach()),
            "train/world_loss_raw": float(output_dict["coflow_world_loss_raw"].detach()),
            "train/action_loss_weighted": float(
                output_dict["coflow_action_loss_weighted"].detach()
            ),
            "train/world_loss_weighted": float(
                output_dict["coflow_world_loss_weighted"].detach()
            ),
        }
        if "coflow_world_loss_weight" in output_dict:
            metrics["train/world_loss_weight"] = float(
                output_dict["coflow_world_loss_weight"].detach()
            )
        return metrics

    def _current_display_task_name(self) -> str:
        """Return the progress/timing label without affecting task sampling."""

        if self._jointflow_cur_task is not None:
            return str(self._jointflow_cur_task)
        framework_cfg = getattr(self.config, "framework", None)
        if bool(
            framework_cfg is not None
            and framework_cfg.get("enable_action_world_coflow", False)
        ):
            return "action_world_coflow"
        if bool(
            framework_cfg is not None
            and framework_cfg.get("enable_world_action_mot", False)
        ):
            return "world_action_mot"
        return "action"

    def _is_log_step(self, sync_gradients: bool) -> bool:
        """Return whether this synchronized optimizer step should be logged."""
        return bool(sync_gradients) and (self.completed_steps + 1) % self.config.trainer.logging_frequency == 0

    def _global_grad_norm(self):
        """Read the unscaled, globally reduced, pre-clipping gradient norm."""
        if getattr(self, "_independent_gradient_clipping", False) and self._last_clip_norm is not None:
            return self._last_clip_norm
        try:
            m = self.model
            if hasattr(m, "get_global_grad_norm"):
                v = m.get_global_grad_norm()
                if v is not None:
                    return float(v)
            eng = getattr(getattr(self.accelerator, "deepspeed_engine_wrapped", None), "engine", None)
            if eng is not None and hasattr(eng, "get_global_grad_norm"):
                v = eng.get_global_grad_norm()
                if v is not None:
                    return float(v)
        except Exception:
            pass
        return self._last_clip_norm

    @staticmethod
    def _shared_vlm_interface_tensors(interface_tensors) -> tuple[torch.Tensor, ...]:
        if not isinstance(interface_tensors, (tuple, list)):
            raise TypeError("mot_shared_vlm_interface must be a tuple/list of tensors")
        tracked = tuple(
            tensor
            for tensor in interface_tensors
            if torch.is_tensor(tensor) and tensor.requires_grad
        )
        if not tracked:
            raise RuntimeError(
                "shared-VLM interface gradient diagnostics found no differentiable "
                "planner-query tensors"
            )
        return tracked

    @staticmethod
    def _shared_vlm_interface_metrics_from_gradients(
        tracked: tuple[torch.Tensor, ...],
        action_grads,
        world_grads,
    ) -> dict[str, float]:
        if len(action_grads) != len(tracked) or len(world_grads) != len(tracked):
            raise ValueError("shared-VLM gradient tuples must match interface tensors")
        squared_norms = torch.zeros(2, device=tracked[0].device, dtype=torch.float32)
        for action_grad, world_grad in zip(
            action_grads,
            world_grads,
            strict=True,
        ):
            if action_grad is None and world_grad is None:
                continue
            action_float = (
                action_grad.detach().float()
                if action_grad is not None
                else None
            )
            world_float = (
                world_grad.detach().float()
                if world_grad is not None
                else None
            )
            if action_float is not None:
                squared_norms[0] += torch.sum(action_float.square())
            if world_float is not None:
                squared_norms[1] += torch.sum(world_float.square())

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(squared_norms, op=dist.ReduceOp.SUM)

        return {
            "train/shared_vlm_interface_grad_norm_action": float(
                squared_norms[0].clamp_min(0.0).sqrt()
            ),
            "train/shared_vlm_interface_grad_norm_world": float(
                squared_norms[1].clamp_min(0.0).sqrt()
            ),
        }

    @staticmethod
    def _shared_vlm_interface_gradient_metrics(
        action_objective: torch.Tensor,
        world_objective: torch.Tensor,
        interface_tensors,
    ) -> dict[str, float]:
        """Standalone exact metric helper retained for audits and unit tests."""

        if not torch.is_tensor(action_objective) or action_objective.ndim != 0:
            raise ValueError("mot_action_objective must be a scalar tensor")
        if not torch.is_tensor(world_objective) or world_objective.ndim != 0:
            raise ValueError("mot_world_objective must be a scalar tensor")
        tracked = VLATrainer._shared_vlm_interface_tensors(interface_tensors)
        action_grads = torch.autograd.grad(
            action_objective,
            tracked,
            retain_graph=True,
            allow_unused=True,
        )
        world_grads = torch.autograd.grad(
            world_objective,
            tracked,
            retain_graph=True,
            allow_unused=True,
        )
        return VLATrainer._shared_vlm_interface_metrics_from_gradients(
            tracked,
            action_grads,
            world_grads,
        )

    @staticmethod
    def _prepare_shared_vlm_interface_gradient_probe(
        action_objective: torch.Tensor,
        world_objective: torch.Tensor,
        interface_tensors,
    ):
        """Prepare one auxiliary gradient before the real optimizer backward.

        The real backward already produces ``g_action + g_world`` at every
        retained planner-query tensor.  Computing only ``g_world`` here lets
        finalization recover ``g_action = g_total - g_world`` exactly, cutting
        the diagnostic from two auxiliary branch backwards to one.  The world
        expert is the smaller branch in the RoboDojo MoT recipes.
        """

        if not torch.is_tensor(action_objective) or action_objective.ndim != 0:
            raise ValueError("mot_action_objective must be a scalar tensor")
        if not torch.is_tensor(world_objective) or world_objective.ndim != 0:
            raise ValueError("mot_world_objective must be a scalar tensor")
        tracked = VLATrainer._shared_vlm_interface_tensors(interface_tensors)
        if any(tensor.is_leaf for tensor in tracked):
            raise RuntimeError(
                "optimized shared-VLM diagnostics require non-leaf planner-query "
                "interface tensors so retained gradients cannot alias parameters"
            )
        for tensor in tracked:
            tensor.retain_grad()
        world_grads = torch.autograd.grad(
            world_objective,
            tracked,
            retain_graph=True,
            allow_unused=True,
        )
        # ``retain_grad`` installs hooks that also observe ``autograd.grad``;
        # clear those auxiliary values so the subsequent real backward stores
        # only g_total rather than g_world + g_total.
        for tensor in tracked:
            tensor.grad = None
        return tracked, world_grads

    @staticmethod
    def _finalize_shared_vlm_interface_gradient_probe(
        probe,
    ) -> dict[str, float]:
        """Recover exact per-task interface gradients after the real backward."""

        tracked, world_grads = probe
        total_grads = tuple(tensor.grad for tensor in tracked)
        for tensor in tracked:
            tensor.grad = None

        action_grads = []
        for total_grad, world_grad in zip(total_grads, world_grads, strict=True):
            if total_grad is None:
                if world_grad is not None:
                    raise RuntimeError(
                        "real optimizer backward did not retain a shared-VLM "
                        "interface gradient required by the diagnostic"
                    )
                action_grads.append(None)
            elif world_grad is None:
                action_grads.append(total_grad)
            else:
                action_grads.append(total_grad - world_grad)

        return VLATrainer._shared_vlm_interface_metrics_from_gradients(
            tracked,
            tuple(action_grads),
            world_grads,
        )

    def _should_measure_shared_vlm_interface_gradients(
        self,
        *,
        sync_gradients: bool,
    ) -> bool:
        return bool(
            sync_gradients
            and getattr(
                self,
                "_shared_vlm_gradient_diagnostics_enabled",
                False,
            )
            and (self.completed_steps + 1)
            % int(self._shared_vlm_gradient_diagnostics_interval)
            == 0
        )

    def _backward_for_current_step(self, loss: torch.Tensor, *, sync: bool) -> None:
        """Backward without letting Accelerate step before strict clipping.

        Accelerate's DeepSpeed wrapper calls ``engine.step()`` inside
        ``accelerator.backward`` at an accumulation boundary.  Independent
        action/world clipping must run after ZeRO reduce-scatter but before
        that step, so the opt-in path calls the engine primitives explicitly.
        Every legacy run continues through ``accelerator.backward`` unchanged.
        """

        if not (
            getattr(self, "_independent_gradient_clipping", False)
            and self._using_deepspeed
        ):
            self.accelerator.backward(loss)
            return
        engine = self.model
        engine.set_gradient_accumulation_boundary(is_boundary=bool(sync))
        engine.backward(loss)

    @torch.no_grad()
    def _clip_independent_gradient_partitions(self) -> dict[str, torch.Tensor]:
        """Clip disjoint action/world gradients and return pre-clip metrics."""

        if not self._independent_gradient_clipping:
            return {}
        if self._using_deepspeed:
            return self._clip_independent_zero2_partitions()

        metrics: dict[str, torch.Tensor] = {}
        for branch, max_norm in self._independent_clip_limits.items():
            parameters = []
            seen: set[int] = set()
            for group in self.optimizer.param_groups:
                if str(group.get("gradient_partition", "")) != branch:
                    continue
                for parameter in group["params"]:
                    if id(parameter) not in seen:
                        seen.add(id(parameter))
                        parameters.append(parameter)
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=float(max_norm))
            norm = torch.as_tensor(norm).detach()
            metrics[f"train/grad_norm_preclip/{branch}"] = norm
            metrics[f"train/grad_clip_scale/{branch}"] = torch.clamp(
                norm.new_tensor(float(max_norm)) / (norm + 1.0e-6),
                max=1.0,
            )
        return metrics

    @torch.no_grad()
    def _clip_independent_zero2_partitions(self) -> dict[str, torch.Tensor]:
        """Scale ZeRO-2 reduced shards independently before engine.step()."""

        zero_optimizer = self.model.optimizer
        inner_groups = list(zero_optimizer.optimizer.param_groups)
        scaled_sq = {
            branch: torch.zeros((), device=self.accelerator.device, dtype=torch.float32)
            for branch in self._independent_clip_limits
        }
        group_branches: dict[int, str] = {}
        for index, group in enumerate(inner_groups):
            branch = str(group.get("gradient_partition", ""))
            if branch not in scaled_sq:
                raise RuntimeError(
                    f"ZeRO optimizer group {index} has invalid gradient_partition={branch!r}"
                )
            gradients = zero_optimizer.averaged_gradients.get(index)
            if gradients is None:
                continue
            params = zero_optimizer.params_in_partition[index]
            group_norm = zero_optimizer.get_grad_norm_direct(gradients, params)
            scaled_sq[branch] = scaled_sq[branch] + group_norm.float().square()
            group_branches[index] = branch

        loss_scale = float(zero_optimizer.loss_scale)
        if loss_scale <= 0.0:
            raise RuntimeError(f"DeepSpeed loss_scale must be positive, got {loss_scale}")
        scales: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}
        for branch, norm_sq in scaled_sq.items():
            norm = norm_sq.sqrt() / loss_scale
            limit = float(self._independent_clip_limits[branch])
            finite_scale = torch.clamp(
                norm.new_tensor(limit) / (norm + 1.0e-6),
                max=1.0,
            )
            # DeepSpeed's get_grad_norm_direct replaces NaN/Inf norms with a
            # negative sentinel. Preserve those gradients unchanged so its
            # overflow detector can skip the step and update the loss scale.
            valid_norm = torch.isfinite(norm) & (norm >= 0.0)
            scale = torch.where(valid_norm, finite_scale, norm.new_ones(()))
            scales[branch] = scale
            metrics[f"train/grad_norm_preclip/{branch}"] = norm.detach()
            metrics[f"train/grad_clip_scale/{branch}"] = scale.detach()

        for index, branch in group_branches.items():
            scale = scales[branch]
            for gradient in zero_optimizer.averaged_gradients[index]:
                gradient.mul_(scale.to(device=gradient.device, dtype=gradient.dtype))
        return metrics

    def _compute_head_sqnorms(self, task):
        """Return the active head's local sharded squared gradient norm.

        This runs after backward and before step with no collective. Inactive
        heads are zero-anchored, so one later scalar all-reduce is sufficient
        to recover the exact global head norm.
        """
        active = self._task_head.get(task) if task is not None else None
        if active is None or active not in self._grad_groups:
            return None, None
        local_sq = group_grad_local_sqnorm(self._grad_groups[active])
        logname = self._task_logname.get(task, task)
        return logname, local_sq

    def _finalize_grad_metrics(self, task, head_logname, head_local_sq) -> dict:
        """Finalize task-scoped total, shared-backbone, and head grad norms.

        Each optimizer step samples one task, so every curve is attributed to
        that task directly. ``log_grad_norms=false`` bypasses this path. The
        head norm requires one scalar all-reduce; the disjoint shared norm is
        recovered with ``sqrt(total^2 - head^2)``.
        """
        m = {}
        logname = head_logname or (self._task_logname.get(task, task) if task is not None else "action")
        total = self._global_grad_norm()
        if total is not None:
            m[f"train/grad_norm_total/{logname}"] = total
        if head_local_sq is not None:
            if self._using_deepspeed and dist.is_available() and dist.is_initialized():
                dist.all_reduce(head_local_sq, op=dist.ReduceOp.SUM)
            head_sq = float(head_local_sq.item())
            head_norm = head_sq ** 0.5
            if task in {"joint_e2e", "joint_detached"}:
                m["train/world_head_grad_norm"] = head_norm
                if total is not None:
                    m["train/policy_grad_norm"] = max(
                        total * total - head_sq,
                        0.0,
                    ) ** 0.5
            else:
                m[f"train/grad_norm_head/{logname}"] = head_norm
            if total is not None and task not in {"joint_e2e", "joint_detached"}:
                m[f"train/grad_norm_shared/{logname}"] = max(total * total - head_sq, 0.0) ** 0.5
        return m
    #######

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            # AIDI merges stdout from every node.  ``is_local_main_process``
            # therefore produced one duplicate progress bar per machine.
            disable=not self.accelerator.is_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            batch_semantic = self._get_next_semantic_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla, batch_semantic=batch_semantic)
            t_end_model = time.perf_counter()
            data_elapsed = t_end_data - t_start_data
            model_elapsed = t_end_model - t_start_model
            task_name = self._current_display_task_name()
            did_optimizer_step = bool(self.accelerator.sync_gradients)

            if did_optimizer_step:
                progress_bar.update(1)
                self.completed_steps += 1

            # Show one progress record per optimizer step. With accumulation,
            # printing every micro-batch duplicates the same step number and
            # makes the task stream look like alternating optimization.
            if did_optimizer_step and self.accelerator.is_main_process:
                progress_bar.set_postfix(
                    {
                        "rank": self.accelerator.process_index,
                        "task": task_name,
                        "data_times": f"{data_elapsed:.3f}",
                        "model_times": f"{model_elapsed:.3f}",
                    }
                )

            action_eval_enabled = bool(self.config.trainer.get("action_eval_enabled", True))
            action_eval_interval = int(self.config.trainer.get("eval_interval", 0))
            if (
                did_optimizer_step
                and action_eval_enabled
                and action_eval_interval > 0
                and self.completed_steps % action_eval_interval == 0
            ):
                step_metrics = self.eval_action_model(step_metrics)

            world_validation = self.config.trainer.get("world_validation", {})
            world_val_enabled = (
                self.vla_val_dataloader is not None
                and hasattr(world_validation, "get")
                and bool(world_validation.get("enabled", False))
            )
            world_val_interval = int(world_validation.get("interval", 0)) if world_val_enabled else 0
            if (
                did_optimizer_step
                and world_val_enabled
                and world_val_interval > 0
                and self.completed_steps % world_val_interval == 0
            ):
                step_metrics.update(self.eval_world_model())

            # Eval/log/save are optimizer-step events.  An accumulation
            # micro-batch keeps ``completed_steps`` unchanged; running these
            # hooks there duplicates expensive evals and checkpoint writes.
            if did_optimizer_step:
                step_metrics["timing/data"] = data_elapsed
                step_metrics["timing/model"] = model_elapsed
                step_metrics[f"timing/model_{task_name}"] = model_elapsed
                self._log_metrics(step_metrics)

            if (
                did_optimizer_step
                and self.completed_steps % self.config.trainer.save_interval == 0
                and self.completed_steps > 0
            ):
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_world_model(self) -> dict[str, float]:
        """Evaluate sampled future-DINO MSE on the fixed held-out split."""

        if self.vla_val_dataloader is None:
            return {}
        cfg = self.config.trainer.get("world_validation", {})
        num_batches = int(cfg.get("num_batches", 4))
        num_samples = int(cfg.get("num_samples", 1))
        num_steps = int(cfg.get("num_inference_timesteps", 10))
        seed = int(cfg.get("seed", 2027))
        task = str(cfg.get("task", "joint_detached"))
        if num_batches <= 0 or num_samples <= 0 or num_steps <= 0:
            raise ValueError(
                "world_validation num_batches/num_samples/num_inference_timesteps must be positive"
            )

        base_model = self.accelerator.unwrap_model(self.model)
        evaluator = getattr(base_model, "evaluate_wam_world_prediction", None)
        if not callable(evaluator):
            raise RuntimeError(
                "trainer.world_validation requires a WAM model exposing "
                "evaluate_wam_world_prediction"
            )
        was_training = bool(base_model.training)
        base_model.eval()
        totals = torch.zeros(6, device=self.accelerator.device, dtype=torch.float64)
        try:
            for batch_index, examples in enumerate(self.vla_val_dataloader):
                if batch_index >= num_batches:
                    break
                device_type = self.accelerator.device.type
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=device_type in {"cuda", "cpu"},
                ):
                    values = evaluator(
                        examples,
                        task=task,
                        seed=seed + batch_index * 1_000_003,
                        num_samples=num_samples,
                        num_inference_timesteps=num_steps,
                    )
                totals += torch.stack(
                    [
                        values["squared_error_sum"],
                        values["element_count"],
                        values["cosine_sum"],
                        values["token_count"],
                        values["copy_squared_error_sum"],
                        values["valid_sample_count"],
                    ]
                ).to(device=totals.device, dtype=totals.dtype)
        finally:
            if was_training:
                base_model.train()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if totals[1].item() <= 0 or totals[3].item() <= 0:
            raise RuntimeError(
                "World validation found no valid t+stride targets. Check the held-out split and future_valid contract."
            )
        mse = totals[0] / totals[1]
        cosine = totals[2] / totals[3]
        copy_mse = totals[4] / totals[1]
        metrics = {
            "val/world_mse": float(mse),
            "val/world_cosine": float(cosine),
            "val/world_copy_mse": float(copy_mse),
            "val/world_mse_vs_copy_ratio": float(mse / copy_mse.clamp_min(1.0e-12)),
            "val/world_valid_samples": float(totals[5]),
        }
        if self.accelerator.is_main_process:
            logger.info(
                "World validation step=%d task=%s batches/rank=%d samples=%d steps=%d "
                "mse=%.6f cosine=%.6f copy_mse=%.6f ratio=%.4f valid=%d",
                self.completed_steps,
                task,
                num_batches,
                num_samples,
                num_steps,
                metrics["val/world_mse"],
                metrics["val/world_cosine"],
                metrics["val/world_copy_mse"],
                metrics["val/world_mse_vs_copy_ratio"],
                int(metrics["val/world_valid_samples"]),
            )
        return metrics

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        base_model = self.accelerator.unwrap_model(self.model)
        was_training = bool(base_model.training)
        base_model.eval()
        try:
            # Let each framework use its configured inference schedule.  The
            # previous hard-coded 20 overrode RoboDojo MoT's train/deploy
            # contract of 10 steps.
            output_dict = base_model.predict_action(examples=examples, use_ddim=True)
        finally:
            if was_training:
                base_model.train()

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None, batch_semantic=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                #######
                jointflow_task = self._sample_jointflow_task()
                if batch_semantic is not None:
                    if jointflow_task is not None:
                        raise RuntimeError(
                            "Event-memory semantic NTP cannot be combined with the "
                            "alternating JointFlow task sampler"
                        )
                    output_dict = self.model.forward(
                        batch_vla,
                        semantic_examples=batch_semantic,
                        global_step=self.completed_steps,
                    )
                    total_loss = output_dict["action_loss"]
                elif jointflow_task is not None:
                    output_dict = self.model.forward(
                        batch_vla,
                        task=jointflow_task,
                        global_step=self.completed_steps,
                    )
                    loss_values = [
                        value
                        for key, value in output_dict.items()
                        if torch.is_tensor(value) and key.endswith("_loss")
                    ]
                    if not loss_values:
                        raise KeyError(
                            f"JointFlow task `{jointflow_task}` returned no tensor loss keys: {output_dict.keys()}"
                        )
                    total_loss = sum(loss_values)
                else:
                    # Co-Flow's scheduled GT->predicted Z16 exposure is tied
                    # to optimizer steps.  No legacy framework receives a new
                    # argument unless the explicit opt-in flag is active.
                    framework_cfg = getattr(self.config, "framework", None)
                    coflow_enabled = bool(
                        framework_cfg is not None
                        and framework_cfg.get("enable_action_world_coflow", False)
                    )
                    if coflow_enabled:
                        output_dict = self.model.forward(batch_vla, global_step=self.completed_steps)
                    else:
                        output_dict = self.model.forward(batch_vla)
                    action_loss = output_dict["action_loss"]
                    total_loss = action_loss
                #######

            sync = bool(self.accelerator.sync_gradients)
            will_log = self._is_log_step(sync)
            shared_vlm_gradient_metrics: dict[str, float] = {}
            shared_vlm_gradient_probe = None
            if self._should_measure_shared_vlm_interface_gradients(
                sync_gradients=sync
            ):
                required = (
                    "mot_action_objective",
                    "mot_world_objective",
                    "mot_shared_vlm_interface",
                )
                missing = [key for key in required if key not in output_dict]
                if missing:
                    raise RuntimeError(
                        "Shared-VLM gradient diagnostics require World--Action "
                        f"MoT outputs {missing}; active output keys="
                        f"{sorted(output_dict)}"
                    )
                shared_vlm_gradient_probe = (
                    self._prepare_shared_vlm_interface_gradient_probe(
                        output_dict["mot_action_objective"],
                        output_dict["mot_world_objective"],
                        output_dict["mot_shared_vlm_interface"],
                    )
                )
            self._backward_for_current_step(total_loss, sync=sync)
            if shared_vlm_gradient_probe is not None:
                shared_vlm_gradient_metrics = (
                    self._finalize_shared_vlm_interface_gradient_probe(
                        shared_vlm_gradient_probe,
                    )
                )

            #######
            independent_clipping = bool(
                getattr(self, "_independent_gradient_clipping", False)
            )
            self._last_clip_norm = None
            independent_clip_metrics: dict[str, torch.Tensor] = {}
            if sync and independent_clipping:
                independent_clip_metrics = self._clip_independent_gradient_partitions()
            elif sync and self.config.trainer.gradient_clipping is not None:
                clipped = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
                if clipped is not None:
                    try:
                        self._last_clip_norm = float(clipped)
                    except Exception:
                        self._last_clip_norm = None

            if will_log and independent_clip_metrics:
                action_norm = independent_clip_metrics[
                    "train/grad_norm_preclip/action"
                ].float()
                world_norm = independent_clip_metrics[
                    "train/grad_norm_preclip/world"
                ].float()
                self._last_clip_norm = float(
                    torch.linalg.vector_norm(torch.stack((action_norm, world_norm)))
                )
            want_global_grad = will_log and bool(
                getattr(self, "_log_global_grad_norm", False)
            )
            # Independent clipping already reports exact pre-clip action/world
            # norms.  The legacy head decomposition is post-clip and would mix
            # incompatible quantities in this opt-in path.
            want_grad = (
                will_log
                and not independent_clipping
                and self._log_grad_norms
                and self._grad_groups is not None
            )

            head_logname, head_local_sq = None, None
            if want_grad:
                try:
                    head_logname, head_local_sq = self._compute_head_sqnorms(jointflow_task)
                except Exception as e:
                    if not self._grad_warned and self.accelerator.is_main_process:
                        logger.warning("per-head grad-norm failed (%s); skipping per-head grad logs hereafter.", e)
                    self._grad_warned = True
                    head_logname, head_local_sq = None, None
            #######

            if independent_clipping and self._using_deepspeed:
                if sync:
                    self.model.step()
            else:
                self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if sync:
                self.lr_scheduler.step()
                #######
                self._jointflow_resample = True
                #######

            #######
            grad_metrics = dict(shared_vlm_gradient_metrics)
            if will_log and independent_clip_metrics:
                grad_metrics.update(
                    {
                        key: float(value)
                        for key, value in independent_clip_metrics.items()
                    }
                )
            if want_global_grad:
                try:
                    total_grad_norm = self._global_grad_norm()
                    if total_grad_norm is not None:
                        grad_metrics["train/grad_norm"] = total_grad_norm
                except Exception as e:
                    if not self._grad_warned and self.accelerator.is_main_process:
                        logger.warning("global grad-norm logging failed (%s); skipping.", e)
                    self._grad_warned = True
            if want_grad:  # Optional detailed shared/head decomposition.
                try:
                    grad_metrics.update(
                        self._finalize_grad_metrics(jointflow_task, head_logname, head_local_sq)
                    )
                except Exception as e:
                    if not self._grad_warned and self.accelerator.is_main_process:
                        logger.warning("grad-norm finalize failed (%s); skipping.", e)
                    self._grad_warned = True
            # Accelerate performs ``step``/``zero_grad`` only on the sync
            # micro-batch.  Clearing at that micro-batch's start discards the
            # gradients accumulated by all preceding micro-batches.  Clearing
            # here, after step and optional gradient logging, preserves the
            # configured effective batch size.
            self.optimizer.zero_grad()
            #######

        #######
        if not will_log:
            return {}
        if jointflow_task is not None:
            metrics = self._build_loss_metrics(output_dict, jointflow_task, total_loss)
        else:
            metrics = self._build_native_loss_metrics(output_dict, total_loss)
        metrics.update(grad_metrics)
        return metrics
        #######

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    global accelerator
    if accelerator is None:
        accelerator = build_training_accelerator(cfg)
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    reproduction_profile = str(
        cfg.framework.get("reproduction_profile", "") or ""
    )
    if reproduction_profile:
        from starVLA.reproducibility.robodojo_rynn50k import (
            ROBODOJO_RYNN50K_PROFILE,
            validate_config as validate_robodojo_rynn50k_config,
        )

        if reproduction_profile == ROBODOJO_RYNN50K_PROFILE:
            reproduction_summary = validate_robodojo_rynn50k_config(cfg)
            logger.info("Frozen reproduction contract verified: %s", reproduction_summary)

    output_dir = setup_directories(cfg=cfg)
    if bool(cfg.trainer.get("seed_before_model_init", False)):
        # Opt-in for controlled ablations.  Use one common construction seed
        # on every rank so optional zero-gated modules can be proven not to
        # perturb the baseline policy initialization.  prepare_training later
        # restores the historical rank-offset runtime seed for data/noise.
        construction_seed = int(getattr(cfg, "seed", 42))
        set_seed(construction_seed)
        logger.info("Deterministic model construction seed=%d", construction_seed)
    vla = build_framework(cfg)
    (
        vla_train_dataloader,
        semantic_train_dataloader,
        vla_val_dataloader,
    ) = prepare_data(
        cfg=cfg, accelerator=accelerator, output_dir=output_dir
    )
    if reproduction_profile == "robodojo_rynnbrain11_causal_dino_mot_50k_v1":
        from starVLA.reproducibility.robodojo_rynn50k import (
            validate_dataset_statistics as validate_robodojo_rynn50k_statistics,
        )

        stats_sha256 = validate_robodojo_rynn50k_statistics(
            output_dir / "dataset_statistics.json"
        )
        logger.info(
            "Frozen RoboDojo v1 statistics verified: sha256=%s",
            stats_sha256,
        )
    #######
    action_cfg = getattr(getattr(cfg, "framework", None), "action_model", None)
    if (
        action_cfg is not None
        and bool(action_cfg.get("use_correlated_noise", False))
    ):
        if not hasattr(vla, "set_action_correlation"):
            raise RuntimeError(
                "framework.action_model.use_correlated_noise=true but the selected framework "
                "does not implement set_action_correlation."
            )
        from starVLA.dataloader.action_correlation import validate_action_correlation_cholesky
        from starVLA.dataloader.jointflow.joint_dataset import compute_action_correlation_cholesky
        import numpy as _np

        #######
        cache_path = os.path.join(output_dir, "action_correlation_cholesky.npy")
        cache_metadata_path = os.path.join(output_dir, "action_correlation_metadata.json")
        _exp = int(action_cfg.get("action_horizon", 0)) * int(action_cfg.get("action_dim", 0))
        if _exp <= 0:
            raise ValueError(
                "correlated-noise requires positive action_horizon and action_dim, "
                f"got horizon={action_cfg.get('action_horizon')}, dim={action_cfg.get('action_dim')}"
            )
        _source_path = action_cfg.get("correlation_cholesky_path")
        _source_chol = None
        if _source_path:
            # Validate on every rank before the barrier. If the shared source is
            # missing/corrupt, all ranks fail promptly instead of leaving peers
            # blocked while only rank 0 raises.
            _source_path = os.path.expandvars(os.path.expanduser(str(_source_path)))
            if not os.path.isfile(_source_path):
                raise FileNotFoundError(
                    "Configured correlation_cholesky_path does not exist; refusing to recompute a different "
                    f"ablation matrix: {_source_path}"
                )
            try:
                _source_chol = validate_action_correlation_cholesky(
                    _np.load(_source_path, allow_pickle=False),
                    expected_size=_exp,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid correlation_cholesky_path {_source_path}: {exc}") from exc

        _correlation_spec = {
            "schema_version": 1,
            "data_root_dir": str(cfg.datasets.vla_data.data_root_dir),
            "data_mix": str(cfg.datasets.vla_data.data_mix),
            "action_horizon": int(action_cfg.get("action_horizon")),
            "action_dim": int(action_cfg.get("action_dim")),
            "matrix_type": str(action_cfg.get("correlation_matrix_type", "covariance")).lower(),
            "beta": float(action_cfg.get("correlation_beta", 0.5)),
            "num_samples": int(action_cfg.get("correlation_num_samples", 4096)),
            "source_path": str(_source_path) if _source_path else None,
        }

        def _write_correlation_metadata() -> None:
            with open(cache_metadata_path, "w", encoding="utf-8") as handle:
                json.dump(_correlation_spec, handle, indent=2, sort_keys=True)

        def _reusable_local_correlation() -> bool:
            if not os.path.isfile(cache_path) or not os.path.isfile(cache_metadata_path):
                return False
            try:
                with open(cache_metadata_path, "r", encoding="utf-8") as handle:
                    cached_spec = json.load(handle)
                if cached_spec != _correlation_spec:
                    logger.warning(
                        "correlated-noise: refusing stale cache because provenance changed: cached=%s current=%s",
                        cached_spec,
                        _correlation_spec,
                    )
                    return False
                validate_action_correlation_cholesky(
                    _np.load(cache_path, allow_pickle=False),
                    expected_size=_exp,
                )
                return True
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("correlated-noise: invalid cache %s; recomputing (%s)", cache_path, exc)
                return False

        if accelerator.is_main_process:
            #######
            if _source_path:
                _np.save(cache_path, _np.asarray(_source_chol))
                _write_correlation_metadata()
                logger.info(
                    "correlated-noise: copied fixed baseline Cholesky(%sx%s) from %s -> %s",
                    _exp,
                    _exp,
                    _source_path,
                    cache_path,
                )
            else:
                _reuse = _reusable_local_correlation()
                if _reuse:
                    logger.info(f"correlated-noise: reusing cached Cholesky ({_exp}x{_exp}) at {cache_path}")
                else:
                    chol = compute_action_correlation_cholesky(
                        vla_train_dataloader.dataset,
                        num_samples=int(action_cfg.get("correlation_num_samples", 4096)),
                        beta=float(action_cfg.get("correlation_beta", 0.5)),
                        matrix_type=str(action_cfg.get("correlation_matrix_type", "covariance")),
                    )
                    chol = validate_action_correlation_cholesky(chol, expected_size=_exp)
                    _np.save(cache_path, chol)
                    _write_correlation_metadata()
                    logger.info(
                        "correlated-noise Cholesky ready: shape=%s, beta=%s, matrix_type=%s -> %s",
                        chol.shape,
                        action_cfg.get("correlation_beta", 0.5),
                        action_cfg.get("correlation_matrix_type", "covariance"),
                        cache_path,
                    )
        if dist.is_initialized():
            dist.barrier()
        try:
            _loaded_chol = validate_action_correlation_cholesky(
                _np.load(cache_path, allow_pickle=False),
                expected_size=_exp,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid correlated-noise Cholesky after synchronization: {cache_path}: {exc}"
            ) from exc
        vla.set_action_correlation(_loaded_chol)
        #######
    #######
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        semantic_train_dataloader=semantic_train_dataloader,
        vla_val_dataloader=vla_val_dataloader,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    # Must happen after YAML/CLI merging: DeepSpeed's config uses "auto" and
    # therefore needs the run-specific accumulation value at construction.
    accelerator = build_training_accelerator(cfg)
    configure_torch_runtime()

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
