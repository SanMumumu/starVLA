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
# 中文注释：JointFlow 多任务训练需要在原生 trainer 内按 framework.tasks.weights 采样任务。
import random
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
from accelerate.utils import set_seed
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

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

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


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader


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
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        #######
        # 中文注释：JointFlow 任务采样器只在配置含 framework.tasks.weights 时启用。
        # 一任务/优化步（累积窗口内同 task）+ rank0 采样后 dist.broadcast 任务索引到各 rank，
        # 彻底避免"靠同种子保持一致"的脆弱性（步数不齐/跳步会失同步 → DDP 参数 ready 不一致挂死）。
        self._jointflow_rng = random.Random(int(getattr(cfg, "seed", 42)))
        self._jointflow_tasks = None
        self._jointflow_weights = None
        self._jointflow_cur_task = None
        self._jointflow_resample = True  # 新优化步起点置 True，窗口内复用已广播的 task
        framework_cfg = getattr(cfg, "framework", None)
        jointflow_cfg = getattr(framework_cfg, "jointflow", None) if framework_cfg is not None else None
        tasks_cfg = getattr(framework_cfg, "tasks", None) if framework_cfg is not None else None
        #######
        # 中文注释：多任务采样既服务 jointflow，也服务 wam（四范式 policy/passive/fdm/idm）；任一开关开 + tasks.weights 即生效。
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
                if self.accelerator.is_main_process:
                    logger.info(
                        "Active QwenGR00T task sampler: tasks=%s weights=%s",
                        self._jointflow_tasks,
                        self._jointflow_weights,
                    )
        #######

        #######
        # 中文注释：per-task loss + 梯度范数日志状态。一任务/优化步 → 在线 grad_norm 都是**单任务**梯度。
        # **梯度记录总开关** `trainer.log_grad_norms`（别名 log_grad_norm_per_head 兼容旧名）。
        #   true  → 记全部 per-task 梯度（grad_norm_total/<task> + shared/<task> + head/<task>）；
        #   false → **一概不算**（连 grad_norm_total 都不取）→ 训练零梯度开销。loss 不受影响、始终记（很便宜）。
        # 需要看梯度时置 true、不需要时 false。默认 false（最快）。
        _gflag = getattr(cfg.trainer, "log_grad_norms", None)
        if _gflag is None:
            _gflag = getattr(cfg.trainer, "log_grad_norm_per_head", False)
        self._log_grad_norms = bool(_gflag)
        self._grad_groups = None          # {shared, policy_head, world_model_head} -> [param,...]（prepare 后构建）
        self._task_head = {}              # task -> 该任务非零梯度的 head 组名
        self._task_logname = {}           # task -> 日志名（idm→inverse）
        self._using_deepspeed = "DEEPSPEED" in str(getattr(accelerator, "distributed_type", "")).upper()
        self._last_clip_norm = None       # 非 DeepSpeed 时 clip_grad_norm_ 的返回（总范数）兜底
        self._grad_warned = False
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

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self._configure_unused_param_anchors()

        #######
        # 中文注释：在 accelerator 包装后，从**未包装**模型构建梯度分组（shared/policy_head/world_model_head）。
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
            except Exception as e:  # 构建失败 → 关闭 per-head 梯度日志，训练照常
                logger.warning("build_task_grad_groups failed (%s); disabling per-head grad logging.", e)
                self._log_grad_norms = False
                self._grad_groups = None
        #######

        self._init_wandb()

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
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

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

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

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

    #######
    # 中文注释：JointFlow 任务采样：未配置 tasks.weights 时返回 None（回落旧 action_loss）。
    # 一任务/优化步——仅在 _jointflow_resample=True（新优化步起点）时由 rank0 采样并 dist.broadcast
    # 任务索引到各 rank，累积窗口内复用同一 task；保证 DDP 各 rank 每步选同一任务、参数 ready 集合一致。
    def _sample_jointflow_task(self) -> str | None:
        if not self._jointflow_tasks:
            return None
        if not self._jointflow_resample and self._jointflow_cur_task is not None:
            return self._jointflow_cur_task
        idx = self._jointflow_rng.choices(range(len(self._jointflow_tasks)), weights=self._jointflow_weights, k=1)[0]
        if dist.is_available() and dist.is_initialized():
            idx_t = torch.tensor([idx], device=self.accelerator.device, dtype=torch.long)
            dist.broadcast(idx_t, src=0)  # 以 rank0 选择为准
            idx = int(idx_t.item())
        self._jointflow_cur_task = self._jointflow_tasks[idx]
        self._jointflow_resample = False
        return self._jointflow_cur_task

    # 中文注释：模型每个 task 返回的 loss key 名 / 日志名（idm→inverse，对齐需求）。
    _TASK_LOSS_KEY = {"policy": "action_loss", "idm": "idm_loss", "fdm": "fdm_loss", "passive": "passive_loss"}
    _TASK_LOGNAME = {"policy": "policy", "idm": "inverse", "fdm": "fdm", "passive": "passive"}

    @classmethod
    def _build_loss_metrics(cls, output_dict: dict, task: str, total_loss: torch.Tensor) -> dict:
        """per-task 原始/加权/total loss（train/ 前缀）。一任务/优化步 → 每步只记当前 task，
        **不为未启用任务写零值**。区分 raw（视觉头原始 MSE，模型返回的 *_loss_raw）与 weighted（× 权重，实际反传）。
        policy/idm 无内部权重 → raw == weighted。"""
        logname = cls._TASK_LOGNAME.get(task, task)
        loss_key = cls._TASK_LOSS_KEY.get(task, f"{task}_loss")
        weighted = output_dict.get(loss_key)
        if weighted is None:  # 兜底：取任一以 _loss 结尾的张量
            for k, v in output_dict.items():
                if torch.is_tensor(v) and k.endswith("_loss"):
                    weighted = v
                    break
        raw = output_dict.get(f"{task}_loss_raw", weighted)
        m = {"train/task": task, "train/loss_total": float(total_loss.detach())}
        m[f"train/loss_{logname}"] = float(weighted.detach())
        m[f"train/loss_{logname}_weighted"] = float(weighted.detach())
        m[f"train/loss_{logname}_raw"] = float(raw.detach())
        return m

    # ---- 梯度范数日志（全部单任务梯度；ZeRO-2 下从 DeepSpeed 取全局范数 + 仅 gather 小 head）----
    def _is_log_step(self, sync_gradients: bool) -> bool:
        """与 _log_metrics 对齐：仅 sync 步、且本步完成后 completed_steps(+1) 命中 logging_frequency 才记。"""
        return bool(sync_gradients) and (self.completed_steps + 1) % self.config.trainer.logging_frequency == 0

    def _global_grad_norm(self):
        """DeepSpeed 全局梯度范数（已 unscale、已跨 rank 规约、已 clip 前）；拿不到回退非-DS 的 clip 返回值。"""
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

    def _compute_head_sqnorms(self, task):
        """当前任务那个 head 的梯度**本地分片**平方和（GPU 标量，**零 collective**）。backward 后、step 前调用。
        一任务/优化步 → 非活跃 head 梯度精确=0（anchor，已验证）→ 只需活跃 head；shared=√(total²−active²) 仍精确。
        返回 (head_logname, local_sq_tensor)；local 平方和由调用方做 **1 次** all_reduce 汇成全局（失同步面=1 次匹配标量规约）。"""
        active = self._task_head.get(task) if task is not None else None
        if active is None or active not in self._grad_groups:
            return None, None
        local_sq = group_grad_local_sqnorm(self._grad_groups[active])   # 零通信；始终返回张量
        logname = self._task_logname.get(task, task)
        return logname, local_sq

    def _finalize_grad_metrics(self, task, head_logname, head_local_sq) -> dict:
        """step 后汇总。**一任务/优化步 → 全部按采样到的任务分曲线**（`.../<task>`），4 任务 1:1:1:1 时
        wandb 每个机制各一条线，可直接叠加对比"每个任务对梯度的影响"。

        全部受总开关 log_grad_norms 控（off 时本函数根本不会被调到，零开销）。on 时：
        - `train/grad_norm_total/<task>`：该任务整步总梯度范数（取自 DeepSpeed 全局，无 collective）；
        - `train/grad_norm_shared/<task>`：该任务对**共享 Qwen 骨干**的梯度（= sqrt(total²−head²)，参数集互不相交 Pythagoras）；
        - `train/grad_norm_head/<task>`：该任务**自己 head** 的梯度（本地分片平方和 + 1 次 all_reduce 汇全局）。"""
        m = {}
        logname = head_logname or (self._task_logname.get(task, task) if task is not None else "action")
        total = self._global_grad_norm()
        if total is not None:
            m[f"train/grad_norm_total/{logname}"] = total
        if head_local_sq is not None:
            # 中文注释：唯一的在线 collective——1 次标量 all_reduce(SUM)，把本地分片平方和汇成全局。各 rank 由 will_log 一致触发 → 匹配、不失同步。
            if self._using_deepspeed and dist.is_available() and dist.is_initialized():
                dist.all_reduce(head_local_sq, op=dist.ReduceOp.SUM)
            head_sq = float(head_local_sq.item())
            m[f"train/grad_norm_head/{logname}"] = head_sq ** 0.5
            if total is not None:
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
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()
            data_elapsed = t_end_data - t_start_data
            model_elapsed = t_end_model - t_start_model
            task_name = str(self._jointflow_cur_task or "action")

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "rank": self.accelerator.process_index,
                        "task": task_name,
                        "data_times": f"{data_elapsed:.3f}",
                        "model_times": f"{model_elapsed:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = data_elapsed
            step_metrics["timing/model"] = model_elapsed
            step_metrics[f"timing/model_{task_name}"] = model_elapsed
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

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

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                #######
                # 中文注释：JointFlow-style 训练复用原生 trainer；配置 tasks.weights 时每步采样一个任务，
                # 模型返回 *_loss / loss_* 指标。没有配置时保持旧 QwenGR00T action_loss 路径。
                jointflow_task = self._sample_jointflow_task()
                if jointflow_task is not None:
                    output_dict = self.model.forward(batch_vla, task=jointflow_task)
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
                    output_dict = self.model.forward(batch_vla)
                    action_loss = output_dict["action_loss"]
                    total_loss = action_loss
                #######

            self.accelerator.backward(total_loss)

            #######
            # 中文注释：梯度统计时序——必须 backward 之后、step/zero_grad 之前。
            # ① 先做 clip（DeepSpeed 下 accelerate clip 实为 no-op/返回 None，真正 clip 在 engine.step 内；
            #    非-DS 时返回总范数，留作 grad_norm_total 兜底）。
            self._last_clip_norm = None
            if self.config.trainer.gradient_clipping is not None:
                clipped = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
                if clipped is not None:
                    try:
                        self._last_clip_norm = float(clipped)
                    except Exception:
                        self._last_clip_norm = None

            sync = bool(self.accelerator.sync_gradients)        # 在 accumulate 块内捕获，块外可能失效
            will_log = self._is_log_step(sync)
            # 中文注释：**梯度记录总开关**——log_grad_norms=false 时 want_grad 恒 False → 既不取 grad_norm_total、
            # 也不算 per-head，训练**零梯度开销**；true 时才在 logging 步算全部 per-task 梯度。
            want_grad = will_log and self._log_grad_norms and self._grad_groups is not None

            # ② step 前：取当前任务 head 的梯度**本地分片**平方和（零 collective）；汇成全局留到 step 后做 1 次 all_reduce。
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

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if sync:
                self.lr_scheduler.step()
                #######
                # 中文注释：优化步完成 → 标记下一步起点重新采样 JointFlow 任务（一任务/优化步，累积窗口内同 task）。
                self._jointflow_resample = True
                #######

            #######
            # 中文注释：step 后（grad_norm_total 此时可从 DeepSpeed 取）汇总梯度指标。仍在 accumulate 块内以保证 engine 状态有效。
            grad_metrics = {}
            if want_grad:  # 受总开关控制：off 时连 grad_norm_total 都不取（零开销）
                try:
                    grad_metrics = self._finalize_grad_metrics(jointflow_task, head_logname, head_local_sq)
                except Exception as e:
                    if not self._grad_warned and self.accelerator.is_main_process:
                        logger.warning("grad-norm finalize failed (%s); skipping.", e)
                    self._grad_warned = True
            #######

        #######
        # 中文注释：只在「即将记录」的步做 .item()/.cpu()（减少 GPU→CPU 同步）；其余步返回 {}。
        if not will_log:
            return {}
        if jointflow_task is not None:
            metrics = self._build_loss_metrics(output_dict, jointflow_task, total_loss)
        else:  # 原生（非多任务）路径
            metrics = {
                "train/task": "action",
                "train/loss_total": float(total_loss.detach()),
                "train/loss_action": float(total_loss.detach()),
                "train/loss_action_raw": float(total_loss.detach()),
                "train/loss_action_weighted": float(total_loss.detach()),
            }
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
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    #######
    # 中文注释：E1.3 correlated noise——若 action_model.use_correlated_noise=true，启动时从 dataloader 估计动作 correlation
    # 的 Cholesky 并注入 action head（启动时算好注入，免离线脚本）。开关关时此段不执行，不影响其它实验。
    action_cfg = getattr(getattr(cfg, "framework", None), "action_model", None)
    if (
        action_cfg is not None
        and bool(action_cfg.get("use_correlated_noise", False))
        and hasattr(vla, "set_action_correlation")
    ):
        from starVLA.dataloader.jointflow.joint_dataset import compute_action_correlation_cholesky
        import numpy as _np

        #######
        # 中文注释：⚠️多机必须只在 rank0 估一次。估计要逐样本 __getitem__（解码视频 + 读 latent），
        # 64 卡各扫 20000 条 = 把 bucket I/O 打爆、启动卡死 30min+。改成：rank0 算(num_samples 可调，默认 4096)
        # → 落盘 → barrier → 所有 rank 从盘读。已有缓存直接加载，免重算（resume/重跑秒过）。
        # 落盘文件供评测 server 读，保证推理初始噪声分布与训练一致（buffer persistent=False 不进 ckpt）。
        cache_path = os.path.join(output_dir, "action_correlation_cholesky.npy")
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
            _source_chol = _np.load(_source_path, allow_pickle=False)
            if tuple(_source_chol.shape) != (_exp, _exp):
                raise ValueError(
                    f"correlation_cholesky_path has shape {_source_chol.shape}, expected ({_exp}, {_exp}): "
                    f"{_source_path}"
                )
            if not _np.isfinite(_source_chol).all():
                raise ValueError(f"correlation_cholesky_path contains non-finite values: {_source_path}")
        if accelerator.is_main_process:
            #######
            # 中文注释：严格消融可通过 correlation_cholesky_path 固定使用已完成 baseline 的矩阵。
            # 指定后必须存在、维度正确且数值有限；失败时直接退出，禁止静默重算引入额外变量。
            if _source_path:
                _np.save(cache_path, _np.asarray(_source_chol))
                logger.info(
                    "correlated-noise: copied fixed baseline Cholesky(%sx%s) from %s -> %s",
                    _exp,
                    _exp,
                    _source_path,
                    cache_path,
                )
            else:
                # 中文注释：未固定外部矩阵时，缓存复用——文件存在且维度正确就直接使用；
                # 维度不符(改了 horizon/action_dim)则重算，防止用到过期 Cholesky。
                _reuse = os.path.exists(cache_path) and tuple(
                    _np.load(cache_path, allow_pickle=False).shape
                ) == (_exp, _exp)
                if _reuse:
                    logger.info(f"correlated-noise: 复用已缓存 Cholesky({_exp}x{_exp}) → {cache_path}")
                else:
                    chol = compute_action_correlation_cholesky(
                        vla_train_dataloader.dataset,
                        num_samples=int(action_cfg.get("correlation_num_samples", 4096)),
                        beta=float(action_cfg.get("correlation_beta", 0.5)),
                        matrix_type=str(action_cfg.get("correlation_matrix_type", "covariance")),
                    )
                    _np.save(cache_path, _np.asarray(chol))
                    logger.info(
                        "correlated-noise Cholesky ready: shape=%s, beta=%s, matrix_type=%s -> %s",
                        chol.shape,
                        action_cfg.get("correlation_beta", 0.5),
                        action_cfg.get("correlation_matrix_type", "covariance"),
                        cache_path,
                    )
        if dist.is_initialized():
            dist.barrier()  # 等 rank0 算好/存好,其余 rank 再读
        _loaded_chol = _np.load(cache_path, allow_pickle=False)
        if tuple(_loaded_chol.shape) != (_exp, _exp) or not _np.isfinite(_loaded_chol).all():
            raise ValueError(f"Invalid correlated-noise Cholesky after synchronization: {cache_path}")
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
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
