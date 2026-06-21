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
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

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

        self._init_wandb()

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

    # 中文注释：把模型返回的 tensor 指标转成可记录标量，同时保留用户要求的关键 loss 名称。
    @staticmethod
    def _collect_jointflow_metrics(output_dict: dict, task: str, total_loss: torch.Tensor) -> dict:
        metrics = {"task": task, "loss": float(total_loss.detach().cpu())}
        alias = {
            "policy_loss": "policy_loss",
            "fdm_loss": "loss_fdm",
            "idm_loss": "idm_loss",
            "passive_loss": "passive_loss",
            "loss_fdm": "loss_fdm",
        }
        for key, value in output_dict.items():
            if torch.is_tensor(value):
                clean_key = key.replace("/", "_")
                metrics[clean_key] = float(value.detach().cpu())
                if key in alias:
                    metrics[alias[key]] = metrics[clean_key]
        if task == "policy" and "policy_loss" not in metrics:
            metrics["policy_loss"] = metrics.get("loss", float(total_loss.detach().cpu()))
        return metrics
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

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
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

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
                #######
                # 中文注释：优化步完成 → 标记下一步起点重新采样 JointFlow 任务（一任务/优化步，累积窗口内同 task）。
                self._jointflow_resample = True
                #######

        #######
        # 中文注释：JointFlow 分支记录 policy/fdm/idm/passive 关键 loss；旧分支保持 action_dit_loss。
        if jointflow_task is not None:
            return self._collect_jointflow_metrics(output_dict, jointflow_task, total_loss)
        return {
            "action_dit_loss": action_loss.item(),
        }
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
    # 中文注释：E1.3 correlated noise——若 action_model.use_correlated_noise=true，启动时从 dataloader 估计动作协方差
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
        if accelerator.is_main_process:
            #######
            # 中文注释：缓存复用——文件存在且维度=action_horizon×action_dim 就直接用,免重扫(秒过)；
            # 维度不符(改了 horizon/action_dim/数据)则重算,防止用到过期 Cholesky。
            _exp = int(action_cfg.get("action_horizon", 0)) * int(action_cfg.get("action_dim", 0))
            _reuse = (
                os.path.exists(cache_path)
                and _exp > 0
                and tuple(_np.load(cache_path).shape) == (_exp, _exp)
            )
            #######
            if _reuse:
                logger.info(f"correlated-noise: 复用已缓存 Cholesky({_exp}x{_exp}) → {cache_path}")
            else:
                chol = compute_action_correlation_cholesky(
                    vla_train_dataloader.dataset,
                    num_samples=int(action_cfg.get("correlation_num_samples", 4096)),
                    beta=float(action_cfg.get("correlation_beta", 0.5)),
                )
                _np.save(cache_path, _np.asarray(chol))
                logger.info(f"correlated-noise Cholesky ready: shape={chol.shape}, beta={action_cfg.get('correlation_beta', 0.5)} → {cache_path}")
        if dist.is_initialized():
            dist.barrier()  # 等 rank0 算好/存好,其余 rank 再读
        vla.set_action_correlation(_np.load(cache_path))
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
