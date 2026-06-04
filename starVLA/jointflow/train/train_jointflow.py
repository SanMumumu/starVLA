"""PR7: JointFlow multi-task training loop.

复用:
- starVLA.training.train_starvla 的 TrainerUtils / optimizer helpers
- starVLA.model.framework.base_framework.build_framework

说明:
本训练入口只服务 QwenJointFlow，显式 import 新 framework 触发 registry，
不修改 starVLA/training/train_starvla.py。
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm

from starVLA.jointflow.data.joint_dataset import build_joint_dataloader
import starVLA.jointflow.framework.qwen_joint_flow  # noqa: F401 - 注册 QwenJointFlow
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    normalize_dotlist_args,
)
from transformers import get_scheduler


logger = get_logger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _get_jointflow_mixed_precision() -> str:
    mixed_precision = os.environ.get("JOINTFLOW_MIXED_PRECISION", "bf16").strip().lower()
    aliases = {
        "0": "no",
        "false": "no",
        "none": "no",
        "off": "no",
        "disable": "no",
        "disabled": "no",
    }
    mixed_precision = aliases.get(mixed_precision, mixed_precision)
    valid = {"no", "fp16", "bf16", "fp8"}
    if mixed_precision not in valid:
        raise ValueError(
            f"Invalid JOINTFLOW_MIXED_PRECISION={mixed_precision!r}; "
            f"expected one of {sorted(valid)}."
        )
    return mixed_precision


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean env {name}={value!r}")


def _build_accelerator() -> Accelerator:
    mixed_precision = _get_jointflow_mixed_precision()
    kwargs_handlers = []
    if _env_flag("JOINTFLOW_FIND_UNUSED_PARAMETERS", True):
        kwargs_handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))
    override = os.environ.get("JOINTFLOW_USE_DEEPSPEED", "").strip().lower()
    if override in {"0", "false", "no"}:
        return Accelerator(mixed_precision=mixed_precision, kwargs_handlers=kwargs_handlers)
    use_deepspeed = override in {"1", "true", "yes"} or os.environ.get("ACCELERATE_USE_DEEPSPEED", "").lower() == "true"
    if use_deepspeed:
        return Accelerator(
            mixed_precision=mixed_precision,
            deepspeed_plugin=DeepSpeedPlugin(),
            kwargs_handlers=kwargs_handlers,
        )
    return Accelerator(mixed_precision=mixed_precision, kwargs_handlers=kwargs_handlers)


accelerator = _build_accelerator()


######### // code // ##########
# 中文注释：按 cfg.framework.tasks.weights 每个 step 采样一个 task。
# 输入权重 dict；输出 task 字符串 policy/fdm/idm/passive。
class JointTaskSampler:
    def __init__(self, weights: dict, seed: int = 42):
        self.tasks = [str(k) for k, v in weights.items() if float(v) > 0]
        self.weights = [float(weights[k]) for k in self.tasks]
        self.task_to_id = {task: idx for idx, task in enumerate(self.tasks)}
        self.rng = random.Random(seed)
        if not self.tasks:
            raise ValueError("At least one JointFlow task weight must be > 0.")

    def sample(self) -> str:
        return self.rng.choices(self.tasks, weights=self.weights, k=1)[0]
######### // code // ##########


######### // code // ##########
# 中文注释：目录、优化器和 scheduler 构造，沿用 starVLA train_starvla 的参数分组方式。
def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
    return output_dir


def setup_optimizer_and_scheduler(model, cfg):
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=torch.cuda.is_available(),
    )
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )
    return optimizer, lr_scheduler
######### // code // ##########


def print_trainable_parameters_safe(model) -> tuple[int, int] | None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return None
    print("model parameter statistics:")
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, "
        f"{num_trainable_params / 10**6:.3f} Trainable"
    )
    return num_params, num_trainable_params
######### // code // ##########


def print_runtime_device_info_safe(accelerator_obj: Accelerator) -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    print("runtime device info:")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"  JOINTFLOW_MIXED_PRECISION={os.environ.get('JOINTFLOW_MIXED_PRECISION', '<unset; default bf16>')}")
    print(
        "  JOINTFLOW_FIND_UNUSED_PARAMETERS="
        f"{os.environ.get('JOINTFLOW_FIND_UNUSED_PARAMETERS', '<unset; default true>')}"
    )
    print(f"  torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count={torch.cuda.device_count()}")
    print(f"  accelerator.device={accelerator_obj.device}")
    print(f"  accelerator.mixed_precision={accelerator_obj.mixed_precision}")
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            print(f"  logical cuda:{idx} name={torch.cuda.get_device_name(idx)}")
######### // code // ##########


def _count_module_parameters(module) -> tuple[int, int]:
    if module is None:
        return 0, 0
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def build_component_parameter_table(model) -> list[dict]:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        llm_note = "Qwen text backbone"
    else:
        llm_note = (
            f"Qwen text backbone; model_id={getattr(backbone, 'model_id', 'unknown')}; "
            "loaded from AutoModelForCausalLM"
        )
    components = [
        ("llm_text_encoder", backbone, llm_note),
        ("dino_encoder", getattr(model, "dino", None), "Live DINOv3; zero means training reads offline latents"),
        ("dino_projector", getattr(model, "dino_proj", None), "DINO 384-dim tokens to Qwen hidden size"),
        ("state_encoder", getattr(model, "state_enc", None), "Current proprio state to state token"),
        ("action_context_encoder", getattr(model, "act_ctx", None), "Clean action chunk to context tokens for FDM"),
        ("query_tokens", getattr(model, "queries", None), "Learned action/image query tokens"),
        ("action_diffusion_head", getattr(model, "action_head", None), "Flow head for policy and IDM actions"),
        ("visual_dino_diffusion_head", getattr(model, "visual_head", None), "Flow head for FDM/passive DINO features"),
    ]
    rows = []
    for name, module, note in components:
        total, trainable = _count_module_parameters(module)
        if module is None:
            status = "not_loaded"
        elif total == 0:
            status = "empty"
        elif trainable == 0:
            status = "frozen"
        elif trainable == total:
            status = "trainable"
        else:
            status = "partially_trainable"
        rows.append(
            {
                "name": name,
                "total": total,
                "trainable": trainable,
                "status": status,
                "note": note,
            }
        )
    return rows


def print_component_parameter_table_safe(model) -> list[dict] | None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return None
    rows = build_component_parameter_table(model)
    print("component parameter table:")
    print(f"{'component':28s} {'total(M)':>12s} {'trainable(M)':>14s} {'status':>20s}  note")
    for row in rows:
        print(
            f"{row['name']:28s} "
            f"{row['total'] / 1e6:12.3f} "
            f"{row['trainable'] / 1e6:14.3f} "
            f"{row['status']:>20s}  "
            f"{row['note']}"
        )
    return rows


def write_parameter_table_to_wandb_summary(rows: list[dict] | None) -> None:
    if not rows:
        return
    for row in rows:
        prefix = f"params/{row['name']}"
        wandb.run.summary[f"{prefix}/total"] = row["total"]
        wandb.run.summary[f"{prefix}/trainable"] = row["trainable"]
        wandb.run.summary[f"{prefix}/status"] = row["status"]
######### // code // ##########


def validate_precision_compatibility_safe(accelerator_obj: Accelerator, model) -> None:
    if accelerator_obj.mixed_precision != "fp16":
        return
    first_bf16_param = None
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype == torch.bfloat16:
            first_bf16_param = name
            break
    if first_bf16_param is None:
        return
    raise RuntimeError(
        "JointFlow is running with accelerator.mixed_precision=fp16, but at least one "
        f"trainable parameter is bfloat16 ({first_bf16_param}). This combination makes "
        "PyTorch GradScaler fail during gradient clipping with "
        "`_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16`. "
        "Launch with bf16 instead, for example: "
        "`JOINTFLOW_MIXED_PRECISION=bf16 bash starVLA/jointflow/scripts/run_jointflow_train.sh "
        "starVLA/jointflow/configs/jointflow_libero.yaml 4,7`."
    )
######### // code // ##########


######### // code // ##########
# 中文注释：JointFlow trainer。每步从一个 batch 派生单一 task loss。
class JointFlowTrainer(TrainerUtils):
    def __init__(self, cfg, model, dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.completed_steps = 0
        self.sampler = JointTaskSampler(cfg.framework.tasks.weights, seed=int(getattr(cfg, "seed", 42)))
        self.task_loss_ema = {task: None for task in self.sampler.tasks}
        self.task_counts = {task: 0 for task in self.sampler.tasks}
        self.task_ema_beta = float(getattr(cfg.trainer, "task_loss_ema_beta", 0.98))
        self.parameter_rows = None

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        set_seed(int(getattr(self.config, "seed", 42)) + rank)
        self._save_initial_configs()
        print_runtime_device_info_safe(self.accelerator)
        freeze_modules = getattr(self.config.trainer, "freeze_modules", None)
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        validate_precision_compatibility_safe(self.accelerator, self.model)
        print_trainable_parameters_safe(self.model)
        self.parameter_rows = print_component_parameter_table_safe(self.model)
        self.model, self.optimizer, self.dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.dataloader,
        )
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="jointflow-train",
            )
            write_parameter_table_to_wandb_summary(self.parameter_rows)

    def _save_initial_configs(self):
        if not self.accelerator.is_main_process:
            return
        output_dir = Path(self.config.output_dir)
        full_cfg = self.config.unwrap() if isinstance(self.config, AccessTrackedConfig) else self.config
        OmegaConf.save(full_cfg, output_dir / "config.full.yaml", resolve=True)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)

    def _get_next_batch(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            return next(self.data_iter)

    def _sample_task_for_all_ranks(self) -> str:
        if not dist.is_initialized():
            return self.sampler.sample()
        if dist.get_rank() == 0:
            task = self.sampler.sample()
            task_id = self.sampler.task_to_id[task]
        else:
            task_id = 0
        task_tensor = torch.tensor([task_id], device=self.accelerator.device, dtype=torch.long)
        dist.broadcast(task_tensor, src=0)
        return self.sampler.tasks[int(task_tensor.item())]

    def _train_step(self, batch):
        task = self._sample_task_for_all_ranks()
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            output = self.model(batch, task=task)
            loss = output[f"{task}_loss"]
            self.accelerator.backward(loss)
            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
            self.optimizer.step()
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
        loss_value = float(loss.detach().cpu())
        self.task_counts[task] += 1
        old_ema = self.task_loss_ema[task]
        self.task_loss_ema[task] = loss_value if old_ema is None else self.task_ema_beta * old_ema + (1.0 - self.task_ema_beta) * loss_value

        metrics = {
            "loss/current": loss_value,
            f"loss/{task}": loss_value,
            "task": task,
            "task_id": self.sampler.task_to_id[task],
        }
        total_task_samples = max(sum(self.task_counts.values()), 1)
        for name in self.sampler.tasks:
            metrics[f"task_sampled/{name}"] = 1.0 if name == task else 0.0
            metrics[f"task_count/{name}"] = self.task_counts[name]
            metrics[f"task_fraction/{name}"] = self.task_counts[name] / total_task_samples
            if self.task_loss_ema[name] is not None:
                metrics[f"loss_ema/{name}"] = self.task_loss_ema[name]
        return metrics

    def _log_metrics(self, metrics):
        if not self.accelerator.is_main_process:
            return
        if self.completed_steps % self.config.trainer.logging_frequency != 0:
            return
        last_lrs = self.lr_scheduler.get_last_lr()
        for i, group in enumerate(self.optimizer.param_groups):
            metrics[f"learning_rate/{group.get('name', str(i))}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
        wandb.log(metrics, step=self.completed_steps)
        logger.info(f"step={self.completed_steps} metrics={metrics}")

    def _save_checkpoint(self):
        if not self.accelerator.is_main_process:
            return
        ckpt = Path(self.config.output_dir) / "checkpoints" / f"steps_{self.completed_steps}_pytorch_model.pt"
        state_dict = self.accelerator.get_state_dict(self.model)
        torch.save(state_dict, ckpt)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(Path(self.config.output_dir) / "config.yaml", use_original_values=False)
        logger.info(f"checkpoint saved: {ckpt}")

    def train(self):
        self.data_iter = iter(self.dataloader)
        progress = tqdm(
            total=self.config.trainer.max_train_steps,
            disable=not self.accelerator.is_local_main_process,
        )
        while self.completed_steps < self.config.trainer.max_train_steps:
            t0 = time.perf_counter()
            batch = self._get_next_batch()
            t1 = time.perf_counter()
            metrics = self._train_step(batch)
            t2 = time.perf_counter()

            if self.accelerator.sync_gradients:
                self.completed_steps += 1
                progress.update(1)
            metrics["timing/data"] = t1 - t0
            metrics["timing/model"] = t2 - t1
            self._log_metrics(metrics)

            if self.completed_steps > 0 and self.completed_steps % self.config.trainer.save_interval == 0:
                self._save_checkpoint()
        progress.close()
        if self.accelerator.is_main_process:
            wandb.finish()
        self.accelerator.wait_for_everyone()
######### // code // ##########


def main(cfg):
    cfg.framework.name = "QwenJointFlow"
    cfg = wrap_config(cfg)
    setup_directories(cfg)
    model = build_framework(cfg)
    dataloader = build_joint_dataloader(cfg)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, cfg)
    trainer = JointFlowTrainer(cfg, model, dataloader, optimizer, lr_scheduler, accelerator)
    trainer.prepare_training()
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/jointflow/configs/jointflow_libero.yaml")
    args, clipargs = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)
    cli_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(clipargs))
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    main(cfg)
