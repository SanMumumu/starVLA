"""PR3/PR6: TriFlow 多任务训练入口.

复用:
- starVLA.jointflow.data.joint_dataset.build_joint_dataloader (数据管线整体复用)
- starVLA.training.trainer_utils.trainer_tools (TrainerUtils / build_param_lr_groups / normalize_dotlist_args)
- starVLA.model.framework.base_framework.build_framework

结构照抄 jointflow v1 train_jointflow.py 的硬化模式（出处注明，不 import 其 trainer 类，
因为 v1 模块顶层会拉起 Qwen/transformers 栈且环境变量前缀/广播内容不同）:
- 每步一个 (task, decoder_flag)，rank0 用 int64[2] 一次性广播 → 全 rank 一致
- 集合通信版 loss 有限性检查 (all-reduce MIN，一起抛错不挂死)
- 非有限梯度跳步 (不污染 AdamW 动量；超过上限再崩)
- accelerate save_state/load_state full-state resume + scheduler/EMA 注册
- TriFlow 新增: EMA (只在真正 optimizer.step 的 sync 步更新)、run_dir 资产落盘
  (config/dataset_statistics/dino_v3_stats/vocab_map —— eval 链路 v1 血泪教训)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

import starVLA.triflow.framework.tri_flow  # noqa: F401 - 注册 TriFlow framework
from starVLA.jointflow.data.joint_dataset import build_joint_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    normalize_dotlist_args,
)
from starVLA.triflow.modules.ema import EmaModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

######### // code // ##########
# 中文注释：集群日志面板不支持 ANSI 彩色，统一关掉（照抄 v1，setdefault 不覆盖用户显式设置）。
for _color_key, _color_val in {
    "NO_COLOR": "1",
    "PY_COLORS": "0",
    "CLICOLOR": "0",
    "FORCE_COLOR": "0",
    "RICH_NO_COLOR": "1",
}.items():
    os.environ.setdefault(_color_key, _color_val)
######### // code // ##########

_RANK_LOG_HANDLES = []


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean env {name}={value!r}")


def _get_triflow_mixed_precision() -> str:
    # 中文注释：TriFlow 默认全程 fp32（88M 小模型；v1 bf16 NaN 血泪教训）。TF32 提供加速。
    mixed_precision = os.environ.get("TRIFLOW_MIXED_PRECISION", "no").strip().lower()
    aliases = {"0": "no", "false": "no", "none": "no", "off": "no", "disable": "no", "disabled": "no"}
    mixed_precision = aliases.get(mixed_precision, mixed_precision)
    valid = {"no", "fp16", "bf16", "fp8"}
    if mixed_precision not in valid:
        raise ValueError(f"Invalid TRIFLOW_MIXED_PRECISION={mixed_precision!r}; expected one of {sorted(valid)}.")
    return mixed_precision


def _maybe_enable_tf32() -> None:
    if not _env_flag("TRIFLOW_TF32", True):
        return
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _format_float(value, precision: int = 4) -> str:
    if value is None:
        return "na"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.{precision}g}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "na"
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"{days}d{hours:02d}h"
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _env_tqdm_enabled() -> bool:
    value = os.environ.get("TRIFLOW_TQDM", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    if value != "auto":
        raise ValueError("Invalid TRIFLOW_TQDM; expected auto/true/false.")
    return sys.stdout.isatty()


def _build_accelerator(gradient_accumulation_steps: int = 1) -> Accelerator:
    kwargs_handlers = []
    if _env_flag("TRIFLOW_FIND_UNUSED_PARAMETERS", False):
        kwargs_handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))
    return Accelerator(
        mixed_precision=_get_triflow_mixed_precision(),
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=kwargs_handlers,
    )


######### // code // ##########
# 中文注释：每步采样一个 (task, decoder_flag)。task 按权重；decoder_flag 只在
# 该任务把 L 作为目标时按 flow.decoder_prob 抽取（ELF 双分支）。
class TriTaskSampler:
    def __init__(self, weights: dict, task_registry: dict, decoder_prob: float, seed: int = 42):
        self.tasks = [str(k) for k, v in weights.items() if float(v) > 0]
        if not self.tasks:
            raise ValueError("At least one TriFlow task weight must be > 0.")
        unknown = [t for t in self.tasks if t not in task_registry]
        if unknown:
            raise ValueError(f"tasks.weights contains unknown tasks {unknown}; known: {sorted(task_registry)}")
        self.weights = [float(weights[k]) for k in self.tasks]
        self.task_to_id = {task: idx for idx, task in enumerate(self.tasks)}
        self.registry = task_registry
        self.decoder_prob = float(decoder_prob)
        self.rng = random.Random(seed)

    def sample(self) -> tuple[str, bool]:
        task = self.rng.choices(self.tasks, weights=self.weights, k=1)[0]
        decoder = False
        if "l" in self.registry[task].target and self.rng.random() < self.decoder_prob:
            decoder = True
        return task, decoder
######### // code // ##########


def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
    return output_dir


def setup_rank_output_control(cfg) -> None:
    # 中文注释：非 0 rank 输出重定向到文件/静默（照抄 v1）。
    mode = os.environ.get("TRIFLOW_RANK_LOG_MODE", "files").strip().lower()
    if mode in {"", "all", "off", "none", "false", "0"}:
        return
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if rank == 0:
        if mode == "files":
            print(f"[triflow] non-zero rank stdout/stderr -> {Path(cfg.output_dir) / 'rank_logs'}/rank*_local*.log", flush=True)
        return
    if mode == "quiet":
        path = os.devnull
    elif mode == "files":
        log_dir = Path(cfg.output_dir) / "rank_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = str(log_dir / f"rank{rank}_local{local_rank}.log")
    else:
        raise ValueError(f"Invalid TRIFLOW_RANK_LOG_MODE={mode!r}")
    sys.stdout.flush()
    sys.stderr.flush()
    handle = open(path, "a", buffering=1)
    _RANK_LOG_HANDLES.append(handle)
    os.dup2(handle.fileno(), 1)
    os.dup2(handle.fileno(), 2)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)


def setup_optimizer_and_scheduler(model, cfg):
    # 中文注释：from-scratch 单参数组（learning_rate 只配 base）；AdamW wd=0 配合
    # embedding 行 RMS 归一（wd 会和归一化打架）。
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


def print_trainable_parameters_safe(model) -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# Parameters (M): {num_params / 1e6:.3f} total, {num_trainable / 1e6:.3f} trainable", flush=True)


def print_component_table_safe(model) -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    rows = [
        ("tower", model.tower, "from-scratch 单塔 transformer"),
        ("embedder", model.embedder, "E 表 + in/out 投影 + pos/type/time/stride"),
        ("dino_live", model.dino, "live DINOv3 (训练为 None, 读离线 latents)"),
    ]
    print("component parameter table:", flush=True)
    for name, module, note in rows:
        if module is None:
            print(f"  {name:12s} {'-':>10s}  not_loaded  {note}", flush=True)
            continue
        total = sum(p.numel() for p in module.parameters())
        print(f"  {name:12s} {total / 1e6:9.2f}M  {note}", flush=True)


######### // code // ##########
# 中文注释：TriFlow trainer（结构照抄 v1 JointFlowTrainer，差异见文件头）。
class TriFlowTrainer(TrainerUtils):
    def __init__(self, cfg, model, dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.completed_steps = 0
        raw_model = model
        self.sampler = TriTaskSampler(
            cfg.framework.tasks.weights,
            task_registry=raw_model.task_registry,
            decoder_prob=float(cfg.framework.flow.get("decoder_prob", 0.2)),
            seed=int(getattr(cfg, "seed", 42)),
        )
        self.task_loss_ema = {task: None for task in self.sampler.tasks}
        self.task_counts = {task: 0 for task in self.sampler.tasks}
        self.task_ema_beta = float(getattr(cfg.trainer, "task_loss_ema_beta", 0.98))
        self.micro_steps_seen = 0
        self.debug_first_micro_steps = int(os.environ.get("TRIFLOW_DEBUG_FIRST_MICRO_STEPS", "8"))
        self.train_start_time = None
        self.last_optimizer_step_time = None
        self.optimizer_step_time_ema = None
        self.step_time_ema_beta = float(os.environ.get("TRIFLOW_STEP_TIME_EMA_BETA", "0.95"))
        self.nonfinite_grad_skips = 0
        self.ema: EmaModel | None = None

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        set_seed(int(getattr(self.config, "seed", 42)) + rank)
        self._save_initial_configs()
        print_trainable_parameters_safe(self.model)
        print_component_table_safe(self.model)
        self.model, self.optimizer, self.dataloader = self.setup_distributed_training(
            self.accelerator, self.model, self.optimizer, self.dataloader
        )
        # 中文注释：scheduler/EMA 都没进 accelerator.prepare()，显式注册才会被
        # save_state/load_state 一并存取（必须在 _maybe_resume 之前注册；v1 同款）。
        self.accelerator.register_for_checkpointing(self.lr_scheduler)
        if bool(self.config.framework.ema.get("enabled", True)):
            unwrapped = self.accelerator.unwrap_model(self.model)
            self.ema = EmaModel(unwrapped, decay=float(self.config.framework.ema.get("decay", 0.9999)))
            self.accelerator.register_for_checkpointing(self.ema)
        self._maybe_resume()
        if self.accelerator.is_main_process:
            wandb_kwargs = {
                "name": self.config.run_id,
                "dir": os.path.join(self.config.output_dir, "wandb"),
                "project": self.config.wandb_project,
                "entity": self.config.wandb_entity,
                "group": "triflow-train",
                "mode": os.environ.get("WANDB_MODE", os.environ.get("TRIFLOW_WANDB_MODE", "online")),
            }
            wandb_run_id = str(getattr(self.config, "wandb_run_id", "") or "").strip()
            if wandb_run_id:
                wandb_kwargs["id"] = wandb_run_id
                if self.completed_steps > 0:
                    wandb_kwargs["resume"] = "allow"
            wandb.init(**wandb_kwargs)

    ######### // code // ##########
    # 中文注释：run_dir 资产落盘（v1 教训：eval 的 read_mode_config 需要完整 config.yaml；
    # server 需要 dataset_statistics.json / dino_v3_stats.json / vocab_map.json）。
    # 训练启动即落齐，eval 不再依赖事后 prepare 脚本。
    def _save_initial_configs(self):
        if not self.accelerator.is_main_process:
            return
        output_dir = Path(self.config.output_dir)
        full_cfg = self.config.unwrap() if isinstance(self.config, AccessTrackedConfig) else self.config
        OmegaConf.save(full_cfg, output_dir / "config.full.yaml", resolve=True)
        OmegaConf.save(full_cfg, output_dir / "config.yaml", resolve=True)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.accessed.yaml", use_original_values=False)

        raw_model = self.model  # prepare 之前调用，model 尚未被 DDP 包裹
        # 1) dataset_statistics.json（action 反归一化真源）
        try:
            dataset = self.dataloader.dataset
            if hasattr(dataset, "save_dataset_statistics"):
                dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
            else:
                from starVLA.dataloader import save_dataset_statistics

                save_dataset_statistics(dataset.dataset_statistics, output_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[triflow][warn] failed to save dataset_statistics.json: {exc}", flush=True)
        # 2) 合并后的 DINO stats（live eval 归一化真源）
        try:
            paths = raw_model._default_dino_stats_paths()
            stats = raw_model._combine_dino_stats([raw_model._read_dino_stats_file(p) for p in paths])
            if stats is not None:
                with open(output_dir / "dino_v3_stats.json", "w", encoding="utf-8") as f:
                    json.dump(stats, f)
        except Exception as exc:  # noqa: BLE001
            print(f"[triflow][warn] failed to save dino_v3_stats.json: {exc}", flush=True)
        # 3) vocab_map.json（文本编解码真源；compact 模式必需）
        try:
            vocab_path = getattr(raw_model.codec, "vocab_map_path", None)
            if vocab_path and Path(vocab_path).exists():
                shutil.copyfile(vocab_path, output_dir / "vocab_map.json")
        except Exception as exc:  # noqa: BLE001
            print(f"[triflow][warn] failed to copy vocab_map.json: {exc}", flush=True)
    ######### // code // ##########

    def _get_next_batch(self):
        # 中文注释：过拟合验收门（PR3-PR5）：TRIFLOW_OVERFIT_ONE_BATCH=1 时
        # 固定返回第一个 batch，验证单 batch loss→~0 与 predict 还原。
        if _env_flag("TRIFLOW_OVERFIT_ONE_BATCH", False):
            cached = getattr(self, "_overfit_batch", None)
            if cached is not None:
                return cached
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        if _env_flag("TRIFLOW_OVERFIT_ONE_BATCH", False):
            self._overfit_batch = batch
        return batch

    ######### // code // ##########
    # 中文注释：(task, decoder_flag) 一次性 int64[2] 广播（v1 task 广播的扩展）。
    # 全 rank 必须同 task 同分支，否则参与 backward 的参数集合不同 → DDP all-reduce 挂死。
    def _sample_task_for_all_ranks(self) -> tuple[str, bool]:
        task, decoder = self.sampler.sample()
        if dist.is_initialized() and _env_flag("TRIFLOW_SYNC_TASK_BROADCAST", True):
            if dist.get_rank() == 0:
                payload = torch.tensor([self.sampler.task_to_id[task], int(decoder)], device=self.accelerator.device, dtype=torch.long)
            else:
                payload = torch.zeros(2, device=self.accelerator.device, dtype=torch.long)
            dist.broadcast(payload, src=0)
            task = self.sampler.tasks[int(payload[0].item())]
            decoder = bool(payload[1].item())
        return task, decoder
    ######### // code // ##########

    def _raise_if_loss_nonfinite(self, loss: torch.Tensor, task: str, batch) -> None:
        # 中文注释：集合通信版有限性检查（v1 同款）：先各自判定再 all-reduce(MIN)，
        # 任一 rank NaN/Inf → 所有 rank 一起抛错，干净失败不挂死。
        local_finite = bool(torch.isfinite(loss.detach()).all())
        all_finite = local_finite
        if dist.is_initialized():
            flag = torch.tensor([1.0 if local_finite else 0.0], device=self.accelerator.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            all_finite = flag.item() > 0.5
        if all_finite:
            return
        samples = [
            {key: ex.get(key, None) for key in ("dataset_name", "trajectory_id", "base_index", "future_index", "lang") if key in ex}
            for ex in batch[:4]
        ]
        raise FloatingPointError(
            f"Non-finite loss before backward: task={task}, local_finite={local_finite}, "
            f"completed_steps={self.completed_steps}, samples={samples}, loss={loss.detach()}"
        )

    def _train_step(self, batch):
        task, decoder = self._sample_task_for_all_ranks()
        grad_norm_value = None
        nonfinite_grad = False
        sub_metrics = {}
        with self.accelerator.accumulate(self.model):
            output = self.model(batch, task=task, decoder_step=decoder)
            loss = output[f"{task}_loss"]
            sub_metrics = {k: v for k, v in output.items() if not torch.is_tensor(v)}
            self._raise_if_loss_nonfinite(loss, task, batch)
            self.accelerator.backward(loss)
            # 中文注释：PR6 测试钩子——TRIFLOW_INJECT_NONFINITE_AT_STEP=N 时在第 N 步
            # 向首个参数梯度注入 inf，验证"跳步不崩、计数、继续训练"。生产不设此 env 即无效。
            inject_at = os.environ.get("TRIFLOW_INJECT_NONFINITE_AT_STEP", "").strip()
            if inject_at and self.accelerator.sync_gradients and self.completed_steps == int(inject_at):
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.view(-1)[0] = float("inf")
                            break
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
                if grad_norm is not None:
                    grad_norm_tensor = grad_norm.detach() if torch.is_tensor(grad_norm) else torch.tensor(float(grad_norm), device=self.accelerator.device)
                    grad_norm_value = float(grad_norm_tensor.float().cpu())
                    nonfinite_grad = not bool(torch.isfinite(grad_norm_tensor).all())
            ######### // code // ##########
            # 中文注释：非有限梯度跳步（v1 同款）：NaN 梯度一旦进 AdamW 动量就永久污染，
            # 必须在 optimizer.step 前拦截 → 丢弃本步梯度继续训练。DDP 下 clip 后的
            # total_norm 各 rank 一致 → 判定一致 → 一起跳，无分叉。
            if nonfinite_grad:
                self.nonfinite_grad_skips += 1
                if self.accelerator.is_main_process:
                    print(
                        f"[triflow][warn] non-finite grad -> skip optimizer.step: task={task} dec={int(decoder)} "
                        f"completed_steps={self.completed_steps} grad_norm={grad_norm_value} total_skips={self.nonfinite_grad_skips}",
                        flush=True,
                    )
                if os.environ.get("TRIFLOW_NONFINITE_GRAD", "skip").strip().lower() == "raise":
                    raise FloatingPointError(
                        f"Non-finite grad norm before optimizer.step: task={task}, completed_steps={self.completed_steps}"
                    )
                self.optimizer.zero_grad(set_to_none=True)
                max_skips = int(os.environ.get("TRIFLOW_MAX_NONFINITE_SKIPS", "100"))
                if max_skips >= 0 and self.nonfinite_grad_skips > max_skips:
                    raise FloatingPointError(
                        f"Too many non-finite grad steps ({self.nonfinite_grad_skips} > {max_skips}); aborting."
                    )
            else:
                self.optimizer.step()
                if self.accelerator.sync_gradients:
                    self.lr_scheduler.step()
                    # 中文注释：EMA 只在“真正成功的 optimizer.step + sync”更新；
                    # 跳步/累积步不更新（否则有效 decay 变 decay^accum，ELF 同款注意事项）。
                    if self.ema is not None:
                        self.ema.update(self.accelerator.unwrap_model(self.model))
                self.optimizer.zero_grad()
            ######### // code // ##########
        loss_value = float(loss.detach().cpu())
        self.task_counts[task] += 1
        old_ema = self.task_loss_ema[task]
        self.task_loss_ema[task] = loss_value if old_ema is None else self.task_ema_beta * old_ema + (1.0 - self.task_ema_beta) * loss_value

        metrics = {
            "loss/current": loss_value,
            f"loss/{task}{'_dec' if decoder else ''}": loss_value,
            "task": task,
            "task_id": self.sampler.task_to_id[task],
            "decoder_step": int(decoder),
        }
        for key, val in sub_metrics.items():
            metrics[f"sub/{key}"] = val
        if grad_norm_value is not None:
            metrics["grad_norm"] = grad_norm_value
        total_task_samples = max(sum(self.task_counts.values()), 1)
        for name in self.sampler.tasks:
            metrics[f"task_count/{name}"] = self.task_counts[name]
            metrics[f"task_fraction/{name}"] = self.task_counts[name] / total_task_samples
            if self.task_loss_ema[name] is not None:
                metrics[f"loss_ema/{name}"] = self.task_loss_ema[name]
        # 中文注释：文本 embedding 健康度监控（塌缩预警）：行 RMS 均值 + 随机行两两余弦。
        # 词表只有几十~几百行，开销可忽略，每步都算（与 logging 步错位会显示 na）。
        with torch.no_grad():
            table = self.accelerator.unwrap_model(self.model).embedder.E.weight
            metrics["text/emb_rms"] = float(table.pow(2).mean(dim=-1).sqrt().mean())
            if table.shape[0] > 4:
                idx = torch.randperm(table.shape[0], device=table.device)[:32]
                sub = torch.nn.functional.normalize(table[idx], dim=-1)
                cos = sub @ sub.t()
                off_diag = cos[~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)]
                metrics["text/emb_pairwise_cos"] = float(off_diag.mean())
        return metrics

    def _log_metrics(self, metrics):
        if not self.accelerator.is_main_process:
            return
        if not self.accelerator.sync_gradients:
            return
        if self.completed_steps % self.config.trainer.logging_frequency != 0:
            return
        max_steps = int(self.config.trainer.max_train_steps)
        steps_left = max(max_steps - int(self.completed_steps), 0)
        progress_frac = float(self.completed_steps) / max(max_steps, 1)
        elapsed = time.perf_counter() - self.train_start_time if self.train_start_time is not None else None
        eta_seconds = self.optimizer_step_time_ema * steps_left if self.optimizer_step_time_ema is not None else None
        if elapsed is not None:
            metrics["time/elapsed_hours"] = elapsed / 3600.0
        if eta_seconds is not None:
            metrics["time/eta_hours"] = eta_seconds / 3600.0
        metrics["progress/percent"] = progress_frac * 100.0
        last_lrs = self.lr_scheduler.get_last_lr()
        for i, group in enumerate(self.optimizer.param_groups):
            metrics[f"learning_rate/{group.get('name', str(i))}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
        wandb.log(metrics, step=self.completed_steps)
        parts = [
            f"[triflow] step={self.completed_steps}",
            f"progress={progress_frac * 100.0:.2f}%",
            f"eta={_format_duration(eta_seconds)}",
            f"task={metrics.get('task')}{'(dec)' if metrics.get('decoder_step') else ''}",
            f"loss={_format_float(metrics.get('loss/current'))}",
            f"grad_norm={_format_float(metrics.get('grad_norm'))}",
            f"emb_rms={_format_float(metrics.get('text/emb_rms'))}",
        ]
        for name in self.sampler.tasks:
            ema = metrics.get(f"loss_ema/{name}", None)
            if ema is not None:
                parts.append(f"ema_{name}={_format_float(ema)}")
        print(" ".join(parts), flush=True)

    ######### // code // ##########
    # 中文注释：full-state checkpoint / resume（v1 同款；EMA 经 register_for_checkpointing
    # 一并存取）。model-only .pt 给 eval；另存 EMA 合并版 *_ema.pt（eval 默认用 EMA）。
    def _trainer_state_path(self, state_dir) -> Path:
        return Path(state_dir) / "triflow_trainer_state.json"

    def _write_trainer_state(self, state_dir) -> None:
        payload = {
            "completed_steps": int(self.completed_steps),
            "nonfinite_grad_skips": int(self.nonfinite_grad_skips),
            "max_train_steps": int(self.config.trainer.max_train_steps),
        }
        with open(self._trainer_state_path(state_dir), "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _resolve_resume_dir(self) -> Path | None:
        raw = os.environ.get("TRIFLOW_RESUME", "").strip()
        if not raw:
            raw = str(getattr(self.config.trainer, "resume_from", "") or "").strip()
        if not raw or raw.lower() in {"none", "no", "off", "false", "0"}:
            return None
        ckpt_root = Path(self.config.output_dir) / "checkpoints"
        if raw.lower() in {"auto", "latest"}:
            if not ckpt_root.is_dir():
                return None
            candidates = []
            for path in ckpt_root.glob("state_steps_*"):
                if not path.is_dir():
                    continue
                try:
                    candidates.append((int(path.name.split("_")[-1]), path))
                except ValueError:
                    continue
            if not candidates:
                return None
            return max(candidates, key=lambda item: item[0])[1]
        path = Path(raw)
        if not path.is_dir():
            if self.accelerator.is_main_process:
                print(f"[triflow][warn] resume path not found, starting fresh: {raw}", flush=True)
            return None
        return path

    def _maybe_resume(self) -> None:
        resume_dir = self._resolve_resume_dir()
        if resume_dir is None:
            return
        if self.accelerator.is_main_process:
            print(f"[triflow] resuming full training state from {resume_dir}", flush=True)
        self.accelerator.load_state(str(resume_dir))
        state_path = self._trainer_state_path(resume_dir)
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.completed_steps = int(payload.get("completed_steps", 0))
            self.nonfinite_grad_skips = int(payload.get("nonfinite_grad_skips", 0))
        else:
            try:
                self.completed_steps = int(Path(resume_dir).name.split("_")[-1])
            except ValueError:
                self.completed_steps = 0
        if self.accelerator.is_main_process:
            print(f"[triflow] resumed at completed_steps={self.completed_steps}", flush=True)

    def _maybe_prune_states(self, ckpt_root) -> None:
        keep = int(os.environ.get("TRIFLOW_KEEP_LAST_STATES", "0"))
        if keep <= 0:
            return
        dirs = []
        for path in Path(ckpt_root).glob("state_steps_*"):
            if not path.is_dir():
                continue
            try:
                dirs.append((int(path.name.split("_")[-1]), path))
            except ValueError:
                continue
        dirs.sort(key=lambda item: item[0])
        for _, path in dirs[:-keep]:
            shutil.rmtree(path, ignore_errors=True)
            print(f"[triflow] pruned old full state: {path}", flush=True)

    def _save_checkpoint(self):
        ckpt_root = Path(self.config.output_dir) / "checkpoints"
        state_dir = ckpt_root / f"state_steps_{self.completed_steps}"
        self.accelerator.save_state(str(state_dir))
        if not self.accelerator.is_main_process:
            return
        self._write_trainer_state(state_dir)
        ckpt = ckpt_root / f"steps_{self.completed_steps}_pytorch_model.pt"
        torch.save(self.accelerator.get_state_dict(self.model), ckpt)
        if self.ema is not None:
            ema_ckpt = ckpt_root / f"steps_{self.completed_steps}_pytorch_model_ema.pt"
            torch.save(self.ema.merged_state_dict(self.accelerator.unwrap_model(self.model)), ema_ckpt)
        print(f"[triflow] checkpoint saved: {ckpt} (+ full state {state_dir})", flush=True)
        self._maybe_prune_states(ckpt_root)
    ######### // code // ##########

    def train(self):
        self.data_iter = iter(self.dataloader)
        self.train_start_time = time.perf_counter()
        self.last_optimizer_step_time = self.train_start_time
        progress = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_main_process or not _env_tqdm_enabled(),
        )
        while self.completed_steps < self.config.trainer.max_train_steps:
            t0 = time.perf_counter()
            batch = self._get_next_batch()
            t1 = time.perf_counter()
            metrics = self._train_step(batch)
            t2 = time.perf_counter()
            self.micro_steps_seen += 1
            if self.accelerator.sync_gradients:
                now = time.perf_counter()
                if self.last_optimizer_step_time is not None:
                    step_seconds = now - self.last_optimizer_step_time
                    if self.optimizer_step_time_ema is None:
                        self.optimizer_step_time_ema = step_seconds
                    else:
                        beta = self.step_time_ema_beta
                        self.optimizer_step_time_ema = beta * self.optimizer_step_time_ema + (1.0 - beta) * step_seconds
                self.last_optimizer_step_time = now
                self.completed_steps += 1
                progress.update(1)
            metrics["timing/data"] = t1 - t0
            metrics["timing/model"] = t2 - t1
            if self.accelerator.is_main_process and self.micro_steps_seen <= self.debug_first_micro_steps:
                print(
                    f"[triflow] micro_step={self.micro_steps_seen} completed_steps={self.completed_steps} "
                    f"sync={self.accelerator.sync_gradients} task={metrics.get('task')} "
                    f"data={t1 - t0:.3f}s model={t2 - t1:.3f}s",
                    flush=True,
                )
            self._log_metrics(metrics)
            if self.completed_steps > 0 and self.completed_steps % self.config.trainer.save_interval == 0 and self.accelerator.sync_gradients:
                self._save_checkpoint()
        progress.close()
        if self.accelerator.is_main_process:
            wandb.finish()
        self.accelerator.wait_for_everyone()
######### // code // ##########


def main(cfg):
    cfg.framework.name = "TriFlow"
    # 中文注释：显式注册 triflow 命名混合（不依赖 framework __init__ 的注册顺序）。
    from starVLA.triflow.data.mixtures import register_triflow_mixtures

    register_triflow_mixtures()
    cfg = wrap_config(cfg)
    _maybe_enable_tf32()
    accelerator = _build_accelerator(int(getattr(cfg.trainer, "gradient_accumulation_steps", 1)))
    setup_directories(cfg)
    setup_rank_output_control(cfg)
    model = build_framework(cfg)
    dataloader = build_joint_dataloader(cfg)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, cfg)
    trainer = TriFlowTrainer(cfg, model, dataloader, optimizer, lr_scheduler, accelerator)
    trainer.prepare_training()
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/triflow/configs/triflow_libero.yaml")
    args, clipargs = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)
    cli_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(clipargs))
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    main(cfg)
