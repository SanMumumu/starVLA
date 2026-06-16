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
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm

from starVLA.jointflow.data.joint_dataset import build_joint_dataloader, compute_null_action_table
# 中文注释：wandb 上行不稳 → 默认 SwanLab（调用面兼容 init/log/finish/run.summary）；
# JOINTFLOW_LOGGER=wandb 可切回。指标字段名全部不变。
from starVLA.jointflow.train.track_logger import get_tracker

wandb = get_tracker()
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

######### // code // ##########
# 中文注释：网页/集群日志面板不支持 ANSI 彩色与 emoji，会把 \x1b[..m 之类转义符打成乱码。
# 这里在训练入口统一关闭各类第三方库（rich/click/transformers 等）的彩色输出。
# 用 setdefault：不覆盖用户/启动脚本里已显式设置的值，只补默认值。
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


######### // code // ##########
# 中文注释：开启 TF32。当前 recipe 让 Qwen backbone + 两个 flow head 全程跑 fp32
# （fp32_forward=true，且 head 内部关了 autocast）。在 A100/H100 上 fp32 matmul
# 不走 tensor core，非常慢。TF32 用 tensor core 做 fp32 matmul、累加仍是 fp32，
# 精度损失可忽略，对 flow-matching 训练稳定性无影响，但能显著加速（常见 2-4x）。
# 设 JOINTFLOW_TF32=0 可关闭。
def _maybe_enable_tf32() -> None:
    if not _env_flag("JOINTFLOW_TF32", True):
        return
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
######### // code // ##########


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
    value = os.environ.get("JOINTFLOW_TQDM", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    if value != "auto":
        raise ValueError("Invalid JOINTFLOW_TQDM; expected auto/true/false.")
    return sys.stdout.isatty()


def _build_accelerator(gradient_accumulation_steps: int = 1) -> Accelerator:
    mixed_precision = _get_jointflow_mixed_precision()
    kwargs_handlers = []
    if _env_flag("JOINTFLOW_FIND_UNUSED_PARAMETERS", False):
        kwargs_handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))
    override = os.environ.get("JOINTFLOW_USE_DEEPSPEED", "").strip().lower()
    if override in {"0", "false", "no"}:
        return Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=kwargs_handlers,
        )
    use_deepspeed = override in {"1", "true", "yes"} or os.environ.get("ACCELERATE_USE_DEEPSPEED", "").lower() == "true"
    if use_deepspeed:
        return Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            deepspeed_plugin=DeepSpeedPlugin(),
            kwargs_handlers=kwargs_handlers,
        )
    return Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=kwargs_handlers,
    )


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


def setup_rank_output_control(cfg) -> None:
    mode = os.environ.get("JOINTFLOW_RANK_LOG_MODE", "files").strip().lower()
    if mode in {"", "all", "off", "none", "false", "0"}:
        return

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if rank == 0:
        if mode == "files":
            log_dir = Path(cfg.output_dir) / "rank_logs"
            print(f"[jointflow] non-zero rank stdout/stderr -> {log_dir}/rank*_local*.log", flush=True)
        return

    if mode == "quiet":
        path = os.devnull
    elif mode == "files":
        log_dir = Path(cfg.output_dir) / "rank_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = str(log_dir / f"rank{rank}_local{local_rank}.log")
    else:
        raise ValueError(
            f"Invalid JOINTFLOW_RANK_LOG_MODE={mode!r}; expected all/off/none, files, or quiet."
        )

    sys.stdout.flush()
    sys.stderr.flush()
    handle = open(path, "a", buffering=1)
    _RANK_LOG_HANDLES.append(handle)
    os.dup2(handle.fileno(), 1)
    os.dup2(handle.fileno(), 2)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)


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
        f"{os.environ.get('JOINTFLOW_FIND_UNUSED_PARAMETERS', '<unset; default false>')}"
    )
    print(
        "  JOINTFLOW_SYNC_TASK_BROADCAST="
        f"{os.environ.get('JOINTFLOW_SYNC_TASK_BROADCAST', '<unset; default false>')}"
    )
    print(
        "  JOINTFLOW_RANK_LOG_MODE="
        f"{os.environ.get('JOINTFLOW_RANK_LOG_MODE', '<unset; default files>')}"
    )
    print(f"  torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count={torch.cuda.device_count()}")
    print(f"  accelerator.device={accelerator_obj.device}")
    print(f"  accelerator.mixed_precision={accelerator_obj.mixed_precision}")
    print(f"  accelerator.gradient_accumulation_steps={accelerator_obj.gradient_accumulation_steps}")
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
        dtype = getattr(backbone, "dtype", "unknown")
        attn_impl = getattr(backbone, "attn_implementation", "unknown")
        fp32_forward = getattr(backbone, "fp32_forward", "unknown")
        llm_note = (
            f"Qwen text backbone; model_id={getattr(backbone, 'model_id', 'unknown')}; "
            f"dtype={dtype}; attn={attn_impl}; fp32_forward={fp32_forward}; "
            "loaded from AutoModelForCausalLM"
        )
    components = [
        ("llm_text_encoder", backbone, llm_note),
        ("dino_encoder", getattr(model, "dino", None), "Frozen online DINOv3 (in-model feature extraction); not optimized"),
        ("dino_projector", getattr(model, "dino_proj", None), "DINO 384-dim tokens to Qwen hidden size"),
        ("state_encoder", getattr(model, "state_enc", None), "Current proprio state to state token"),
        ("action_context_encoder", getattr(model, "act_ctx", None), "Clean action chunk to context tokens for FDM"),
        ("action_query_tokens", getattr(model, "action_queries", None), "Learned action query tokens for policy/IDM"),
        (
            "future_dino_query_tokens",
            getattr(model, "future_dino_queries", None),
            "Learned spatial future-DINO query tokens for FDM/passive",
        ),
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
        self.micro_steps_seen = 0
        self.debug_first_micro_steps = int(os.environ.get("JOINTFLOW_DEBUG_FIRST_MICRO_STEPS", "8"))
        self.train_start_time = None
        self.last_optimizer_step_time = None
        self.optimizer_step_time_ema = None
        self.step_time_ema_beta = float(os.environ.get("JOINTFLOW_STEP_TIME_EMA_BETA", "0.95"))
        self.nonfinite_grad_skips = 0

    ######### // code // ##########
    # 中文注释：覆盖父类 TrainerUtils.freeze_backbones。
    # 目的：(1) 去掉 emoji / "####" 等彩色打印（网页/集群日志渲染不了，会变成乱码）；
    #      (2) 修正父类里 `if dist.get_rank == 0`（漏了括号，拿函数对象和 0 比，恒为 False）的 rank 判断 bug；
    #      (3) 行为保持一致：按逗号分隔的相对路径冻结对应子模块的参数。
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        frozen = []
        if freeze_modules and isinstance(freeze_modules, str):
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]
            for path in patterns:
                module = model
                ok = True
                for attr in path.split("."):
                    if not hasattr(module, attr):
                        ok = False
                        break
                    module = getattr(module, attr)
                if not ok:
                    if is_rank0:
                        print(f"[jointflow] freeze: module path not found, skip: {path}", flush=True)
                    continue
                for param in module.parameters():
                    param.requires_grad = False
                frozen.append(path)
        if is_rank0:
            print(f"[jointflow] frozen modules: {frozen}", flush=True)
        return model
    ######### // code // ##########

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
        # 中文注释：lr_scheduler 没有进 accelerator.prepare()，因此 save_state 默认不会保存它。
        # 显式注册后，accelerator.save_state/load_state 会把 scheduler 状态(last_epoch 等)
        # 一并存/取，保证 resume 后 LR(尤其 cosine)接得上。必须在 _maybe_resume() 之前注册。
        self.accelerator.register_for_checkpointing(self.lr_scheduler)
        self._maybe_resume()
        if self.accelerator.is_main_process:
            wandb_kwargs = {
                "name": self.config.run_id,
                "dir": os.path.join(self.config.output_dir, "wandb"),
                "project": self.config.wandb_project,
                "entity": self.config.wandb_entity,
                "group": "jointflow-train",
                "mode": os.environ.get("WANDB_MODE", os.environ.get("JOINTFLOW_WANDB_MODE", "online")),
            }
            wandb_run_id = str(getattr(self.config, "wandb_run_id", "") or "").strip()
            if wandb_run_id:
                wandb_kwargs["id"] = wandb_run_id
                # 中文注释：带 run_id 且是 resume 续训时，允许 wandb 续接同一个 run(x 轴接上)。
                if self.completed_steps > 0:
                    wandb_kwargs["resume"] = "allow"
            wandb.init(**wandb_kwargs)
            write_parameter_table_to_wandb_summary(self.parameter_rows)

    def _save_initial_configs(self):
        if not self.accelerator.is_main_process:
            return
        output_dir = Path(self.config.output_dir)
        full_cfg = self.config.unwrap() if isinstance(self.config, AccessTrackedConfig) else self.config
        OmegaConf.save(full_cfg, output_dir / "config.full.yaml", resolve=True)
        # 中文注释：eval 的 read_mode_config 读取 run_dir/config.yaml 来重建 framework。
        # 这里 config.yaml 必须是“完整”配置:access-tracked 精简版会漏掉未被显式属性访问的子树
        # (实测漏过 framework.visual_model → eval 用默认 hidden=768 与 ckpt=1024 size mismatch、
        # 甚至缺 trainer → from_pretrained 报 Missing key trainer)。因此 config.yaml 直接落
        # 完整解析后的配置;access 精简版另存 config.accessed.yaml 备查。
        OmegaConf.save(full_cfg, output_dir / "config.yaml", resolve=True)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.accessed.yaml", use_original_values=False)

    def _get_next_batch(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            return next(self.data_iter)

    def _sample_task_for_all_ranks(self) -> str:
        # 中文注释：每个 step 所有 rank 必须采到“同一个 task”。否则不同 rank 跑不同机制
        # （policy 用 action_head、fdm 用 visual_head ...），参与 backward 的参数集合不同，
        # DDP 的梯度 all-reduce 在不同 rank 上对不上 → 直接挂死。
        # 这里在分布式下默认从 rank0 broadcast task_id，强一致；不再依赖各 rank RNG 恰好同步。
        # 如需关闭可设 JOINTFLOW_SYNC_TASK_BROADCAST=false（仅在确认 RNG 同步时）。
        task = self.sampler.sample()
        if dist.is_initialized() and _env_flag("JOINTFLOW_SYNC_TASK_BROADCAST", True):
            task_id = self.sampler.task_to_id[task] if dist.get_rank() == 0 else 0
            task_tensor = torch.tensor([task_id], device=self.accelerator.device, dtype=torch.long)
            dist.broadcast(task_tensor, src=0)
            task = self.sampler.tasks[int(task_tensor.item())]
        return task

    ######### // code // ##########
    # 中文注释：集合通信版的 loss 有限性检查。
    # 单 rank 直接 raise 会让“坏 rank”抛错退出、其它 rank 卡在下一次 all-reduce → NCCL 挂死。
    # 这里先各自判定，再 all-reduce(MIN) 汇总：只要任一 rank 出现 NaN/Inf，所有 rank 一起抛错，
    # 得到干净的同步失败而不是挂死。
    def _raise_if_loss_nonfinite(self, loss: torch.Tensor, task: str, batch) -> None:
        local_finite = bool(torch.isfinite(loss.detach()).all())
        all_finite = local_finite
        if dist.is_initialized():
            flag = torch.tensor([1.0 if local_finite else 0.0], device=self.accelerator.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            all_finite = flag.item() > 0.5
        if all_finite:
            return
        samples = [
            {
                key: ex.get(key, None)
                for key in ("dataset_name", "trajectory_id", "base_index", "future_index", "lang")
                if key in ex
            }
            for ex in batch[:4]
        ]
        raise FloatingPointError(
            f"Non-finite loss before backward: task={task}, local_finite={local_finite}, "
            f"completed_steps={self.completed_steps}, samples={samples}, loss={loss.detach()}"
        )
    ######### // code // ##########

    def _nonfinite_grad_summaries(self, max_items: int = 12) -> list[dict]:
        rows = []
        model = self.accelerator.unwrap_model(self.model)
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None or not torch.is_tensor(grad) or not grad.is_floating_point():
                continue
            finite = torch.isfinite(grad)
            if bool(finite.all()):
                continue
            detached = grad.detach()
            row = {
                "name": name,
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "nan": int(torch.isnan(detached).sum().item()),
                "inf": int(torch.isinf(detached).sum().item()),
                "finite": int(finite.sum().item()),
                "numel": int(detached.numel()),
            }
            if bool(finite.any()):
                values = detached[finite].float()
                row.update(
                    {
                        "min": float(values.min().item()),
                        "max": float(values.max().item()),
                        "mean": float(values.mean().item()),
                    }
                )
            rows.append(row)
            if len(rows) >= max_items:
                break
        return rows

    def _train_step(self, batch):
        task = self._sample_task_for_all_ranks()
        grad_norm_value = None
        nonfinite_grad = False
        with self.accelerator.accumulate(self.model):
            output = self.model(batch, task=task)
            loss = output[f"{task}_loss"]
            # 中文注释：多卡安全的 NaN/Inf 检查（所有 rank 一起判定/抛错，避免单 rank 抛错挂死）。
            self._raise_if_loss_nonfinite(loss, task, batch)
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
                if grad_norm is not None:
                    grad_norm_tensor = grad_norm.detach() if torch.is_tensor(grad_norm) else torch.tensor(float(grad_norm), device=self.accelerator.device)
                    grad_norm_value = float(grad_norm_tensor.float().cpu())
                    nonfinite_grad = not bool(torch.isfinite(grad_norm_tensor).all())
            ######### // code // ##########
            # 中文注释：非有限梯度（NaN/Inf）处理——把“崩溃”改成“跳过这一步”。
            # 为什么必须在 optimizer.step() 之前拦截：一旦把 NaN 梯度喂进 AdamW，它的
            # 一阶/二阶动量 (exp_avg / exp_avg_sq) 会被永久污染成 NaN，之后每一步都产
            # NaN 参数 → 整个训练不可恢复。因此发现非有限梯度就丢弃本 step 的累积梯度：
            # 不 optimizer.step、不 lr_scheduler.step，只 zero_grad 清掉坏梯度，训练继续。
            #
            # 多卡 DDP 安全性：DDP 在 backward(sync step) 已把梯度 all-reduce 同步，
            # clip_grad_norm_ 在每个 rank 得到完全相同的 total_norm，于是所有 rank 对
            # nonfinite_grad 的判定一致 → 一起跳过，不会出现 rank 间分叉导致 NCCL 挂死。
            #
            # 提示：之前 bad_grads 里“所有参数 100% NaN”是 clip_grad_norm_ 把全部梯度
            # ×clip_coef(=NaN) 抹出来的假象，不代表每个算子都各自产了 NaN；真正源头是
            # 某一个/少数梯度先变 NaN（偶发数值尖刺）。
            # 用 JOINTFLOW_NONFINITE_GRAD=raise 可恢复旧的“直接崩溃”行为。
            if nonfinite_grad:
                self.nonfinite_grad_skips += 1
                if self.accelerator.is_main_process:
                    samples = [
                        {
                            key: ex.get(key, None)
                            for key in ("dataset_name", "trajectory_id", "base_index", "future_index", "lang")
                            if key in ex
                        }
                        for ex in batch[:4]
                    ]
                    print(
                        f"[jointflow][warn] non-finite grad -> skip optimizer.step: task={task} "
                        f"completed_steps={self.completed_steps} grad_norm={grad_norm_value} "
                        f"total_skips={self.nonfinite_grad_skips} samples={samples}",
                        flush=True,
                    )
                if os.environ.get("JOINTFLOW_NONFINITE_GRAD", "skip").strip().lower() == "raise":
                    raise FloatingPointError(
                        f"Non-finite grad norm before optimizer.step: task={task}, "
                        f"completed_steps={self.completed_steps}, grad_norm={grad_norm_value}"
                    )
                self.optimizer.zero_grad(set_to_none=True)
                max_skips = int(os.environ.get("JOINTFLOW_MAX_NONFINITE_SKIPS", "100"))
                if max_skips >= 0 and self.nonfinite_grad_skips > max_skips:
                    raise FloatingPointError(
                        f"Too many non-finite grad steps ({self.nonfinite_grad_skips} > {max_skips}); "
                        "this looks like real divergence rather than a transient spike. Aborting."
                    )
            else:
                self.optimizer.step()
                if self.accelerator.sync_gradients:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()
            ######### // code // ##########
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
        if grad_norm_value is not None:
            metrics["grad_norm"] = grad_norm_value
        # 中文注释（CED C4）：framework forward 返回的子指标（loss/act、loss/ident、
        # metric/probe_mse_val、stat/* 等标量）全部透传给 wandb；key 即 wandb 字段名。
        for key, value in output.items():
            if key == f"{task}_loss":
                continue
            if torch.is_tensor(value) and value.numel() == 1:
                metrics[key] = float(value.detach().cpu())
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
            metrics["time/elapsed_seconds"] = elapsed
            metrics["time/elapsed_hours"] = elapsed / 3600.0
        if eta_seconds is not None:
            metrics["time/eta_seconds"] = eta_seconds
            metrics["time/eta_hours"] = eta_seconds / 3600.0
        if self.optimizer_step_time_ema is not None:
            metrics["time/optimizer_step_seconds_ema"] = self.optimizer_step_time_ema
        metrics["progress/fraction"] = progress_frac
        metrics["progress/percent"] = progress_frac * 100.0
        metrics["progress/steps_left"] = steps_left
        last_lrs = self.lr_scheduler.get_last_lr()
        for i, group in enumerate(self.optimizer.param_groups):
            metrics[f"learning_rate/{group.get('name', str(i))}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
        wandb.log(metrics, step=self.completed_steps)
        task = metrics.get("task", "unknown")
        parts = [
            f"[jointflow] step={self.completed_steps}",
            f"progress={progress_frac * 100.0:.2f}%",
            f"left={steps_left}",
            f"elapsed={_format_duration(elapsed)}",
            f"eta={_format_duration(eta_seconds)}",
            f"step_ema={_format_float(self.optimizer_step_time_ema)}s",
            f"task={task}",
            f"loss={_format_float(metrics.get('loss/current'))}",
            f"grad_norm={_format_float(metrics.get('grad_norm'))}",
            f"data={_format_float(metrics.get('timing/data'))}s",
            f"model={_format_float(metrics.get('timing/model'))}s",
        ]
        for name in self.sampler.tasks:
            ema = metrics.get(f"loss_ema/{name}", None)
            if ema is not None:
                parts.append(f"ema_{name}={_format_float(ema)}")
        parts.extend(
            [
                f"lr_backbone={_format_float(metrics.get('learning_rate/backbone'), 3)}",
                f"lr_action={_format_float(metrics.get('learning_rate/action_head'), 3)}",
                f"lr_visual={_format_float(metrics.get('learning_rate/visual_head'), 3)}",
            ]
        )
        print(" ".join(parts), flush=True)

    ######### // code // ##########
    # 中文注释：full-state checkpoint / resume 支持。
    # 现状(改前)只存了 model 权重的 .pt，没有 optimizer/scheduler/step，也没有任何 resume 逻辑，
    # 任何一次中断(NaN 崩溃/抢占/OOM/节点故障)都=白练。这里:
    #   - _save_checkpoint: 所有 rank 调 accelerator.save_state() 存“完整训练态”
    #     (model+optimizer+已注册的 scheduler+RNG)，并保留原来给 eval 用的 model-only .pt；
    #   - _maybe_resume:   prepare 之后从 state 目录 load_state，并恢复 completed_steps。
    # resume 入口(任选其一，env 优先)：
    #   - env  JOINTFLOW_RESUME=auto|latest         自动找 output_dir 下最新 state_steps_*
    #   - env  JOINTFLOW_RESUME=/abs/path/state_xxx 指定目录
    #   - cfg.trainer.resume_from: 同上
    # 注意：本实现不快进 dataloader(数据顺序会从头洗)，但 model/optimizer/scheduler/step 都接得上，
    # 对这类训练足够；要精确续数据可后续再接 accelerate 的 skip_first_batches。
    def _trainer_state_path(self, state_dir) -> Path:
        return Path(state_dir) / "jointflow_trainer_state.json"

    def _write_trainer_state(self, state_dir) -> None:
        import json

        payload = {
            "completed_steps": int(self.completed_steps),
            "nonfinite_grad_skips": int(self.nonfinite_grad_skips),
            "max_train_steps": int(self.config.trainer.max_train_steps),
        }
        with open(self._trainer_state_path(state_dir), "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _resolve_resume_dir(self) -> Path | None:
        raw = os.environ.get("JOINTFLOW_RESUME", "").strip()
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
                    step = int(path.name.split("_")[-1])
                except ValueError:
                    continue
                candidates.append((step, path))
            if not candidates:
                return None
            return max(candidates, key=lambda item: item[0])[1]
        path = Path(raw)
        if not path.is_dir():
            if self.accelerator.is_main_process:
                print(
                    f"[jointflow][warn] resume path not found, starting fresh: {raw}",
                    flush=True,
                )
            return None
        return path

    def _maybe_resume(self) -> None:
        resume_dir = self._resolve_resume_dir()
        if resume_dir is None:
            return
        is_main = self.accelerator.is_main_process
        if is_main:
            print(f"[jointflow] resuming full training state from {resume_dir}", flush=True)
        # load_state 是集合操作，所有 rank 一起调用，恢复 model+optimizer+scheduler+RNG。
        self.accelerator.load_state(str(resume_dir))
        # 中文注释：所有 rank 都读同一个 json(output_dir 共享存储)，保证 completed_steps 全 rank 一致；
        # 否则各 rank while 循环步数不同 → 下次 all-reduce 对不上 → DDP 挂死。
        state_path = self._trainer_state_path(resume_dir)
        if state_path.exists():
            import json

            with open(state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.completed_steps = int(payload.get("completed_steps", 0))
            self.nonfinite_grad_skips = int(payload.get("nonfinite_grad_skips", 0))
        else:
            try:
                self.completed_steps = int(Path(resume_dir).name.split("_")[-1])
            except ValueError:
                self.completed_steps = 0
        if is_main:
            print(
                f"[jointflow] resumed at completed_steps={self.completed_steps} "
                f"(nonfinite_grad_skips={self.nonfinite_grad_skips})",
                flush=True,
            )

    def _maybe_prune_states(self, ckpt_root) -> None:
        # 中文注释：full state 体积大(model+AdamW≈3x 权重)。可选只保留最近 N 份完整态。
        # 默认 0=全部保留(不删任何东西，避免意外丢 checkpoint)。只删本程序产生的 state_steps_* 目录。
        keep = int(os.environ.get("JOINTFLOW_KEEP_LAST_STATES", "0"))
        if keep <= 0:
            return
        import shutil

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
            print(f"[jointflow] pruned old full state: {path}", flush=True)

    def _save_checkpoint(self):
        ckpt_root = Path(self.config.output_dir) / "checkpoints"
        state_dir = ckpt_root / f"state_steps_{self.completed_steps}"
        # 中文注释：save_state 是集合操作，必须所有 rank 一起调用(不能放在 is_main 之后)。
        self.accelerator.save_state(str(state_dir))
        if not self.accelerator.is_main_process:
            return
        self._write_trainer_state(state_dir)
        # 保留原有 model-only .pt(给 eval/部署，产物不变)。
        ckpt = ckpt_root / f"steps_{self.completed_steps}_pytorch_model.pt"
        torch.save(self.accelerator.get_state_dict(self.model), ckpt)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(Path(self.config.output_dir) / "config.yaml", use_original_values=False)
        print(f"[jointflow] checkpoint saved: {ckpt} (+ full state {state_dir})", flush=True)
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
                    metrics["time/optimizer_step_seconds"] = step_seconds
                self.last_optimizer_step_time = now
                self.completed_steps += 1
                progress.update(1)
            metrics["timing/data"] = t1 - t0
            metrics["timing/model"] = t2 - t1
            if self.accelerator.is_main_process and self.micro_steps_seen <= self.debug_first_micro_steps:
                print(
                    "[jointflow] "
                    f"micro_step={self.micro_steps_seen} completed_steps={self.completed_steps} "
                    f"sync_gradients={self.accelerator.sync_gradients} task={metrics.get('task')} "
                    f"data={t1 - t0:.3f}s model={t2 - t1:.3f}s",
                    flush=True,
                )
            self._log_metrics(metrics)

            if self.completed_steps > 0 and self.completed_steps % self.config.trainer.save_interval == 0:
                self._save_checkpoint()
        progress.close()
        # 中文注释（收尾顺序坑）：必须先 barrier、再 rank0 单独 finish。
        # 反过来（先 rank0 finish 再 barrier）会让 rank1-7 卡在 wait_for_everyone 这个
        # NCCL barrier 上等 rank0，而 rank0 卡在 swanlab.finish() 的网络上传里 → 超过 NCCL
        # 超时 → 其它 rank SIGABRT(exitcode -6)，整个 job 被标 failed（虽然 ckpt 早已存好）。
        # 现在：8 个 rank 跑完 100k 后毫秒级到齐 barrier，之后再无集合通信，rank0 慢慢传日志、
        # rank1-7 直接退出，互不阻塞。finish 再用 try/except 兜底，避免日志网络异常污染退出码。
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            try:
                wandb.finish()
            except Exception as exc:  # noqa: BLE001
                print(f"[jointflow][warn] tracker finish failed (training already complete): {exc}", flush=True)
######### // code // ##########


def main(cfg):
    cfg.framework.name = "QwenJointFlow"
    cfg = wrap_config(cfg)
    _maybe_enable_tf32()  # 中文注释：开启 TF32 加速 fp32 matmul（见 _maybe_enable_tf32）
    accelerator = _build_accelerator(int(getattr(cfg.trainer, "gradient_accumulation_steps", 1)))
    setup_directories(cfg)
    setup_rank_output_control(cfg)
    model = build_framework(cfg)
    dataloader = build_joint_dataloader(cfg)
    # 中文注释（CED C2）：effect 任务需要"归一化空间的空动作"查表（per-dataset 常数），
    # 从 dataloader 持有的 mixture 数据集的活 Normalizer 上算出并注入 model。
    ced_cfg = cfg.framework.get("ced", None)
    if ced_cfg is not None and bool(ced_cfg.get("enabled", False)):
        null_table = compute_null_action_table(dataloader.dataset)
        model.set_null_action_table(null_table)
        if accelerator.is_main_process:
            print(f"[jointflow][ced] null-action table ready for datasets: {sorted(null_table)}", flush=True)
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
