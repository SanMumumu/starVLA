"""PR4/PR5/PR6: QwenJointFlow framework.

复用:
- starVLA.model.framework.base_framework.baseframework
- starVLA.model.framework.share_tools.merge_framework_config
- starVLA.model.modules.action_model.GR00T_ActionHeader.FlowmatchingActionHead
- starVLA.model.tools.FRAMEWORK_REGISTRY

说明:
本文件是 JointFlow 的唯一模型 API。它只注册新 framework，不修改
starVLA/model/framework 下的任何现有文件。
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from omegaconf import OmegaConf
from PIL import Image


######### // code // ##########
# 中文注释：是否启用逐张量的 finite 断言。
# 单卡 debug 默认开（方便定位 NaN/Inf 来源）；多卡训练默认关，
# 因为这些断言在 forward 里逐 rank 抛异常，会造成只有一张卡抛错、
# 其它卡卡在下一次 all-reduce → NCCL 挂死。多卡下统一交给 trainer 里
# 的“集合通信版” loss 有限性检查（所有 rank 一起判定、一起抛错）。
# 需要在多卡下也打开逐张量断言时，设 JOINTFLOW_ASSERT_FINITE=1。
def _assert_finite_enabled() -> bool:
    flag = os.environ.get("JOINTFLOW_ASSERT_FINITE", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    # 未显式设置：分布式（多卡）下默认关闭，单进程下默认开启。
    return not dist.is_initialized()
######### // code // ##########

from starVLA.jointflow.backbone.qwen2_text_interface import _QWen2_Text_Interface
from starVLA.jointflow.modules.attention_mask import build_block_causal_mask
from starVLA.jointflow.modules.dino_v3 import DINOv3Backbone, resolve_dino_spec
from starVLA.jointflow.modules.joint_modules import (
    ActionContextEncoder,
    ActionQueryTokenBank,
    DinoProjector,
    FutureDinoQueryTokenBank,
    StateEncoder,
)
from starVLA.jointflow.modules.visual_dino_flow_head import VisualFlowMatchingHead
from starVLA.jointflow.patches.qwen2_text_patch import apply_qwen2_text_block_mask_patch
from starVLA.jointflow.data.mix_registry import resolve_data_mix
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.tools import FRAMEWORK_REGISTRY


######### // code // ##########
# 中文注释：QwenJointFlow 默认配置。YAML 中 framework 节点会覆盖这些默认值。
# action_model.state_dim 固定为 0：state 只作为 Qwen token 注入，避免 action head 内部分支重复注入。
@dataclass
class QwenJointFlowDefaultConfig:
    name: str = "QwenJointFlow"
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "playground/Pretrained_models/Qwen2.5-0.5B",
            "attn_implementation": "eager",
            "torch_dtype": "float32",
            "fp32_forward": True,
            "vl_hidden_dim": 896,
            "max_text_length": 256,
        }
    )
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
        }
    )
    state: dict = field(default_factory=lambda: {"inject_mode": "token", "n_state_tokens": 1, "state_dim": 8})
    tasks: dict = field(
        default_factory=lambda: {
            "weights": {"policy": 1.0, "fdm": 0.5, "idm": 0.5, "passive": 0.5},
            "hybrid_mask": True,
            "attention_mask_neg_value": -1.0e4,
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 0,
            "action_horizon": 8,
            "repeated_diffusion_steps": 1,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 8,
            "diffusion_model_cfg": {
                "cross_attention_dim": 896,
                "dropout": 0.1,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 8,
                "output_dim": 768,
                "positional_embeddings": None,
            },
        }
    )
    visual_model: dict = field(
        default_factory=lambda: {
            "d_dino": 384,
            "n_query": 196,
            "max_image_queries": 256,
            "hidden_size": 768,
            "cross_attention_dim": 896,
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
        }
    )
######### // code // ##########


######### // code // ##########
# 中文注释：QwenJointFlow 主模型。forward 每次只跑一个 task；
# policy/idm 使用 action flow loss，fdm/passive 使用 visual DINO flow loss。
@FRAMEWORK_REGISTRY.register("QwenJointFlow")
class QwenJointFlowVLA(baseframework):
    def __init__(self, config=None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenJointFlowDefaultConfig, config)
        apply_qwen2_text_block_mask_patch()

        self.backbone = _QWen2_Text_Interface(self.config)
        hidden_size = int(self.backbone.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = hidden_size
        self.config.framework.action_model.state_dim = 0
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = hidden_size
        self.config.framework.visual_model.cross_attention_dim = hidden_size

        action_cfg = self.config.framework.action_model
        dino_cfg = self.config.framework.dino
        visual_cfg = self.config.framework.visual_model
        state_cfg = self.config.framework.state

        self.action_horizon = int(action_cfg.action_horizon)
        self.action_dim = int(action_cfg.action_dim)
        # 中文注释：在线 DINO。dino_spec 由 model_size/weights 等解析而来（见 resolve_dino_spec），
        # 是“一处指定尺寸”的真源：embed_dim 据此确定，并同步给 visual_model.d_dino，
        # 保证 visual flow head 的 x_embed/x_decode 维度与 DINO 输出对齐（改尺寸不用再手动改两处）。
        self._dino_spec = resolve_dino_spec(dino_cfg)
        self.d_dino = int(self._dino_spec["embed_dim"])
        self.config.framework.visual_model.d_dino = self.d_dino
        self.state_inject_mode = state_cfg.get("inject_mode", "token")

        self.dino_proj = DinoProjector(d_dino=self.d_dino, hidden_size=hidden_size)
        self.act_ctx = ActionContextEncoder(action_dim=self.action_dim, hidden_size=hidden_size)
        self.state_enc = StateEncoder(
            state_dim=int(state_cfg.get("state_dim", 8)),
            hidden_size=hidden_size,
            n_state_tokens=int(state_cfg.get("n_state_tokens", 1)),
        )
        self.action_queries = ActionQueryTokenBank(action_horizon=self.action_horizon, hidden_size=hidden_size)
        self.future_dino_queries = FutureDinoQueryTokenBank(
            max_queries=int(visual_cfg.get("max_image_queries", visual_cfg.get("n_query", 196))),
            hidden_size=hidden_size,
        )

        self.action_head = FlowmatchingActionHead(self.config)
        self.visual_head = VisualFlowMatchingHead(self.config)
        # 中文注释：在线模式默认 load_live_backbone=true，构造时即加载 frozen DINOv3；
        # __init__ 阶段不 .to(device)，随整模型 .to() 一起搬。requires_grad=False，不进优化器更新。
        self.dino = None
        if bool(dino_cfg.get("load_live_backbone", False)):
            self.dino = DINOv3Backbone(**self._dino_spec)

        self.register_buffer("_dino_mean", torch.zeros(self.d_dino), persistent=False)
        self.register_buffer("_dino_std", torch.ones(self.d_dino), persistent=False)
        self._dino_stats_source = None
        self._load_dino_stats(dino_cfg.get("stats_path", None))

    @staticmethod
    def _zero_grad_anchor_for_modules(modules: list[nn.Module | None], ref: torch.Tensor) -> torch.Tensor:
        anchor = ref.new_zeros(())
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

    def _unused_task_param_anchor(self, task: str, ref: torch.Tensor) -> torch.Tensor:
        modules: list[nn.Module | None] = []
        if task in {"policy", "idm"}:
            modules.extend([self.visual_head, self.future_dino_queries, self.act_ctx])
        elif task == "fdm":
            modules.extend([self.action_head, self.action_queries])
        elif task == "passive":
            modules.extend([self.action_head, self.action_queries, self.act_ctx])
        else:
            raise ValueError(f"Unsupported JointFlow task `{task}`")
        if self.state_inject_mode != "token":
            modules.append(self.state_enc)
        return self._zero_grad_anchor_for_modules(modules, ref)

    ######### // code // ##########
    # 中文注释：对“全部可训练子模块”做 zero-grad anchor。
    # 用途：当某个 rank 因为 future_valid 全为 False 走了 early-return（没调用对应 head），
    # 而其它 rank 走了正常路径（调用了 head）时，DDP 要求每个 rank 在每次 backward
    # 都把所有参数标记为 ready。这里把所有可训练参数都挂上 0 梯度，
    # 保证 early-return 这一支也让每个参数 ready，从而消除多卡 reduction 卡死。
    # （已用 2 卡 DDP find_unused_parameters=False 验证：partial anchor 会挂死，full anchor 通过。）
    def _all_trainable_anchor(self, ref: torch.Tensor) -> torch.Tensor:
        modules = [
            self.backbone,
            self.dino_proj,
            self.act_ctx,
            self.state_enc,
            self.action_queries,
            self.future_dino_queries,
            self.action_head,
            self.visual_head,
        ]
        return self._zero_grad_anchor_for_modules(modules, ref)
    ######### // code // ##########

    @staticmethod
    def _tensor_debug_summary(tensor: torch.Tensor) -> dict:
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        summary = {
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "numel": int(detached.numel()),
            "finite": int(finite.sum().item()),
            "nan": int(torch.isnan(detached).sum().item()) if detached.is_floating_point() else 0,
            "inf": int(torch.isinf(detached).sum().item()) if detached.is_floating_point() else 0,
        }
        if bool(finite.any()):
            values = detached[finite].float()
            summary.update(
                {
                    "min": float(values.min().item()),
                    "max": float(values.max().item()),
                    "mean": float(values.mean().item()),
                    "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
                }
            )
        return summary

    def _sample_debug_summary(self, examples: List[dict], max_items: int = 4) -> list[dict]:
        keys = ("dataset_name", "trajectory_id", "base_index", "future_index", "future_valid_steps", "future_stride", "lang")
        rows = []
        for ex in examples[:max_items]:
            row = {key: ex.get(key, None) for key in keys if key in ex}
            if "lang" in row:
                row["lang"] = str(row["lang"])[:120]
            rows.append(row)
        return rows

    def _assert_finite(self, name: str, tensor: torch.Tensor | None, task: str, examples: List[dict]) -> None:
        # 中文注释：多卡训练默认跳过逐张量断言，避免单 rank 抛异常导致 NCCL 挂死。
        if not _assert_finite_enabled():
            return
        if tensor is None or not torch.is_tensor(tensor) or not tensor.is_floating_point():
            return
        if bool(torch.isfinite(tensor).all()):
            return
        summary = self._tensor_debug_summary(tensor)
        samples = self._sample_debug_summary(examples)
        raise FloatingPointError(
            f"Non-finite tensor in QwenJointFlow task={task} name={name}: "
            f"summary={summary}, samples={samples}"
        )

    def _read_dino_stats_file(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _default_dino_stats_paths(self) -> list[Path]:
        datasets_cfg = getattr(self.config, "datasets", None)
        vla_cfg = getattr(datasets_cfg, "vla_data", None)
        if vla_cfg is None:
            return []
        data_root = vla_cfg.get("data_root_dir", None)
        data_mix = vla_cfg.get("data_mix", None)
        if not data_root or not data_mix:
            return []
        feature_dir = vla_cfg.get("dino_feature_dir", "latents")
        try:
            mixture = resolve_data_mix(str(data_mix))
        except Exception:
            return []
        paths = []
        for data_name, _, _ in mixture:
            path = Path(data_root) / str(data_name) / str(feature_dir) / "dino_v3_stats.json"
            if path.exists():
                paths.append(path)
        return paths

    def _combine_dino_stats(self, stats_list: list[dict]) -> dict | None:
        if not stats_list:
            return None
        if len(stats_list) == 1:
            return stats_list[0]

        total_count = 0.0
        mean_acc = None
        second_moment_acc = None
        for stats in stats_list:
            count = float(stats.get("count", 0))
            if count <= 0:
                continue
            mean = torch.tensor(stats["mean"], dtype=torch.float64)
            std = torch.tensor(stats["std"], dtype=torch.float64).clamp_min(1e-12)
            second_moment = std.square() + mean.square()
            if mean_acc is None:
                mean_acc = count * mean
                second_moment_acc = count * second_moment
            else:
                mean_acc += count * mean
                second_moment_acc += count * second_moment
            total_count += count
        if total_count <= 0 or mean_acc is None or second_moment_acc is None:
            return None
        mean = mean_acc / total_count
        var = (second_moment_acc / total_count - mean.square()).clamp_min(1e-12)
        return {"mean": mean.tolist(), "std": var.sqrt().tolist(), "count": int(total_count)}

    def _load_dino_stats(self, stats_path: str | None) -> None:
        source = None
        if stats_path:
            path = Path(stats_path)
            if not path.exists():
                return
            stats = self._read_dino_stats_file(path)
            source = path.as_posix()
        else:
            paths = self._default_dino_stats_paths()
            stats = self._combine_dino_stats([self._read_dino_stats_file(path) for path in paths])
            source = ",".join(path.as_posix() for path in paths) if paths else None
            if stats is None:
                return

        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32).clamp_min(1e-6)
        if mean.numel() != self.d_dino or std.numel() != self.d_dino:
            # 中文注释：stats 维度与当前 DINO embed_dim 对不上（常见于换了 dino.model_size
            # 之后命中旧尺寸残留的离线 stats）。在线模式下 stats 是可选项：不崩，退回恒等归一化
            # （直接用 DINOv3 自带 LayerNorm 后的 patch token，数值已较稳定）。
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(
                    f"[QwenJointFlow][warn] DINO stats dim {mean.numel()} != embed_dim {self.d_dino} "
                    f"(source={source}); skipping stats, using identity normalization.",
                    flush=True,
                )
            return
        self._dino_mean.copy_(mean)
        self._dino_std.copy_(std)
        self._dino_stats_source = source

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _stack_field(self, examples: List[dict], key: str, required: bool = True) -> torch.Tensor | None:
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

    def _get_view_keys(self, examples: List[dict], num_views: int) -> list[str]:
        keys = examples[0].get("dino_view_keys") or examples[0].get("view_keys")
        if keys:
            return [str(k) for k in keys]
        return [f"view_{i}" for i in range(num_views)]

    def _select_future_dino(self, dino_1: torch.Tensor, examples: List[dict]) -> torch.Tensor:
        if dino_1.ndim == 3:
            return dino_1
        if dino_1.ndim != 4:
            raise ValueError(f"Expected dino_1 [B,V,N,D] or [B,N,D], got {tuple(dino_1.shape)}")
        view_keys = self._get_view_keys(examples, dino_1.shape[1])
        preferred = list(self.config.framework.dino.get("future_view_keys", []))
        chosen = 0
        for name in preferred:
            if name in view_keys:
                chosen = view_keys.index(name)
                break
        return dino_1[:, chosen]

    def _flatten_dino(self, dino: torch.Tensor) -> torch.Tensor:
        if dino.ndim == 4:
            bsz, views, n_tokens, dim = dino.shape
            dino = dino.reshape(bsz, views * n_tokens, dim)
        elif dino.ndim != 3:
            raise ValueError(f"Expected DINO features [B,V,N,D] or [B,N,D], got {tuple(dino.shape)}")

        pool = int(self.config.framework.dino.get("dino_pool", 1))
        if pool > 1:
            n_tokens = dino.shape[1]
            side = int(n_tokens**0.5)
            if side * side == n_tokens and side % pool == 0:
                dino = dino.view(dino.shape[0], side, side, dino.shape[-1])
                dino = dino.view(dino.shape[0], side // pool, pool, side // pool, pool, dino.shape[-1]).mean(dim=(2, 4))
                dino = dino.reshape(dino.shape[0], -1, dino.shape[-1])
        return dino

    def _normalize_live_dino(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self._dino_mean.to(z.device, z.dtype)) / self._dino_std.to(z.device, z.dtype)

    ######### // code // ##########
    # 中文注释：在线 DINO 工具方法。
    # 训练时 dataset 返回原始图像 image_0/image_1（[V,H,W,C] uint8）；eval 时 policy server
    # 给的是 `image`（PIL 列表）。这里统一在 GPU 上跑 frozen DINOv3 → 标准化特征 [B,V,N,D]。
    def _ensure_dino(self) -> DINOv3Backbone:
        if self.dino is None:
            # 中文注释：兜底惰性构造（理论上 load_live_backbone=true 已在 __init__ 建好）。
            self.dino = DINOv3Backbone(**self._dino_spec).to(self.device)
        return self.dino

    def _collect_images(self, examples: List[dict], keys: list[str]):
        # 返回 batch 维的 view 列表：[[view0,view1,...], ...]；任一样本缺图返回 None。
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
                if val.ndim == 4:  # [V,H,W,C]
                    views = [val[i] for i in range(val.shape[0])]
                elif val.ndim == 3:  # [H,W,C]
                    views = [val]
                else:
                    raise ValueError(f"Unsupported image array ndim={val.ndim} for keys={keys}")
            elif isinstance(val, (list, tuple)):
                views = list(val)
            else:
                views = [val]
            batch_views.append(views)
        return batch_views

    def _run_dino_on_images(self, examples: List[dict], keys: list[str], required: bool) -> torch.Tensor | None:
        batch_views = self._collect_images(examples, keys)
        if batch_views is None:
            if required:
                raise KeyError(f"Online DINO needs one of {keys} in the batch (got none).")
            return None
        dino = self._ensure_dino()
        n_views = len(batch_views[0])
        flat = [img for views in batch_views for img in views]
        # 中文注释：DINO 强制 fp32（关 autocast），与 backbone / flow head 数值口径一致，
        # 保证作为 flow 目标的 dino_1 特征稳定、可复现。
        device_type = self.device.type
        autocast_ctx = torch.autocast(device_type=device_type, enabled=False) if device_type in {"cuda", "cpu"} else nullcontext()
        with autocast_ctx:
            tensor = dino.preprocess_batch(flat)
            feats = dino(tensor)  # [B*V, N, D]
        bsz = len(batch_views)
        feats = feats.view(bsz, n_views, feats.shape[1], feats.shape[2]).float()
        return self._normalize_live_dino(feats)
    ######### // code // ##########

    def _examples_to_batch(self, examples: List[dict], require_future_dino: bool, require_action: bool) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        # 中文注释：优先用离线特征 dino_0/dino_1（若存在）；否则在线对原始图像跑 DINO。
        # 训练在线：image_0/image_1；eval：policy server 给 `image`（当前观测，等价 image_0）。
        dino_0 = self._stack_field(examples, "dino_0", required=False)
        if dino_0 is None:
            dino_0 = self._run_dino_on_images(examples, ["image_0", "image"], required=True)
        dino_1 = self._stack_field(examples, "dino_1", required=False)
        if dino_1 is None and require_future_dino:
            dino_1 = self._run_dino_on_images(examples, ["image_1"], required=True)
        batch = {
            "examples": examples,
            "instructions": [str(ex.get("lang", "")) for ex in examples],
            "dino_0": dino_0,
            "dino_1": dino_1,
            "action": self._stack_field(examples, "action", required=require_action),
            "state": self._stack_field(examples, "state", required=False),
            "future_valid": self._stack_field(examples, "future_valid", required=False),
            "future_valid_steps": self._stack_field(examples, "future_valid_steps", required=False),
        }
        if batch["future_valid"] is None and require_future_dino:
            batch["future_valid"] = torch.ones(len(examples), device=self.device, dtype=torch.float32)
        return batch

    def _future_valid_mask(self, batch: dict, cond: torch.Tensor) -> torch.Tensor:
        valid = batch.get("future_valid", None)
        if valid is None:
            return torch.ones(cond.shape[0], device=cond.device, dtype=torch.bool)
        valid = valid.to(device=cond.device)
        return valid.reshape(valid.shape[0], -1)[:, 0] > 0.5

    def _assemble_sequence(self, task: str, batch: dict):
        examples = batch["examples"]
        text_embeds, text_valid_lens, _ = self.backbone.embed_text(batch["instructions"])
        target_dtype = text_embeds.dtype

        blocks: list[torch.Tensor] = [text_embeds]
        block_sizes: list[int] = [text_embeds.shape[1]]
        block_names: list[str] = ["text"]

        if self.state_inject_mode == "token":
            state_tokens = self.state_enc(
                batch["state"],
                batch_size=text_embeds.shape[0],
                device=text_embeds.device,
                dtype=torch.float32,
            ).to(dtype=target_dtype)
            blocks.append(state_tokens)
            block_sizes.append(state_tokens.shape[1])
            block_names.append("state")
        elif self.state_inject_mode != "none":
            raise ValueError(f"Unsupported state.inject_mode={self.state_inject_mode}")

        dino0 = self._flatten_dino(batch["dino_0"]).to(text_embeds.device, dtype=torch.float32)
        img0_tokens = self.dino_proj(dino0).to(dtype=target_dtype)
        blocks.append(img0_tokens)
        block_sizes.append(img0_tokens.shape[1])
        block_names.append("img0")

        action = batch["action"]
        if task == "fdm":
            if action is None:
                raise KeyError("fdm requires `action` context.")
            action_ctx = self.act_ctx(action[:, -self.action_horizon :, : self.action_dim]).to(dtype=target_dtype)
            blocks.append(action_ctx)
            block_sizes.append(action_ctx.shape[1])
            block_names.append("action_ctx")

        if task == "idm":
            if batch["dino_1"] is None:
                raise KeyError("idm requires `dino_1`.")
            dino1_ctx = self._flatten_dino(batch["dino_1"]).to(text_embeds.device, dtype=torch.float32)
            img1_tokens = self.dino_proj(dino1_ctx).to(dtype=target_dtype)
            blocks.append(img1_tokens)
            block_sizes.append(img1_tokens.shape[1])
            block_names.append("img1")

        query_start = sum(block_sizes)
        if task in {"policy", "idm"}:
            query = self.action_queries(text_embeds.shape[0], device=text_embeds.device).to(dtype=target_dtype)
            query_kind = "action_query"
        elif task in {"fdm", "passive"}:
            if batch["dino_1"] is None:
                raise KeyError(f"{task} requires `dino_1`.")
            target = self._select_future_dino(batch["dino_1"], examples)
            n_query = int(target.shape[1])
            query = self.future_dino_queries(
                text_embeds.shape[0],
                n_query=n_query,
                device=text_embeds.device,
            ).to(dtype=target_dtype)
            query_kind = "future_dino_query"
        else:
            raise ValueError(f"Unsupported JointFlow task `{task}`")

        blocks.append(query)
        block_sizes.append(query.shape[1])
        block_names.append(query_kind)
        query_slice = slice(query_start, query_start + query.shape[1])

        inputs_embeds = torch.cat(blocks, dim=1)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0).expand(inputs_embeds.shape[0], -1)
        attn4d = build_block_causal_mask(
            block_sizes=block_sizes,
            text_valid_lens=text_valid_lens,
            dtype=torch.float32,
            device=inputs_embeds.device,
            hybrid=bool(self.config.framework.tasks.get("hybrid_mask", True)),
            neg_value=float(self.config.framework.tasks.get("attention_mask_neg_value", -1.0e4)),
        )
        return inputs_embeds, attn4d, position_ids, query_slice, block_names, block_sizes

    def forward(self, examples: List[dict] = None, task: str = "policy", **kwargs) -> dict:
        batch = self._examples_to_batch(
            examples,
            require_future_dino=task in {"fdm", "idm", "passive"},
            require_action=task in {"policy", "fdm", "idm"},
        )
        self._assert_finite("batch.dino_0", batch.get("dino_0"), task, batch["examples"])
        self._assert_finite("batch.dino_1", batch.get("dino_1"), task, batch["examples"])
        self._assert_finite("batch.action", batch.get("action"), task, batch["examples"])
        self._assert_finite("batch.state", batch.get("state"), task, batch["examples"])
        inputs_embeds, attn4d, position_ids, query_slice, _, _ = self._assemble_sequence(task, batch)
        self._assert_finite("inputs_embeds", inputs_embeds, task, batch["examples"])
        self._assert_finite("attention_mask_4d", attn4d, task, batch["examples"])
        hidden = self.backbone(inputs_embeds=inputs_embeds, attention_mask_4d=attn4d, position_ids=position_ids)
        self._assert_finite("backbone.hidden", hidden, task, batch["examples"])
        cond = hidden[:, query_slice].float()
        self._assert_finite("cond", cond, task, batch["examples"])

        if task in {"policy", "idm"}:
            actions = batch["action"]
            if actions is None:
                raise KeyError(f"{task} requires `action` labels.")
            target = actions[:, -self.action_horizon :, : self.action_dim].float()
            self._assert_finite("target.action", target, task, batch["examples"])
            if task == "idm":
                valid_mask = self._future_valid_mask(batch, cond)
                if not bool(valid_mask.any()):
                    # 中文注释：本 rank 整个 batch 的 future 都无效 → 不调用 action_head。
                    # 必须用 full anchor（覆盖所有可训练参数，含 action_head），否则当
                    # 别的 rank 走了正常路径用到 action_head 时，两边 ready 的参数集合不一致 → DDP 挂死。
                    loss = cond.sum() * 0.0
                    return {f"{task}_loss": loss + self._all_trainable_anchor(loss)}
                cond = cond[valid_mask]
                target = target[valid_mask]
            # 中文注释：与 visual_head / backbone 保持一致——强制 action head 的 DiT 在 fp32 下计算。
            # accelerate(bf16) 默认会把整个 model.forward 包进 autocast，backbone 和 visual_head
            # 都显式 opt-out 跑 fp32，唯独 FlowmatchingActionHead 没有，导致 policy/idm 的 24 层
            # DiT 实际在 bf16 下跑，是数值尖刺/NaN 的高风险来源。这里本地 opt-out（不改共享的
            # GR00T_ActionHeader.py），让 policy/idm 与 fdm/passive 数值口径一致。
            device_type = cond.device.type
            ac = torch.autocast(device_type=device_type, enabled=False) if device_type in {"cuda", "cpu"} else nullcontext()
            with ac:
                loss = self.action_head(cond.float(), target.float(), state=None, encoder_attention_mask=None)
        else:
            target = self._select_future_dino(batch["dino_1"], batch["examples"]).to(cond.device, dtype=torch.float32)
            self._assert_finite("target.dino_1", target, task, batch["examples"])
            valid_mask = self._future_valid_mask(batch, cond)
            if not bool(valid_mask.any()):
                # 中文注释：fdm/passive 同理，本 rank future 全无效时不调用 visual_head。
                # 用 full anchor 保证所有可训练参数 ready，避免多卡 reduction 不一致挂死。
                loss = cond.sum() * 0.0
                return {f"{task}_loss": loss + self._all_trainable_anchor(loss)}
            cond = cond[valid_mask]
            target = target[valid_mask]
            loss = self.visual_head(cond, target)
        self._assert_finite(f"{task}.loss", loss, task, batch["examples"])
        return {f"{task}_loss": loss + self._unused_task_param_anchor(task, loss)}

    def compute_loss(self, tag: str, batch, loss_scale: dict = None):
        if tag != "vla":
            return None
        task = getattr(self, "_current_task", "policy")
        out = self.forward(batch, task=task)
        scale = (loss_scale or {}).get(tag, 1.0)
        return {k: v * scale for k, v in out.items()}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        batch = self._examples_to_batch(examples, require_future_dino=False, require_action=False)
        inputs_embeds, attn4d, position_ids, query_slice, _, _ = self._assemble_sequence("policy", batch)
        hidden = self.backbone(inputs_embeds=inputs_embeds, attention_mask_4d=attn4d, position_ids=position_ids)
        cond = hidden[:, query_slice].float()
        pred_actions = self.action_head.predict_action(cond, state=None, encoder_attention_mask=None)
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
######### // code // ##########


######### // code // ##########
# 中文注释：模块自检入口。加载真实 Qwen，使用假 DINO/action/state 跑 4 个 task
# forward 和 predict_action 形状检查。
def _make_fake_batch(batch_size: int, views: int, n_tokens: int, d_dino: int, action_horizon: int, action_dim: int):
    batch = []
    for idx in range(batch_size):
        batch.append(
            {
                "dino_0": np.random.randn(views, n_tokens, d_dino).astype("float32"),
                "dino_1": np.random.randn(views, n_tokens, d_dino).astype("float32"),
                "action": np.random.uniform(-1, 1, size=(action_horizon, action_dim)).astype("float32"),
                "state": np.random.uniform(-1, 1, size=(1, 8)).astype("float32"),
                "lang": f"fake LIBERO instruction {idx}",
                "dino_view_keys": ["video.primary_image", "video.wrist_image"][:views],
            }
        )
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/jointflow/configs/jointflow_libero.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.name = "QwenJointFlow"
    cfg.framework.dino.load_live_backbone = False
    cfg.framework.action_model.diffusion_model_cfg.num_layers = 1
    cfg.framework.visual_model.num_layers = 1

    model = QwenJointFlowVLA(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    n_tokens = int(cfg.framework.visual_model.n_query)
    batch = _make_fake_batch(
        batch_size=2,
        views=2,
        n_tokens=n_tokens,
        d_dino=int(cfg.framework.visual_model.d_dino),
        action_horizon=int(cfg.framework.action_model.action_horizon),
        action_dim=int(cfg.framework.action_model.action_dim),
    )
    for task_name in ["policy", "fdm", "idm", "passive"]:
        out = model(batch, task=task_name)
        print(task_name, {k: float(v.detach().cpu()) for k, v in out.items()})
    pred = model.predict_action(batch)["normalized_actions"]
    print("predict_action", pred.shape)
######### // code // ##########
