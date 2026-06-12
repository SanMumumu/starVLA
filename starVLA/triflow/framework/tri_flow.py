"""PR2/PR3: TriFlow framework — 单塔全生成式三模态 (V/L/A) 扩散模型.

复用:
- starVLA.model.framework.base_framework.baseframework / FRAMEWORK_REGISTRY
- starVLA.model.framework.share_tools.merge_framework_config
- starVLA.jointflow.modules.dino_v3.DINOv3Backbone (eval 时 live 提特征, 只 import)
- starVLA.jointflow.data.mix_registry.resolve_data_mix (DINO stats 定位, 只 import)
- Third_github/ELF 训练/采样配方 (经 starVLA.triflow.modules.flow 移植)

设计 (用户硬约束):
- 零 LLM / 零 VLM / 零预训练语言权重; state 完全不输入。
- 一条序列 [v | l | a | vf] 进 from-scratch 单塔; 干净块=条件(t=1), 加噪块=目标(各自 t);
  任务 = config 驱动的 (cond, target) 噪声模式 (policy/v2l/fdm/passive/idm/joint/...)。
- mask: 噪声块看一切, 干净块只看干净块; 文本 pad 列全程屏蔽。
- loss: 各目标块 速度MSE 按 (有效token×维度) 取均值 → 模态权重加权求和;
  L 目标另有 ELF 解码分支 (decoder_step=True 时 CE, tied unembedding)。
- DDP: 一步一任务; 未参与本步计算图的组件挂 zero-grad 锚点 (full coverage,
  v1 血泪教训; 覆盖正确性由 scripts/smoke_test.py 的梯度覆盖断言保证)。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from starVLA.jointflow.data.mix_registry import resolve_data_mix
from starVLA.jointflow.modules.dino_v3 import DINOv3Backbone
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.triflow.data.text_vocab import build_codec_from_cfg
from starVLA.triflow.modules.attention_mask import build_clean_noisy_mask
from starVLA.triflow.modules.blocks import BlockEmbedder
from starVLA.triflow.modules.flow import (
    add_noise,
    masked_token_mean,
    ode_step,
    sample_decoder_lambda,
    sample_t_logit_normal,
    sde_renoise,
    uniform_t_grid,
    velocity_from_x0,
)
from starVLA.triflow.modules.tri_tower import TriFlowTower
from starVLA.triflow.tasks.registry import TaskSpec, build_task_registry


######### // code // ##########
# 中文注释：TriFlow 默认配置。YAML 的 framework 节点覆盖这些默认值 (merge_framework_config)。
@dataclass
class TriFlowDefaultConfig:
    name: str = "TriFlow"
    tower: dict = field(
        default_factory=lambda: {
            "hidden_size": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "qk_norm": True,
            "dropout": 0.0,
        }
    )
    text: dict = field(
        default_factory=lambda: {
            "tokenizer_path": "playground/Pretrained_models/Qwen2.5-0.5B",
            "vocab_mode": "compact",
            "vocab_map_path": "playground/Pretrained_models/triflow_assets/vocab_map_libero.json",
            "d_text": 256,
            "max_len": 32,
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
            "n_patches": 196,
            "max_views": 4,
            "future_view_keys": ["video.primary_image", "observation.images.image", "primary_image", "agentview"],
            "stats_path": None,
            "load_live_backbone": False,
        }
    )
    action_model: dict = field(default_factory=lambda: {"action_horizon": 8, "action_dim": 7})
    flow: dict = field(
        default_factory=lambda: {
            "t_eps": 0.05,
            "denoiser_p_mean": -1.5,
            "denoiser_p_std": 0.8,
            "noise_scale": {"vision": 2.0, "text": 2.0, "action": 1.0},
            "decoder_prob": 0.2,
            "decoder_p_mean": 0.8,
            "decoder_p_std": 0.8,
            "decoder_noise_scale": 2.0,
            "self_conditioning": False,
        }
    )
    losses: dict = field(default_factory=lambda: {"modality_weights": {"a": 1.0, "l": 1.0, "vf": 1.0}})
    sampling: dict = field(
        default_factory=lambda: {
            "method": "ode",
            "sde_gamma": 1.0,
            "num_steps": {"action": 16, "text": 32, "future": 32, "joint": 32},
        }
    )
    tasks: dict = field(
        default_factory=lambda: {
            "attention_mask_neg_value": -1.0e4,
            "specs": {
                "policy": {"cond": ["v", "l"], "target": ["a"]},
                "v2l": {"cond": ["v"], "target": ["l"]},
                "fdm": {"cond": ["v", "l", "a"], "target": ["vf"]},
                "passive": {"cond": ["v"], "target": ["vf"]},
                "idm": {"cond": ["v", "vf"], "target": ["a"]},
                "joint": {"cond": ["v", "l"], "target": ["a", "vf"]},
            },
            "weights": {"policy": 1.0, "v2l": 0.5, "fdm": 0.5, "passive": 0.3, "idm": 0.5, "joint": 0.5},
        }
    )
    ema: dict = field(default_factory=lambda: {"enabled": True, "decay": 0.9999})
######### // code // ##########


# 中文注释：目标块 → 模态噪声尺度 / 模态权重 / 采样步数键 的映射表。
_KIND_TO_NOISE_KEY = {"v": "vision", "vf": "vision", "l": "text", "a": "action"}


######### // code // ##########
# 中文注释：TriFlow 主模型。forward 每步只跑一个任务 (decoder_step 由 trainer 广播决定)。
@FRAMEWORK_REGISTRY.register("TriFlow")
class TriFlowVLA(baseframework):
    def __init__(self, config=None, **kwargs) -> None:
        super().__init__()
        # 中文注释：注册 triflow 命名混合（triflow_libero_all 等），必须早于任何
        # resolve_data_mix 调用（本类的 _default_dino_stats_paths / 外部 dataloader）。
        from starVLA.triflow.data.mixtures import register_triflow_mixtures

        register_triflow_mixtures()
        self.config = merge_framework_config(TriFlowDefaultConfig, config)
        fw = self.config.framework

        # --- v1 血泪教训硬断言：LIBERO lerobot action 已是 delta，必须 abs 透传 ---
        datasets_cfg = getattr(self.config, "datasets", None)
        vla_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
        if vla_cfg is not None:
            action_mode = str(vla_cfg.get("action_mode", "abs"))
            if action_mode != "abs":
                raise ValueError(
                    f"TriFlow requires datasets.vla_data.action_mode=abs (got `{action_mode}`). "
                    "LIBERO lerobot actions are already delta; re-differencing breaks eval (v1 0%-SR bug)."
                )

        # --- 文本编解码器（只用分词文件 + vocab_map；零预训练权重） ---
        env_vocab = os.environ.get("TRIFLOW_VOCAB_MAP", "").strip()
        if env_vocab:
            fw.text.vocab_map_path = env_vocab
        self.codec = build_codec_from_cfg(fw)

        tower_cfg = fw.tower
        dino_cfg = fw.dino
        self.hidden_size = int(tower_cfg.get("hidden_size", 768))
        self.action_horizon = int(fw.action_model.get("action_horizon", 8))
        self.action_dim = int(fw.action_model.get("action_dim", 7))
        self.d_dino = int(dino_cfg.get("embed_dim", 384))
        self.n_patches = int(dino_cfg.get("n_patches", 196))

        # --- 模态块嵌入器 + 单塔 ---
        self.embedder = BlockEmbedder(
            vocab_size=int(self.codec.vocab_size),
            d_text=int(fw.text.get("d_text", 256)),
            max_text_len=int(fw.text.get("max_len", 32)),
            d_dino=self.d_dino,
            n_patches=self.n_patches,
            max_views=int(dino_cfg.get("max_views", 4)),
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            hidden_size=self.hidden_size,
        )
        self.tower = TriFlowTower(
            hidden_size=self.hidden_size,
            depth=int(tower_cfg.get("depth", 12)),
            num_heads=int(tower_cfg.get("num_heads", 12)),
            mlp_ratio=float(tower_cfg.get("mlp_ratio", 4.0)),
            qk_norm=bool(tower_cfg.get("qk_norm", True)),
            dropout=float(tower_cfg.get("dropout", 0.0)),
        )

        # --- 任务注册表（config 驱动） ---
        self.task_registry: dict[str, TaskSpec] = build_task_registry(fw.tasks)

        # --- DINO 统计（live eval 归一化用）与可选 live backbone ---
        self.dino = None
        if bool(dino_cfg.get("load_live_backbone", False)):
            self.dino = self._build_live_dino()
        self.register_buffer("_dino_mean", torch.zeros(self.d_dino), persistent=False)
        self.register_buffer("_dino_std", torch.ones(self.d_dino), persistent=False)
        self._dino_stats_source = None
        self._load_dino_stats(dino_cfg.get("stats_path", None))

    # ------------------------------------------------------------------
    # DDP 锚点：组件名 → 参数；按任务规格推导"本步用到的组件集合"，其余全部挂 0 梯度。
    # 正确性由 scripts/smoke_test.py 的梯度覆盖断言验证（每个 (task,branch) backward 后
    # 所有可训练参数都必须有 grad）。
    # ------------------------------------------------------------------
    def _component_map(self) -> dict[str, nn.Module | nn.Parameter]:
        emb = self.embedder
        return {
            "tower": self.tower,
            "E": emb.E,
            "in_v": emb.in_proj_v,
            "in_l": emb.in_proj_l,
            "in_a": emb.in_proj_a,
            "out_v": emb.out_proj_v,
            "out_l": emb.out_proj_l,
            "out_a": emb.out_proj_a,
            "dec_proj": emb.dec_proj,
            "dec_gain": emb.dec_gain,
            "pos_patch": emb.pos_patch,
            "view": emb.view_emb,
            "pos_text": emb.pos_text,
            "pos_action": emb.pos_action,
            "type": emb.type_emb,
            "time": emb.time_embedder,
            "stride": emb.stride_embedder,
        }

    @staticmethod
    def used_components(spec: TaskSpec, decoder_step: bool) -> set[str]:
        used = {"tower", "type", "time"}
        for kind in spec.blocks:
            if kind == "v":
                used |= {"in_v", "pos_patch", "view"}
            elif kind == "vf":
                used |= {"in_v", "pos_patch", "stride"}
            elif kind == "l":
                used |= {"E", "in_l", "pos_text"}
            elif kind == "a":
                used |= {"in_a", "pos_action"}
        for kind in spec.target:
            if kind == "a":
                used.add("out_a")
            elif kind == "vf":
                used.add("out_v")
            elif kind == "l":
                if decoder_step:
                    used |= {"dec_proj", "dec_gain"}  # tied 表 E 已在输入侧计入
                else:
                    used.add("out_l")
        return used

    @staticmethod
    def _params_of(component: nn.Module | nn.Parameter):
        if isinstance(component, nn.Parameter):
            return [component]
        return list(component.parameters())

    def _unused_param_anchor(self, spec: TaskSpec, decoder_step: bool, ref: torch.Tensor) -> torch.Tensor:
        used = self.used_components(spec, decoder_step)
        anchor = ref.new_zeros(())
        seen: set[int] = set()
        for name, component in self._component_map().items():
            if name in used:
                continue
            for param in self._params_of(component):
                if not param.requires_grad or param.numel() == 0 or id(param) in seen:
                    continue
                seen.add(id(param))
                anchor = anchor + param.reshape(-1)[0].to(anchor.dtype) * 0.0
        return anchor

    # ------------------------------------------------------------------
    # DINO stats（v1 同款：单文件或按 data_mix 合并多数据集统计）
    # ------------------------------------------------------------------
    def _read_dino_stats_file(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _default_dino_stats_paths(self) -> list[Path]:
        datasets_cfg = getattr(self.config, "datasets", None)
        vla_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
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
            stats = self._combine_dino_stats([self._read_dino_stats_file(p) for p in paths])
            source = ",".join(p.as_posix() for p in paths) if paths else None
            if stats is None:
                return
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32).clamp_min(1e-6)
        if mean.numel() != self.d_dino or std.numel() != self.d_dino:
            raise ValueError(f"DINO stats dim mismatch: {mean.numel()} vs expected {self.d_dino}")
        self._dino_mean.copy_(mean)
        self._dino_std.copy_(std)
        self._dino_stats_source = source

    # ------------------------------------------------------------------
    # batch 构造（v1 同款模式；无 state）
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _stack_field(self, examples: List[dict], key: str, required: bool = True) -> torch.Tensor | None:
        vals = [ex.get(key, None) for ex in examples]
        if vals[0] is None:
            if required:
                raise KeyError(f"Missing required field `{key}` in TriFlow batch.")
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
            return dino.reshape(bsz, views * n_tokens, dim)
        if dino.ndim != 3:
            raise ValueError(f"Expected DINO features [B,V,N,D] or [B,N,D], got {tuple(dino.shape)}")
        return dino

    def _build_live_dino(self) -> DINOv3Backbone:
        dino_cfg = self.config.framework.dino
        return DINOv3Backbone(
            name=dino_cfg.get("name", "dinov3_vits16"),
            hf_model_id=dino_cfg.get("hf_model_id", "facebook/dinov3-vits16-pretrain-lvd1689m"),
            repo_or_dir=dino_cfg.get("repo_or_dir", "facebookresearch/dinov3"),
            weights=dino_cfg.get("weights", None),
            loader=dino_cfg.get("loader", "auto"),
            image_size=int(dino_cfg.get("image_size", 224)),
            patch_size=int(dino_cfg.get("patch_size", 16)),
            embed_dim=int(dino_cfg.get("embed_dim", 384)),
        )

    def _normalize_live_dino(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self._dino_mean.to(z.device, z.dtype)) / self._dino_std.to(z.device, z.dtype)

    def _live_dino_0(self, examples: List[dict]) -> torch.Tensor:
        if self.dino is None:
            self.dino = self._build_live_dino().to(self.device)
        images = []
        for ex in examples:
            imgs = ex.get("image", None)
            if imgs is None:
                raise KeyError("TriFlow inference requires either `dino_0` or live `image` inputs.")
            if isinstance(imgs, Image.Image):
                imgs = [imgs]
            images.append(imgs)
        tensor = self.dino.prepare_dino_input(images)
        feats = self.dino(tensor)
        bsz = len(images)
        views = len(images[0])
        feats = feats.view(bsz, views, feats.shape[1], feats.shape[2])
        return self._normalize_live_dino(feats.float())

    def _examples_to_batch(self, examples: List[dict], spec: TaskSpec, for_inference: bool = False) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        need_action = "a" in spec.target or "a" in spec.cond
        # 中文注释：dino_1 必需性——vf 作条件(idm)总是需要；vf 作目标只在训练时需要（监督标签），
        # 推理时 vf 由噪声生成，不需要 dino_1。
        need_future = ("vf" in spec.cond) or ((not for_inference) and "vf" in spec.target)

        batch: dict = {"examples": examples}
        batch["instructions"] = [str(ex.get("lang", "")) for ex in examples]
        batch["dino_0"] = self._stack_field(examples, "dino_0", required=False)
        if batch["dino_0"] is None:
            batch["dino_0"] = self._live_dino_0(examples)
        batch["dino_1"] = self._stack_field(examples, "dino_1", required=need_future)
        # 中文注释：action 必需性与 dino_1 同理——`a` 作目标只在训练需要（监督标签），
        # 推理时由噪声生成（policy/idm/joint 的 client 不发 action）；`a` 作条件（fdm）
        # 则训练/推理都必须提供。
        required_action = ("a" in spec.target) and (not for_inference)
        batch["action"] = self._stack_field(examples, "action", required=required_action)
        if batch["action"] is None and "a" in spec.cond:
            raise KeyError(f"task `{spec.name}` requires `action` as condition input.")
        batch["future_valid"] = self._stack_field(examples, "future_valid", required=False)
        batch["future_valid_steps"] = self._stack_field(examples, "future_valid_steps", required=False)
        batch["future_stride"] = self._stack_field(examples, "future_stride", required=False)
        return batch

    # ------------------------------------------------------------------
    # 干净 latent / 序列组装
    # ------------------------------------------------------------------
    def _clean_latents(self, batch: dict, spec: TaskSpec) -> dict:
        """各模态干净 latent x0 + 文本辅助量（在归一空间）。"""
        bsz = batch["dino_0"].shape[0]
        device = self.device
        out: dict = {"x0": {}, "bsz": bsz}

        present = set(spec.blocks)
        if "v" in present:
            out["x0"]["v"] = self._flatten_dino(batch["dino_0"]).float()
        if "l" in present:
            ids, lens = self.codec.encode_batch(batch["instructions"])
            ids = ids.to(device)
            lens = lens.to(device)
            out["text_ids"] = ids
            out["text_lens"] = lens
            out["x0"]["l"] = self.embedder.embed_text_x0(ids)
        if "a" in present:
            action = batch["action"]
            # 中文注释：a 作目标且在推理时（client 不发 action）没有标签 → 不构造 x0，
            # 目标块由噪声生成；a 作条件（fdm）时 _examples_to_batch 已保证非 None。
            if action is not None:
                out["x0"]["a"] = action[:, -self.action_horizon :, : self.action_dim].float()
        if "vf" in present:
            if batch["dino_1"] is not None:
                out["x0"]["vf"] = self._select_future_dino(batch["dino_1"], batch["examples"]).float()
            steps = batch.get("future_valid_steps", None)
            stride = batch.get("future_stride", None)
            if steps is not None and stride is not None:
                out["stride01"] = (steps.reshape(bsz) / stride.reshape(bsz).clamp_min(1.0)).clamp(0.0, 1.0)
            else:
                # 中文注释：推理默认预测整 stride 之后的未来帧
                out["stride01"] = torch.ones(bsz, device=device)
        valid = batch.get("future_valid", None)
        out["future_valid"] = (
            valid.reshape(bsz, -1)[:, 0] > 0.5 if valid is not None else torch.ones(bsz, device=device, dtype=torch.bool)
        )
        return out

    def _assemble(
        self,
        spec: TaskSpec,
        latents: dict[str, torch.Tensor],
        t_map: dict[str, torch.Tensor],
        clean: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, slice]]:
        """latents: 各块进塔的 latent（条件=干净 x0，目标=加噪 z）。返回 tokens/mask/slices。"""
        bsz = clean["bsz"]
        blocks, sizes, slices = [], [], {}
        text_block_idx = None
        offset = 0
        for idx, kind in enumerate(spec.blocks):
            tok = self.embedder.embed_block(
                kind,
                latents[kind],
                t_map[kind],
                stride01=clean.get("stride01", None) if kind == "vf" else None,
            )
            blocks.append(tok)
            sizes.append(tok.shape[1])
            slices[kind] = slice(offset, offset + tok.shape[1])
            offset += tok.shape[1]
            # 中文注释：只有 L 作为"条件"时才屏蔽 pad 列；L 作为"目标"时是全长生成画布
            # （ELF 默认 loss_mask=ones 的同款做法：模型学会 EOS 后输出 PAD，
            #  推理无需已知长度，且训练/推理注意力分布一致）。
            if kind == "l" and kind not in spec.target:
                text_block_idx = idx

        tokens = torch.cat(blocks, dim=1)
        mask = build_clean_noisy_mask(
            block_sizes=sizes,
            noisy_flags=list(spec.noisy_flags),
            batch_size=bsz,
            dtype=tokens.dtype,
            device=tokens.device,
            neg_value=float(self.config.framework.tasks.get("attention_mask_neg_value", -1.0e4)),
            text_block_idx=text_block_idx,
            text_valid_lens=clean.get("text_lens", None),
        )
        return tokens, mask, slices

    # ------------------------------------------------------------------
    # 训练 forward
    # ------------------------------------------------------------------
    def _target_token_mask(self, kind: str, spec: TaskSpec, clean: dict, n_tokens: int) -> torch.Tensor:
        """每个目标块的逐 token loss mask [B,n]。"""
        bsz = clean["bsz"]
        device = self.device
        if kind == "l":
            # 中文注释：L 目标块全长算 loss（含 EOS 后的 PAD 位置）——
            # 与 _assemble 的"目标块不做 pad 屏蔽"配套，让模型学会自己结束句子。
            return torch.ones(bsz, n_tokens, device=device)
        if kind == "vf":
            return clean["future_valid"].float().reshape(-1, 1).expand(bsz, n_tokens).clone()
        if kind == "a":
            # idm：vf 为条件且 future 无效时，该样本的 (V,V') 对退化（dino_1==dino_0），动作目标剔除
            if "vf" in spec.cond:
                return clean["future_valid"].float().reshape(-1, 1).expand(bsz, n_tokens).clone()
            return torch.ones(bsz, n_tokens, device=device)
        raise ValueError(f"Unexpected target kind `{kind}`")

    def forward(self, examples: List[dict] = None, task: str = "policy", decoder_step: bool = False, **kwargs) -> dict:
        if task not in self.task_registry:
            raise ValueError(f"Unknown TriFlow task `{task}`. Known: {sorted(self.task_registry)}")
        spec = self.task_registry[task]
        if decoder_step and "l" not in spec.target:
            raise ValueError(f"decoder_step=True only valid when `l` is a target (task={task})")

        flow_cfg = self.config.framework.flow
        t_eps = float(flow_cfg.get("t_eps", 0.05))
        batch = self._examples_to_batch(examples, spec)
        clean = self._clean_latents(batch, spec)
        bsz = clean["bsz"]
        device = self.device

        # --- 加噪：条件块 t=1 干净；目标块各自采样 t（解码分支的 L 块按 λ 噪声、t=1） ---
        latents: dict[str, torch.Tensor] = {}
        t_map: dict[str, torch.Tensor] = {}
        z_map: dict[str, torch.Tensor] = {}
        noise_map: dict[str, torch.Tensor] = {}
        clean_t = torch.ones(bsz, device=device)
        for kind in spec.blocks:
            x0 = clean["x0"][kind]
            if kind not in spec.target:
                latents[kind] = x0
                t_map[kind] = clean_t
                continue
            noise_scale = float(flow_cfg.noise_scale.get(_KIND_TO_NOISE_KEY[kind], 1.0))
            if kind == "l" and decoder_step:
                lam = sample_decoder_lambda(
                    bsz, x0.shape[1], float(flow_cfg.get("decoder_p_mean", 0.8)),
                    float(flow_cfg.get("decoder_p_std", 0.8)), device,
                )
                eps = torch.randn_like(x0) * float(flow_cfg.get("decoder_noise_scale", 2.0))
                z = lam * x0.detach() + (1.0 - lam) * eps
                t_kind = clean_t
            else:
                t_kind = sample_t_logit_normal(
                    bsz, float(flow_cfg.get("denoiser_p_mean", -1.5)), float(flow_cfg.get("denoiser_p_std", 0.8)), device
                )
                noise = torch.randn_like(x0)
                z = add_noise(x0, noise, t_kind, noise_scale)
                noise_map[kind] = noise
            latents[kind] = z
            t_map[kind] = t_kind
            z_map[kind] = z

        tokens, mask, slices = self._assemble(spec, latents, t_map, clean)
        hidden = self.tower(tokens, mask)

        # --- 各目标块 loss ---
        modality_weights = self.config.framework.losses.get("modality_weights", {})
        total = tokens.new_zeros(())
        metrics: dict = {}
        for kind in spec.target:
            h_k = hidden[:, slices[kind]]
            token_mask = self._target_token_mask(kind, spec, clean, h_k.shape[1])
            if kind == "l" and decoder_step:
                logits = self.embedder.decoder_logits(h_k)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), clean["text_ids"].reshape(-1), reduction="none"
                ).reshape(bsz, -1)
                block_loss = masked_token_mean(ce, token_mask)
                metrics[f"{task}/ce"] = float(block_loss.detach())
            else:
                x0_pred = self.embedder.project_out(kind, h_k)
                z = z_map[kind]
                t_kind = t_map[kind]
                v_pred = velocity_from_x0(x0_pred, z, t_kind, t_eps)
                # 中文注释：回归目标整体 stop-grad —— 防止 E 表通过"把目标拉向预测"走捷径塌缩
                v_tgt = velocity_from_x0(clean["x0"][kind].detach(), z.detach(), t_kind, t_eps)
                per_token = (v_pred - v_tgt).pow(2).mean(dim=-1)
                block_loss = masked_token_mean(per_token, token_mask)
                metrics[f"{task}/v_mse_{kind}"] = float(block_loss.detach())
            total = total + float(modality_weights.get(kind, 1.0)) * block_loss

        anchor = self._unused_param_anchor(spec, decoder_step, total)
        out = {f"{task}_loss": total + anchor}
        out.update(metrics)
        return out

    def compute_loss(self, tag: str, batch, loss_scale: dict = None):
        if tag != "vla":
            return None
        task = getattr(self, "_current_task", "policy")
        decoder_step = bool(getattr(self, "_current_decoder_step", False))
        out = self.forward(batch, task=task, decoder_step=decoder_step)
        scale = (loss_scale or {}).get(tag, 1.0)
        return {k: v * scale for k, v in out.items() if torch.is_tensor(v)}

    # ------------------------------------------------------------------
    # 采样（Euler ODE，可选 SDE 回噪）；条件块独立成块，永不被重新加噪。
    # ------------------------------------------------------------------
    def _num_steps_for(self, spec: TaskSpec, num_steps: int | None) -> int:
        if num_steps is not None:
            return int(num_steps)
        steps_cfg = self.config.framework.sampling.get("num_steps", {})
        targets = set(spec.target)
        if len(targets) > 1:
            return int(steps_cfg.get("joint", 32))
        key = {"a": "action", "l": "text", "vf": "future"}[next(iter(targets))]
        return int(steps_cfg.get(key, 32))

    @torch.inference_mode()
    def _generate(self, task: str, examples: List[dict], num_steps: int | None = None) -> dict:
        spec = self.task_registry[task]
        flow_cfg = self.config.framework.flow
        sampling_cfg = self.config.framework.sampling
        t_eps = float(flow_cfg.get("t_eps", 0.05))
        method = str(sampling_cfg.get("method", "ode"))
        gamma = float(sampling_cfg.get("sde_gamma", 1.0))

        batch = self._examples_to_batch(examples, spec, for_inference=True)
        clean = self._clean_latents(batch, spec)
        bsz = clean["bsz"]
        device = self.device
        clean_t = torch.ones(bsz, device=device)

        # 目标块从纯噪声出发（t=0: z = ε·noise_scale）；vf 推理无 dino_1，形状由 n_patches 决定
        target_shapes = {
            "a": (bsz, self.action_horizon, self.action_dim),
            "l": (bsz, int(self.codec.max_len), int(self.embedder.d_text)),
            "vf": (bsz, self.n_patches, self.d_dino),
        }
        scale = {k: float(flow_cfg.noise_scale.get(_KIND_TO_NOISE_KEY[k], 1.0)) for k in spec.target}
        z = {k: torch.randn(target_shapes[k], device=device) * scale[k] for k in spec.target}

        steps = self._num_steps_for(spec, num_steps)
        grid = uniform_t_grid(steps, device)

        def forward_x0(z_now: dict, t_now: float) -> dict:
            latents, t_map = {}, {}
            for kind in spec.blocks:
                if kind in spec.target:
                    latents[kind] = z_now[kind]
                    t_map[kind] = torch.full((bsz,), float(t_now), device=device)
                else:
                    latents[kind] = clean["x0"][kind]
                    t_map[kind] = clean_t
            tokens, mask, slices = self._assemble(spec, latents, t_map, clean)
            hidden = self.tower(tokens, mask)
            return {k: self.embedder.project_out(k, hidden[:, slices[k]]) for k in spec.target}, hidden, slices

        for i in range(steps):
            t_cur, t_next = float(grid[i]), float(grid[i + 1])
            if method == "sde" and 0.0 < t_cur < 1.0:
                t_back_map = {}
                for kind in spec.target:
                    z[kind], t_back = sde_renoise(z[kind], t_cur, t_next, gamma, scale[kind])
                    t_back_map[kind] = t_back
                t_eval = t_back_map[next(iter(spec.target))]
            else:
                t_eval = t_cur
            x0_pred, _, _ = forward_x0(z, t_eval)
            t_tensor = torch.full((bsz,), float(t_eval), device=device)
            for kind in spec.target:
                v = velocity_from_x0(x0_pred[kind], z[kind], t_tensor, t_eps)
                z[kind] = z[kind] + (t_next - t_eval) * v

        result: dict = {"z": z, "clean": clean, "spec": spec}
        # L 目标：t=1 处单次 forward 过解码头 → argmax（ELF _dlm_decode_batch）
        if "l" in spec.target:
            latents, t_map = {}, {}
            for kind in spec.blocks:
                latents[kind] = z[kind] if kind in spec.target else clean["x0"][kind]
                t_map[kind] = clean_t
            tokens, mask, slices = self._assemble(spec, latents, t_map, clean)
            hidden = self.tower(tokens, mask)
            logits = self.embedder.decoder_logits(hidden[:, slices["l"]])
            result["text_token_ids"] = torch.argmax(logits, dim=-1)
        return result

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        out = self._generate("policy", examples, num_steps=kwargs.get("num_steps", None))
        return {"normalized_actions": out["z"]["a"].detach().float().cpu().numpy()}

    @torch.inference_mode()
    def predict_language(self, examples: List[dict], **kwargs) -> dict:
        out = self._generate("v2l", examples, num_steps=kwargs.get("num_steps", None))
        ids = out["text_token_ids"].detach().cpu()
        return {"texts": self.codec.decode_ids(ids), "token_ids": ids.numpy()}

    @torch.inference_mode()
    def predict_future(self, examples: List[dict], task: str | None = None, **kwargs) -> dict:
        if task is None:
            has_action = examples[0].get("action", None) is not None and "fdm" in self.task_registry
            task = "fdm" if has_action else "passive"
        out = self._generate(task, examples, num_steps=kwargs.get("num_steps", None))
        z_norm = out["z"]["vf"].detach().float()
        feats = z_norm * self._dino_std.reshape(1, 1, -1) + self._dino_mean.reshape(1, 1, -1)
        return {
            "future_features": feats.cpu().numpy(),
            "normalized_features": z_norm.cpu().numpy(),
            "task": task,
        }

    @torch.inference_mode()
    def predict_joint(self, examples: List[dict], **kwargs) -> dict:
        out = self._generate("joint", examples, num_steps=kwargs.get("num_steps", None))
        z_norm = out["z"]["vf"].detach().float()
        feats = z_norm * self._dino_std.reshape(1, 1, -1) + self._dino_mean.reshape(1, 1, -1)
        return {
            "normalized_actions": out["z"]["a"].detach().float().cpu().numpy(),
            "future_features": feats.cpu().numpy(),
            "normalized_features": z_norm.cpu().numpy(),
        }
######### // code // ##########
