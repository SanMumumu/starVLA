"""Probe whether JointFlow FDM predictions actually use action context.

This script compares future-DINO predictions under matched observations with:

- the correct action chunk (`fdm`)
- no action context (`passive`)
- zeroed action chunk (`fdm`)
- negated/reversed action chunk (`fdm`)
- shuffled action chunk from another sample (`fdm`)

All comparisons are done in the normalized DINO feature space used for training.
For prediction comparisons, every variant in a repeat starts from the same
diffusion noise so pairwise differences mostly reflect conditioning changes,
not sampling randomness.

python tools/debug_action_causality.py \
    --ckpt_path ../../Z_ws/jointflow_libero_object/checkpoints_all/steps_5000_pytorch_model.pt \
    --num_samples 32 \
    --batch_size 4 \
    --save_visuals \
    --save_rae_rgb \
    --output_dir /mnt/hwdata/wangsen/starVLA/starVLA/starVLA/jointflow/tools/output5K \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def _ensure_starvla_importable() -> None:
    """Allow running as `python tools/debug_action_causality.py` from this dir."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "starVLA"
        if (candidate / "jointflow").is_dir() and (candidate / "model").is_dir():
            sys.path.insert(0, str(parent))
            return


_ensure_starvla_importable()

from starVLA.jointflow.data.joint_dataset import build_joint_dataset  # noqa: E402
from starVLA.jointflow.data.mix_registry import register_jointflow_mixtures  # noqa: E402
import starVLA.jointflow.framework.qwen_joint_flow  # noqa: F401,E402 - register framework
from starVLA.model.framework.base_framework import build_framework  # noqa: E402
from starVLA.model.framework.share_tools import apply_config_compat  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _jointflow_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(raw: str | None, *, must_exist: bool = True) -> Path | None:
    if raw is None:
        return None
    raw = os.path.expandvars(os.path.expanduser(str(raw)))
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        cwd = Path.cwd()
        candidates.append(cwd / path)
        candidates.extend(parent / path for parent in cwd.parents)
        root = _repo_root()
        candidates.append(root / path)
        candidates.extend(parent / path for parent in root.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        checked = ", ".join(str(c) for c in candidates[:6])
        raise FileNotFoundError(f"Path not found: {raw!r}. Checked: {checked}")
    return path.resolve()


def _default_output_dir(ckpt_path: Path, args: argparse.Namespace) -> Path:
    stem = ckpt_path.stem.replace("_pytorch_model", "")
    name = f"{stem}_action_probe_seed{int(args.seed)}_n{int(args.num_samples)}"
    return (_jointflow_dir() / "tools" / "action_causality_outputs" / name).resolve()


def _sanitize_name(value: str) -> str:
    out = []
    for ch in str(value):
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "".join(out).strip("_") or "item"


def _infer_dino_token_count(ex: dict) -> int | None:
    value = ex.get("dino_1", None)
    if value is None:
        value = ex.get("dino_0", None)
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim < 2:
        return None
    return int(arr.shape[-2])


def _infer_rae_image_size(examples: list[dict], patch_size: int, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    for ex in examples:
        n_tokens = _infer_dino_token_count(ex)
        if n_tokens is None:
            continue
        side = int(round(n_tokens ** 0.5))
        if side * side == n_tokens:
            return side * int(patch_size)
    return 224


def _rae_stats_grid_shape(decoder_dir: Path) -> tuple[int, int] | None:
    stats_path = decoder_dir / "stats.pt"
    if not stats_path.exists():
        return None
    try:
        stats = torch.load(str(stats_path), map_location="cpu", weights_only=False)
        mean = stats.get("mean", None) if isinstance(stats, dict) else None
        if torch.is_tensor(mean) and mean.ndim == 3:
            return int(mean.shape[-2]), int(mean.shape[-1])
    except Exception:
        return None
    return None


def _resolve_if_exists(raw) -> Path | None:
    if raw is None:
        return None
    try:
        return _resolve_path(str(raw), must_exist=True)
    except FileNotFoundError:
        return None


def _config_score(path: Path) -> int:
    try:
        cfg = OmegaConf.load(str(path))
    except Exception:
        return -1
    score = 0
    data_root = OmegaConf.select(cfg, "datasets.vla_data.data_root_dir", default=None)
    base_vlm = OmegaConf.select(cfg, "framework.qwenvl.base_vlm", default=None)
    if _resolve_if_exists(data_root) is not None:
        score += 4
    if _resolve_if_exists(base_vlm) is not None:
        score += 2
    if OmegaConf.select(cfg, "framework.visual_model", default=None) is not None:
        score += 1
    if OmegaConf.select(cfg, "trainer", default=None) is not None:
        score += 1
    return score


def _default_config_for_ckpt(ckpt_path: Path) -> Path:
    run_dir = ckpt_path.parents[1]
    candidates = [
        run_dir / "config.full.yaml",
        run_dir / "config.yaml",
        _repo_root() / "jointflow" / "configs" / "jointflow_libero_object.yaml",
        _repo_root() / "starVLA" / "jointflow" / "configs" / "jointflow_libero_object.yaml",
        Path(__file__).resolve().parents[1] / "configs" / "jointflow_libero_object.yaml",
    ]
    existing = [path.resolve() for path in candidates if path.exists()]
    if existing:
        # Prefer configs already rewritten for the current machine. A complete
        # config with dead bucket paths is less useful for this local probe than
        # a run_dir/config.yaml that has valid data/model paths.
        return max(existing, key=_config_score)
    raise FileNotFoundError(
        "Could not infer config YAML. Pass --config_yaml explicitly. "
        f"Looked under run_dir={run_dir} and local configs/."
    )


def _relocate_cfg_path(cfg, dotted_key: str) -> None:
    value = OmegaConf.select(cfg, dotted_key, default=None)
    resolved = _resolve_if_exists(value)
    if resolved is not None:
        OmegaConf.update(cfg, dotted_key, str(resolved), force_add=True)


def _load_config(args: argparse.Namespace, ckpt_path: Path):
    config_path = _resolve_path(args.config_yaml) if args.config_yaml else _default_config_for_ckpt(ckpt_path)
    cfg = OmegaConf.load(str(config_path))
    apply_config_compat(cfg)
    if args.data_root_dir is not None:
        cfg.datasets.vla_data.data_root_dir = str(_resolve_path(args.data_root_dir))
    else:
        _relocate_cfg_path(cfg, "datasets.vla_data.data_root_dir")
    if args.data_mix is not None:
        cfg.datasets.vla_data.data_mix = str(args.data_mix)
    if args.dino_feature_dir is not None:
        cfg.datasets.vla_data.dino_feature_dir = str(args.dino_feature_dir)
    _relocate_cfg_path(cfg, "framework.qwenvl.base_vlm")

    # This probe consumes offline dino_0/dino_1 features from the dataset.
    cfg.framework.dino.load_live_backbone = False
    if args.num_inference_timesteps is not None:
        cfg.framework.visual_model.num_inference_timesteps = int(args.num_inference_timesteps)
    return cfg, config_path


def _normalize_checkpoint_state(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "module"):
            value = obj.get(key)
            if isinstance(value, dict) and value:
                if all(isinstance(k, str) for k in value.keys()):
                    return value
    return obj


def _strip_module_prefix_if_needed(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_model(cfg, ckpt_path: Path, device: torch.device, *, strict: bool):
    model = build_framework(cfg)
    obj = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = _strip_module_prefix_if_needed(_normalize_checkpoint_state(obj))
    incompatible = model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    return model, incompatible


class RAEDinoDecoder:
    """Decode JointFlow-normalized DINO tokens with a RAEv2 stage-1 decoder."""

    def __init__(
        self,
        *,
        repo_dir: Path,
        decoder_dir: Path,
        decoder_config: Path,
        device: torch.device,
        image_size: int = 224,
        patch_size: int = 16,
        dino_dim: int = 384,
        allow_token_interpolate: bool = False,
    ) -> None:
        src_dir = repo_dir / "src"
        if not src_dir.is_dir():
            raise FileNotFoundError(f"RAEv2 src directory not found: {src_dir}")
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from stage1.decoders import GeneralDecoder
        from transformers import AutoConfig

        decoder_ckpt = decoder_dir / "decoder.pt"
        if not decoder_ckpt.exists():
            raise FileNotFoundError(f"RAE decoder checkpoint not found: {decoder_ckpt}")

        cfg = AutoConfig.from_pretrained(str(decoder_config))
        cfg.hidden_size = int(dino_dim)
        cfg.patch_size = int(patch_size)
        cfg.image_size = int(image_size)
        num_patches = (int(image_size) // int(patch_size)) ** 2

        decoder = GeneralDecoder(cfg, num_patches=num_patches)
        state_dict = torch.load(str(decoder_ckpt), map_location="cpu", weights_only=False)
        decoder_state = decoder.state_dict()
        loadable = {}
        skipped_shape = []
        skipped_shape_keys = set()
        for key, value in state_dict.items():
            if key in decoder_state and tuple(decoder_state[key].shape) == tuple(value.shape):
                loadable[key] = value
            elif key in decoder_state:
                skipped_shape.append((key, tuple(value.shape), tuple(decoder_state[key].shape)))
                skipped_shape_keys.add(key)
        incompatible = decoder.load_state_dict(loadable, strict=False)
        if skipped_shape:
            skipped_text = ", ".join(f"{k}: ckpt{src}->model{dst}" for k, src, dst in skipped_shape[:4])
            print(f"[warn] RAE decoder skipped shape-mismatched keys: {skipped_text}", flush=True)
        missing_keys = [key for key in incompatible.missing_keys if key not in skipped_shape_keys]
        if missing_keys:
            print(f"[warn] RAE decoder missing keys: {len(missing_keys)}", flush=True)
        if incompatible.unexpected_keys:
            print(f"[warn] RAE decoder unexpected keys: {len(incompatible.unexpected_keys)}", flush=True)

        self.decoder = decoder.to(device).eval()
        self.device = device
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.dino_dim = int(dino_dim)
        self.num_patches = int(num_patches)
        self.allow_token_interpolate = bool(allow_token_interpolate)
        self.decoder_ckpt = decoder_ckpt
        self.stats_path = decoder_dir / "stats.pt"
        self.stats_grid_shape = _rae_stats_grid_shape(decoder_dir)
        # Per-channel raw-DINO mean/std the decoder was trained to consume. The
        # diffusion latents are stats-normalized during training and de-normalized
        # back to this scale right before decoding, so faithful reconstruction
        # requires inputs in *this* space, not JointFlow's normalization scale.
        self.latent_mean, self.latent_std = self._load_perchannel_stats(self.stats_path)

    def _load_perchannel_stats(self, stats_path: Path):
        if not stats_path.exists():
            return None, None
        try:
            stats = torch.load(str(stats_path), map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] failed to read RAE stats {stats_path}: {exc}", flush=True)
            return None, None
        if not isinstance(stats, dict):
            return None, None
        mean = stats.get("mean", None)
        var = stats.get("var", None)
        std = stats.get("std", None)
        if not torch.is_tensor(mean):
            return None, None
        if not torch.is_tensor(std):
            if torch.is_tensor(var):
                std = var.clamp_min(0).sqrt()
            else:
                return None, None
        # stats may be stored per-position (D, H, W); reduce to per-channel (D,).
        reduce_dims = tuple(range(1, mean.ndim))
        mean = mean.float().mean(dim=reduce_dims) if reduce_dims else mean.float()
        std = std.float().mean(dim=reduce_dims) if reduce_dims else std.float()
        if int(mean.numel()) != self.dino_dim:
            print(
                f"[warn] RAE stats channel dim {int(mean.numel())} != dino_dim {self.dino_dim}; "
                "skipping RAE-space denormalization.",
                flush=True,
            )
            return None, None
        return mean.to(self.device), std.to(self.device)

    @property
    def has_latent_stats(self) -> bool:
        return self.latent_mean is not None and self.latent_std is not None

    def denorm_to_decoder_space(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Map per-channel-standardized DINO tokens into the decoder's raw-DINO scale."""
        if not self.has_latent_stats:
            return z_norm
        mean = self.latent_mean.to(z_norm.device, dtype=torch.float32)
        std = self.latent_std.to(z_norm.device, dtype=torch.float32)
        return z_norm.float() * std + mean

    @torch.inference_mode()
    def decode_raw_tokens(self, raw_tokens: torch.Tensor, batch_size: int = 2) -> torch.Tensor:
        outs = []
        raw_tokens = raw_tokens.to(device=self.device, dtype=torch.float32)
        if raw_tokens.shape[1] != self.num_patches and not self.allow_token_interpolate:
            raise ValueError(
                f"RAE token count mismatch: got {raw_tokens.shape[1]} tokens, "
                f"decoder expects {self.num_patches} tokens for image_size={self.image_size}, "
                f"patch_size={self.patch_size}. Use --rae_image_size {int(round(raw_tokens.shape[1] ** 0.5)) * self.patch_size} "
                "or pass --rae_allow_token_interpolate to permit latent interpolation."
            )
        for start in range(0, raw_tokens.shape[0], int(batch_size)):
            chunk = raw_tokens[start : start + int(batch_size)]
            out = self.decoder(chunk, drop_cls_token=False).logits
            rgb = self.decoder.unpatchify(out).clamp(0, 1)
            outs.append(rgb.detach().cpu())
        return torch.cat(outs, dim=0)


def _as_numpy_action(action) -> np.ndarray:
    if torch.is_tensor(action):
        return action.detach().cpu().numpy().astype(np.float32)
    return np.asarray(action, dtype=np.float32)


def transform_action(action: np.ndarray, mode: str) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if mode == "negate":
        return -action
    if mode == "reverse_time":
        return action[::-1].copy()
    if mode == "negate_reverse_time":
        return (-action[::-1]).copy()
    raise ValueError(f"Unsupported opposite_mode={mode!r}")


def _replace_actions(examples: list[dict], make_action: Callable[[int, dict], np.ndarray]) -> list[dict]:
    out = []
    for idx, ex in enumerate(examples):
        new = dict(ex)
        new["action"] = np.ascontiguousarray(make_action(idx, ex), dtype=np.float32)
        out.append(new)
    return out


def _drop_actions(examples: list[dict]) -> list[dict]:
    out = []
    for ex in examples:
        new = dict(ex)
        new.pop("action", None)
        out.append(new)
    return out


def make_variants(examples: list[dict], opposite_mode: str) -> dict[str, tuple[str, list[dict]]]:
    actions = [_as_numpy_action(ex["action"]) for ex in examples]
    variants = {
        "correct_action": ("fdm", examples),
        "no_action_passive": ("passive", _drop_actions(examples)),
        "zero_action": ("fdm", _replace_actions(examples, lambda _idx, ex: np.zeros_like(_as_numpy_action(ex["action"])))),
        "opposite_action": (
            "fdm",
            _replace_actions(examples, lambda _idx, ex: transform_action(_as_numpy_action(ex["action"]), opposite_mode)),
        ),
    }
    if len(actions) > 1:
        shuffled = actions[1:] + actions[:1]
        variants["shuffled_action"] = ("fdm", _replace_actions(examples, lambda idx, _ex: shuffled[idx]))
    return variants


def _metadata(ex: dict) -> dict:
    keys = ("dataset_name", "trajectory_id", "base_index", "future_index", "future_valid_steps", "future_stride", "lang")
    row = {}
    for key in keys:
        if key not in ex:
            continue
        value = ex[key]
        if isinstance(value, np.generic):
            value = value.item()
        if key == "lang":
            value = str(value)
        row[key] = value
    return row


def collect_examples(dataset, *, num_samples: int, seed: int, start_index: int | None, include_invalid: bool) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = len(dataset)
    examples: list[dict] = []
    seen: set[int] = set()
    attempts = 0
    max_attempts = max(num_samples * 80, 1000)
    while len(examples) < num_samples and attempts < max_attempts and len(seen) < n:
        if start_index is None:
            idx = int(rng.integers(0, n))
        else:
            idx = int(start_index + attempts)
            if idx >= n:
                break
        attempts += 1
        if idx in seen:
            continue
        seen.add(idx)
        ex = dataset[idx]
        if not include_invalid and not bool(ex.get("future_valid", True)):
            continue
        examples.append(ex)
    if len(examples) < num_samples:
        print(f"[warn] collected only {len(examples)} valid samples out of requested {num_samples}", flush=True)
    return examples


@dataclass
class MeanMeter:
    total: float = 0.0
    count: int = 0

    def add(self, value: float, n: int = 1) -> None:
        if value is None or not math.isfinite(float(value)):
            return
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def mean(self) -> float | None:
        if self.count <= 0:
            return None
        return self.total / self.count


@dataclass
class Aggregates:
    data: dict[str, dict[str, MeanMeter]] = field(default_factory=dict)

    def add(self, row: str, metric: str, value: float, n: int = 1) -> None:
        self.data.setdefault(row, {}).setdefault(metric, MeanMeter()).add(value, n)

    def to_dict(self) -> dict[str, dict[str, float | None]]:
        return {row: {metric: meter.mean for metric, meter in metrics.items()} for row, metrics in self.data.items()}


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a.float() - b.float()) ** 2).mean()


def _rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _mse(a, b).sqrt()


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean()


def _per_sample_metrics(pred: torch.Tensor, target: torch.Tensor, current: torch.Tensor) -> dict[str, list[float]]:
    return {
        "target_mse": ((pred - target) ** 2).float().mean(dim=(1, 2)).detach().cpu().tolist(),
        "target_rmse": (((pred - target) ** 2).float().mean(dim=(1, 2)).sqrt()).detach().cpu().tolist(),
        "target_cos": F.cosine_similarity(pred.float(), target.float(), dim=-1).mean(dim=1).detach().cpu().tolist(),
        "current_mse": ((pred - current) ** 2).float().mean(dim=(1, 2)).detach().cpu().tolist(),
    }


@torch.inference_mode()
def compute_condition(model, examples: list[dict], task: str):
    require_action = task == "fdm"
    batch = model._examples_to_batch(examples, require_future_dino=True, require_action=require_action)
    inputs_embeds, attn4d, position_ids, query_slice, _, _ = model._assemble_sequence(task, batch)
    hidden = model.backbone(inputs_embeds=inputs_embeds, attention_mask_4d=attn4d, position_ids=position_ids)
    cond = hidden[:, query_slice].float()
    target = model._select_future_dino(batch["dino_1"], batch["examples"]).to(cond.device, dtype=torch.float32)
    current = model._select_future_dino(batch["dino_0"], batch["examples"]).to(cond.device, dtype=torch.float32)
    valid_mask = model._future_valid_mask(batch, cond)
    return cond[valid_mask], target[valid_mask], current[valid_mask], valid_mask.detach().cpu().numpy().astype(bool)


@torch.inference_mode()
def predict_visual_from_noise(model, cond: torch.Tensor, noise: torch.Tensor, steps: int | None = None) -> torch.Tensor:
    head = model.visual_head
    cond = cond.float()
    z = noise.to(device=cond.device, dtype=torch.float32).clone()
    steps = int(steps or head.num_inference_timesteps)
    dt = 1.0 / float(steps)
    for step in range(steps):
        t_cont = step / float(steps)
        t_discretized = int(t_cont * head.num_timestep_buckets)
        timestep = torch.full((z.shape[0],), t_discretized, device=z.device, dtype=torch.long)
        hidden = head._embed_noisy(z)
        out = head.model(hidden_states=hidden, encoder_hidden_states=cond, timestep=timestep)
        pred_velocity = head.x_decode(out).float()
        z = z + dt * pred_velocity
    return z.float()


def _flow_loss_with_seed(model, cond: torch.Tensor, target: torch.Tensor, seed: int) -> float:
    devices = []
    if cond.device.type == "cuda":
        devices = [cond.device.index if cond.device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if cond.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        loss = model.visual_head(cond.float(), target.float())
    return _scalar(loss)


def _add_prediction_metrics(
    aggr: Aggregates,
    name: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    correct_pred: torch.Tensor | None,
    correct_cond: torch.Tensor | None,
    cond: torch.Tensor | None,
    n: int,
) -> None:
    target_mse = _scalar(_mse(pred, target))
    current_to_target_mse = _scalar(_mse(current, target))
    aggr.add(name, "target_mse", target_mse, n)
    aggr.add(name, "target_rmse", math.sqrt(max(target_mse, 0.0)), n)
    aggr.add(name, "target_cos", _scalar(_cos(pred, target)), n)
    aggr.add(name, "mse_vs_current_frame", _scalar(_mse(pred, current)), n)
    if current_to_target_mse > 0:
        aggr.add(name, "mse_over_copy_baseline", target_mse / current_to_target_mse, n)

    if correct_pred is not None:
        pair_mse = _scalar(_mse(pred, correct_pred))
        aggr.add(name, "pred_mse_to_correct", pair_mse, n)
        aggr.add(name, "pred_rmse_to_correct", math.sqrt(max(pair_mse, 0.0)), n)
        aggr.add(name, "pred_cos_to_correct", _scalar(_cos(pred, correct_pred)), n)
        if current_to_target_mse > 0:
            aggr.add(name, "pred_change_over_future_change", math.sqrt(pair_mse) / math.sqrt(current_to_target_mse), n)
    if correct_cond is not None and cond is not None:
        cond_mse = _scalar(_mse(cond, correct_cond))
        aggr.add(name, "cond_mse_to_correct", cond_mse, n)
        aggr.add(name, "cond_cos_to_correct", _scalar(_cos(cond, correct_cond)), n)


def _format_float(value) -> str:
    if value is None:
        return "na"
    return f"{float(value):.6g}"


def print_table(title: str, rows: dict[str, dict[str, float | None]], columns: Iterable[str]) -> None:
    columns = list(columns)
    widths = {"variant": max(7, max((len(k) for k in rows.keys()), default=7))}
    for col in columns:
        widths[col] = max(len(col), 10)
    print(f"\n{title}")
    print(" ".join(["variant".ljust(widths["variant"])] + [c.rjust(widths[c]) for c in columns]))
    print(" ".join(["-" * widths["variant"]] + ["-" * widths[c] for c in columns]))
    for name, metrics in rows.items():
        print(
            " ".join(
                [name.ljust(widths["variant"])]
                + [_format_float(metrics.get(col)).rjust(widths[col]) for col in columns]
            )
        )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _tensor_to_pil_rgb(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _array_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max(initial=0.0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr[..., :3], mode="RGB")


def _tokens_to_grid(tokens: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = tokens.detach().float().cpu().numpy() if torch.is_tensor(tokens) else np.asarray(tokens, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected tokens [N,D], got {arr.shape}")
    side = int(round(arr.shape[0] ** 0.5))
    if side * side != arr.shape[0]:
        raise ValueError(f"Token count is not square: {arr.shape[0]}")
    return arr.reshape(side, side, arr.shape[-1]).astype(np.float32)


def _normalize_image_channel(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        lo, hi = float(x.min(initial=0.0)), float(x.max(initial=1.0))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _heatmap_to_rgb(x: np.ndarray) -> Image.Image:
    x = _normalize_image_channel(x)
    rgb = np.stack(
        [
            x,
            0.5 - 0.5 * np.cos(np.pi * x),
            1.0 - x,
        ],
        axis=-1,
    )
    rgb = (rgb * 255.0).round().astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _pca_feature_rgb(grid: np.ndarray) -> Image.Image:
    h, w, d = grid.shape
    flat = grid.reshape(h * w, d).astype(np.float32)
    flat = flat - flat.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(flat, full_matrices=False)
        proj = flat @ vh[:3].T
        if proj.shape[1] < 3:
            proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])))
        rgb = np.stack([_normalize_image_channel(proj[:, i]) for i in range(3)], axis=-1)
        rgb = (rgb.reshape(h, w, 3) * 255.0).round().astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")
    except np.linalg.LinAlgError:
        return _heatmap_to_rgb(np.linalg.norm(grid, axis=-1))


def _save_feature_map(tokens: torch.Tensor, out_prefix: Path, image_size: int) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    grid = _tokens_to_grid(tokens)
    np.save(str(out_prefix) + ".npy", grid)
    norm_img = _heatmap_to_rgb(np.linalg.norm(grid, axis=-1))
    pca_img = _pca_feature_rgb(grid)
    norm_img.resize((image_size, image_size), Image.BICUBIC).save(str(out_prefix) + "_norm.png")
    pca_img.resize((image_size, image_size), Image.BICUBIC).save(str(out_prefix) + "_pca.png")


def _select_view_index_from_example(model, ex: dict) -> int:
    keys = ex.get("dino_view_keys") or ex.get("view_keys") or []
    keys = [str(k) for k in keys]
    preferred = list(model.config.framework.dino.get("future_view_keys", []))
    for name in preferred:
        if name in keys:
            return keys.index(name)
    return 0


def _find_single_dataset(dataset, ex: dict):
    datasets = getattr(dataset, "datasets", None)
    if datasets is None:
        return dataset
    name = str(ex.get("dataset_name", ""))
    for ds in datasets:
        if str(getattr(ds, "dataset_name", "")) == name:
            return ds
    return datasets[0] if datasets else None


def _read_rgb_views(dataset, ex: dict, frame_index_key: str) -> dict[str, Image.Image]:
    ds = _find_single_dataset(dataset, ex)
    if ds is None:
        return {}
    trajectory_id = int(ex["trajectory_id"])
    frame_index = int(ex[frame_index_key])
    ds.curr_traj_data = ds.get_trajectory_data(trajectory_id)
    out = {}
    for video_key in ds.modality_keys.get("video", []):
        try:
            frames = ds.get_video(trajectory_id, video_key, frame_index)
            if len(frames) > 0:
                out[str(video_key)] = _array_to_pil_rgb(frames[0])
        except Exception as exc:
            print(f"[warn] failed to read RGB {video_key} {trajectory_id=} {frame_index=}: {exc}", flush=True)
    return out


def _jointflow_norm_to_raw_dino(model, z_norm: torch.Tensor) -> torch.Tensor:
    mean = model._dino_mean.to(z_norm.device, dtype=torch.float32)
    std = model._dino_std.to(z_norm.device, dtype=torch.float32)
    return z_norm.float() * std + mean


def _save_visual_batch(
    *,
    output_dir: Path,
    dataset,
    model,
    rae_decoder: RAEDinoDecoder | None,
    examples: list[dict],
    valid_mask: np.ndarray,
    preds: dict[str, torch.Tensor],
    target: torch.Tensor,
    current: torch.Tensor,
    global_start: int,
    max_visual_samples: int,
    image_size: int,
    decode_batch_size: int,
) -> int:
    if max_visual_samples <= 0:
        return 0
    kept_examples = [ex for ex, valid in zip(examples, valid_mask) if bool(valid)]
    if not kept_examples:
        return 0

    variants = {"current_dino": current, "target_future_dino": target, **preds}
    saved = 0
    for local_idx, ex in enumerate(kept_examples):
        if saved >= max_visual_samples:
            break
        meta = _metadata(ex)
        sample_name = (
            f"sample_{global_start + local_idx:04d}_"
            f"traj{meta.get('trajectory_id', 'na')}_base{meta.get('base_index', 'na')}"
        )
        sample_dir = output_dir / "visualizations" / _sanitize_name(sample_name)
        rgb_dir = sample_dir / "rgb"
        fmap_dir = sample_dir / "dino_maps"
        recon_dir = sample_dir / "rae_rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        fmap_dir.mkdir(parents=True, exist_ok=True)
        if rae_decoder is not None:
            recon_dir.mkdir(parents=True, exist_ok=True)

        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        current_views = _read_rgb_views(dataset, ex, "base_index")
        future_views = _read_rgb_views(dataset, ex, "future_index")
        selected_view_idx = _select_view_index_from_example(model, ex)
        for view_idx, (view_key, img) in enumerate(current_views.items()):
            safe_key = _sanitize_name(view_key)
            img.save(rgb_dir / f"current_view{view_idx}_{safe_key}.png")
            if view_idx == selected_view_idx:
                img.save(rgb_dir / "current_selected.png")
        for view_idx, (view_key, img) in enumerate(future_views.items()):
            safe_key = _sanitize_name(view_key)
            img.save(rgb_dir / f"future_view{view_idx}_{safe_key}.png")
            if view_idx == selected_view_idx:
                img.save(rgb_dir / "future_selected.png")

        to_decode_names: list[str] = []
        to_decode_tokens: list[torch.Tensor] = []
        for name, tensor in variants.items():
            feature = tensor[local_idx].detach().float().cpu()
            _save_feature_map(feature, fmap_dir / _sanitize_name(name), image_size=image_size)
            if rae_decoder is not None:
                to_decode_names.append(name)
                # The feature is per-channel-standardized DINO (JointFlow's working
                # space). The RAE decoder expects raw DINO at its own extraction
                # scale, so re-scale with the decoder's stats.pt when available.
                # Falling back to JointFlow's stats yields ~2.5x-too-small inputs
                # that drive the decoder out of distribution (scrambled patches).
                if rae_decoder.has_latent_stats:
                    raw = rae_decoder.denorm_to_decoder_space(feature.to(rae_decoder.device))
                else:
                    raw = _jointflow_norm_to_raw_dino(model, feature.to(model.device))
                to_decode_tokens.append(raw)

        if rae_decoder is not None and to_decode_tokens:
            raw_tokens = torch.stack(to_decode_tokens, dim=0)
            decoded = rae_decoder.decode_raw_tokens(raw_tokens, batch_size=decode_batch_size)
            for name, rgb in zip(to_decode_names, decoded):
                _tensor_to_pil_rgb(rgb).save(recon_dir / f"{_sanitize_name(name)}.png")
        saved += 1
    return saved


def run_probe(args: argparse.Namespace) -> dict:
    ckpt_path = _resolve_path(args.ckpt_path)
    cfg, config_path = _load_config(args, ckpt_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = _resolve_path(args.output_dir, must_exist=False) if args.output_dir else None
    if output_dir is None and args.save_visuals:
        output_dir = _default_output_dir(ckpt_path, args)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.out_json is None:
            args.out_json = str(output_dir / "summary.json")
        if args.per_sample_jsonl is None:
            args.per_sample_jsonl = str(output_dir / "samples.jsonl")

    print(f"[jointflow-action-probe] ckpt={ckpt_path}", flush=True)
    print(f"[jointflow-action-probe] config={config_path}", flush=True)
    print(f"[jointflow-action-probe] device={device}", flush=True)
    if output_dir is not None:
        print(f"[jointflow-action-probe] output_dir={output_dir}", flush=True)

    register_jointflow_mixtures()
    dataset = build_joint_dataset(cfg, mode=args.mode)
    examples = collect_examples(
        dataset,
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        start_index=args.start_index,
        include_invalid=bool(args.include_invalid),
    )
    if not examples:
        raise RuntimeError("No examples collected. Check dataset path/config and future_valid filtering.")
    rae_image_size = _infer_rae_image_size(examples, int(args.rae_patch_size), args.rae_image_size)
    first_token_count = _infer_dino_token_count(examples[0])
    dino_grid_side = int(round(first_token_count ** 0.5)) if first_token_count else None

    model, incompatible = load_model(cfg, ckpt_path, device, strict=not args.non_strict_load)
    if args.non_strict_load:
        print(f"[jointflow-action-probe] missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}")

    decode_rae_rgb = bool(args.save_visuals and args.save_rae_rgb and not args.no_rae_decode)
    rae_decoder = None
    rae_decoder_dir_for_summary = None
    rae_decoder_config_for_summary = None
    rae_stats_grid = None
    if args.save_visuals:
        try:
            rae_decoder_dir_for_summary = _resolve_path(args.rae_decoder_dir)
            rae_stats_grid = _rae_stats_grid_shape(rae_decoder_dir_for_summary)
        except FileNotFoundError:
            rae_decoder_dir_for_summary = None
    if rae_stats_grid is not None and dino_grid_side is not None and tuple(rae_stats_grid) != (dino_grid_side, dino_grid_side):
        print(
            "[warn] RAE stage1 stats grid "
            f"{rae_stats_grid[0]}x{rae_stats_grid[1]} does not match JointFlow DINO grid "
            f"{dino_grid_side}x{dino_grid_side}. RAE RGB decoding is not a strict visualization.",
            flush=True,
        )
    if decode_rae_rgb:
        rae_repo = _resolve_path(args.rae_repo)
        rae_decoder_dir = rae_decoder_dir_for_summary or _resolve_path(args.rae_decoder_dir)
        rae_config = _resolve_path(args.rae_decoder_config)
        rae_decoder_config_for_summary = rae_config
        rae_device = torch.device(args.rae_device or str(device))
        print(f"[jointflow-action-probe] loading RAE decoder from {rae_decoder_dir}", flush=True)
        rae_decoder = RAEDinoDecoder(
            repo_dir=rae_repo,
            decoder_dir=rae_decoder_dir,
            decoder_config=rae_config,
            device=rae_device,
            image_size=int(rae_image_size),
            patch_size=int(args.rae_patch_size),
            dino_dim=int(model.d_dino),
            allow_token_interpolate=bool(args.rae_allow_token_interpolate),
        )

    aggr = Aggregates()
    per_sample_rows: list[dict] = []
    started = time.perf_counter()
    total_valid = 0
    chunk_count = 0
    visuals_saved = 0

    for start in range(0, len(examples), int(args.batch_size)):
        chunk = examples[start : start + int(args.batch_size)]
        chunk_count += 1
        variants = make_variants(chunk, args.opposite_mode)

        conds: dict[str, torch.Tensor] = {}
        target = None
        current = None
        valid_mask = None
        for name, (task, variant_examples) in variants.items():
            cond, target_i, current_i, valid_i = compute_condition(model, variant_examples, task)
            conds[name] = cond
            if target is None:
                target = target_i
                current = current_i
                valid_mask = valid_i
            elif target_i.shape != target.shape or not torch.allclose(target_i, target):
                raise RuntimeError(f"Target mismatch while building variant {name}")

        if target is None or current is None or valid_mask is None:
            continue
        n_valid = int(target.shape[0])
        if n_valid <= 0:
            continue
        total_valid += n_valid

        copy_mse = _scalar(_mse(current, target))
        aggr.add("current_copy", "target_mse", copy_mse, n_valid)
        aggr.add("current_copy", "target_rmse", math.sqrt(max(copy_mse, 0.0)), n_valid)
        aggr.add("current_copy", "target_cos", _scalar(_cos(current, target)), n_valid)
        aggr.add("current_copy", "mse_vs_current_frame", 0.0, n_valid)
        aggr.add("current_copy", "mse_over_copy_baseline", 1.0, n_valid)

        preds_by_repeat: list[dict[str, torch.Tensor]] = []
        for repeat in range(int(args.num_repeats)):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(args.seed) + repeat)
            noise = torch.randn(tuple(target.shape), generator=gen, dtype=torch.float32).to(device)
            preds = {
                name: predict_visual_from_noise(model, cond, noise, steps=args.num_inference_timesteps)
                for name, cond in conds.items()
            }
            preds_by_repeat.append(preds)
            correct_pred = preds["correct_action"]
            correct_cond = conds["correct_action"]
            for name, pred in preds.items():
                _add_prediction_metrics(
                    aggr,
                    name,
                    pred,
                    target,
                    current,
                    correct_pred=None if name == "correct_action" else correct_pred,
                    correct_cond=None if name == "correct_action" else correct_cond,
                    cond=None if name == "correct_action" else conds[name],
                    n=n_valid,
                )

            if args.per_sample_jsonl:
                kept_examples = [ex for ex, valid in zip(chunk, valid_mask) if bool(valid)]
                for name, pred in preds.items():
                    sample_metrics = _per_sample_metrics(pred, target, current)
                    if name != "correct_action":
                        sample_metrics["pred_mse_to_correct"] = (
                            ((pred - correct_pred) ** 2).float().mean(dim=(1, 2)).detach().cpu().tolist()
                        )
                    for idx, ex in enumerate(kept_examples):
                        row = {"variant": name, "repeat": repeat, **_metadata(ex)}
                        row.update({k: float(v[idx]) for k, v in sample_metrics.items()})
                        per_sample_rows.append(row)

            if (
                args.save_visuals
                and output_dir is not None
                and repeat == int(args.visual_repeat)
                and visuals_saved < int(args.max_visual_samples)
            ):
                visuals_saved += _save_visual_batch(
                    output_dir=output_dir,
                    dataset=dataset,
                    model=model,
                    rae_decoder=rae_decoder,
                    examples=chunk,
                    valid_mask=valid_mask,
                    preds=preds,
                    target=target,
                    current=current,
                    global_start=start,
                    max_visual_samples=int(args.max_visual_samples) - visuals_saved,
                    image_size=int(rae_image_size),
                    decode_batch_size=int(args.rae_decode_batch_size),
                )

        if args.compute_flow_loss:
            for name, cond in conds.items():
                loss = _flow_loss_with_seed(model, cond, target, seed=int(args.seed))
                aggr.add(name, "flow_loss", loss, n_valid)

        if args.verbose:
            print(f"[jointflow-action-probe] batch={chunk_count} valid={n_valid} elapsed={time.perf_counter() - started:.1f}s", flush=True)

    rows = aggr.to_dict()
    pred_cols = [
        "target_mse",
        "target_rmse",
        "target_cos",
        "mse_over_copy_baseline",
        "pred_rmse_to_correct",
        "pred_cos_to_correct",
        "pred_change_over_future_change",
    ]
    if args.compute_flow_loss:
        pred_cols.append("flow_loss")
    print_table("Prediction metrics in normalized DINO space", rows, pred_cols)

    cond_rows = {k: v for k, v in rows.items() if k not in {"current_copy", "correct_action"}}
    print_table("Condition / prediction sensitivity vs correct_action", cond_rows, ["cond_mse_to_correct", "cond_cos_to_correct", "pred_mse_to_correct"])

    result = {
        "ckpt_path": str(ckpt_path),
        "config_path": str(config_path),
        "device": str(device),
        "num_samples_requested": int(args.num_samples),
        "num_samples_collected": len(examples),
        "num_valid_evaluated": int(total_valid),
        "batch_size": int(args.batch_size),
        "num_repeats": int(args.num_repeats),
        "opposite_mode": str(args.opposite_mode),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "visuals_saved": int(visuals_saved),
        "rae_decoder": {
            "enabled": bool(rae_decoder is not None),
            "decoder_dir": str(rae_decoder_dir_for_summary) if rae_decoder_dir_for_summary is not None else None,
            "decoder_config": str(rae_decoder_config_for_summary) if rae_decoder_config_for_summary is not None else None,
            "device": str(args.rae_device or device) if decode_rae_rgb else None,
            "image_size": int(rae_image_size),
            "patch_size": int(args.rae_patch_size),
            "allow_token_interpolate": bool(args.rae_allow_token_interpolate),
            "save_rae_rgb": bool(args.save_rae_rgb),
            "jointflow_dino_grid": [int(dino_grid_side), int(dino_grid_side)] if dino_grid_side else None,
            "rae_stats_grid": list(rae_stats_grid) if rae_stats_grid is not None else None,
            "strict_match": bool(rae_stats_grid is not None and dino_grid_side is not None and tuple(rae_stats_grid) == (dino_grid_side, dino_grid_side)),
            "note": "RAE RGB is disabled unless --save_rae_rgb is passed. It is only trustworthy when the RAE stage1 encoder contract matches the stored JointFlow DINO features.",
        },
        "metrics": rows,
        "sample_metadata_preview": [_metadata(ex) for ex in examples[: min(5, len(examples))]],
        "interpretation_hint": {
            "action_helpful": "correct_action target_mse should be clearly below no_action_passive/opposite_action and below current_copy.",
            "action_ignored": "opposite/no_action predictions have pred_cos_to_correct near 1 and pred_change_over_future_change near 0.",
            "video_prior": "current_copy is already strong and correct_action barely improves mse_over_copy_baseline.",
        },
    }

    if args.out_json:
        out_path = _resolve_path(args.out_json, must_exist=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nwrote summary: {out_path}", flush=True)

    if args.per_sample_jsonl:
        per_sample_path = _resolve_path(args.per_sample_jsonl, must_exist=False)
        _write_jsonl(per_sample_path, per_sample_rows)
        print(f"wrote per-sample metrics: {per_sample_path}", flush=True)

    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug whether JointFlow future-DINO predictions use action context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="Z_ws/jointflow_libero_object/checkpoints_all/steps_50000_pytorch_model.pt",
        help="Path to model-only JointFlow checkpoint.",
    )
    parser.add_argument("--config_yaml", type=str, default=None, help="Training config YAML; inferred from ckpt run_dir if omitted.")
    parser.add_argument("--data_root_dir", type=str, default=None, help="Override cfg.datasets.vla_data.data_root_dir.")
    parser.add_argument("--data_mix", type=str, default=None, help="Override cfg.datasets.vla_data.data_mix.")
    parser.add_argument("--dino_feature_dir", type=str, default=None, help="Override cfg.datasets.vla_data.dino_feature_dir.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val"], help="Dataset split mode.")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of valid dataset samples to probe.")
    parser.add_argument("--batch_size", type=int, default=4, help="Forward batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling and diffusion-noise seed.")
    parser.add_argument("--start_index", type=int, default=None, help="Use sequential dataset indices from this offset instead of random samples.")
    parser.add_argument("--include_invalid", action="store_true", help="Do not skip future_valid=False samples.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0 or cpu. Defaults to cuda if available.")
    parser.add_argument("--num_repeats", type=int, default=1, help="Repeat predictions with different shared initial noises.")
    parser.add_argument("--num_inference_timesteps", type=int, default=None, help="Override visual flow inference steps.")
    parser.add_argument(
        "--opposite_mode",
        type=str,
        default="negate",
        choices=["negate", "reverse_time", "negate_reverse_time"],
        help="How to construct the opposite-action baseline.",
    )
    parser.add_argument("--compute_flow_loss", action="store_true", help="Also report visual flow-matching loss for each variant.")
    parser.add_argument("--non_strict_load", action="store_true", help="Load checkpoint with strict=False.")
    parser.add_argument("--output_dir", type=str, default=None, help="Run directory for summary JSON, sample JSONL, and visualization files.")
    parser.add_argument("--out_json", type=str, default=None, help="Optional path for aggregate JSON summary.")
    parser.add_argument("--per_sample_jsonl", type=str, default=None, help="Optional path for per-sample JSONL metrics.")
    parser.add_argument("--save_visuals", action="store_true", help="Save RGB observations and DINO feature maps.")
    parser.add_argument("--max_visual_samples", type=int, default=8, help="Maximum valid samples to save under output_dir/visualizations.")
    parser.add_argument("--visual_repeat", type=int, default=0, help="Prediction repeat index to visualize.")
    parser.add_argument("--save_rae_rgb", action="store_true", help="Also save experimental RAE decoded RGBs under rae_rgb/.")
    parser.add_argument("--no_rae_decode", action="store_true", help="Compatibility flag; forces RAE decoded RGB outputs off.")
    parser.add_argument("--rae_repo", type=str, default="../../Third_github/RAEv2", help="Path to the RAEv2 reference repository.")
    parser.add_argument(
        "--rae_decoder_dir",
        type=str,
        default="tools/stage1_imagenet_dinov3s-k1",
        help="Directory containing decoder.pt and stats.pt.",
    )
    parser.add_argument(
        "--rae_decoder_config",
        type=str,
        default="../../Third_github/RAEv2/configs/decoder/ViTXL",
        help="RAEv2 decoder config directory or config.json.",
    )
    parser.add_argument("--rae_device", type=str, default=None, help="Device for RAE decoder; defaults to --device.")
    parser.add_argument("--rae_image_size", type=int, default=None, help="RAE decoder output image size. Defaults to sqrt(num_dino_tokens) * patch_size.")
    parser.add_argument("--rae_patch_size", type=int, default=16, help="RAE decoder patch size.")
    parser.add_argument("--rae_decode_batch_size", type=int, default=2, help="RAE decoder batch size for visual outputs.")
    parser.add_argument("--rae_allow_token_interpolate", action="store_true", help="Allow RAE decoder to interpolate latent tokens when token count and image_size disagree.")
    parser.add_argument("--verbose", action="store_true", help="Print per-batch progress.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
