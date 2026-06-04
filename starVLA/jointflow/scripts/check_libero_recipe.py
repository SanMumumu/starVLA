#!/usr/bin/env python
"""Audit JointFlow LIBERO training recipe and feature scales.

This script intentionally does not load Qwen or run training. It checks the
high-impact recipe switches that can silently destroy LIBERO score:

- action normalization modes from the active robot DataConfig
- sampler/shuffle-related config
- action DiT hidden-size consistency
- task sampling ratios
- effective global batch
- DINO raw and normalized feature ranges
- one small dataloader batch, if available
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionTransform
from starVLA.jointflow.data.joint_dataset import build_joint_dataloader
from starVLA.jointflow.data.mix_registry import resolve_data_mix
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    flat = x.reshape(-1)
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "q01": float(np.quantile(flat, 0.01)),
        "q99": float(np.quantile(flat, 0.99)),
    }


def _print_stats(name: str, x: np.ndarray) -> None:
    s = _stats(x)
    print(
        f"{name:34s} "
        f"min={s['min']:9.4f} max={s['max']:9.4f} "
        f"mean={s['mean']:9.4f} std={s['std']:9.4f} "
        f"q01={s['q01']:9.4f} q99={s['q99']:9.4f}"
    )


def _flatten_transforms(transforms: Any) -> list[Any]:
    if isinstance(transforms, ComposedModalityTransform):
        out = []
        for transform in transforms.transforms:
            out.extend(_flatten_transforms(transform))
        return out
    return [transforms]


def _action_norm_modes(robot_type: str) -> dict[str, str]:
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modes: dict[str, str] = {}
    for transform in _flatten_transforms(data_config.transform()):
        if isinstance(transform, StateActionTransform):
            for key, mode in dict(transform.normalization_modes).items():
                if str(key).startswith("action."):
                    modes[str(key)] = str(mode)
    return modes


def _audit_action_norm(mixture: list[tuple[str, float, str]]) -> None:
    print("\n[action normalization]")
    seen = set()
    for _, _, robot_type in mixture:
        if robot_type in seen:
            continue
        seen.add(robot_type)
        data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        action_keys = list(getattr(data_config, "action_keys", []))
        modes = _action_norm_modes(robot_type)
        print(f"robot_type={robot_type}")
        missing = []
        for key in action_keys:
            mode = modes.get(str(key), "<missing>")
            print(f"  {key:24s} {mode}")
            if mode == "<missing>":
                missing.append(str(key))
        if missing:
            print(f"  [WARN] action keys without explicit normalization: {missing}")
        else:
            print("  [OK] every action key has an explicit normalization mode")


def _audit_recipe(cfg, num_gpus: int) -> None:
    vla_cfg = cfg.datasets.vla_data
    trainer = cfg.trainer
    weights = {str(k): float(v) for k, v in cfg.framework.tasks.weights.items()}
    weight_sum = max(sum(weights.values()), 1e-12)
    print("\n[recipe]")
    print(f"data_mix={vla_cfg.data_mix}")
    print(f"action_horizon={vla_cfg.action_horizon}")
    print(f"world_model.future_stride={_cfg_get(_cfg_get(vla_cfg, 'world_model', None), 'future_stride', '<unset>')}")
    print(f"sequential_step_sampling={vla_cfg.get('sequential_step_sampling', '<unset>')}")
    if bool(vla_cfg.get("sequential_step_sampling", False)):
        print("  [WARN] sequential_step_sampling is true; LIBERO training should usually shuffle.")
    else:
        print("  [OK] sequential_step_sampling is false")

    per_device = int(vla_cfg.per_device_batch_size)
    grad_accum = int(trainer.get("gradient_accumulation_steps", 1))
    print(f"effective_global_batch={per_device} per_device * {num_gpus} gpus * {grad_accum} grad_accum = {per_device * num_gpus * grad_accum}")
    print(f"max_train_steps={trainer.max_train_steps}")
    print(f"num_warmup_steps={trainer.num_warmup_steps}")
    print(f"lr_scheduler_type={trainer.lr_scheduler_type}")
    print(f"task_weights={weights}")
    print(f"policy_step_fraction={weights.get('policy', 0.0) / weight_sum:.3f}")


def _audit_action_dit(cfg) -> None:
    action_cfg = cfg.framework.action_model
    dit_cfg = action_cfg.diffusion_model_cfg
    heads = int(dit_cfg.get("num_attention_heads", 12))
    head_dim = int(dit_cfg.get("attention_head_dim", 64))
    inner = heads * head_dim
    output_dim = int(dit_cfg.get("output_dim", inner))
    mlp_hidden = int(action_cfg.get("hidden_size", inner))
    print("\n[action DiT shape]")
    print(f"action_model_type={action_cfg.action_model_type}")
    print(f"DiT inner_dim=num_attention_heads*attention_head_dim={heads}*{head_dim}={inner}")
    print(f"diffusion_model_cfg.output_dim={output_dim}")
    print(f"action_model.hidden_size={mlp_hidden}  # action decoder/context MLP width")
    if output_dim != inner:
        print("  [WARN] output_dim != DiT inner_dim; check whether this is intentional.")
    else:
        print("  [OK] output_dim matches DiT inner_dim")
    if mlp_hidden != inner:
        print("  [INFO] action_model.hidden_size differs from DiT inner_dim; in GR00T_ActionHead this is the decoder MLP width, not the DiT width.")


def _load_latent(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu")
    latent = obj["latent"] if isinstance(obj, dict) else obj
    if not torch.is_tensor(latent):
        latent = torch.as_tensor(latent)
    return latent.float().numpy()


def _audit_dino_raw_ranges(cfg, mixture: list[tuple[str, float, str]], num_raw_files: int) -> None:
    print("\n[dino latent raw/norm ranges]")
    root = Path(cfg.datasets.vla_data.data_root_dir)
    feature_dir = str(cfg.datasets.vla_data.get("dino_feature_dir", "latents"))
    for data_name, _, _ in mixture:
        latents_root = root / data_name / feature_dir
        stats_path = latents_root / "dino_v3_stats.json"
        print(f"\ndataset={data_name}")
        if not stats_path.exists():
            print(f"  [WARN] missing stats: {stats_path}")
            continue
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.maximum(np.asarray(stats["std"], dtype=np.float32), 1e-6)
        print(f"  stats_count={stats.get('count', '<missing>')}")
        _print_stats("  dino_stats.mean", mean)
        _print_stats("  dino_stats.std", std)
        if num_raw_files <= 0:
            continue

        paths = sorted((latents_root / "chunk-000").glob("*/*.pth"))[:num_raw_files]
        if not paths:
            print(f"  [WARN] no latent .pth files found under {latents_root / 'chunk-000'}")
            continue
        for path in paths:
            raw = _load_latent(path)
            norm = (raw - mean) / std
            print(f"  file={path.relative_to(latents_root)} shape={tuple(raw.shape)}")
            _print_stats("    raw_dino", raw)
            _print_stats("    norm_dino", norm)


def _audit_batch_ranges(cfg, batch_size: int, skip_dataloader: bool) -> None:
    if skip_dataloader:
        return
    print("\n[dataloader batch ranges]")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.datasets.vla_data.per_device_batch_size = int(batch_size)
    cfg.datasets.vla_data.num_workers = 0
    try:
        loader = build_joint_dataloader(cfg)
        batch = next(iter(loader))
    except Exception as exc:
        print(f"  [WARN] unable to build/read dataloader batch: {exc}")
        return
    action = np.stack([np.asarray(ex["action"], dtype=np.float32) for ex in batch], axis=0)
    dino0 = np.stack([np.asarray(ex["dino_0"], dtype=np.float32) for ex in batch], axis=0)
    dino1 = np.stack([np.asarray(ex["dino_1"], dtype=np.float32) for ex in batch], axis=0)
    future_valid = np.asarray([float(ex.get("future_valid", False)) for ex in batch], dtype=np.float32)
    _print_stats("  batch.action(normed)", action)
    _print_stats("  batch.dino_0(normed)", dino0)
    _print_stats("  batch.dino_1(normed)", dino1)
    print(f"  future_valid_fraction={future_valid.mean():.3f}")
    if abs(float(dino0.mean())) > 0.25 or float(dino0.std()) < 0.5 or float(dino0.std()) > 1.8:
        print("  [WARN] dino_0 normalized range looks unusual; check dino_v3_stats.json and latent layout.")
    else:
        print("  [OK] dino_0 is z-score normalized to a roughly standard scale")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/jointflow/configs/jointflow_libero_highscore_b24.yaml")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_raw_files", type=int, default=1)
    parser.add_argument("--skip_dataloader", action="store_true")
    args, extra = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cli_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(extra))
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_config_compat(cfg)
    mixture = resolve_data_mix(str(cfg.datasets.vla_data.data_mix))

    print(f"[config] {args.config_yaml}")
    _audit_recipe(cfg, num_gpus=int(args.num_gpus))
    _audit_action_norm(mixture)
    _audit_action_dit(cfg)
    _audit_dino_raw_ranges(cfg, mixture, num_raw_files=int(args.num_raw_files))
    _audit_batch_ranges(cfg, batch_size=int(args.batch_size), skip_dataloader=bool(args.skip_dataloader))

    print("\n[conclusion]")
    print("DINO features should be normalized for the flow head. JointFlow uses per-channel z-score normalization, not strict [-1, 1] scaling.")
    print("For actions, verify every action key above has an explicit normalization mode and that batch.action(normed) is in a sane range.")


if __name__ == "__main__":
    main()
