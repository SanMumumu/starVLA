"""CED E0.1：数据级诊断——动作可从 (z_t, z_H) 恢复吗（离线 IDM probe）。

机制 gate：若平动维 R² 过低，说明 LIBERO 的 DINO 特征对动作差分根本不可辨识，
CED 的 loss_ident/loss_align 没有信息来源，**项目应停下来换数据（human video）再议**。

特征 = concat[pool(z_t), pool(z_H), pool(z_H−z_t)]（主视角 patch 平均池化，3·D 维），
标签 = 归一化动作 chunk a.flatten()。按轨迹划 train/val 防泄露。

运行（starVLA env）：
  python starVLA/jointflow/scripts/diag_offline_idm_probe.py \
    --config_yaml starVLA/jointflow/configs/jointflow_libero_ced.yaml \
    --data_root_dir /mnt/hwdata/wangsen/starVLA/DATA/LEBERO/libero \
    --out playground/Debug_Checkpoints/ced_diag/E0.1
输出：<out>/idm_probe.json + <out>/idm_probe_r2.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.jointflow.data.joint_dataset import build_joint_dataset
from starVLA.jointflow.scripts._probe_common import collect_probe_samples, save_json, train_probe

LIBERO_DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/jointflow/configs/jointflow_libero_ced.yaml")
    parser.add_argument("--data_root_dir", type=str, default=None)
    parser.add_argument("--data_mix", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--probe_hidden", type=int, default=512)
    parser.add_argument("--val_mod", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="playground/Debug_Checkpoints/ced_diag/E0.1")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.load(args.config_yaml)
    if args.data_root_dir:
        cfg.datasets.vla_data.data_root_dir = args.data_root_dir
    if args.data_mix:
        cfg.datasets.vla_data.data_mix = args.data_mix
    # 中文注释：诊断优先走离线 latents（快、免视频解码）；没有 latents 时再在线提特征。
    cfg.datasets.vla_data.online_dino = "auto"
    cfg.datasets.vla_data.num_workers = 0
    dataset = build_joint_dataset(cfg, mode="train")

    dino = None
    dino_stats = None
    probe_sample = dataset.datasets[0][0]
    if "dino_0" not in probe_sample:
        from starVLA.jointflow.modules.dino_v3 import DINOv3Backbone, resolve_dino_spec

        dino = DINOv3Backbone(**resolve_dino_spec(cfg.framework.dino)).to(device).eval()
        print("[E0.1] no offline latents -> live frozen DINO extraction (slower)")

    preferred = list(cfg.framework.dino.get("future_view_keys", []))
    z_t, z_h, actions, traj_keys = collect_probe_samples(
        dataset, num_samples=args.num_samples, seed=args.seed, preferred_views=preferred,
        dino=dino, dino_stats=dino_stats, device=device,
    )
    print(f"[E0.1] collected {len(traj_keys)} samples, z {z_t.shape}, action {actions.shape}")

    feats = np.concatenate([z_t.mean(axis=1), z_h.mean(axis=1), (z_h - z_t).mean(axis=1)], axis=1)
    labels = actions.reshape(actions.shape[0], -1)
    result = train_probe(
        feats, labels, traj_keys, hidden=args.probe_hidden, epochs=args.epochs,
        val_mod=args.val_mod, seed=args.seed, device=device,
    )

    # 中文注释：逐"动作维"汇总（对 horizon 上同一维取均值）：R²[H·D] → R²[D]。
    horizon, dim = actions.shape[1], actions.shape[2]
    r2 = np.asarray(result["r2_per_dim"]).reshape(horizon, dim)
    r2_by_dim = r2.mean(axis=0)
    dim_names = LIBERO_DIM_NAMES if dim == len(LIBERO_DIM_NAMES) else [f"dim{i}" for i in range(dim)]
    translation_r2 = float(r2_by_dim[: min(6, dim)].mean())
    result.update(
        {
            "r2_by_action_dim": {name: float(v) for name, v in zip(dim_names, r2_by_dim)},
            "r2_by_dim_first_step": {name: float(v) for name, v in zip(dim_names, r2[0])},
            "translation_r2_mean": translation_r2,
            "num_samples": int(actions.shape[0]),
            "gate_hint": "平动维 R² 显著>0 才支撑 CED；接近 0 → LIBERO 撑不住该机制，停下换数据",
        }
    )
    out_dir = Path(args.out)
    save_json(out_dir / "idm_probe.json", result)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dim_names, r2_by_dim, color=["tab:blue"] * min(6, dim) + ["tab:orange"] * max(dim - 6, 0))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("val R^2 (mean over horizon)")
    ax.set_title(f"E0.1 offline IDM probe  |  translation R^2 = {translation_r2:.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / "idm_probe_r2.png", dpi=150)
    print(f"[E0.1] translation R^2 mean = {translation_r2:.4f} -> {out_dir/'idm_probe_r2.png'}")


if __name__ == "__main__":
    main()
