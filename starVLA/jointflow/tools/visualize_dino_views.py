"""Visualize JointFlow offline DINO latents for one LIBERO sample.

This script saves per-view heatmaps for dino_0, dino_1, and their difference.
It visualizes DINO patch-token norms; DINO latents are not RGB images.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from starVLA.jointflow.data.joint_dataset import build_joint_dataset
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", default="starVLA/jointflow/configs/jointflow_libero.yaml")
    parser.add_argument("--data_root_dir", default="/mnt/hwdata/wangsen/starVLA/DATA/LEBERO/libero")
    parser.add_argument("--data_mix", default="libero_all")
    parser.add_argument("--dataset_name", default="libero_object_no_noops_1.0.0_lerobot")
    parser.add_argument("--trajectory_id", type=int, default=448)
    parser.add_argument("--base_index", type=int, default=150)
    parser.add_argument("--out_dir", default="Z_ws/jointflow_dino_views")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _load_cfg(args: argparse.Namespace):
    cfg = OmegaConf.load(args.config_yaml)
    override = OmegaConf.from_dotlist(
        normalize_dotlist_args(
            [
                "--datasets.vla_data.data_root_dir",
                args.data_root_dir,
                "--datasets.vla_data.data_mix",
                args.data_mix,
                "--datasets.vla_data.per_device_batch_size",
                "1",
                "--datasets.vla_data.num_workers",
                "0",
                "--datasets.vla_data.dino_feature_dir",
                "latents",
                "--datasets.vla_data.dino_feature_layout",
                "lingbot_episode",
            ]
        )
    )
    return apply_config_compat(OmegaConf.merge(cfg, override))


def _find_single_dataset(mixture, dataset_name: str):
    for dataset in mixture.datasets:
        if dataset.dataset_name == dataset_name:
            return dataset
    names = [dataset.dataset_name for dataset in mixture.datasets]
    raise ValueError(f"Dataset {dataset_name!r} not found. Available datasets: {names}")


def _get_exact_sample(dataset, trajectory_id: int, base_index: int) -> dict:
    raw_data = dataset.get_step_data(trajectory_id, base_index)
    data = dataset.transforms(raw_data)
    return dataset._pack_sample(data)


def _first_or_square_map(tokens: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(tokens.astype(np.float32), axis=-1)
    side = int(round(norm.shape[0] ** 0.5))
    if side * side == norm.shape[0]:
        return norm.reshape(side, side)
    return norm[None, :]


def _zscore_for_display(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32)
    std = float(mat.std())
    if std < 1e-6:
        return mat - float(mat.mean())
    return (mat - float(mat.mean())) / std


def _save_view_figure(sample: dict, view_idx: int, view_name: str, out_dir: Path, dpi: int) -> None:
    d0 = np.asarray(sample["dino_0"])[view_idx]
    d1 = np.asarray(sample["dino_1"])[view_idx]
    m0 = _first_or_square_map(d0)
    m1 = _first_or_square_map(d1)
    md = np.linalg.norm((d1 - d0).astype(np.float32), axis=-1).reshape(m0.shape)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    mats = [
        (m0, "dino_0 patch norm"),
        (m1, "dino_1 patch norm"),
        (md, "|dino_1 - dino_0| patch norm"),
    ]
    for ax, (mat, title) in zip(axes, mats):
        im = ax.imshow(mat, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"{view_name} | traj={sample['trajectory_id']} base={sample['base_index']} "
        f"future={sample['future_index']} valid={sample['future_valid']}"
    )
    fig.savefig(out_dir / f"view{view_idx}_{view_name}_dino_norms.png", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    mats = [
        (_zscore_for_display(m0), "dino_0 norm z-score"),
        (_zscore_for_display(m1), "dino_1 norm z-score"),
        (_zscore_for_display(md), "diff norm z-score"),
    ]
    for ax, (mat, title) in zip(axes, mats):
        im = ax.imshow(mat, cmap="coolwarm", vmin=-3, vmax=3)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{view_name} normalized display")
    fig.savefig(out_dir / f"view{view_idx}_{view_name}_dino_norms_zscore.png", dpi=dpi)
    plt.close(fig)


def _save_overview(sample: dict, out_dir: Path, dpi: int) -> None:
    dino_0 = np.asarray(sample["dino_0"])
    dino_1 = np.asarray(sample["dino_1"])
    view_keys = list(sample.get("dino_view_keys", []))
    num_views = int(dino_0.shape[0])

    fig, axes = plt.subplots(num_views, 3, figsize=(12, 4 * num_views), constrained_layout=True)
    if num_views == 1:
        axes = axes[None, :]

    for view_idx in range(num_views):
        view_name = view_keys[view_idx] if view_idx < len(view_keys) else f"view{view_idx}"
        m0 = _first_or_square_map(dino_0[view_idx])
        m1 = _first_or_square_map(dino_1[view_idx])
        md = np.linalg.norm((dino_1[view_idx] - dino_0[view_idx]).astype(np.float32), axis=-1).reshape(m0.shape)
        for ax, mat, title in [
            (axes[view_idx, 0], m0, f"{view_name}\ndino_0"),
            (axes[view_idx, 1], m1, f"{view_name}\ndino_1"),
            (axes[view_idx, 2], md, f"{view_name}\ndiff"),
        ]:
            im = ax.imshow(mat, cmap="viridis")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"{sample['dataset_name']} | traj={sample['trajectory_id']} "
        f"base={sample['base_index']} future={sample['future_index']} "
        f"steps={sample['future_valid_steps']}/{sample['future_stride']} | {sample['lang']}"
    )
    fig.savefig(out_dir / "all_views_dino_overview.png", dpi=dpi)
    plt.close(fig)


def _write_summary(sample: dict, out_dir: Path) -> None:
    d0 = np.asarray(sample["dino_0"])
    d1 = np.asarray(sample["dino_1"])
    action = np.asarray(sample["action"])
    state = np.asarray(sample["state"])
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        for key in [
            "dataset_name",
            "trajectory_id",
            "base_index",
            "future_index",
            "future_valid",
            "future_valid_steps",
            "future_stride",
            "dino_view_keys",
            "lang",
        ]:
            f.write(f"{key}: {sample.get(key)}\n")
        f.write(f"dino_0 shape: {d0.shape}, min={d0.min():.6f}, max={d0.max():.6f}, mean={d0.mean():.6f}, std={d0.std():.6f}\n")
        f.write(f"dino_1 shape: {d1.shape}, min={d1.min():.6f}, max={d1.max():.6f}, mean={d1.mean():.6f}, std={d1.std():.6f}\n")
        f.write(f"action shape: {action.shape}\n")
        f.write(f"state shape: {state.shape}\n")
        f.write(f"first action: {action[0]}\n")
        f.write(f"last action: {action[-1]}\n")
        f.write(f"state[0]: {state.reshape(-1, state.shape[-1])[0]}\n")


def main() -> None:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(args)
    mixture = build_joint_dataset(cfg, mode="train")
    dataset = _find_single_dataset(mixture, args.dataset_name)
    sample = _get_exact_sample(dataset, args.trajectory_id, args.base_index)

    view_keys = list(sample.get("dino_view_keys", []))
    for view_idx in range(np.asarray(sample["dino_0"]).shape[0]):
        view_name = view_keys[view_idx] if view_idx < len(view_keys) else f"view{view_idx}"
        safe_view_name = view_name.replace("/", "__").replace(".", "_")
        _save_view_figure(sample, view_idx, safe_view_name, out_dir, args.dpi)
    _save_overview(sample, out_dir, args.dpi)
    _write_summary(sample, out_dir)

    print(f"saved DINO view visualizations to: {out_dir.resolve()}")
    for path in sorted(out_dir.glob("*.png")):
        print(path)
    print(out_dir / "summary.txt")


if __name__ == "__main__":
    main()
