"""CED 诊断脚本共享件：样本特征收集 + MLP probe 训练核心。

复用:
- starVLA.jointflow.data.joint_dataset.build_joint_dataset
- starVLA.jointflow.modules.dino_v3.DINOv3Backbone / resolve_dino_spec（在线样本兜底提特征）
被 diag_offline_idm_probe.py / probe_weak_readout.py 共用。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


######### // code // ##########
# 中文注释：按 future_view_keys 偏好挑主视角下标（与 framework._select_future_dino 同规则）。
def pick_main_view(view_keys: list[str], preferred: list[str]) -> int:
    for name in preferred:
        if name in view_keys:
            return view_keys.index(name)
    return 0


# 中文注释：从 dataset 随机收集 num_samples 个 future_valid 样本的 (z_t, z_H, action, traj_key)。
# 离线样本直读 dino_0/dino_1（已 z-norm）；在线样本（image_0/image_1）用 frozen DINO 现场提特征
# 后按 stats 归一化（dino_stats=(mean,std) 由调用方提供，None 则不归一化并打印警告）。
def collect_probe_samples(dataset, *, num_samples: int, seed: int, preferred_views: list[str],
                          dino=None, dino_stats=None, device="cuda"):
    rng = np.random.default_rng(seed)
    n = len(dataset)
    feats_t, feats_h, actions, traj_keys = [], [], [], []
    seen: set[int] = set()
    attempts, max_attempts = 0, max(num_samples * 50, 2000)
    while len(actions) < num_samples and attempts < max_attempts:
        idx = int(rng.integers(0, n))
        attempts += 1
        if idx in seen:
            continue
        seen.add(idx)
        ex = dataset[idx]
        if not bool(ex.get("future_valid", True)):
            continue
        view = pick_main_view(list(ex.get("dino_view_keys", [])), preferred_views)
        if "dino_0" in ex:
            z_t = np.asarray(ex["dino_0"])[view]
            z_h = np.asarray(ex["dino_1"])[view]
        else:
            assert dino is not None, "在线样本需要传入 frozen DINO backbone"
            from PIL import Image

            imgs = [Image.fromarray(np.asarray(ex["image_0"])[view]), Image.fromarray(np.asarray(ex["image_1"])[view])]
            with torch.no_grad():
                feats = dino(dino.preprocess_batch(imgs).to(device)).float().cpu().numpy()
            z_t, z_h = feats[0], feats[1]
            if dino_stats is not None:
                mean, std = dino_stats
                z_t = (z_t - mean) / std
                z_h = (z_h - mean) / std
        feats_t.append(z_t.astype(np.float32))
        feats_h.append(z_h.astype(np.float32))
        actions.append(np.asarray(ex["action"], dtype=np.float32))
        traj_keys.append(f"{ex.get('dataset_name', 'ds')}::{int(ex.get('trajectory_id', -1))}")
    return np.stack(feats_t), np.stack(feats_h), np.stack(actions), traj_keys
######### // code // ##########


######### // code // ##########
# 中文注释：probe 训练核心。X [N,F] → y [N,Dout]，按轨迹 hash 划 train/val（防同轨迹泄露），
# 2 层 MLP + Adam，返回 val 上逐维 R²（1 − MSE/Var）与训练曲线。
def train_probe(x: np.ndarray, y: np.ndarray, traj_keys: list[str], *, hidden=512, epochs=200,
                batch_size=256, lr=1e-3, val_mod=10, seed=0, device="cuda") -> dict:
    torch.manual_seed(seed)
    is_val = np.array([abs(hash(k)) % val_mod == 0 for k in traj_keys])
    if is_val.all() or (~is_val).all():
        raise RuntimeError(f"轨迹划分退化：val 占比 {is_val.mean():.2f}，调整 val_mod 或加样本")
    xt = torch.as_tensor(x[~is_val], device=device)
    yt = torch.as_tensor(y[~is_val], device=device)
    xv = torch.as_tensor(x[is_val], device=device)
    yv = torch.as_tensor(y[is_val], device=device)

    probe = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.GELU(), nn.Linear(hidden, y.shape[1])).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    curve = []
    for epoch in range(epochs):
        perm = torch.randperm(xt.shape[0], device=device)
        for i in range(0, xt.shape[0], batch_size):
            sel = perm[i : i + batch_size]
            loss = ((probe(xt[sel]) - yt[sel]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % max(1, epochs // 20) == 0:
            with torch.no_grad():
                val_mse = ((probe(xv) - yv) ** 2).mean().item()
            curve.append({"epoch": epoch + 1, "val_mse": val_mse})
    with torch.no_grad():
        pred = probe(xv)
        mse_dim = ((pred - yv) ** 2).mean(dim=0)
        var_dim = yv.var(dim=0, unbiased=False).clamp_min(1e-12)
        r2_dim = (1.0 - mse_dim / var_dim).cpu().numpy()
    return {
        "r2_per_dim": r2_dim.tolist(),
        "val_mse": float(((pred - yv) ** 2).mean().item()),
        "num_train": int(xt.shape[0]),
        "num_val": int(xv.shape[0]),
        "curve": curve,
    }


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[ced-diag] wrote {path}")
######### // code // ##########
