"""CED E0.2 + V2：模型级诊断——配对 action-ablation gap 与 Δ 热力图。

对每个样本配对计算（共享 noise、t=0、fp32、eval 模式）：
  gap = L(cond⁰) − L(cond⁺)   （逐 patch；>0 表示真动作让 future 预测更准 = FDM 在用动作）
按 ‖z_H−z_t‖ top-10% 变化 patch 拆 gap_dyn / gap_static。基线 ckpt 预期 ≈0（correlational 解），
CED 训练后 gap_dyn 应单调上升——这是机制是否生效的直接读数。
V2 定性图：‖Δ_p‖（内生动作效果定位）vs ‖z_H−z_t‖（原始变化）并排。

复用：starVLA/jointflow/tools/debug_action_causality.py 的 _load_config/load_model/
collect_examples/compute_condition（ckpt→run_dir config 自动发现等行为完全一致）。

运行（starVLA env）：
  python starVLA/jointflow/scripts/diag_action_gap.py \
    --ckpt_path <run_dir>/checkpoints/steps_XXXX_pytorch_model.pt \
    --out playground/Debug_Checkpoints/ced_diag/E0.2
输出：<out>/action_gap.json + <out>/delta_heatmap_*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from starVLA.jointflow.data.joint_dataset import build_joint_dataset, compute_null_action_table
from starVLA.jointflow.scripts._probe_common import save_json
from starVLA.jointflow.tools.debug_action_causality import (
    _load_config,
    _resolve_path,
    collect_examples,
    compute_condition,
    load_model,
)


######### // code // ##########
# 中文注释：单 chunk 的配对 gap：K 个共享 noise draw 平均逐 patch loss（t=0），
# 返回 (gap_per_patch [B,N], delta_norm [B,N], change [B,N])。
@torch.inference_mode()
def paired_gap(model, cond_pos, cond_nul, z_h, z_t, num_noise: int):
    head = model.visual_head
    gap = torch.zeros(z_h.shape[0], z_h.shape[1], device=z_h.device)
    delta_norm = None
    for k in range(num_noise):
        eps = torch.randn_like(z_h)
        _, v_pos, pp_pos = head(cond_pos, z_h, noise=eps, t=0.0, return_pred=True)
        _, v_nul, pp_nul = head(cond_nul, z_h, noise=eps, t=0.0, return_pred=True)
        gap += pp_nul - pp_pos
        if k == 0:
            delta_norm = (v_pos.float() - v_nul.float()).norm(dim=-1)
    change = (z_h.float() - z_t.float()).norm(dim=-1)
    return gap / num_noise, delta_norm, change
######### // code // ##########


def _grid(x: np.ndarray) -> np.ndarray:
    side = int(round(len(x) ** 0.5))
    return x[: side * side].reshape(side, side)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_yaml", type=str, default=None)
    parser.add_argument("--data_root_dir", type=str, default=None)
    parser.add_argument("--data_mix", type=str, default=None)
    parser.add_argument("--dino_feature_dir", type=str, default=None)
    parser.add_argument("--num_inference_timesteps", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_noise", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--num_viz", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="playground/Debug_Checkpoints/ced_diag/E0.2")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = _resolve_path(args.ckpt_path)
    cfg, config_path = _load_config(args, ckpt)
    print(f"[E0.2] config={config_path}")
    dataset = build_joint_dataset(cfg, mode="train")
    examples = collect_examples(dataset, num_samples=args.num_samples, seed=args.seed, start_index=None, include_invalid=False)
    model, incompatible = load_model(cfg, ckpt, device, strict=False)
    if incompatible.missing_keys:
        print(f"[E0.2] missing keys (ok for baseline ckpt without CED modules): {incompatible.missing_keys[:4]} ...")
    model.set_null_action_table(compute_null_action_table(dataset))

    records, viz = [], []
    g_all = g_dyn = g_sta = 0.0
    n_all = 0
    for i in range(0, len(examples), args.chunk):
        chunk = examples[i : i + args.chunk]
        cond_pos, z_h, z_t, valid = compute_condition(model, chunk, "fdm")
        kept = [ex for ex, keep in zip(chunk, valid.tolist()) if keep]
        if not kept:
            continue
        action = torch.as_tensor(np.stack([np.asarray(ex["action"], dtype=np.float32) for ex in kept]))
        null = model.make_null_action(action, kept)
        chunk_nul = [{**ex, "action": null[j].cpu().numpy()} for j, ex in enumerate(kept)]
        cond_nul, _, _, _ = compute_condition(model, chunk_nul, "fdm")

        gap, delta_norm, change = paired_gap(model, cond_pos, cond_nul, z_h, z_t, args.num_noise)
        k = max(1, int(round(0.1 * change.shape[1])))
        thresh = change.topk(k, dim=1).values[:, -1:]
        dyn = change >= thresh
        for j, ex in enumerate(kept):
            rec = {
                "dataset_name": str(ex.get("dataset_name")),
                "trajectory_id": int(ex.get("trajectory_id", -1)),
                "base_index": int(ex.get("base_index", -1)),
                "gap_all": float(gap[j].mean()),
                "gap_dyn": float(gap[j][dyn[j]].mean()),
                "gap_static": float(gap[j][~dyn[j]].mean()) if bool((~dyn[j]).any()) else None,
                "delta_norm_mean": float(delta_norm[j].mean()),
            }
            records.append(rec)
            g_all += rec["gap_all"]
            g_dyn += rec["gap_dyn"]
            g_sta += rec["gap_static"] or 0.0
            n_all += 1
            if len(viz) < args.num_viz:
                viz.append((ex, delta_norm[j].cpu().numpy(), change[j].cpu().numpy()))

    assert n_all > 0, "没有有效样本"
    summary = {
        "ckpt": str(ckpt),
        "num_samples": n_all,
        "num_noise": args.num_noise,
        "gap_all_mean": g_all / n_all,
        "gap_dyn_mean": g_dyn / n_all,
        "gap_static_mean": g_sta / n_all,
        "readout_hint": "基线预期 ≈0；CED 训练后 gap_dyn 应显著>0 且随训练上升",
        "per_sample": records,
    }
    out_dir = Path(args.out)
    save_json(out_dir / "action_gap.json", summary)
    print(f"[E0.2] gap_all={summary['gap_all_mean']:.6f} gap_dyn={summary['gap_dyn_mean']:.6f} gap_static={summary['gap_static_mean']:.6f}")

    # ---- V2：‖Δ_p‖ vs ‖z_H−z_t‖ 并排热力图（有原图则加第三列）----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for idx, (ex, dmap, cmap_) in enumerate(viz):
        has_img = "image_0" in ex
        fig, axes = plt.subplots(1, 3 if has_img else 2, figsize=(9 if has_img else 6, 3))
        axes[0].imshow(_grid(dmap), cmap="inferno")
        axes[0].set_title("|Δ| (action effect)")
        axes[1].imshow(_grid(cmap_), cmap="viridis")
        axes[1].set_title("|z_H − z_t| (raw change)")
        if has_img:
            view = 0
            axes[2].imshow(np.asarray(ex["image_0"])[view])
            axes[2].set_title("o_t")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{ex.get('dataset_name')} traj={ex.get('trajectory_id')} t={ex.get('base_index')}", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"delta_heatmap_{idx:02d}.png", dpi=150)
        plt.close(fig)
    print(f"[E0.2] wrote {len(viz)} heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
