"""裸 ckpt → 可评测 run_dir 组装（从集群拖回的 ckpt 一条命令补齐 eval 资产）.

复用:
- starVLA.jointflow.data.joint_dataset.build_joint_dataloader (生成 dataset_statistics)
- starVLA.jointflow.data.mix_registry.resolve_data_mix (定位各套件 DINO stats)

背景 (v1 血泪教训): eval 链路用 run_dir = ckpt.parents[1]，要求布局
  <run_dir>/checkpoints/<ckpt>.pt + config.yaml + dataset_statistics.json
  (+ dino_v3_stats.json + vocab_map.json)。
集群训练的 run_dir 资产在 bucket 上；本地只拖 ckpt 时用本脚本按"本地数据根"重建资产
(统计与训练同源同算法，结果一致)。

用法:
  python starVLA/triflow/scripts/prepare_eval_run.py \
      --ckpt Z_tri/steps_100000_pytorch_model_ema.pt \
      --config_yaml starVLA/triflow/configs/triflow_libero_large.yaml \
      --run_dir Z_tri/e003_libero_all_large
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.triflow.data.mixtures import register_triflow_mixtures


######### // code // ##########
# 中文注释：多数据集 DINO stats 按样本数加权合并（与 TriFlowVLA._combine_dino_stats 同算法）。
def combine_dino_stats(stats_list: list[dict]) -> dict | None:
    if not stats_list:
        return None
    if len(stats_list) == 1:
        return stats_list[0]
    total = 0.0
    mean_acc = None
    m2_acc = None
    for stats in stats_list:
        count = float(stats.get("count", 0))
        if count <= 0:
            continue
        mean = torch.tensor(stats["mean"], dtype=torch.float64)
        std = torch.tensor(stats["std"], dtype=torch.float64).clamp_min(1e-12)
        m2 = std.square() + mean.square()
        mean_acc = count * mean if mean_acc is None else mean_acc + count * mean
        m2_acc = count * m2 if m2_acc is None else m2_acc + count * m2
        total += count
    if total <= 0 or mean_acc is None:
        return None
    mean = mean_acc / total
    var = (m2_acc / total - mean.square()).clamp_min(1e-12)
    return {"mean": mean.tolist(), "std": var.sqrt().tolist(), "count": int(total)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="裸 ckpt（.pt，建议 *_ema.pt）；可逗号分隔多个")
    parser.add_argument("--config_yaml", type=str, required=True, help="训练所用配置（数据根需指向本地）")
    parser.add_argument("--run_dir", type=str, required=True, help="目标 run_dir（将创建 checkpoints/ 等）")
    parser.add_argument("--copy", action="store_true", help="复制 ckpt（默认硬链接：零拷贝且对 .resolve() 透明）")
    args = parser.parse_args()

    register_triflow_mixtures()
    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1) ckpt 放入 checkpoints/（软链或复制）
    for raw in str(args.ckpt).split(","):
        src = Path(raw.strip()).resolve()
        assert src.exists(), f"ckpt not found: {src}"
        dst = ckpt_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        # 中文注释：默认硬链接——软链会被消费方 Path.resolve() 跟踪回裸文件路径，
        # 导致 run_dir=ckpt.parents[1] 算到错误目录（实测踩坑）；硬链接无此问题且零拷贝。
        if args.copy:
            shutil.copyfile(src, dst)
        else:
            try:
                os.link(src, dst)
            except OSError:  # 跨文件系统退化为复制
                shutil.copyfile(src, dst)
        print(f"[prepare_eval_run] ckpt -> {dst}{'' if args.copy else ' (hardlink)'}")

    # 2) config.yaml / config.full.yaml（完整解析版；from_pretrained 必需）
    cfg = OmegaConf.load(args.config_yaml)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    OmegaConf.save(cfg, run_dir / "config.full.yaml", resolve=True)
    print(f"[prepare_eval_run] config.yaml <- {args.config_yaml}")

    # 3) vocab_map.json（紧凑词表真源）
    vocab_src = Path(str(cfg.framework.text.vocab_map_path))
    if vocab_src.exists():
        shutil.copyfile(vocab_src, run_dir / "vocab_map.json")
        print(f"[prepare_eval_run] vocab_map.json <- {vocab_src}")
    else:
        print(f"[prepare_eval_run][warn] vocab_map missing: {vocab_src}（先跑 build_vocab）")

    # 4) dino_v3_stats.json（live DINO 归一化真源；按 data_mix 合并各套件统计）
    from starVLA.jointflow.data.mix_registry import resolve_data_mix

    paths = []
    for name, _, _ in resolve_data_mix(str(cfg.datasets.vla_data.data_mix)):
        p = Path(cfg.datasets.vla_data.data_root_dir) / name / str(cfg.datasets.vla_data.get("dino_feature_dir", "latents")) / "dino_v3_stats.json"
        if p.exists():
            paths.append(p)
    stats = combine_dino_stats([json.load(open(p)) for p in paths])
    if stats is not None:
        with open(run_dir / "dino_v3_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f)
        print(f"[prepare_eval_run] dino_v3_stats.json <- merged {len(paths)} suites (count={stats['count']})")

    # 5) dataset_statistics.json（动作反归一化真源；与训练同源生成）
    from starVLA.jointflow.data.joint_dataset import build_joint_dataloader

    cfg.datasets.vla_data.num_workers = 0
    dataset = build_joint_dataloader(cfg).dataset
    dataset.save_dataset_statistics(run_dir / "dataset_statistics.json")
    print("[prepare_eval_run] dataset_statistics.json generated")
    print(f"[prepare_eval_run] DONE -> {run_dir}")
######### // code // ##########


if __name__ == "__main__":
    main()
