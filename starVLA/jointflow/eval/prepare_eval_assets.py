"""Generate eval-time assets next to a JointFlow checkpoint.

所属：JointFlow LIBERO eval。
复用：
- starVLA.jointflow.data.joint_dataset.build_joint_dataset（构 LeRobotMixtureDataset）
- LeRobotMixtureDataset.save_dataset_statistics（写 dataset_statistics.json）
- starVLA.jointflow.data.mix_registry.resolve_data_mix（取 mixture 里的数据集名）
- starVLA.model.framework.share_tools.apply_config_compat

为何需要本脚本（关键缺口）：
JointFlow 训练（train_jointflow.py）只保存 config.yaml + 权重，**从不保存
dataset_statistics.json**；但 server 端 read_mode_config / PolicyNormProcessor
强制要求 ckpt 同级 run_dir 下存在 dataset_statistics.json，否则起 server 直接
assert 失败。本脚本离线补齐该文件（动作/状态反归一化与状态归一化都依赖它）。
顺带把各数据集的 DINOv3 归一化统计合并成一个 dino_v3_stats.json 落到 run_dir，
作为 eval 跨机时 framework 加载 DINO norm 的可移植兜底。

为何不改训练代码：用户要求"只在 jointflow 下新增、不污染外部"，且训练已完成；
用独立的离线 prep 步骤补资产，比回灌训练入口更安全。

run_dir 约定：与 read_mode_config 一致——ckpt 形如 <run_dir>/checkpoints/x.pt，
故 run_dir = ckpt.parents[1]。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.jointflow.data.joint_dataset import build_joint_dataset
from starVLA.jointflow.data.mix_registry import resolve_data_mix
from starVLA.model.framework.share_tools import apply_config_compat


######### // code // ##########
# 中文注释：从 ckpt 路径或显式 run_dir 解析出 run_dir，并读取其中的 config.yaml。
# 注意：这里**不能**用 read_mode_config——它会断言 dataset_statistics.json 已存在，
# 而本脚本正是要生成该文件。所以直接 OmegaConf.load(config.yaml)。
def _resolve_run_dir_and_cfg(ckpt_path: str | None, run_dir: str | None):
    if run_dir:
        run_dir_path = Path(run_dir)
    elif ckpt_path:
        ckpt = Path(ckpt_path)
        run_dir_path = ckpt.parents[1]
    else:
        raise ValueError("Provide either --ckpt_path or --run_dir.")

    config_yaml = run_dir_path / "config.yaml"
    if not config_yaml.exists():
        raise FileNotFoundError(f"Missing config.yaml at {config_yaml}; cannot build dataset for stats.")
    cfg = OmegaConf.load(str(config_yaml))
    apply_config_compat(cfg)
    return run_dir_path, cfg
######### // code // ##########


######### // code // ##########
# 中文注释：生成 dataset_statistics.json。
# 构出与训练同款的 LeRobotMixtureDataset（含 state/action modality），其内部
# 已计算/合并好统计，直接 save_dataset_statistics 落盘。幂等：已存在则跳过。
def generate_dataset_statistics(run_dir: Path, cfg, force: bool = False) -> Path:
    out_path = run_dir / "dataset_statistics.json"
    if out_path.exists() and not force:
        print(f"[prepare_eval_assets] dataset_statistics.json exists, skip: {out_path}")
        return out_path
    print("[prepare_eval_assets] building dataset to compute statistics ...")
    dataset = build_joint_dataset(cfg, mode="train")
    dataset.save_dataset_statistics(out_path)
    print(f"[prepare_eval_assets] wrote {out_path}")
    return out_path
######### // code // ##########


######### // code // ##########
# 中文注释：count 加权合并多个数据集的 DINO per-channel mean/std。
# 与 framework._combine_dino_stats 同公式：用二阶矩合并再开方得到合并 std。
def _combine_dino_stats(stats_list: list[dict]) -> dict | None:
    stats_list = [s for s in stats_list if s]
    if not stats_list:
        return None
    if len(stats_list) == 1:
        return stats_list[0]

    dim = len(stats_list[0]["mean"])
    total = 0.0
    mean_acc = [0.0] * dim
    second_acc = [0.0] * dim
    for stats in stats_list:
        count = float(stats.get("count", 0)) or 1.0
        mean = stats["mean"]
        std = stats["std"]
        for i in range(dim):
            m = float(mean[i])
            s = max(float(std[i]), 1e-12)
            mean_acc[i] += count * m
            second_acc[i] += count * (s * s + m * m)
        total += count
    if total <= 0:
        return None
    mean = [mean_acc[i] / total for i in range(dim)]
    std = [math.sqrt(max(second_acc[i] / total - mean[i] * mean[i], 1e-12)) for i in range(dim)]
    return {"mean": mean, "std": std, "count": int(total)}
######### // code // ##########


######### // code // ##########
# 中文注释：把 mixture 里各数据集的 latents/dino_v3_stats.json 合并写到 run_dir，
# 作为 eval 跨机加载 DINO norm 的兜底。挂载了数据根时 framework 也能自行加载，
# 此处只是冗余保险。找不到任何 per-dataset 统计则跳过（不报错）。
def bundle_dino_stats(run_dir: Path, cfg, force: bool = False) -> Path | None:
    out_path = run_dir / "dino_v3_stats.json"
    if out_path.exists() and not force:
        print(f"[prepare_eval_assets] dino_v3_stats.json exists, skip: {out_path}")
        return out_path

    vla = cfg.datasets.vla_data
    data_root = vla.get("data_root_dir", None)
    data_mix = vla.get("data_mix", None)
    feature_dir = vla.get("dino_feature_dir", "latents")
    if not data_root or not data_mix:
        print("[prepare_eval_assets] no data_root_dir/data_mix in cfg; skip DINO stats bundling.")
        return None

    per_dataset = []
    for data_name, _, _ in resolve_data_mix(str(data_mix)):
        stats_path = Path(data_root) / str(data_name) / str(feature_dir) / "dino_v3_stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                per_dataset.append(json.load(f))
        else:
            print(f"[prepare_eval_assets] missing per-dataset DINO stats: {stats_path}")

    combined = _combine_dino_stats(per_dataset)
    if combined is None:
        print("[prepare_eval_assets] no DINO stats found; skip bundling (framework will read data root at eval).")
        return None
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"[prepare_eval_assets] wrote {out_path} (combined from {len(per_dataset)} datasets)")
    return out_path
######### // code // ##########


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to <run_dir>/checkpoints/x.pt")
    parser.add_argument("--run_dir", type=str, default=None, help="Run dir holding config.yaml (overrides ckpt_path).")
    parser.add_argument("--force", action="store_true", help="Regenerate even if outputs exist.")
    parser.add_argument("--skip_dino_stats", action="store_true", help="Do not bundle dino_v3_stats.json.")
    args = parser.parse_args()

    run_dir, cfg = _resolve_run_dir_and_cfg(args.ckpt_path, args.run_dir)
    print(f"[prepare_eval_assets] run_dir={run_dir}")
    generate_dataset_statistics(run_dir, cfg, force=args.force)
    if not args.skip_dino_stats:
        bundle_dino_stats(run_dir, cfg, force=args.force)
    print("[prepare_eval_assets] done.")


if __name__ == "__main__":
    main()
