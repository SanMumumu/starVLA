"""Aggregate per-shard JointFlow LIBERO eval results into per-suite + total SR.

所属：JointFlow LIBERO eval（结果汇总）。
复用：仅标准库。
说明：各 client 分片把结果写成 <results_dir>/shard*.json（结构见
eval_libero_jointflow.py）。本脚本把同一套件的 successes/episodes 累加，
按 (suite,task) 去重合并（分片不重叠，正常不会撞），输出总体与分套件成功率。
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


######### // code // ##########
# 中文注释：读取 results_dir 下所有 shard*.json，合并 per_suite -> per_task 计数。
def aggregate(results_dir: str, pattern: str = "shard*.json") -> dict:
    files = sorted(glob.glob(str(Path(results_dir) / pattern)))
    if not files:
        raise FileNotFoundError(f"No shard result files matching {pattern} under {results_dir}")

    suites: dict = {}
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for suite, sd in data.get("per_suite", {}).items():
            agg = suites.setdefault(suite, {"successes": 0, "episodes": 0, "per_task": {}})
            for task_id, td in sd.get("per_task", {}).items():
                # 中文注释：分片不重叠；若重复出现同 task（异常），取累加以暴露问题。
                cur = agg["per_task"].setdefault(task_id, {"successes": 0, "episodes": 0})
                cur["successes"] += int(td.get("successes", 0))
                cur["episodes"] += int(td.get("episodes", 0))
            agg["successes"] += int(sd.get("successes", 0))
            agg["episodes"] += int(sd.get("episodes", 0))

    total_s = sum(v["successes"] for v in suites.values())
    total_e = sum(v["episodes"] for v in suites.values())
    summary = {
        "num_shard_files": len(files),
        "per_suite": {
            s: {
                "successes": v["successes"],
                "episodes": v["episodes"],
                "success_rate": v["successes"] / max(v["episodes"], 1),
                "per_task": v["per_task"],
            }
            for s, v in suites.items()
        },
        "total": {
            "successes": total_s,
            "episodes": total_e,
            "success_rate": total_s / max(total_e, 1),
        },
    }
    return summary
######### // code // ##########


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, required=True, help="Dir containing shard*.json")
    p.add_argument("--pattern", type=str, default="shard*.json")
    p.add_argument("--out", type=str, default="", help="Optional combined summary JSON path")
    args = p.parse_args()

    summary = aggregate(args.results_dir, args.pattern)

    print("=" * 60)
    print(f"JointFlow LIBERO eval summary ({summary['num_shard_files']} shard files)")
    print("=" * 60)
    for suite, sd in sorted(summary["per_suite"].items()):
        print(f"  {suite:18s} SR={sd['success_rate'] * 100:6.2f}%  ({sd['successes']}/{sd['episodes']})")
    t = summary["total"]
    print("-" * 60)
    print(f"  {'TOTAL':18s} SR={t['success_rate'] * 100:6.2f}%  ({t['successes']}/{t['episodes']})")
    print("=" * 60)

    out = args.out or str(Path(args.results_dir) / "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
