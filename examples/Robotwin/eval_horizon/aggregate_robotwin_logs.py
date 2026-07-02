#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

SUCCESS_RE = re.compile(r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>\s*([0-9.]+)%")


def parse_log(path: Path) -> dict:
    matches = SUCCESS_RE.findall(path.read_text(errors="replace"))
    if not matches:
        return {
            "log": str(path),
            "success": 0,
            "total": 0,
            "rate": 0.0,
            "complete": False,
        }

    success, total, rate_pct = matches[-1]
    success_i = int(success)
    total_i = int(total)
    return {
        "log": str(path),
        "success": success_i,
        "total": total_i,
        "rate": success_i / total_i if total_i else 0.0,
        "reported_rate": float(rate_pct) / 100.0,
        "complete": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    records = []
    by_mode: dict[str, dict[str, float | int]] = {}

    for log_path in sorted((args.output_dir / "logs").glob("*/*.log")):
        mode = log_path.parent.name
        task = log_path.name.split("_shard", 1)[0]
        rec = parse_log(log_path)
        rec["mode"] = mode
        rec["task"] = task
        records.append(rec)

        bucket = by_mode.setdefault(mode, {"success": 0, "total": 0, "tasks": 0, "complete_tasks": 0})
        bucket["success"] += rec["success"]
        bucket["total"] += rec["total"]
        bucket["tasks"] += 1
        bucket["complete_tasks"] += int(rec["complete"])

    for bucket in by_mode.values():
        total = int(bucket["total"])
        bucket["rate"] = float(bucket["success"]) / total if total else 0.0

    overall_success = sum(int(rec["success"]) for rec in records)
    overall_total = sum(int(rec["total"]) for rec in records)
    result = {
        "output_dir": str(args.output_dir),
        "modes": by_mode,
        "overall": {
            "success": overall_success,
            "total": overall_total,
            "rate": overall_success / overall_total if overall_total else 0.0,
            "tasks": len(records),
            "complete_tasks": sum(int(rec["complete"]) for rec in records),
        },
        "records": records,
    }

    out = args.output_dir / "aggregate_results.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("RoboTwin aggregate")
    for mode, bucket in sorted(by_mode.items()):
        print(
            f"{mode}: {int(bucket['success'])}/{int(bucket['total'])} "
            f"({float(bucket['rate']) * 100:.2f}%), "
            f"tasks={int(bucket['complete_tasks'])}/{int(bucket['tasks'])}"
        )
    print(
        f"overall: {overall_success}/{overall_total} "
        f"({result['overall']['rate'] * 100:.2f}%), "
        f"tasks={result['overall']['complete_tasks']}/{result['overall']['tasks']}"
    )
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
