#!/usr/bin/env python3
import argparse
import json
import pathlib
from collections import defaultdict
from typing import Any

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DISPLAY = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
}


def rate(success: int, total: int) -> dict[str, Any]:
    return {
        "success": success,
        "total": total,
        "rate": success / total if total else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--expected-total", type=int, default=10030)
    args = p.parse_args()

    root = pathlib.Path(args.output_dir)
    records: dict[str, dict[str, Any]] = {}
    missing = []
    for shard_id in range(args.num_shards):
        path = root / "shards" / f"shard_{shard_id:02d}.jsonl"
        if not path.is_file():
            missing.append(str(path))
            continue
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"WARNING: ignore corrupt line {path}:{line_no}")
                    continue
                key = rec["key"]
                if key in records and records[key] != rec:
                    raise RuntimeError(f"conflicting duplicate: {key}")
                records[key] = rec

    if missing:
        raise RuntimeError("missing shard files:\n" + "\n".join(missing))

    suite = defaultdict(lambda: [0, 0])
    category = defaultdict(lambda: [0, 0])
    suite_category = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for rec in records.values():
        ok = int(bool(rec["success"]))
        s = rec["suite"]
        c = rec["category"]
        suite[s][0] += ok
        suite[s][1] += 1
        category[c][0] += ok
        category[c][1] += 1
        suite_category[s][c][0] += ok
        suite_category[s][c][1] += 1

    suite_results = {s: rate(*suite[s]) for s in SUITES}
    category_results = {c: rate(*v) for c, v in sorted(category.items())}
    suite_category_results = {
        s: {c: rate(*v) for c, v in sorted(suite_category[s].items())}
        for s in SUITES
    }
    macro_avg = sum(suite_results[s]["rate"] for s in SUITES) / len(SUITES)
    total_success = sum(int(r["success"]) for r in records.values())
    total = len(records)

    result = {
        "complete": total == args.expected_total,
        "num_records": total,
        "expected_total": args.expected_total,
        "suite_results": suite_results,
        "macro_suite_average": macro_avg,
        "overall_micro": rate(total_success, total),
        "category_results": category_results,
        "suite_category_results": suite_category_results,
    }
    out = root / "aggregate_results.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n========== LIBERO-Plus: four suites ==========")
    print(f"{'Suite':<12}{'Success':>10}{'Total':>10}{'Rate':>11}")
    for s in SUITES:
        x = suite_results[s]
        print(f"{DISPLAY[s]:<12}{x['success']:>10}{x['total']:>10}{x['rate']*100:>10.2f}%")
    print(f"{'Macro Avg':<12}{'-':>10}{'-':>10}{macro_avg*100:>10.2f}%")
    micro = result["overall_micro"]
    print(f"{'Micro All':<12}{micro['success']:>10}{micro['total']:>10}{micro['rate']*100:>10.2f}%")

    print("\n========== LIBERO-Plus: perturbations ==========")
    print(f"{'Category':<30}{'Success':>10}{'Total':>10}{'Rate':>11}")
    for c, x in category_results.items():
        print(f"{c:<30}{x['success']:>10}{x['total']:>10}{x['rate']*100:>10.2f}%")

    print(f"\nrecords: {total}/{args.expected_total}")
    print(f"saved: {out}")
    if total != args.expected_total:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
