#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


EP_RE = re.compile(r"Total episodes:\s*(\d+)")
SUCCESS_RE = re.compile(r"# successes:\s*(\d+)\s*\(([0-9.]+)%\)")

SUITE_DESC = {
    "libero_spatial": "spatial",
    "libero_object": "object",
    "libero_goal": "goal",
    "libero_10": "long-horizon",
    "libero_90": "90-task",
}


def parse_eval_log(path: Path):
    text = path.read_text(errors="ignore")

    ep_matches = list(EP_RE.finditer(text))
    succ_matches = list(SUCCESS_RE.finditer(text))

    if not ep_matches or not succ_matches:
        return None, None

    episodes = int(ep_matches[-1].group(1))
    successes = int(succ_matches[-1].group(1))
    return successes, episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--suites", nargs="+", required=True)
    ap.add_argument("--num-trials-per-task", type=int, default=50)
    ap.add_argument("--max-tasks", type=int, default=-1)
    args = ap.parse_args()

    root = Path(args.root)
    out_file = root / "aggregate.txt"

    lines = []
    lines.append("=" * 72)
    lines.append("LIBERO aggregate")
    lines.append(f"root: {root}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Components:")
    for suite in args.suites:
        lines.append(f"  {suite:<15} = {SUITE_DESC.get(suite, 'unknown')}")
    lines.append("")

    total_successes = 0
    total_episodes = 0

    for suite in args.suites:
        suite_successes = 0
        suite_episodes = 0

        logs = sorted(root.glob(f"shard_*/{suite}/eval.log"))

        for log in logs:
            successes, episodes = parse_eval_log(log)
            if successes is None or episodes is None:
                continue

            suite_successes += successes
            suite_episodes += episodes

        rate = 100.0 * suite_successes / suite_episodes if suite_episodes > 0 else 0.0
        lines.append(f"{suite:<15}: {suite_successes:>4}/{suite_episodes:<4} = {rate:6.2f}%")

        total_successes += suite_successes
        total_episodes += suite_episodes

    lines.append("-" * 72)
    total_rate = 100.0 * total_successes / total_episodes if total_episodes > 0 else 0.0
    lines.append(f"{'TOTAL':<15}: {total_successes:>4}/{total_episodes:<4} = {total_rate:6.2f}%")
    lines.append("=" * 72)

    text = "\n".join(lines) + "\n"
    print(text, end="")

    root.mkdir(parents=True, exist_ok=True)
    out_file.write_text(text)
    print(f"[saved] {out_file}")


if __name__ == "__main__":
    main()
