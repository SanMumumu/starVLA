#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

SUITE_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}

SUCCESS_RE = re.compile(r"Success:\s*(True|False|true|false|1|0)")
TOTAL_RATE_RE = re.compile(r"Total success rate:\s*([0-9.]+)")
TOTAL_EPISODES_RE = re.compile(r"Total episodes:\s*(\d+)")


def parse_log(path: Path) -> tuple[int, int]:
    text = path.read_text(errors="replace")
    successes = 0
    episodes = 0

    for match in SUCCESS_RE.finditer(text):
        episodes += 1
        successes += match.group(1).lower() in {"true", "1"}

    if episodes > 0:
        return episodes, successes

    rate_matches = TOTAL_RATE_RE.findall(text)
    episode_matches = TOTAL_EPISODES_RE.findall(text)
    if rate_matches and episode_matches:
        episodes = int(episode_matches[-1])
        successes = round(float(rate_matches[-1]) * episodes)
        return episodes, successes

    return 0, 0


def expected_episodes(suite: str, num_trials_per_task: int, max_tasks: int) -> int | None:
    num_tasks = SUITE_TASK_COUNTS.get(suite)
    if num_tasks is None:
        return None
    if max_tasks > 0:
        num_tasks = min(num_tasks, max_tasks)
    return num_tasks * num_trials_per_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--suites", nargs="+", required=True)
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--max-tasks", type=int, default=-1)
    args = parser.parse_args()

    summary = {
        "root": str(args.root),
        "num_trials_per_task": args.num_trials_per_task,
        "max_tasks": args.max_tasks,
        "suites": {},
    }
    overall_episodes = 0
    overall_successes = 0

    print("LIBERO shard aggregate")
    print(f"root: {args.root}")
    print()

    for suite in args.suites:
        log_paths = sorted(args.root.glob(f"shard_*/{suite}/eval.log"))
        episodes = 0
        successes = 0
        rows = []

        for log_path in log_paths:
            shard_episodes, shard_successes = parse_log(log_path)
            episodes += shard_episodes
            successes += shard_successes
            rows.append(
                {
                    "log": str(log_path),
                    "episodes": shard_episodes,
                    "successes": shard_successes,
                    "success_rate": (shard_successes / shard_episodes if shard_episodes else None),
                }
            )

        expected = expected_episodes(suite, args.num_trials_per_task, args.max_tasks)
        rate = successes / episodes if episodes else 0.0
        complete = expected is None or episodes == expected

        print(
            f"{suite}: {successes}/{episodes} ({rate * 100:.2f}%)"
            + (f", expected={expected}, complete={complete}" if expected is not None else "")
        )

        summary["suites"][suite] = {
            "episodes": episodes,
            "successes": successes,
            "success_rate": rate,
            "expected_episodes": expected,
            "complete": complete,
            "logs": rows,
        }
        overall_episodes += episodes
        overall_successes += successes

    overall_rate = overall_successes / overall_episodes if overall_episodes else 0.0
    summary["overall"] = {
        "episodes": overall_episodes,
        "successes": overall_successes,
        "success_rate": overall_rate,
    }

    print()
    print(f"overall: {overall_successes}/{overall_episodes} ({overall_rate * 100:.2f}%)")

    args.root.mkdir(parents=True, exist_ok=True)
    out_path = args.root / "aggregate_results.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
