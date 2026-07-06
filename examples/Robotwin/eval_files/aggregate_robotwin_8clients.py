#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

RATE_RE = re.compile(
    r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>\s*([0-9]+(?:\.[0-9]+)?)%"
)


@dataclass
class Result:
    job_id: int
    mode: str
    task: str
    slot: int
    gpu: str
    port: int
    log: Path
    successes: int | None
    episodes: int | None
    rate_pct: float | None


def parse_log(path: Path) -> tuple[int | None, int | None, float | None]:
    if not path.is_file():
        return None, None, None

    matches = RATE_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        return None, None, None

    success, episodes, rate = matches[-1]
    return int(success), int(episodes), float(rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []

    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            log = Path(row["log"])
            successes, episodes, rate = parse_log(log)
            results.append(
                Result(
                    job_id=int(row["job_id"]),
                    mode=row["mode"],
                    task=row["task"],
                    slot=int(row["slot"]),
                    gpu=row["gpu"],
                    port=int(row["port"]),
                    log=log,
                    successes=successes,
                    episodes=episodes,
                    rate_pct=rate,
                )
            )

    results.sort(key=lambda item: item.job_id)
    csv_path = args.output_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "job_id",
                "mode",
                "task",
                "slot",
                "gpu",
                "port",
                "successes",
                "episodes",
                "success_rate_pct",
                "status",
                "log",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.job_id,
                    item.mode,
                    item.task,
                    item.slot,
                    item.gpu,
                    item.port,
                    "" if item.successes is None else item.successes,
                    "" if item.episodes is None else item.episodes,
                    "" if item.rate_pct is None else f"{item.rate_pct:.4f}",
                    "complete" if item.rate_pct is not None else "missing",
                    str(item.log),
                ]
            )

    by_mode: dict[str, list[Result]] = defaultdict(list)
    for item in results:
        by_mode[item.mode].append(item)

    lines = [
        "StarVLA RoboTwin evaluation summary",
        f"Manifest: {args.manifest}",
        f"Jobs: {len(results)}",
        "",
    ]

    all_complete_rates: list[float] = []
    all_successes = 0
    all_episodes = 0

    for mode, mode_results in sorted(by_mode.items()):
        complete = [item for item in mode_results if item.rate_pct is not None]
        missing = [item for item in mode_results if item.rate_pct is None]

        macro = (
            sum(item.rate_pct for item in complete if item.rate_pct is not None)
            / len(complete)
            if complete
            else float("nan")
        )
        successes = sum(item.successes or 0 for item in complete)
        episodes = sum(item.episodes or 0 for item in complete)
        micro = 100.0 * successes / episodes if episodes else float("nan")

        all_complete_rates.extend(
            item.rate_pct for item in complete if item.rate_pct is not None
        )
        all_successes += successes
        all_episodes += episodes

        lines.extend(
            [
                f"[{mode}]",
                f"completed tasks: {len(complete)}/{len(mode_results)}",
                f"macro task average: {macro:.4f}%",
                f"micro episode success: {micro:.4f}% ({successes}/{episodes})",
            ]
        )
        if missing:
            lines.append("missing: " + ", ".join(item.task for item in missing))
        lines.append("")

    overall_macro = (
        sum(all_complete_rates) / len(all_complete_rates)
        if all_complete_rates
        else float("nan")
    )
    overall_micro = (
        100.0 * all_successes / all_episodes if all_episodes else float("nan")
    )
    lines.extend(
        [
            "[overall]",
            f"completed task-mode jobs: {len(all_complete_rates)}/{len(results)}",
            f"macro task-mode average: {overall_macro:.4f}%",
            f"micro episode success: {overall_micro:.4f}% "
            f"({all_successes}/{all_episodes})",
            "",
            f"Detailed CSV: {csv_path}",
        ]
    )

    summary_path = args.output_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
