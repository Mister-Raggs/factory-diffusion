"""Aggregate per-seed Phase 1 probe reports into a provisional decision."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--path-glob", default="**/report_seed_*.json")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase1"))
    parser.add_argument("--minimum-skip-fraction", type=float, default=0.2)
    parser.add_argument("--maximum-first-action-error", type=float, default=0.05)
    args = parser.parse_args()

    paths = sorted(args.input_root.glob(args.path_glob))
    if not paths:
        parser.error(f"no report_seed_*.json files found under {args.input_root}")
    reports = [json.loads(path.read_text()) for path in paths]
    cache_policies = {json.dumps(report.get("cache_policy"), sort_keys=True) for report in reports}
    if len(cache_policies) != 1:
        parser.error("all reports must use the same cache_policy")
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        for row in report["online_exact_scheduler_runs"]:
            grouped[float(row["threshold"])].append(row)

    thresholds = []
    for threshold, rows in sorted(grouped.items()):
        thresholds.append(
            {
                "threshold": threshold,
                "runs": len(rows),
                "complete": len(rows) == len(reports),
                "mean_skip_fraction": _mean([row["skip_fraction"] for row in rows]),
                "minimum_skip_fraction": min(row["skip_fraction"] for row in rows),
                "mean_first_action_max_error": _mean(
                    [row["first_action_max_error"] for row in rows]
                ),
                "maximum_first_action_max_error": max(
                    row["first_action_max_error"] for row in rows
                ),
                "mean_action_chunk_mse": _mean([row["action_chunk_mse"] for row in rows]),
                "maximum_action_chunk_error": max(row["action_chunk_max_error"] for row in rows),
                "median_observed_cpu_speedup": statistics.median(
                    row["observed_speedup"] for row in rows
                ),
            }
        )

    candidates = [
        row
        for row in thresholds
        if row["complete"]
        and row["minimum_skip_fraction"] >= args.minimum_skip_fraction
        and row["maximum_first_action_max_error"] <= args.maximum_first_action_error
    ]
    exact_zero = next((row for row in thresholds if row["threshold"] == 0), None)
    zero_is_exact = bool(
        exact_zero
        and exact_zero["maximum_first_action_max_error"] == 0
        and exact_zero["maximum_action_chunk_error"] == 0
    )
    decision = (
        "proceed-to-cuda-and-closed-loop-validation" if candidates and zero_is_exact else "stop"
    )
    summary = {
        "decision": decision,
        "decision_is_provisional": True,
        "seeds": sorted(report["seed"] for report in reports),
        "checkpoint": reports[0]["checkpoint"],
        "checkpoint_revision": reports[0]["checkpoint_revision"],
        "device": reports[0]["device"],
        "cache_policy": reports[0]["cache_policy"],
        "mean_denoiser_fraction": _mean(
            [report["baseline"]["denoiser_fraction"] for report in reports]
        ),
        "criteria": {
            "minimum_skip_fraction_each_seed": args.minimum_skip_fraction,
            "maximum_first_action_error_each_seed": args.maximum_first_action_error,
            "threshold_zero_must_be_exact": True,
        },
        "candidate_thresholds": [row["threshold"] for row in candidates],
        "thresholds": thresholds,
        "limitations": [
            (
                "The conditioning vectors are synthetic but normalized; this is not "
                "task-success evidence."
            ),
            "CPU timing is functional and must not be presented as a CUDA performance claim.",
            "The action-error criterion is a screening bound, not a task-derived tolerance.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Phase 1 preliminary result",
        "",
        f"Decision: **{decision}** (provisional).",
        "",
        f"Checkpoint: `{summary['checkpoint']}` at `{summary['checkpoint_revision']}`.",
        f"Seeds: `{summary['seeds']}`. Device: `{summary['device']}`.",
        f"Mean denoiser share: `{summary['mean_denoiser_fraction']:.3f}`.",
        "",
        (
            "| Threshold | Mean skip | Worst first-action error | Mean chunk MSE | "
            "Median CPU speedup* |"
        ),
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in thresholds:
        lines.append(
            f"| {row['threshold']:.4f} | {row['mean_skip_fraction']:.1%} | "
            f"{row['maximum_first_action_max_error']:.6f} | {row['mean_action_chunk_mse']:.6g} | "
            f"{row['median_observed_cpu_speedup']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "\\* Functional local timing only; not a publishable CUDA claim.",
            "",
            "This gate only establishes that adaptive reuse has measurable skip opportunities "
            "with bounded normalized action deviation. Task success and CUDA latency remain "
            "Phase 2.",
        ]
    )
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
