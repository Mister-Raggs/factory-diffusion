"""Run Experiment 2D: post-hoc one-chunk rollout-sensitivity diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from factory_diffusion.baselines.pusht_keypoints import (
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
    load_policy,
)
from factory_diffusion.closed_loop import PushTNormalization
from factory_diffusion.rollout_sensitivity import (
    collect_teacher_counterfactuals,
    summarize_sensitivity,
)
from factory_diffusion.schedules import standard_ddim_schedule

OPTIMIZED_SCHEDULES = {
    2: (70, 0),
    3: (80, 10, 0),
    4: (90, 50, 10, 0),
    5: (90, 70, 30, 10, 0),
}
DEFAULT_SEEDS = tuple(range(10))
DEFAULT_BUDGETS = (2, 3, 4, 5)


def _integer_sequence(raw: str) -> tuple[int, ...]:
    raw = raw.strip()
    if ":" in raw:
        parts = [int(value) for value in raw.split(":")]
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError("ranges use start:stop or start:stop:step")
        values = tuple(range(*parts))
    else:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected a non-empty sequence of unique integers")
    return values


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=_integer_sequence, default=DEFAULT_SEEDS)
    parser.add_argument("--budgets", type=_integer_sequence, default=DEFAULT_BUDGETS)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/pusht-keypoints"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface/hub"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiment2/rollout-sensitivity"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if any(budget not in OPTIMIZED_SCHEDULES for budget in args.budgets):
        parser.error("--budgets must be drawn from 2,3,4,5")

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    device = _device(args.device)
    policy = load_policy(
        device=device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    policy.eval()
    normalization = PushTNormalization.from_stats_file(args.dataset_root / "meta" / "stats.json")
    teacher_schedule = standard_ddim_schedule(
        policy.diffusion.noise_scheduler,
        policy.diffusion.num_inference_steps,
    )
    standard_schedules = {
        budget: standard_ddim_schedule(policy.diffusion.noise_scheduler, budget)
        for budget in args.budgets
    }
    schedules = {
        **{(budget, "standard"): standard_schedules[budget] for budget in args.budgets},
        **{(budget, "optimized"): OPTIMIZED_SCHEDULES[budget] for budget in args.budgets},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "partial.json"
    report_path = args.output_dir / "report.json"
    signature = {
        "seeds": list(args.seeds),
        "budgets": list(args.budgets),
        "max_steps": args.max_steps,
        "teacher_schedule": list(teacher_schedule),
        "checkpoint_revision": CHECKPOINT_REVISION,
    }
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing.get("run_signature") != signature:
            parser.error(f"{report_path} belongs to a different run configuration")
        print(f"already complete: {report_path}")
        return
    rows = []
    completed_seeds: set[int] = set()
    prior_wall_seconds = 0.0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("run_signature") != signature:
            parser.error(f"{progress_path} belongs to a different run configuration")
        rows = list(progress.get("rows", []))
        completed_seeds = {int(seed) for seed in progress.get("completed_seeds", [])}
        prior_wall_seconds = float(progress.get("wall_seconds", 0.0))

    started = time.perf_counter()
    for seed in args.seeds:
        if seed in completed_seeds:
            print(f"skip completed seed={seed}", flush=True)
            continue
        seed_rows = collect_teacher_counterfactuals(
            policy.diffusion,
            normalization=normalization,
            seed=seed,
            teacher_schedule=teacher_schedule,
            schedules=schedules,
            max_steps=args.max_steps,
        )
        rows.extend(row.to_dict() for row in seed_rows)
        completed_seeds.add(seed)
        _write_json_atomic(
            progress_path,
            {
                "status": "experiment-2d-partial",
                "run_signature": signature,
                "completed_seeds": sorted(completed_seeds),
                "wall_seconds": prior_wall_seconds + time.perf_counter() - started,
                "rows": rows,
            },
        )
        print(
            f"seed={seed} snapshots={len(seed_rows) // (2 * len(args.budgets))} "
            f"rows={len(seed_rows)}",
            flush=True,
        )

    summary = summarize_sensitivity(rows)
    report = {
        "status": "experiment-2d-rollout-sensitivity-diagnostic",
        "decision": "diagnostic-aligned" if summary["diagnostic_alignment"] else "stop-proxy",
        "run_signature": signature,
        "post_hoc": True,
        "checkpoint": CHECKPOINT_REPO,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "device": device,
        "standard_schedules": {
            str(budget): list(schedule) for budget, schedule in standard_schedules.items()
        },
        "optimized_schedules": {
            str(budget): list(OPTIMIZED_SCHEDULES[budget]) for budget in args.budgets
        },
        "summary": summary,
        "wall_seconds": prior_wall_seconds + time.perf_counter() - started,
        "rows": rows,
        "limitations": [
            "This post-hoc diagnostic cannot change Experiment 2 or Experiment 3 decisions.",
            "Branches cover one eight-action chunk from states visited by DDIM-10 only.",
            "Fresh branch environments reproduce body state but not Pymunk collision caches.",
        ],
    }
    _write_json_atomic(report_path, report)
    print(f"decision={report['decision']}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
