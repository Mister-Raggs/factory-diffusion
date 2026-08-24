"""Run Experiment 3: paired closed-loop PushT schedule evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

import torch

from factory_diffusion.baselines.pusht_data import DATASET_REPO, DATASET_REVISION
from factory_diffusion.baselines.pusht_keypoints import (
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
    load_policy,
)
from factory_diffusion.closed_loop import (
    PushTNormalization,
    rollout_pusht_episode,
    summarize_paired_episodes,
)
from factory_diffusion.schedules import standard_ddim_schedule

OPTIMIZED_SCHEDULES = {
    2: (70, 0),
    3: (80, 10, 0),
    4: (90, 50, 10, 0),
    5: (90, 70, 30, 10, 0),
}
FULL_EVALUATION_SEEDS = tuple(range(50))
FULL_BUDGETS = (2, 3, 4, 5)


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


def _run_signature(*, seeds: tuple[int, ...], budgets: tuple[int, ...], max_steps: int) -> dict:
    return {
        "seeds": list(seeds),
        "budgets": list(budgets),
        "max_steps": max_steps,
        "optimized_schedules": {
            str(budget): list(OPTIMIZED_SCHEDULES[budget]) for budget in budgets
        },
        "checkpoint_revision": CHECKPOINT_REVISION,
        "dataset_revision": DATASET_REVISION,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=_integer_sequence, default=FULL_EVALUATION_SEEDS)
    parser.add_argument("--budgets", type=_integer_sequence, default=FULL_BUDGETS)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/pusht-keypoints"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface/hub"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment3/closed-loop"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if any(budget not in OPTIMIZED_SCHEDULES for budget in args.budgets):
        parser.error("--budgets must be drawn from 2,3,4,5")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    progress_path = args.output_dir / "partial.json"
    signature = _run_signature(
        seeds=args.seeds,
        budgets=args.budgets,
        max_steps=args.max_steps,
    )
    if report_path.exists():
        existing_report = json.loads(report_path.read_text())
        if existing_report.get("run_signature") != signature:
            parser.error(f"{report_path} belongs to a different run configuration")
        print(f"already complete: {report_path}")
        return
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("run_signature") != signature:
            parser.error(f"{progress_path} belongs to a different run configuration")
        rows = list(progress.get("rows", []))
        prior_wall_seconds = float(progress.get("wall_seconds", 0.0))
    else:
        rows = []
        prior_wall_seconds = 0.0

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

    standard_schedules = {
        budget: standard_ddim_schedule(policy.diffusion.noise_scheduler, budget)
        for budget in args.budgets
    }
    started = time.perf_counter()
    total_episodes = len(args.seeds) * len(args.budgets) * 2
    completed = {(int(row["seed"]), int(row["budget"]), str(row["method"])) for row in rows}
    for seed_index, seed in enumerate(args.seeds):
        for budget in args.budgets:
            methods = [
                ("standard", standard_schedules[budget]),
                ("optimized", OPTIMIZED_SCHEDULES[budget]),
            ]
            if (seed_index + budget) % 2:
                methods.reverse()
            for method, schedule in methods:
                key = (seed, budget, method)
                if key in completed:
                    print(
                        f"skip completed seed={seed} k={budget} method={method}",
                        flush=True,
                    )
                    continue
                result = rollout_pusht_episode(
                    policy.diffusion,
                    normalization=normalization,
                    method=method,
                    schedule=schedule,
                    seed=seed,
                    max_steps=args.max_steps,
                )
                rows.append(result.to_dict())
                completed.add(key)
                _write_json_atomic(
                    progress_path,
                    {
                        "status": "experiment-3-partial",
                        "run_signature": signature,
                        "completed_episodes": len(rows),
                        "total_episodes": total_episodes,
                        "wall_seconds": prior_wall_seconds + time.perf_counter() - started,
                        "rows": rows,
                    },
                )
                print(
                    f"episode {len(rows)}/{total_episodes} seed={seed} k={budget} "
                    f"method={method} success={result.success} "
                    f"coverage={result.max_coverage:.3f} nfe={result.total_nfe}",
                    flush=True,
                )

    statistical_summary = summarize_paired_episodes(
        rows,
        budgets=args.budgets,
        seeds=args.seeds,
    )
    full_protocol = (
        args.seeds == FULL_EVALUATION_SEEDS
        and args.budgets == FULL_BUDGETS
        and args.max_steps == 300
    )
    if full_protocol:
        decision = statistical_summary["decision"]
    else:
        decision = "smoke-only"
        statistical_summary["decision"] = decision
    report = {
        "status": "experiment-3-closed-loop-pusht",
        "decision": decision,
        "full_protocol": full_protocol,
        "run_signature": signature,
        "checkpoint": CHECKPOINT_REPO,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "dataset": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "gym_pusht_version": importlib.metadata.version("gym-pusht"),
        "lerobot_version": importlib.metadata.version("lerobot"),
        "device": device,
        "seeds": list(args.seeds),
        "budgets": list(args.budgets),
        "max_steps": args.max_steps,
        "standard_schedules": {
            str(budget): list(schedule) for budget, schedule in standard_schedules.items()
        },
        "optimized_schedules": {
            str(budget): list(OPTIMIZED_SCHEDULES[budget]) for budget in args.budgets
        },
        "protocol": {
            "observation_type": "environment_state_agent_pos",
            "n_obs_steps": 2,
            "n_action_steps": 8,
            "success_threshold": 0.95,
            "noninferiority_margin": -0.05,
            "pairing": "same environment seed and per-query initial diffusion noise",
            "rendering": False,
        },
        "wall_seconds": prior_wall_seconds + time.perf_counter() - started,
        "summary": statistical_summary,
        "rows": rows,
        "limitations": [
            "The keypoint policy does not include visual perception latency or error.",
            "Equal NFE is a compute control, not a hardware wall-clock claim.",
            "Smoke configurations validate integration only and cannot change the protocol.",
        ],
    }
    _write_json_atomic(report_path, report)
    print(f"decision={decision}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
