"""Run Experiment 2: calibration-only optimization of few-step DDIM schedules."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch import Tensor

from factory_diffusion.baselines.pusht_data import (
    DATASET_REPO,
    DATASET_REVISION,
    PushTConditioningSample,
    load_real_conditioning_samples,
)
from factory_diffusion.baselines.pusht_keypoints import (
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
    load_policy,
)
from factory_diffusion.evaluation import run_explicit_schedule_sampler
from factory_diffusion.schedule_search import (
    ScheduleScore,
    paired_bootstrap_mean_ci,
    per_sample_action_errors,
    select_schedule,
)
from factory_diffusion.schedules import grid_schedules, standard_ddim_schedule


def _int_list(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _action_pixel_scale(dataset_root: Path) -> Tensor:
    stats = json.loads((dataset_root / "meta" / "stats.json").read_text())
    minimum = torch.tensor(stats["action"]["min"], dtype=torch.float32)
    maximum = torch.tensor(stats["action"]["max"], dtype=torch.float32)
    return (maximum - minimum) / 2


def _sample_manifest(samples: list[PushTConditioningSample]) -> list[dict]:
    return [
        {
            "sample_index": sample_index,
            "dataset_index": sample.dataset_index,
            "episode_index": sample.episode_index,
            "frame_index": sample.frame_index,
            "phase": sample.phase,
            "history_is_padded": sample.history_is_padded,
        }
        for sample_index, sample in enumerate(samples)
    ]


def _make_noise(
    *,
    count: int,
    seed: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    noise = []
    for sample_index in range(count):
        try:
            generator = torch.Generator(device=device).manual_seed(seed + sample_index)
        except RuntimeError:
            generator = torch.Generator().manual_seed(seed + sample_index)
        noise.append(
            torch.randn(
                (horizon, action_dim),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
    return torch.stack(noise)


def _run_schedule_batches(
    diffusion,
    conditioning: Tensor,
    noise: Tensor,
    schedule: tuple[int, ...],
    *,
    batch_size: int,
) -> tuple[Tensor, float]:
    actions = []
    wall_ms = 0.0
    for start in range(0, len(conditioning), batch_size):
        stop = min(len(conditioning), start + batch_size)
        run = run_explicit_schedule_sampler(
            diffusion,
            global_cond=conditioning[start:stop],
            noise=noise[start:stop],
            timesteps=schedule,
        )
        actions.append(run.actions.cpu())
        wall_ms += run.wall_ms
    return torch.cat(actions), wall_ms


def _score(
    reference: Tensor,
    candidate: Tensor,
    *,
    schedule: tuple[int, ...],
    pixel_scale: Tensor,
) -> ScheduleScore:
    errors = per_sample_action_errors(
        reference,
        candidate,
        pixel_scale=pixel_scale,
    )
    return ScheduleScore(
        schedule=schedule,
        mean_action_chunk_mse_normalized=float(errors.action_chunk_mse_normalized.mean()),
        mean_first_action_error_pixels=float(errors.first_action_max_pixels.mean()),
    )


def _heldout_rows(
    *,
    samples: list[PushTConditioningSample],
    sample_offset: int,
    budget: int,
    method: str,
    schedule: tuple[int, ...],
    reference: Tensor,
    candidate: Tensor,
    pixel_scale: Tensor,
) -> list[dict]:
    errors = per_sample_action_errors(
        reference,
        candidate,
        pixel_scale=pixel_scale,
    )
    rows = []
    for local_index, sample in enumerate(samples):
        rows.append(
            {
                "sample_index": sample_offset + local_index,
                "dataset_index": sample.dataset_index,
                "episode_index": sample.episode_index,
                "frame_index": sample.frame_index,
                "phase": sample.phase,
                "budget": budget,
                "method": method,
                "schedule": list(schedule),
                "actual_nfe": len(schedule),
                "first_action_max_error_normalized": float(
                    errors.first_action_max_normalized[local_index]
                ),
                "first_action_max_error_pixels": float(errors.first_action_max_pixels[local_index]),
                "action_chunk_mse_normalized": float(
                    errors.action_chunk_mse_normalized[local_index]
                ),
                "action_chunk_max_error_normalized": float(
                    errors.action_chunk_max_normalized[local_index]
                ),
            }
        )
    return rows


def _summarize(rows: list[dict], budgets: list[int], *, bootstrap_seed: int) -> list[dict]:
    summaries = []
    for budget in budgets:
        methods = {
            method: [row for row in rows if row["budget"] == budget and row["method"] == method]
            for method in ("standard", "optimized")
        }
        standard_by_sample = {row["sample_index"]: row for row in methods["standard"]}
        optimized_by_sample = {row["sample_index"]: row for row in methods["optimized"]}
        sample_indices = sorted(standard_by_sample)
        if sample_indices != sorted(optimized_by_sample):
            raise RuntimeError("standard and optimized rows are not paired")

        mse_differences = torch.tensor(
            [
                optimized_by_sample[index]["action_chunk_mse_normalized"]
                - standard_by_sample[index]["action_chunk_mse_normalized"]
                for index in sample_indices
            ]
        )
        first_differences = torch.tensor(
            [
                optimized_by_sample[index]["first_action_max_error_pixels"]
                - standard_by_sample[index]["first_action_max_error_pixels"]
                for index in sample_indices
            ]
        )
        mse_ci = paired_bootstrap_mean_ci(
            mse_differences,
            seed=bootstrap_seed + budget,
        )
        first_ci = paired_bootstrap_mean_ci(
            first_differences,
            seed=bootstrap_seed + 100 + budget,
        )
        record = {"budget": budget}
        for method, selected in methods.items():
            record[method] = {
                "schedule": selected[0]["schedule"],
                "samples": len(selected),
                "mean_first_action_error_pixels": statistics.fmean(
                    row["first_action_max_error_pixels"] for row in selected
                ),
                "max_first_action_error_pixels": max(
                    row["first_action_max_error_pixels"] for row in selected
                ),
                "mean_action_chunk_mse_normalized": statistics.fmean(
                    row["action_chunk_mse_normalized"] for row in selected
                ),
            }
        record["optimized_minus_standard"] = {
            "mean_first_action_error_pixels": float(first_differences.mean()),
            "mean_first_action_error_pixels_ci95": list(first_ci),
            "mean_action_chunk_mse_normalized": float(mse_differences.mean()),
            "mean_action_chunk_mse_normalized_ci95": list(mse_ci),
        }
        record["optimized_wins"] = (
            record["optimized"]["mean_action_chunk_mse_normalized"]
            < record["standard"]["mean_action_chunk_mse_normalized"]
            and record["optimized"]["mean_first_action_error_pixels"]
            <= record["standard"]["mean_first_action_error_pixels"]
        )
        summaries.append(record)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--calibration-samples", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", type=_int_list, default=_int_list("2,3,4,5"))
    parser.add_argument("--grid-step", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/pusht-keypoints"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface/hub"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment2/schedules"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.samples < 2:
        parser.error("--samples must be at least two")
    if not 1 <= args.calibration_samples < args.samples:
        parser.error("--calibration-samples must be positive and smaller than --samples")
    if args.grid_step < 1 or args.grid_step >= 100:
        parser.error("--grid-step must be between 1 and 99")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    device = _device(args.device)
    policy = load_policy(
        device=device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    policy.eval()
    parameter = next(policy.diffusion.parameters())
    num_train_timesteps = int(policy.diffusion.noise_scheduler.config.num_train_timesteps)
    grid = tuple(range(0, num_train_timesteps, args.grid_step))
    if any(budget > len(grid) for budget in args.budgets):
        parser.error("a requested budget exceeds the candidate-grid size")

    samples = load_real_conditioning_samples(
        count=args.samples,
        seed=args.seed,
        root=args.dataset_root,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    conditioning = torch.stack([sample.global_cond for sample in samples])
    noise = _make_noise(
        count=args.samples,
        seed=args.seed,
        horizon=policy.config.horizon,
        action_dim=policy.config.action_feature.shape[0],
        device=parameter.device,
        dtype=parameter.dtype,
    )
    pixel_scale = _action_pixel_scale(args.dataset_root)
    reference_schedule = standard_ddim_schedule(
        policy.diffusion.noise_scheduler,
        policy.diffusion.num_inference_steps,
    )
    reference, reference_wall_ms = _run_schedule_batches(
        policy.diffusion,
        conditioning,
        noise,
        reference_schedule,
        batch_size=args.batch_size,
    )
    print(f"reference complete schedule={reference_schedule}", flush=True)

    calibration_conditioning = conditioning[: args.calibration_samples]
    calibration_noise = noise[: args.calibration_samples]
    calibration_reference = reference[: args.calibration_samples]
    selected_schedules = {}
    calibration_scores = []
    search_started = time.perf_counter()
    for budget in args.budgets:
        standard_schedule = standard_ddim_schedule(policy.diffusion.noise_scheduler, budget)
        candidates = set(grid_schedules(budget, grid=grid))
        candidates.add(standard_schedule)
        ordered_candidates = sorted(candidates)
        scores = []
        for candidate_index, schedule in enumerate(ordered_candidates, 1):
            candidate, wall_ms = _run_schedule_batches(
                policy.diffusion,
                calibration_conditioning,
                calibration_noise,
                schedule,
                batch_size=args.batch_size,
            )
            score = _score(
                calibration_reference,
                candidate,
                schedule=schedule,
                pixel_scale=pixel_scale,
            )
            scores.append(score)
            calibration_scores.append(
                {
                    "budget": budget,
                    "schedule": list(schedule),
                    "is_standard": schedule == standard_schedule,
                    "mean_action_chunk_mse_normalized": score.mean_action_chunk_mse_normalized,
                    "mean_first_action_error_pixels": score.mean_first_action_error_pixels,
                    "wall_ms": wall_ms,
                }
            )
            if candidate_index % 25 == 0 or candidate_index == len(ordered_candidates):
                print(
                    f"search k={budget} {candidate_index}/{len(ordered_candidates)}",
                    flush=True,
                )
        selected = select_schedule(scores)
        selected_schedules[budget] = selected.schedule
        print(
            f"selected k={budget} schedule={selected.schedule} "
            f"calibration_mse={selected.mean_action_chunk_mse_normalized:.6g}",
            flush=True,
        )
    search_wall_ms = (time.perf_counter() - search_started) * 1000

    heldout_conditioning = conditioning[args.calibration_samples :]
    heldout_noise = noise[args.calibration_samples :]
    heldout_reference = reference[args.calibration_samples :]
    heldout_samples = samples[args.calibration_samples :]
    rows = []
    for budget in args.budgets:
        standard_schedule = standard_ddim_schedule(policy.diffusion.noise_scheduler, budget)
        optimized_schedule = selected_schedules[budget]
        standard_actions, _ = _run_schedule_batches(
            policy.diffusion,
            heldout_conditioning,
            heldout_noise,
            standard_schedule,
            batch_size=args.batch_size,
        )
        if optimized_schedule == standard_schedule:
            optimized_actions = standard_actions
        else:
            optimized_actions, _ = _run_schedule_batches(
                policy.diffusion,
                heldout_conditioning,
                heldout_noise,
                optimized_schedule,
                batch_size=args.batch_size,
            )
        rows.extend(
            _heldout_rows(
                samples=heldout_samples,
                sample_offset=args.calibration_samples,
                budget=budget,
                method="standard",
                schedule=standard_schedule,
                reference=heldout_reference,
                candidate=standard_actions,
                pixel_scale=pixel_scale,
            )
        )
        rows.extend(
            _heldout_rows(
                samples=heldout_samples,
                sample_offset=args.calibration_samples,
                budget=budget,
                method="optimized",
                schedule=optimized_schedule,
                reference=heldout_reference,
                candidate=optimized_actions,
                pixel_scale=pixel_scale,
            )
        )
        print(f"heldout k={budget} complete", flush=True)

    summary = _summarize(rows, args.budgets, bootstrap_seed=args.seed + 1000)
    wins = sum(record["optimized_wins"] for record in summary)
    full_protocol = (
        args.samples == 100
        and args.calibration_samples == 25
        and args.budgets == [2, 3, 4, 5]
        and args.grid_step == 10
        and args.seed == 0
    )
    if not full_protocol:
        decision = "smoke-only"
    else:
        decision = "proceed-to-closed-loop" if wins >= 3 else "stop-schedule-optimization"
    report = {
        "status": "experiment-2-schedule-optimization",
        "decision": decision,
        "full_protocol": full_protocol,
        "decision_rule": (
            "optimized schedule must lower held-out mean chunk MSE without increasing "
            "mean first-action pixel error at three of four budgets"
        ),
        "optimized_budget_wins": wins,
        "device": device,
        "seed": args.seed,
        "checkpoint": CHECKPOINT_REPO,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "dataset": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "samples": args.samples,
        "calibration_samples": args.calibration_samples,
        "evaluation_samples": args.samples - args.calibration_samples,
        "batch_size": args.batch_size,
        "budgets": args.budgets,
        "candidate_grid": list(grid),
        "reference_schedule": list(reference_schedule),
        "reference_wall_ms": reference_wall_ms,
        "search_wall_ms": search_wall_ms,
        "selected_schedules": {
            str(budget): list(schedule) for budget, schedule in selected_schedules.items()
        },
        "sample_manifest": _sample_manifest(samples),
        "calibration_scores": calibration_scores,
        "summary": summary,
        "rows": rows,
        "limitations": [
            "Schedule selection optimizes offline DDIM-10 action fidelity, not task success.",
            "Temporal thirds are coverage strata, not semantic contact labels.",
            "CPU timing is diagnostic and is not an acceleration claim.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for record in summary:
        print(
            f"k={record['budget']} standard={record['standard']['schedule']} "
            f"optimized={record['optimized']['schedule']} "
            f"standard_px={record['standard']['mean_first_action_error_pixels']:.3f} "
            f"optimized_px={record['optimized']['mean_first_action_error_pixels']:.3f} "
            f"win={record['optimized_wins']}",
        )
    print(f"decision={decision} wins={wins}/{len(args.budgets)}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
