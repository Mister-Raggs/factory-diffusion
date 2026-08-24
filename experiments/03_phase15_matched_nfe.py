"""Run the Phase 1.5 real-conditioning, matched-NFE comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from factory_diffusion.analysis import sweep_baseline_path
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
from factory_diffusion.cache import AdaptiveCacheConfig
from factory_diffusion.evaluation import (
    compare_action_tensors,
    run_cached_sampler,
    run_fixed_sampler,
    run_uncached_sampler,
)


def _float_list(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated non-negative numbers")
    return values


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


def _select_thresholds(
    calibration_traces,
    thresholds: list[float],
    budgets: list[int],
    *,
    warmup_steps: int,
    force_compute_last: int,
) -> tuple[dict[int, float], dict[str, dict[str, float]]]:
    recomputations: dict[float, list[int]] = defaultdict(list)
    for trace in calibration_traces:
        for row in sweep_baseline_path(
            trace,
            thresholds,
            warmup_steps=warmup_steps,
            force_compute_last=force_compute_last,
        ):
            recomputations[row.threshold].append(row.recomputed_steps)

    diagnostics = {
        str(threshold): {
            "mean_recomputations": statistics.fmean(values),
            "stdev_recomputations": statistics.pstdev(values),
        }
        for threshold, values in recomputations.items()
    }
    selected = {}
    for budget in budgets:
        selected[budget] = min(
            thresholds,
            key=lambda threshold: (
                abs(diagnostics[str(threshold)]["mean_recomputations"] - budget),
                diagnostics[str(threshold)]["stdev_recomputations"],
                threshold,
            ),
        )
    return selected, diagnostics


def _action_pixel_scale(dataset_root: Path) -> torch.Tensor:
    stats = json.loads((dataset_root / "meta" / "stats.json").read_text())
    minimum = torch.tensor(stats["action"]["min"], dtype=torch.float32)
    maximum = torch.tensor(stats["action"]["max"], dtype=torch.float32)
    return (maximum - minimum) / 2


def _error_record(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    pixel_scale: torch.Tensor,
) -> dict:
    normalized = compare_action_tensors(reference, candidate)
    pixel_reference = reference.float() * pixel_scale.to(reference.device)
    pixel_candidate = candidate.float() * pixel_scale.to(candidate.device)
    pixels = compare_action_tensors(pixel_reference, pixel_candidate)
    return {
        "first_action_max_error_normalized": normalized.first_action_max_error,
        "action_chunk_mse_normalized": normalized.action_chunk_mse,
        "action_chunk_max_error_normalized": normalized.action_chunk_max_error,
        "first_action_max_error_pixels": pixels.first_action_max_error,
        "action_chunk_mse_pixels": pixels.action_chunk_mse,
        "action_chunk_max_error_pixels": pixels.action_chunk_max_error,
    }


def _sample_manifest(samples: list[PushTConditioningSample]) -> list[dict]:
    return [
        {
            "dataset_index": sample.dataset_index,
            "episode_index": sample.episode_index,
            "frame_index": sample.frame_index,
            "phase": sample.phase,
            "history_is_padded": sample.history_is_padded,
        }
        for sample in samples
    ]


def _summarize(rows: list[dict], budgets: list[int]) -> list[dict]:
    summaries = []
    for budget in budgets:
        for method in ("ddim", "fixed", "adaptive"):
            selected = [row for row in rows if row["budget"] == budget and row["method"] == method]
            summaries.append(
                {
                    "budget": budget,
                    "method": method,
                    "samples": len(selected),
                    "mean_first_action_error_normalized": statistics.fmean(
                        row["first_action_max_error_normalized"] for row in selected
                    ),
                    "max_first_action_error_normalized": max(
                        row["first_action_max_error_normalized"] for row in selected
                    ),
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
            )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--calibration-samples", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", type=_int_list, default=_int_list("5,6,7,8"))
    parser.add_argument(
        "--thresholds",
        type=_float_list,
        default=_float_list("0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.4"),
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--force-compute-last", type=int, default=2)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/pusht-keypoints"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface/hub"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase15/matched-nfe"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.samples < 2:
        parser.error("--samples must be at least two")
    if not 1 <= args.calibration_samples < args.samples:
        parser.error("--calibration-samples must be positive and smaller than --samples")
    if any(budget >= 10 for budget in args.budgets):
        parser.error("matched budgets must be below the DDIM-10 reference")

    device = _device(args.device)
    policy = load_policy(
        device=device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    policy.eval()
    parameter = next(policy.diffusion.parameters())
    samples = load_real_conditioning_samples(
        count=args.samples,
        seed=args.seed,
        root=args.dataset_root,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    pixel_scale = _action_pixel_scale(args.dataset_root)

    references = []
    noises = []
    for sample_index, sample in enumerate(samples):
        generator = torch.Generator(device=parameter.device).manual_seed(args.seed + sample_index)
        noise = torch.randn(
            (1, policy.config.horizon, policy.config.action_feature.shape[0]),
            device=parameter.device,
            dtype=parameter.dtype,
            generator=generator,
        )
        reference = run_uncached_sampler(
            policy.diffusion,
            global_cond=sample.global_cond.unsqueeze(0),
            noise=noise,
            scheduler_seed=args.seed + sample_index,
        )
        references.append(reference)
        noises.append(noise)
        print(f"reference {sample_index + 1}/{len(samples)} phase={sample.phase}", flush=True)

    selected_thresholds, calibration = _select_thresholds(
        [run.trace for run in references[: args.calibration_samples]],
        args.thresholds,
        args.budgets,
        warmup_steps=args.warmup_steps,
        force_compute_last=args.force_compute_last,
    )

    rows = []
    for sample_index in range(args.calibration_samples, len(samples)):
        sample = samples[sample_index]
        reference = references[sample_index]
        noise = noises[sample_index]
        conditioning = sample.global_cond.unsqueeze(0)
        metadata = {
            "sample_index": sample_index,
            "dataset_index": sample.dataset_index,
            "episode_index": sample.episode_index,
            "frame_index": sample.frame_index,
            "phase": sample.phase,
            "history_is_padded": sample.history_is_padded,
        }
        for budget in args.budgets:
            ddim = run_uncached_sampler(
                policy.diffusion,
                global_cond=conditioning,
                noise=noise,
                scheduler_seed=args.seed + sample_index,
                num_inference_steps=budget,
            )
            fixed = run_fixed_sampler(
                policy.diffusion,
                global_cond=conditioning,
                noise=noise,
                recomputations=budget,
                scheduler_seed=args.seed + sample_index,
                warmup_steps=args.warmup_steps,
                force_compute_last=args.force_compute_last,
            )
            adaptive = run_cached_sampler(
                policy.diffusion,
                global_cond=conditioning,
                noise=noise,
                cache_config=AdaptiveCacheConfig(
                    threshold=selected_thresholds[budget],
                    warmup_steps=args.warmup_steps,
                    force_compute_last=args.force_compute_last,
                    target_recomputations=budget,
                ),
                scheduler_seed=args.seed + sample_index,
            )

            for method, candidate, nfe in (
                ("ddim", ddim.actions, len(ddim.trace.steps)),
                ("fixed", fixed.actions, sum(step.recomputed for step in fixed.steps)),
                ("adaptive", adaptive.actions, sum(step.recomputed for step in adaptive.steps)),
            ):
                rows.append(
                    {
                        **metadata,
                        "budget": budget,
                        "method": method,
                        "actual_nfe": nfe,
                        "threshold": selected_thresholds[budget] if method == "adaptive" else None,
                        **_error_record(reference.actions, candidate, pixel_scale),
                    }
                )
        print(
            f"evaluation {sample_index - args.calibration_samples + 1}/"
            f"{len(samples) - args.calibration_samples} phase={sample.phase}",
            flush=True,
        )

    report = {
        "status": "phase-1.5-matched-nfe",
        "device": device,
        "seed": args.seed,
        "checkpoint": CHECKPOINT_REPO,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "dataset": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "conditioning": "real normalized two-frame PushT keypoint observations",
        "samples": args.samples,
        "calibration_samples": args.calibration_samples,
        "evaluation_samples": args.samples - args.calibration_samples,
        "budgets": args.budgets,
        "threshold_candidates": args.thresholds,
        "selected_thresholds": {str(key): value for key, value in selected_thresholds.items()},
        "calibration_threshold_nfe": calibration,
        "sample_manifest": _sample_manifest(samples),
        "summary": _summarize(rows, args.budgets),
        "rows": rows,
        "limitations": [
            "This report measures action fidelity, not closed-loop task success.",
            "CPU/MPS timings are diagnostic only and are intentionally omitted from claims.",
            "Thresholds are selected only by calibration NFE, never by evaluation action error.",
            "Progress thirds are temporal strata, not verified contact-state labels.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["summary"]:
        print(
            f"k={row['budget']} method={row['method']} "
            f"mean_first_px={row['mean_first_action_error_pixels']:.3f} "
            f"mean_mse={row['mean_action_chunk_mse_normalized']:.6g}"
        )
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
