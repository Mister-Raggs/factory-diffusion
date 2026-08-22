"""Run the complete Phase 1 probe on a real pretrained action U-Net."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from factory_diffusion.analysis import sweep_baseline_path
from factory_diffusion.baselines.pusht_keypoints import (
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
    load_policy,
)
from factory_diffusion.cache import AdaptiveCacheConfig
from factory_diffusion.evaluation import compare_runs, run_cached_sampler, run_uncached_sampler


def _parse_thresholds(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("thresholds must be comma-separated non-negative numbers")
    return values


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=_parse_thresholds("0,0.03,0.05,0.075,0.1,0.15,0.25"),
    )
    parser.add_argument(
        "--online-thresholds",
        type=_parse_thresholds,
        default=_parse_thresholds("0,0.075,0.1,0.15,0.2,0.25"),
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--force-compute-last", type=int, default=2)
    parser.add_argument("--max-consecutive-skips", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface/hub"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1/pusht-keypoints"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--print-report", action="store_true")
    args = parser.parse_args()

    device = _device(args.device)
    torch.manual_seed(args.seed)
    policy = load_policy(
        device=device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    policy.eval()
    diffusion = policy.diffusion

    parameter = next(diffusion.parameters())
    generator = torch.Generator(device=parameter.device).manual_seed(args.seed)
    global_cond = torch.empty((1, 36), device=parameter.device, dtype=parameter.dtype).uniform_(
        -1, 1, generator=generator
    )
    noise = torch.randn(
        (1, policy.config.horizon, policy.config.action_feature.shape[0]),
        device=parameter.device,
        dtype=parameter.dtype,
        generator=generator,
    )

    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be non-negative")
    for warmup_index in range(args.warmup_runs):
        with torch.inference_mode():
            diffusion.conditional_sample(
                noise.shape[0],
                global_cond=global_cond,
                noise=noise.detach().clone(),
                generator=torch.Generator(device=parameter.device).manual_seed(
                    args.seed + warmup_index
                ),
            )

    baseline = run_uncached_sampler(
        diffusion,
        global_cond=global_cond,
        noise=noise,
        scheduler_seed=args.seed,
    )
    offline = sweep_baseline_path(
        baseline.trace,
        args.thresholds,
        warmup_steps=args.warmup_steps,
        force_compute_last=args.force_compute_last,
        max_consecutive_skips=args.max_consecutive_skips,
    )

    online = []
    for threshold in args.online_thresholds:
        cached = run_cached_sampler(
            diffusion,
            global_cond=global_cond,
            noise=noise,
            cache_config=AdaptiveCacheConfig(
                threshold=threshold,
                warmup_steps=args.warmup_steps,
                force_compute_last=args.force_compute_last,
                max_consecutive_skips=args.max_consecutive_skips,
            ),
            scheduler_seed=args.seed,
        )
        paired = compare_runs(
            baseline,
            cached,
            n_obs_steps=policy.config.n_obs_steps,
            n_action_steps=policy.config.n_action_steps,
        )
        online.append(
            {
                "threshold": threshold,
                "recomputed_steps": len(cached.steps) - cached.skipped_steps,
                "skipped_steps": cached.skipped_steps,
                "skip_fraction": cached.skip_fraction,
                "cached_wall_ms": cached.wall_ms,
                "observed_speedup": paired.observed_speedup,
                "first_action_max_error": paired.first_action_max_error,
                "action_chunk_mse": paired.action_chunk_mse,
                "action_chunk_max_error": paired.action_chunk_max_error,
                "step_reasons": [step.reason for step in cached.steps],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline.trace.save(args.output_dir / f"trace_seed_{args.seed}")
    report = {
        "status": "phase-1-probe",
        "checkpoint": CHECKPOINT_REPO,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "lerobot_version": "0.4.4",
        "device": device,
        "seed": args.seed,
        "warmup_runs": args.warmup_runs,
        "conditioning": "fixed synthetic normalized keypoint conditioning",
        "cache_policy": {
            "warmup_steps": args.warmup_steps,
            "force_compute_last": args.force_compute_last,
            "max_consecutive_skips": args.max_consecutive_skips,
        },
        "baseline": {
            "wall_ms": baseline.wall_ms,
            "denoiser_ms": baseline.trace.total_model_ms,
            "denoiser_fraction": baseline.trace.total_model_ms / baseline.wall_ms,
            "denoising_steps": len(baseline.trace.steps),
        },
        "offline_baseline_path_sweep": [asdict(row) for row in offline],
        "online_exact_scheduler_runs": online,
        "limitations": [
            "CPU/MPS timings are functional measurements, not publication CUDA results.",
            (
                "Offline replay follows baseline inputs and does not include "
                "scheduler-path divergence."
            ),
            (
                "Synthetic normalized conditioning tests the real pretrained U-Net, "
                "not PushT task success."
            ),
            (
                "The legacy checkpoint normalization buffers are not used by this "
                "direct denoiser probe."
            ),
        ],
    }
    report_path = args.output_dir / f"report_seed_{args.seed}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.print_report:
        print(json.dumps(report, indent=2))
    else:
        for row in online:
            print(
                f"threshold={row['threshold']:.4f} skip={row['skip_fraction']:.1%} "
                f"first_error={row['first_action_max_error']:.6f} "
                f"chunk_mse={row['action_chunk_mse']:.6g}"
            )
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
