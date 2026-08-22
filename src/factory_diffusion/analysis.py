"""Offline analysis for uncached denoising traces."""

from __future__ import annotations

from dataclasses import dataclass

from factory_diffusion.cache import AdaptiveCacheConfig, AdaptiveResidualCache
from factory_diffusion.trace import DenoisingTrace


@dataclass(frozen=True)
class OfflineReplaySummary:
    """Approximate cache behavior along the original uncached sample path."""

    threshold: float
    total_steps: int
    recomputed_steps: int
    skipped_steps: int
    skip_fraction: float
    model_output_mse: float
    model_output_max_error: float
    recomputed_indices: tuple[int, ...]
    skipped_indices: tuple[int, ...]


def replay_baseline_path(
    trace: DenoisingTrace,
    config: AdaptiveCacheConfig,
) -> OfflineReplaySummary:
    """Replay cache decisions on recorded baseline inputs.

    This is an inexpensive feasibility screen. It does not model scheduler-path
    divergence after a cached output; selected thresholds must subsequently be
    run online against the real scheduler.
    """

    if not trace.steps:
        raise ValueError("cannot replay an empty trace")
    cache = AdaptiveResidualCache(config)
    cache.reset(total_steps=len(trace.steps))
    squared_error_sum = 0.0
    element_count = 0
    max_error = 0.0
    recomputed: list[int] = []
    skipped: list[int] = []

    for record in trace.steps:
        result = cache.run(
            record.index,
            record.model_input,
            lambda output=record.model_output: output,
        )
        error = (result.output.float() - record.model_output.float()).abs()
        squared_error_sum += float(error.square().sum())
        element_count += error.numel()
        max_error = max(max_error, float(error.max()))
        (recomputed if result.recomputed else skipped).append(record.index)

    return OfflineReplaySummary(
        threshold=config.threshold,
        total_steps=len(trace.steps),
        recomputed_steps=len(recomputed),
        skipped_steps=len(skipped),
        skip_fraction=len(skipped) / len(trace.steps),
        model_output_mse=squared_error_sum / element_count,
        model_output_max_error=max_error,
        recomputed_indices=tuple(recomputed),
        skipped_indices=tuple(skipped),
    )


def sweep_baseline_path(
    trace: DenoisingTrace,
    thresholds: list[float],
    *,
    warmup_steps: int = 2,
    force_compute_last: int = 2,
    max_consecutive_skips: int | None = None,
) -> list[OfflineReplaySummary]:
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return [
        replay_baseline_path(
            trace,
            AdaptiveCacheConfig(
                threshold=threshold,
                warmup_steps=warmup_steps,
                force_compute_last=force_compute_last,
                max_consecutive_skips=max_consecutive_skips,
            ),
        )
        for threshold in thresholds
    ]
