"""Calibration and paired statistics for DDIM schedule selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PerSampleActionError:
    first_action_max_normalized: Tensor
    action_chunk_mse_normalized: Tensor
    action_chunk_max_normalized: Tensor
    first_action_max_pixels: Tensor


@dataclass(frozen=True)
class ScheduleScore:
    schedule: tuple[int, ...]
    mean_action_chunk_mse_normalized: float
    mean_first_action_error_pixels: float


def per_sample_action_errors(
    reference: Tensor,
    candidate: Tensor,
    *,
    pixel_scale: Tensor,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
) -> PerSampleActionError:
    """Return paired action errors without reducing the batch dimension."""

    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate action tensors must have equal shapes")
    if reference.ndim != 3:
        raise ValueError("action tensors must have shape (batch, horizon, action_dim)")
    if tuple(pixel_scale.shape) != (reference.shape[-1],):
        raise ValueError("pixel_scale must contain one value per action dimension")
    action_start = n_obs_steps - 1
    action_stop = action_start + n_action_steps
    if action_start < 0 or action_stop > reference.shape[1]:
        raise ValueError("requested action chunk is outside the policy horizon")

    difference = (
        candidate[:, action_start:action_stop].float()
        - reference[:, action_start:action_stop].float()
    ).abs()
    pixel_difference = difference * pixel_scale.to(difference.device, difference.dtype)
    return PerSampleActionError(
        first_action_max_normalized=difference[:, 0].amax(dim=-1),
        action_chunk_mse_normalized=difference.square().mean(dim=(1, 2)),
        action_chunk_max_normalized=difference.amax(dim=(1, 2)),
        first_action_max_pixels=pixel_difference[:, 0].amax(dim=-1),
    )


def select_schedule(scores: list[ScheduleScore]) -> ScheduleScore:
    """Select using calibration chunk MSE, then first action error and schedule."""

    if not scores:
        raise ValueError("at least one schedule score is required")
    return min(
        scores,
        key=lambda score: (
            score.mean_action_chunk_mse_normalized,
            score.mean_first_action_error_pixels,
            score.schedule,
        ),
    )


def paired_bootstrap_mean_ci(
    differences: Tensor,
    *,
    seed: int,
    resamples: int = 5000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap a confidence interval for a paired mean difference."""

    values = differences.detach().float().cpu().flatten()
    if values.numel() < 2:
        raise ValueError("paired bootstrap requires at least two differences")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        values.numel(),
        (resamples, values.numel()),
        generator=generator,
    )
    means = values[indices].mean(dim=1)
    tail = (1 - confidence) / 2
    quantiles = torch.quantile(means, torch.tensor([tail, 1 - tail]))
    return float(quantiles[0]), float(quantiles[1])
