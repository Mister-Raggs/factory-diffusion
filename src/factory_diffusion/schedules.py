"""Explicit DDIM schedules for few-step robot-policy sampling."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

from torch import Tensor


def validate_ddim_schedule(
    timesteps: Iterable[int],
    *,
    num_train_timesteps: int,
    require_terminal_zero: bool = True,
) -> tuple[int, ...]:
    """Return a validated, strictly descending sequence of training timesteps."""

    schedule = tuple(int(timestep) for timestep in timesteps)
    if not schedule:
        raise ValueError("a DDIM schedule must contain at least one timestep")
    if num_train_timesteps < 1:
        raise ValueError("num_train_timesteps must be positive")
    if any(timestep < 0 or timestep >= num_train_timesteps for timestep in schedule):
        raise ValueError("schedule timesteps must be within the training range")
    if any(
        current <= following for current, following in zip(schedule, schedule[1:], strict=False)
    ):
        raise ValueError("schedule timesteps must be strictly descending")
    if require_terminal_zero and schedule[-1] != 0:
        raise ValueError("a complete DDIM schedule must terminate at timestep zero")
    return schedule


def grid_schedules(
    budget: int,
    *,
    grid: Iterable[int] = range(0, 100, 10),
) -> tuple[tuple[int, ...], ...]:
    """Enumerate descending schedules of an exact budget that terminate at zero."""

    candidates = tuple(sorted({int(timestep) for timestep in grid}))
    if not candidates or candidates[0] != 0:
        raise ValueError("the candidate grid must contain timestep zero")
    if budget < 1 or budget > len(candidates):
        raise ValueError("budget must be between one and the candidate-grid size")

    schedules = [
        tuple(sorted((*interior, 0), reverse=True))
        for interior in itertools.combinations(candidates[1:], budget - 1)
    ]
    return tuple(schedules)


def standard_ddim_schedule(scheduler: Any, budget: int) -> tuple[int, ...]:
    """Ask Diffusers for its configured default schedule without retaining mutation."""

    previous_steps = getattr(scheduler, "num_inference_steps", None)
    previous_timesteps = getattr(scheduler, "timesteps", None)
    try:
        scheduler.set_timesteps(budget)
        schedule = tuple(int(timestep) for timestep in scheduler.timesteps.tolist())
    finally:
        scheduler.num_inference_steps = previous_steps
        if previous_timesteps is not None:
            scheduler.timesteps = previous_timesteps
    return validate_ddim_schedule(
        schedule,
        num_train_timesteps=int(scheduler.config.num_train_timesteps),
    )


def ddim_step_to(
    scheduler: Any,
    model_output: Tensor,
    timestep: int,
    previous_timestep: int,
    sample: Tensor,
) -> Tensor:
    """Apply a deterministic DDIM transition to an explicit previous timestep."""

    if previous_timestep >= timestep:
        raise ValueError("previous_timestep must be smaller than timestep")
    if model_output.shape != sample.shape:
        raise ValueError("model_output and sample must have equal shapes")

    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(
        device=sample.device,
        dtype=sample.dtype,
    )
    if previous_timestep >= 0:
        alpha_prod_previous = scheduler.alphas_cumprod[previous_timestep]
    else:
        alpha_prod_previous = scheduler.final_alpha_cumprod
    alpha_prod_previous = alpha_prod_previous.to(device=sample.device, dtype=sample.dtype)
    beta_prod_t = 1 - alpha_prod_t

    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        predicted_original = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        predicted_epsilon = model_output
    elif prediction_type == "sample":
        predicted_original = model_output
        predicted_epsilon = (sample - alpha_prod_t.sqrt() * predicted_original) / beta_prod_t.sqrt()
    elif prediction_type == "v_prediction":
        predicted_original = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        predicted_epsilon = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
    else:
        raise ValueError(f"unsupported DDIM prediction type: {prediction_type}")

    if scheduler.config.thresholding:
        predicted_original = scheduler._threshold_sample(predicted_original)
    elif scheduler.config.clip_sample:
        predicted_original = predicted_original.clamp(
            -scheduler.config.clip_sample_range,
            scheduler.config.clip_sample_range,
        )

    direction = (1 - alpha_prod_previous).sqrt() * predicted_epsilon
    return alpha_prod_previous.sqrt() * predicted_original + direction
