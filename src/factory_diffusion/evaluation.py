"""Uncached and cached evaluation on an actual diffusion sampler."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from factory_diffusion.cache import (
    AdaptiveCacheConfig,
    AdaptiveResidualCache,
    CacheStep,
    FixedResidualCache,
    guarded_uniform_schedule,
)
from factory_diffusion.integrations.lerobot import ResidualCache, ResidualReuseDenoiser
from factory_diffusion.trace import DenoisingTrace, TraceDenoiser


def _synchronize(tensor: Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)
    elif tensor.device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except RuntimeError:
        return torch.Generator().manual_seed(seed)


@dataclass(frozen=True)
class UncachedRun:
    actions: Tensor
    trace: DenoisingTrace
    wall_ms: float


@dataclass(frozen=True)
class CachedRun:
    actions: Tensor
    steps: tuple[CacheStep, ...]
    wall_ms: float

    @property
    def skipped_steps(self) -> int:
        return sum(not step.recomputed for step in self.steps)

    @property
    def skip_fraction(self) -> float:
        return self.skipped_steps / len(self.steps)


@dataclass(frozen=True)
class PairedRun:
    baseline: UncachedRun
    cached: CachedRun
    first_action_max_error: float
    action_chunk_mse: float
    action_chunk_max_error: float

    @property
    def observed_speedup(self) -> float:
        return self.baseline.wall_ms / self.cached.wall_ms


@dataclass(frozen=True)
class ActionError:
    first_action_max_error: float
    action_chunk_mse: float
    action_chunk_max_error: float


def compare_action_tensors(
    reference: Tensor,
    candidate: Tensor,
    *,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
) -> ActionError:
    """Compare the action chunk that LeRobot would actually execute."""

    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate action tensors must have equal shapes")
    action_start = n_obs_steps - 1
    action_stop = action_start + n_action_steps
    reference_chunk = reference[:, action_start:action_stop]
    candidate_chunk = candidate[:, action_start:action_stop]
    chunk_error = (candidate_chunk.float() - reference_chunk.float()).abs()
    return ActionError(
        first_action_max_error=float(chunk_error[:, 0].max()),
        action_chunk_mse=float(chunk_error.square().mean()),
        action_chunk_max_error=float(chunk_error.max()),
    )


def _validate_diffusion(diffusion: nn.Module) -> tuple[nn.Module, int]:
    unet = getattr(diffusion, "unet", None)
    if not isinstance(unet, nn.Module):
        raise TypeError("diffusion must expose a torch module at .unet")
    if not hasattr(diffusion, "num_inference_steps"):
        raise TypeError("diffusion must expose .num_inference_steps")
    total_steps = int(diffusion.num_inference_steps)
    if total_steps < 1:
        raise ValueError("diffusion.num_inference_steps must be positive")
    return unet, total_steps


@torch.inference_mode()
def run_uncached_sampler(
    diffusion: nn.Module,
    *,
    global_cond: Tensor,
    noise: Tensor,
    scheduler_seed: int = 0,
    num_inference_steps: int | None = None,
) -> UncachedRun:
    """Capture an exact uncached trajectory and its denoiser timing."""

    original_unet, configured_steps = _validate_diffusion(diffusion)
    total_steps = configured_steps if num_inference_steps is None else int(num_inference_steps)
    if total_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    tracer = TraceDenoiser(original_unet, auto_total_steps=total_steps)
    tracer.eval()
    try:
        diffusion.unet = tracer
        diffusion.num_inference_steps = total_steps
        _synchronize(noise)
        started = time.perf_counter()
        actions = diffusion.conditional_sample(
            noise.shape[0],
            global_cond=global_cond,
            noise=noise.detach().clone(),
            generator=_generator(noise.device, scheduler_seed),
        )
        _synchronize(actions)
        wall_ms = (time.perf_counter() - started) * 1000
        trace = tracer.trace()
    finally:
        diffusion.unet = original_unet
        diffusion.num_inference_steps = configured_steps
    return UncachedRun(actions=actions.detach(), trace=trace, wall_ms=wall_ms)


@torch.inference_mode()
def run_residual_reuse_sampler(
    diffusion: nn.Module,
    *,
    global_cond: Tensor,
    noise: Tensor,
    cache: ResidualCache,
    scheduler_seed: int = 0,
) -> CachedRun:
    """Run an exact reused trajectory, including scheduler-path divergence."""

    original_unet, total_steps = _validate_diffusion(diffusion)
    cached = ResidualReuseDenoiser(original_unet, cache, auto_total_steps=total_steps)
    cached.eval()
    try:
        diffusion.unet = cached
        _synchronize(noise)
        started = time.perf_counter()
        actions = diffusion.conditional_sample(
            noise.shape[0],
            global_cond=global_cond,
            noise=noise.detach().clone(),
            generator=_generator(noise.device, scheduler_seed),
        )
        _synchronize(actions)
        wall_ms = (time.perf_counter() - started) * 1000
    finally:
        diffusion.unet = original_unet
    return CachedRun(actions=actions.detach(), steps=tuple(cached.steps), wall_ms=wall_ms)


def run_cached_sampler(
    diffusion: nn.Module,
    *,
    global_cond: Tensor,
    noise: Tensor,
    cache_config: AdaptiveCacheConfig,
    scheduler_seed: int = 0,
) -> CachedRun:
    return run_residual_reuse_sampler(
        diffusion,
        global_cond=global_cond,
        noise=noise,
        cache=AdaptiveResidualCache(cache_config),
        scheduler_seed=scheduler_seed,
    )


def run_fixed_sampler(
    diffusion: nn.Module,
    *,
    global_cond: Tensor,
    noise: Tensor,
    recomputations: int,
    scheduler_seed: int = 0,
    warmup_steps: int = 2,
    force_compute_last: int = 2,
) -> CachedRun:
    """Run fixed transformation reuse with an exact denoiser-call budget."""

    _, total_steps = _validate_diffusion(diffusion)
    schedule = guarded_uniform_schedule(
        total_steps,
        recomputations,
        warmup_steps=warmup_steps,
        force_compute_last=force_compute_last,
    )
    return run_residual_reuse_sampler(
        diffusion,
        global_cond=global_cond,
        noise=noise,
        cache=FixedResidualCache(schedule),
        scheduler_seed=scheduler_seed,
    )


def compare_runs(
    baseline: UncachedRun,
    cached: CachedRun,
    *,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
) -> PairedRun:
    """Compare the action chunk that LeRobot would actually execute."""

    error = compare_action_tensors(
        baseline.actions,
        cached.actions,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )
    return PairedRun(
        baseline=baseline,
        cached=cached,
        first_action_max_error=error.first_action_max_error,
        action_chunk_mse=error.action_chunk_mse,
        action_chunk_max_error=error.action_chunk_max_error,
    )


def run_paired_sampler(
    diffusion: nn.Module,
    *,
    global_cond: Tensor,
    noise: Tensor,
    cache_config: AdaptiveCacheConfig,
    scheduler_seed: int = 0,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
) -> PairedRun:
    baseline = run_uncached_sampler(
        diffusion,
        global_cond=global_cond,
        noise=noise,
        scheduler_seed=scheduler_seed,
    )
    cached = run_cached_sampler(
        diffusion,
        global_cond=global_cond,
        noise=noise,
        cache_config=cache_config,
        scheduler_seed=scheduler_seed,
    )
    return compare_runs(
        baseline,
        cached,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )
