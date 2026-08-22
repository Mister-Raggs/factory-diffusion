"""Uncached and cached evaluation on an actual diffusion sampler."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from factory_diffusion.cache import AdaptiveCacheConfig, CacheStep
from factory_diffusion.integrations.lerobot import CachedDenoiser
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
) -> UncachedRun:
    """Capture an exact uncached trajectory and its denoiser timing."""

    original_unet, total_steps = _validate_diffusion(diffusion)
    tracer = TraceDenoiser(original_unet, auto_total_steps=total_steps)
    tracer.eval()
    try:
        diffusion.unet = tracer
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
    return UncachedRun(actions=actions.detach(), trace=trace, wall_ms=wall_ms)


@torch.inference_mode()
def run_cached_sampler(
    diffusion: nn.Module,
    *,
    global_cond: Tensor,
    noise: Tensor,
    cache_config: AdaptiveCacheConfig,
    scheduler_seed: int = 0,
) -> CachedRun:
    """Run an exact cached trajectory, including scheduler-path divergence."""

    original_unet, total_steps = _validate_diffusion(diffusion)
    cached = CachedDenoiser(original_unet, cache_config, auto_total_steps=total_steps)
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


def compare_runs(
    baseline: UncachedRun,
    cached: CachedRun,
    *,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
) -> PairedRun:
    """Compare the action chunk that LeRobot would actually execute."""

    if baseline.actions.shape != cached.actions.shape:
        raise ValueError("baseline and cached action tensors must have equal shapes")
    action_start = n_obs_steps - 1
    action_stop = action_start + n_action_steps
    baseline_chunk = baseline.actions[:, action_start:action_stop]
    cached_chunk = cached.actions[:, action_start:action_stop]
    chunk_error = (cached_chunk.float() - baseline_chunk.float()).abs()
    first_error = chunk_error[:, 0]
    return PairedRun(
        baseline=baseline,
        cached=cached,
        first_action_max_error=float(first_error.max()),
        action_chunk_mse=float(chunk_error.square().mean()),
        action_chunk_max_error=float(chunk_error.max()),
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
