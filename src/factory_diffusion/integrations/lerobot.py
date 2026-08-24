"""A minimal wrapper around LeRobot's temporal diffusion U-Net."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from torch import Tensor, nn

from factory_diffusion.cache import AdaptiveCacheConfig, AdaptiveResidualCache, CacheStep


class ResidualCache(Protocol):
    total_steps: int | None
    next_step: int

    def reset(self, total_steps: int | None) -> None: ...

    def run(
        self,
        step_index: int,
        model_input: Tensor,
        compute: Callable[[], Tensor],
    ) -> CacheStep: ...


class ResidualReuseDenoiser(nn.Module):
    """Wrap a same-shape PyTorch denoiser with a per-trajectory cache.

    Call ``begin_trajectory`` immediately before LeRobot enters its scheduler
    loop. Each subsequent ``forward`` consumes exactly one denoising step.
    Telemetry is retained in ``steps`` for offline analysis.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        cache: ResidualCache,
        *,
        auto_total_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.cache = cache
        self.steps: list[CacheStep] = []
        self._active = False
        self.auto_total_steps = auto_total_steps

    def begin_trajectory(self, total_steps: int) -> None:
        self.cache.reset(total_steps)
        self.steps.clear()
        self._active = True

    def end_trajectory(self) -> None:
        self._active = False

    def forward(
        self,
        sample: Tensor,
        timestep: Tensor | int,
        global_cond: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        if self.training:
            return self.denoiser(sample, timestep, global_cond=global_cond, **kwargs)
        if not self._active:
            if self.auto_total_steps is None:
                raise RuntimeError("begin_trajectory(total_steps) must be called before denoising")
            self.begin_trajectory(self.auto_total_steps)

        step_index = self.cache.next_step

        def compute() -> Tensor:
            return self.denoiser(sample, timestep, global_cond=global_cond, **kwargs)

        result = self.cache.run(step_index, sample, compute)
        self.steps.append(result)
        if self.cache.total_steps == self.cache.next_step:
            self.end_trajectory()
        return result.output


class CachedDenoiser(ResidualReuseDenoiser):
    """Adaptive-residual specialization retained as the public LeRobot adapter."""

    def __init__(
        self,
        denoiser: nn.Module,
        config: AdaptiveCacheConfig | None = None,
        *,
        auto_total_steps: int | None = None,
    ) -> None:
        super().__init__(
            denoiser,
            AdaptiveResidualCache(config),
            auto_total_steps=auto_total_steps,
        )


def install_on_lerobot_policy(
    policy: nn.Module,
    config: AdaptiveCacheConfig | None = None,
) -> CachedDenoiser:
    """Install caching on a LeRobot 0.4.4 ``DiffusionPolicy`` instance.

    The function relies only on LeRobot's small public object shape instead of
    importing or copying its modeling module. It should be called after loading
    the checkpoint and before evaluation. Training forwards bypass the cache.
    """

    diffusion = getattr(policy, "diffusion", None)
    if diffusion is None or not hasattr(diffusion, "unet"):
        raise TypeError("expected a LeRobot DiffusionPolicy with diffusion.unet")
    if isinstance(diffusion.unet, CachedDenoiser):
        raise ValueError("the policy denoiser is already cache-wrapped")

    total_steps = int(diffusion.num_inference_steps)
    wrapped = CachedDenoiser(diffusion.unet, config, auto_total_steps=total_steps)
    diffusion.unet = wrapped
    return wrapped
