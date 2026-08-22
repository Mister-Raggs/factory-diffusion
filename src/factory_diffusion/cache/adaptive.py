"""Training-free adaptive transformation caching for tensor denoisers.

The cache is intentionally independent of LeRobot. A caller supplies the
current denoiser input and a zero-argument function that performs the expensive
model evaluation. The returned tensor always has the same shape as the input.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AdaptiveCacheConfig:
    """Controls when a denoiser evaluation may be replaced by cached reuse."""

    threshold: float = 0.0
    warmup_steps: int = 2
    force_compute_last: int = 2
    max_consecutive_skips: int | None = None
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")
        if self.warmup_steps < 1:
            raise ValueError("warmup_steps must be at least one")
        if self.force_compute_last < 0:
            raise ValueError("force_compute_last must be non-negative")
        if self.max_consecutive_skips is not None and self.max_consecutive_skips < 1:
            raise ValueError("max_consecutive_skips must be positive when provided")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class CacheStep:
    """Result and telemetry for one denoising step."""

    output: Tensor
    recomputed: bool
    reason: str
    predicted_error: float
    accumulated_error: float
    sensitivity: float | None
    compute_ms: float


class AdaptiveResidualCache:
    """Reuse a cached ``model_output - model_input`` transformation.

    ``reset`` must be called before each independent denoising trajectory. The
    implementation clones detached tensors so cache state cannot retain an
    autograd graph or be mutated by the scheduler.
    """

    def __init__(self, config: AdaptiveCacheConfig | None = None) -> None:
        self.config = config or AdaptiveCacheConfig()
        self.reset(total_steps=None)

    def reset(self, total_steps: int | None) -> None:
        if total_steps is not None and total_steps < 1:
            raise ValueError("total_steps must be positive when provided")
        self.total_steps = total_steps
        self.next_step = 0
        self.accumulated_error = 0.0
        self.sensitivity: float | None = None
        self.consecutive_skips = 0
        self._previous_input: Tensor | None = None
        self._last_computed_input: Tensor | None = None
        self._last_computed_output: Tensor | None = None
        self._cached_transformation: Tensor | None = None

    @staticmethod
    def _mean_absolute(tensor: Tensor) -> Tensor:
        return tensor.detach().float().abs().mean()

    def _must_compute(self, step_index: int) -> str | None:
        if step_index < self.config.warmup_steps:
            return "warmup"
        if self.total_steps is not None:
            final_region = max(0, self.total_steps - self.config.force_compute_last)
            if step_index >= final_region:
                return "final-step"
        if self._cached_transformation is None or self.sensitivity is None:
            return "uninitialized"
        if (
            self.config.max_consecutive_skips is not None
            and self.consecutive_skips >= self.config.max_consecutive_skips
        ):
            return "skip-limit"
        return None

    def _record_computation(self, model_input: Tensor, output: Tensor) -> None:
        if output.shape != model_input.shape:
            raise ValueError(
                "adaptive transformation reuse requires model input and output to have equal shapes"
            )

        if self._last_computed_input is not None and self._last_computed_output is not None:
            input_change = self._mean_absolute(model_input - self._last_computed_input)
            output_change = self._mean_absolute(output - self._last_computed_output)
            if float(input_change) > self.config.epsilon:
                self.sensitivity = float(output_change / input_change)

        clean_input = model_input.detach().clone()
        clean_output = output.detach().clone()
        self._last_computed_input = clean_input
        self._last_computed_output = clean_output
        self._cached_transformation = clean_output - clean_input
        self.accumulated_error = 0.0
        self.consecutive_skips = 0

    def run(
        self,
        step_index: int,
        model_input: Tensor,
        compute: Callable[[], Tensor],
    ) -> CacheStep:
        """Return a computed or cached denoiser output for one ordered step."""

        if step_index != self.next_step:
            raise ValueError(f"expected denoising step {self.next_step}, received {step_index}")
        if self.total_steps is not None and step_index >= self.total_steps:
            raise ValueError("step_index exceeds the trajectory length supplied to reset")

        forced_reason = self._must_compute(step_index)
        predicted_error = 0.0

        if forced_reason is None:
            assert self._previous_input is not None
            assert self._last_computed_output is not None
            input_change = self._mean_absolute(model_input - self._previous_input)
            output_norm = self._mean_absolute(self._last_computed_output)
            predicted_error = self.sensitivity * float(
                input_change / output_norm.clamp_min(self.config.epsilon)
            )
            candidate_error = self.accumulated_error + predicted_error
            if candidate_error < self.config.threshold:
                assert self._cached_transformation is not None
                output = model_input + self._cached_transformation.to(
                    device=model_input.device, dtype=model_input.dtype
                )
                self.accumulated_error = candidate_error
                self.consecutive_skips += 1
                reason = "cached"
                recomputed = False
            else:
                forced_reason = "threshold"

        if forced_reason is not None:
            if model_input.device.type == "cuda":
                torch.cuda.synchronize(model_input.device)
            elif model_input.device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
            started = time.perf_counter()
            output = compute()
            if isinstance(output, Tensor):
                if output.device.type == "cuda":
                    torch.cuda.synchronize(output.device)
                elif output.device.type == "mps" and hasattr(torch, "mps"):
                    torch.mps.synchronize()
            compute_ms = (time.perf_counter() - started) * 1000
            if not isinstance(output, Tensor):
                raise TypeError("compute must return a torch.Tensor")
            self._record_computation(model_input, output)
            reason = forced_reason
            recomputed = True
        else:
            compute_ms = 0.0

        self._previous_input = model_input.detach().clone()
        self.next_step += 1
        return CacheStep(
            output=output,
            recomputed=recomputed,
            reason=reason,
            predicted_error=predicted_error,
            accumulated_error=self.accumulated_error,
            sensitivity=self.sensitivity,
            compute_ms=compute_ms,
        )
