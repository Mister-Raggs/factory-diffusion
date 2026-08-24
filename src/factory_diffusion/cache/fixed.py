"""Fixed-schedule transformation reuse for matched-NFE comparisons."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import torch
from torch import Tensor

from factory_diffusion.cache.adaptive import CacheStep


def guarded_uniform_schedule(
    total_steps: int,
    recomputations: int,
    *,
    warmup_steps: int = 2,
    force_compute_last: int = 2,
) -> tuple[int, ...]:
    """Place an exact compute budget while preserving warmup and final guards."""

    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 1 <= recomputations <= total_steps:
        raise ValueError("recomputations must be between one and total_steps")
    if not 0 <= warmup_steps <= total_steps:
        raise ValueError("warmup_steps must be between zero and total_steps")
    if not 0 <= force_compute_last <= total_steps:
        raise ValueError("force_compute_last must be between zero and total_steps")

    mandatory = {0}
    mandatory.update(range(warmup_steps))
    mandatory.update(range(total_steps - force_compute_last, total_steps))
    if recomputations < len(mandatory):
        raise ValueError("recomputations cannot be smaller than the initial/final guard union")

    candidates = [step for step in range(total_steps) if step not in mandatory]
    needed = recomputations - len(mandatory)
    selected = set(mandatory)
    if needed:
        for position in range(needed):
            candidate_index = round((position + 1) * (len(candidates) + 1) / (needed + 1)) - 1
            candidate_index = max(0, min(candidate_index, len(candidates) - 1))
            selected.add(candidates[candidate_index])

    if len(selected) != recomputations:
        raise RuntimeError("failed to construct an exact fixed compute schedule")
    return tuple(sorted(selected))


class FixedResidualCache:
    """Reuse the most recently computed output-input transformation."""

    def __init__(self, compute_steps: Iterable[int]) -> None:
        self.compute_steps = frozenset(int(step) for step in compute_steps)
        if not self.compute_steps or min(self.compute_steps) < 0:
            raise ValueError("compute_steps must contain non-negative indices")
        self.reset(total_steps=None)

    def reset(self, total_steps: int | None) -> None:
        if total_steps is not None:
            if total_steps < 1:
                raise ValueError("total_steps must be positive when provided")
            if max(self.compute_steps) >= total_steps:
                raise ValueError("compute_steps contains an index outside total_steps")
            if 0 not in self.compute_steps:
                raise ValueError("compute_steps must include step zero")
        self.total_steps = total_steps
        self.next_step = 0
        self.recomputed_steps = 0
        self._cached_transformation: Tensor | None = None

    @staticmethod
    def _synchronize(tensor: Tensor) -> None:
        if tensor.device.type == "cuda":
            torch.cuda.synchronize(tensor.device)
        elif tensor.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def run(
        self,
        step_index: int,
        model_input: Tensor,
        compute: Callable[[], Tensor],
    ) -> CacheStep:
        if step_index != self.next_step:
            raise ValueError(f"expected denoising step {self.next_step}, received {step_index}")
        if self.total_steps is not None and step_index >= self.total_steps:
            raise ValueError("step_index exceeds the trajectory length supplied to reset")

        if step_index in self.compute_steps:
            self._synchronize(model_input)
            started = time.perf_counter()
            output = compute()
            if not isinstance(output, Tensor):
                raise TypeError("compute must return a torch.Tensor")
            self._synchronize(output)
            compute_ms = (time.perf_counter() - started) * 1000
            if output.shape != model_input.shape:
                raise ValueError(
                    "fixed transformation reuse requires model input and output "
                    "to have equal shapes"
                )
            self._cached_transformation = output.detach().clone() - model_input.detach().clone()
            self.recomputed_steps += 1
            recomputed = True
            reason = "fixed-schedule"
        else:
            if self._cached_transformation is None:
                raise RuntimeError("fixed schedule cannot skip before its first computation")
            output = model_input + self._cached_transformation.to(
                device=model_input.device, dtype=model_input.dtype
            )
            compute_ms = 0.0
            recomputed = False
            reason = "fixed-reuse"

        self.next_step += 1
        return CacheStep(
            output=output,
            recomputed=recomputed,
            reason=reason,
            predicted_error=0.0,
            accumulated_error=0.0,
            sensitivity=None,
            compute_ms=compute_ms,
        )
