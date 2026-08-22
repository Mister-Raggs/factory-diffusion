"""Denoiser trace capture and baseline-path diagnostics."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def _synchronize(tensor: Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)
    elif tensor.device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _mean_absolute(tensor: Tensor) -> float:
    return float(tensor.detach().float().abs().mean())


def _timestep_value(timestep: Tensor | int | float) -> float:
    if isinstance(timestep, Tensor):
        return float(timestep.detach().flatten()[0])
    return float(timestep)


@dataclass(frozen=True)
class DenoisingStep:
    """One fully computed denoiser call."""

    index: int
    timestep: float
    model_input: Tensor
    model_output: Tensor
    elapsed_ms: float


@dataclass(frozen=True)
class StepDiagnostics:
    """Step-to-step measurements computed from an uncached trace."""

    index: int
    timestep: float
    elapsed_ms: float
    input_change_l1: float | None
    output_change_l1: float | None
    output_norm_l1: float
    sensitivity: float | None
    transformation_drift_l1: float | None
    relative_transformation_drift: float | None


@dataclass
class DenoisingTrace:
    """A complete uncached denoising trajectory."""

    steps: list[DenoisingStep]

    @property
    def total_model_ms(self) -> float:
        return sum(step.elapsed_ms for step in self.steps)

    def diagnostics(self, epsilon: float = 1e-8) -> list[StepDiagnostics]:
        rows: list[StepDiagnostics] = []
        previous: DenoisingStep | None = None
        for step in self.steps:
            output_norm = _mean_absolute(step.model_output)
            input_change = None
            output_change = None
            sensitivity = None
            transformation_drift = None
            relative_transformation_drift = None
            if previous is not None:
                input_change = _mean_absolute(step.model_input - previous.model_input)
                output_change = _mean_absolute(step.model_output - previous.model_output)
                if input_change > epsilon:
                    sensitivity = output_change / input_change
                previous_transform = previous.model_output - previous.model_input
                current_transform = step.model_output - step.model_input
                transformation_drift = _mean_absolute(current_transform - previous_transform)
                transform_norm = _mean_absolute(previous_transform)
                relative_transformation_drift = transformation_drift / max(transform_norm, epsilon)
            rows.append(
                StepDiagnostics(
                    index=step.index,
                    timestep=step.timestep,
                    elapsed_ms=step.elapsed_ms,
                    input_change_l1=input_change,
                    output_change_l1=output_change,
                    output_norm_l1=output_norm,
                    sensitivity=sensitivity,
                    transformation_drift_l1=transformation_drift,
                    relative_transformation_drift=relative_transformation_drift,
                )
            )
            previous = step
        return rows

    def save(self, directory: str | Path) -> None:
        """Save tensors and human-readable diagnostics without pickled objects."""

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, Tensor] = {}
        metadata = []
        for step in self.steps:
            tensors[f"input_{step.index:04d}"] = step.model_input.detach().cpu()
            tensors[f"output_{step.index:04d}"] = step.model_output.detach().cpu()
            metadata.append(
                {"index": step.index, "timestep": step.timestep, "elapsed_ms": step.elapsed_ms}
            )
        torch.save(tensors, target / "trace_tensors.pt")
        (target / "trace.json").write_text(
            json.dumps(
                {
                    "steps": metadata,
                    "diagnostics": [asdict(row) for row in self.diagnostics()],
                    "total_model_ms": self.total_model_ms,
                },
                indent=2,
            )
            + "\n"
        )


class TraceDenoiser(nn.Module):
    """Record every invocation while preserving the wrapped denoiser output."""

    def __init__(self, denoiser: nn.Module, *, auto_total_steps: int | None = None) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.auto_total_steps = auto_total_steps
        self.steps: list[DenoisingStep] = []
        self._active = False
        self._total_steps: int | None = None

    def begin_trajectory(self, total_steps: int) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.steps.clear()
        self._total_steps = total_steps
        self._active = True

    def forward(
        self,
        sample: Tensor,
        timestep: Tensor | int,
        global_cond: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        if not self._active:
            if self.auto_total_steps is None:
                raise RuntimeError("begin_trajectory(total_steps) must be called before tracing")
            self.begin_trajectory(self.auto_total_steps)

        _synchronize(sample)
        started = time.perf_counter()
        output = self.denoiser(sample, timestep, global_cond=global_cond, **kwargs)
        _synchronize(output)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.steps.append(
            DenoisingStep(
                index=len(self.steps),
                timestep=_timestep_value(timestep),
                model_input=sample.detach().clone(),
                model_output=output.detach().clone(),
                elapsed_ms=elapsed_ms,
            )
        )
        if self._total_steps == len(self.steps):
            self._active = False
        return output

    def trace(self) -> DenoisingTrace:
        if self._active:
            raise RuntimeError("the denoising trajectory is not complete")
        return DenoisingTrace(list(self.steps))
