"""Boundary implemented later by the main Factory SRE repository."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FactorySREEnvironment(Protocol):
    """Minimal closed-loop contract required by the evaluation harness."""

    def reset(self, seed: int) -> dict[str, Any]: ...

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]: ...

    def task_succeeded(self, info: dict[str, Any]) -> bool: ...


@runtime_checkable
class FactorySREExpert(Protocol):
    """Scripted controller used to collect imitation-learning episodes."""

    def action(self, observation: dict[str, Any]) -> Any: ...
