"""Closed-loop PushT control with frozen explicit DDIM schedules."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from factory_diffusion.baselines.pusht_data import min_max_normalize
from factory_diffusion.evaluation import run_explicit_schedule_sampler
from factory_diffusion.schedules import validate_ddim_schedule

PUSHT_ACTION_MIN = 0.0
PUSHT_ACTION_MAX = 512.0


@dataclass(frozen=True)
class PushTNormalization:
    state_min: Tensor
    state_max: Tensor
    environment_min: Tensor
    environment_max: Tensor
    action_min: Tensor
    action_max: Tensor

    @classmethod
    def from_stats_file(cls, path: str | Path) -> PushTNormalization:
        stats = json.loads(Path(path).read_text())

        def value(feature: str, statistic: str) -> Tensor:
            try:
                return torch.tensor(stats[feature][statistic], dtype=torch.float32)
            except KeyError as error:
                raise KeyError(f"stats are missing {feature}.{statistic}") from error

        return cls(
            state_min=value("observation.state", "min"),
            state_max=value("observation.state", "max"),
            environment_min=value("observation.environment_state", "min"),
            environment_max=value("observation.environment_state", "max"),
            action_min=value("action", "min"),
            action_max=value("action", "max"),
        )

    def global_conditioning(
        self,
        states: Sequence[Tensor],
        environment_states: Sequence[Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if len(states) != 2 or len(environment_states) != 2:
            raise ValueError("PushT conditioning requires exactly two observation frames")
        state = torch.stack(tuple(states)).to(device=device, dtype=dtype)
        environment = torch.stack(tuple(environment_states)).to(device=device, dtype=dtype)
        normalized_state = min_max_normalize(state, self.state_min, self.state_max)
        normalized_environment = min_max_normalize(
            environment,
            self.environment_min,
            self.environment_max,
        )
        return torch.cat((normalized_state, normalized_environment), dim=-1).flatten().unsqueeze(0)

    def unnormalize_actions(self, actions: Tensor) -> Tensor:
        minimum = self.action_min.to(device=actions.device, dtype=actions.dtype)
        maximum = self.action_max.to(device=actions.device, dtype=actions.dtype)
        return (actions + 1) * (maximum - minimum) / 2 + minimum


@dataclass(frozen=True)
class EpisodeResult:
    method: str
    budget: int
    schedule: tuple[int, ...]
    seed: int
    success: bool
    steps: int
    steps_to_success: int | None
    max_coverage: float
    sum_reward: float
    policy_queries: int
    total_nfe: int
    clipped_action_values: int
    executed_action_values: int

    @property
    def clipping_fraction(self) -> float:
        if self.executed_action_values == 0:
            return 0.0
        return self.clipped_action_values / self.executed_action_values

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["schedule"] = list(self.schedule)
        record["clipping_fraction"] = self.clipping_fraction
        return record


class ScheduledPushTController:
    """Execute eight-action chunks sampled with one frozen DDIM schedule."""

    def __init__(
        self,
        diffusion: nn.Module,
        *,
        schedule: Sequence[int],
        normalization: PushTNormalization,
        episode_seed: int,
        noise_seed_base: int = 1_000_000,
        sampler: Callable[..., Any] = run_explicit_schedule_sampler,
    ) -> None:
        scheduler = getattr(diffusion, "noise_scheduler", None)
        if scheduler is None:
            raise TypeError("diffusion must expose a noise_scheduler")
        self.diffusion = diffusion
        self.schedule = validate_ddim_schedule(
            schedule,
            num_train_timesteps=int(scheduler.config.num_train_timesteps),
        )
        self.normalization = normalization
        self.episode_seed = int(episode_seed)
        self.noise_seed_base = int(noise_seed_base)
        self.sampler = sampler
        self._states: deque[Tensor] = deque(maxlen=2)
        self._environment_states: deque[Tensor] = deque(maxlen=2)
        self._actions: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self.policy_queries = 0
        self.clipped_action_values = 0
        self.executed_action_values = 0

    @property
    def total_nfe(self) -> int:
        return self.policy_queries * len(self.schedule)

    def _append_observation(self, observation: Mapping[str, np.ndarray]) -> None:
        if "agent_pos" not in observation or "environment_state" not in observation:
            raise KeyError("PushT observation must contain agent_pos and environment_state")
        state = torch.as_tensor(observation["agent_pos"], dtype=torch.float32)
        environment = torch.as_tensor(observation["environment_state"], dtype=torch.float32)
        if tuple(state.shape) != (2,) or tuple(environment.shape) != (16,):
            raise ValueError("unexpected PushT keypoint observation shape")
        self._states.append(state)
        self._environment_states.append(environment)
        if len(self._states) == 1:
            self._states.appendleft(state.clone())
            self._environment_states.appendleft(environment.clone())

    def _noise(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        seed = self.noise_seed_base + self.episode_seed * 10_000 + self.policy_queries
        try:
            generator = torch.Generator(device=device).manual_seed(seed)
        except RuntimeError:
            generator = torch.Generator().manual_seed(seed)
        config = getattr(self.diffusion, "config", None)
        if config is None:
            raise TypeError("diffusion must expose its config")
        return torch.randn(
            (1, int(config.horizon), int(config.action_feature.shape[0])),
            device=device,
            dtype=dtype,
            generator=generator,
        )

    def _refill_actions(self) -> None:
        parameter = next(self.diffusion.parameters())
        conditioning = self.normalization.global_conditioning(
            tuple(self._states),
            tuple(self._environment_states),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        noise = self._noise(device=parameter.device, dtype=parameter.dtype)
        run = self.sampler(
            self.diffusion,
            global_cond=conditioning,
            noise=noise,
            timesteps=self.schedule,
        )
        config = self.diffusion.config
        start = int(config.n_obs_steps) - 1
        stop = start + int(config.n_action_steps)
        normalized = run.actions[0, start:stop]
        raw = self.normalization.unnormalize_actions(normalized)
        clipped = raw.clamp(min=PUSHT_ACTION_MIN, max=PUSHT_ACTION_MAX)
        masks = raw.ne(clipped)
        for action, mask in zip(clipped.cpu().numpy(), masks.cpu().numpy(), strict=True):
            self._actions.append((action.astype(np.float32), mask))
        self.policy_queries += 1

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        self._append_observation(observation)
        if not self._actions:
            self._refill_actions()
        action, clipped_mask = self._actions.popleft()
        self.clipped_action_values += int(clipped_mask.sum())
        self.executed_action_values += int(clipped_mask.size)
        return action


def rollout_pusht_episode(
    diffusion: nn.Module,
    *,
    normalization: PushTNormalization,
    method: str,
    schedule: Sequence[int],
    seed: int,
    max_steps: int = 300,
) -> EpisodeResult:
    """Run one deterministic, non-rendered keypoint PushT episode."""

    try:
        import gym_pusht  # noqa: F401
        import gymnasium as gym
    except ImportError as error:  # pragma: no cover - optional simulation dependency
        raise RuntimeError("closed-loop PushT requires the lerobot pusht extra") from error

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    schedule = tuple(int(timestep) for timestep in schedule)
    controller = ScheduledPushTController(
        diffusion,
        schedule=schedule,
        normalization=normalization,
        episode_seed=seed,
    )
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=max_steps,
    )
    success = False
    steps_to_success = None
    max_coverage = 0.0
    sum_reward = 0.0
    steps = 0
    try:
        observation, _ = env.reset(seed=seed)
        for step in range(1, max_steps + 1):
            action = controller.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            steps = step
            coverage = float(info.get("coverage", reward))
            max_coverage = max(max_coverage, coverage)
            sum_reward += float(reward)
            if bool(info.get("is_success", False)):
                success = True
                steps_to_success = step
            if terminated or truncated:
                break
    finally:
        env.close()

    return EpisodeResult(
        method=method,
        budget=len(schedule),
        schedule=schedule,
        seed=seed,
        success=success,
        steps=steps,
        steps_to_success=steps_to_success,
        max_coverage=max_coverage,
        sum_reward=sum_reward,
        policy_queries=controller.policy_queries,
        total_nfe=controller.total_nfe,
        clipped_action_values=controller.clipped_action_values,
        executed_action_values=controller.executed_action_values,
    )


def bootstrap_mean_interval(
    values: Tensor,
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, float | list[float]]:
    """Return a mean, two-sided interval, and one-sided lower bound."""

    values = values.detach().float().cpu()
    if values.ndim not in (1, 2) or values.shape[0] < 2:
        raise ValueError("bootstrap values must contain at least two paired seed rows")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(values.shape[0], (resamples, values.shape[0]), generator=generator)
    means = values[indices].mean(dim=tuple(range(1, values.ndim + 1)))
    quantiles = torch.quantile(means, torch.tensor([0.025, 0.05, 0.975]))
    return {
        "mean": float(values.mean()),
        "ci95": [float(quantiles[0]), float(quantiles[2])],
        "lower95_one_sided": float(quantiles[1]),
    }


def summarize_paired_episodes(
    rows: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int],
    seeds: Sequence[int],
    bootstrap_seed: int = 3000,
) -> dict[str, Any]:
    """Summarize paired standard/optimized episodes with seed-block inference."""

    keyed = {(int(row["seed"]), int(row["budget"]), str(row["method"])): row for row in rows}
    expected = {
        (int(seed), int(budget), method)
        for seed in seeds
        for budget in budgets
        for method in ("standard", "optimized")
    }
    if set(keyed) != expected:
        raise ValueError("rows do not form the requested complete paired design")

    success_matrix = torch.tensor(
        [
            [
                float(keyed[(seed, budget, "optimized")]["success"])
                - float(keyed[(seed, budget, "standard")]["success"])
                for budget in budgets
            ]
            for seed in seeds
        ]
    )
    per_budget = []
    for budget_index, budget in enumerate(budgets):
        standard = [keyed[(seed, budget, "standard")] for seed in seeds]
        optimized = [keyed[(seed, budget, "optimized")] for seed in seeds]
        differences = success_matrix[:, budget_index]
        per_budget.append(
            {
                "budget": budget,
                "standard_successes": sum(bool(row["success"]) for row in standard),
                "optimized_successes": sum(bool(row["success"]) for row in optimized),
                "success_difference": bootstrap_mean_interval(
                    differences,
                    seed=bootstrap_seed + budget,
                ),
                "standard_only_successes": sum(
                    bool(left["success"]) and not bool(right["success"])
                    for left, right in zip(standard, optimized, strict=True)
                ),
                "optimized_only_successes": sum(
                    bool(right["success"]) and not bool(left["success"])
                    for left, right in zip(standard, optimized, strict=True)
                ),
                "mean_max_coverage_standard": float(
                    np.mean([float(row["max_coverage"]) for row in standard])
                ),
                "mean_max_coverage_optimized": float(
                    np.mean([float(row["max_coverage"]) for row in optimized])
                ),
            }
        )

    pooled = bootstrap_mean_interval(success_matrix, seed=bootstrap_seed)
    noninferior = bool(
        float(pooled["lower95_one_sided"]) > -0.05
        and all(float(record["success_difference"]["mean"]) >= -0.05 for record in per_budget)
    )
    superior = bool(float(pooled["ci95"][0]) > 0)
    decision = (
        "superior" if superior else "non-inferior" if noninferior else "inconclusive-or-negative"
    )
    return {
        "pooled_success_difference": pooled,
        "per_budget": per_budget,
        "decision": decision,
    }
