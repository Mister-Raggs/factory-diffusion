"""One-chunk physical-sensitivity diagnostics for frozen DDIM schedules."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from factory_diffusion.closed_loop import (
    PUSHT_ACTION_MAX,
    PUSHT_ACTION_MIN,
    PushTNormalization,
)
from factory_diffusion.evaluation import run_explicit_schedule_sampler
from factory_diffusion.schedules import validate_ddim_schedule


@dataclass(frozen=True)
class BodyState:
    position: tuple[float, float]
    velocity: tuple[float, float]
    angle: float
    angular_velocity: float


@dataclass(frozen=True)
class PushTSnapshot:
    agent: BodyState
    block: BodyState


@dataclass(frozen=True)
class BranchOutcome:
    observation: Mapping[str, np.ndarray]
    coverage: float
    contacts: int
    executed_steps: int
    success: bool


@dataclass(frozen=True)
class SensitivityRow:
    seed: int
    query_index: int
    environment_step: int
    budget: int
    method: str
    schedule: tuple[int, ...]
    teacher_contact: bool
    teacher_coverage: float
    candidate_coverage: float
    coverage_difference: float
    absolute_coverage_error: float
    action_rmse_pixels: float
    keypoint_rmse_pixels: float
    agent_rmse_pixels: float
    teacher_contacts: int
    candidate_contacts: int
    teacher_executed_steps: int
    candidate_executed_steps: int
    candidate_success: bool
    clipped_action_values: int

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["schedule"] = list(self.schedule)
        return record


def _body_state(body: Any) -> BodyState:
    return BodyState(
        position=(float(body.position.x), float(body.position.y)),
        velocity=(float(body.velocity.x), float(body.velocity.y)),
        angle=float(body.angle),
        angular_velocity=float(body.angular_velocity),
    )


def capture_snapshot(env: Any) -> PushTSnapshot:
    """Capture the dynamic body state needed to reproduce a local PushT branch."""

    unwrapped = env.unwrapped
    return PushTSnapshot(agent=_body_state(unwrapped.agent), block=_body_state(unwrapped.block))


def restore_snapshot(env: Any, snapshot: PushTSnapshot) -> Mapping[str, np.ndarray]:
    """Reset a branch environment and restore positions and velocities."""

    state = [*snapshot.agent.position, *snapshot.block.position, snapshot.block.angle]
    env.reset(options={"reset_to_state": state})
    unwrapped = env.unwrapped
    for body, saved in ((unwrapped.agent, snapshot.agent), (unwrapped.block, snapshot.block)):
        body.position = saved.position
        body.velocity = saved.velocity
        body.angle = saved.angle
        body.angular_velocity = saved.angular_velocity
        unwrapped.space.reindex_shapes_for_body(body)
    unwrapped.n_contact_points = 0
    return unwrapped.get_obs()


def deterministic_noise(
    diffusion: nn.Module,
    *,
    episode_seed: int,
    query_index: int,
    noise_seed_base: int = 1_000_000,
) -> Tensor:
    parameter = next(diffusion.parameters())
    seed = noise_seed_base + int(episode_seed) * 10_000 + int(query_index)
    try:
        generator = torch.Generator(device=parameter.device).manual_seed(seed)
    except RuntimeError:
        generator = torch.Generator().manual_seed(seed)
    config = diffusion.config
    return torch.randn(
        (1, int(config.horizon), int(config.action_feature.shape[0])),
        device=parameter.device,
        dtype=parameter.dtype,
        generator=generator,
    )


def sample_action_chunk(
    diffusion: nn.Module,
    *,
    normalization: PushTNormalization,
    states: Sequence[Tensor],
    environment_states: Sequence[Tensor],
    noise: Tensor,
    schedule: Sequence[int],
    sampler: Callable[..., Any] = run_explicit_schedule_sampler,
) -> tuple[np.ndarray, int]:
    """Sample and unnormalize the eight actions executed by the PushT policy."""

    parameter = next(diffusion.parameters())
    schedule = validate_ddim_schedule(
        schedule,
        num_train_timesteps=int(diffusion.noise_scheduler.config.num_train_timesteps),
    )
    conditioning = normalization.global_conditioning(
        states,
        environment_states,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    run = sampler(
        diffusion,
        global_cond=conditioning,
        noise=noise,
        timesteps=schedule,
    )
    start = int(diffusion.config.n_obs_steps) - 1
    stop = start + int(diffusion.config.n_action_steps)
    raw = normalization.unnormalize_actions(run.actions[0, start:stop])
    clipped = raw.clamp(min=PUSHT_ACTION_MIN, max=PUSHT_ACTION_MAX)
    return clipped.cpu().numpy().astype(np.float32), int(raw.ne(clipped).sum())


def execute_branch(env: Any, snapshot: PushTSnapshot, actions: np.ndarray) -> BranchOutcome:
    """Execute one action chunk from a restored simulator snapshot."""

    observation = restore_snapshot(env, snapshot)
    coverage = float(env.unwrapped._get_coverage())
    contacts = 0
    success = coverage >= 0.95
    executed_steps = 0
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        executed_steps += 1
        coverage = float(info.get("coverage", reward))
        contacts += int(info.get("n_contacts", 0))
        success = success or bool(info.get("is_success", False))
        if terminated or truncated:
            break
    return BranchOutcome(
        observation=observation,
        coverage=coverage,
        contacts=contacts,
        executed_steps=executed_steps,
        success=success,
    )


def _append_observation(
    states: deque[Tensor],
    environment_states: deque[Tensor],
    observation: Mapping[str, np.ndarray],
) -> None:
    state = torch.as_tensor(observation["agent_pos"], dtype=torch.float32)
    environment = torch.as_tensor(observation["environment_state"], dtype=torch.float32)
    states.append(state)
    environment_states.append(environment)
    if len(states) == 1:
        states.appendleft(state.clone())
        environment_states.appendleft(environment.clone())


def collect_teacher_counterfactuals(
    diffusion: nn.Module,
    *,
    normalization: PushTNormalization,
    seed: int,
    teacher_schedule: Sequence[int],
    schedules: Mapping[tuple[int, str], Sequence[int]],
    max_steps: int = 300,
) -> list[SensitivityRow]:
    """Collect paired one-chunk branches along one DDIM-10 teacher trajectory."""

    try:
        import gym_pusht  # noqa: F401
        import gymnasium as gym
    except ImportError as error:  # pragma: no cover - optional simulation dependency
        raise RuntimeError("rollout sensitivity requires the lerobot pusht extra") from error

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    teacher_schedule = tuple(int(timestep) for timestep in teacher_schedule)
    candidate_schedules = {
        (int(budget), str(method)): tuple(int(timestep) for timestep in schedule)
        for (budget, method), schedule in schedules.items()
    }
    main_env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=max_steps,
    )
    branch_env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=max_steps,
    )
    states: deque[Tensor] = deque(maxlen=2)
    environment_states: deque[Tensor] = deque(maxlen=2)
    rows: list[SensitivityRow] = []
    try:
        observation, _ = main_env.reset(seed=seed)
        environment_step = 0
        query_index = 0
        done = False
        while environment_step < max_steps and not done:
            _append_observation(states, environment_states, observation)
            noise = deterministic_noise(
                diffusion,
                episode_seed=seed,
                query_index=query_index,
            )
            teacher_actions, _ = sample_action_chunk(
                diffusion,
                normalization=normalization,
                states=tuple(states),
                environment_states=tuple(environment_states),
                noise=noise,
                schedule=teacher_schedule,
            )
            snapshot = capture_snapshot(main_env)
            teacher = execute_branch(branch_env, snapshot, teacher_actions)

            for (budget, method), schedule in candidate_schedules.items():
                candidate_actions, clipped = sample_action_chunk(
                    diffusion,
                    normalization=normalization,
                    states=tuple(states),
                    environment_states=tuple(environment_states),
                    noise=noise,
                    schedule=schedule,
                )
                candidate = execute_branch(branch_env, snapshot, candidate_actions)
                action_rmse = float(np.sqrt(np.mean((candidate_actions - teacher_actions) ** 2)))
                keypoint_rmse = float(
                    np.sqrt(
                        np.mean(
                            (
                                candidate.observation["environment_state"]
                                - teacher.observation["environment_state"]
                            )
                            ** 2
                        )
                    )
                )
                agent_rmse = float(
                    np.sqrt(
                        np.mean(
                            (candidate.observation["agent_pos"] - teacher.observation["agent_pos"])
                            ** 2
                        )
                    )
                )
                coverage_difference = candidate.coverage - teacher.coverage
                rows.append(
                    SensitivityRow(
                        seed=seed,
                        query_index=query_index,
                        environment_step=environment_step,
                        budget=budget,
                        method=method,
                        schedule=schedule,
                        teacher_contact=teacher.contacts > 0,
                        teacher_coverage=teacher.coverage,
                        candidate_coverage=candidate.coverage,
                        coverage_difference=coverage_difference,
                        absolute_coverage_error=abs(coverage_difference),
                        action_rmse_pixels=action_rmse,
                        keypoint_rmse_pixels=keypoint_rmse,
                        agent_rmse_pixels=agent_rmse,
                        teacher_contacts=teacher.contacts,
                        candidate_contacts=candidate.contacts,
                        teacher_executed_steps=teacher.executed_steps,
                        candidate_executed_steps=candidate.executed_steps,
                        candidate_success=candidate.success,
                        clipped_action_values=clipped,
                    )
                )

            for action_index, action in enumerate(teacher_actions):
                if action_index > 0:
                    _append_observation(states, environment_states, observation)
                observation, _, terminated, truncated, _ = main_env.step(action)
                environment_step += 1
                done = bool(terminated or truncated)
                if done or environment_step >= max_steps:
                    break
            query_index += 1
    finally:
        main_env.close()
        branch_env.close()
    return rows


def summarize_sensitivity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize paired optimized-minus-standard rollout divergences."""

    if not rows:
        raise ValueError("at least one sensitivity row is required")
    metrics = (
        "action_rmse_pixels",
        "keypoint_rmse_pixels",
        "agent_rmse_pixels",
        "absolute_coverage_error",
        "coverage_difference",
    )
    keyed = {
        (
            int(row["seed"]),
            int(row["query_index"]),
            int(row["budget"]),
            str(row["method"]),
        ): row
        for row in rows
    }
    budgets = sorted({int(row["budget"]) for row in rows})
    summary = []
    for budget in budgets:
        pairs = sorted(
            {
                (seed, query)
                for seed, query, candidate_budget, _ in keyed
                if candidate_budget == budget
                and (seed, query, budget, "standard") in keyed
                and (seed, query, budget, "optimized") in keyed
            }
        )
        record: dict[str, Any] = {"budget": budget, "paired_snapshots": len(pairs)}
        for phase, selected in (
            ("all", pairs),
            (
                "contact",
                [
                    pair
                    for pair in pairs
                    if bool(keyed[(pair[0], pair[1], budget, "standard")]["teacher_contact"])
                ],
            ),
        ):
            phase_record: dict[str, Any] = {"paired_snapshots": len(selected)}
            for metric in metrics:
                standard = np.asarray(
                    [
                        float(keyed[(seed, query, budget, "standard")][metric])
                        for seed, query in selected
                    ]
                )
                optimized = np.asarray(
                    [
                        float(keyed[(seed, query, budget, "optimized")][metric])
                        for seed, query in selected
                    ]
                )
                phase_record[metric] = {
                    "standard_mean": float(standard.mean()) if len(standard) else None,
                    "optimized_mean": float(optimized.mean()) if len(optimized) else None,
                    "optimized_minus_standard_mean": (
                        float((optimized - standard).mean()) if len(standard) else None
                    ),
                }
                if len(selected):
                    seed_differences = []
                    for seed in sorted({seed for seed, _ in selected}):
                        seed_indices = [
                            index
                            for index, (selected_seed, _) in enumerate(selected)
                            if selected_seed == seed
                        ]
                        seed_differences.append(
                            float((optimized[seed_indices] - standard[seed_indices]).mean())
                        )
                    seed_values = np.asarray(seed_differences)
                    rng = np.random.default_rng(20_000 + budget)
                    bootstrap_indices = rng.integers(
                        0,
                        len(seed_values),
                        size=(10_000, len(seed_values)),
                    )
                    bootstrap_means = seed_values[bootstrap_indices].mean(axis=1)
                    phase_record[metric].update(
                        {
                            "paired_seeds": len(seed_values),
                            "seed_balanced_difference": float(seed_values.mean()),
                            "seed_bootstrap_ci95": [
                                float(np.quantile(bootstrap_means, 0.025)),
                                float(np.quantile(bootstrap_means, 0.975)),
                            ],
                        }
                    )
            record[phase] = phase_record
        summary.append(record)

    by_budget = {record["budget"]: record for record in summary}
    primary_differences = {
        budget: by_budget[budget]["all"]["keypoint_rmse_pixels"]["seed_balanced_difference"]
        for budget in budgets
    }
    diagnostic_alignment = bool(
        3 in primary_differences
        and 5 in primary_differences
        and primary_differences[3] <= 0
        and primary_differences[5] > 0
    )
    return {
        "per_budget": summary,
        "primary_metric": "one-chunk final keypoint RMSE from DDIM-10",
        "diagnostic_alignment": diagnostic_alignment,
        "interpretation": (
            "proxy matches the observed Experiment 3 signs at NFE 3 and 5"
            if diagnostic_alignment
            else "proxy does not match the observed Experiment 3 signs at NFE 3 and 5"
        ),
    }
