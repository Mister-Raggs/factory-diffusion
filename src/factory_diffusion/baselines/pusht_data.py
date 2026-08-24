"""Real keypoint-only PushT conditioning for the Phase 1.5 gate."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

DATASET_REPO = "lerobot/pusht_keypoints"
DATASET_REVISION = "ace8c161a68bc025c21a5f29f85b86a9a2c5e64b"
OBSERVATION_DELTAS_SECONDS = (-0.1, 0.0)


@dataclass(frozen=True)
class PushTConditioningSample:
    global_cond: Tensor
    dataset_index: int
    episode_index: int
    frame_index: int
    phase: str
    history_is_padded: bool


def min_max_normalize(values: Tensor, minimum: Tensor, maximum: Tensor) -> Tensor:
    """Match LeRobot's MIN_MAX processor and map values to [-1, 1]."""

    minimum = minimum.to(device=values.device, dtype=values.dtype)
    maximum = maximum.to(device=values.device, dtype=values.dtype)
    denominator = maximum - minimum
    epsilon = torch.tensor(1e-8, device=values.device, dtype=values.dtype)
    denominator = torch.where(denominator == 0, epsilon, denominator)
    return 2 * (values - minimum) / denominator - 1


def prepare_global_conditioning(
    state: Tensor,
    environment_state: Tensor,
    *,
    state_min: Tensor,
    state_max: Tensor,
    environment_min: Tensor,
    environment_max: Tensor,
) -> Tensor:
    """Normalize and flatten two PushT observation frames to 36 dimensions."""

    if state.shape != (2, 2):
        raise ValueError(f"expected state shape (2, 2), received {tuple(state.shape)}")
    if environment_state.shape != (2, 16):
        raise ValueError(
            f"expected environment_state shape (2, 16), received {tuple(environment_state.shape)}"
        )
    normalized_state = min_max_normalize(state, state_min, state_max)
    normalized_environment = min_max_normalize(
        environment_state,
        environment_min,
        environment_max,
    )
    return torch.cat((normalized_state, normalized_environment), dim=-1).flatten()


def phase_stratified_indices(
    episodes: Sequence[Mapping[str, Any]],
    count: int,
    *,
    seed: int,
) -> list[tuple[int, str]]:
    """Sample dataset rows evenly from episode-progress thirds."""

    if count < 1:
        raise ValueError("count must be positive")
    buckets: dict[str, list[int]] = {"early": [], "middle": [], "late": []}
    labels = tuple(buckets)
    for episode in episodes:
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        if stop <= start:
            raise ValueError("episode bounds must be non-empty and increasing")
        length = stop - start
        for offset in range(length):
            phase_index = min(2, (offset * 3) // length)
            buckets[labels[phase_index]].append(start + offset)

    allocations = {label: count // 3 for label in labels}
    for label in labels[: count % 3]:
        allocations[label] += 1
    if any(allocations[label] > len(buckets[label]) for label in labels):
        raise ValueError("requested more samples than a phase bucket contains")

    rng = random.Random(seed)
    selected = [
        (index, label)
        for label in labels
        for index in rng.sample(buckets[label], allocations[label])
    ]
    rng.shuffle(selected)
    return selected


def _stat_tensor(stats: Mapping[str, Any], feature: str, statistic: str) -> Tensor:
    try:
        value = stats[feature][statistic]
    except KeyError as error:
        raise KeyError(f"dataset stats are missing {feature}.{statistic}") from error
    return torch.as_tensor(value)


def load_real_conditioning_samples(
    *,
    count: int,
    seed: int,
    root: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    repo_id: str = DATASET_REPO,
    revision: str = DATASET_REVISION,
) -> list[PushTConditioningSample]:
    """Download/load the pinned tabular dataset and return normalized samples."""

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:  # pragma: no cover - exercised without optional dependency
        raise RuntimeError("load_real_conditioning_samples requires the lerobot extra") from error

    dataset = LeRobotDataset(
        repo_id,
        root=root,
        revision=revision,
        download_videos=False,
        delta_timestamps={
            "observation.state": list(OBSERVATION_DELTAS_SECONDS),
            "observation.environment_state": list(OBSERVATION_DELTAS_SECONDS),
        },
    )
    episodes = [dataset.meta.episodes[index] for index in range(len(dataset.meta.episodes))]
    selected = phase_stratified_indices(episodes, count, seed=seed)
    stats = dataset.meta.stats
    state_min = _stat_tensor(stats, "observation.state", "min")
    state_max = _stat_tensor(stats, "observation.state", "max")
    environment_min = _stat_tensor(stats, "observation.environment_state", "min")
    environment_max = _stat_tensor(stats, "observation.environment_state", "max")

    samples = []
    for dataset_index, phase in selected:
        item = dataset[dataset_index]
        global_cond = prepare_global_conditioning(
            item["observation.state"],
            item["observation.environment_state"],
            state_min=state_min,
            state_max=state_max,
            environment_min=environment_min,
            environment_max=environment_max,
        ).to(device=device, dtype=dtype)
        samples.append(
            PushTConditioningSample(
                global_cond=global_cond,
                dataset_index=dataset_index,
                episode_index=int(item["episode_index"]),
                frame_index=int(item["frame_index"]),
                phase=phase,
                history_is_padded=bool(
                    item["observation.state_is_pad"].any()
                    or item["observation.environment_state_is_pad"].any()
                ),
            )
        )
    return samples
