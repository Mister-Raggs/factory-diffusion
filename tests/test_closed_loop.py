from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from factory_diffusion.closed_loop import (
    PushTNormalization,
    ScheduledPushTController,
    bootstrap_mean_interval,
    summarize_paired_episodes,
)


class FakeDiffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(
            num_train_timesteps=100,
            horizon=16,
            action_feature=SimpleNamespace(shape=(2,)),
            n_obs_steps=2,
            n_action_steps=8,
        )
        self.noise_scheduler = SimpleNamespace(config=self.config)


def normalization() -> PushTNormalization:
    return PushTNormalization(
        state_min=torch.zeros(2),
        state_max=torch.full((2,), 10.0),
        environment_min=torch.zeros(16),
        environment_max=torch.full((16,), 20.0),
        action_min=torch.zeros(2),
        action_max=torch.full((2,), 100.0),
    )


class ClosedLoopTest(unittest.TestCase):
    def test_stats_loading_and_action_unnormalization(self) -> None:
        stats = {
            "observation.state": {"min": [0, 0], "max": [10, 10]},
            "observation.environment_state": {"min": [0] * 16, "max": [20] * 16},
            "action": {"min": [0, 0], "max": [100, 100]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            path.write_text(json.dumps(stats))
            loaded = PushTNormalization.from_stats_file(path)

        torch.testing.assert_close(
            loaded.unnormalize_actions(torch.tensor([[-1.0, 1.0]])),
            torch.tensor([[0.0, 100.0]]),
        )

    def test_controller_pads_history_reuses_chunk_and_counts_nfe(self) -> None:
        calls = []

        def sampler(diffusion, *, global_cond, noise, timesteps):
            del diffusion
            calls.append((global_cond.clone(), noise.clone(), timesteps))
            actions = torch.zeros((1, 16, 2))
            actions[:, 1, 0] = 15.0  # unnormalizes above the environment bound
            return SimpleNamespace(actions=actions)

        controller = ScheduledPushTController(
            FakeDiffusion(),
            schedule=(70, 0),
            normalization=normalization(),
            episode_seed=4,
            sampler=sampler,
        )
        observation = {
            "agent_pos": np.array([5.0, 5.0], dtype=np.float32),
            "environment_state": np.full(16, 10.0, dtype=np.float32),
        }
        actions = [controller.act(observation) for _ in range(9)]

        self.assertEqual(len(calls), 2)
        self.assertEqual(tuple(calls[0][0].shape), (1, 36))
        self.assertEqual(controller.policy_queries, 2)
        self.assertEqual(controller.total_nfe, 4)
        self.assertEqual(actions[0][0], 512.0)
        self.assertEqual(controller.clipped_action_values, 2)
        self.assertEqual(controller.executed_action_values, 18)

    def test_bootstrap_and_paired_summary_preserve_seed_blocks(self) -> None:
        rows = []
        for seed in range(4):
            for budget in (2, 3):
                rows.extend(
                    [
                        {
                            "seed": seed,
                            "budget": budget,
                            "method": "standard",
                            "success": False,
                            "max_coverage": 0.5,
                        },
                        {
                            "seed": seed,
                            "budget": budget,
                            "method": "optimized",
                            "success": True,
                            "max_coverage": 0.8,
                        },
                    ]
                )
        summary = summarize_paired_episodes(rows, budgets=(2, 3), seeds=range(4))

        self.assertEqual(summary["decision"], "superior")
        self.assertEqual(summary["pooled_success_difference"]["mean"], 1.0)
        self.assertEqual(summary["per_budget"][0]["optimized_only_successes"], 4)
        interval = bootstrap_mean_interval(torch.tensor([-1.0, -1.0]), seed=0, resamples=10)
        self.assertEqual(interval["ci95"], [-1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
