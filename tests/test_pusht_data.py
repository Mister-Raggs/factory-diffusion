from __future__ import annotations

import unittest

import torch

from factory_diffusion.baselines.pusht_data import (
    min_max_normalize,
    phase_stratified_indices,
    prepare_global_conditioning,
)


class PushTDataTest(unittest.TestCase):
    def test_min_max_normalization_matches_expected_range(self) -> None:
        values = torch.tensor([[0.0, 15.0], [10.0, 20.0]])
        normalized = min_max_normalize(
            values,
            torch.tensor([0.0, 10.0]),
            torch.tensor([10.0, 20.0]),
        )
        torch.testing.assert_close(
            normalized,
            torch.tensor([[-1.0, 0.0], [1.0, 1.0]]),
        )

    def test_global_conditioning_has_checkpoint_shape(self) -> None:
        conditioning = prepare_global_conditioning(
            torch.zeros((2, 2)),
            torch.zeros((2, 16)),
            state_min=torch.zeros(2),
            state_max=torch.ones(2),
            environment_min=torch.zeros(16),
            environment_max=torch.ones(16),
        )
        self.assertEqual(conditioning.shape, (36,))
        self.assertTrue(torch.all(conditioning == -1))

    def test_phase_sampling_is_balanced_and_deterministic(self) -> None:
        episodes = [
            {"dataset_from_index": 0, "dataset_to_index": 9},
            {"dataset_from_index": 9, "dataset_to_index": 18},
        ]
        first = phase_stratified_indices(episodes, 9, seed=7)
        second = phase_stratified_indices(episodes, 9, seed=7)

        self.assertEqual(first, second)
        phases = {phase for _, phase in first}
        counts = {phase: sum(value == phase for _, value in first) for phase in phases}
        self.assertEqual(counts, {"early": 3, "middle": 3, "late": 3})


if __name__ == "__main__":
    unittest.main()
