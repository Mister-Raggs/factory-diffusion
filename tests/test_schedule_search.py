from __future__ import annotations

import unittest

import torch

from factory_diffusion.schedule_search import (
    ScheduleScore,
    paired_bootstrap_mean_ci,
    per_sample_action_errors,
    select_schedule,
)


class ScheduleSearchTest(unittest.TestCase):
    def test_per_sample_errors_preserve_pairing(self) -> None:
        reference = torch.zeros((2, 10, 2))
        candidate = reference.clone()
        candidate[0, 1, 0] = 0.5
        candidate[1, 1, 1] = 0.25

        error = per_sample_action_errors(
            reference,
            candidate,
            pixel_scale=torch.tensor([100.0, 200.0]),
        )

        torch.testing.assert_close(
            error.first_action_max_normalized,
            torch.tensor([0.5, 0.25]),
        )
        torch.testing.assert_close(
            error.first_action_max_pixels,
            torch.tensor([50.0, 50.0]),
        )

    def test_selection_uses_chunk_mse_before_first_action(self) -> None:
        selected = select_schedule(
            [
                ScheduleScore((80, 0), 0.2, 0.1),
                ScheduleScore((90, 0), 0.1, 10.0),
            ]
        )

        self.assertEqual(selected.schedule, (90, 0))

    def test_paired_bootstrap_is_deterministic(self) -> None:
        differences = torch.tensor([-3.0, -2.0, -1.0, -2.0])
        first = paired_bootstrap_mean_ci(differences, seed=4, resamples=1000)
        second = paired_bootstrap_mean_ci(differences, seed=4, resamples=1000)

        self.assertEqual(first, second)
        self.assertLess(first[1], 0)


if __name__ == "__main__":
    unittest.main()
