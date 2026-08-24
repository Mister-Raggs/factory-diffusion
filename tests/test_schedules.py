from __future__ import annotations

import math
import unittest

import torch

from factory_diffusion.schedules import (
    ddim_step_to,
    grid_schedules,
    standard_ddim_schedule,
    validate_ddim_schedule,
)


class ScheduleTest(unittest.TestCase):
    def test_grid_schedules_have_exact_budget_and_terminal_zero(self) -> None:
        for budget in (2, 3, 4, 5):
            with self.subTest(budget=budget):
                schedules = grid_schedules(budget)
                self.assertEqual(len(schedules), math.comb(9, budget - 1))
                self.assertTrue(all(len(schedule) == budget for schedule in schedules))
                self.assertTrue(all(schedule[-1] == 0 for schedule in schedules))

    def test_invalid_schedules_are_rejected(self) -> None:
        invalid = [(), (90, 80), (90, 80, 80, 0), (80, 90, 0), (100, 0), (-1,)]
        for schedule in invalid:
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                validate_ddim_schedule(schedule, num_train_timesteps=100)

    def test_explicit_transition_matches_standard_diffusers_schedule(self) -> None:
        try:
            from diffusers import DDIMScheduler
        except ImportError as error:  # pragma: no cover - optional dependency
            self.skipTest(str(error))

        scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        schedule = standard_ddim_schedule(scheduler, 5)
        scheduler.set_timesteps(5)
        standard = torch.randn((2, 16, 2), generator=torch.Generator().manual_seed(7))
        explicit = standard.clone()

        for index, timestep in enumerate(schedule):
            standard_output = standard * 0.1 + timestep / 1000
            explicit_output = explicit * 0.1 + timestep / 1000
            standard = scheduler.step(standard_output, timestep, standard).prev_sample
            previous_timestep = schedule[index + 1] if index + 1 < len(schedule) else -1
            explicit = ddim_step_to(
                scheduler,
                explicit_output,
                timestep,
                previous_timestep,
                explicit,
            )

        torch.testing.assert_close(explicit, standard)


if __name__ == "__main__":
    unittest.main()
