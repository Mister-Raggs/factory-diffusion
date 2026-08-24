from __future__ import annotations

import unittest

import torch

from factory_diffusion.cache import FixedResidualCache, guarded_uniform_schedule


class FixedResidualCacheTest(unittest.TestCase):
    def test_guarded_schedules_have_exact_budgets(self) -> None:
        for budget in (5, 6, 7, 8):
            with self.subTest(budget=budget):
                schedule = guarded_uniform_schedule(10, budget)
                self.assertEqual(len(schedule), budget)
                self.assertTrue({0, 1, 8, 9}.issubset(schedule))

    def test_fixed_cache_reuses_transformation_on_unscheduled_steps(self) -> None:
        cache = FixedResidualCache((0, 2, 3))
        cache.reset(total_steps=4)
        calls = 0
        results = []
        for step in range(4):
            model_input = torch.tensor([float(step)])

            def compute(value: torch.Tensor = model_input) -> torch.Tensor:
                nonlocal calls
                calls += 1
                return value + 3

            results.append(cache.run(step, model_input, compute))

        self.assertEqual(calls, 3)
        torch.testing.assert_close(results[1].output, torch.tensor([4.0]))
        self.assertEqual(results[1].reason, "fixed-reuse")

    def test_step_zero_is_computed_even_without_a_warmup_guard(self) -> None:
        schedule = guarded_uniform_schedule(
            total_steps=4,
            recomputations=2,
            warmup_steps=0,
            force_compute_last=1,
        )

        self.assertEqual(schedule, (0, 3))


if __name__ == "__main__":
    unittest.main()
