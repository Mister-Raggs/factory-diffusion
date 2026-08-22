from __future__ import annotations

import unittest

import torch

from factory_diffusion.cache import AdaptiveCacheConfig, AdaptiveResidualCache


class AdaptiveResidualCacheTest(unittest.TestCase):
    def test_constant_transformation_can_be_reused(self) -> None:
        cache = AdaptiveResidualCache(
            AdaptiveCacheConfig(threshold=1.0, warmup_steps=2, force_compute_last=1)
        )
        cache.reset(total_steps=4)
        calls = 0
        reasons = []

        for step in range(4):
            model_input = torch.tensor([float(step)])

            def compute() -> torch.Tensor:
                nonlocal calls
                calls += 1
                return model_input + 2

            result = cache.run(step, model_input, compute)
            torch.testing.assert_close(result.output, model_input + 2)
            reasons.append(result.reason)

        self.assertEqual(calls, 3)
        self.assertEqual(reasons, ["warmup", "warmup", "cached", "final-step"])

    def test_zero_threshold_disables_skipping(self) -> None:
        cache = AdaptiveResidualCache(AdaptiveCacheConfig(threshold=0.0, warmup_steps=2))
        cache.reset(total_steps=4)

        results = []
        for step in range(4):
            model_input = torch.tensor([float(step)])
            results.append(cache.run(step, model_input, lambda: model_input.square()))

        self.assertTrue(all(result.recomputed for result in results))

    def test_reset_prevents_state_from_crossing_trajectories(self) -> None:
        cache = AdaptiveResidualCache(AdaptiveCacheConfig(threshold=10.0, warmup_steps=1))
        cache.reset(total_steps=2)
        cache.run(0, torch.tensor([0.0]), lambda: torch.tensor([1.0]))

        cache.reset(total_steps=2)
        result = cache.run(0, torch.tensor([5.0]), lambda: torch.tensor([9.0]))

        self.assertTrue(result.recomputed)
        self.assertEqual(result.reason, "warmup")
        self.assertIsNone(result.sensitivity)

    def test_steps_must_be_ordered(self) -> None:
        cache = AdaptiveResidualCache()
        cache.reset(total_steps=3)

        with self.assertRaisesRegex(ValueError, "expected denoising step 0"):
            cache.run(1, torch.zeros(1), lambda: torch.zeros(1))

    def test_invalid_config_is_rejected(self) -> None:
        invalid = [
            {"threshold": -1},
            {"warmup_steps": 0},
            {"force_compute_last": -1},
            {"max_consecutive_skips": 0},
            {"epsilon": 0},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AdaptiveCacheConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
