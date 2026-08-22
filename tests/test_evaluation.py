from __future__ import annotations

import unittest

import torch
from torch import nn

from factory_diffusion.cache import AdaptiveCacheConfig
from factory_diffusion.evaluation import compare_runs, run_cached_sampler, run_uncached_sampler


class AdditiveDenoiser(nn.Module):
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep, global_cond
        return sample + 2


class FakeDiffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = AdditiveDenoiser()
        self.num_inference_steps = 4

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: torch.Tensor,
        generator: torch.Generator,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        del batch_size, generator
        sample = noise
        for timestep in range(self.num_inference_steps):
            output = self.unet(sample, timestep, global_cond=global_cond)
            sample = sample + output * 0.1
        return sample


class EvaluationTest(unittest.TestCase):
    def test_exact_online_pair_restores_model_and_preserves_actions(self) -> None:
        diffusion = FakeDiffusion()
        original = diffusion.unet
        noise = torch.zeros((1, 10, 2))
        conditioning = torch.zeros((1, 4))

        baseline = run_uncached_sampler(
            diffusion,
            global_cond=conditioning,
            noise=noise,
        )
        cached = run_cached_sampler(
            diffusion,
            global_cond=conditioning,
            noise=noise,
            cache_config=AdaptiveCacheConfig(
                threshold=2.0,
                warmup_steps=2,
                force_compute_last=1,
            ),
        )
        paired = compare_runs(baseline, cached, n_obs_steps=2, n_action_steps=8)

        self.assertIs(diffusion.unet, original)
        self.assertEqual(cached.skipped_steps, 1)
        self.assertEqual(paired.action_chunk_mse, 0.0)
        self.assertEqual(paired.first_action_max_error, 0.0)


if __name__ == "__main__":
    unittest.main()
