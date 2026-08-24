from __future__ import annotations

import unittest

import torch
from torch import nn

from factory_diffusion.cache import AdaptiveCacheConfig
from factory_diffusion.evaluation import (
    compare_runs,
    run_cached_sampler,
    run_explicit_schedule_sampler,
    run_fixed_sampler,
    run_uncached_sampler,
)
from factory_diffusion.schedules import standard_ddim_schedule


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


class FakeDDIMDiffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            from diffusers import DDIMScheduler
        except ImportError as error:  # pragma: no cover - optional dependency
            raise unittest.SkipTest(str(error)) from error
        self.unet = AdditiveDenoiser()
        self.num_inference_steps = 5
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: torch.Tensor,
        generator: torch.Generator,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        del batch_size
        sample = noise
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            output = self.unet(sample, timestep, global_cond=global_cond)
            sample = self.noise_scheduler.step(
                output,
                timestep,
                sample,
                generator=generator,
            ).prev_sample
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

    def test_reduced_step_run_restores_configured_steps(self) -> None:
        diffusion = FakeDiffusion()
        run = run_uncached_sampler(
            diffusion,
            global_cond=torch.zeros((1, 4)),
            noise=torch.zeros((1, 10, 2)),
            num_inference_steps=2,
        )

        self.assertEqual(len(run.trace.steps), 2)
        self.assertEqual(diffusion.num_inference_steps, 4)

    def test_fixed_sampler_uses_exact_budget(self) -> None:
        diffusion = FakeDiffusion()
        run = run_fixed_sampler(
            diffusion,
            global_cond=torch.zeros((1, 4)),
            noise=torch.zeros((1, 10, 2)),
            recomputations=3,
            warmup_steps=1,
            force_compute_last=1,
        )

        self.assertEqual(sum(step.recomputed for step in run.steps), 3)

    def test_explicit_standard_schedule_matches_native_sampler(self) -> None:
        diffusion = FakeDDIMDiffusion()
        noise = torch.randn((2, 10, 2), generator=torch.Generator().manual_seed(11))
        conditioning = torch.zeros((2, 4))
        schedule = standard_ddim_schedule(diffusion.noise_scheduler, 5)

        native = run_uncached_sampler(
            diffusion,
            global_cond=conditioning,
            noise=noise,
        )
        explicit = run_explicit_schedule_sampler(
            diffusion,
            global_cond=conditioning,
            noise=noise,
            timesteps=schedule,
        )

        torch.testing.assert_close(explicit.actions, native.actions)
        self.assertEqual(len(explicit.trace.steps), 5)


if __name__ == "__main__":
    unittest.main()
