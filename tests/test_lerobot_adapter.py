from __future__ import annotations

import unittest

import torch
from torch import nn

from factory_diffusion.cache import AdaptiveCacheConfig
from factory_diffusion.integrations.lerobot import CachedDenoiser, install_on_lerobot_policy


class AdditiveDenoiser(nn.Module):
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep, global_cond
        return sample + 3


class FakeDiffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = AdditiveDenoiser()
        self.num_inference_steps = 3


class FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diffusion = FakeDiffusion()


class CachedDenoiserTest(unittest.TestCase):
    def test_adapter_resets_and_closes_a_trajectory(self) -> None:
        adapter = CachedDenoiser(
            AdditiveDenoiser(),
            AdaptiveCacheConfig(threshold=2.0, warmup_steps=2, force_compute_last=0),
        )
        adapter.eval()
        adapter.begin_trajectory(total_steps=3)

        for step in range(3):
            sample = torch.tensor([float(step)])
            torch.testing.assert_close(adapter(sample, step), sample + 3)

        self.assertEqual(len(adapter.steps), 3)
        self.assertFalse(adapter._active)
        with self.assertRaisesRegex(RuntimeError, "begin_trajectory"):
            adapter(torch.zeros(1), 0)

    def test_install_uses_policy_step_count_and_auto_resets(self) -> None:
        policy = FakePolicy()
        wrapped = install_on_lerobot_policy(
            policy,
            AdaptiveCacheConfig(threshold=2.0, warmup_steps=2, force_compute_last=0),
        )
        policy.eval()

        for step in range(3):
            sample = torch.tensor([float(step)])
            torch.testing.assert_close(policy.diffusion.unet(sample, step), sample + 3)

        self.assertIs(policy.diffusion.unet, wrapped)
        self.assertFalse(wrapped._active)
        self.assertEqual(len(wrapped.steps), 3)

    def test_training_bypasses_cache(self) -> None:
        adapter = CachedDenoiser(AdditiveDenoiser(), auto_total_steps=3)
        adapter.train()

        sample = torch.tensor([1.0])
        torch.testing.assert_close(adapter(sample, 0), sample + 3)
        self.assertEqual(adapter.steps, [])


if __name__ == "__main__":
    unittest.main()
