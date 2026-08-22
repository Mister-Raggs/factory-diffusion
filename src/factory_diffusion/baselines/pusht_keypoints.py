"""Compatibility loader for the official PushT keypoint checkpoint.

The checkpoint predates LeRobot's current typed configuration format. Its U-Net
weights are compatible with LeRobot 0.4.4, but normalization buffers belong to
the legacy policy wrapper and are intentionally not used by the direct
denoising probe.
"""

from __future__ import annotations

from pathlib import Path

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

CHECKPOINT_REPO = "lerobot/diffusion_pusht_keypoints"
CHECKPOINT_REVISION = "58570fc39828d28efa5457aa297a52be27ac3a10"


def build_config(device: str = "cpu") -> DiffusionConfig:
    """Translate the checkpoint's legacy config to LeRobot 0.4.4."""

    return DiffusionConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
            "observation.environment_state": PolicyFeature(FeatureType.ENV, (16,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        normalization_mapping={
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        horizon=16,
        n_obs_steps=2,
        n_action_steps=8,
        noise_scheduler_type="DDIM",
        num_inference_steps=10,
        num_train_timesteps=100,
        device=device,
        use_amp=False,
    )


def load_policy(
    *,
    device: str = "cpu",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> DiffusionPolicy:
    """Load the pinned checkpoint using the translated typed configuration."""

    return DiffusionPolicy.from_pretrained(
        CHECKPOINT_REPO,
        config=build_config(device),
        revision=CHECKPOINT_REVISION,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
