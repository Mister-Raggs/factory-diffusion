from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("lerobot"), "requires the lerobot extra")
class PushTBaselineTest(unittest.TestCase):
    def test_legacy_checkpoint_translation_matches_action_unet(self) -> None:
        from factory_diffusion.baselines.pusht_keypoints import (
            CHECKPOINT_REVISION,
            build_config,
        )

        config = build_config("cpu")

        self.assertEqual(CHECKPOINT_REVISION, "58570fc39828d28efa5457aa297a52be27ac3a10")
        self.assertEqual(config.num_inference_steps, 10)
        self.assertEqual(config.noise_scheduler_type, "DDIM")
        self.assertEqual(config.horizon, 16)
        self.assertEqual(config.action_feature.shape, (2,))
        self.assertEqual(config.env_state_feature.shape, (16,))
        self.assertEqual(config.image_features, {})


if __name__ == "__main__":
    unittest.main()
