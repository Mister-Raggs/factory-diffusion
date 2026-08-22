from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from factory_diffusion.analysis import replay_baseline_path, sweep_baseline_path
from factory_diffusion.cache import AdaptiveCacheConfig
from factory_diffusion.trace import DenoisingTrace, TraceDenoiser


class AdditiveDenoiser(nn.Module):
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep, global_cond
        return sample + 2


class TraceAnalysisTest(unittest.TestCase):
    def _trace(self) -> DenoisingTrace:
        recorder = TraceDenoiser(AdditiveDenoiser(), auto_total_steps=4)
        for step in range(4):
            sample = torch.tensor([[float(step)]])
            torch.testing.assert_close(recorder(sample, step), sample + 2)
        return recorder.trace()

    def test_diagnostics_measure_stable_transformation(self) -> None:
        diagnostics = self._trace().diagnostics()

        self.assertIsNone(diagnostics[0].transformation_drift_l1)
        for row in diagnostics[1:]:
            self.assertIsNotNone(row.sensitivity)
            self.assertIsNotNone(row.transformation_drift_l1)
            self.assertIsNotNone(row.relative_transformation_drift)
            self.assertAlmostEqual(row.sensitivity, 1.0)
            self.assertAlmostEqual(row.transformation_drift_l1, 0.0)
            self.assertAlmostEqual(row.relative_transformation_drift, 0.0)

    def test_offline_replay_reports_skips_and_exact_outputs(self) -> None:
        replay = replay_baseline_path(
            self._trace(),
            AdaptiveCacheConfig(threshold=1.0, warmup_steps=2, force_compute_last=1),
        )

        self.assertEqual(replay.skipped_indices, (2,))
        self.assertEqual(replay.model_output_mse, 0.0)
        self.assertEqual(replay.model_output_max_error, 0.0)

    def test_sweep_requires_thresholds(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one threshold"):
            sweep_baseline_path(self._trace(), [])

    def test_trace_save_uses_tensor_and_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._trace().save(directory)
            target = Path(directory)
            tensors = torch.load(target / "trace_tensors.pt", weights_only=True)
            metadata = json.loads((target / "trace.json").read_text())

        self.assertIn("input_0000", tensors)
        self.assertEqual(len(metadata["diagnostics"]), 4)


if __name__ == "__main__":
    unittest.main()
