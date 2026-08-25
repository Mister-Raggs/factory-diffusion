from __future__ import annotations

import unittest

from factory_diffusion.rollout_sensitivity import summarize_sensitivity


def row(
    *,
    budget: int,
    method: str,
    query: int,
    keypoint_error: float,
    contact: bool,
) -> dict:
    return {
        "seed": 0,
        "query_index": query,
        "budget": budget,
        "method": method,
        "teacher_contact": contact,
        "action_rmse_pixels": keypoint_error * 2,
        "keypoint_rmse_pixels": keypoint_error,
        "agent_rmse_pixels": keypoint_error / 2,
        "absolute_coverage_error": keypoint_error / 10,
        "coverage_difference": -keypoint_error / 10,
    }


class RolloutSensitivityTest(unittest.TestCase):
    def test_summary_pairs_snapshots_and_checks_expected_signs(self) -> None:
        rows = []
        for query, contact in ((0, False), (1, True)):
            rows.extend(
                [
                    row(
                        budget=3,
                        method="standard",
                        query=query,
                        keypoint_error=2.0,
                        contact=contact,
                    ),
                    row(
                        budget=3,
                        method="optimized",
                        query=query,
                        keypoint_error=1.0,
                        contact=contact,
                    ),
                    row(
                        budget=5,
                        method="standard",
                        query=query,
                        keypoint_error=1.0,
                        contact=contact,
                    ),
                    row(
                        budget=5,
                        method="optimized",
                        query=query,
                        keypoint_error=3.0,
                        contact=contact,
                    ),
                ]
            )

        summary = summarize_sensitivity(rows)

        self.assertTrue(summary["diagnostic_alignment"])
        self.assertEqual(summary["per_budget"][0]["paired_snapshots"], 2)
        self.assertEqual(summary["per_budget"][0]["contact"]["paired_snapshots"], 1)
        self.assertEqual(
            summary["per_budget"][0]["all"]["keypoint_rmse_pixels"][
                "optimized_minus_standard_mean"
            ],
            -1.0,
        )
        self.assertEqual(
            summary["per_budget"][0]["all"]["keypoint_rmse_pixels"]["seed_balanced_difference"],
            -1.0,
        )

    def test_empty_rows_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            summarize_sensitivity([])


if __name__ == "__main__":
    unittest.main()
