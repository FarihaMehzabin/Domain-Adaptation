from __future__ import annotations

import unittest

from paper_v1.evaluation.forgetting import compute_forgetting


class ForgettingTest(unittest.TestCase):
    def test_forgetting_matches_expected_drop(self) -> None:
        history = [
            {"stage_name": "s0", "domain_metrics": {"d0": {"macro_auroc": 0.8}}},
            {"stage_name": "s1", "domain_metrics": {"d0": {"macro_auroc": 0.7}, "d1": {"macro_auroc": 0.9}}},
        ]
        summary = compute_forgetting(history)
        self.assertAlmostEqual(summary["final_average_forgetting"], 0.1)


if __name__ == "__main__":
    unittest.main()
