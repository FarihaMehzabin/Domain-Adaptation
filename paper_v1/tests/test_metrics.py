from __future__ import annotations

import unittest

import numpy as np

from paper_v1.evaluation.metrics import compute_multilabel_metrics


class MetricsTest(unittest.TestCase):
    def test_perfect_predictions_score_perfectly(self) -> None:
        y_true = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
        y_prob = y_true.copy()
        metrics = compute_multilabel_metrics(y_true, y_prob)
        self.assertAlmostEqual(metrics["macro_auroc"], 1.0)
        self.assertAlmostEqual(metrics["macro_auprc"], 1.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
