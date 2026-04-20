"""Calibration helpers."""

from __future__ import annotations

import numpy as np

from paper_v1.evaluation.metrics import binary_brier_score, expected_calibration_error


def summarize_calibration(y_true: np.ndarray, y_prob: np.ndarray, *, num_bins: int = 10) -> dict[str, float]:
    return {
        "brier_score": binary_brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, num_bins=num_bins),
    }
