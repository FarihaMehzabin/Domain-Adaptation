import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/workspace")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_nih_on_mimic import (  # noqa: E402
    LABELS,
    build_model,
    compute_binary_metrics,
    save_predictions_csv,
)


def test_model_forward_shape() -> None:
    model = build_model()
    batch = torch.randn(3, 3, 224, 224)
    logits = model(batch)
    assert logits.shape == (3, len(LABELS))


def test_metric_function_handles_normal_labels() -> None:
    targets = np.array(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
        ],
        dtype=np.float32,
    )
    probabilities = np.array(
        [
            [0.9, 0.2, 0.8, 0.1, 0.7],
            [0.2, 0.8, 0.3, 0.9, 0.4],
            [0.7, 0.7, 0.2, 0.3, 0.8],
            [0.1, 0.4, 0.9, 0.8, 0.2],
        ],
        dtype=np.float32,
    )

    metrics = compute_binary_metrics(targets, probabilities)

    assert metrics["macro_auroc"] is not None
    assert metrics["macro_auprc"] is not None
    assert metrics["defined_auroc_labels"] == len(LABELS)
    assert metrics["defined_auprc_labels"] == len(LABELS)


def test_metric_function_handles_label_with_no_positives() -> None:
    targets = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
        ],
        dtype=np.float32,
    )
    probabilities = np.array(
        [
            [0.9, 0.2, 0.8, 0.1, 0.3],
            [0.2, 0.8, 0.3, 0.9, 0.4],
            [0.7, 0.7, 0.2, 0.3, 0.2],
            [0.1, 0.4, 0.9, 0.8, 0.1],
        ],
        dtype=np.float32,
    )

    metrics = compute_binary_metrics(targets, probabilities)

    effusion = metrics["per_label"]["Effusion"]
    assert effusion["auroc"] is None
    assert effusion["auprc"] is None
    assert effusion["auroc_defined"] is False
    assert effusion["auprc_defined"] is False
    assert "reason" in effusion


def test_prediction_csv_has_expected_probability_columns(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        [
            {
                "resolved_path": "/tmp/example-a.jpg",
                "abs_path": "/tmp/example-a.jpg",
                "subject_id": 101,
                "study_id": 1001,
                "dicom_id": "dicom-a",
                **{label: 0 for label in LABELS},
            },
            {
                "resolved_path": "/tmp/example-b.jpg",
                "abs_path": "/tmp/example-b.jpg",
                "subject_id": 102,
                "study_id": 1002,
                "dicom_id": "dicom-b",
                **{label: 1 for label in LABELS},
            },
        ]
    )
    probabilities = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9, 0.95],
        ],
        dtype=np.float32,
    )

    output_path = tmp_path / "predictions.csv"
    save_predictions_csv(dataframe, probabilities, output_path, path_column="abs_path")
    saved = pd.read_csv(output_path)

    expected_probability_columns = {f"pred_{label}" for label in LABELS}
    assert expected_probability_columns.issubset(set(saved.columns))
