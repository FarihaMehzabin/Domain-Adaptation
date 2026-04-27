import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/workspace")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adapt_head_only_mimic import (  # noqa: E402
    LABELS,
    build_model,
    check_leakage,
    freeze_backbone,
    load_checkpoint,
    train_one_epoch,
    verify_frozen_backbone,
)


def make_row(
    row_id: str,
    subject_id: int,
    study_id: int,
    label_values: dict[str, int],
) -> dict[str, object]:
    return {
        "dicom_id": f"dicom-{row_id}",
        "subject_id": subject_id,
        "study_id": study_id,
        "abs_path": f"/tmp/image-{row_id}.jpg",
        **{label: int(label_values.get(label, 0)) for label in LABELS},
    }


def run_single_training_step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)

    model = build_model()
    summary = freeze_backbone(model)
    verify_frozen_backbone(summary)

    classifier_before = model.classifier.weight.detach().clone()
    backbone_before = model.features.conv0.weight.detach().clone()

    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

    classifier_after = model.classifier.weight.detach().clone()
    backbone_after = model.features.conv0.weight.detach().clone()
    return classifier_before, classifier_after, backbone_before, backbone_after


def test_model_loads_with_five_outputs() -> None:
    warnings: list[str] = []
    model, metadata = load_checkpoint(
        ROOT / "checkpoints" / "nih_2k_densenet121_best.pt",
        torch.device("cpu"),
        warnings,
    )

    batch = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (2, len(LABELS))
    assert metadata["label_names"] == LABELS
    assert warnings == []


def test_freezing_leaves_only_classifier_parameters_trainable() -> None:
    model = build_model()

    summary = freeze_backbone(model)
    verify_frozen_backbone(summary)

    assert summary["trainable_parameter_names"] == ["classifier.weight", "classifier.bias"]
    assert summary["backbone_trainable_names"] == []
    assert summary["classifier_frozen_names"] == []


def test_leakage_checker_catches_subject_overlap() -> None:
    support = pd.DataFrame([make_row("support", 7, 700, {})])
    val = pd.DataFrame([make_row("val", 7, 701, {})])
    test = pd.DataFrame([make_row("test", 9, 900, {})])

    checks, failures = check_leakage({"support": support, "val": val, "test": test})

    assert checks["subject_overlap"]["support_vs_val"]["count"] == 1
    assert any("subject_id leakage" in failure for failure in failures)


def test_one_training_step_changes_classifier_weights() -> None:
    classifier_before, classifier_after, _, _ = run_single_training_step()

    assert not torch.allclose(classifier_before, classifier_after)


def test_one_training_step_keeps_frozen_backbone_weights_unchanged() -> None:
    _, _, backbone_before, backbone_after = run_single_training_step()

    assert torch.allclose(backbone_before, backbone_after)
