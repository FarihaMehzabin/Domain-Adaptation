import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/workspace")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adapt_lastblock_mimic import (  # noqa: E402
    LABELS,
    StageFailure,
    build_model,
    check_leakage,
    enable_lastblock_finetune,
    load_checkpoint,
    summarize_parameter_trainability,
    train_one_epoch,
    verify_lastblock_trainability,
)


TRAINABLE_PREFIXES = ("features.denseblock4.", "features.norm5.", "classifier.")


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


def run_single_training_step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(13)

    model = build_model()
    summary = enable_lastblock_finetune(model)
    verify_lastblock_trainability(summary)

    classifier_before = model.classifier.weight.detach().clone()
    lastblock_before = model.features.denseblock4.denselayer1.conv2.weight.detach().clone()
    early_before = model.features.conv0.weight.detach().clone()

    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

    classifier_after = model.classifier.weight.detach().clone()
    lastblock_after = model.features.denseblock4.denselayer1.conv2.weight.detach().clone()
    early_after = model.features.conv0.weight.detach().clone()
    return (
        classifier_before,
        classifier_after,
        lastblock_before,
        lastblock_after,
        early_before,
        early_after,
    )


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


def test_lastblock_mask_leaves_only_denseblock4_norm5_and_classifier_trainable() -> None:
    model = build_model()

    summary = enable_lastblock_finetune(model)
    verify_lastblock_trainability(summary)

    assert summary["unexpected_trainable_names"] == []
    assert summary["unexpected_frozen_names"] == []
    assert all(name.startswith(TRAINABLE_PREFIXES) for name in summary["trainable_parameter_names"])
    assert "classifier.weight" in summary["trainable_parameter_names"]
    assert "classifier.bias" in summary["trainable_parameter_names"]
    assert any(name.startswith("features.denseblock4.") for name in summary["trainable_parameter_names"])
    assert any(name.startswith("features.norm5.") for name in summary["trainable_parameter_names"])
    assert "features.conv0.weight" in summary["frozen_parameter_names"]
    assert any(name.startswith("features.denseblock3.") for name in summary["frozen_parameter_names"])


def test_leakage_checker_catches_subject_overlap() -> None:
    support = pd.DataFrame([make_row("support", 7, 700, {})])
    val = pd.DataFrame([make_row("val", 7, 701, {})])
    test = pd.DataFrame([make_row("test", 9, 900, {})])

    checks, failures = check_leakage({"support": support, "val": val, "test": test})

    assert checks["subject_overlap"]["support_vs_val"]["count"] == 1
    assert any("subject_id leakage" in failure for failure in failures)


def test_one_training_step_changes_classifier_weights() -> None:
    classifier_before, classifier_after, _, _, _, _ = run_single_training_step()

    assert not torch.allclose(classifier_before, classifier_after)


def test_one_training_step_changes_denseblock4_weights() -> None:
    _, _, lastblock_before, lastblock_after, _, _ = run_single_training_step()

    assert not torch.allclose(lastblock_before, lastblock_after)


def test_one_training_step_does_not_change_conv0_weights() -> None:
    _, _, _, _, early_before, early_after = run_single_training_step()

    assert torch.allclose(early_before, early_after)


def test_trainability_check_fails_if_early_backbone_parameter_is_trainable() -> None:
    model = build_model()
    enable_lastblock_finetune(model)
    model.features.conv0.weight.requires_grad = True

    summary = summarize_parameter_trainability(model)

    with pytest.raises(StageFailure, match="unexpectedly trainable"):
        verify_lastblock_trainability(summary)
