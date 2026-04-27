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

from scripts.adapt_lora_mimic import (  # noqa: E402
    LABELS,
    LoRAConv2d,
    LoRALinear,
    StageFailure,
    build_lora_model_from_saved_state,
    build_model,
    check_leakage,
    inject_lora_modules,
    load_checkpoint,
    summarize_parameter_trainability,
    train_one_epoch,
    verify_lora_trainability,
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


def build_test_lora_model() -> tuple[torch.nn.Module, dict[str, object]]:
    model = build_model()
    target_module_names = inject_lora_modules(model, rank=4, alpha=4.0, dropout=0.0)
    summary = summarize_parameter_trainability(
        model=model,
        target_module_names=target_module_names,
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )
    verify_lora_trainability(summary)
    return model, summary


def run_single_training_step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)

    model, _ = build_test_lora_model()
    classifier_lora_before = model.classifier.lora_up.weight.detach().clone()
    denseblock4_lora_before = (
        model.features.denseblock4.denselayer1.conv2.lora_up.weight.detach().clone()
    )
    early_base_before = model.features.conv0.weight.detach().clone()

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

    classifier_lora_after = model.classifier.lora_up.weight.detach().clone()
    denseblock4_lora_after = (
        model.features.denseblock4.denselayer1.conv2.lora_up.weight.detach().clone()
    )
    early_base_after = model.features.conv0.weight.detach().clone()
    return (
        classifier_lora_before,
        classifier_lora_after,
        denseblock4_lora_before,
        denseblock4_lora_after,
        early_base_before,
        early_base_after,
    )


def test_model_loads_with_five_outputs_and_lora_injection() -> None:
    warnings: list[str] = []
    base_model, metadata = load_checkpoint(
        ROOT / "checkpoints" / "nih_2k_densenet121_best.pt",
        torch.device("cpu"),
        warnings,
    )
    target_module_names = inject_lora_modules(base_model, rank=4, alpha=4.0, dropout=0.0)

    batch = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = base_model(batch)

    assert logits.shape == (2, len(LABELS))
    assert metadata["label_names"] == LABELS
    assert warnings == []
    assert len(target_module_names) == 33
    assert isinstance(base_model.classifier, LoRALinear)
    assert isinstance(base_model.features.denseblock4.denselayer1.conv1, LoRAConv2d)


def test_only_lora_parameters_are_trainable() -> None:
    _, summary = build_test_lora_model()

    assert summary["unexpected_trainable_names"] == []
    assert summary["missing_trainable_names"] == []
    assert summary["target_module_count"] == 33
    assert all(".lora_" in name for name in summary["trainable_parameter_names"])
    assert all(".lora_" not in name for name in summary["frozen_parameter_names"] if name.endswith(".weight"))


def test_leakage_checker_catches_subject_overlap() -> None:
    support = pd.DataFrame([make_row("support", 7, 700, {})])
    val = pd.DataFrame([make_row("val", 7, 701, {})])
    test = pd.DataFrame([make_row("test", 9, 900, {})])

    checks, failures = check_leakage({"support": support, "val": val, "test": test})

    assert checks["subject_overlap"]["support_vs_val"]["count"] == 1
    assert any("subject_id leakage" in failure for failure in failures)


def test_one_training_step_changes_classifier_lora_weights() -> None:
    classifier_before, classifier_after, _, _, _, _ = run_single_training_step()

    assert not torch.allclose(classifier_before, classifier_after)


def test_one_training_step_changes_denseblock4_lora_weights() -> None:
    _, _, denseblock4_before, denseblock4_after, _, _ = run_single_training_step()

    assert not torch.allclose(denseblock4_before, denseblock4_after)


def test_one_training_step_does_not_change_frozen_early_backbone_weights() -> None:
    _, _, _, _, early_before, early_after = run_single_training_step()

    assert torch.allclose(early_before, early_after)


def test_trainability_check_fails_if_base_parameter_is_trainable() -> None:
    model, summary = build_test_lora_model()
    model.features.conv0.weight.requires_grad = True
    target_module_names = summary["target_module_names"]
    summary = summarize_parameter_trainability(
        model=model,
        target_module_names=target_module_names,
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )

    with pytest.raises(StageFailure, match="unexpectedly trainable"):
        verify_lora_trainability(summary)


def test_saved_lora_state_can_be_reloaded_into_same_wrapped_architecture() -> None:
    model, _ = build_test_lora_model()
    state_dict = model.state_dict()

    reloaded = build_lora_model_from_saved_state(
        state_dict=state_dict,
        rank=4,
        alpha=4.0,
        dropout=0.0,
        device=torch.device("cpu"),
    )

    batch = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = reloaded(batch)

    assert logits.shape == (2, len(LABELS))
