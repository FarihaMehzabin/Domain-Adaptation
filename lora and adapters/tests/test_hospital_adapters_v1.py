import sys
from pathlib import Path

import torch

ROOT = Path("/workspace")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hospital_adapters_v1.common import DEFAULT_LABELS, build_densenet121_model, load_base_checkpoint  # noqa: E402
from experiments.hospital_adapters_v1.models.hospital_adapter import (  # noqa: E402
    HospitalAdapterClassifier,
    ResidualFeatureAdapter,
    apply_adapter_checkpoint,
    build_adapter_checkpoint_payload,
    configure_trainable_parameters,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)


def build_identity_model() -> HospitalAdapterClassifier:
    base_model = build_densenet121_model(num_labels=len(DEFAULT_LABELS))
    adapter = ResidualFeatureAdapter(
        input_dim=base_model.classifier.in_features,
        bottleneck_dim=128,
        dropout=0.1,
        scale_init=1e-3,
    )
    return HospitalAdapterClassifier(base_model=base_model, adapter=adapter)


def test_identity_adapter_matches_base_model_logits() -> None:
    torch.manual_seed(7)
    model = build_identity_model()
    images = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        base_logits = model.base_model(images)
        adapter_logits = model(images)

    max_diff = float((base_logits - adapter_logits).abs().max().item())
    assert max_diff < 1e-7


def test_default_trainable_parameters_are_only_adapter_and_bias() -> None:
    model = build_identity_model()
    summary = configure_trainable_parameters(
        model,
        train_classifier_head=False,
        train_hospital_bias=True,
    )

    assert summary["unexpectedly_trainable_base_names"] == []
    assert "hospital_bias" in summary["trainable_parameter_names"]
    assert "base_model.classifier.weight" not in summary["trainable_parameter_names"]
    assert all(
        name == "hospital_bias" or name.startswith("adapter.")
        for name in summary["trainable_parameter_names"]
    )


def test_classifier_head_can_be_enabled_explicitly() -> None:
    model = build_identity_model()
    summary = configure_trainable_parameters(
        model,
        train_classifier_head=True,
        train_hospital_bias=True,
    )

    assert "base_model.classifier.weight" in summary["trainable_parameter_names"]
    assert "base_model.classifier.bias" in summary["trainable_parameter_names"]


def test_adapter_checkpoint_round_trip_preserves_outputs(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = build_identity_model()
    configure_trainable_parameters(model, train_classifier_head=True, train_hospital_bias=True)

    with torch.no_grad():
        model.adapter.up.weight.fill_(0.01)
        model.adapter.up.bias.fill_(0.02)
        model.hospital_bias.fill_(0.03)
        model.base_model.classifier.weight.fill_(0.04)
        model.base_model.classifier.bias.fill_(0.05)

    payload = build_adapter_checkpoint_payload(
        model,
        epoch=3,
        target_hospital="mimic",
        source_hospital="nih",
        label_names=list(DEFAULT_LABELS),
        pooled_feature_dim=model.pooled_feature_dim,
        adapter_bottleneck=model.adapter.bottleneck_dim,
        adapter_dropout=model.adapter.dropout_p,
        adapter_scale_init=1e-3,
        base_checkpoint_path="/workspace/checkpoints/nih_2k_densenet121_best.pt",
        classifier_head_trained=True,
        best_metric_name="target_val_macro_auroc",
        best_metric_value=0.8,
    )
    checkpoint_path = tmp_path / "adapter.pt"
    save_adapter_checkpoint(payload, checkpoint_path)

    reloaded = build_identity_model()
    reloaded.base_model.load_state_dict(model.base_model.state_dict(), strict=True)
    loaded_payload = load_adapter_checkpoint(checkpoint_path, torch.device("cpu"))
    apply_adapter_checkpoint(reloaded, loaded_payload)

    images = torch.randn(2, 3, 64, 64)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        original_logits = model(images)
        reloaded_logits = reloaded(images)

    assert torch.allclose(original_logits, reloaded_logits)


def test_real_base_checkpoint_loads_with_expected_label_order() -> None:
    model, metadata, label_names = load_base_checkpoint(
        ROOT / "checkpoints" / "nih_2k_densenet121_best.pt",
        torch.device("cpu"),
    )

    batch = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (2, len(DEFAULT_LABELS))
    assert label_names == DEFAULT_LABELS
    assert metadata["label_names"] == DEFAULT_LABELS
