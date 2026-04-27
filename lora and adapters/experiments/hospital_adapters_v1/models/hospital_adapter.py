#!/usr/bin/env python3
"""Hospital-specific residual feature adapters for DenseNet-121."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hospital_adapters_v1.common import StageFailure  # noqa: E402


class ResidualFeatureAdapter(nn.Module):
    """Residual bottleneck adapter initialized as an identity map."""

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 128,
        dropout: float = 0.1,
        scale_init: float = 1e-3,
        use_vector_scale: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.dropout_p = float(dropout)
        self.use_vector_scale = bool(use_vector_scale)

        self.norm = nn.LayerNorm(self.input_dim)
        self.down = nn.Linear(self.input_dim, self.bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.up = nn.Linear(self.bottleneck_dim, self.input_dim)

        scale_shape = (self.input_dim,) if self.use_vector_scale else ()
        self.scale = nn.Parameter(torch.full(scale_shape, float(scale_init), dtype=torch.float32))

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.dropout(self.activation(self.down(self.norm(features)))))
        return features + self.scale * residual


def get_classifier_module(base_model: nn.Module) -> nn.Linear:
    classifier = getattr(base_model, "classifier", None)
    if not isinstance(classifier, nn.Linear):
        raise StageFailure(
            "Hospital adapter experiment expects a DenseNet-style nn.Linear classifier head on the base model."
        )
    return classifier


def extract_pooled_features(base_model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(base_model, "forward_features") and callable(getattr(base_model, "forward_features")):
        features = base_model.forward_features(images)
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            return torch.flatten(features, 1)
        if features.ndim == 2:
            return features
        raise StageFailure(
            f"Unsupported forward_features output shape for pooled extraction: {tuple(features.shape)}"
        )

    features_module = getattr(base_model, "features", None)
    if features_module is None:
        raise StageFailure("Base model does not expose .features, so pooled feature extraction is unavailable.")

    features = features_module(images)
    features = F.relu(features, inplace=False)
    features = F.adaptive_avg_pool2d(features, (1, 1))
    return torch.flatten(features, 1)


class HospitalAdapterClassifier(nn.Module):
    """DenseNet classifier augmented with a hospital-specific feature adapter."""

    def __init__(
        self,
        base_model: nn.Module,
        adapter: ResidualFeatureAdapter,
        hospital_bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        classifier = self.get_classifier()
        bias_shape = (classifier.out_features,)
        if hospital_bias is None:
            bias_tensor = torch.zeros(bias_shape, dtype=classifier.weight.dtype)
        else:
            bias_tensor = hospital_bias.detach().clone().to(dtype=classifier.weight.dtype)
        self.hospital_bias = nn.Parameter(bias_tensor)
        self.pooled_feature_dim = int(classifier.in_features)
        self.num_labels = int(classifier.out_features)

    def get_classifier(self) -> nn.Linear:
        return get_classifier_module(self.base_model)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return extract_pooled_features(self.base_model, images)

    def forward_with_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self.extract_features(images)
        adapted = self.adapter(pooled)
        logits = self.get_classifier()(adapted) + self.hospital_bias
        return logits, pooled, adapted

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_with_features(images)
        return logits


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True


def configure_trainable_parameters(
    model: HospitalAdapterClassifier,
    *,
    train_classifier_head: bool = False,
    train_hospital_bias: bool = True,
) -> dict[str, Any]:
    freeze_module(model.base_model)
    unfreeze_module(model.adapter)
    model.hospital_bias.requires_grad = bool(train_hospital_bias)
    if train_classifier_head:
        unfreeze_module(model.get_classifier())

    total_parameters = 0
    trainable_parameters = 0
    trainable_parameter_names: list[str] = []
    frozen_classifier_parameter_names: list[str] = []
    unexpectedly_trainable_base_names: list[str] = []

    for name, parameter in model.named_parameters():
        total_parameters += parameter.numel()
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()
            trainable_parameter_names.append(name)
            if name.startswith("base_model.") and not name.startswith("base_model.classifier."):
                unexpectedly_trainable_base_names.append(name)
        elif name.startswith("base_model.classifier."):
            frozen_classifier_parameter_names.append(name)

    return {
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "trainable_parameter_names": trainable_parameter_names,
        "frozen_classifier_parameter_names": frozen_classifier_parameter_names,
        "unexpectedly_trainable_base_names": unexpectedly_trainable_base_names,
    }


def set_adapter_train_mode(model: HospitalAdapterClassifier, train_classifier_head: bool) -> None:
    model.eval()
    model.adapter.train()
    if train_classifier_head:
        model.get_classifier().train()


def build_adapter_checkpoint_payload(
    model: HospitalAdapterClassifier,
    *,
    epoch: int,
    target_hospital: str,
    source_hospital: str | None,
    label_names: list[str],
    pooled_feature_dim: int,
    adapter_bottleneck: int,
    adapter_dropout: float,
    adapter_scale_init: float,
    base_checkpoint_path: str,
    classifier_head_trained: bool,
    best_metric_name: str,
    best_metric_value: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "epoch": int(epoch),
        "target_hospital": target_hospital,
        "source_hospital": source_hospital,
        "label_names": list(label_names),
        "pooled_feature_dim": int(pooled_feature_dim),
        "adapter_bottleneck": int(adapter_bottleneck),
        "adapter_dropout": float(adapter_dropout),
        "adapter_scale_init": float(adapter_scale_init),
        "base_checkpoint_path": str(base_checkpoint_path),
        "classifier_head_trained": bool(classifier_head_trained),
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "adapter_state_dict": model.adapter.state_dict(),
        "hospital_bias": model.hospital_bias.detach().cpu(),
        "classifier_state_dict": model.get_classifier().state_dict() if classifier_head_trained else None,
    }
    if extra:
        payload.update(extra)
    return payload


def save_adapter_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_adapter_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise StageFailure(f"Missing adapter checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise StageFailure(f"Could not load adapter checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise StageFailure(f"Adapter checkpoint {path} must be a dictionary payload.")

    required_keys = [
        "adapter_state_dict",
        "hospital_bias",
        "target_hospital",
        "label_names",
        "pooled_feature_dim",
        "adapter_bottleneck",
        "adapter_dropout",
        "base_checkpoint_path",
        "classifier_head_trained",
    ]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise StageFailure(f"Adapter checkpoint {path} is missing required keys: {missing_keys}")
    return checkpoint


def apply_adapter_checkpoint(model: HospitalAdapterClassifier, checkpoint: dict[str, Any]) -> None:
    model.adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)

    hospital_bias = checkpoint["hospital_bias"]
    if isinstance(hospital_bias, torch.Tensor):
        bias_tensor = hospital_bias.to(dtype=model.hospital_bias.dtype, device=model.hospital_bias.device)
    else:
        bias_tensor = torch.tensor(hospital_bias, dtype=model.hospital_bias.dtype, device=model.hospital_bias.device)
    if tuple(bias_tensor.shape) != tuple(model.hospital_bias.shape):
        raise StageFailure(
            "Adapter checkpoint hospital bias shape does not match model output shape: "
            f"{tuple(bias_tensor.shape)} vs {tuple(model.hospital_bias.shape)}"
        )
    with torch.no_grad():
        model.hospital_bias.copy_(bias_tensor)

    classifier_state = checkpoint.get("classifier_state_dict")
    if checkpoint.get("classifier_head_trained"):
        if classifier_state is None:
            raise StageFailure("Adapter checkpoint says the classifier head was trained but has no classifier state.")
        model.get_classifier().load_state_dict(classifier_state, strict=True)
