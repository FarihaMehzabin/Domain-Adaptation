"""Loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if mask is not None:
        losses = losses * mask
        normalizer = torch.clamp(mask.sum(), min=1.0)
        return losses.sum() / normalizer
    return losses.mean()


def sigmoid_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    student = torch.sigmoid(student_logits / temperature)
    teacher = torch.sigmoid(teacher_logits / temperature)
    return F.mse_loss(student, teacher)


def gate_sparsity_loss(gate: torch.Tensor) -> torch.Tensor:
    return gate.mean()


def old_prototype_margin_preservation_loss(
    current_logits: torch.Tensor,
    source_logits: torch.Tensor,
    label_indices: torch.Tensor,
    state_signs: torch.Tensor,
    *,
    slack: float = 0.25,
) -> torch.Tensor:
    if current_logits.numel() == 0:
        return current_logits.new_zeros(())
    gathered_current = current_logits.gather(1, label_indices.view(-1, 1)).squeeze(1)
    gathered_source = source_logits.gather(1, label_indices.view(-1, 1)).squeeze(1)
    signed_drop = state_signs * (gathered_source - gathered_current)
    return torch.relu(signed_drop - slack).mean()


def trust_region_weights(
    matched_support: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    support_threshold: float = 0.7,
) -> torch.Tensor:
    return torch.relu(matched_support - support_threshold) * (1.0 - uncertainty)


def residual_trust_region_loss(
    residual: torch.Tensor,
    matched_support: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    support_threshold: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = trust_region_weights(
        matched_support,
        uncertainty,
        support_threshold=support_threshold,
    )
    return torch.mean(weights * (residual**2)), weights
