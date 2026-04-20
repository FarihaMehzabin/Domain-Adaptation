"""Label-wise disagreement-gated trust-region adaptation."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from paper_v1.models.prototype_memory import PrototypeMemoryModule
from paper_v1.models.residual_adapter import ResidualLogitAdapter


class _SharedLabelWiseGate(nn.Module):
    def __init__(self, *, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gate_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_gate = self.net(gate_inputs).squeeze(-1)
        gate = torch.sigmoid(raw_gate)
        return raw_gate, gate


def _labelwise_signals(
    previous_logits: torch.Tensor,
    retrieval: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    source_probabilities = torch.sigmoid(previous_logits)
    disagreement = torch.abs(source_probabilities - retrieval["prior_probabilities"])
    uncertainty = 1.0 - torch.abs((2.0 * source_probabilities) - 1.0)
    negative_support_margin = torch.relu(-retrieval["support_margin"])
    correction_score = disagreement + uncertainty + retrieval["matched_distance"] + negative_support_margin
    return {
        "source_probabilities": source_probabilities,
        "disagreement": disagreement,
        "uncertainty": uncertainty,
        "negative_support_margin": negative_support_margin,
        "correction_score": correction_score,
    }


class LabelWiseTrustRegionAdaptationModel(nn.Module):
    def __init__(
        self,
        previous_model: nn.Module,
        memory_module: PrototypeMemoryModule,
        *,
        feature_dim: int,
        num_labels: int,
        gate_hidden_dim: int = 16,
        bottleneck_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.previous_model = deepcopy(previous_model)
        for parameter in self.previous_model.parameters():
            parameter.requires_grad = False
        self.memory_module = memory_module
        self.num_labels = num_labels
        self.gate = _SharedLabelWiseGate(hidden_dim=gate_hidden_dim)
        self.adapter = ResidualLogitAdapter(feature_dim, num_labels, bottleneck_dim=bottleneck_dim, dropout=dropout)

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            previous_logits = self.previous_model(embeddings)
        retrieval = self.memory_module.retrieve(embeddings, previous_probabilities=torch.sigmoid(previous_logits))
        signals = _labelwise_signals(previous_logits, retrieval)
        prior_probabilities = retrieval["prior_probabilities"]
        gate_inputs = torch.stack(
            [
                signals["disagreement"],
                signals["uncertainty"],
                retrieval["matched_distance"],
                retrieval["support_margin"],
            ],
            dim=-1,
        )
        raw_gate, gate = self.gate(gate_inputs)
        residual = self.adapter(embeddings, previous_logits, prior_probabilities)
        logits = previous_logits + (gate * residual)
        return {
            "previous_logits": previous_logits,
            "source_probabilities": signals["source_probabilities"],
            "prior_probabilities": prior_probabilities,
            "retrieval_summary": retrieval["retrieval_summary"],
            "positive_support": retrieval["positive_support"],
            "negative_support": retrieval["negative_support"],
            "matched_support": retrieval["matched_support"],
            "opposing_support": retrieval["opposing_support"],
            "matched_distance": retrieval["matched_distance"],
            "support_margin": retrieval["support_margin"],
            "disagreement": signals["disagreement"],
            "uncertainty": signals["uncertainty"],
            "negative_support_margin": signals["negative_support_margin"],
            "correction_score": signals["correction_score"],
            "gate_inputs": gate_inputs,
            "residual": residual,
            "raw_gate": raw_gate,
            "gate": gate,
            "logits": logits,
        }


class TopKLabelWiseTrustRegionCorrectionModel(nn.Module):
    def __init__(
        self,
        previous_model: nn.Module,
        memory_module: PrototypeMemoryModule,
        *,
        feature_dim: int,
        num_labels: int,
        top_k: int,
        bottleneck_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.previous_model = deepcopy(previous_model)
        for parameter in self.previous_model.parameters():
            parameter.requires_grad = False
        self.memory_module = memory_module
        self.num_labels = num_labels
        self.top_k = int(top_k)
        self.adapter = ResidualLogitAdapter(feature_dim, num_labels, bottleneck_dim=bottleneck_dim, dropout=dropout)

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            previous_logits = self.previous_model(embeddings)
        retrieval = self.memory_module.retrieve(embeddings, previous_probabilities=torch.sigmoid(previous_logits))
        signals = _labelwise_signals(previous_logits, retrieval)
        residual = self.adapter(embeddings, previous_logits, retrieval["prior_probabilities"])
        selection_score = signals["correction_score"]
        top_k = max(1, min(self.top_k, self.num_labels))
        _, selected_indices = torch.topk(selection_score, k=top_k, dim=1)
        mask = torch.zeros_like(selection_score)
        mask.scatter_(1, selected_indices, 1.0)
        logits = previous_logits + (mask * residual)
        return {
            "previous_logits": previous_logits,
            "source_probabilities": signals["source_probabilities"],
            "prior_probabilities": retrieval["prior_probabilities"],
            "retrieval_summary": retrieval["retrieval_summary"],
            "positive_support": retrieval["positive_support"],
            "negative_support": retrieval["negative_support"],
            "matched_support": retrieval["matched_support"],
            "opposing_support": retrieval["opposing_support"],
            "matched_distance": retrieval["matched_distance"],
            "support_margin": retrieval["support_margin"],
            "disagreement": signals["disagreement"],
            "uncertainty": signals["uncertainty"],
            "negative_support_margin": signals["negative_support_margin"],
            "correction_score": selection_score,
            "residual": residual,
            "raw_gate": selection_score,
            "gate": mask,
            "selected_label_count": mask.sum(dim=1, keepdim=True),
            "logits": logits,
        }
