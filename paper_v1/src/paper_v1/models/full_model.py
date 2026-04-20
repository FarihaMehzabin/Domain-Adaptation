"""Main continual adaptation model."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from paper_v1.models.gating import NoveltyGate
from paper_v1.models.prototype_memory import PrototypeMemoryModule
from paper_v1.models.residual_adapter import ResidualLogitAdapter


class ContinualAdaptationModel(nn.Module):
    def __init__(
        self,
        previous_model: nn.Module,
        memory_module: PrototypeMemoryModule,
        *,
        feature_dim: int,
        num_labels: int,
        gate_hidden_dim: int = 64,
        bottleneck_dim: int = 128,
        dropout: float = 0.1,
        old_like_similarity_threshold: float | None = None,
        old_like_gate_cap: float | None = None,
    ) -> None:
        super().__init__()
        self.previous_model = deepcopy(previous_model)
        for parameter in self.previous_model.parameters():
            parameter.requires_grad = False
        self.memory_module = memory_module
        self.gate = NoveltyGate(feature_dim, num_labels, hidden_dim=gate_hidden_dim)
        self.adapter = ResidualLogitAdapter(feature_dim, num_labels, bottleneck_dim=bottleneck_dim, dropout=dropout)
        self.old_like_similarity_threshold = old_like_similarity_threshold
        self.old_like_gate_cap = old_like_gate_cap

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            previous_logits = self.previous_model(embeddings)
        prior_probabilities, retrieval_summary = self.memory_module(embeddings)
        residual = self.adapter(embeddings, previous_logits, prior_probabilities)
        raw_gate = self.gate(embeddings, previous_logits, prior_probabilities, retrieval_summary)
        gate = raw_gate
        old_like_mask = torch.zeros_like(gate)
        if self.old_like_similarity_threshold is not None and self.old_like_gate_cap is not None:
            similarity = retrieval_summary[:, :1]
            old_like_mask = (similarity >= self.old_like_similarity_threshold).to(gate.dtype)
            gate_cap = torch.full_like(gate, float(self.old_like_gate_cap))
            gate = torch.where(old_like_mask > 0, torch.minimum(gate, gate_cap), gate)
        logits = previous_logits + (gate * residual)
        return {
            "previous_logits": previous_logits,
            "prior_probabilities": prior_probabilities,
            "retrieval_summary": retrieval_summary,
            "residual": residual,
            "raw_gate": raw_gate,
            "gate": gate,
            "old_like_mask": old_like_mask,
            "logits": logits,
        }
