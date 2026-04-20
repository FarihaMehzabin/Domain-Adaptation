"""Novelty and oldness gate."""

from __future__ import annotations

import torch
import torch.nn as nn


class NoveltyGate(nn.Module):
    def __init__(self, feature_dim: int, num_labels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        input_dim = feature_dim + (2 * num_labels) + 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        previous_logits: torch.Tensor,
        prior_probabilities: torch.Tensor,
        retrieval_summary: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [embeddings, previous_logits, prior_probabilities, retrieval_summary],
            dim=1,
        )
        return torch.sigmoid(self.net(features))
