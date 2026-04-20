"""Small residual logit corrector."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualLogitAdapter(nn.Module):
    def __init__(self, feature_dim: int, num_labels: int, bottleneck_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        input_dim = feature_dim + (2 * num_labels)
        self.net = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, num_labels),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        previous_logits: torch.Tensor,
        prior_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([embeddings, previous_logits, prior_probabilities], dim=1)
        return self.net(features)
