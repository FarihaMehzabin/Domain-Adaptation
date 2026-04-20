"""Simple frozen-embedding multilabel head."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, feature_dim: int, num_labels: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_labels = num_labels
        self.linear = nn.Linear(feature_dim, num_labels)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings)
