"""Tiny affine correction on top of frozen previous-stage logits."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class TinyLogitCorrection(nn.Module):
    def __init__(self, previous_model: nn.Module, *, num_labels: int) -> None:
        super().__init__()
        self.previous_model = deepcopy(previous_model)
        for parameter in self.previous_model.parameters():
            parameter.requires_grad = False
        self.logit_scale = nn.Parameter(torch.ones(num_labels))
        self.logit_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            previous_logits = self.previous_model(embeddings)
        logits = (previous_logits * self.logit_scale.unsqueeze(0)) + self.logit_bias.unsqueeze(0)
        return {
            "previous_logits": previous_logits,
            "logits": logits,
        }
