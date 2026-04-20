from __future__ import annotations

import unittest

import numpy as np
import torch

from paper_v1.models.full_model import ContinualAdaptationModel
from paper_v1.models.linear_head import LinearHead
from paper_v1.models.prototype_memory import PrototypeBank


class StageTransitionTest(unittest.TestCase):
    def test_frozen_previous_model_and_zero_residual_match_previous_logits(self) -> None:
        previous = LinearHead(feature_dim=4, num_labels=2)
        bank = PrototypeBank(
            prototype_vectors=np.zeros((1, 4), dtype=np.float32),
            soft_labels=np.zeros((1, 2), dtype=np.float32),
            cluster_counts=np.ones((1,), dtype=np.int64),
            source_domains=["d0"],
            label_indices=[0],
            states=["positive"],
            label_names=["a", "b"],
        )
        model = ContinualAdaptationModel(previous, bank.to_module(top_k=1, temperature=0.1), feature_dim=4, num_labels=2)
        for parameter in model.adapter.parameters():
            torch.nn.init.zeros_(parameter)
        embeddings = torch.randn(3, 4)
        outputs = model(embeddings)
        with torch.no_grad():
            previous_logits = previous(embeddings)
        self.assertTrue(torch.allclose(outputs["logits"], previous_logits))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.previous_model.parameters()))


if __name__ == "__main__":
    unittest.main()
