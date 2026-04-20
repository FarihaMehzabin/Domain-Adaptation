from __future__ import annotations

import unittest

import numpy as np
import torch

from paper_v1.models.full_model import ContinualAdaptationModel
from paper_v1.models.labelwise_trust_region import (
    LabelWiseTrustRegionAdaptationModel,
    TopKLabelWiseTrustRegionCorrectionModel,
)
from paper_v1.models.linear_head import LinearHead
from paper_v1.models.prototype_memory import PrototypeBank
from paper_v1.models.tiny_logit_correction import TinyLogitCorrection


class PostmortemModelsTest(unittest.TestCase):
    def _bank(self) -> PrototypeBank:
        return PrototypeBank(
            prototype_vectors=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            soft_labels=np.asarray([[1.0, 0.0]], dtype=np.float32),
            cluster_counts=np.ones((1,), dtype=np.int64),
            source_domains=["d0_nih"],
            label_indices=[0],
            states=["positive"],
            label_names=["a", "b"],
        )

    def test_gate_clipping_caps_old_like_samples(self) -> None:
        previous = LinearHead(feature_dim=4, num_labels=2)
        model = ContinualAdaptationModel(
            previous,
            self._bank().to_module(top_k=1, temperature=0.1),
            feature_dim=4,
            num_labels=2,
            old_like_similarity_threshold=0.5,
            old_like_gate_cap=0.1,
        )
        for parameter in model.gate.parameters():
            torch.nn.init.zeros_(parameter)
        final_linear = model.gate.net[2]
        torch.nn.init.constant_(final_linear.bias, 10.0)
        embeddings = torch.asarray([[10.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        outputs = model(embeddings)
        self.assertGreater(float(outputs["raw_gate"][0, 0]), 0.9)
        self.assertLessEqual(float(outputs["gate"][0, 0]), 0.1 + 1.0e-6)
        self.assertEqual(float(outputs["old_like_mask"][0, 0]), 1.0)

    def test_tiny_logit_correction_starts_at_identity(self) -> None:
        previous = LinearHead(feature_dim=4, num_labels=2)
        model = TinyLogitCorrection(previous, num_labels=2)
        embeddings = torch.randn(3, 4)
        outputs = model(embeddings)
        with torch.no_grad():
            previous_logits = previous(embeddings)
        self.assertTrue(torch.allclose(outputs["logits"], previous_logits))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.previous_model.parameters()))

    def test_labelwise_model_outputs_per_label_gate_and_preserves_previous_when_residual_zero(self) -> None:
        previous = LinearHead(feature_dim=4, num_labels=2)
        model = LabelWiseTrustRegionAdaptationModel(
            previous,
            self._bank().to_module(top_k=1, temperature=0.1),
            feature_dim=4,
            num_labels=2,
        )
        for parameter in model.adapter.parameters():
            torch.nn.init.zeros_(parameter)
        for parameter in model.gate.parameters():
            torch.nn.init.zeros_(parameter)
        embeddings = torch.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        outputs = model(embeddings)
        with torch.no_grad():
            previous_logits = previous(embeddings)
        self.assertEqual(outputs["gate"].shape, (2, 2))
        self.assertEqual(outputs["raw_gate"].shape, (2, 2))
        self.assertTrue(torch.allclose(outputs["logits"], previous_logits))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.previous_model.parameters()))

    def test_topk_model_emits_binary_mask_with_requested_budget(self) -> None:
        previous = LinearHead(feature_dim=4, num_labels=2)
        model = TopKLabelWiseTrustRegionCorrectionModel(
            previous,
            self._bank().to_module(top_k=1, temperature=0.1),
            feature_dim=4,
            num_labels=2,
            top_k=1,
        )
        for parameter in model.adapter.parameters():
            torch.nn.init.zeros_(parameter)
        embeddings = torch.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        outputs = model(embeddings)
        with torch.no_grad():
            previous_logits = previous(embeddings)
        self.assertEqual(outputs["gate"].shape, (2, 2))
        self.assertTrue(torch.all((outputs["gate"] == 0.0) | (outputs["gate"] == 1.0)))
        self.assertTrue(torch.all(outputs["gate"].sum(dim=1) == 1.0))
        self.assertTrue(torch.allclose(outputs["selected_label_count"], torch.ones((2, 1))))
        self.assertTrue(torch.allclose(outputs["logits"], previous_logits))


if __name__ == "__main__":
    unittest.main()
