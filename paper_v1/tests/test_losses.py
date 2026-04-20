from __future__ import annotations

import unittest

import torch

from paper_v1.training.losses import (
    masked_bce_with_logits,
    old_prototype_margin_preservation_loss,
    residual_trust_region_loss,
    sigmoid_distillation_loss,
    trust_region_weights,
)


class LossesTest(unittest.TestCase):
    def test_masked_bce_is_zero_for_perfect_logits(self) -> None:
        logits = torch.tensor([[10.0, -10.0]])
        targets = torch.tensor([[1.0, 0.0]])
        mask = torch.ones_like(targets)
        loss = masked_bce_with_logits(logits, targets, mask=mask)
        self.assertLess(float(loss), 1.0e-4)

    def test_distillation_zero_when_logits_match(self) -> None:
        logits = torch.tensor([[1.0, -1.0]])
        loss = sigmoid_distillation_loss(logits, logits)
        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_margin_preservation_zero_when_logits_match(self) -> None:
        logits = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
        label_indices = torch.tensor([0, 1], dtype=torch.long)
        state_signs = torch.tensor([1.0, -1.0])
        loss = old_prototype_margin_preservation_loss(logits, logits, label_indices, state_signs, slack=0.25)
        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_trust_region_weights_threshold_support(self) -> None:
        matched_support = torch.tensor([[0.6, 0.9]])
        uncertainty = torch.tensor([[0.1, 0.2]])
        weights = trust_region_weights(matched_support, uncertainty, support_threshold=0.7)
        self.assertAlmostEqual(float(weights[0, 0]), 0.0, places=7)
        self.assertGreater(float(weights[0, 1]), 0.0)

    def test_residual_trust_region_loss_positive_with_weighted_residual(self) -> None:
        residual = torch.tensor([[1.0, 2.0]])
        matched_support = torch.tensor([[0.8, 0.95]])
        uncertainty = torch.tensor([[0.0, 0.0]])
        loss, weights = residual_trust_region_loss(
            residual,
            matched_support,
            uncertainty,
            support_threshold=0.7,
        )
        self.assertGreater(float(loss), 0.0)
        self.assertTrue(torch.all(weights > 0.0))


if __name__ == "__main__":
    unittest.main()
