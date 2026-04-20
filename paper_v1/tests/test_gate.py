from __future__ import annotations

import unittest

import torch

from paper_v1.models.gating import NoveltyGate


class GateTest(unittest.TestCase):
    def test_gate_outputs_probabilities(self) -> None:
        gate = NoveltyGate(feature_dim=4, num_labels=2, hidden_dim=8)
        outputs = gate(
            torch.randn(3, 4),
            torch.randn(3, 2),
            torch.rand(3, 2),
            torch.randn(3, 2),
        )
        self.assertEqual(outputs.shape, (3, 1))
        self.assertTrue(torch.all(outputs >= 0.0))
        self.assertTrue(torch.all(outputs <= 1.0))


if __name__ == "__main__":
    unittest.main()
