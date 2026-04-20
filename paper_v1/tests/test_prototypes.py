from __future__ import annotations

import unittest

import numpy as np
import torch

from paper_v1.models.prototype_memory import build_label_state_prototypes


class PrototypeMemoryTest(unittest.TestCase):
    def test_builder_returns_expected_shapes(self) -> None:
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [-1.0, 0.0],
                [-0.9, -0.1],
            ],
            dtype=np.float32,
        )
        labels = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        bank = build_label_state_prototypes(embeddings, labels, domain="d0", positive_k=1, negative_k=1, seed=7)
        self.assertGreaterEqual(bank.num_prototypes, 2)
        self.assertEqual(bank.soft_labels.shape[1], 2)
        self.assertGreater(bank.memory_size_bytes(), 0)

    def test_retrieve_exposes_labelwise_support_features(self) -> None:
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [-1.0, 0.0],
                [-0.9, -0.1],
            ],
            dtype=np.float32,
        )
        labels = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        bank = build_label_state_prototypes(embeddings, labels, domain="d0", positive_k=1, negative_k=1, seed=11)
        module = bank.to_module(top_k=1, temperature=0.1)
        retrieval = module.retrieve(
            torch.asarray([[1.0, 0.0]], dtype=torch.float32),
            previous_probabilities=torch.asarray([[0.9, 0.1]], dtype=torch.float32),
        )
        self.assertEqual(retrieval["positive_support"].shape, (1, 2))
        self.assertEqual(retrieval["negative_support"].shape, (1, 2))
        self.assertEqual(retrieval["matched_distance"].shape, (1, 2))
        self.assertEqual(retrieval["support_margin"].shape, (1, 2))
        self.assertGreater(float(retrieval["matched_support"][0, 0]), float(retrieval["opposing_support"][0, 0]))


if __name__ == "__main__":
    unittest.main()
