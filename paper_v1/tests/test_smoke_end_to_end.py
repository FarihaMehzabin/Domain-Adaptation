from __future__ import annotations

import tempfile
import unittest

from paper_v1.runners.run_smoke import run


class SmokeEndToEndTest(unittest.TestCase):
    def test_smoke_runner_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run(
                {
                    "experiment_name": "test_smoke",
                    "output_root": tmp_dir,
                    "seed": 1337,
                    "feature_dim": 8,
                    "num_labels": 7,
                    "training": {
                        "epochs": 3,
                        "batch_size": 16,
                        "lr": 0.01,
                        "weight_decay": 0.0001,
                        "patience": 2,
                        "distill_weight": 0.5,
                        "temperature": 2.0,
                        "l2_anchor_weight": 0.0001,
                        "ewc_weight": 1.0,
                        "replay_weight": 1.0,
                        "fisher_max_batches": 2
                    },
                    "memory": {"positive_k": 2, "negative_k": 1},
                    "main_method": {
                        "epochs": 2,
                        "batch_size": 16,
                        "lr": 0.01,
                        "weight_decay": 0.0001,
                        "patience": 2,
                        "distill_weight": 0.5,
                        "distill_temperature": 2.0,
                        "prototype_replay_weight": 1.0,
                        "residual_zero_weight": 0.1,
                        "top_k": 2,
                        "temperature": 0.2,
                        "gate_hidden_dim": 16,
                        "bottleneck_dim": 16,
                        "dropout": 0.0
                    }
                }
            )
            self.assertIn("summary_csv", result)


if __name__ == "__main__":
    unittest.main()
