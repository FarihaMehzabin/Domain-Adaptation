import unittest

from paper_v1.analysis.postmortem import selection_count_distribution_rows
from paper_v1.analysis.seed_sweep import add_source_only_deltas, aggregate_seed_rows, promotion_decision


class SeedSweepTest(unittest.TestCase):
    def test_aggregate_seed_rows_computes_mean_and_std(self) -> None:
        rows = [
            {
                "method": "source_only",
                "seed": 1,
                "d0_nih_test_macro_auroc": 0.8,
                "d1_chexpert_test_macro_auroc": 0.7,
                "seen_average_macro_auroc": 0.75,
                "nih_forgetting_macro_auroc": 0.0,
                "average_macro_ece": 0.1,
                "average_brier_score": 0.2,
            },
            {
                "method": "source_only",
                "seed": 2,
                "d0_nih_test_macro_auroc": 0.8,
                "d1_chexpert_test_macro_auroc": 0.7,
                "seen_average_macro_auroc": 0.75,
                "nih_forgetting_macro_auroc": 0.0,
                "average_macro_ece": 0.1,
                "average_brier_score": 0.2,
            },
        ]
        aggregate = aggregate_seed_rows(rows)
        self.assertEqual(aggregate[0]["seen_average_macro_auroc_mean"], 0.75)
        self.assertEqual(aggregate[0]["seen_average_macro_auroc_std"], 0.0)

    def test_promotion_decision_requires_all_criteria(self) -> None:
        rows = add_source_only_deltas(
            [
                {
                    "method": "source_only",
                    "num_seeds": 3,
                    "d0_nih_test_macro_auroc_mean": 0.84,
                    "d0_nih_test_macro_auroc_std": 0.0,
                    "d0_nih_test_macro_auroc_min": 0.84,
                    "d0_nih_test_macro_auroc_max": 0.84,
                    "d1_chexpert_test_macro_auroc_mean": 0.85,
                    "d1_chexpert_test_macro_auroc_std": 0.0,
                    "d1_chexpert_test_macro_auroc_min": 0.85,
                    "d1_chexpert_test_macro_auroc_max": 0.85,
                    "seen_average_macro_auroc_mean": 0.845,
                    "seen_average_macro_auroc_std": 0.0,
                    "seen_average_macro_auroc_min": 0.845,
                    "seen_average_macro_auroc_max": 0.845,
                    "nih_forgetting_macro_auroc_mean": 0.0,
                    "nih_forgetting_macro_auroc_std": 0.0,
                    "nih_forgetting_macro_auroc_min": 0.0,
                    "nih_forgetting_macro_auroc_max": 0.0,
                    "average_macro_ece_mean": 0.06,
                    "average_macro_ece_std": 0.0,
                    "average_macro_ece_min": 0.06,
                    "average_macro_ece_max": 0.06,
                    "average_brier_score_mean": 0.08,
                    "average_brier_score_std": 0.0,
                    "average_brier_score_min": 0.08,
                    "average_brier_score_max": 0.08,
                },
                {
                    "method": "harder_gate_clipping",
                    "num_seeds": 3,
                    "d0_nih_test_macro_auroc_mean": 0.843,
                    "d0_nih_test_macro_auroc_std": 0.002,
                    "d0_nih_test_macro_auroc_min": 0.841,
                    "d0_nih_test_macro_auroc_max": 0.845,
                    "d1_chexpert_test_macro_auroc_mean": 0.853,
                    "d1_chexpert_test_macro_auroc_std": 0.003,
                    "d1_chexpert_test_macro_auroc_min": 0.85,
                    "d1_chexpert_test_macro_auroc_max": 0.856,
                    "seen_average_macro_auroc_mean": 0.848,
                    "seen_average_macro_auroc_std": 0.002,
                    "seen_average_macro_auroc_min": 0.846,
                    "seen_average_macro_auroc_max": 0.85,
                    "nih_forgetting_macro_auroc_mean": 0.002,
                    "nih_forgetting_macro_auroc_std": 0.001,
                    "nih_forgetting_macro_auroc_min": 0.001,
                    "nih_forgetting_macro_auroc_max": 0.004,
                    "average_macro_ece_mean": 0.062,
                    "average_macro_ece_std": 0.001,
                    "average_macro_ece_min": 0.061,
                    "average_macro_ece_max": 0.063,
                    "average_brier_score_mean": 0.079,
                    "average_brier_score_std": 0.001,
                    "average_brier_score_min": 0.078,
                    "average_brier_score_max": 0.08,
                },
            ]
        )
        decision = promotion_decision(
            rows,
            candidate_method="harder_gate_clipping",
            max_mean_forgetting=0.005,
            max_forgetting_any_seed=0.01,
            max_seen_average_std=0.005,
            max_chexpert_std=0.01,
        )
        self.assertTrue(decision["promote_to_stage2_candidate"])

    def test_selection_count_distribution_rows_normalize(self) -> None:
        rows = selection_count_distribution_rows(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            method="topk",
            domain_name="nih",
            seed=7,
        )
        by_count = {row["selected_label_count"]: row for row in rows}
        self.assertAlmostEqual(by_count[1]["fraction"], 2.0 / 3.0, places=7)
        self.assertAlmostEqual(by_count[2]["fraction"], 1.0 / 3.0, places=7)


if __name__ == "__main__":
    unittest.main()
