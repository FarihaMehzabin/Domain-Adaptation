"""Helpers for multiseed experiment summaries and promotion decisions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


SUMMARY_METRICS = (
    "d0_nih_test_macro_auroc",
    "d1_chexpert_test_macro_auroc",
    "seen_average_macro_auroc",
    "nih_forgetting_macro_auroc",
    "average_macro_ece",
    "average_brier_score",
)


def seed_result_row(method: str, seed: int, summary_row: dict[str, Any]) -> dict[str, Any]:
    row = {"method": method, "seed": int(seed)}
    row.update(summary_row)
    return row


def aggregate_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for method, method_rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {"method": method, "num_seeds": len(method_rows)}
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in method_rows], dtype=np.float64)
            aggregate[f"{metric}_mean"] = float(np.mean(values))
            aggregate[f"{metric}_std"] = float(np.std(values))
            aggregate[f"{metric}_min"] = float(np.min(values))
            aggregate[f"{metric}_max"] = float(np.max(values))
        aggregate_rows.append(aggregate)
    return aggregate_rows


def add_source_only_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method = {row["method"]: row for row in rows}
    source_row = by_method["source_only"]
    augmented: list[dict[str, Any]] = []
    for row in rows:
        augmented_row = dict(row)
        augmented_row["seen_average_delta_vs_source_only_mean"] = float(
            row["seen_average_macro_auroc_mean"] - source_row["seen_average_macro_auroc_mean"]
        )
        augmented_row["d1_chexpert_delta_vs_source_only_mean"] = float(
            row["d1_chexpert_test_macro_auroc_mean"] - source_row["d1_chexpert_test_macro_auroc_mean"]
        )
        augmented_row["average_macro_ece_delta_vs_source_only_mean"] = float(
            row["average_macro_ece_mean"] - source_row["average_macro_ece_mean"]
        )
        augmented.append(augmented_row)
    return augmented


def promotion_decision(
    aggregate_rows: list[dict[str, Any]],
    *,
    candidate_method: str,
    max_mean_forgetting: float,
    max_forgetting_any_seed: float,
    max_seen_average_std: float,
    max_chexpert_std: float,
    min_chexpert_gain: float = 0.0,
    min_seen_average_gain: float = 0.0,
) -> dict[str, Any]:
    by_method = {row["method"]: row for row in aggregate_rows}
    source_row = by_method["source_only"]
    candidate_row = by_method[candidate_method]
    criteria = {
        "mean_seen_average_above_source_only": bool(
            candidate_row["seen_average_macro_auroc_mean"]
            >= source_row["seen_average_macro_auroc_mean"] + min_seen_average_gain
        ),
        "mean_forgetting_near_zero": bool(candidate_row["nih_forgetting_macro_auroc_mean"] <= max_mean_forgetting),
        "max_forgetting_near_zero": bool(candidate_row["nih_forgetting_macro_auroc_max"] <= max_forgetting_any_seed),
        "mean_chexpert_gain_present": bool(
            candidate_row["d1_chexpert_test_macro_auroc_mean"]
            >= source_row["d1_chexpert_test_macro_auroc_mean"] + min_chexpert_gain
        ),
        "seen_average_variance_not_crazy": bool(candidate_row["seen_average_macro_auroc_std"] <= max_seen_average_std),
        "chexpert_variance_not_crazy": bool(candidate_row["d1_chexpert_test_macro_auroc_std"] <= max_chexpert_std),
    }
    return {
        "candidate_method": candidate_method,
        "source_only_reference": source_row,
        "candidate_summary": candidate_row,
        "criteria": criteria,
        "promote_to_stage2_candidate": bool(all(criteria.values())),
    }
