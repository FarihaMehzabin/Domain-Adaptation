"""Forgetting summaries."""

from __future__ import annotations

from typing import Any


def compute_forgetting(stage_history: list[dict[str, Any]], *, metric_key: str = "macro_auroc") -> dict[str, Any]:
    best_seen: dict[str, float] = {}
    forgetting_by_stage: list[dict[str, Any]] = []
    for stage_index, stage_entry in enumerate(stage_history):
        domain_metrics = stage_entry["domain_metrics"]
        stage_forgetting: dict[str, float] = {}
        for domain, metrics in domain_metrics.items():
            value = metrics.get(metric_key)
            if value is None:
                continue
            previous_best = best_seen.get(domain)
            if previous_best is not None:
                stage_forgetting[domain] = max(previous_best - value, 0.0)
                best_seen[domain] = max(previous_best, value)
            else:
                best_seen[domain] = value
        forgetting_by_stage.append(
            {
                "stage_index": stage_index,
                "stage_name": stage_entry.get("stage_name", f"stage_{stage_index}"),
                "per_domain_forgetting": stage_forgetting,
            }
        )
    final_domains = forgetting_by_stage[-1]["per_domain_forgetting"] if forgetting_by_stage else {}
    final_average = sum(final_domains.values()) / len(final_domains) if final_domains else 0.0
    return {
        "stages": forgetting_by_stage,
        "final_average_forgetting": final_average,
    }
