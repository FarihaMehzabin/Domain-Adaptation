"""Fixed-alpha retrieval mixing baseline."""

from __future__ import annotations

from paper_v1.training.stage_adapt import evaluate_fixed_alpha_mix, select_fixed_alpha


def run_fixed_alpha_mix(**kwargs):
    return evaluate_fixed_alpha_mix(**kwargs)


__all__ = ["run_fixed_alpha_mix", "select_fixed_alpha"]
