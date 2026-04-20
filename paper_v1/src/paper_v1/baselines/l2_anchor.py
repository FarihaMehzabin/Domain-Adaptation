"""L2-anchor baseline."""

from __future__ import annotations

from paper_v1.training.stage_adapt import train_linear_adaptation


def run_l2_anchor(**kwargs):
    return train_linear_adaptation(method_name="l2_anchor", **kwargs)
