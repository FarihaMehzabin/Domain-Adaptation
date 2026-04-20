"""LwF baseline."""

from __future__ import annotations

from paper_v1.training.stage_adapt import train_linear_adaptation


def run_lwf(**kwargs):
    return train_linear_adaptation(method_name="lwf", **kwargs)
