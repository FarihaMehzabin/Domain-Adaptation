"""EWC baseline."""

from __future__ import annotations

from paper_v1.training.stage_adapt import train_linear_adaptation


def run_ewc(**kwargs):
    return train_linear_adaptation(method_name="ewc", **kwargs)
