"""VQ summary replay baseline."""

from __future__ import annotations

from paper_v1.training.stage_adapt import train_linear_adaptation


def run_vq_summary_replay(**kwargs):
    return train_linear_adaptation(method_name="vq_summary_replay", **kwargs)
