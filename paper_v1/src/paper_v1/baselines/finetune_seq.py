"""Sequential fine-tuning baseline."""

from __future__ import annotations

from paper_v1.training.stage_adapt import train_linear_adaptation


def run_finetune_seq(**kwargs):
    return train_linear_adaptation(method_name="finetune", **kwargs)
