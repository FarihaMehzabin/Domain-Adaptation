"""Source-only baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from paper_v1.training.stage_adapt import _evaluate_and_write, load_linear_head_checkpoint


def run_source_only(
    *,
    checkpoint_path: str | Path,
    eval_datasets: dict[str, Any],
    output_dir: str | Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model = load_linear_head_checkpoint(checkpoint_path, device=device)
    return _evaluate_and_write(model, eval_datasets, batch_size=batch_size, device=device, output_dir=Path(output_dir))
