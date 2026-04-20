"""Run directory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from paper_v1.utils.io import ensure_dir


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    metrics: Path
    reports: Path
    artifacts: Path


def init_run_paths(output_root: str | Path, experiment_name: str) -> RunPaths:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = ensure_dir(Path(output_root) / f"{timestamp}__{experiment_name}")
    checkpoints = ensure_dir(root / "checkpoints")
    metrics = ensure_dir(root / "metrics")
    reports = ensure_dir(root / "reports")
    artifacts = ensure_dir(root / "artifacts")
    return RunPaths(root=root, checkpoints=checkpoints, metrics=metrics, reports=reports, artifacts=artifacts)
