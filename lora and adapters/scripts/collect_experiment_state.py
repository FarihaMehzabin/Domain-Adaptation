#!/usr/bin/env python3
"""Collect experiment artifacts into a machine-readable CSV index."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "reports" / "discovered_artifacts.csv"

SCAN_DIRS = [
    "results",
    "outputs",
    "logs",
    "runs",
    "wandb",
    "checkpoints",
    "lightning_logs",
    "tensorboard",
    "experiments",
    "reports",
    "manifests",
    "configs",
]


@dataclass(frozen=True)
class ArtifactRecord:
    relative_path: str
    category: str
    suffix: str
    size_bytes: int
    modified_utc: str


def category_for_path(path: Path) -> str:
    name = path.name
    suffix = path.suffix.lower()

    if name.startswith("events.out.tfevents"):
        return "tensorboard_event"
    if suffix == ".pt":
        return "checkpoint"
    if suffix == ".log":
        return "log"
    if suffix == ".out":
        return "stdout_stderr_log"
    if suffix == ".csv":
        if "prediction" in name:
            return "prediction_csv"
        if "manifest" in name or path.parent.name == "manifests":
            return "manifest_csv"
        return "csv"
    if suffix == ".json":
        return "report_json" if path.parent.name == "reports" else "json"
    if suffix == ".md":
        return "report_md" if path.parent.name == "reports" else "markdown"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "image"
    if suffix in {".yaml", ".yml", ".toml", ".ini"}:
        return "config"
    if suffix == ".ipynb":
        return "notebook"
    return "other"


def utc_timestamp(stat_mtime: float) -> str:
    return datetime.fromtimestamp(stat_mtime, tz=timezone.utc).isoformat()


def iter_records() -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    seen: set[Path] = set()

    for dirname in SCAN_DIRS:
        base_dir = ROOT / dirname
        if not base_dir.exists() or not base_dir.is_dir():
            continue

        for path in sorted(base_dir.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            stat = path.stat()
            records.append(
                ArtifactRecord(
                    relative_path=str(path.relative_to(ROOT)),
                    category=category_for_path(path),
                    suffix=path.suffix.lower(),
                    size_bytes=int(stat.st_size),
                    modified_utc=utc_timestamp(stat.st_mtime),
                )
            )

    records.sort(key=lambda item: item.relative_path)
    return records


def write_csv(records: list[ArtifactRecord], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "category", "suffix", "size_bytes", "modified_utc"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def main() -> int:
    records = iter_records()
    write_csv(records, OUTPUT_CSV)
    print(f"wrote {len(records)} artifact rows to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
