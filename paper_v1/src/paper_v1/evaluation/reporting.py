"""Artifact reporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper_v1.utils.io import ensure_dir, write_csv, write_json, write_text


def write_stage_metrics(output_dir: str | Path, filename: str, metrics: dict[str, Any]) -> Path:
    output_path = ensure_dir(output_dir) / filename
    write_json(output_path, metrics)
    return output_path


def write_summary_table(output_dir: str | Path, filename: str, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError("summary table requires at least one row")
    return write_csv(Path(output_dir) / filename, list(rows[0].keys()), rows)


def write_stage_report(
    output_dir: str | Path,
    filename: str,
    *,
    title: str,
    sections: list[tuple[str, list[str]]],
) -> Path:
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.append(f"## {heading}")
        if body:
            lines.extend(body)
        else:
            lines.append("- none")
        lines.append("")
    return write_text(Path(output_dir) / filename, "\n".join(lines))
