"""Small IO helpers for configs, reports, and artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def write_text(path: str | Path, text: str) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def write_csv(path: str | Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def resolve_path(base_dir: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()
