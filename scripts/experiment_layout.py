#!/usr/bin/env python3
"""Shared helpers for numbered experiment storage and discovery."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
STATUS_DIRS = ("active", "complete", "archived")
INDEX_FILENAME = "INDEX.csv"
RUN_METADATA_FILENAME = "run.json"


@dataclass(frozen=True)
class ExperimentEntry:
    experiment_id: str
    experiment_number: int
    name: str
    slug: str
    status: str
    path: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or fallback


def extract_experiment_number(name: str) -> int | None:
    match = re.match(r"^exp(\d+)(?:__|$)", name)
    if match is None:
        return None
    return int(match.group(1))


def strip_experiment_number_prefix(name: str) -> str:
    if name.startswith("exp") and "__" in name:
        _, remainder = name.split("__", 1)
        return remainder
    return name


def is_experiment_dir(path: Path) -> bool:
    return path.is_dir() and extract_experiment_number(path.name) is not None


def status_root(experiments_root: Path, status: str) -> Path:
    if status not in STATUS_DIRS:
        raise SystemExit(f"Unsupported experiment status '{status}'. Expected one of: {', '.join(STATUS_DIRS)}.")
    return experiments_root / status


def infer_status(experiments_root: Path, path: Path) -> str:
    for status in STATUS_DIRS:
        if path.parent == experiments_root / status:
            return status
    return "legacy"


def iter_experiment_entries(experiments_root: Path = DEFAULT_EXPERIMENTS_ROOT) -> list[ExperimentEntry]:
    if not experiments_root.exists():
        return []

    entries: list[ExperimentEntry] = []
    seen: set[Path] = set()

    def register(path: Path, status: str) -> None:
        resolved = path.resolve()
        if resolved in seen or not is_experiment_dir(path):
            return
        seen.add(resolved)
        number = extract_experiment_number(path.name)
        if number is None:
            return
        entries.append(
            ExperimentEntry(
                experiment_id=f"exp{number:04d}",
                experiment_number=number,
                name=path.name,
                slug=strip_experiment_number_prefix(path.name),
                status=status,
                path=path,
            )
        )

    for child in sorted(experiments_root.iterdir(), key=lambda item: item.name):
        if child.name in STATUS_DIRS:
            continue
        register(child, "legacy")

    for status in STATUS_DIRS:
        root = experiments_root / status
        if not root.exists():
            continue
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            register(child, status)

    return sorted(entries, key=lambda item: (item.experiment_number, item.name))


def next_experiment_number(experiments_root: Path = DEFAULT_EXPERIMENTS_ROOT) -> int:
    entries = iter_experiment_entries(experiments_root)
    if not entries:
        return 1
    return max(entry.experiment_number for entry in entries) + 1


def find_experiment(
    reference: str | Path,
    *,
    experiments_root: Path = DEFAULT_EXPERIMENTS_ROOT,
) -> ExperimentEntry:
    if isinstance(reference, Path):
        candidate_path = reference
    else:
        text = reference.strip()
        candidate_path = Path(text) if "/" in text or text.startswith(".") else None

    if candidate_path is not None:
        if candidate_path.exists() and is_experiment_dir(candidate_path):
            number = extract_experiment_number(candidate_path.name)
            if number is None:
                raise SystemExit(f"Experiment path does not look like an experiment directory: {candidate_path}")
            return ExperimentEntry(
                experiment_id=f"exp{number:04d}",
                experiment_number=number,
                name=candidate_path.name,
                slug=strip_experiment_number_prefix(candidate_path.name),
                status=infer_status(experiments_root, candidate_path),
                path=candidate_path,
            )
        raise SystemExit(f"Experiment path not found: {candidate_path}")

    text = str(reference).strip()
    entries = iter_experiment_entries(experiments_root)
    matches = [entry for entry in entries if entry.name == text or entry.experiment_id == text]
    if not matches:
        raise SystemExit(f"Experiment not found: {text}")
    if len(matches) > 1:
        choices = ", ".join(str(entry.path) for entry in matches)
        raise SystemExit(f"Experiment reference '{text}' is ambiguous: {choices}")
    return matches[0]


def find_experiment_dir(
    reference: str | Path,
    *,
    experiments_root: Path = DEFAULT_EXPERIMENTS_ROOT,
) -> Path:
    return find_experiment(reference, experiments_root=experiments_root).path


def resolve_experiment_identity(
    *,
    experiments_root: Path,
    requested_name: str | None,
    generated_slug: str,
    overwrite: bool,
    id_width: int = 4,
    status: str = "active",
) -> tuple[int, str, str, Path]:
    experiments_root.mkdir(parents=True, exist_ok=True)
    target_root = status_root(experiments_root, status)
    target_root.mkdir(parents=True, exist_ok=True)

    requested = (requested_name or "").strip() or None
    base_name = requested or generated_slug
    explicit_number = extract_experiment_number(base_name)
    if explicit_number is None:
        experiment_number = next_experiment_number(experiments_root)
        experiment_name = f"exp{experiment_number:0{id_width}d}__{base_name}"
    else:
        experiment_number = explicit_number
        experiment_name = base_name

    experiment_id = f"exp{experiment_number:0{id_width}d}"
    experiment_dir = target_root / experiment_name

    for entry in iter_experiment_entries(experiments_root):
        if entry.name != experiment_name:
            continue
        if entry.path == experiment_dir:
            break
        raise SystemExit(
            f"Experiment name already exists at {entry.path}. "
            "Choose a different name or move/archive the existing run first."
        )

    if experiment_dir.exists() and not overwrite:
        raise SystemExit(
            f"Experiment directory already exists: {experiment_dir}\n"
            "Pass --overwrite to reuse it or choose a different --experiment-name."
        )

    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_number, experiment_id, experiment_name, experiment_dir


def read_run_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_run_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def ensure_standard_layout(
    experiment_dir: Path,
    *,
    stage_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    stage_names = stage_names or []
    metadata = dict(metadata or {})

    paths = {
        "stages": experiment_dir / "stages",
        "outputs": experiment_dir / "outputs",
        "logs": experiment_dir / "logs",
        "summary": experiment_dir / "summary.md",
        "run_json": experiment_dir / RUN_METADATA_FILENAME,
    }
    for key in ("stages", "outputs", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)

    for index, stage_name in enumerate(stage_names, start=1):
        stage_dir = paths["stages"] / f"{index:02d}_{slugify(stage_name, fallback=f'stage-{index:02d}')}"
        stage_dir.mkdir(parents=True, exist_ok=True)

    summary_title = metadata.get("summary_title") or experiment_dir.name
    if not paths["summary"].exists():
        paths["summary"].write_text(f"# {summary_title}\n\n", encoding="utf-8")

    run_metadata = read_run_metadata(paths["run_json"])
    if not run_metadata:
        run_metadata["created_at"] = utc_now_iso()
    run_metadata.update(metadata)
    run_metadata.setdefault("experiment_name", experiment_dir.name)
    run_metadata.setdefault("experiment_id", experiment_dir.name.split("__", 1)[0])
    run_metadata.setdefault("slug", strip_experiment_number_prefix(experiment_dir.name))
    run_metadata["status"] = metadata.get("status") or run_metadata.get("status") or "active"
    run_metadata["path"] = str(experiment_dir)
    run_metadata["updated_at"] = utc_now_iso()
    write_run_metadata(paths["run_json"], run_metadata)
    return paths


def move_experiment(
    reference: str | Path,
    *,
    experiments_root: Path = DEFAULT_EXPERIMENTS_ROOT,
    status: str,
) -> Path:
    entry = find_experiment(reference, experiments_root=experiments_root)
    target_root = status_root(experiments_root, status)
    target_root.mkdir(parents=True, exist_ok=True)
    target_path = target_root / entry.name
    if entry.path == target_path:
        return target_path
    if target_path.exists():
        raise SystemExit(f"Target experiment directory already exists: {target_path}")
    entry.path.rename(target_path)
    run_json_path = target_path / RUN_METADATA_FILENAME
    metadata = read_run_metadata(run_json_path)
    if metadata:
        metadata["status"] = status
        metadata["path"] = str(target_path)
        metadata["updated_at"] = utc_now_iso()
        write_run_metadata(run_json_path, metadata)
    return target_path


def write_index(
    experiments_root: Path = DEFAULT_EXPERIMENTS_ROOT,
    *,
    index_path: Path | None = None,
) -> Path:
    experiments_root.mkdir(parents=True, exist_ok=True)
    destination = index_path or (experiments_root / INDEX_FILENAME)
    entries = iter_experiment_entries(experiments_root)
    fieldnames = [
        "exp_id",
        "status",
        "name",
        "slug",
        "path",
        "parents",
        "kind",
        "family",
        "backbone",
        "source_domain",
        "target_domain",
        "tags",
        "created_at",
        "updated_at",
        "has_run_json",
        "has_summary",
    ]

    rows: list[dict[str, str]] = []
    for entry in entries:
        run_json_path = entry.path / RUN_METADATA_FILENAME
        summary_path = entry.path / "summary.md"
        metadata = read_run_metadata(run_json_path)
        parents = metadata.get("parents") or []
        tags = metadata.get("tags") or []
        rows.append(
            {
                "exp_id": entry.experiment_id,
                "status": entry.status,
                "name": entry.name,
                "slug": entry.slug,
                "path": str(entry.path.relative_to(experiments_root)),
                "parents": ";".join(str(item) for item in parents),
                "kind": str(metadata.get("kind") or ""),
                "family": str(metadata.get("family") or ""),
                "backbone": str(metadata.get("backbone") or ""),
                "source_domain": str(metadata.get("source_domain") or ""),
                "target_domain": str(metadata.get("target_domain") or ""),
                "tags": ";".join(str(item) for item in tags),
                "created_at": str(metadata.get("created_at") or ""),
                "updated_at": str(metadata.get("updated_at") or ""),
                "has_run_json": "1" if run_json_path.exists() else "0",
                "has_summary": "1" if summary_path.exists() else "0",
            }
        )

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination
