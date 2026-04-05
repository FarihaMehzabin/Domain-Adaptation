#!/usr/bin/env python3
"""Generate split-aware fused embeddings from any number of embedding sources."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'numpy'.") from exc


DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_SPLITS = ["train", "val", "test"]
DEFAULT_OPERATION_LABEL = "fused_embedding_generation"
DEFAULT_EXPERIMENT_PREFIX = f"{DEFAULT_OPERATION_LABEL}__embeddings"
DEFAULT_EXPERIMENT_ID_WIDTH = 4
DEFAULT_ROW_CHUNK_SIZE = 4096

ID_PARSER_CHOICES = ("identity", "stem", "basename")
NORMALIZATION_CHOICES = ("l2", "none")
ALIGNMENT_CHOICES = ("reference", "intersection")
FUSION_CHOICES = ("concat",)

AUTO_ID_COLUMNS = ("sample_id", "row_id", "report_id", "image_id", "id")
AUTO_PATH_COLUMNS = ("image_path", "report_path", "path")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: Path
    embeddings_relpath: str = "embeddings.npy"
    ids_relpath: str | None = None
    id_column: str | None = None
    id_parser: str | None = None
    weight: float = 1.0
    reference: bool = False


@dataclass(frozen=True)
class SidecarSpec:
    relative_path: str
    format: str
    parser: str
    column: str | None = None


@dataclass
class LoadedSource:
    spec: SourceSpec
    split: str
    split_dir: Path
    embeddings_path: Path
    embeddings_shape: tuple[int, int]
    sidecar: SidecarSpec
    raw_items: list[str]
    row_ids: list[str]
    id_to_index: dict[str, int]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2, sort_keys=True), encoding="utf-8")


def slugify(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or fallback


def ensure_operation_prefix(name: str, operation_label: str = DEFAULT_OPERATION_LABEL) -> str:
    normalized_label = slugify(operation_label, fallback="operation")
    explicit_match = re.match(r"^(exp\d+__)(.+)$", name)
    if explicit_match is not None:
        numbered_prefix, remainder = explicit_match.groups()
        if re.match(rf"^{re.escape(normalized_label)}(?:__|$)", remainder):
            return name
        return f"{numbered_prefix}{normalized_label}__{remainder}"

    if re.match(rf"^{re.escape(normalized_label)}(?:__|$)", name):
        return name
    return f"{normalized_label}__{name}"


def extract_experiment_number(name: str) -> int | None:
    match = re.match(r"^exp(\d+)(?:__|$)", name)
    if match is None:
        return None
    return int(match.group(1))


def next_experiment_number(experiments_root: Path) -> int:
    if not experiments_root.exists():
        return 1

    max_number = 0
    for child in experiments_root.iterdir():
        if not child.is_dir():
            continue
        experiment_number = extract_experiment_number(child.name)
        if experiment_number is None:
            continue
        max_number = max(max_number, experiment_number)
    return max_number + 1


def resolve_experiment_identity(
    *,
    experiments_root: Path,
    requested_name: str | None,
    generated_slug: str,
    overwrite: bool,
    id_width: int = DEFAULT_EXPERIMENT_ID_WIDTH,
) -> tuple[int, str, str, Path]:
    experiments_root.mkdir(parents=True, exist_ok=True)

    if requested_name:
        requested_name = requested_name.strip()
        if not requested_name:
            raise SystemExit("--experiment-name cannot be empty.")
    base_name = ensure_operation_prefix(requested_name or generated_slug)

    explicit_number = extract_experiment_number(base_name)
    if explicit_number is not None:
        experiment_number = explicit_number
        experiment_name = base_name
    else:
        experiment_number = next_experiment_number(experiments_root)
        experiment_name = f"exp{experiment_number:0{id_width}d}__{base_name}"

    experiment_id = f"exp{experiment_number:0{id_width}d}"
    experiment_dir = experiments_root / experiment_name
    if experiment_dir.exists() and not overwrite:
        raise SystemExit(
            f"Experiment directory already exists: {experiment_dir}\n"
            "Pass --overwrite to reuse it or choose a different --experiment-name."
        )
    return experiment_number, experiment_id, experiment_name, experiment_dir


def parse_bool_flag(value: str, *, field_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise SystemExit(f"Invalid boolean for {field_name}: {value}")


def parse_source_spec(raw_value: str) -> SourceSpec:
    fields: dict[str, str] = {}
    for part in raw_value.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        if "=" not in cleaned:
            raise SystemExit(
                f"Invalid --source entry '{raw_value}'. Expected comma-separated key=value pairs."
            )
        key, value = cleaned.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if not key or not value:
            raise SystemExit(f"Invalid --source entry '{raw_value}'. Empty key or value.")
        fields[key] = value

    required = {"name", "root"}
    missing = sorted(required - set(fields))
    if missing:
        raise SystemExit(f"--source is missing required keys: {', '.join(missing)}")

    parser = fields.get("id_parser")
    if parser is not None and parser not in ID_PARSER_CHOICES:
        raise SystemExit(
            f"Unsupported id_parser '{parser}' in --source. Expected one of: {', '.join(ID_PARSER_CHOICES)}"
        )

    weight_text = fields.get("weight", "1.0")
    try:
        weight = float(weight_text)
    except ValueError as exc:
        raise SystemExit(f"Invalid weight '{weight_text}' in --source.") from exc
    if not math.isfinite(weight):
        raise SystemExit(f"Weight must be finite in --source, got {weight_text}.")

    reference = parse_bool_flag(fields.get("reference", "false"), field_name="reference")

    return SourceSpec(
        name=slugify(fields["name"], fallback="source"),
        root=Path(fields["root"]).expanduser(),
        embeddings_relpath=fields.get("embeddings", "embeddings.npy"),
        ids_relpath=fields.get("ids"),
        id_column=fields.get("id_column"),
        id_parser=parser,
        weight=weight,
        reference=reference,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse split-aware embedding experiments by aligning rows on stable IDs, concatenating the "
            "selected sources, and optionally L2 normalizing the fused output."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help=(
            "Embedding source spec as comma-separated key=value pairs. "
            "Required keys: name, root. Optional keys: embeddings, ids, id_column, id_parser, "
            "weight, reference. Example: "
            "name=image,root=/workspace/experiments/exp0001__...,ids=image_paths.txt,id_parser=stem"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help=f"Splits to fuse. Default: {' '.join(DEFAULT_SPLITS)}",
    )
    parser.add_argument(
        "--fusion",
        choices=FUSION_CHOICES,
        default="concat",
        help="Fusion operator applied after alignment. The PDF-backed default is concat.",
    )
    parser.add_argument(
        "--normalize-output",
        choices=NORMALIZATION_CHOICES,
        default="l2",
        help="Normalization applied to the fused embedding before saving. Default: l2",
    )
    parser.add_argument(
        "--alignment",
        choices=ALIGNMENT_CHOICES,
        default="reference",
        help=(
            "How to choose the output row set. reference: keep the reference source order and require "
            "every other source to cover it. intersection: keep only the common IDs in reference order."
        ),
    )
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=DEFAULT_ROW_CHUNK_SIZE,
        help=f"Chunk size used when copying aligned embeddings. Default: {DEFAULT_ROW_CHUNK_SIZE}",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=DEFAULT_EXPERIMENTS_ROOT,
        help=f"Where fused experiment directories will be created. Default: {DEFAULT_EXPERIMENTS_ROOT}",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name. If omitted, a name is generated from the fusion config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reuse the target experiment directory if it already exists.",
    )
    return parser.parse_args()


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    head = path.read_bytes()[:64]
    return head.startswith(b"version https://git-lfs.github.com/spec/") or head.startswith(b"version https://")


def read_csv_column(path: Path, column: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or column not in reader.fieldnames:
        raise SystemExit(f"CSV sidecar {path} does not contain column '{column}'.")

    values: list[str] = []
    for row in reader:
        value = row.get(column)
        if value is None:
            raise SystemExit(f"CSV sidecar {path} contains a row without column '{column}'.")
        values.append(str(value).strip())
    return values


def read_sidecar_items(path: Path, *, format_name: str, column: str | None) -> list[str]:
    if format_name == "lines":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if format_name == "json_list":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit(f"Expected JSON list in sidecar: {path}")
        return [str(item).strip() for item in payload if str(item).strip()]
    if format_name == "csv_column":
        if not column:
            raise SystemExit(f"CSV sidecar requires a column name: {path}")
        return [item for item in read_csv_column(path, column) if item]
    raise SystemExit(f"Unsupported sidecar format '{format_name}' for {path}")


def parse_row_id(raw_item: str, parser_name: str) -> str:
    cleaned = raw_item.strip()
    if not cleaned:
        raise ValueError("row ID source item is empty")
    if parser_name == "identity":
        return cleaned
    if parser_name == "stem":
        return Path(cleaned).stem
    if parser_name == "basename":
        return Path(cleaned).name
    raise ValueError(f"Unsupported parser '{parser_name}'")


def choose_csv_sidecar(path: Path) -> SidecarSpec | None:
    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return None

    for column in AUTO_ID_COLUMNS:
        if column in reader.fieldnames:
            return SidecarSpec(relative_path=path.name, format="csv_column", parser="identity", column=column)
    for column in AUTO_PATH_COLUMNS:
        if column in reader.fieldnames:
            return SidecarSpec(relative_path=path.name, format="csv_column", parser="stem", column=column)
    return None


def autodetect_sidecar(split_dir: Path) -> SidecarSpec:
    preferred_exact = [
        SidecarSpec(relative_path="image_paths.txt", format="lines", parser="stem"),
        SidecarSpec(relative_path="report_ids.json", format="json_list", parser="identity"),
        SidecarSpec(relative_path="sample_ids.json", format="json_list", parser="identity"),
    ]
    for candidate in preferred_exact:
        if (split_dir / candidate.relative_path).exists():
            return candidate

    json_candidates = sorted(
        path
        for path in split_dir.glob("*_ids.json")
        if path.is_file() and path.name not in {"row_ids.json"}
    )
    if len(json_candidates) == 1:
        return SidecarSpec(relative_path=json_candidates[0].name, format="json_list", parser="identity")
    if len(json_candidates) > 1:
        names = ", ".join(path.name for path in json_candidates)
        raise SystemExit(
            f"Auto-detection is ambiguous in {split_dir}: multiple *_ids.json files found ({names}). "
            "Pass ids=... in --source."
        )

    txt_candidates = sorted(path for path in split_dir.glob("*_paths.txt") if path.is_file())
    if len(txt_candidates) == 1:
        return SidecarSpec(relative_path=txt_candidates[0].name, format="lines", parser="stem")
    if len(txt_candidates) > 1:
        names = ", ".join(path.name for path in txt_candidates)
        raise SystemExit(
            f"Auto-detection is ambiguous in {split_dir}: multiple *_paths.txt files found ({names}). "
            "Pass ids=... and id_parser=... in --source."
        )

    csv_candidates = sorted(path for path in split_dir.glob("*manifest.csv") if path.is_file())
    for candidate in csv_candidates:
        sidecar = choose_csv_sidecar(candidate)
        if sidecar is not None:
            return sidecar

    raise SystemExit(
        f"Could not auto-detect a row-ID sidecar in {split_dir}. Pass ids=... in --source."
    )


def resolve_sidecar(split_dir: Path, spec: SourceSpec) -> SidecarSpec:
    if spec.ids_relpath is None:
        return autodetect_sidecar(split_dir)

    path = split_dir / spec.ids_relpath
    if not path.exists():
        raise SystemExit(f"Configured sidecar for source '{spec.name}' not found: {path}")

    if path.suffix.lower() == ".json":
        parser = spec.id_parser or "identity"
        return SidecarSpec(relative_path=spec.ids_relpath, format="json_list", parser=parser, column=spec.id_column)
    if path.suffix.lower() == ".txt":
        parser = spec.id_parser or "stem"
        return SidecarSpec(relative_path=spec.ids_relpath, format="lines", parser=parser, column=spec.id_column)
    if path.suffix.lower() == ".csv":
        if not spec.id_column:
            raise SystemExit(
                f"Source '{spec.name}' uses CSV sidecar {path}, so id_column=... is required in --source."
            )
        parser = spec.id_parser or ("stem" if spec.id_column in AUTO_PATH_COLUMNS else "identity")
        return SidecarSpec(
            relative_path=spec.ids_relpath,
            format="csv_column",
            parser=parser,
            column=spec.id_column,
        )
    raise SystemExit(
        f"Unsupported sidecar extension for source '{spec.name}': {path.suffix}. "
        "Supported extensions: .txt, .json, .csv"
    )


def build_id_index(row_ids: list[str], *, source_name: str, split: str) -> dict[str, int]:
    index: dict[str, int] = {}
    duplicates: list[str] = []
    for row_idx, row_id in enumerate(row_ids):
        if not row_id:
            raise SystemExit(f"Source '{source_name}' split '{split}' contains an empty row ID.")
        if row_id in index:
            if len(duplicates) < 5:
                duplicates.append(row_id)
            continue
        index[row_id] = row_idx
    if duplicates:
        raise SystemExit(
            f"Source '{source_name}' split '{split}' contains duplicate row IDs. Examples: {duplicates}"
        )
    return index


def load_source_split(spec: SourceSpec, split: str) -> LoadedSource:
    split_dir = spec.root / split
    if not split_dir.exists():
        raise SystemExit(f"Source '{spec.name}' split directory not found: {split_dir}")

    embeddings_path = split_dir / spec.embeddings_relpath
    if not embeddings_path.exists():
        raise SystemExit(f"Embeddings not found for source '{spec.name}' split '{split}': {embeddings_path}")
    if is_lfs_pointer(embeddings_path):
        raise SystemExit(
            f"Embeddings for source '{spec.name}' split '{split}' appear to be a Git LFS pointer: {embeddings_path}"
        )

    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise SystemExit(
            f"Expected a 2D embeddings array for source '{spec.name}' split '{split}', got {embeddings.shape}"
        )

    sidecar = resolve_sidecar(split_dir, spec)
    sidecar_path = split_dir / sidecar.relative_path
    raw_items = read_sidecar_items(sidecar_path, format_name=sidecar.format, column=sidecar.column)
    if len(raw_items) != int(embeddings.shape[0]):
        raise SystemExit(
            f"Source '{spec.name}' split '{split}' has {embeddings.shape[0]} embedding rows but "
            f"{len(raw_items)} sidecar rows in {sidecar_path}."
        )

    row_ids: list[str] = []
    for item in raw_items:
        try:
            row_ids.append(parse_row_id(item, sidecar.parser))
        except ValueError as exc:
            raise SystemExit(
                f"Failed to derive row ID for source '{spec.name}' split '{split}' from item '{item}': {exc}"
            ) from exc

    return LoadedSource(
        spec=spec,
        split=split,
        split_dir=split_dir,
        embeddings_path=embeddings_path,
        embeddings_shape=(int(embeddings.shape[0]), int(embeddings.shape[1])),
        sidecar=sidecar,
        raw_items=raw_items,
        row_ids=row_ids,
        id_to_index=build_id_index(row_ids, source_name=spec.name, split=split),
    )


def choose_reference_index(sources: list[SourceSpec]) -> int:
    flagged = [idx for idx, spec in enumerate(sources) if spec.reference]
    if len(flagged) > 1:
        raise SystemExit("At most one --source may set reference=true.")
    if flagged:
        return flagged[0]
    return 0


def build_aligned_row_ids(
    loaded_sources: list[LoadedSource],
    *,
    reference_index: int,
    alignment_mode: str,
) -> list[str]:
    reference_ids = loaded_sources[reference_index].row_ids
    if alignment_mode == "reference":
        return list(reference_ids)

    if alignment_mode != "intersection":
        raise SystemExit(f"Unsupported alignment mode: {alignment_mode}")

    common_ids = set(reference_ids)
    for idx, source in enumerate(loaded_sources):
        if idx == reference_index:
            continue
        common_ids &= set(source.row_ids)
    aligned = [row_id for row_id in reference_ids if row_id in common_ids]
    if not aligned:
        raise SystemExit("No common row IDs were found across the selected sources.")
    return aligned


def build_alignment_indices(
    loaded_sources: list[LoadedSource],
    *,
    ordered_row_ids: list[str],
    alignment_mode: str,
) -> dict[str, np.ndarray]:
    indices: dict[str, np.ndarray] = {}
    for source in loaded_sources:
        missing: list[str] = []
        aligned: list[int] = []
        for row_id in ordered_row_ids:
            index = source.id_to_index.get(row_id)
            if index is None:
                if len(missing) < 5:
                    missing.append(row_id)
                continue
            aligned.append(index)

        if alignment_mode == "reference" and len(aligned) != len(ordered_row_ids):
            raise SystemExit(
                f"Source '{source.spec.name}' split '{source.split}' is missing "
                f"{len(ordered_row_ids) - len(aligned)} IDs required by the reference source. "
                f"Examples: {missing}"
            )

        indices[source.spec.name] = np.asarray(aligned, dtype=np.int64)
    return indices


def build_component_offsets(loaded_sources: list[LoadedSource]) -> list[dict[str, Any]]:
    offsets: list[dict[str, Any]] = []
    start = 0
    for source in loaded_sources:
        width = source.embeddings_shape[1]
        offsets.append(
            {
                "source": source.spec.name,
                "start_col": start,
                "end_col_exclusive": start + width,
                "width": width,
                "weight": source.spec.weight,
            }
        )
        start += width
    return offsets


def assemble_fused_embeddings(
    loaded_sources: list[LoadedSource],
    *,
    alignment_indices: dict[str, np.ndarray],
    num_rows: int,
    row_chunk_size: int,
) -> np.ndarray:
    total_dim = sum(source.embeddings_shape[1] for source in loaded_sources)
    fused = np.empty((num_rows, total_dim), dtype=np.float32)

    offset = 0
    for source in loaded_sources:
        source_dim = source.embeddings_shape[1]
        aligned_indices = alignment_indices[source.spec.name]
        embeddings = np.load(source.embeddings_path, mmap_mode="r")
        if embeddings.ndim != 2:
            raise SystemExit(
                f"Expected a 2D embeddings array for source '{source.spec.name}', got {embeddings.shape}"
            )

        for start in range(0, num_rows, row_chunk_size):
            stop = min(start + row_chunk_size, num_rows)
            chunk_rows = aligned_indices[start:stop]
            block = np.asarray(embeddings[chunk_rows], dtype=np.float32)
            if source.spec.weight != 1.0:
                block *= np.float32(source.spec.weight)
            fused[start:stop, offset : offset + source_dim] = block

        offset += source_dim
    return fused


def normalize_rows_in_place(embeddings: np.ndarray, *, row_chunk_size: int) -> dict[str, Any]:
    zero_norm_rows: list[int] = []
    pre_norm_min = math.inf
    pre_norm_max = 0.0
    pre_norm_sum = 0.0
    count = 0

    for start in range(0, embeddings.shape[0], row_chunk_size):
        stop = min(start + row_chunk_size, embeddings.shape[0])
        rows = embeddings[start:stop]
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        flat_norms = norms[:, 0]
        if flat_norms.size:
            pre_norm_min = min(pre_norm_min, float(np.min(flat_norms)))
            pre_norm_max = max(pre_norm_max, float(np.max(flat_norms)))
            pre_norm_sum += float(np.sum(flat_norms))
            count += int(flat_norms.size)
        for offset, value in enumerate(flat_norms):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                if len(zero_norm_rows) < 5:
                    zero_norm_rows.append(start + offset)
        if zero_norm_rows:
            continue
        rows /= norms

    if zero_norm_rows:
        raise SystemExit(
            f"Cannot L2 normalize fused embeddings because some rows have non-finite or zero norm. "
            f"Examples: {zero_norm_rows}"
        )

    post_norms = np.linalg.norm(embeddings, axis=1)
    return {
        "method": "l2",
        "rows": int(embeddings.shape[0]),
        "pre_norm_min": float(pre_norm_min if count else 0.0),
        "pre_norm_max": float(pre_norm_max if count else 0.0),
        "pre_norm_mean": float(pre_norm_sum / count) if count else 0.0,
        "post_norm_min": float(np.min(post_norms)) if post_norms.size else 0.0,
        "post_norm_max": float(np.max(post_norms)) if post_norms.size else 0.0,
        "post_norm_mean": float(np.mean(post_norms)) if post_norms.size else 0.0,
    }


def summarize_row_norms(embeddings: np.ndarray) -> dict[str, Any]:
    if embeddings.size == 0:
        return {"method": "none", "rows": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "method": "none",
        "rows": int(embeddings.shape[0]),
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
        "mean": float(np.mean(norms)),
    }


def write_alignment_manifest(
    path: Path,
    *,
    ordered_row_ids: list[str],
    loaded_sources: list[LoadedSource],
    alignment_indices: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_index", "row_id", *[f"{source.spec.name}_item" for source in loaded_sources]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for row_index, row_id in enumerate(ordered_row_ids):
            row = [row_index, row_id]
            for source in loaded_sources:
                source_row = int(alignment_indices[source.spec.name][row_index])
                row.append(source.raw_items[source_row])
            writer.writerow(row)


def copy_reference_sidecar(
    output_dir: Path,
    *,
    reference_source: LoadedSource,
    reference_indices: np.ndarray,
) -> str | None:
    relative_name = Path(reference_source.sidecar.relative_path).name
    destination = output_dir / relative_name

    aligned_items = [reference_source.raw_items[int(idx)] for idx in reference_indices]
    if reference_source.sidecar.format == "lines":
        destination.write_text("\n".join(aligned_items) + "\n", encoding="utf-8")
        return relative_name
    if reference_source.sidecar.format == "json_list":
        destination.write_text(json.dumps(aligned_items, indent=2) + "\n", encoding="utf-8")
        return relative_name
    return None


def generate_default_slug(
    source_specs: list[SourceSpec],
    *,
    fusion: str,
    normalize_output: str,
) -> str:
    source_names = "-".join(spec.name for spec in source_specs)
    return f"{DEFAULT_EXPERIMENT_PREFIX}__{source_names}__{fusion}__{normalize_output}"


def save_split_outputs(
    output_dir: Path,
    *,
    embeddings: np.ndarray,
    ordered_row_ids: list[str],
    loaded_sources: list[LoadedSource],
    alignment_indices: dict[str, np.ndarray],
    reference_index: int,
    run_meta: dict[str, Any],
) -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings.astype(np.float32))
    (output_dir / "row_ids.json").write_text(json.dumps(ordered_row_ids, indent=2) + "\n", encoding="utf-8")
    write_alignment_manifest(
        output_dir / "alignment_manifest.csv",
        ordered_row_ids=ordered_row_ids,
        loaded_sources=loaded_sources,
        alignment_indices=alignment_indices,
    )
    reference_sidecar_name = copy_reference_sidecar(
        output_dir,
        reference_source=loaded_sources[reference_index],
        reference_indices=alignment_indices[loaded_sources[reference_index].spec.name],
    )
    write_json(output_dir / "run_meta.json", run_meta)
    return reference_sidecar_name


def main() -> int:
    args = parse_args()
    if args.row_chunk_size <= 0:
        raise SystemExit("--row-chunk-size must be a positive integer.")

    source_specs = [parse_source_spec(item) for item in args.source]
    source_names = [spec.name for spec in source_specs]
    if len(source_names) != len(set(source_names)):
        raise SystemExit(f"Source names must be unique. Received: {source_names}")
    if len(source_specs) < 2:
        raise SystemExit("At least two --source entries are required for fusion.")

    reference_index = choose_reference_index(source_specs)

    generated_slug = generate_default_slug(
        source_specs,
        fusion=args.fusion,
        normalize_output=args.normalize_output,
    )
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=args.experiments_root,
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=bool(args.overwrite),
    )

    experiment_dir.mkdir(parents=True, exist_ok=True)

    split_output_dirs: dict[str, str] = {}
    split_summaries: dict[str, dict[str, Any]] = {}
    for split in dedupe_preserve_order([str(item).strip() for item in args.splits if str(item).strip()]):
        loaded_sources = [load_source_split(spec, split) for spec in source_specs]
        ordered_row_ids = build_aligned_row_ids(
            loaded_sources,
            reference_index=reference_index,
            alignment_mode=args.alignment,
        )
        alignment_indices = build_alignment_indices(
            loaded_sources,
            ordered_row_ids=ordered_row_ids,
            alignment_mode=args.alignment,
        )
        num_rows = len(ordered_row_ids)
        if num_rows == 0:
            raise SystemExit(f"Split '{split}' produced zero aligned rows.")

        if args.fusion != "concat":
            raise SystemExit(f"Unsupported fusion mode: {args.fusion}")

        fused_embeddings = assemble_fused_embeddings(
            loaded_sources,
            alignment_indices=alignment_indices,
            num_rows=num_rows,
            row_chunk_size=args.row_chunk_size,
        )
        if args.normalize_output == "l2":
            norm_summary = normalize_rows_in_place(fused_embeddings, row_chunk_size=args.row_chunk_size)
        else:
            norm_summary = summarize_row_norms(fused_embeddings)

        component_offsets = build_component_offsets(loaded_sources)
        source_split_meta: list[dict[str, Any]] = []
        for source in loaded_sources:
            aligned_rows = int(alignment_indices[source.spec.name].shape[0])
            source_split_meta.append(
                {
                    "name": source.spec.name,
                    "root": source.spec.root,
                    "split_dir": source.split_dir,
                    "embeddings_path": source.embeddings_path,
                    "embeddings_shape": list(source.embeddings_shape),
                    "weight": source.spec.weight,
                    "sidecar": {
                        "relative_path": source.sidecar.relative_path,
                        "format": source.sidecar.format,
                        "parser": source.sidecar.parser,
                        "column": source.sidecar.column,
                    },
                    "reference": bool(source.spec.reference),
                    "available_rows": int(source.embeddings_shape[0]),
                    "aligned_rows": aligned_rows,
                }
            )

        output_dir = experiment_dir / split
        run_meta = {
            "run_date_utc": utc_now_iso(),
            "split": split,
            "experiment_dir": experiment_dir,
            "fusion": args.fusion,
            "normalize_output": args.normalize_output,
            "alignment": args.alignment,
            "num_rows": num_rows,
            "fused_dim": int(fused_embeddings.shape[1]),
            "row_chunk_size": int(args.row_chunk_size),
            "reference_source": loaded_sources[reference_index].spec.name,
            "row_id_source": loaded_sources[reference_index].sidecar.relative_path,
            "component_offsets": component_offsets,
            "source_order": [source.spec.name for source in loaded_sources],
            "sources": source_split_meta,
            "row_norm_summary": norm_summary,
        }
        reference_sidecar_name = save_split_outputs(
            output_dir,
            embeddings=fused_embeddings,
            ordered_row_ids=ordered_row_ids,
            loaded_sources=loaded_sources,
            alignment_indices=alignment_indices,
            reference_index=reference_index,
            run_meta=run_meta,
        )
        if reference_sidecar_name is not None:
            run_meta["reference_sidecar_output"] = reference_sidecar_name
            write_json(output_dir / "run_meta.json", run_meta)

        split_output_dirs[split] = str(output_dir)
        split_summaries[split] = {
            "num_rows": num_rows,
            "fused_dim": int(fused_embeddings.shape[1]),
            "reference_source": loaded_sources[reference_index].spec.name,
            "row_norm_summary": norm_summary,
        }
        print(
            f"{split}: fused {num_rows} rows into dim={fused_embeddings.shape[1]} "
            f"(normalize={args.normalize_output}, reference={loaded_sources[reference_index].spec.name})"
        )

    experiment_meta = {
        "argv": sys.argv,
        "run_date_utc": utc_now_iso(),
        "experiment_id": experiment_id,
        "experiment_number": experiment_number,
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        "experiments_root": args.experiments_root,
        "operation_label": DEFAULT_OPERATION_LABEL,
        "fusion": args.fusion,
        "normalize_output": args.normalize_output,
        "alignment": args.alignment,
        "row_chunk_size": int(args.row_chunk_size),
        "splits": dedupe_preserve_order([str(item).strip() for item in args.splits if str(item).strip()]),
        "reference_source": source_specs[reference_index].name,
        "source_order": [spec.name for spec in source_specs],
        "sources": [
            {
                "name": spec.name,
                "root": spec.root,
                "embeddings_relpath": spec.embeddings_relpath,
                "ids_relpath": spec.ids_relpath,
                "id_column": spec.id_column,
                "id_parser": spec.id_parser,
                "weight": spec.weight,
                "reference": spec.reference,
            }
            for spec in source_specs
        ],
        "split_output_dirs": split_output_dirs,
        "split_summaries": split_summaries,
    }
    write_json(experiment_dir / "experiment_meta.json", experiment_meta)
    print(f"wrote fused embeddings to {experiment_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
