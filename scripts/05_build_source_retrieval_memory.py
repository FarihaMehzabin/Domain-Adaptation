#!/usr/bin/env python3
"""Build a source-domain retrieval memory from a frozen embedding experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import random
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'faiss'. Install faiss-cpu before running "
        "/workspace/scripts/05_build_source_retrieval_memory.py."
    ) from exc

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'numpy'.") from exc


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_nih_cxr14_all14.csv")
DEFAULT_EMBEDDING_ROOT = Path(
    "/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2"
)
DEFAULT_BASELINE_EXPERIMENT_DIR = Path(
    "/workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4"
)
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_OPERATION_LABEL = "source_retrieval_memory_building"
DEFAULT_EXPERIMENT_ID_WIDTH = 4
DEFAULT_SPLIT = "train"
DEFAULT_SELF_RETRIEVAL_SAMPLE_SIZE = 512
DEFAULT_LABEL_AGREEMENT_QUERIES = 200
DEFAULT_QUALITATIVE_QUERIES = 10
DEFAULT_TOP_K = 5
DEFAULT_SEED = 3407

AUTO_ID_COLUMNS = ("sample_id", "row_id", "report_id", "image_id", "id")
AUTO_PATH_COLUMNS = ("image_path", "report_path", "path")
METADATA_COLUMNS = ("dataset", "split", "patient_id", "study_id", "view_raw", "view_group", "sex", "age")


@dataclass(frozen=True)
class ManifestRecord:
    row_id: str
    image_path: str
    labels: tuple[float, ...]
    metadata: dict[str, str]


@dataclass(frozen=True)
class SidecarSpec:
    relative_path: str
    format: str
    parser: str
    column: str | None = None


@dataclass(frozen=True)
class SourceSplit:
    split_dir: Path
    embeddings_path: Path
    embeddings: np.ndarray
    sidecar: SidecarSpec
    sidecar_items: list[str]
    row_ids: list[str]
    image_paths: list[str]
    alignment_rows: list[dict[str, str]] | None
    run_meta: dict[str, Any] | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str, *, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or fallback


def ensure_operation_prefix(name: str, operation_label: str = DEFAULT_OPERATION_LABEL) -> str:
    normalized_label = slugify(operation_label, fallback="operation")
    if name.startswith("exp") and "__" in name:
        prefix, remainder = name.split("__", 1)
        if remainder.startswith(normalized_label):
            return name
        return f"{prefix}__{normalized_label}__{remainder}"
    if name.startswith(normalized_label):
        return name
    return f"{normalized_label}__{name}"


def strip_experiment_number_prefix(name: str) -> str:
    if name.startswith("exp") and "__" in name:
        _, remainder = name.split("__", 1)
        return remainder
    return name


def extract_experiment_number(name: str) -> int | None:
    if not name.startswith("exp"):
        return None
    prefix = name.split("__", 1)[0]
    digits = prefix.removeprefix("exp")
    if not digits.isdigit():
        return None
    return int(digits)


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
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_number, experiment_id, experiment_name, experiment_dir


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
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"CSV file not found: {path}")
    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise SystemExit(f"CSV has no header row: {path}")
    return [{key: (value or "") for key, value in row.items()} for row in reader]


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
        cleaned = str(value).strip()
        if cleaned:
            values.append(cleaned)
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
        return read_csv_column(path, column)
    raise SystemExit(f"Unsupported sidecar format '{format_name}' for {path}")


def parse_row_id(raw_item: str, parser_name: str) -> str:
    cleaned = raw_item.strip()
    if not cleaned:
        raise SystemExit("Encountered an empty row identity source item.")
    if parser_name == "identity":
        return cleaned
    if parser_name == "stem":
        return Path(cleaned).stem
    if parser_name == "basename":
        return Path(cleaned).name
    raise SystemExit(f"Unsupported parser '{parser_name}'.")


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
        SidecarSpec(relative_path="row_ids.json", format="json_list", parser="identity"),
        SidecarSpec(relative_path="image_paths.txt", format="lines", parser="stem"),
        SidecarSpec(relative_path="report_ids.json", format="json_list", parser="identity"),
        SidecarSpec(relative_path="sample_ids.json", format="json_list", parser="identity"),
    ]
    for candidate in preferred_exact:
        if (split_dir / candidate.relative_path).exists():
            return candidate

    json_candidates = sorted(path for path in split_dir.glob("*_ids.json") if path.is_file())
    if len(json_candidates) == 1:
        return SidecarSpec(relative_path=json_candidates[0].name, format="json_list", parser="identity")
    if len(json_candidates) > 1:
        names = ", ".join(path.name for path in json_candidates)
        raise SystemExit(
            f"Auto-detection is ambiguous in {split_dir}: multiple *_ids.json files found ({names})."
        )

    txt_candidates = sorted(path for path in split_dir.glob("*_paths.txt") if path.is_file())
    if len(txt_candidates) == 1:
        return SidecarSpec(relative_path=txt_candidates[0].name, format="lines", parser="stem")
    if len(txt_candidates) > 1:
        names = ", ".join(path.name for path in txt_candidates)
        raise SystemExit(
            f"Auto-detection is ambiguous in {split_dir}: multiple *_paths.txt files found ({names})."
        )

    csv_candidates = sorted(path for path in split_dir.glob("*manifest.csv") if path.is_file())
    for candidate in csv_candidates:
        sidecar = choose_csv_sidecar(candidate)
        if sidecar is not None:
            return sidecar

    raise SystemExit(f"Could not auto-detect a row-identity sidecar in {split_dir}.")


def load_embedding_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"Embeddings file not found: {path}")
    array = np.load(path)
    if array.ndim != 2:
        raise SystemExit(f"Expected 2D embeddings in {path}, found shape {array.shape}")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise SystemExit(f"Embeddings contain NaN or inf values: {path}")
    return np.ascontiguousarray(array)


def load_manifest_records(
    manifest_csv: Path,
    *,
    split: str,
) -> tuple[list[str], list[str], dict[str, ManifestRecord]]:
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV not found: {manifest_csv}")
    manifest_text = manifest_csv.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(manifest_text))
    if reader.fieldnames is None:
        raise SystemExit(f"Manifest CSV has no header: {manifest_csv}")
    required_columns = {"dataset", "split", "image_path"}
    if not required_columns.issubset(reader.fieldnames):
        raise SystemExit(f"Manifest CSV must contain columns: {sorted(required_columns)}")
    label_columns = [field for field in reader.fieldnames if field.startswith("label_")]
    if not label_columns:
        raise SystemExit("Manifest CSV does not contain any label_... columns.")
    label_names = [column.removeprefix("label_") for column in label_columns]

    by_row_id: dict[str, ManifestRecord] = {}
    for row in reader:
        dataset = (row.get("dataset") or "").strip()
        if dataset and dataset != "nih_cxr14":
            continue
        current_split = (row.get("split") or "").strip().lower()
        if current_split != split:
            continue
        image_path = (row.get("image_path") or "").strip()
        if not image_path:
            continue
        row_id = Path(image_path).stem
        labels: list[float] = []
        for label_column in label_columns:
            raw_value = str(row.get(label_column) or "0").strip()
            try:
                labels.append(float(raw_value))
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid value '{raw_value}' in column '{label_column}' for row '{row_id}'."
                ) from exc
        if row_id in by_row_id:
            raise SystemExit(f"Duplicate manifest row_id '{row_id}' found in split '{split}'.")
        metadata = {key: str(row.get(key) or "").strip() for key in METADATA_COLUMNS}
        by_row_id[row_id] = ManifestRecord(
            row_id=row_id,
            image_path=image_path,
            labels=tuple(labels),
            metadata=metadata,
        )

    if not by_row_id:
        raise SystemExit(f"Manifest contains no rows for split '{split}'.")
    return label_columns, label_names, by_row_id


def read_optional_lines(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise SystemExit(f"Expected non-empty text file: {path}")
    return values


def load_source_split(embedding_root: Path, *, split: str) -> SourceSplit:
    split_dir = embedding_root / split
    if not split_dir.exists():
        raise SystemExit(f"Split directory not found for '{split}': {split_dir}")

    embeddings_path = split_dir / "embeddings.npy"
    embeddings = load_embedding_array(embeddings_path)

    sidecar = autodetect_sidecar(split_dir)
    sidecar_path = split_dir / sidecar.relative_path
    sidecar_items = read_sidecar_items(sidecar_path, format_name=sidecar.format, column=sidecar.column)
    if len(sidecar_items) != int(embeddings.shape[0]):
        raise SystemExit(
            f"Split '{split}' has {embeddings.shape[0]} embedding rows but "
            f"{len(sidecar_items)} sidecar rows in {sidecar_path}."
        )
    row_ids = [parse_row_id(item, sidecar.parser) for item in sidecar_items]

    image_paths_path = split_dir / "image_paths.txt"
    image_paths = read_optional_lines(image_paths_path) or []
    if image_paths and len(image_paths) != len(row_ids):
        raise SystemExit(
            f"Split '{split}' has {len(row_ids)} row IDs but {len(image_paths)} image paths in {image_paths_path}."
        )

    alignment_manifest_path = split_dir / "alignment_manifest.csv"
    alignment_rows = read_csv_rows(alignment_manifest_path) if alignment_manifest_path.exists() else None
    if alignment_rows is not None and len(alignment_rows) != len(row_ids):
        raise SystemExit(
            f"Split '{split}' has {len(row_ids)} row IDs but {len(alignment_rows)} rows in {alignment_manifest_path}."
        )

    run_meta_path = split_dir / "run_meta.json"
    run_meta = read_json(run_meta_path) if run_meta_path.exists() else None

    return SourceSplit(
        split_dir=split_dir,
        embeddings_path=embeddings_path,
        embeddings=embeddings,
        sidecar=sidecar,
        sidecar_items=sidecar_items,
        row_ids=row_ids,
        image_paths=image_paths,
        alignment_rows=alignment_rows,
        run_meta=run_meta,
    )


def validate_split_alignment(source_split: SourceSplit, manifest_records: dict[str, ManifestRecord]) -> None:
    missing: list[str] = []
    for row_id in source_split.row_ids:
        if row_id not in manifest_records and len(missing) < 5:
            missing.append(row_id)
    if missing:
        raise SystemExit(
            f"Found row IDs in split data that are missing from the manifest. Examples: {missing}"
        )

    if source_split.image_paths:
        for index, (row_id, image_path) in enumerate(zip(source_split.row_ids, source_split.image_paths)):
            expected = manifest_records[row_id].image_path
            if Path(expected).name != Path(image_path).name:
                raise SystemExit(
                    f"Image path mismatch at row {index}: manifest has '{expected}', split sidecar has '{image_path}'."
                )

    if source_split.alignment_rows is not None:
        for index, row in enumerate(source_split.alignment_rows):
            row_id = str(row.get("row_id") or "").strip()
            if row_id != source_split.row_ids[index]:
                raise SystemExit(
                    f"alignment_manifest.csv row_id mismatch at row {index}: "
                    f"expected '{source_split.row_ids[index]}', found '{row_id}'."
                )
            if source_split.image_paths:
                image_item = str(row.get("image_item") or "").strip()
                if image_item and image_item != source_split.image_paths[index]:
                    raise SystemExit(
                        f"alignment_manifest.csv image_item mismatch at row {index}: "
                        f"expected '{source_split.image_paths[index]}', found '{image_item}'."
                    )


def build_labels_and_rows(
    row_ids: list[str],
    manifest_records: dict[str, ManifestRecord],
) -> tuple[np.ndarray, list[ManifestRecord], list[str]]:
    labels = np.zeros((len(row_ids), len(next(iter(manifest_records.values())).labels)), dtype=np.float32)
    rows: list[ManifestRecord] = []
    image_paths: list[str] = []
    for index, row_id in enumerate(row_ids):
        record = manifest_records[row_id]
        labels[index] = np.asarray(record.labels, dtype=np.float32)
        rows.append(record)
        image_paths.append(record.image_path)
    return labels, rows, image_paths


def summarize_values(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def normalize_rows(embeddings: np.ndarray) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    raw_norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.isfinite(raw_norms).all():
        raise SystemExit("Raw embedding norms contain NaN or inf values.")
    zero_norms = int(np.count_nonzero(raw_norms <= 0.0))
    if zero_norms > 0:
        raise SystemExit(f"Found {zero_norms} zero-norm embeddings; cannot normalize.")
    normalized = embeddings / raw_norms[:, None].astype(np.float32)
    normalized = np.ascontiguousarray(normalized.astype(np.float32))
    if not np.isfinite(normalized).all():
        raise SystemExit("Normalized embeddings contain NaN or inf values.")
    normalized_norms = np.linalg.norm(normalized.astype(np.float64), axis=1)
    return normalized, summarize_values(raw_norms), summarize_values(normalized_norms)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    if embeddings.ndim != 2:
        raise SystemExit(f"Expected 2D embeddings, got shape {embeddings.shape}")
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(np.ascontiguousarray(embeddings.astype(np.float32)))
    return index


def labels_to_names(label_row: np.ndarray, label_names: list[str]) -> list[str]:
    indices = np.flatnonzero(label_row > 0.5)
    return [label_names[int(index)] for index in indices.tolist()]


def sample_indices(num_rows: int, sample_size: int, seed: int) -> np.ndarray:
    actual_size = min(num_rows, sample_size)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_rows, size=actual_size, replace=False).astype(np.int64))


def search_without_self(
    index: faiss.IndexFlatIP,
    query_vectors: np.ndarray,
    query_indices: np.ndarray,
    top_k: int,
) -> tuple[list[list[int]], list[list[float]]]:
    k_search = min(index.ntotal, max(top_k + 5, top_k + 1))
    scores, indices = index.search(np.ascontiguousarray(query_vectors.astype(np.float32)), k_search)
    all_neighbors: list[list[int]] = []
    all_scores: list[list[float]] = []
    for query_row, query_index in enumerate(query_indices.tolist()):
        neighbors: list[int] = []
        similarities: list[float] = []
        for match_index, score in zip(indices[query_row].tolist(), scores[query_row].tolist()):
            if match_index < 0 or match_index == query_index:
                continue
            neighbors.append(int(match_index))
            similarities.append(float(score))
            if len(neighbors) >= top_k:
                break
        all_neighbors.append(neighbors)
        all_scores.append(similarities)
    return all_neighbors, all_scores


def run_self_retrieval_check(
    index: faiss.IndexFlatIP,
    normalized_embeddings: np.ndarray,
    labels: np.ndarray,
    row_ids: list[str],
    label_names: list[str],
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    query_indices = sample_indices(normalized_embeddings.shape[0], sample_size, seed)
    query_vectors = normalized_embeddings[query_indices]
    k_search = min(index.ntotal, 5)
    scores, indices = index.search(np.ascontiguousarray(query_vectors.astype(np.float32)), k_search)

    top1 = indices[:, 0]
    self_hit = top1 == query_indices
    top5_contains_self = np.array(
        [query_index in row.tolist() for query_index, row in zip(query_indices.tolist(), indices)],
        dtype=bool,
    )

    miss_examples: list[dict[str, Any]] = []
    for row_index, hit in enumerate(self_hit.tolist()):
        if hit:
            continue
        query_index = int(query_indices[row_index])
        retrieved_index = int(top1[row_index])
        miss_examples.append(
            {
                "query_index": query_index,
                "query_row_id": row_ids[query_index],
                "query_labels": labels_to_names(labels[query_index], label_names),
                "retrieved_index": retrieved_index,
                "retrieved_row_id": row_ids[retrieved_index],
                "retrieved_labels": labels_to_names(labels[retrieved_index], label_names),
                "similarity": float(scores[row_index, 0]),
            }
        )
        if len(miss_examples) >= 10:
            break

    return {
        "sample_size": int(query_indices.shape[0]),
        "top1_self_hit_rate": float(self_hit.mean()),
        "top5_contains_self_rate": float(top5_contains_self.mean()),
        "top1_similarity_mean": float(scores[:, 0].mean()),
        "top1_similarity_min": float(scores[:, 0].min()),
        "miss_examples": miss_examples,
    }


def random_neighbors_excluding(query_index: int, num_rows: int, top_k: int, rng: random.Random) -> np.ndarray:
    picks: list[int] = []
    seen = {query_index}
    while len(picks) < min(top_k, num_rows - 1):
        value = rng.randrange(num_rows)
        if value in seen:
            continue
        seen.add(value)
        picks.append(value)
    return np.asarray(picks, dtype=np.int64)


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(a, b).sum()
    return float(intersection / union)


def run_label_agreement_check(
    index: faiss.IndexFlatIP,
    normalized_embeddings: np.ndarray,
    labels: np.ndarray,
    row_ids: list[str],
    label_names: list[str],
    sample_size: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    query_indices = sample_indices(normalized_embeddings.shape[0], sample_size, seed)
    query_vectors = normalized_embeddings[query_indices]
    neighbors, neighbor_scores = search_without_self(index, query_vectors, query_indices, top_k=top_k)

    global_prevalence = labels.mean(axis=0)
    rng = random.Random(seed)

    positive_prevalence_values: list[float] = []
    chance_prevalence_values: list[float] = []
    positive_jaccard_values: list[float] = []
    random_jaccard_values: list[float] = []
    per_label_neighbor_prevalence: dict[str, list[float]] = {name: [] for name in label_names}
    per_label_chance_prevalence: dict[str, list[float]] = {name: [] for name in label_names}
    query_summaries: list[dict[str, Any]] = []
    positive_query_count = 0

    for row_index, query_index in enumerate(query_indices.tolist()):
        neighbor_indices = neighbors[row_index]
        if not neighbor_indices:
            continue
        query_labels = labels[query_index] > 0.5
        positive_label_indices = np.flatnonzero(query_labels)
        if positive_label_indices.size == 0:
            continue
        positive_query_count += 1

        neighbor_label_matrix = labels[np.asarray(neighbor_indices, dtype=np.int64)] > 0.5
        prevalence_vector = neighbor_label_matrix.mean(axis=0)
        random_neighbor_indices = random_neighbors_excluding(query_index, labels.shape[0], top_k, rng)
        random_label_matrix = labels[random_neighbor_indices] > 0.5

        shared_positive_prevalence = prevalence_vector[positive_label_indices]
        chance_prevalence = global_prevalence[positive_label_indices]
        positive_prevalence_values.extend(shared_positive_prevalence.astype(np.float64).tolist())
        chance_prevalence_values.extend(chance_prevalence.astype(np.float64).tolist())

        for label_index in positive_label_indices.tolist():
            label_name = label_names[int(label_index)]
            per_label_neighbor_prevalence[label_name].append(float(prevalence_vector[label_index]))
            per_label_chance_prevalence[label_name].append(float(global_prevalence[label_index]))

        for neighbor_row in neighbor_label_matrix:
            positive_jaccard_values.append(jaccard_similarity(query_labels, neighbor_row))
        for random_row in random_label_matrix:
            random_jaccard_values.append(jaccard_similarity(query_labels, random_row))

        query_summaries.append(
            {
                "query_index": int(query_index),
                "query_row_id": row_ids[query_index],
                "query_labels": labels_to_names(labels[query_index], label_names),
                "neighbor_row_ids": [row_ids[index] for index in neighbor_indices],
                "neighbor_similarities": [float(score) for score in neighbor_scores[row_index]],
            }
        )

    per_label_summary: dict[str, Any] = {}
    for label_name in label_names:
        observed_values = per_label_neighbor_prevalence.get(label_name, [])
        chance_values = per_label_chance_prevalence.get(label_name, [])
        observed_mean = float(np.mean(observed_values)) if observed_values else None
        chance_mean = (
            float(np.mean(chance_values))
            if chance_values
            else float(global_prevalence[label_names.index(label_name)])
        )
        lift = None
        if observed_mean is not None and chance_mean > 0.0:
            lift = float(observed_mean / chance_mean)
        per_label_summary[label_name] = {
            "positive_queries": int(len(observed_values)),
            "mean_neighbor_prevalence": observed_mean,
            "chance_prevalence": chance_mean,
            "prevalence_lift": lift,
        }

    observed_prevalence = float(np.mean(positive_prevalence_values)) if positive_prevalence_values else None
    chance_prevalence = float(np.mean(chance_prevalence_values)) if chance_prevalence_values else None
    prevalence_lift = None
    if observed_prevalence is not None and chance_prevalence and chance_prevalence > 0.0:
        prevalence_lift = float(observed_prevalence / chance_prevalence)

    observed_jaccard = float(np.mean(positive_jaccard_values)) if positive_jaccard_values else None
    random_jaccard = float(np.mean(random_jaccard_values)) if random_jaccard_values else None
    jaccard_lift = None
    if observed_jaccard is not None and random_jaccard and random_jaccard > 0.0:
        jaccard_lift = float(observed_jaccard / random_jaccard)

    return {
        "num_queries": int(query_indices.shape[0]),
        "positive_query_count": int(positive_query_count),
        "top_k": int(top_k),
        "global_label_prevalence": {name: float(global_prevalence[index]) for index, name in enumerate(label_names)},
        "mean_query_positive_neighbor_prevalence": observed_prevalence,
        "chance_query_positive_prevalence": chance_prevalence,
        "positive_prevalence_lift": prevalence_lift,
        "mean_positive_jaccard": observed_jaccard,
        "mean_random_positive_jaccard": random_jaccard,
        "positive_jaccard_lift": jaccard_lift,
        "per_label": per_label_summary,
        "sample_queries": query_summaries[:10],
    }


def build_qualitative_neighbors(
    index: faiss.IndexFlatIP,
    normalized_embeddings: np.ndarray,
    labels: np.ndarray,
    rows: list[ManifestRecord],
    row_ids: list[str],
    image_paths: list[str],
    label_names: list[str],
    sample_size: int,
    top_k: int,
    seed: int,
) -> list[dict[str, Any]]:
    query_indices = sample_indices(normalized_embeddings.shape[0], sample_size, seed)
    query_vectors = normalized_embeddings[query_indices]
    neighbors, neighbor_scores = search_without_self(index, query_vectors, query_indices, top_k=top_k)

    samples: list[dict[str, Any]] = []
    for row_index, query_index in enumerate(query_indices.tolist()):
        query_row = rows[query_index]
        entry = {
            "query": {
                "index": int(query_index),
                "row_id": row_ids[query_index],
                "file_id": Path(image_paths[query_index]).name,
                "image_path": image_paths[query_index],
                "labels": labels_to_names(labels[query_index], label_names),
                "metadata": query_row.metadata,
            },
            "neighbors": [],
        }
        for rank, (neighbor_index, similarity) in enumerate(
            zip(neighbors[row_index], neighbor_scores[row_index]),
            start=1,
        ):
            neighbor_row = rows[neighbor_index]
            entry["neighbors"].append(
                {
                    "rank": int(rank),
                    "index": int(neighbor_index),
                    "row_id": row_ids[neighbor_index],
                    "file_id": Path(image_paths[neighbor_index]).name,
                    "image_path": image_paths[neighbor_index],
                    "similarity": float(similarity),
                    "labels": labels_to_names(labels[neighbor_index], label_names),
                    "metadata": neighbor_row.metadata,
                }
            )
        samples.append(entry)
    return samples


def write_items_jsonl(
    path: Path,
    rows: list[ManifestRecord],
    row_ids: list[str],
    image_paths: list[str],
    labels: np.ndarray,
    label_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, (row, row_id, image_path, label_row) in enumerate(zip(rows, row_ids, image_paths, labels)):
            payload = {
                "index": int(index),
                "row_id": row_id,
                "file_id": Path(image_path).name,
                "image_path": image_path,
                "labels": labels_to_names(label_row, label_names),
                "metadata": row.metadata,
            }
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


def script_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def human_size(num_bytes: int) -> str:
    units = ("B", "K", "M", "G", "T")
    value = float(num_bytes)
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)}{units[unit_index]}"
    return f"{value:.2f}{units[unit_index]}"


def list_hash_lines(paths: list[Path]) -> list[str]:
    return [f"{sha256_file(path)}  {path}" for path in paths if path.exists()]


def format_shell_command(argv: list[str]) -> str:
    return shlex.join(argv)


def build_recreation_report(
    *,
    experiment_dir: Path,
    experiment_id: str,
    operation_label: str,
    script_path: Path,
    argv_exact: list[str],
    argv_fresh: list[str],
    manifest_csv: Path,
    embedding_root: Path,
    baseline_experiment_dir: Path,
    split: str,
    row_count: int,
    embedding_dim: int,
    label_names: list[str],
    source_split: SourceSplit,
    experiment_meta: dict[str, Any],
    output_paths: list[Path],
) -> str:
    size_lines = [f"- {path.name}: `{human_size(path.stat().st_size)}`" for path in output_paths if path.exists()]
    hash_block = "\n".join(list_hash_lines(output_paths))
    total_size = sum(path.stat().st_size for path in output_paths if path.exists())
    run_meta = source_split.run_meta or {}
    raw_norm_summary = experiment_meta["embedding_sanity"]["raw_norm_summary"]
    normalized_norm_summary = experiment_meta["embedding_sanity"]["normalized_norm_summary"]
    self_retrieval = experiment_meta["sanity_checks"]["self_retrieval"]
    label_agreement = experiment_meta["sanity_checks"]["label_agreement"]
    faiss_version = getattr(faiss, "__version__", "unknown")

    lines = [
        "# Source Retrieval Memory Recreation Report",
        "",
        "## Scope",
        "",
        "This report documents how to recreate the source retrieval-memory experiment stored at:",
        "",
        f"`{experiment_dir}`",
        "",
        "The producing script is:",
        "",
        f"`{script_path}`",
        "",
        "Script SHA-256:",
        "",
        f"`{script_sha256(script_path)}`",
        "",
        "## Final Experiment Identity",
        "",
        f"- Experiment directory: `{experiment_dir}`",
        f"- Experiment id: `{experiment_id}`",
        f"- Operation label: `{operation_label}`",
        f"- Source embedding root: `{embedding_root}`",
        f"- Baseline reference experiment: `{baseline_experiment_dir}`",
        f"- Manifest: `{manifest_csv}`",
        f"- Split used for memory: `{split}`",
        f"- Memory granularity: `instance`",
        f"- Retrieval key family: `fused image+report embeddings`",
        f"- Stored rows: `{row_count:,}`",
        f"- Embedding dimension: `{embedding_dim}`",
        f"- Label count: `{len(label_names)}`",
        f"- Label names: `{' '.join(label_names)}`",
        f"- Index type: `faiss.IndexFlatIP`",
        f"- Similarity metric: `inner product on L2-normalized vectors`",
        "",
        "## Environment",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- NumPy: `{np.__version__}`",
        f"- faiss: `{faiss_version}`",
        f"- Platform: `{platform.platform()}`",
        "",
        "## Exact Recreation Command",
        "",
        "If you want to recreate the same directory name in place, use this command:",
        "",
        "```bash",
        format_shell_command(argv_exact),
        "```",
        "",
        "If you want a fresh numbered run instead of overwriting the existing directory, use:",
        "",
        "```bash",
        format_shell_command(argv_fresh),
        "```",
        "",
        "## Preconditions",
        "",
        f"- The fused embedding experiment must already exist at `{embedding_root}`.",
        f"- The selected source baseline should already exist at `{baseline_experiment_dir}`.",
        f"- The manifest must be present at `{manifest_csv}`.",
        f"- The source split `{split}` must contain `embeddings.npy`, a row-identity sidecar, and aligned image paths.",
        "- The required Python packages must be importable: `numpy`, `faiss`.",
        "",
        "## Input Summary",
        "",
        f"- Split directory: `{source_split.split_dir}`",
        f"- Source embeddings: `{source_split.embeddings_path}`",
        f"- Source sidecar: `{source_split.sidecar.relative_path}`",
        f"- Source sidecar parser: `{source_split.sidecar.parser}`",
        f"- Source rows: `{row_count:,}`",
        f"- Source embedding dim: `{embedding_dim}`",
        f"- Run-meta fusion mode: `{run_meta.get('fusion', 'unknown')}`",
        f"- Run-meta source order: `{' '.join(run_meta.get('source_order', []))}`",
        "",
        "## Expected Outputs",
        "",
        "- `.gitignore`",
        "- `experiment_meta.json`",
        "- `recreation_report.md`",
        "- `embeddings.npy`",
        "- `labels.npy`",
        "- `row_ids.json`",
        "- `image_paths.txt`",
        "- `items.jsonl`",
        "- `sanity_report.json`",
        "- `qualitative_neighbors.json`",
        "- `index.faiss`",
        "",
        "## Expected Counts And Sanity",
        "",
        f"- Memory rows: `{row_count:,}`",
        f"- Memory embedding dimension: `{embedding_dim}`",
        f"- Raw norm mean before defensive normalization: `{raw_norm_summary['mean']:.8f}`",
        f"- Post-normalization norm mean: `{normalized_norm_summary['mean']:.8f}`",
        f"- Self-retrieval top-1 hit rate on sample: `{self_retrieval['top1_self_hit_rate']:.6f}`",
        f"- Self-retrieval top-5 contains-self rate on sample: `{self_retrieval['top5_contains_self_rate']:.6f}`",
        f"- Positive prevalence lift in sampled label-agreement check: `{label_agreement['positive_prevalence_lift']:.6f}`",
        f"- Positive Jaccard lift in sampled label-agreement check: `{label_agreement['positive_jaccard_lift']:.6f}`",
        "",
        "## Output Sizes",
        "",
    ]
    lines.extend(size_lines)
    lines.extend(
        [
            f"- Total output size: `{human_size(total_size)}`",
            "",
            "## Final Artifact SHA-256",
            "",
            "```text",
            hash_block,
            "```",
            "",
            "## Important Reproduction Notes",
            "",
            "- `experiment_meta.json`, `sanity_report.json`, `qualitative_neighbors.json`, and `recreation_report.md` include timestamps or sampled summaries, so their hashes can change on rerun.",
            "- `embeddings.npy` and `index.faiss` are large duplicated retrieval artifacts and are intentionally ignored by the experiment-local `.gitignore` for plain Git commits.",
            "- `labels.npy`, `row_ids.json`, `image_paths.txt`, and `items.jsonl` are the main pushable provenance/value artifacts for this memory stage.",
            "- The script defensively re-normalizes the fused embeddings even though the source fused experiment already reports unit-length rows.",
            "- The FAISS index is frozen after build. No training or hyperparameter tuning happens in `exp0005`.",
            "",
            "## Agent Handoff Text",
            "",
            "```text",
            (
                "Use /workspace/scripts/05_build_source_retrieval_memory.py and the report "
                f"{experiment_dir / 'recreation_report.md'} to recreate the frozen NIH CXR14 source retrieval memory "
                f"from {embedding_root}. Build the train-only instance memory, verify the sanity_report metrics, and "
                "confirm that embeddings.npy and index.faiss are present locally even though they are excluded from plain Git."
            ),
            "```",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a train-time retrieval memory from frozen source-domain embeddings."
    )
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--baseline-experiment-dir", type=Path, default=DEFAULT_BASELINE_EXPERIMENT_DIR)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["train"])
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--self-retrieval-sample-size", type=int, default=DEFAULT_SELF_RETRIEVAL_SAMPLE_SIZE)
    parser.add_argument("--label-agreement-queries", type=int, default=DEFAULT_LABEL_AGREEMENT_QUERIES)
    parser.add_argument("--qualitative-queries", type=int, default=DEFAULT_QUALITATIVE_QUERIES)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    embedding_root = args.embedding_root.resolve()
    baseline_experiment_dir = args.baseline_experiment_dir.resolve()
    manifest_csv = args.manifest_csv.resolve()
    experiments_root = args.experiments_root.resolve()
    split = args.split

    generated_slug = "nih_cxr14_exp0003_fused_train_instance_memory_e100_p4"
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=args.overwrite,
    )

    script_path = Path(__file__).resolve()
    script_hash = script_sha256(script_path)

    label_columns, label_names, manifest_records = load_manifest_records(manifest_csv, split=split)
    source_split = load_source_split(embedding_root, split=split)
    validate_split_alignment(source_split, manifest_records)
    labels, rows, image_paths = build_labels_and_rows(source_split.row_ids, manifest_records)

    normalized_embeddings, raw_norm_summary, normalized_norm_summary = normalize_rows(source_split.embeddings)
    index = build_faiss_index(normalized_embeddings)

    self_retrieval = run_self_retrieval_check(
        index=index,
        normalized_embeddings=normalized_embeddings,
        labels=labels,
        row_ids=source_split.row_ids,
        label_names=label_names,
        sample_size=args.self_retrieval_sample_size,
        seed=args.seed,
    )
    label_agreement = run_label_agreement_check(
        index=index,
        normalized_embeddings=normalized_embeddings,
        labels=labels,
        row_ids=source_split.row_ids,
        label_names=label_names,
        sample_size=args.label_agreement_queries,
        top_k=args.top_k,
        seed=args.seed + 1,
    )
    qualitative_neighbors = build_qualitative_neighbors(
        index=index,
        normalized_embeddings=normalized_embeddings,
        labels=labels,
        rows=rows,
        row_ids=source_split.row_ids,
        image_paths=image_paths,
        label_names=label_names,
        sample_size=args.qualitative_queries,
        top_k=args.top_k,
        seed=args.seed + 2,
    )

    gitignore_path = experiment_dir / ".gitignore"
    embeddings_path = experiment_dir / "embeddings.npy"
    labels_path = experiment_dir / "labels.npy"
    row_ids_path = experiment_dir / "row_ids.json"
    image_paths_path = experiment_dir / "image_paths.txt"
    items_path = experiment_dir / "items.jsonl"
    sanity_path = experiment_dir / "sanity_report.json"
    qualitative_path = experiment_dir / "qualitative_neighbors.json"
    index_path = experiment_dir / "index.faiss"
    experiment_meta_path = experiment_dir / "experiment_meta.json"
    recreation_report_path = experiment_dir / "recreation_report.md"

    gitignore_path.write_text("/embeddings.npy\n/index.faiss\n", encoding="utf-8")
    np.save(embeddings_path, normalized_embeddings.astype(np.float32))
    np.save(labels_path, labels.astype(np.float32))
    row_ids_path.write_text(json.dumps(source_split.row_ids, indent=2), encoding="utf-8")
    image_paths_path.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
    write_items_jsonl(items_path, rows, source_split.row_ids, image_paths, labels, label_names)
    write_json(
        sanity_path,
        {
            "embedding_sanity": {
                "row_count": int(normalized_embeddings.shape[0]),
                "embedding_dim": int(normalized_embeddings.shape[1]),
                "no_nan_or_inf": bool(np.isfinite(normalized_embeddings).all()),
                "raw_norm_summary": raw_norm_summary,
                "normalized_norm_summary": normalized_norm_summary,
            },
            "self_retrieval": self_retrieval,
            "label_agreement": label_agreement,
            "qualitative_neighbors_path": str(qualitative_path),
        },
    )
    write_json(qualitative_path, qualitative_neighbors)
    faiss.write_index(index, str(index_path))

    baseline_meta_path = baseline_experiment_dir / "experiment_meta.json"
    baseline_meta = read_json(baseline_meta_path) if baseline_meta_path.exists() else None

    experiment_meta = {
        "argv": sys.argv,
        "baseline_experiment_dir": str(baseline_experiment_dir),
        "baseline_meta_path": str(baseline_meta_path),
        "baseline_reference": baseline_meta,
        "embedding_root": str(embedding_root),
        "experiment_dir": str(experiment_dir),
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_number": experiment_number,
        "label_columns": label_columns,
        "label_names": label_names,
        "manifest_csv": str(manifest_csv),
        "operation_label": DEFAULT_OPERATION_LABEL,
        "run_date_utc": utc_now_iso(),
        "script_path": str(script_path),
        "script_sha256": script_hash,
        "seed": args.seed,
        "split": split,
        "split_input": {
            "alignment_manifest_path": str(source_split.split_dir / "alignment_manifest.csv")
            if (source_split.split_dir / "alignment_manifest.csv").exists()
            else None,
            "embedding_dim": int(source_split.embeddings.shape[1]),
            "embeddings_path": str(source_split.embeddings_path),
            "image_paths_path": str(source_split.split_dir / "image_paths.txt")
            if (source_split.split_dir / "image_paths.txt").exists()
            else None,
            "num_rows": int(source_split.embeddings.shape[0]),
            "row_id_sidecar": {
                "column": source_split.sidecar.column,
                "format": source_split.sidecar.format,
                "parser": source_split.sidecar.parser,
                "relative_path": source_split.sidecar.relative_path,
            },
            "run_meta_path": str(source_split.split_dir / "run_meta.json")
            if (source_split.split_dir / "run_meta.json").exists()
            else None,
        },
        "memory": {
            "granularity": "instance",
            "row_count": int(normalized_embeddings.shape[0]),
            "embedding_dim": int(normalized_embeddings.shape[1]),
            "value_type": "multilabel_float32",
        },
        "index": {
            "metric": "inner_product_on_l2_normalized_vectors",
            "ntotal": int(index.ntotal),
            "type": "faiss.IndexFlatIP",
        },
        "artifacts": {
            "gitignore": str(gitignore_path),
            "embeddings": str(embeddings_path),
            "labels": str(labels_path),
            "row_ids": str(row_ids_path),
            "image_paths": str(image_paths_path),
            "items": str(items_path),
            "sanity_report": str(sanity_path),
            "qualitative_neighbors": str(qualitative_path),
            "index": str(index_path),
            "recreation_report": str(recreation_report_path),
        },
        "embedding_sanity": {
            "raw_norm_summary": raw_norm_summary,
            "normalized_norm_summary": normalized_norm_summary,
            "source_embeddings_already_near_unit_norm": bool(abs(raw_norm_summary["mean"] - 1.0) < 1e-3),
        },
        "sanity_checks": {
            "self_retrieval": self_retrieval,
            "label_agreement": label_agreement,
        },
        "source_run_meta": source_split.run_meta,
    }
    write_json(experiment_meta_path, experiment_meta)

    argv_exact = [
        "python",
        str(script_path),
        "--embedding-root",
        str(embedding_root),
        "--baseline-experiment-dir",
        str(baseline_experiment_dir),
        "--manifest-csv",
        str(manifest_csv),
        "--split",
        split,
        "--self-retrieval-sample-size",
        str(args.self_retrieval_sample_size),
        "--label-agreement-queries",
        str(args.label_agreement_queries),
        "--qualitative-queries",
        str(args.qualitative_queries),
        "--top-k",
        str(args.top_k),
        "--seed",
        str(args.seed),
        "--experiment-name",
        experiment_name,
        "--overwrite",
    ]
    argv_fresh = [
        "python",
        str(script_path),
        "--embedding-root",
        str(embedding_root),
        "--baseline-experiment-dir",
        str(baseline_experiment_dir),
        "--manifest-csv",
        str(manifest_csv),
        "--split",
        split,
        "--self-retrieval-sample-size",
        str(args.self_retrieval_sample_size),
        "--label-agreement-queries",
        str(args.label_agreement_queries),
        "--qualitative-queries",
        str(args.qualitative_queries),
        "--top-k",
        str(args.top_k),
        "--seed",
        str(args.seed),
        "--experiment-name",
        strip_experiment_number_prefix(experiment_name),
    ]
    output_paths = [
        gitignore_path,
        experiment_meta_path,
        recreation_report_path,
        embeddings_path,
        labels_path,
        row_ids_path,
        image_paths_path,
        items_path,
        sanity_path,
        qualitative_path,
        index_path,
    ]
    recreation_report = build_recreation_report(
        experiment_dir=experiment_dir,
        experiment_id=experiment_id,
        operation_label=DEFAULT_OPERATION_LABEL,
        script_path=script_path,
        argv_exact=argv_exact,
        argv_fresh=argv_fresh,
        manifest_csv=manifest_csv,
        embedding_root=embedding_root,
        baseline_experiment_dir=baseline_experiment_dir,
        split=split,
        row_count=int(normalized_embeddings.shape[0]),
        embedding_dim=int(normalized_embeddings.shape[1]),
        label_names=label_names,
        source_split=source_split,
        experiment_meta=experiment_meta,
        output_paths=output_paths,
    )
    recreation_report_path.write_text(recreation_report + "\n", encoding="utf-8")

    print(f"[saved] experiment_dir={experiment_dir}")
    print(
        "[embedding_sanity] "
        f"rows={normalized_embeddings.shape[0]} dim={normalized_embeddings.shape[1]} "
        f"raw_norm_mean={raw_norm_summary['mean']:.8f} "
        f"post_norm_mean={normalized_norm_summary['mean']:.8f}"
    )
    print(
        "[self_retrieval] "
        f"sample_size={self_retrieval['sample_size']} "
        f"top1_self_hit_rate={self_retrieval['top1_self_hit_rate']:.6f} "
        f"top5_contains_self_rate={self_retrieval['top5_contains_self_rate']:.6f}"
    )
    print(
        "[label_agreement] "
        f"queries={label_agreement['num_queries']} "
        f"positive_queries={label_agreement['positive_query_count']} "
        f"top_k={label_agreement['top_k']} "
        f"positive_prevalence_lift={label_agreement['positive_prevalence_lift']:.6f} "
        f"positive_jaccard_lift={label_agreement['positive_jaccard_lift']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
