#!/usr/bin/env python3
"""Evaluate validation-only memory probabilities from a frozen source retrieval memory."""

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
        "/workspace/scripts/06_evaluate_source_memory_only.py."
    ) from exc

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'numpy'.") from exc


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_nih_cxr14_all14.csv")
DEFAULT_MEMORY_ROOT = Path(
    "/workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory"
)
DEFAULT_QUERY_EMBEDDING_ROOT = Path(
    "/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2"
)
DEFAULT_BASELINE_EXPERIMENT_DIR = Path(
    "/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear"
)
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_OPERATION_LABEL = "source_memory_only_evaluation"
DEFAULT_EXPERIMENT_ID_WIDTH = 4
DEFAULT_SPLIT = "val"
DEFAULT_QUALITATIVE_QUERIES = 10
DEFAULT_SEED = 3407
DEFAULT_SELECTION_METRIC = "macro_auroc"
DEFAULT_ECE_BINS = 15
SWEEP_K_VALUES = [1, 3, 5, 10, 20, 50]
SWEEP_TAU_VALUES = [1, 5, 10, 20, 40]


@dataclass(frozen=True)
class ManifestRecord:
    row_id: str
    image_path: str
    labels: tuple[float, ...]


@dataclass(frozen=True)
class SidecarSpec:
    relative_path: str
    format: str
    parser: str
    column: str | None = None


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


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Text file not found: {path}")
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise SystemExit(f"Text file is empty: {path}")
    return values


def read_sidecar_items(path: Path, *, format_name: str, column: str | None) -> list[str]:
    if format_name == "lines":
        return read_lines(path)
    if format_name == "json_list":
        payload = read_json(path)
        if not isinstance(payload, list):
            raise SystemExit(f"Expected JSON list in sidecar: {path}")
        values = [str(item).strip() for item in payload if str(item).strip()]
        if not values:
            raise SystemExit(f"JSON sidecar is empty: {path}")
        return values
    if format_name == "csv_column":
        text = path.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None or not column or column not in reader.fieldnames:
            raise SystemExit(f"CSV sidecar {path} does not contain column '{column}'.")
        values: list[str] = []
        for row in reader:
            value = row.get(column)
            if value is None:
                raise SystemExit(f"CSV sidecar {path} contains a row without column '{column}'.")
            cleaned = str(value).strip()
            if cleaned:
                values.append(cleaned)
        if not values:
            raise SystemExit(f"CSV sidecar is empty: {path}")
        return values
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


def autodetect_sidecar(split_dir: Path) -> SidecarSpec:
    preferred_exact = [
        SidecarSpec(relative_path="row_ids.json", format="json_list", parser="identity"),
        SidecarSpec(relative_path="image_paths.txt", format="lines", parser="stem"),
        SidecarSpec(relative_path="report_ids.json", format="json_list", parser="identity"),
    ]
    for candidate in preferred_exact:
        if (split_dir / candidate.relative_path).exists():
            return candidate
    raise SystemExit(f"Could not auto-detect a row-identity sidecar in {split_dir}.")


def load_embedding_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"Array file not found: {path}")
    array = np.load(path)
    if array.ndim != 2:
        raise SystemExit(f"Expected 2D array at {path}, found shape {array.shape}.")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise SystemExit(f"Array contains NaN or inf values: {path}")
    return np.ascontiguousarray(array)


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
    if int(np.count_nonzero(raw_norms <= 0.0)) > 0:
        raise SystemExit("Found zero-norm embeddings; cannot normalize.")
    normalized = embeddings / raw_norms[:, None].astype(np.float32)
    normalized = np.ascontiguousarray(normalized.astype(np.float32))
    if not np.isfinite(normalized).all():
        raise SystemExit("Normalized embeddings contain NaN or inf values.")
    normalized_norms = np.linalg.norm(normalized.astype(np.float64), axis=1)
    return normalized, summarize_values(raw_norms), summarize_values(normalized_norms)


def load_manifest_records(
    manifest_csv: Path,
    *,
    split: str,
) -> tuple[list[str], list[str], dict[str, ManifestRecord]]:
    text = manifest_csv.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise SystemExit(f"Manifest CSV has no header: {manifest_csv}")
    required = {"dataset", "split", "image_path"}
    if not required.issubset(reader.fieldnames):
        raise SystemExit(f"Manifest CSV must contain columns: {sorted(required)}")
    label_columns = [field for field in reader.fieldnames if field.startswith("label_")]
    if not label_columns:
        raise SystemExit("Manifest CSV does not contain any label_... columns.")
    label_names = [column.removeprefix("label_") for column in label_columns]

    records: dict[str, ManifestRecord] = {}
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
        if row_id in records:
            raise SystemExit(f"Duplicate manifest row_id '{row_id}' found in split '{split}'.")
        records[row_id] = ManifestRecord(row_id=row_id, image_path=image_path, labels=tuple(labels))
    if not records:
        raise SystemExit(f"Manifest contains no rows for split '{split}'.")
    return label_columns, label_names, records


def load_query_split(query_embedding_root: Path, *, split: str) -> tuple[np.ndarray, list[str], list[str], dict[str, Any] | None]:
    split_dir = query_embedding_root / split
    if not split_dir.exists():
        raise SystemExit(f"Split directory not found for '{split}': {split_dir}")
    embeddings = load_embedding_array(split_dir / "embeddings.npy")
    sidecar = autodetect_sidecar(split_dir)
    sidecar_items = read_sidecar_items(split_dir / sidecar.relative_path, format_name=sidecar.format, column=sidecar.column)
    if len(sidecar_items) != int(embeddings.shape[0]):
        raise SystemExit(
            f"Split '{split}' has {embeddings.shape[0]} embedding rows but {len(sidecar_items)} sidecar rows."
        )
    row_ids = [parse_row_id(item, sidecar.parser) for item in sidecar_items]
    image_paths_path = split_dir / "image_paths.txt"
    image_paths = read_lines(image_paths_path) if image_paths_path.exists() else []
    if image_paths and len(image_paths) != len(row_ids):
        raise SystemExit(
            f"Split '{split}' has {len(row_ids)} row IDs but {len(image_paths)} image paths in {image_paths_path}."
        )
    run_meta_path = split_dir / "run_meta.json"
    run_meta = read_json(run_meta_path) if run_meta_path.exists() else None
    return embeddings, row_ids, image_paths, run_meta


def build_labels_from_records(row_ids: list[str], records: dict[str, ManifestRecord]) -> tuple[np.ndarray, list[str]]:
    labels = np.zeros((len(row_ids), len(next(iter(records.values())).labels)), dtype=np.float32)
    image_paths: list[str] = []
    missing: list[str] = []
    for index, row_id in enumerate(row_ids):
        record = records.get(row_id)
        if record is None:
            if len(missing) < 5:
                missing.append(row_id)
            continue
        labels[index] = np.asarray(record.labels, dtype=np.float32)
        image_paths.append(record.image_path)
    if missing:
        raise SystemExit(f"Query split contains row IDs missing from manifest. Examples: {missing}")
    return labels, image_paths


def validate_query_alignment(query_row_ids: list[str], sidecar_image_paths: list[str], manifest_image_paths: list[str]) -> None:
    if sidecar_image_paths and len(sidecar_image_paths) != len(manifest_image_paths):
        raise SystemExit("Validation image-path alignment failed due to row-count mismatch.")
    for index, row_id in enumerate(query_row_ids):
        expected = Path(manifest_image_paths[index]).stem
        if expected != row_id:
            raise SystemExit(
                f"Validation row-id mismatch at row {index}: manifest has '{expected}', embeddings have '{row_id}'."
            )
        if sidecar_image_paths:
            if Path(sidecar_image_paths[index]).name != Path(manifest_image_paths[index]).name:
                raise SystemExit(
                    f"Validation image path mismatch at row {index}: "
                    f"manifest has '{manifest_image_paths[index]}', sidecar has '{sidecar_image_paths[index]}'."
                )


def load_faiss_index(index_path: Path, train_embeddings: np.ndarray | None) -> tuple[faiss.Index, dict[str, Any]]:
    details: dict[str, Any] = {
        "index_path": str(index_path),
        "loaded_from_disk": False,
        "rebuilt_from_embeddings": False,
        "load_error": None,
    }
    if index_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            details["loaded_from_disk"] = True
            return index, details
        except Exception as exc:  # pragma: no cover
            details["load_error"] = f"{type(exc).__name__}: {exc}"
    if train_embeddings is None:
        raise SystemExit("FAISS index is unavailable and no train embeddings were provided to rebuild it.")
    index = faiss.IndexFlatIP(int(train_embeddings.shape[1]))
    index.add(np.ascontiguousarray(train_embeddings.astype(np.float32)))
    details["rebuilt_from_embeddings"] = True
    return index, details


def load_memory_embeddings(memory_root: Path, memory_meta: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    memory_embeddings_path = memory_root / "embeddings.npy"
    if memory_embeddings_path.exists():
        return load_embedding_array(memory_embeddings_path), {"source": str(memory_embeddings_path), "kind": "memory_artifact"}

    fallback_path = Path(
        str(memory_meta["baseline_reference"]["split_inputs"]["train"]["embeddings_path"])
        if memory_meta.get("baseline_reference")
        else ""
    )
    if fallback_path.exists():
        return load_embedding_array(fallback_path), {"source": str(fallback_path), "kind": "baseline_reference_train_embeddings"}

    source_meta = memory_meta.get("source_run_meta") or {}
    if isinstance(source_meta, dict):
        for source in source_meta.get("sources", []):
            pass
    raise SystemExit(
        "Could not locate train embeddings for the memory. Expected either "
        f"{memory_embeddings_path} or a valid fallback embedding path in exp0008 metadata."
    )


def check_memory_consistency(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    train_row_ids: list[str],
    train_image_paths: list[str],
) -> None:
    row_count = int(train_embeddings.shape[0])
    if train_labels.shape[0] != row_count:
        raise SystemExit(f"Train labels row count {train_labels.shape[0]} does not match train embeddings {row_count}.")
    if len(train_row_ids) != row_count:
        raise SystemExit(f"Train row_id count {len(train_row_ids)} does not match train embeddings {row_count}.")
    if len(train_image_paths) != row_count:
        raise SystemExit(f"Train image_path count {len(train_image_paths)} does not match train embeddings {row_count}.")
    for index, (row_id, image_path) in enumerate(zip(train_row_ids, train_image_paths)):
        if Path(image_path).stem != row_id:
            raise SystemExit(
                f"Train row-id mismatch at row {index}: row_ids has '{row_id}', image_paths has '{Path(image_path).stem}'."
            )


def compute_memory_probabilities(
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    train_labels: np.ndarray,
    *,
    k: int,
    tau: float,
) -> np.ndarray:
    sliced_indices = np.ascontiguousarray(neighbor_indices[:, :k], dtype=np.int64)
    sliced_scores = np.ascontiguousarray(neighbor_scores[:, :k], dtype=np.float32)
    scaled_scores = sliced_scores * float(tau)
    scaled_scores -= scaled_scores.max(axis=1, keepdims=True)
    weights = np.exp(scaled_scores.astype(np.float64))
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
    neighbor_label_matrix = train_labels[sliced_indices]
    probabilities = np.sum(weights[:, :, None] * neighbor_label_matrix.astype(np.float64), axis=1)
    probabilities = np.ascontiguousarray(probabilities.astype(np.float32))
    if not np.isfinite(probabilities).all():
        raise SystemExit(f"Computed memory probabilities contain NaN or inf for k={k}, tau={tau}.")
    if probabilities.min() < -1e-6 or probabilities.max() > 1.0 + 1e-6:
        raise SystemExit(
            f"Memory probabilities are out of range for k={k}, tau={tau}: "
            f"min={probabilities.min()}, max={probabilities.max()}."
        )
    return probabilities


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(y.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    total = sorted_scores.shape[0]
    while start < total:
        end = start + 1
        while end < total and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[start:end] = average_rank
        start = end
    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks
    sum_ranks_pos = float(full_ranks[y == 1].sum())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def binary_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.int64)
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, y_sorted.shape[0] + 1, dtype=np.float64)
    ap = float(precision[y_sorted == 1].sum() / positives)
    return ap


def binary_f1(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    preds = (np.asarray(probs, dtype=np.float64) >= float(threshold)).astype(np.int64)
    tp = float(np.sum((preds == 1) & (y == 1)))
    fp = float(np.sum((preds == 1) & (y == 0)))
    fn = float(np.sum((preds == 0) & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    return float((2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0)


def binary_ece(y_true: np.ndarray, probs: np.ndarray, num_bins: int = DEFAULT_ECE_BINS) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total = float(p.shape[0])
    for idx in range(num_bins):
        left = bins[idx]
        right = bins[idx + 1]
        if idx == num_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(p[mask].mean())
        empirical = float(y[mask].mean())
        ece += (count / total) * abs(confidence - empirical)
    return float(ece)


def tune_thresholds(y_true: np.ndarray, probs: np.ndarray, label_names: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    thresholds = np.full((len(label_names),), 0.5, dtype=np.float32)
    payload: dict[str, Any] = {}
    for idx, label_name in enumerate(label_names):
        y = np.asarray(y_true[:, idx], dtype=np.int64)
        p = np.asarray(probs[:, idx], dtype=np.float64)
        positives = int(y.sum())
        if positives == 0:
            payload[label_name] = {
                "threshold": 0.5,
                "best_f1": 0.0,
                "prevalence": float(y.mean()),
                "reason": "no_positive_examples_in_val",
            }
            continue

        order = np.argsort(-p, kind="mergesort")
        y_sorted = y[order]
        p_sorted = p[order]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        positives_float = float(positives)
        fn = positives_float - tp
        precision = tp / np.clip(tp + fp, 1.0, None)
        recall = tp / positives_float
        f1 = (2.0 * precision * recall) / np.clip(precision + recall, 1e-12, None)
        best_value = float(np.max(f1))
        best_indices = np.flatnonzero(np.isclose(f1, best_value, atol=1e-12, rtol=0.0))
        best_threshold = float(np.min(p_sorted[best_indices]))
        thresholds[idx] = best_threshold
        payload[label_name] = {
            "threshold": best_threshold,
            "best_f1": best_value,
            "prevalence": float(y.mean()),
        }
    return thresholds, payload


def evaluate_probabilities(
    y_true: np.ndarray,
    probs: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    per_label: dict[str, dict[str, Any]] = {}
    macro_auroc_values: list[float] = []
    macro_ap_values: list[float] = []
    macro_f1_values: list[float] = []
    macro_ece_values: list[float] = []

    thresholds, threshold_payload = tune_thresholds(y_true, probs, label_names)
    macro_f1_tuned_values: list[float] = []

    for idx, label_name in enumerate(label_names):
        targets = y_true[:, idx]
        scores = probs[:, idx]
        auroc = binary_auroc(targets, scores)
        average_precision = binary_average_precision(targets, scores)
        f1_at_0p5 = binary_f1(targets, scores, 0.5)
        f1_at_tuned = binary_f1(targets, scores, float(thresholds[idx]))
        ece = binary_ece(targets, scores)

        if auroc is not None:
            macro_auroc_values.append(auroc)
        if average_precision is not None:
            macro_ap_values.append(average_precision)
        macro_f1_values.append(f1_at_0p5)
        macro_f1_tuned_values.append(f1_at_tuned)
        macro_ece_values.append(ece)

        per_label[label_name] = {
            "auroc": auroc,
            "average_precision": average_precision,
            "ece": ece,
            "f1_at_0.5": f1_at_0p5,
            "f1_at_tuned_threshold": f1_at_tuned,
            "threshold": float(thresholds[idx]),
            "positive_count": int(targets.sum()),
        }

    return {
        "macro_auroc": float(np.mean(macro_auroc_values)) if macro_auroc_values else None,
        "macro_average_precision": float(np.mean(macro_ap_values)) if macro_ap_values else None,
        "macro_ece": float(np.mean(macro_ece_values)) if macro_ece_values else None,
        "macro_f1_at_0.5": float(np.mean(macro_f1_values)) if macro_f1_values else None,
        "diagnostic_macro_f1_at_tuned_thresholds": (
            float(np.mean(macro_f1_tuned_values)) if macro_f1_tuned_values else None
        ),
        "thresholds": threshold_payload,
        "per_label": per_label,
    }


def select_best_sweep_row(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise SystemExit("Sweep results are empty; cannot select best configuration.")

    def metric_value(row: dict[str, Any], key: str) -> float:
        value = row.get(key)
        return float("-inf") if value is None else float(value)

    max_auroc = max(metric_value(row, "macro_auroc") for row in rows)
    auroc_candidates = [
        row for row in rows if np.isclose(metric_value(row, "macro_auroc"), max_auroc, atol=1e-12, rtol=0.0)
    ]

    max_ap = max(metric_value(row, "macro_average_precision") for row in auroc_candidates)
    ap_candidates = [
        row
        for row in auroc_candidates
        if np.isclose(metric_value(row, "macro_average_precision"), max_ap, atol=1e-12, rtol=0.0)
    ]

    min_k = min(int(row["k"]) for row in ap_candidates)
    k_candidates = [row for row in ap_candidates if int(row["k"]) == min_k]
    tau_values = [int(row["tau"]) for row in k_candidates]
    selected_tau = 10 if 10 in tau_values else min(tau_values)
    final_candidates = [row for row in k_candidates if int(row["tau"]) == selected_tau]
    best_row = final_candidates[0]

    trace = {
        "max_macro_auroc": max_auroc,
        "macro_auroc_tied_candidates": len(auroc_candidates),
        "max_macro_average_precision_within_auroc_ties": max_ap,
        "macro_average_precision_tied_candidates": len(ap_candidates),
        "smallest_k_within_metric_ties": min_k,
        "k_tied_candidates": len(k_candidates),
        "selected_tau_after_k_tie_break": selected_tau,
        "final_candidates": len(final_candidates),
        "tie_break_rule": [
            "highest validation macro AUROC",
            "higher validation macro average precision",
            "smaller k",
            "tau = 10 if among tied settings, otherwise smaller tau",
        ],
    }
    return best_row, trace


def choose_qualitative_query_indices(labels: np.ndarray, count: int) -> list[tuple[int, str]]:
    label_counts = labels.sum(axis=1)
    negative_indices = np.flatnonzero(label_counts == 0)
    single_positive_indices = np.flatnonzero(label_counts == 1)
    multi_positive_indices = np.flatnonzero(label_counts >= 2)

    def spread_select(indices: np.ndarray, desired: int) -> list[int]:
        if desired <= 0 or indices.size == 0:
            return []
        if indices.size <= desired:
            return indices.astype(np.int64).tolist()
        positions = np.linspace(0, indices.size - 1, num=desired)
        chosen = np.unique(np.rint(positions).astype(np.int64))
        return indices[chosen].astype(np.int64).tolist()[:desired]

    selected: list[tuple[int, str]] = []
    seen: set[int] = set()
    desired_counts = [
        ("negative_or_unlabeled", negative_indices, 3),
        ("single_positive", single_positive_indices, 3),
        ("multi_positive", multi_positive_indices, count - 6),
    ]
    for category_name, candidates, desired in desired_counts:
        for query_index in spread_select(candidates, desired):
            if query_index in seen:
                continue
            selected.append((query_index, category_name))
            seen.add(query_index)
    if len(selected) < count:
        for query_index in np.arange(labels.shape[0], dtype=np.int64).tolist():
            if query_index in seen:
                continue
            category_name = (
                "negative_or_unlabeled"
                if int(label_counts[query_index]) == 0
                else "single_positive"
                if int(label_counts[query_index]) == 1
                else "multi_positive"
            )
            selected.append((int(query_index), category_name))
            seen.add(int(query_index))
            if len(selected) >= count:
                break
    return selected[:count]


def labels_to_names(label_row: np.ndarray, label_names: list[str]) -> list[str]:
    indices = np.flatnonzero(label_row > 0.5)
    return [label_names[int(index)] for index in indices.tolist()]


def infer_observation(query_labels: list[str], retrieved_label_lists: list[list[str]]) -> str:
    query_label_set = set(query_labels)
    retrieved_sets = [set(labels) for labels in retrieved_label_lists]
    overlap_count = sum(bool(query_label_set & labels) for labels in retrieved_sets)
    positive_neighbor_count = sum(bool(labels) for labels in retrieved_sets)

    if not query_label_set:
        if positive_neighbor_count <= 1:
            return "mostly negative neighbors"
        if positive_neighbor_count <= 2:
            return "partial clinical overlap"
        return "failure / suspicious retrieval"
    if overlap_count >= max(3, len(retrieved_sets) // 2 + 1):
        return "strong label match"
    if overlap_count >= 1:
        return "partial clinical overlap"
    if positive_neighbor_count == 0:
        return "mostly negative neighbors"
    return "failure / suspicious retrieval"


def build_qualitative_neighbors(
    selected_queries: list[tuple[int, str]],
    val_row_ids: list[str],
    val_image_paths: list[str],
    val_labels: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    train_row_ids: list[str],
    train_labels: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for query_index, category_name in selected_queries:
        query_label_names = labels_to_names(val_labels[query_index], label_names)
        current_neighbor_indices = neighbor_indices[query_index].astype(np.int64).tolist()
        current_neighbor_scores = neighbor_scores[query_index].astype(np.float64).tolist()
        neighbor_label_lists = [labels_to_names(train_labels[index], label_names) for index in current_neighbor_indices]
        payload.append(
            {
                "query_category": category_name,
                "validation_row_id": val_row_ids[query_index],
                "validation_image_path": val_image_paths[query_index],
                "true_positive_labels": query_label_names,
                "top_k_retrieved_train_row_ids": [train_row_ids[index] for index in current_neighbor_indices],
                "top_k_similarities": [float(score) for score in current_neighbor_scores],
                "top_k_retrieved_train_labels": neighbor_label_lists,
                "observation": infer_observation(query_label_names, neighbor_label_lists),
            }
        )
    return payload


def format_shell_command(argv: list[str]) -> str:
    return shlex.join(argv)


def build_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_metric(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def build_recreation_report(
    *,
    experiment_dir: Path,
    experiment_id: str,
    operation_label: str,
    script_path: Path,
    argv_exact: list[str],
    argv_fresh: list[str],
    memory_root: Path,
    query_embedding_root: Path,
    baseline_experiment_dir: Path,
    manifest_csv: Path,
    split: str,
    row_count: int,
    label_names: list[str],
    memory_rows: int,
    memory_dim: int,
    query_dim: int,
    norm_summary_before: dict[str, float],
    norm_summary_after: dict[str, float],
    index_details: dict[str, Any],
    best_row: dict[str, Any],
    best_metrics: dict[str, Any],
    sweep_rows: list[dict[str, Any]],
    output_paths: list[Path],
) -> str:
    size_lines = [f"- {path.name}: `{human_size(path.stat().st_size)}`" for path in output_paths if path.exists()]
    total_size = sum(path.stat().st_size for path in output_paths if path.exists())
    hash_lines = [f"{sha256_file(path)}  {path}" for path in output_paths if path.exists()]
    sweep_table = build_markdown_table(
        ["k", "tau", "Macro AUROC", "Macro AP", "Macro ECE", "Macro F1 @ 0.5", "Best"],
        [
            [
                str(int(row["k"])),
                str(int(row["tau"])),
                format_metric(row.get("macro_auroc")),
                format_metric(row.get("macro_average_precision")),
                format_metric(row.get("macro_ece")),
                format_metric(row.get("macro_f1_at_0.5")),
                "<- best" if int(row["k"]) == int(best_row["k"]) and int(row["tau"]) == int(best_row["tau"]) else "",
            ]
            for row in sweep_rows
        ],
    )
    faiss_version = getattr(faiss, "__version__", "unknown")
    lines = [
        "# Source Memory-Only Evaluation Recreation Report",
        "",
        "## Scope",
        "",
        "This report documents how to recreate the validation-only memory evaluation experiment stored at:",
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
        f"- Memory root: `{memory_root}`",
        f"- Query embedding root: `{query_embedding_root}`",
        f"- Baseline reference experiment: `{baseline_experiment_dir}`",
        f"- Manifest: `{manifest_csv}`",
        f"- Evaluation split: `{split}`",
        f"- Validation query rows: `{row_count:,}`",
        f"- Train memory rows: `{memory_rows:,}`",
        f"- Query embedding dimension: `{query_dim}`",
        f"- Memory embedding dimension: `{memory_dim}`",
        f"- Label count: `{len(label_names)}`",
        f"- Label names: `{' '.join(label_names)}`",
        f"- Selection metric: `{DEFAULT_SELECTION_METRIC}`",
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
        f"- The memory experiment must already exist at `{memory_root}`.",
        f"- The query embeddings must already exist at `{query_embedding_root / split}`.",
        f"- The manifest must be present at `{manifest_csv}`.",
        "- The required Python packages must be importable: `numpy`, `faiss`.",
        "- If the `exp0008` local FAISS index is missing, this script can rebuild it from the local train embeddings.",
        "",
        "## Input Summary",
        "",
        f"- Query split directory: `{query_embedding_root / split}`",
        f"- Query rows: `{row_count:,}`",
        f"- Query embedding dim: `{query_dim}`",
        f"- Memory rows: `{memory_rows:,}`",
        f"- Memory embedding dim: `{memory_dim}`",
        f"- Index loaded from disk: `{str(index_details['loaded_from_disk']).lower()}`",
        f"- Index rebuilt from embeddings: `{str(index_details['rebuilt_from_embeddings']).lower()}`",
        "",
        "## Sweep Summary",
        "",
        sweep_table,
        "",
        "## Best Configuration",
        "",
        f"- Best `k`: `{int(best_row['k'])}`",
        f"- Best `tau`: `{int(best_row['tau'])}`",
        f"- Validation macro AUROC: `{format_metric(best_metrics['macro_auroc'])}`",
        f"- Validation macro average precision: `{format_metric(best_metrics['macro_average_precision'])}`",
        f"- Validation macro ECE: `{format_metric(best_metrics['macro_ece'])}`",
        f"- Validation macro F1 @ 0.5: `{format_metric(best_metrics['macro_f1_at_0.5'])}`",
        f"- Diagnostic macro F1 @ tuned thresholds: `{format_metric(best_metrics['diagnostic_macro_f1_at_tuned_thresholds'])}`",
        "",
        "## Query Normalization",
        "",
        f"- Raw norm mean: `{norm_summary_before['mean']:.8f}`",
        f"- Post-normalization norm mean: `{norm_summary_after['mean']:.8f}`",
        "",
        "## Expected Outputs",
        "",
        "- `experiment_meta.json`",
        "- `recreation_report.md`",
        "- `sweep_results.json`",
        "- `best_config.json`",
        "- `best_val_metrics.json`",
        "- `val_probabilities.npy`",
        "- `qualitative_neighbors.json`",
        "- `memory_only_selection.md`",
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
            "\n".join(hash_lines),
            "```",
            "",
            "## Important Reproduction Notes",
            "",
            "- All configuration selection in `exp0009` is validation-only.",
            "- `val_probabilities.npy` contains the best-config validation probabilities and is small enough for plain Git.",
            "- If the local `exp0008/index.faiss` is unavailable, rerun `exp0008` or keep the local train embeddings available so the index can be rebuilt.",
            "- Threshold-tuned F1 is recorded as diagnostic only because thresholds are chosen on the same validation split.",
            "",
            "## Agent Handoff Text",
            "",
            "```text",
            (
                "Use /workspace/scripts/06_evaluate_source_memory_only.py and the report "
                f"{experiment_dir / 'recreation_report.md'} to recreate the validation-only memory evaluation for "
                f"{memory_root}. Sweep k and tau on the val split, select the best config by macro AUROC, and verify "
                "the saved best_config.json, best_val_metrics.json, and val_probabilities.npy artifacts."
            ),
            "```",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen train memory on validation queries only.")
    parser.add_argument("--memory-root", type=Path, default=DEFAULT_MEMORY_ROOT)
    parser.add_argument("--query-embedding-root", type=Path, default=DEFAULT_QUERY_EMBEDDING_ROOT)
    parser.add_argument("--baseline-experiment-dir", type=Path, default=DEFAULT_BASELINE_EXPERIMENT_DIR)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["val"])
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--qualitative-queries", type=int, default=DEFAULT_QUALITATIVE_QUERIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    memory_root = args.memory_root.resolve()
    query_embedding_root = args.query_embedding_root.resolve()
    baseline_experiment_dir = args.baseline_experiment_dir.resolve()
    manifest_csv = args.manifest_csv.resolve()
    experiments_root = args.experiments_root.resolve()
    split = args.split

    generated_slug = "nih_cxr14_exp0008_val"
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=args.overwrite,
    )

    script_path = Path(__file__).resolve()
    script_hash = script_sha256(script_path)

    memory_meta = read_json(memory_root / "experiment_meta.json")
    train_labels = load_embedding_array(memory_root / "labels.npy")
    train_row_ids = read_json(memory_root / "row_ids.json")
    if not isinstance(train_row_ids, list) or not train_row_ids:
        raise SystemExit(f"Invalid row_ids.json in {memory_root}")
    train_row_ids = [str(item) for item in train_row_ids]
    train_image_paths = read_lines(memory_root / "image_paths.txt")

    train_embeddings, train_embedding_details = load_memory_embeddings(memory_root, memory_meta)
    check_memory_consistency(train_embeddings, train_labels, train_row_ids, train_image_paths)

    index, index_details = load_faiss_index(memory_root / "index.faiss", train_embeddings)
    if int(index.ntotal) != train_embeddings.shape[0]:
        raise SystemExit(
            f"FAISS index row count {int(index.ntotal)} does not match train embeddings {train_embeddings.shape[0]}."
        )

    label_columns, label_names, manifest_records = load_manifest_records(manifest_csv, split=split)
    val_embeddings, val_row_ids, val_sidecar_image_paths, query_run_meta = load_query_split(query_embedding_root, split=split)
    val_labels, val_manifest_image_paths = build_labels_from_records(val_row_ids, manifest_records)
    validate_query_alignment(val_row_ids, val_sidecar_image_paths, val_manifest_image_paths)
    normalized_val_embeddings, val_norm_summary_before, val_norm_summary_after = normalize_rows(val_embeddings)

    max_k = max(SWEEP_K_VALUES)
    if int(index.ntotal) < max_k:
        raise SystemExit(f"Memory contains only {int(index.ntotal)} rows, which is less than max sweep k={max_k}.")
    neighbor_scores, neighbor_indices = index.search(np.ascontiguousarray(normalized_val_embeddings.astype(np.float32)), max_k)

    sweep_rows: list[dict[str, Any]] = []
    metrics_by_config: dict[str, Any] = {}
    probabilities_by_config: dict[str, np.ndarray] = {}
    for k in SWEEP_K_VALUES:
        for tau in SWEEP_TAU_VALUES:
            probabilities = compute_memory_probabilities(
                neighbor_indices,
                neighbor_scores,
                train_labels,
                k=k,
                tau=float(tau),
            )
            metrics = evaluate_probabilities(val_labels, probabilities, label_names)
            row = {
                "k": int(k),
                "tau": int(tau),
                "macro_auroc": metrics["macro_auroc"],
                "macro_average_precision": metrics["macro_average_precision"],
                "macro_ece": metrics["macro_ece"],
                "macro_f1_at_0.5": metrics["macro_f1_at_0.5"],
                "diagnostic_macro_f1_at_tuned_thresholds": metrics["diagnostic_macro_f1_at_tuned_thresholds"],
            }
            sweep_rows.append(row)
            key = f"k={k},tau={tau}"
            metrics_by_config[key] = metrics
            probabilities_by_config[key] = probabilities

    best_row, best_trace = select_best_sweep_row(sweep_rows)
    best_key = f"k={int(best_row['k'])},tau={int(best_row['tau'])}"
    best_metrics = metrics_by_config[best_key]
    best_probabilities = probabilities_by_config[best_key]

    selected_queries = choose_qualitative_query_indices(val_labels, args.qualitative_queries)
    qualitative_neighbors = build_qualitative_neighbors(
        selected_queries,
        val_row_ids,
        val_manifest_image_paths,
        val_labels,
        neighbor_indices[:, : int(best_row["k"])],
        neighbor_scores[:, : int(best_row["k"])],
        train_row_ids,
        train_labels,
        label_names,
    )

    experiment_meta_path = experiment_dir / "experiment_meta.json"
    recreation_report_path = experiment_dir / "recreation_report.md"
    sweep_results_path = experiment_dir / "sweep_results.json"
    best_config_path = experiment_dir / "best_config.json"
    best_val_metrics_path = experiment_dir / "best_val_metrics.json"
    val_probabilities_path = experiment_dir / "val_probabilities.npy"
    qualitative_neighbors_path = experiment_dir / "qualitative_neighbors.json"
    selection_summary_path = experiment_dir / "memory_only_selection.md"

    write_json(
        sweep_results_path,
        {
            "configs": sweep_rows,
            "metrics_by_config": metrics_by_config,
            "selection_trace": best_trace,
        },
    )
    write_json(
        best_config_path,
        {
            "k": int(best_row["k"]),
            "tau": int(best_row["tau"]),
            "selection_metric": DEFAULT_SELECTION_METRIC,
            "selection_trace": best_trace,
        },
    )
    write_json(best_val_metrics_path, best_metrics)
    np.save(val_probabilities_path, best_probabilities.astype(np.float32))
    write_json(qualitative_neighbors_path, qualitative_neighbors)

    selection_summary_lines = [
        "# Memory-Only Selection",
        "",
        "The canonical memory-only validation configuration for the current source memory stage is:",
        "",
        f"- k: `{int(best_row['k'])}`",
        f"- tau: `{int(best_row['tau'])}`",
        f"- validation macro AUROC: `{format_metric(best_metrics['macro_auroc'])}`",
        f"- validation macro average precision: `{format_metric(best_metrics['macro_average_precision'])}`",
        f"- validation macro ECE: `{format_metric(best_metrics['macro_ece'])}`",
        f"- validation macro F1 @ 0.5: `{format_metric(best_metrics['macro_f1_at_0.5'])}`",
    ]
    selection_summary_path.write_text("\n".join(selection_summary_lines) + "\n", encoding="utf-8")

    baseline_meta_path = baseline_experiment_dir / "experiment_meta.json"
    baseline_meta = read_json(baseline_meta_path) if baseline_meta_path.exists() else None
    experiment_meta = {
        "argv": sys.argv,
        "baseline_experiment_dir": str(baseline_experiment_dir),
        "baseline_meta_path": str(baseline_meta_path),
        "baseline_reference": baseline_meta,
        "experiment_dir": str(experiment_dir),
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_number": experiment_number,
        "index_details": index_details,
        "label_columns": label_columns,
        "label_names": label_names,
        "manifest_csv": str(manifest_csv),
        "memory_root": str(memory_root),
        "memory_source_embeddings": train_embedding_details,
        "memory_summary": {
            "row_count": int(train_embeddings.shape[0]),
            "embedding_dim": int(train_embeddings.shape[1]),
        },
        "operation_label": DEFAULT_OPERATION_LABEL,
        "query_embedding_root": str(query_embedding_root),
        "query_split": {
            "embedding_dim": int(val_embeddings.shape[1]),
            "num_rows": int(val_embeddings.shape[0]),
            "split": split,
            "run_meta": query_run_meta,
        },
        "query_norm_summary": {
            "raw": val_norm_summary_before,
            "normalized": val_norm_summary_after,
        },
        "run_date_utc": utc_now_iso(),
        "script_path": str(script_path),
        "script_sha256": script_hash,
        "seed": args.seed,
        "selection_metric": DEFAULT_SELECTION_METRIC,
        "best_config": {
            "k": int(best_row["k"]),
            "tau": int(best_row["tau"]),
        },
        "best_metrics": best_metrics,
        "artifacts": {
            "experiment_meta": str(experiment_meta_path),
            "recreation_report": str(recreation_report_path),
            "sweep_results": str(sweep_results_path),
            "best_config": str(best_config_path),
            "best_val_metrics": str(best_val_metrics_path),
            "val_probabilities": str(val_probabilities_path),
            "qualitative_neighbors": str(qualitative_neighbors_path),
            "selection_summary": str(selection_summary_path),
        },
    }
    write_json(experiment_meta_path, experiment_meta)

    argv_exact = [
        "python",
        str(script_path),
        "--memory-root",
        str(memory_root),
        "--query-embedding-root",
        str(query_embedding_root),
        "--baseline-experiment-dir",
        str(baseline_experiment_dir),
        "--manifest-csv",
        str(manifest_csv),
        "--split",
        split,
        "--qualitative-queries",
        str(args.qualitative_queries),
        "--seed",
        str(args.seed),
        "--experiment-name",
        experiment_name,
        "--overwrite",
    ]
    argv_fresh = [
        "python",
        str(script_path),
        "--memory-root",
        str(memory_root),
        "--query-embedding-root",
        str(query_embedding_root),
        "--baseline-experiment-dir",
        str(baseline_experiment_dir),
        "--manifest-csv",
        str(manifest_csv),
        "--split",
        split,
        "--qualitative-queries",
        str(args.qualitative_queries),
        "--seed",
        str(args.seed),
        "--experiment-name",
        strip_experiment_number_prefix(experiment_name),
    ]
    output_paths = [
        experiment_meta_path,
        recreation_report_path,
        sweep_results_path,
        best_config_path,
        best_val_metrics_path,
        val_probabilities_path,
        qualitative_neighbors_path,
        selection_summary_path,
    ]
    recreation_report = build_recreation_report(
        experiment_dir=experiment_dir,
        experiment_id=experiment_id,
        operation_label=DEFAULT_OPERATION_LABEL,
        script_path=script_path,
        argv_exact=argv_exact,
        argv_fresh=argv_fresh,
        memory_root=memory_root,
        query_embedding_root=query_embedding_root,
        baseline_experiment_dir=baseline_experiment_dir,
        manifest_csv=manifest_csv,
        split=split,
        row_count=int(val_embeddings.shape[0]),
        label_names=label_names,
        memory_rows=int(train_embeddings.shape[0]),
        memory_dim=int(train_embeddings.shape[1]),
        query_dim=int(val_embeddings.shape[1]),
        norm_summary_before=val_norm_summary_before,
        norm_summary_after=val_norm_summary_after,
        index_details=index_details,
        best_row=best_row,
        best_metrics=best_metrics,
        sweep_rows=sweep_rows,
        output_paths=output_paths,
    )
    recreation_report_path.write_text(recreation_report + "\n", encoding="utf-8")

    print(f"[saved] experiment_dir={experiment_dir}")
    print(
        "[best_config] "
        f"k={int(best_row['k'])} tau={int(best_row['tau'])} "
        f"macro_auroc={format_metric(best_metrics['macro_auroc'])} "
        f"macro_ap={format_metric(best_metrics['macro_average_precision'])} "
        f"macro_ece={format_metric(best_metrics['macro_ece'])} "
        f"macro_f1_0p5={format_metric(best_metrics['macro_f1_at_0.5'])}"
    )
    print(
        "[query_norms] "
        f"raw_mean={val_norm_summary_before['mean']:.8f} "
        f"post_mean={val_norm_summary_after['mean']:.8f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
