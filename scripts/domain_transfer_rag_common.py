#!/usr/bin/env python3
"""Common helpers for the new domain-transfer image-only retrieval branch."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import experiment_layout
import faiss  # type: ignore
import numpy as np
import torch
from torch import nn


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest/manifest_common_labels_pilot5h.csv")
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_EMBEDDING_ROOT = experiment_layout.find_experiment_dir("exp0014", experiments_root=DEFAULT_EXPERIMENTS_ROOT)
DEFAULT_BASELINE_EXPERIMENT_DIR = experiment_layout.find_experiment_dir(
    "exp0015",
    experiments_root=DEFAULT_EXPERIMENTS_ROOT,
)
DEFAULT_SOURCE_DOMAIN = "d0_nih"
DEFAULT_SOURCE_SPLIT = "train"
DEFAULT_SELECTION_DOMAIN = "d0_nih"
DEFAULT_SELECTION_SPLIT = "val"
DEFAULT_SEED = 3407
DEFAULT_ECE_BINS = 15
DEFAULT_BATCH_SIZE = 2048
DEFAULT_QUALITATIVE_QUERIES = 10
SWEEP_K_VALUES = [1, 3, 5, 10, 20, 50]
SWEEP_TAU_VALUES = [1.0, 5.0, 10.0, 20.0, 40.0]
ALPHA_GRID = [round(value, 1) for value in np.linspace(0.0, 1.0, num=11).tolist()]

AUTO_ID_COLUMNS = ("row_id", "sample_id", "report_id", "image_id", "id")
AUTO_PATH_COLUMNS = ("image_path", "report_path", "path")

BASELINE_ALIAS_TO_FILENAME = {
    "d0_val": "d0_val_metrics.json",
    "d0_test": "d0_test_metrics.json",
    "d1_transfer": "d1_transfer_metrics.json",
    "d2_transfer": "d2_transfer_metrics.json",
}


@dataclass(frozen=True)
class ManifestRecord:
    row_id: str
    domain: str
    dataset: str
    split: str
    image_path: str
    labels: tuple[float, ...]


@dataclass(frozen=True)
class SidecarSpec:
    relative_path: str
    format: str
    parser: str
    column: str | None = None


@dataclass(frozen=True)
class EmbeddingSplit:
    domain: str
    split: str
    split_dir: Path
    embeddings_path: Path
    embeddings: np.ndarray
    sidecar: SidecarSpec
    row_ids: list[str]
    image_paths: list[str]
    run_meta: dict[str, Any] | None


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_labels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or fallback


def ensure_operation_prefix(name: str, operation_label: str) -> str:
    normalized = slugify(operation_label, fallback="operation")
    if name.startswith("exp") and "__" in name:
        prefix, remainder = name.split("__", 1)
        if remainder.startswith(normalized):
            return name
        return f"{prefix}__{normalized}__{remainder}"
    if name.startswith(normalized):
        return name
    return f"{normalized}__{name}"


def extract_experiment_number(name: str) -> int | None:
    match = re.match(r"^exp(\d+)(?:__|$)", name)
    if match is None:
        return None
    return int(match.group(1))


def next_experiment_number(experiments_root: Path) -> int:
    return experiment_layout.next_experiment_number(experiments_root)


def resolve_experiment_identity(
    *,
    experiments_root: Path,
    requested_name: str | None,
    generated_slug: str,
    operation_label: str,
    overwrite: bool,
    id_width: int = 4,
) -> tuple[int, str, str, Path]:
    requested = (requested_name or "").strip() or None
    base_name = ensure_operation_prefix(requested or generated_slug, operation_label)
    return experiment_layout.resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=base_name if requested else None,
        generated_slug=base_name,
        overwrite=overwrite,
        id_width=id_width,
    )


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, torch.Tensor):
        return to_serializable(value.detach().cpu().numpy())
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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_serializable(payload), sort_keys=True) + "\n")


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
    if not values:
        raise SystemExit(f"CSV sidecar is empty: {path}")
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
    preferred = [
        SidecarSpec(relative_path="row_ids.json", format="json_list", parser="identity"),
        SidecarSpec(relative_path="image_paths.txt", format="lines", parser="stem"),
        SidecarSpec(relative_path="report_ids.json", format="json_list", parser="identity"),
    ]
    for candidate in preferred:
        if (split_dir / candidate.relative_path).exists():
            return candidate
    for candidate in sorted(split_dir.glob("*.csv")):
        chosen = choose_csv_sidecar(candidate)
        if chosen is not None:
            return chosen
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
    normalized_norms = np.linalg.norm(normalized.astype(np.float64), axis=1)
    if not np.isfinite(normalized).all():
        raise SystemExit("Normalized embeddings contain NaN or inf values.")
    return normalized, summarize_values(raw_norms), summarize_values(normalized_norms)


def load_manifest_records(
    manifest_csv: Path,
    *,
    domain: str,
    split: str,
) -> tuple[list[str], list[str], dict[str, ManifestRecord]]:
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV not found: {manifest_csv}")
    text = manifest_csv.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise SystemExit(f"Manifest CSV has no header: {manifest_csv}")
    required = {"domain", "split", "image_path"}
    if not required.issubset(reader.fieldnames):
        raise SystemExit(f"Manifest CSV must contain columns: {sorted(required)}")
    label_columns = [field for field in reader.fieldnames if field.startswith("label_")]
    if not label_columns:
        raise SystemExit("Manifest CSV does not contain any label_... columns.")
    label_names = [column.removeprefix("label_") for column in label_columns]

    records: dict[str, ManifestRecord] = {}
    normalized_domain = domain.strip()
    normalized_split = split.strip().lower()
    for row in reader:
        current_domain = (row.get("domain") or "").strip()
        current_split = (row.get("split") or "").strip().lower()
        if current_domain != normalized_domain or current_split != normalized_split:
            continue
        row_id = (row.get("row_id") or "").strip()
        image_path = (row.get("image_path") or "").strip()
        if not row_id:
            if not image_path:
                continue
            row_id = Path(image_path).stem
        if not image_path:
            continue
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
            raise SystemExit(f"Duplicate manifest row_id '{row_id}' found in {domain}/{split}.")
        records[row_id] = ManifestRecord(
            row_id=row_id,
            domain=current_domain,
            dataset=(row.get("dataset") or "").strip(),
            split=current_split,
            image_path=image_path,
            labels=tuple(labels),
        )
    if not records:
        raise SystemExit(f"Manifest contains no rows for {domain}/{split}.")
    return label_columns, label_names, records


def load_embedding_split(embedding_root: Path, *, domain: str, split: str) -> EmbeddingSplit:
    split_dir = embedding_root / domain / split
    if not split_dir.exists():
        raise SystemExit(f"Split directory not found for {domain}/{split}: {split_dir}")
    embeddings_path = split_dir / "embeddings.npy"
    embeddings = load_embedding_array(embeddings_path)
    sidecar = autodetect_sidecar(split_dir)
    sidecar_path = split_dir / sidecar.relative_path
    sidecar_items = read_sidecar_items(sidecar_path, format_name=sidecar.format, column=sidecar.column)
    if len(sidecar_items) != int(embeddings.shape[0]):
        raise SystemExit(
            f"{domain}/{split} has {embeddings.shape[0]} embedding rows but {len(sidecar_items)} sidecar rows."
        )
    row_ids = [parse_row_id(item, sidecar.parser) for item in sidecar_items]
    image_paths_path = split_dir / "image_paths.txt"
    image_paths = read_lines(image_paths_path) if image_paths_path.exists() else []
    if image_paths and len(image_paths) != len(row_ids):
        raise SystemExit(
            f"{domain}/{split} has {len(row_ids)} row IDs but {len(image_paths)} image paths in {image_paths_path}."
        )
    run_meta_path = split_dir / "run_meta.json"
    run_meta = read_json(run_meta_path) if run_meta_path.exists() else None
    return EmbeddingSplit(
        domain=domain,
        split=split,
        split_dir=split_dir,
        embeddings_path=embeddings_path,
        embeddings=embeddings,
        sidecar=sidecar,
        row_ids=row_ids,
        image_paths=image_paths,
        run_meta=run_meta,
    )


def build_labels_from_records(
    row_ids: list[str],
    records: dict[str, ManifestRecord],
) -> tuple[np.ndarray, list[str], list[str]]:
    labels = np.zeros((len(row_ids), len(next(iter(records.values())).labels)), dtype=np.float32)
    image_paths: list[str] = []
    expected_row_ids: list[str] = []
    missing: list[str] = []
    for index, row_id in enumerate(row_ids):
        record = records.get(row_id)
        if record is None:
            if len(missing) < 5:
                missing.append(row_id)
            continue
        labels[index] = np.asarray(record.labels, dtype=np.float32)
        image_paths.append(record.image_path)
        expected_row_ids.append(record.row_id)
    if missing:
        raise SystemExit(f"Embedding split contains row IDs missing from manifest. Examples: {missing}")
    return labels, image_paths, expected_row_ids


def validate_query_alignment(
    row_ids: list[str],
    expected_row_ids: list[str],
    sidecar_image_paths: list[str],
    manifest_image_paths: list[str],
) -> None:
    if len(expected_row_ids) != len(row_ids):
        raise SystemExit("Row-id alignment failed due to row-count mismatch.")
    if sidecar_image_paths and len(sidecar_image_paths) != len(manifest_image_paths):
        raise SystemExit("Image-path alignment failed due to row-count mismatch.")
    for index, row_id in enumerate(row_ids):
        expected = expected_row_ids[index]
        if expected != row_id:
            raise SystemExit(
                f"Row-id mismatch at row {index}: manifest has '{expected}', embeddings have '{row_id}'."
            )
        if sidecar_image_paths:
            if Path(sidecar_image_paths[index]).name != Path(manifest_image_paths[index]).name:
                raise SystemExit(
                    f"Image path mismatch at row {index}: "
                    f"manifest has '{manifest_image_paths[index]}', sidecar has '{sidecar_image_paths[index]}'."
                )


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    if embeddings.ndim != 2:
        raise SystemExit(f"Expected 2D embeddings, got shape {embeddings.shape}.")
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(np.ascontiguousarray(embeddings.astype(np.float32)))
    return index


def search_index(index: faiss.IndexFlatIP, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    scores, indices = index.search(np.ascontiguousarray(queries.astype(np.float32)), int(top_k))
    if scores.shape != indices.shape:
        raise SystemExit("FAISS returned mismatched score/index shapes.")
    return np.asarray(indices, dtype=np.int64), np.asarray(scores, dtype=np.float32)


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
    probabilities = np.sum(weights[:, :, None] * train_labels[sliced_indices].astype(np.float64), axis=1)
    probabilities = np.ascontiguousarray(probabilities.astype(np.float32))
    if not np.isfinite(probabilities).all():
        raise SystemExit(f"Computed memory probabilities contain NaN or inf for k={k}, tau={tau}.")
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
    return float(precision[y_sorted == 1].sum() / positives)


def binary_f1(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    preds = (np.asarray(probs, dtype=np.float64) >= float(threshold)).astype(np.int64)
    tp = float(np.sum((preds == 1) & (y == 1)))
    fp = float(np.sum((preds == 1) & (y == 0)))
    fn = float(np.sum((preds == 0) & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


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
                "reason": "no_positive_examples",
            }
            continue
        order = np.argsort(-p, kind="mergesort")
        y_sorted = y[order]
        p_sorted = p[order]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        fn = float(positives) - tp
        precision = tp / np.clip(tp + fp, 1.0, None)
        recall = tp / float(positives)
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
    thresholds, threshold_payload = tune_thresholds(y_true, probs, label_names)
    return evaluate_probabilities_with_frozen_thresholds(
        y_true,
        probs,
        label_names,
        thresholds=thresholds,
        threshold_payload=threshold_payload,
    )


def evaluate_probabilities_with_frozen_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    label_names: list[str],
    *,
    thresholds: np.ndarray,
    threshold_payload: dict[str, Any],
) -> dict[str, Any]:
    label_metrics: dict[str, dict[str, Any]] = {}
    macro_auroc_values: list[float] = []
    macro_ap_values: list[float] = []
    macro_ece_values: list[float] = []
    macro_f1_values: list[float] = []
    macro_f1_tuned_values: list[float] = []

    for idx, label_name in enumerate(label_names):
        targets = np.asarray(y_true[:, idx], dtype=np.int64)
        scores = np.asarray(probs[:, idx], dtype=np.float64)
        threshold = float(thresholds[idx])
        auroc = binary_auroc(targets, scores)
        average_precision = binary_average_precision(targets, scores)
        f1_at_0p5 = binary_f1(targets, scores, 0.5)
        f1_at_tuned = binary_f1(targets, scores, threshold)
        ece = binary_ece(targets, scores)

        if auroc is not None:
            macro_auroc_values.append(auroc)
        if average_precision is not None:
            macro_ap_values.append(average_precision)
        macro_ece_values.append(ece)
        macro_f1_values.append(f1_at_0p5)
        macro_f1_tuned_values.append(f1_at_tuned)

        label_metrics[label_name] = {
            "auroc": auroc,
            "average_precision": average_precision,
            "ece": ece,
            "f1_at_0.5": f1_at_0p5,
            "f1_at_tuned_threshold": f1_at_tuned,
            "positive_count": int(targets.sum()),
            "negative_count": int(targets.shape[0] - targets.sum()),
            "prevalence": float(targets.mean()),
            "threshold_used": threshold,
        }

    return {
        "macro": {
            "auroc": float(np.mean(macro_auroc_values)) if macro_auroc_values else None,
            "average_precision": float(np.mean(macro_ap_values)) if macro_ap_values else None,
            "ece": float(np.mean(macro_ece_values)) if macro_ece_values else None,
            "f1_at_0.5": float(np.mean(macro_f1_values)) if macro_f1_values else None,
            "f1_at_tuned_thresholds": (
                float(np.mean(macro_f1_tuned_values)) if macro_f1_tuned_values else None
            ),
        },
        "label_metrics": label_metrics,
        "thresholds": threshold_payload,
    }


def extract_thresholds(
    threshold_payload: dict[str, Any],
    label_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    values: list[float] = []
    normalized: dict[str, Any] = {}
    for label_name in label_names:
        current = threshold_payload.get(label_name)
        if not isinstance(current, dict) or "threshold" not in current:
            raise SystemExit(f"Missing frozen threshold for label '{label_name}'.")
        threshold = float(current["threshold"])
        values.append(threshold)
        normalized[label_name] = {
            "threshold": threshold,
            "best_f1": current.get("best_f1"),
            "prevalence": current.get("prevalence"),
        }
    return np.asarray(values, dtype=np.float32), normalized


def load_baseline_thresholds(
    baseline_experiment_dir: Path,
    label_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = read_json(baseline_experiment_dir / "d0_val_f1_thresholds.json").get("labels")
    if not isinstance(payload, dict):
        raise SystemExit(
            f"Baseline validation thresholds are missing from {baseline_experiment_dir / 'd0_val_f1_thresholds.json'}."
        )
    return extract_thresholds(payload, label_names)


def baseline_metrics_path(baseline_experiment_dir: Path, split_alias: str) -> Path:
    filename = BASELINE_ALIAS_TO_FILENAME.get(split_alias)
    if filename is None:
        raise SystemExit(f"Unsupported baseline split alias: {split_alias}")
    return baseline_experiment_dir / filename


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def reconstruct_baseline_probabilities(
    *,
    checkpoint_path: Path,
    normalized_embeddings: np.ndarray,
    label_names: list[str],
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Unexpected checkpoint payload type: {type(checkpoint)!r}")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise SystemExit("Checkpoint does not contain a 'state_dict' dictionary.")
    if set(state_dict.keys()) != {"classifier.weight", "classifier.bias"}:
        raise SystemExit(
            "Only linear baseline checkpoints are supported in this new retrieval branch. "
            f"Found keys: {sorted(state_dict.keys())}"
        )
    weight = state_dict["classifier.weight"]
    bias = state_dict["classifier.bias"]
    input_dim = int(weight.shape[1])
    output_dim = int(weight.shape[0])
    if normalized_embeddings.shape[1] != input_dim:
        raise SystemExit(
            f"Embedding dim {normalized_embeddings.shape[1]} does not match checkpoint input dim {input_dim}."
        )
    if output_dim != len(label_names):
        raise SystemExit(f"Checkpoint output dim {output_dim} does not match label count {len(label_names)}.")

    model = LinearProbe(input_dim=input_dim, num_labels=output_dim)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    features = torch.from_numpy(normalized_embeddings.astype(np.float32))
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            logits = model(features[start:end])
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            batches.append(probs)
    probabilities = np.ascontiguousarray(np.concatenate(batches, axis=0).astype(np.float32))
    return probabilities, {
        "batch_size": int(batch_size),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_label_names": checkpoint.get("label_names"),
        "input_dim": input_dim,
        "output_dim": output_dim,
    }


def compare_metrics_to_archived(
    reconstructed_metrics: dict[str, Any],
    archived_metrics: dict[str, Any],
    label_names: list[str],
) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    max_abs_delta = 0.0
    macro_deltas: dict[str, Any] = {}
    macro_key_map = {
        "auroc": "auroc",
        "average_precision": "average_precision",
        "ece": "ece",
        "f1_at_0.5": "f1_at_0.5",
        "f1_at_tuned_thresholds": "f1_at_tuned_thresholds",
    }
    for key, archived_key in macro_key_map.items():
        archived_value = archived_metrics["macro"].get(archived_key)
        reconstructed_value = reconstructed_metrics["macro"].get(key)
        delta = None
        if archived_value is not None and reconstructed_value is not None:
            delta = float(reconstructed_value - archived_value)
            max_abs_delta = max(max_abs_delta, abs(delta))
        macro_deltas[key] = {
            "archived": archived_value,
            "reconstructed": reconstructed_value,
            "delta": delta,
        }

    for label_name in label_names:
        archived_label = archived_metrics["label_metrics"][label_name]
        reconstructed_label = reconstructed_metrics["label_metrics"][label_name]
        current: dict[str, Any] = {}
        for metric_name in ("auroc", "average_precision", "ece", "f1_at_0.5", "f1_at_tuned_threshold"):
            archived_value = archived_label.get(metric_name)
            reconstructed_value = reconstructed_label.get(metric_name)
            delta = None
            if archived_value is not None and reconstructed_value is not None:
                delta = float(reconstructed_value - archived_value)
                max_abs_delta = max(max_abs_delta, abs(delta))
            current[metric_name] = {
                "archived": archived_value,
                "reconstructed": reconstructed_value,
                "delta": delta,
            }
        per_label[label_name] = current

    return {
        "macro": macro_deltas,
        "per_label": per_label,
        "max_abs_delta": float(max_abs_delta),
        "matches_archived_metrics_within_5e-4": bool(max_abs_delta <= 5e-4),
    }


def select_best_retrieval_row(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise SystemExit("Sweep results are empty.")

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
    tau_values = [float(row["tau"]) for row in k_candidates]
    selected_tau = 10.0 if any(np.isclose(tau, 10.0, atol=1e-12, rtol=0.0) for tau in tau_values) else min(tau_values)
    final_candidates = [
        row for row in k_candidates if np.isclose(float(row["tau"]), selected_tau, atol=1e-12, rtol=0.0)
    ]
    best_row = final_candidates[0]
    return best_row, {
        "max_macro_auroc": max_auroc,
        "max_macro_average_precision": max_ap,
        "selected_k": min_k,
        "selected_tau": selected_tau,
        "tie_break_rule": [
            "highest validation macro AUROC",
            "higher validation macro average precision",
            "smaller k",
            "tau = 10 if present among ties, otherwise smaller tau",
        ],
    }


def mix_probabilities(base_probabilities: np.ndarray, memory_probabilities: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit(f"Alpha must be in [0, 1], found {alpha}.")
    mixed = alpha * base_probabilities.astype(np.float64) + (1.0 - alpha) * memory_probabilities.astype(np.float64)
    mixed = np.ascontiguousarray(mixed.astype(np.float32))
    if not np.isfinite(mixed).all():
        raise SystemExit(f"Mixed probabilities contain NaN or inf for alpha={alpha}.")
    return mixed


def select_best_alpha_row(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise SystemExit("Alpha sweep results are empty.")

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
    max_alpha = max(float(row["alpha"]) for row in ap_candidates)
    final_candidates = [
        row for row in ap_candidates if np.isclose(float(row["alpha"]), max_alpha, atol=1e-12, rtol=0.0)
    ]
    best_row = final_candidates[0]
    return best_row, {
        "max_macro_auroc": max_auroc,
        "max_macro_average_precision": max_ap,
        "selected_alpha": max_alpha,
        "tie_break_rule": [
            "highest validation macro AUROC",
            "higher validation macro average precision",
            "larger alpha",
        ],
    }


def format_metric(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def format_alpha_value(value: float) -> str:
    return f"{float(value):.1f}"


def split_alias_from_domain_split(domain: str, split: str) -> str:
    mapping = {
        ("d0_nih", "train"): "d0_train",
        ("d0_nih", "val"): "d0_val",
        ("d0_nih", "test"): "d0_test",
        ("d1_chexpert", "val"): "d1_transfer",
        ("d2_mimic", "test"): "d2_transfer",
        ("d2_mimic", "val"): "d2_val",
    }
    return mapping.get((domain, split), f"{domain}_{split}")


def labels_to_names(label_row: np.ndarray, label_names: list[str]) -> list[str]:
    indices = np.flatnonzero(np.asarray(label_row, dtype=np.float64) > 0.5)
    return [label_names[int(index)] for index in indices.tolist()]


def choose_qualitative_query_indices(labels: np.ndarray, count: int) -> list[int]:
    if count <= 0:
        return []
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

    selected: list[int] = []
    seen: set[int] = set()
    for candidates, desired in (
        (negative_indices, 3),
        (single_positive_indices, 3),
        (multi_positive_indices, count - 6),
    ):
        for query_index in spread_select(candidates, desired):
            if query_index in seen:
                continue
            selected.append(query_index)
            seen.add(query_index)
    if len(selected) < count:
        for query_index in np.arange(labels.shape[0], dtype=np.int64).tolist():
            if query_index in seen:
                continue
            selected.append(int(query_index))
            seen.add(int(query_index))
            if len(selected) >= count:
                break
    return selected[:count]


def build_qualitative_neighbors(
    *,
    query_indices: list[int],
    query_row_ids: list[str],
    query_image_paths: list[str],
    query_labels: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    train_row_ids: list[str],
    train_labels: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for query_index in query_indices:
        current_neighbor_indices = neighbor_indices[query_index].astype(np.int64).tolist()
        current_neighbor_scores = neighbor_scores[query_index].astype(np.float64).tolist()
        payload.append(
            {
                "query_row_id": query_row_ids[query_index],
                "query_image_path": query_image_paths[query_index] if query_image_paths else None,
                "query_positive_labels": labels_to_names(query_labels[query_index], label_names),
                "retrieved_row_ids": [train_row_ids[index] for index in current_neighbor_indices],
                "retrieved_similarities": [float(score) for score in current_neighbor_scores],
                "retrieved_positive_labels": [
                    labels_to_names(train_labels[index], label_names) for index in current_neighbor_indices
                ],
            }
        )
    return payload


def script_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    return script_sha256(path)


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def build_simple_recreation_report(
    *,
    experiment_dir: Path,
    script_path: Path,
    argv: list[str],
    summary_lines: list[str],
) -> str:
    lines = [
        "# Recreation Report",
        "",
        f"- Experiment directory: `{experiment_dir}`",
        f"- Script: `{script_path}`",
        f"- Script SHA-256: `{script_sha256(script_path)}`",
        f"- Run date UTC: `{utc_now_iso()}`",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(argv),
        "```",
        "",
        "## Summary",
        "",
    ]
    lines.extend(summary_lines)
    return "\n".join(lines) + "\n"
