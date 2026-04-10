#!/usr/bin/env python3
"""Evaluate validation-only probability mixing between frozen baseline and memory probabilities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'numpy'.") from exc

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'torch'.") from exc


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_nih_cxr14_all14.csv")
DEFAULT_MEMORY_EVAL_ROOT = Path(
    "/workspace/experiments/exp0006__source_memory_only_evaluation__nih_cxr14_exp0005_val_e100_p4"
)
DEFAULT_BASELINE_EXPERIMENT_DIR = Path(
    "/workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4"
)
DEFAULT_QUERY_EMBEDDING_ROOT = Path(
    "/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2"
)
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_OPERATION_LABEL = "source_probability_mixing_evaluation"
DEFAULT_EXPERIMENT_ID_WIDTH = 4
DEFAULT_SPLIT = "val"
DEFAULT_BATCH_SIZE = 2048
DEFAULT_SEED = 3407
DEFAULT_ECE_BINS = 15
DEFAULT_SELECTION_METRIC = "macro_auroc"
ALPHA_GRID = [round(value, 1) for value in np.linspace(0.0, 1.0, num=11).tolist()]


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


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_labels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


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


def unique_preserving_order(values: list[float]) -> list[float]:
    unique_values: list[float] = []
    for value in values:
        if value in unique_values:
            continue
        unique_values.append(value)
    return unique_values


def format_alpha_value(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise SystemExit(f"Encountered a non-finite alpha value: {value!r}")
    if numeric.is_integer():
        return f"{numeric:.1f}"
    return f"{numeric:.12g}"


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


def load_query_split(query_embedding_root: Path, *, split: str) -> tuple[np.ndarray, list[str], list[str]]:
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
    return embeddings, row_ids, image_paths


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
        raise SystemExit(f"Validation split contains row IDs missing from manifest. Examples: {missing}")
    return labels, image_paths


def validate_query_alignment(row_ids: list[str], sidecar_image_paths: list[str], manifest_image_paths: list[str]) -> None:
    if sidecar_image_paths and len(sidecar_image_paths) != len(manifest_image_paths):
        raise SystemExit("Validation image-path alignment failed due to row-count mismatch.")
    for index, row_id in enumerate(row_ids):
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


def evaluate_probabilities(y_true: np.ndarray, probs: np.ndarray, label_names: list[str]) -> dict[str, Any]:
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


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def reconstruct_baseline_probabilities(
    *,
    checkpoint_path: Path,
    val_embeddings: np.ndarray,
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
        raise SystemExit(f"Unsupported checkpoint keys for linear probe: {sorted(state_dict.keys())}")

    weight = state_dict["classifier.weight"]
    bias = state_dict["classifier.bias"]
    input_dim = int(weight.shape[1])
    output_dim = int(weight.shape[0])
    if val_embeddings.shape[1] != input_dim:
        raise SystemExit(
            f"Validation embedding dim {val_embeddings.shape[1]} does not match checkpoint input dim {input_dim}."
        )
    if output_dim != len(label_names):
        raise SystemExit(f"Checkpoint output dim {output_dim} does not match label count {len(label_names)}.")

    model = LinearProbe(input_dim=input_dim, num_labels=output_dim)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    features = torch.from_numpy(val_embeddings.astype(np.float32))
    probability_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            logits = model(features[start:end])
            probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            probability_batches.append(probabilities)
    probs = np.ascontiguousarray(np.concatenate(probability_batches, axis=0).astype(np.float32))

    summary = {
        "batch_size": int(batch_size),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_label_names": checkpoint.get("label_names"),
        "input_dim": input_dim,
        "output_dim": output_dim,
    }
    return probs, summary


def compare_baseline_to_archived(
    reconstructed_metrics: dict[str, Any],
    archived_val_metrics: dict[str, Any],
    label_names: list[str],
) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    max_abs_delta = 0.0
    macro_keys = {
        "macro_auroc": ("macro", "auroc"),
        "macro_average_precision": ("macro", "average_precision"),
        "macro_ece": ("macro", "ece"),
        "macro_f1_at_0.5": ("macro", "f1_at_0.5"),
    }
    macro_deltas: dict[str, Any] = {}
    for target_key, archived_path in macro_keys.items():
        archived_value = archived_val_metrics[archived_path[0]][archived_path[1]]
        reconstructed_value = reconstructed_metrics[target_key]
        delta = None
        if archived_value is not None and reconstructed_value is not None:
            delta = float(reconstructed_value - archived_value)
            max_abs_delta = max(max_abs_delta, abs(delta))
        macro_deltas[target_key] = {
            "archived": archived_value,
            "reconstructed": reconstructed_value,
            "delta": delta,
        }

    for label_name in label_names:
        archived_label = archived_val_metrics["label_metrics"][label_name]
        reconstructed_label = reconstructed_metrics["per_label"][label_name]
        current: dict[str, Any] = {}
        for metric_name, archived_key, reconstructed_key in (
            ("auroc", "auroc", "auroc"),
            ("average_precision", "average_precision", "average_precision"),
            ("ece", "ece", "ece"),
            ("f1_at_0.5", "f1_at_0.5", "f1_at_0.5"),
        ):
            archived_value = archived_label.get(archived_key)
            reconstructed_value = reconstructed_label.get(reconstructed_key)
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
        "comparison_scope": "forward_metrics_only_excluding_threshold_retuning",
        "matches_archived_metrics_within_5e-4": bool(max_abs_delta <= 5e-4),
    }


def mix_probabilities(p_base: np.ndarray, p_mem: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit(f"Alpha must be in [0, 1], found {alpha}.")
    mixed = alpha * p_base.astype(np.float64) + (1.0 - alpha) * p_mem.astype(np.float64)
    mixed = np.ascontiguousarray(mixed.astype(np.float32))
    if not np.isfinite(mixed).all():
        raise SystemExit(f"Mixed probabilities contain NaN or inf for alpha={alpha}.")
    if mixed.min() < -1e-6 or mixed.max() > 1.0 + 1e-6:
        raise SystemExit(
            f"Mixed probabilities are out of range for alpha={alpha}: min={mixed.min()}, max={mixed.max()}."
        )
    return mixed


def select_best_row(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise SystemExit("Probability-mixing sweep rows are empty.")

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
    trace = {
        "max_macro_auroc": max_auroc,
        "macro_auroc_tied_candidates": len(auroc_candidates),
        "max_macro_average_precision_within_auroc_ties": max_ap,
        "macro_average_precision_tied_candidates": len(ap_candidates),
        "largest_alpha_within_metric_ties": max_alpha,
        "final_candidates": len(final_candidates),
        "tie_break_rule": [
            "highest validation macro AUROC",
            "higher validation macro average precision",
            "larger alpha",
        ],
    }
    return best_row, trace


def format_shell_command(argv: list[str]) -> str:
    return shlex.join(argv)


def format_metric(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def build_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_recreation_report(
    *,
    experiment_dir: Path,
    experiment_id: str,
    operation_label: str,
    script_path: Path,
    argv_exact: list[str],
    argv_fresh: list[str],
    memory_eval_root: Path,
    baseline_experiment_dir: Path,
    query_embedding_root: Path,
    manifest_csv: Path,
    split: str,
    row_count: int,
    label_names: list[str],
    alpha_values: list[float],
    best_row: dict[str, Any],
    best_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    baseline_comparison: dict[str, Any],
    sweep_rows: list[dict[str, Any]],
    output_paths: list[Path],
) -> str:
    size_lines = [f"- {path.name}: `{human_size(path.stat().st_size)}`" for path in output_paths if path.exists()]
    total_size = sum(path.stat().st_size for path in output_paths if path.exists())
    hash_lines = [f"{sha256_file(path)}  {path}" for path in output_paths if path.exists()]
    sweep_table = build_markdown_table(
        ["alpha", "Macro AUROC", "Macro AP", "Macro ECE", "Macro F1 @ 0.5", "Best"],
        [
            [
                format_alpha_value(float(row["alpha"])),
                format_metric(row.get("macro_auroc")),
                format_metric(row.get("macro_average_precision")),
                format_metric(row.get("macro_ece")),
                format_metric(row.get("macro_f1_at_0.5")),
                "<- best" if np.isclose(float(row["alpha"]), float(best_row["alpha"]), atol=1e-12, rtol=0.0) else "",
            ]
            for row in sweep_rows
        ],
    )
    delta_auroc = None
    if best_metrics["macro_auroc"] is not None and baseline_metrics["macro_auroc"] is not None:
        delta_auroc = float(best_metrics["macro_auroc"] - baseline_metrics["macro_auroc"])
    delta_ap = None
    if best_metrics["macro_average_precision"] is not None and baseline_metrics["macro_average_precision"] is not None:
        delta_ap = float(best_metrics["macro_average_precision"] - baseline_metrics["macro_average_precision"])
    faiss_note = "Not used directly in this stage; memory probabilities come from exp0006."
    lines = [
        "# Source Probability-Mixing Recreation Report",
        "",
        "## Scope",
        "",
        "This report documents how to recreate the validation-only probability-mixing experiment stored at:",
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
        f"- Memory-evaluation root: `{memory_eval_root}`",
        f"- Baseline experiment: `{baseline_experiment_dir}`",
        f"- Query embedding root: `{query_embedding_root}`",
        f"- Manifest: `{manifest_csv}`",
        f"- Evaluation split: `{split}`",
        f"- Validation rows: `{row_count:,}`",
        f"- Label count: `{len(label_names)}`",
        f"- Label names: `{' '.join(label_names)}`",
        f"- Selection metric: `{DEFAULT_SELECTION_METRIC}`",
        f"- Alpha values: `{' '.join(format_alpha_value(value) for value in alpha_values)}`",
        "",
        "## Environment",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- NumPy: `{np.__version__}`",
        f"- PyTorch: `{torch.__version__}`",
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
        f"- The memory-only evaluation must already exist at `{memory_eval_root}`.",
        f"- The baseline experiment must already exist at `{baseline_experiment_dir}`.",
        f"- The query embeddings must already exist at `{query_embedding_root / split}`.",
        f"- The manifest must be present at `{manifest_csv}`.",
        "- The required Python packages must be importable: `numpy`, `torch`.",
        "",
        "## Input Summary",
        "",
        f"- Baseline checkpoint: `{baseline_experiment_dir / 'best.ckpt'}`",
        f"- Memory probabilities: `{memory_eval_root / 'val_probabilities.npy'}`",
        f"- Query embedding split: `{query_embedding_root / split}`",
        f"- Validation rows: `{row_count:,}`",
        f"- FAISS note: `{faiss_note}`",
        "",
        "## Sweep Summary",
        "",
        sweep_table,
        "",
        "## Best Configuration",
        "",
        f"- Best alpha: `{float(best_row['alpha']):.1f}`",
        f"- Validation macro AUROC: `{format_metric(best_metrics['macro_auroc'])}`",
        f"- Validation macro average precision: `{format_metric(best_metrics['macro_average_precision'])}`",
        f"- Validation macro ECE: `{format_metric(best_metrics['macro_ece'])}`",
        f"- Validation macro F1 @ 0.5: `{format_metric(best_metrics['macro_f1_at_0.5'])}`",
        f"- Diagnostic macro F1 @ tuned thresholds: `{format_metric(best_metrics['diagnostic_macro_f1_at_tuned_thresholds'])}`",
        "",
        "## Baseline Comparison",
        "",
        f"- Frozen baseline validation macro AUROC: `{format_metric(baseline_metrics['macro_auroc'])}`",
        f"- Frozen baseline validation macro average precision: `{format_metric(baseline_metrics['macro_average_precision'])}`",
        f"- Best mixed minus baseline macro AUROC: `{format_metric(delta_auroc)}`",
        f"- Best mixed minus baseline macro average precision: `{format_metric(delta_ap)}`",
        f"- Baseline reconstruction matches archived exp0004 forward metrics within 5e-4: `{str(baseline_comparison['matches_archived_metrics_within_5e-4']).lower()}`",
        f"- Baseline reconstruction max absolute metric delta: `{baseline_comparison['max_abs_delta']:.12f}`",
        "",
        "## Expected Outputs",
        "",
        "- `experiment_meta.json`",
        "- `recreation_report.md`",
        "- `sweep_results.json`",
        "- `best_config.json`",
        "- `best_val_metrics.json`",
        "- `val_mixed_probabilities.npy`",
        "- `probability_mixing_selection.md`",
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
            "- All selection in `exp0007` is validation-only.",
            "- `alpha=1.0` corresponds to the frozen baseline alone, and `alpha=0.0` corresponds to the selected memory-only probabilities from `exp0006`.",
            "- `val_mixed_probabilities.npy` stores the best-config mixed probabilities in validation row order.",
            "- Tied settings are resolved conservatively in favor of larger alpha.",
            "",
            "## Agent Handoff Text",
            "",
            "```text",
            (
                "Use /workspace/scripts/07_evaluate_probability_mixing.py and the report "
                f"{experiment_dir / 'recreation_report.md'} to recreate the validation-only probability-mixing stage "
                f"that combines {baseline_experiment_dir} with {memory_eval_root}. Reconstruct the frozen baseline "
                "validation probabilities, mix them with the exp0006 memory probabilities across alpha in [0.0, 1.0], "
                "and verify the saved best_config.json, best_val_metrics.json, and val_mixed_probabilities.npy artifacts."
            ),
            "```",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation-only probability mixing.")
    parser.add_argument("--memory-eval-root", type=Path, default=DEFAULT_MEMORY_EVAL_ROOT)
    parser.add_argument("--baseline-experiment-dir", type=Path, default=DEFAULT_BASELINE_EXPERIMENT_DIR)
    parser.add_argument("--query-embedding-root", type=Path, default=DEFAULT_QUERY_EMBEDDING_ROOT)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["val"])
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--alpha-values", type=float, nargs="+", default=ALPHA_GRID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    memory_eval_root = args.memory_eval_root.resolve()
    baseline_experiment_dir = args.baseline_experiment_dir.resolve()
    query_embedding_root = args.query_embedding_root.resolve()
    manifest_csv = args.manifest_csv.resolve()
    experiments_root = args.experiments_root.resolve()
    split = args.split
    alpha_values = [float(value) for value in unique_preserving_order(list(args.alpha_values))]
    if not alpha_values:
        raise SystemExit("Alpha sweep must contain at least one value.")
    if any(value < 0.0 or value > 1.0 for value in alpha_values):
        raise SystemExit(f"All alpha values must be in [0, 1]. Received: {alpha_values}")

    generated_slug = "nih_cxr14_exp0006_val_e100_p4"
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=args.overwrite,
    )

    script_path = Path(__file__).resolve()
    script_hash = script_sha256(script_path)

    label_columns, label_names, manifest_records = load_manifest_records(manifest_csv, split=split)
    val_embeddings, val_row_ids, val_sidecar_image_paths = load_query_split(query_embedding_root, split=split)
    val_labels, val_manifest_image_paths = build_labels_from_records(val_row_ids, manifest_records)
    validate_query_alignment(val_row_ids, val_sidecar_image_paths, val_manifest_image_paths)
    normalized_val_embeddings, query_norm_summary_before, query_norm_summary_after = normalize_rows(val_embeddings)

    memory_eval_meta = read_json(memory_eval_root / "experiment_meta.json")
    memory_best_config = read_json(memory_eval_root / "best_config.json")
    memory_best_metrics = read_json(memory_eval_root / "best_val_metrics.json")
    memory_probabilities = load_embedding_array(memory_eval_root / "val_probabilities.npy")
    if memory_probabilities.shape != val_labels.shape:
        raise SystemExit(
            f"Memory probability shape {memory_probabilities.shape} does not match validation labels {val_labels.shape}."
        )

    baseline_config = read_json(baseline_experiment_dir / "config.json")
    baseline_val_metrics_archived = read_json(baseline_experiment_dir / "val_metrics.json")
    baseline_checkpoint_path = baseline_experiment_dir / "best.ckpt"
    baseline_probabilities, baseline_reconstruction = reconstruct_baseline_probabilities(
        checkpoint_path=baseline_checkpoint_path,
        val_embeddings=normalized_val_embeddings,
        label_names=label_names,
        batch_size=int(args.batch_size),
    )
    if baseline_probabilities.shape != val_labels.shape:
        raise SystemExit(
            f"Baseline probability shape {baseline_probabilities.shape} does not match validation labels {val_labels.shape}."
        )
    baseline_metrics = evaluate_probabilities(val_labels, baseline_probabilities, label_names)
    baseline_comparison = compare_baseline_to_archived(baseline_metrics, baseline_val_metrics_archived, label_names)

    sweep_rows: list[dict[str, Any]] = []
    metrics_by_alpha: dict[str, Any] = {}
    mixed_by_alpha: dict[str, np.ndarray] = {}
    for alpha in alpha_values:
        mixed_probabilities = mix_probabilities(baseline_probabilities, memory_probabilities, float(alpha))
        metrics = evaluate_probabilities(val_labels, mixed_probabilities, label_names)
        row = {
            "alpha": float(alpha),
            "macro_auroc": metrics["macro_auroc"],
            "macro_average_precision": metrics["macro_average_precision"],
            "macro_ece": metrics["macro_ece"],
            "macro_f1_at_0.5": metrics["macro_f1_at_0.5"],
            "diagnostic_macro_f1_at_tuned_thresholds": metrics["diagnostic_macro_f1_at_tuned_thresholds"],
        }
        sweep_rows.append(row)
        alpha_key = format_alpha_value(float(alpha))
        metrics_by_alpha[alpha_key] = metrics
        mixed_by_alpha[alpha_key] = mixed_probabilities

    best_row, selection_trace = select_best_row(sweep_rows)
    best_alpha_key = format_alpha_value(float(best_row["alpha"]))
    best_metrics = metrics_by_alpha[best_alpha_key]
    best_mixed_probabilities = mixed_by_alpha[best_alpha_key]

    experiment_meta_path = experiment_dir / "experiment_meta.json"
    recreation_report_path = experiment_dir / "recreation_report.md"
    sweep_results_path = experiment_dir / "sweep_results.json"
    best_config_path = experiment_dir / "best_config.json"
    best_val_metrics_path = experiment_dir / "best_val_metrics.json"
    val_mixed_probabilities_path = experiment_dir / "val_mixed_probabilities.npy"
    selection_summary_path = experiment_dir / "probability_mixing_selection.md"

    write_json(
        sweep_results_path,
        {
            "alphas": sweep_rows,
            "alpha_values": alpha_values,
            "metrics_by_alpha": metrics_by_alpha,
            "baseline_metrics": baseline_metrics,
            "memory_best_config": memory_best_config,
            "memory_best_metrics": memory_best_metrics,
            "selection_trace": selection_trace,
        },
    )
    write_json(
        best_config_path,
        {
            "alpha": float(best_row["alpha"]),
            "selection_metric": DEFAULT_SELECTION_METRIC,
            "selection_trace": selection_trace,
            "memory_best_config": memory_best_config,
        },
    )
    write_json(best_val_metrics_path, best_metrics)
    np.save(val_mixed_probabilities_path, best_mixed_probabilities.astype(np.float32))

    delta_auroc = None
    if best_metrics["macro_auroc"] is not None and baseline_metrics["macro_auroc"] is not None:
        delta_auroc = float(best_metrics["macro_auroc"] - baseline_metrics["macro_auroc"])
    delta_ap = None
    if best_metrics["macro_average_precision"] is not None and baseline_metrics["macro_average_precision"] is not None:
        delta_ap = float(best_metrics["macro_average_precision"] - baseline_metrics["macro_average_precision"])

    summary_lines = [
        "# Probability Mixing Selection",
        "",
        "The canonical validation-only probability-mixing configuration for the current source stage is:",
        "",
        f"- alpha: `{format_alpha_value(float(best_row['alpha']))}`",
        f"- validation macro AUROC: `{format_metric(best_metrics['macro_auroc'])}`",
        f"- validation macro average precision: `{format_metric(best_metrics['macro_average_precision'])}`",
        f"- validation macro ECE: `{format_metric(best_metrics['macro_ece'])}`",
        f"- validation macro F1 @ 0.5: `{format_metric(best_metrics['macro_f1_at_0.5'])}`",
        f"- delta vs frozen baseline macro AUROC: `{format_metric(delta_auroc)}`",
        f"- delta vs frozen baseline macro average precision: `{format_metric(delta_ap)}`",
    ]
    selection_summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    baseline_meta_path = baseline_experiment_dir / "experiment_meta.json"
    baseline_meta = read_json(baseline_meta_path) if baseline_meta_path.exists() else None
    experiment_meta = {
        "argv": sys.argv,
        "baseline_experiment_dir": str(baseline_experiment_dir),
        "baseline_meta_path": str(baseline_meta_path),
        "baseline_reference": baseline_meta,
        "baseline_reconstruction": baseline_reconstruction,
        "baseline_reconstruction_metric_check": baseline_comparison,
        "experiment_dir": str(experiment_dir),
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_number": experiment_number,
        "label_columns": label_columns,
        "label_names": label_names,
        "manifest_csv": str(manifest_csv),
        "memory_eval_root": str(memory_eval_root),
        "memory_eval_reference": memory_eval_meta,
        "memory_best_config": memory_best_config,
        "memory_best_metrics": memory_best_metrics,
        "operation_label": DEFAULT_OPERATION_LABEL,
        "query_embedding_root": str(query_embedding_root),
        "query_norm_summary": {
            "raw": query_norm_summary_before,
            "normalized": query_norm_summary_after,
        },
        "run_date_utc": utc_now_iso(),
        "script_path": str(script_path),
        "script_sha256": script_hash,
        "seed": args.seed,
        "alpha_values": alpha_values,
        "selection_metric": DEFAULT_SELECTION_METRIC,
        "best_config": {
            "alpha": float(best_row["alpha"]),
        },
        "baseline_metrics": baseline_metrics,
        "best_metrics": best_metrics,
        "artifacts": {
            "experiment_meta": str(experiment_meta_path),
            "recreation_report": str(recreation_report_path),
            "sweep_results": str(sweep_results_path),
            "best_config": str(best_config_path),
            "best_val_metrics": str(best_val_metrics_path),
            "val_mixed_probabilities": str(val_mixed_probabilities_path),
            "selection_summary": str(selection_summary_path),
        },
    }
    write_json(experiment_meta_path, experiment_meta)

    argv_exact = [
        "python",
        str(script_path),
        "--memory-eval-root",
        str(memory_eval_root),
        "--baseline-experiment-dir",
        str(baseline_experiment_dir),
        "--query-embedding-root",
        str(query_embedding_root),
        "--manifest-csv",
        str(manifest_csv),
        "--split",
        split,
        "--batch-size",
        str(args.batch_size),
        "--alpha-values",
        *[format_alpha_value(value) for value in alpha_values],
        "--seed",
        str(args.seed),
        "--experiment-name",
        experiment_name,
        "--overwrite",
    ]
    argv_fresh = [
        "python",
        str(script_path),
        "--memory-eval-root",
        str(memory_eval_root),
        "--baseline-experiment-dir",
        str(baseline_experiment_dir),
        "--query-embedding-root",
        str(query_embedding_root),
        "--manifest-csv",
        str(manifest_csv),
        "--split",
        split,
        "--batch-size",
        str(args.batch_size),
        "--alpha-values",
        *[format_alpha_value(value) for value in alpha_values],
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
        val_mixed_probabilities_path,
        selection_summary_path,
    ]
    recreation_report = build_recreation_report(
        experiment_dir=experiment_dir,
        experiment_id=experiment_id,
        operation_label=DEFAULT_OPERATION_LABEL,
        script_path=script_path,
        argv_exact=argv_exact,
        argv_fresh=argv_fresh,
        memory_eval_root=memory_eval_root,
        baseline_experiment_dir=baseline_experiment_dir,
        query_embedding_root=query_embedding_root,
        manifest_csv=manifest_csv,
        split=split,
        row_count=int(val_embeddings.shape[0]),
        label_names=label_names,
        alpha_values=alpha_values,
        best_row=best_row,
        best_metrics=best_metrics,
        baseline_metrics=baseline_metrics,
        baseline_comparison=baseline_comparison,
        sweep_rows=sweep_rows,
        output_paths=output_paths,
    )
    recreation_report_path.write_text(recreation_report + "\n", encoding="utf-8")

    print(f"[saved] experiment_dir={experiment_dir}")
    print(
        "[best_config] "
        f"alpha={format_alpha_value(float(best_row['alpha']))} "
        f"macro_auroc={format_metric(best_metrics['macro_auroc'])} "
        f"macro_ap={format_metric(best_metrics['macro_average_precision'])} "
        f"macro_ece={format_metric(best_metrics['macro_ece'])} "
        f"macro_f1_0p5={format_metric(best_metrics['macro_f1_at_0.5'])}"
    )
    print(
        "[baseline_compare] "
        f"baseline_macro_auroc={format_metric(baseline_metrics['macro_auroc'])} "
        f"delta_vs_baseline={format_metric(delta_auroc)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
