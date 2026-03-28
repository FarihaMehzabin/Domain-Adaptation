#!/usr/bin/env python3
"""Train a frozen-embedding multilabel baseline for NIH CXR14."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import shlex
import sys
import time
from contextlib import nullcontext
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
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'torch'.") from exc


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_nih_cxr14_all14.csv")
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_WORKERS = 0
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5
DEFAULT_SEED = 1337
DEFAULT_DEVICE = "auto"
DEFAULT_OPERATION_LABEL = "source_baseline_training"
DEFAULT_EXPERIMENT_ID_WIDTH = 4
DEFAULT_SELECTION_METRIC = "macro_auroc"
DEFAULT_ECE_BINS = 15

AUTO_ID_COLUMNS = ("sample_id", "row_id", "report_id", "image_id", "id")
AUTO_PATH_COLUMNS = ("image_path", "report_path", "path")


@dataclass(frozen=True)
class ManifestRecord:
    split: str
    row_id: str
    image_path: str
    labels: tuple[float, ...]


@dataclass(frozen=True)
class SidecarSpec:
    relative_path: str
    format: str
    parser: str
    column: str | None = None


@dataclass
class SplitData:
    split: str
    split_dir: Path
    embeddings_path: Path
    embeddings_shape: tuple[int, int]
    sidecar: SidecarSpec
    raw_items: list[str]
    row_ids: list[str]
    labels: np.ndarray
    image_paths: list[str]


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_path: Path, labels: np.ndarray) -> None:
        self.embeddings_path = embeddings_path
        self.labels = np.asarray(labels, dtype=np.float32)
        self._embeddings: np.ndarray | None = None
        shape = np.load(embeddings_path, mmap_mode="r").shape
        if len(shape) != 2:
            raise SystemExit(f"Expected 2D embeddings in {embeddings_path}, found shape {shape}")
        self.num_rows = int(shape[0])

    def __len__(self) -> int:
        return self.num_rows

    def _array(self) -> np.ndarray:
        if self._embeddings is None:
            self._embeddings = np.load(self.embeddings_path, mmap_mode="r")
        return self._embeddings

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        features = np.asarray(self._array()[index], dtype=np.float32)
        return {
            "features": torch.from_numpy(features),
            "targets": torch.from_numpy(self.labels[index]),
        }


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_labels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
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
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_serializable(payload), sort_keys=True) + "\n")


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


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but CUDA is not available.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit("Requested --device mps, but MPS is not available.")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_autocast_context(device: torch.device, fp16_on_cuda: bool) -> Any:
    if bool(fp16_on_cuda and device.type == "cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


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
        SidecarSpec(relative_path="image_paths.txt", format="lines", parser="stem"),
        SidecarSpec(relative_path="report_ids.json", format="json_list", parser="identity"),
        SidecarSpec(relative_path="sample_ids.json", format="json_list", parser="identity"),
        SidecarSpec(relative_path="row_ids.json", format="json_list", parser="identity"),
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


def load_manifest_records(manifest_csv: Path) -> tuple[list[str], dict[str, dict[str, ManifestRecord]]]:
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

    by_split: dict[str, dict[str, ManifestRecord]] = {"train": {}, "val": {}, "test": {}}
    for row in reader:
        dataset = (row.get("dataset") or "").strip()
        if dataset and dataset != "nih_cxr14":
            continue
        split = (row.get("split") or "").strip().lower()
        if split not in by_split:
            continue
        image_path = (row.get("image_path") or "").strip()
        if not image_path:
            continue
        row_id = Path(image_path).stem
        labels = []
        for label_column in label_columns:
            raw_value = str(row.get(label_column) or "0").strip()
            try:
                labels.append(float(raw_value))
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid value '{raw_value}' in column '{label_column}' for row '{row_id}'."
                ) from exc
        if row_id in by_split[split]:
            raise SystemExit(f"Duplicate manifest row_id '{row_id}' found in split '{split}'.")
        by_split[split][row_id] = ManifestRecord(
            split=split,
            row_id=row_id,
            image_path=image_path,
            labels=tuple(labels),
        )
    return label_columns, by_split


def load_split_data(
    *,
    embedding_root: Path,
    split: str,
    manifest_records: dict[str, ManifestRecord],
    num_labels: int,
) -> SplitData:
    split_dir = embedding_root / split
    if not split_dir.exists():
        raise SystemExit(f"Split directory not found for '{split}': {split_dir}")

    embeddings_path = split_dir / "embeddings.npy"
    if not embeddings_path.exists():
        raise SystemExit(f"Missing embeddings.npy for split '{split}': {embeddings_path}")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise SystemExit(f"Expected 2D embeddings in {embeddings_path}, found {embeddings.shape}")

    sidecar = autodetect_sidecar(split_dir)
    sidecar_path = split_dir / sidecar.relative_path
    raw_items = read_sidecar_items(sidecar_path, format_name=sidecar.format, column=sidecar.column)
    if len(raw_items) != int(embeddings.shape[0]):
        raise SystemExit(
            f"Split '{split}' has {embeddings.shape[0]} embedding rows but {len(raw_items)} sidecar rows in {sidecar_path}."
        )

    row_ids = [parse_row_id(item, sidecar.parser) for item in raw_items]
    labels = np.zeros((len(row_ids), num_labels), dtype=np.float32)
    image_paths: list[str] = []
    missing: list[str] = []
    for idx, row_id in enumerate(row_ids):
        record = manifest_records.get(row_id)
        if record is None:
            if len(missing) < 5:
                missing.append(row_id)
            continue
        labels[idx] = np.asarray(record.labels, dtype=np.float32)
        image_paths.append(record.image_path)

    if missing:
        raise SystemExit(
            f"Split '{split}' contains row IDs that are missing from the manifest. Examples: {missing}"
        )

    if len(image_paths) != len(row_ids):
        raise SystemExit(f"Split '{split}' label alignment failed due to missing manifest rows.")

    return SplitData(
        split=split,
        split_dir=split_dir,
        embeddings_path=embeddings_path,
        embeddings_shape=(int(embeddings.shape[0]), int(embeddings.shape[1])),
        sidecar=sidecar,
        raw_items=raw_items,
        row_ids=row_ids,
        labels=labels,
        image_paths=image_paths,
    )


def build_dataloader(
    split_data: SplitData,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    dataset = EmbeddingDataset(split_data.embeddings_path, split_data.labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    positive = labels.sum(axis=0)
    negative = float(labels.shape[0]) - positive
    safe_positive = np.where(positive > 0.0, positive, 1.0)
    ratio = negative / safe_positive
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    ratio = np.where(positive > 0.0, ratio, 1.0)
    return torch.tensor(ratio.astype(np.float32))


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


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


def binary_f1_stats(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    preds = (np.asarray(probs, dtype=np.float64) >= float(threshold)).astype(np.int64)
    tp = float(np.sum((preds == 1) & (y == 1)))
    fp = float(np.sum((preds == 1) & (y == 0)))
    fn = float(np.sum((preds == 0) & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive_count": float(preds.sum()),
        "true_positive_count": float(y.sum()),
    }


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


def tune_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    label_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    thresholds = np.full((len(label_names),), 0.5, dtype=np.float32)
    label_payload: dict[str, Any] = {}
    for idx, label_name in enumerate(label_names):
        y = np.asarray(y_true[:, idx], dtype=np.int64)
        p = np.asarray(probs[:, idx], dtype=np.float64)
        positives = int(y.sum())
        if positives == 0:
            label_payload[label_name] = {
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
        fn = positives - tp
        denom = (2 * tp) + fp + fn
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=np.float64), where=denom > 0)
        best_index = int(np.argmax(f1))
        threshold = float(np.clip(p_sorted[best_index], 1e-6, 1.0 - 1e-6))
        thresholds[idx] = threshold
        label_payload[label_name] = {
            "threshold": threshold,
            "best_f1": float(f1[best_index]),
            "prevalence": float(y.mean()),
            "reason": "argmax_exact_f1_on_val",
        }

    macro_threshold = float(np.mean(thresholds)) if thresholds.size else None
    return thresholds, {
        "selection_split": "val",
        "selection_metric": "per_label_f1",
        "macro_threshold": macro_threshold,
        "labels": label_payload,
    }


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def summarize_split_metrics(
    *,
    split: str,
    loss: float,
    targets: np.ndarray,
    logits: np.ndarray,
    label_names: list[str],
    tuned_thresholds: np.ndarray,
) -> dict[str, Any]:
    probs = sigmoid_np(logits.astype(np.float64))
    label_metrics: dict[str, Any] = {}
    aurocs: list[float | None] = []
    aps: list[float | None] = []
    eces: list[float] = []
    f1_half: list[float] = []
    f1_tuned: list[float] = []

    for idx, label_name in enumerate(label_names):
        y = np.asarray(targets[:, idx], dtype=np.int64)
        p = np.asarray(probs[:, idx], dtype=np.float64)
        auroc = binary_auroc(y, p)
        ap = binary_average_precision(y, p)
        ece = binary_ece(y, p, num_bins=DEFAULT_ECE_BINS)
        half_stats = binary_f1_stats(y, p, 0.5)
        tuned_stats = binary_f1_stats(y, p, float(tuned_thresholds[idx]))
        aurocs.append(auroc)
        aps.append(ap)
        eces.append(ece)
        f1_half.append(half_stats["f1"])
        f1_tuned.append(tuned_stats["f1"])
        label_metrics[label_name] = {
            "prevalence": float(y.mean()),
            "positive_count": int(y.sum()),
            "negative_count": int(y.shape[0] - y.sum()),
            "auroc": auroc,
            "average_precision": ap,
            "ece": ece,
            "f1_at_0.5": half_stats["f1"],
            "f1_at_tuned_threshold": tuned_stats["f1"],
            "precision_at_0.5": half_stats["precision"],
            "recall_at_0.5": half_stats["recall"],
            "precision_at_tuned_threshold": tuned_stats["precision"],
            "recall_at_tuned_threshold": tuned_stats["recall"],
            "threshold_used": float(tuned_thresholds[idx]),
        }

    macro_auroc = mean_or_none(aurocs)
    macro_ap = mean_or_none(aps)
    macro_ece = mean_or_none(eces)
    macro_f1_half = mean_or_none(f1_half)
    macro_f1_tuned = mean_or_none(f1_tuned)

    return {
        "split": split,
        "num_examples": int(targets.shape[0]),
        "loss": float(loss),
        "macro": {
            "auroc": macro_auroc,
            "average_precision": macro_ap,
            "ece": macro_ece,
            "f1_at_0.5": macro_f1_half,
            "f1_at_tuned_thresholds": macro_f1_tuned,
        },
        "valid_label_counts": {
            "macro_auroc": int(sum(value is not None for value in aurocs)),
            "macro_average_precision": int(sum(value is not None for value in aps)),
        },
        "label_metrics": label_metrics,
    }


def train_one_epoch(
    *,
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    fp16_on_cuda: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with get_autocast_context(device, fp16_on_cuda):
            logits = model(features)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise SystemExit("Training loader produced zero examples.")
    return total_loss / total_examples


@torch.no_grad()
def evaluate_model(
    *,
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    fp16_on_cuda: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_chunks: list[torch.Tensor] = []
    targets_chunks: list[torch.Tensor] = []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        with get_autocast_context(device, fp16_on_cuda):
            logits = model(features)
            loss = criterion(logits, targets)
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        logits_chunks.append(logits.detach().cpu())
        targets_chunks.append(targets.detach().cpu())
    if total_examples == 0:
        raise SystemExit("Evaluation loader produced zero examples.")
    logits = torch.cat(logits_chunks, dim=0).numpy().astype(np.float32)
    targets = torch.cat(targets_chunks, dim=0).numpy().astype(np.float32)
    return total_loss / total_examples, logits, targets


def selection_tuple(summary: dict[str, Any]) -> tuple[float, float, float]:
    macro = summary["macro"]
    auroc = float(macro["auroc"]) if macro["auroc"] is not None else float("-inf")
    ap = float(macro["average_precision"]) if macro["average_precision"] is not None else float("-inf")
    loss = float(summary["loss"])
    return (auroc, ap, -loss)


def build_checkpoint_payload(
    *,
    model: nn.Module,
    epoch: int,
    best_summary: dict[str, Any],
    label_names: list[str],
    tuned_thresholds: np.ndarray | None,
) -> dict[str, Any]:
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return {
        "epoch": int(epoch),
        "state_dict": state_dict,
        "best_summary": to_serializable(best_summary),
        "label_names": list(label_names),
        "tuned_thresholds": tuned_thresholds.tolist() if tuned_thresholds is not None else None,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_size(num_bytes: int) -> str:
    units = ["B", "K", "M", "G", "T"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)}{unit}"
            if size >= 100:
                return f"{size:.0f}{unit}"
            if size >= 10:
                return f"{size:.1f}{unit}"
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def format_metric(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{float(value):.6f}"


def canonical_python_command(argv: list[str], *, experiment_name: str | None, overwrite: bool) -> list[str]:
    if not argv:
        raise SystemExit("argv is empty; cannot build recreation command.")
    result = [str(Path(argv[0]).resolve())]
    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--experiment-name":
            idx += 2
            continue
        if arg == "--overwrite":
            idx += 1
            continue
        result.append(arg)
        idx += 1
    if experiment_name is not None:
        result.extend(["--experiment-name", experiment_name])
    if overwrite:
        result.append("--overwrite")
    return ["python", *result]


def format_bash_command(argv: list[str]) -> str:
    return " \\\n  ".join(shlex.quote(part) for part in argv)


def render_recreation_report(
    *,
    experiment_dir: Path,
    config: dict[str, Any],
    experiment_meta: dict[str, Any],
    split_data: dict[str, SplitData],
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    threshold_payload: dict[str, Any],
    script_path: Path,
    script_sha256: str,
) -> str:
    same_dir_command = canonical_python_command(
        config["argv"],
        experiment_name=config["experiment_name"],
        overwrite=True,
    )
    fresh_command = canonical_python_command(
        config["argv"],
        experiment_name=strip_experiment_number_prefix(config["experiment_name"]),
        overwrite=False,
    )

    root_files = [
        experiment_dir / "config.json",
        experiment_dir / "experiment_meta.json",
        experiment_dir / "best.ckpt",
        experiment_dir / "val_metrics.json",
        experiment_dir / "test_metrics.json",
        experiment_dir / "val_f1_thresholds.json",
        experiment_dir / "train_log.jsonl",
    ]
    hash_lines = [
        f"{sha256_file(path)}  {path}"
        for path in root_files
        if path.exists()
    ]
    output_lines = [f"- `{path.name}`" for path in root_files if path.exists()]
    output_sizes = [f"- {path.name}: `{format_size(path.stat().st_size)}`" for path in root_files if path.exists()]

    environment_lines = [
        f"- Python: `{platform.python_version()}`",
        f"- NumPy: `{np.__version__}`",
        f"- PyTorch: `{torch.__version__}`",
        f"- CUDA available: `{str(torch.cuda.is_available()).lower()}`",
    ]
    if torch.cuda.is_available():
        environment_lines.append(f"- GPU used: `{torch.cuda.get_device_name(torch.cuda.current_device())}`")

    split_summary_lines: list[str] = []
    for split in ("train", "val", "test"):
        current = split_data[split]
        split_summary_lines.extend(
            [
                f"- {split.title()} rows: `{current.embeddings_shape[0]:,}`",
                f"- {split.title()} embedding dim: `{current.embeddings_shape[1]}`",
                f"- {split.title()} sidecar: `{current.sidecar.relative_path}`",
                f"- {split.title()} ID parser: `{current.sidecar.parser}`",
            ]
        )

    report_lines = [
        "# Source Baseline Recreation Report",
        "",
        "## Scope",
        "",
        "This report documents how to recreate the frozen-embedding multilabel baseline experiment stored at:",
        "",
        f"`{experiment_dir}`",
        "",
        "The producing script is:",
        "",
        f"`{script_path}`",
        "",
        "Script SHA-256:",
        "",
        f"`{script_sha256}`",
        "",
        "## Final Experiment Identity",
        "",
        f"- Experiment directory: `{experiment_dir}`",
        f"- Experiment id: `{config['experiment_id']}`",
        f"- Operation label: `{config['operation_label']}`",
        f"- Input embedding root: `{config['embedding_root']}`",
        f"- Manifest: `{config['manifest_csv']}`",
        f"- Label count: `{len(config['label_names'])}`",
        f"- Selection metric: `{config['selection_metric']}`",
        f"- Batch size: `{config['batch_size']}`",
        f"- Epoch budget: `{config['epochs']}`",
        f"- Patience: `{config['patience']}`",
        f"- Learning rate: `{config['lr']}`",
        f"- Weight decay: `{config['weight_decay']}`",
        f"- Seed: `{config['seed']}`",
        f"- Device requested: `{config['device_requested']}`",
        f"- Device resolved during run: `{config['device_resolved']}`",
        f"- Mixed precision on CUDA: `{str(config['fp16_on_cuda']).lower()}`",
        f"- Best epoch: `{experiment_meta['best_epoch']}`",
        "",
        "## Environment",
        "",
        *environment_lines,
        "",
        "## Exact Recreation Command",
        "",
        "If you want to recreate the same directory name in place, use this command:",
        "",
        "```bash",
        format_bash_command(same_dir_command),
        "```",
        "",
        "If you want a fresh numbered run instead of overwriting the existing directory, use:",
        "",
        "```bash",
        format_bash_command(fresh_command),
        "```",
        "",
        "## Preconditions",
        "",
        f"- The embedding experiment must already exist at `{config['embedding_root']}`.",
        f"- The manifest must be present at `{config['manifest_csv']}`.",
        "- The embedding experiment must contain `train/embeddings.npy`, `val/embeddings.npy`, and `test/embeddings.npy`.",
        "- Each split must have a row-identity sidecar such as `image_paths.txt` or `report_ids.json`.",
        "- The sidecar-derived row IDs must match `Path(image_path).stem` from the manifest for every split row.",
        "- The required Python packages must be importable: `numpy`, `torch`.",
        "",
        "## Input Summary",
        "",
        *split_summary_lines,
        "",
        "## Expected Outputs",
        "",
        *output_lines,
        "",
        "## Output Sizes",
        "",
        *output_sizes,
        "",
        "## Final Metrics",
        "",
        f"- Validation macro AUROC: `{format_metric(val_metrics['macro']['auroc'])}`",
        f"- Validation macro average precision: `{format_metric(val_metrics['macro']['average_precision'])}`",
        f"- Validation macro ECE: `{format_metric(val_metrics['macro']['ece'])}`",
        f"- Validation macro F1 @ 0.5: `{format_metric(val_metrics['macro']['f1_at_0.5'])}`",
        f"- Validation macro F1 @ tuned thresholds: `{format_metric(val_metrics['macro']['f1_at_tuned_thresholds'])}`",
        f"- Test macro AUROC: `{format_metric(test_metrics['macro']['auroc'])}`",
        f"- Test macro average precision: `{format_metric(test_metrics['macro']['average_precision'])}`",
        f"- Test macro ECE: `{format_metric(test_metrics['macro']['ece'])}`",
        f"- Test macro F1 @ 0.5: `{format_metric(test_metrics['macro']['f1_at_0.5'])}`",
        f"- Test macro F1 @ tuned thresholds: `{format_metric(test_metrics['macro']['f1_at_tuned_thresholds'])}`",
        f"- Macro mean tuned threshold: `{format_metric(threshold_payload['macro_threshold'])}`",
        "",
        "## Final Artifact SHA-256",
        "",
        "```text",
        *hash_lines,
        "```",
        "",
        "## Important Reproduction Notes",
        "",
        "- `config.json`, `experiment_meta.json`, and `recreation_report.md` include timestamps, so their hashes change on rerun.",
        "- Thresholds are tuned only on the validation split and then reused unchanged for test metrics.",
        "- The model is a single linear layer on top of frozen embeddings; the input embedding experiment determines the feature space.",
        "- The trainer auto-detects the split sidecar and aligns labels by sample ID rather than by trusting row position in the manifest.",
        "- After a successful run, keep the generated `recreation_report.md`, commit the experiment directory, and push before starting the next experiment.",
        "",
        "## Agent Handoff Text",
        "",
        "```text",
        (
            f"Use {script_path} and the report {experiment_dir / 'recreation_report.md'} to recreate "
            f"the frozen NIH CXR14 source baseline for embeddings from {config['embedding_root']}. "
            "Run the exact command in the report, verify the saved metrics and checkpoint hashes, "
            "and confirm that validation-tuned thresholds are reused for test evaluation."
        ),
        "```",
        "",
    ]
    return "\n".join(report_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen-embedding multilabel baseline on NIH CXR14 embeddings using a single linear head, "
            "validation-driven early stopping, and validation-tuned per-label thresholds."
        )
    )
    parser.add_argument("--embedding-root", type=Path, required=True, help="Embedding experiment root to train on.")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=DEFAULT_MANIFEST_CSV,
        help=f"Manifest CSV with NIH labels. Default: {DEFAULT_MANIFEST_CSV}",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=DEFAULT_EXPERIMENTS_ROOT,
        help=f"Where baseline experiment directories will be created. Default: {DEFAULT_EXPERIMENTS_ROOT}",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name. If omitted, a numbered name is generated.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Default: {DEFAULT_BATCH_SIZE}")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help=f"Default: {DEFAULT_NUM_WORKERS}")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Default: {DEFAULT_EPOCHS}")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"Default: {DEFAULT_LR}")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"Default: {DEFAULT_WEIGHT_DECAY}",
    )
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help=f"Default: {DEFAULT_PATIENCE}")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Default: {DEFAULT_SEED}")
    parser.add_argument(
        "--selection-metric",
        choices=("macro_auroc", "macro_average_precision"),
        default=DEFAULT_SELECTION_METRIC,
        help=f"Validation metric used for early stopping. Default: {DEFAULT_SELECTION_METRIC}",
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help=f"Default: {DEFAULT_DEVICE}")
    parser.add_argument("--fp16-on-cuda", action="store_true", help="Enable CUDA AMP during training and evaluation.")
    parser.add_argument("--overwrite", action="store_true", help="Reuse the target experiment directory if it exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0.")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.lr <= 0.0:
        raise SystemExit("--lr must be positive.")
    if args.weight_decay < 0.0:
        raise SystemExit("--weight-decay must be >= 0.")
    if args.patience < 0:
        raise SystemExit("--patience must be >= 0.")

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    manifest_csv = args.manifest_csv.resolve()
    embedding_root = args.embedding_root.resolve()
    label_columns, manifest_by_split = load_manifest_records(manifest_csv)
    label_names = [column.removeprefix("label_") for column in label_columns]

    split_data = {
        split: load_split_data(
            embedding_root=embedding_root,
            split=split,
            manifest_records=manifest_by_split[split],
            num_labels=len(label_columns),
        )
        for split in ("train", "val", "test")
    }

    embedding_dims = {payload.embeddings_shape[1] for payload in split_data.values()}
    if len(embedding_dims) != 1:
        raise SystemExit(f"Embedding dimensions differ across splits: {sorted(embedding_dims)}")
    input_dim = embedding_dims.pop()

    embedding_slug = slugify(embedding_root.name, fallback="embedding_root")
    generated_slug = f"{DEFAULT_OPERATION_LABEL}__{embedding_slug}__linear"
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=args.experiments_root.resolve(),
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=bool(args.overwrite),
    )

    train_loader = build_dataloader(
        split_data["train"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=True,
    )
    val_loader = build_dataloader(
        split_data["val"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
    )
    test_loader = build_dataloader(
        split_data["test"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
    )

    model = LinearProbe(input_dim=input_dim, num_labels=len(label_names)).to(device)
    pos_weight = compute_pos_weight(split_data["train"].labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16_on_cuda and device.type == "cuda"))

    script_path = Path(__file__).resolve()
    script_sha256 = sha256_file(script_path)
    config = {
        "argv": list(sys.argv),
        "run_date_utc": utc_now_iso(),
        "operation_label": DEFAULT_OPERATION_LABEL,
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "embedding_root": str(embedding_root),
        "manifest_csv": str(manifest_csv),
        "label_columns": label_columns,
        "label_names": label_names,
        "input_dim": int(input_dim),
        "num_labels": len(label_names),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "selection_metric": str(args.selection_metric),
        "device_requested": str(args.device),
        "device_resolved": str(device),
        "fp16_on_cuda": bool(args.fp16_on_cuda),
        "script_path": str(script_path),
        "script_sha256": script_sha256,
        "split_inputs": {
            split: {
                "embeddings_path": str(payload.embeddings_path),
                "num_rows": int(payload.embeddings_shape[0]),
                "embedding_dim": int(payload.embeddings_shape[1]),
                "sidecar_path": str(payload.split_dir / payload.sidecar.relative_path),
                "sidecar_format": payload.sidecar.format,
                "sidecar_parser": payload.sidecar.parser,
                "first_row_id": payload.row_ids[0] if payload.row_ids else None,
            }
            for split, payload in split_data.items()
        },
        "train_label_positive_counts": {
            label_name: int(split_data["train"].labels[:, idx].sum())
            for idx, label_name in enumerate(label_names)
        },
    }
    write_json(experiment_dir / "config.json", config)
    (experiment_dir / "train_log.jsonl").write_text("", encoding="utf-8")

    print(f"[info] experiment_dir={experiment_dir}")
    print(f"[info] embedding_root={embedding_root}")
    print(f"[info] input_dim={input_dim} labels={len(label_names)} device={device}")

    best_epoch = 0
    best_summary: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_loss = train_one_epoch(
            loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            fp16_on_cuda=bool(args.fp16_on_cuda),
        )
        val_loss, val_logits, val_targets = evaluate_model(
            loader=val_loader,
            model=model,
            criterion=criterion,
            device=device,
            fp16_on_cuda=bool(args.fp16_on_cuda),
        )
        current_summary = summarize_split_metrics(
            split="val",
            loss=val_loss,
            targets=val_targets,
            logits=val_logits,
            label_names=label_names,
            tuned_thresholds=np.full((len(label_names),), 0.5, dtype=np.float32),
        )
        improved = best_summary is None or selection_tuple(current_summary) > selection_tuple(best_summary)
        if improved:
            best_epoch = epoch
            best_summary = current_summary
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                build_checkpoint_payload(
                    model=model,
                    epoch=epoch,
                    best_summary=current_summary,
                    label_names=label_names,
                    tuned_thresholds=None,
                ),
                experiment_dir / "best.ckpt",
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_log = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_macro_auroc": current_summary["macro"]["auroc"],
            "val_macro_average_precision": current_summary["macro"]["average_precision"],
            "val_macro_ece": current_summary["macro"]["ece"],
            "val_macro_f1_at_0.5": current_summary["macro"]["f1_at_0.5"],
            "improved": bool(improved),
            "elapsed_sec": float(time.time() - epoch_started),
        }
        append_jsonl(experiment_dir / "train_log.jsonl", epoch_log)
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_macro_auroc={format_metric(current_summary['macro']['auroc'])} "
            f"improved={str(improved).lower()}"
        )

        if args.patience >= 0 and epochs_without_improvement > args.patience:
            print(f"[early-stop] epoch={epoch} patience={args.patience}")
            break

    if best_state is None or best_summary is None:
        raise SystemExit("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    val_loss, val_logits, val_targets = evaluate_model(
        loader=val_loader,
        model=model,
        criterion=criterion,
        device=device,
        fp16_on_cuda=bool(args.fp16_on_cuda),
    )
    test_loss, test_logits, test_targets = evaluate_model(
        loader=test_loader,
        model=model,
        criterion=criterion,
        device=device,
        fp16_on_cuda=bool(args.fp16_on_cuda),
    )

    val_probs = sigmoid_np(val_logits.astype(np.float64))
    tuned_thresholds, threshold_payload = tune_thresholds(val_targets, val_probs, label_names)
    val_metrics = summarize_split_metrics(
        split="val",
        loss=val_loss,
        targets=val_targets,
        logits=val_logits,
        label_names=label_names,
        tuned_thresholds=tuned_thresholds,
    )
    test_metrics = summarize_split_metrics(
        split="test",
        loss=test_loss,
        targets=test_targets,
        logits=test_logits,
        label_names=label_names,
        tuned_thresholds=tuned_thresholds,
    )

    torch.save(
        build_checkpoint_payload(
            model=model,
            epoch=best_epoch,
            best_summary=val_metrics,
            label_names=label_names,
            tuned_thresholds=tuned_thresholds,
        ),
        experiment_dir / "best.ckpt",
    )
    write_json(experiment_dir / "val_metrics.json", val_metrics)
    write_json(experiment_dir / "test_metrics.json", test_metrics)
    write_json(experiment_dir / "val_f1_thresholds.json", threshold_payload)

    experiment_meta = {
        "argv": list(sys.argv),
        "run_date_utc": utc_now_iso(),
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "operation_label": DEFAULT_OPERATION_LABEL,
        "embedding_root": str(embedding_root),
        "manifest_csv": str(manifest_csv),
        "device_resolved": str(device),
        "selection_metric": str(args.selection_metric),
        "best_epoch": int(best_epoch),
        "stopped_early": bool(best_epoch < args.epochs),
        "input_dim": int(input_dim),
        "num_labels": int(len(label_names)),
        "split_inputs": config["split_inputs"],
        "val_metrics_path": str(experiment_dir / "val_metrics.json"),
        "test_metrics_path": str(experiment_dir / "test_metrics.json"),
        "thresholds_path": str(experiment_dir / "val_f1_thresholds.json"),
        "checkpoint_path": str(experiment_dir / "best.ckpt"),
        "macro_metrics": {
            "val": val_metrics["macro"],
            "test": test_metrics["macro"],
        },
    }
    write_json(experiment_dir / "experiment_meta.json", experiment_meta)

    recreation_report = render_recreation_report(
        experiment_dir=experiment_dir,
        config=config,
        experiment_meta=experiment_meta,
        split_data=split_data,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        threshold_payload=threshold_payload,
        script_path=script_path,
        script_sha256=script_sha256,
    )
    (experiment_dir / "recreation_report.md").write_text(recreation_report, encoding="utf-8")

    print(
        "[done] "
        f"best_epoch={best_epoch} "
        f"val_macro_auroc={format_metric(val_metrics['macro']['auroc'])} "
        f"test_macro_auroc={format_metric(test_metrics['macro']['auroc'])} "
        f"output_dir={experiment_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
