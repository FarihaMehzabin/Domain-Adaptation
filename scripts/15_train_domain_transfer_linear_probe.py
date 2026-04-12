#!/usr/bin/env python3
"""Train and evaluate an image-only multilabel head for D0/D1/D2 transfer."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shlex
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import experiment_layout
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest/manifest_common_labels_nih_train_val_test_chexpert_mimic.csv")
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_WORKERS = 0
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5
DEFAULT_SEED = 1337
DEFAULT_DEVICE = "auto"
DEFAULT_OPERATION_LABEL = "domain_transfer_head_training"
DEFAULT_EXPERIMENT_ID_WIDTH = 4
DEFAULT_ECE_BINS = 15
DEFAULT_HEAD_TYPE = "linear"
DEFAULT_MLP_DROPOUT = 0.2


@dataclass(frozen=True)
class ManifestRecord:
    domain: str
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
    alias: str
    domain: str
    split: str
    split_dir: Path
    embeddings_path: Path
    embeddings_shape: tuple[int, ...]
    sidecar: SidecarSpec
    row_ids: list[str]
    labels: np.ndarray
    image_paths: list[str]


@dataclass(frozen=True)
class EvaluationPlan:
    name: str
    train_alias: str
    selection_alias: str
    primary_test_alias: str | None
    split_specs: tuple[tuple[str, str, str], ...]
    output_name_map: dict[str, str]
    thresholds_filename: str


AUTO_ID_COLUMNS = ("row_id", "sample_id", "image_id", "id")
AUTO_PATH_COLUMNS = ("image_path", "report_path", "path")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


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


def extract_experiment_number(name: str) -> int | None:
    if not name.startswith("exp"):
        return None
    prefix = name.split("__", 1)[0]
    digits = prefix.removeprefix("exp")
    if not digits.isdigit():
        return None
    return int(digits)


def next_experiment_number(experiments_root: Path) -> int:
    return experiment_layout.next_experiment_number(experiments_root)


def resolve_experiment_identity(
    *,
    experiments_root: Path,
    requested_name: str | None,
    generated_slug: str,
    overwrite: bool,
    id_width: int = DEFAULT_EXPERIMENT_ID_WIDTH,
) -> tuple[int, str, str, Path]:
    requested = (requested_name or "").strip() or None
    base_name = ensure_operation_prefix(requested or generated_slug)
    return experiment_layout.resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=base_name if requested else None,
        generated_slug=base_name,
        overwrite=overwrite,
        id_width=id_width,
    )


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
    reader = csv.DictReader(text.splitlines())
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
    reader = csv.DictReader(text.splitlines())
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
    ]
    for candidate in preferred_exact:
        if (split_dir / candidate.relative_path).exists():
            return candidate

    json_candidates = sorted(path for path in split_dir.glob("*_ids.json") if path.is_file())
    if len(json_candidates) == 1:
        return SidecarSpec(relative_path=json_candidates[0].name, format="json_list", parser="identity")
    if len(json_candidates) > 1:
        names = ", ".join(path.name for path in json_candidates)
        raise SystemExit(f"Ambiguous sidecar in {split_dir}: multiple *_ids.json files found ({names}).")

    txt_candidates = sorted(path for path in split_dir.glob("*_paths.txt") if path.is_file())
    if len(txt_candidates) == 1:
        return SidecarSpec(relative_path=txt_candidates[0].name, format="lines", parser="stem")
    if len(txt_candidates) > 1:
        names = ", ".join(path.name for path in txt_candidates)
        raise SystemExit(f"Ambiguous sidecar in {split_dir}: multiple *_paths.txt files found ({names}).")

    csv_candidates = sorted(path for path in split_dir.glob("*manifest.csv") if path.is_file())
    for candidate in csv_candidates:
        sidecar = choose_csv_sidecar(candidate)
        if sidecar is not None:
            return sidecar
    raise SystemExit(f"Could not auto-detect a row identity sidecar in {split_dir}.")


def load_manifest_records(manifest_csv: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, ManifestRecord]]]:
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV not found: {manifest_csv}")
    text = manifest_csv.read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise SystemExit(f"Manifest CSV has no header: {manifest_csv}")
    required_columns = {"domain", "split", "row_id", "image_path"}
    if not required_columns.issubset(reader.fieldnames):
        raise SystemExit(f"Manifest CSV must contain columns: {sorted(required_columns)}")
    label_columns = [field for field in reader.fieldnames if field.startswith("label_")]
    if not label_columns:
        raise SystemExit("Manifest CSV does not contain any label_... columns.")

    by_key: dict[tuple[str, str], dict[str, ManifestRecord]] = {}
    for row in reader:
        domain = (row.get("domain") or "").strip()
        split = (row.get("split") or "").strip().lower()
        row_id = (row.get("row_id") or "").strip()
        image_path = (row.get("image_path") or "").strip()
        if not domain or not split or not row_id or not image_path:
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
        key = (domain, split)
        current = by_key.setdefault(key, {})
        if row_id in current:
            raise SystemExit(f"Duplicate manifest row_id '{row_id}' found in domain={domain} split={split}.")
        current[row_id] = ManifestRecord(
            domain=domain,
            split=split,
            row_id=row_id,
            image_path=image_path,
            labels=tuple(labels),
        )
    return [field.removeprefix("label_") for field in label_columns], by_key


def resolve_split_dir(embedding_root: Path, *, embedding_layout: str, domain: str, split: str) -> Path:
    if embedding_layout == "domain_split":
        return embedding_root / domain / split
    if embedding_layout == "source_only":
        if domain != "d0_nih":
            raise SystemExit(
                f"embedding-layout=source_only only supports D0 NIH source evaluation, got domain={domain}."
            )
        return embedding_root / split
    raise SystemExit(f"Unsupported --embedding-layout: {embedding_layout}")


def load_split_data(
    *,
    alias: str,
    embedding_root: Path,
    embedding_layout: str,
    domain: str,
    split: str,
    manifest_records: dict[str, ManifestRecord],
    num_labels: int,
    max_rows: int | None,
) -> SplitData:
    split_dir = resolve_split_dir(embedding_root, embedding_layout=embedding_layout, domain=domain, split=split)
    if not split_dir.exists():
        raise SystemExit(f"Split directory not found for alias={alias}: {split_dir}")
    embeddings_path = split_dir / "embeddings.npy"
    if not embeddings_path.exists():
        raise SystemExit(f"Missing embeddings.npy for alias={alias}: {embeddings_path}")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.ndim not in {2, 3}:
        raise SystemExit(
            f"Expected 2D or 3D embeddings in {embeddings_path}, found shape {embeddings.shape}"
        )

    sidecar = autodetect_sidecar(split_dir)
    raw_items = read_sidecar_items(
        split_dir / sidecar.relative_path,
        format_name=sidecar.format,
        column=sidecar.column,
    )
    if len(raw_items) != int(embeddings.shape[0]):
        raise SystemExit(
            f"Alias={alias} has {embeddings.shape[0]} embedding rows but {len(raw_items)} sidecar rows."
        )
    row_ids = [parse_row_id(item, sidecar.parser) for item in raw_items]
    if max_rows is not None:
        row_ids = row_ids[:max_rows]

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
            f"Alias={alias} contains row IDs missing from the manifest. Examples: {missing}"
        )
    if len(image_paths) != len(row_ids):
        raise SystemExit(f"Alias={alias} label alignment failed due to missing manifest rows.")

    effective_shape = (len(row_ids), *tuple(int(dim) for dim in embeddings.shape[1:]))
    return SplitData(
        alias=alias,
        domain=domain,
        split=split,
        split_dir=split_dir,
        embeddings_path=embeddings_path,
        embeddings_shape=effective_shape,
        sidecar=sidecar,
        row_ids=row_ids,
        labels=labels,
        image_paths=image_paths,
    )


def normalize_mlp_hidden_dims(raw_dims: list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    if not raw_dims:
        return tuple()
    dims = tuple(int(dim) for dim in raw_dims)
    for dim in dims:
        if dim <= 0:
            raise SystemExit("--mlp-hidden-dims values must be positive.")
    return dims


def format_float_slug(value: float) -> str:
    return str(value).replace(".", "p")


def pool_feature_row(feature: np.ndarray, pooling: str) -> np.ndarray:
    array = np.asarray(feature, dtype=np.float32)
    if array.ndim == 1:
        if pooling in {"avg", "cls", "flatten"}:
            return array
        raise SystemExit(f"Unsupported token pooling mode for 1D features: {pooling}")
    if array.ndim != 2:
        raise SystemExit(f"Expected a 1D or 2D feature row, found shape {array.shape}")
    if pooling == "avg":
        return array.mean(axis=0).astype(np.float32)
    if pooling == "cls":
        return array[0].astype(np.float32)
    if pooling == "flatten":
        return array.reshape(-1).astype(np.float32)
    raise SystemExit(f"Unsupported --token-pooling value: {pooling}")


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        embeddings_path: Path,
        labels: np.ndarray,
        *,
        num_rows: int,
        token_pooling: str,
        l2_normalize_features: bool,
    ) -> None:
        self.embeddings_path = embeddings_path
        self.labels = np.asarray(labels, dtype=np.float32)
        self.num_rows = int(num_rows)
        self.token_pooling = token_pooling
        self.l2_normalize_features = l2_normalize_features
        self._embeddings: np.ndarray | None = None
        shape = np.load(embeddings_path, mmap_mode="r").shape
        if shape[0] < self.num_rows:
            raise SystemExit(
                f"Requested {self.num_rows} rows from {embeddings_path}, but only {shape[0]} are available."
            )

    def __len__(self) -> int:
        return self.num_rows

    def _array(self) -> np.ndarray:
        if self._embeddings is None:
            self._embeddings = np.load(self.embeddings_path, mmap_mode="r")
        return self._embeddings

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raw_feature = np.asarray(self._array()[index], dtype=np.float32)
        feature = np.array(pool_feature_row(raw_feature, self.token_pooling), dtype=np.float32, copy=True)
        if self.l2_normalize_features:
            norm = float(np.linalg.norm(feature))
            if norm > 0.0:
                feature = feature / norm
        return {
            "features": torch.from_numpy(feature),
            "targets": torch.from_numpy(self.labels[index]),
        }


def build_dataloader(
    split_data: SplitData,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
    token_pooling: str,
    l2_normalize_features: bool,
) -> DataLoader:
    dataset = EmbeddingDataset(
        split_data.embeddings_path,
        split_data.labels,
        num_rows=len(split_data.row_ids),
        token_pooling=token_pooling,
        l2_normalize_features=l2_normalize_features,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_labels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class SmallMLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], num_labels: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            linear = nn.Linear(previous_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.ReLU(), nn.Dropout(dropout)])
            previous_dim = hidden_dim
        output = nn.Linear(previous_dim, num_labels)
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.classifier = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def build_probe_model(
    *,
    head_type: str,
    input_dim: int,
    num_labels: int,
    mlp_hidden_dims: tuple[int, ...],
    mlp_dropout: float,
) -> nn.Module:
    if head_type == "linear":
        return LinearProbe(input_dim=input_dim, num_labels=num_labels)
    if head_type == "mlp":
        return SmallMLPProbe(
            input_dim=input_dim,
            hidden_dims=mlp_hidden_dims,
            num_labels=num_labels,
            dropout=mlp_dropout,
        )
    raise SystemExit(f"Unsupported --head-type value: {head_type}")


def describe_head(
    *,
    head_type: str,
    mlp_hidden_dims: tuple[int, ...],
    mlp_dropout: float,
) -> str:
    if head_type == "linear":
        return "linear"
    dims = "x".join(str(dim) for dim in mlp_hidden_dims)
    return f"mlp(hidden={dims},dropout={mlp_dropout})"


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


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
    return float(precision[y_sorted == 1].sum() / positives)


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
        mask = (p >= left) & (p <= right) if idx == num_bins - 1 else (p >= left) & (p < right)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(p[mask].mean())
        empirical = float(y[mask].mean())
        ece += (count / total) * abs(confidence - empirical)
    return float(ece)


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def tune_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    label_names: list[str],
    *,
    selection_split: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    thresholds = np.full((len(label_names),), 0.5, dtype=np.float32)
    payload: dict[str, Any] = {}
    for idx, label_name in enumerate(label_names):
        y = np.asarray(y_true[:, idx], dtype=np.int64)
        p = np.asarray(probs[:, idx], dtype=np.float64)
        positives = int(y.sum())
        if positives == 0:
            payload[label_name] = {"threshold": 0.5, "best_f1": 0.0, "reason": "no_positive_examples"}
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
        payload[label_name] = {"threshold": threshold, "best_f1": float(f1[best_index])}
    return thresholds, {
        "selection_split": selection_split,
        "selection_metric": "per_label_f1",
        "macro_threshold": float(thresholds.mean()) if thresholds.size else None,
        "labels": payload,
    }


def summarize_split_metrics(
    *,
    split_alias: str,
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
    return {
        "split_alias": split_alias,
        "num_examples": int(targets.shape[0]),
        "loss": float(loss),
        "macro": {
            "auroc": mean_or_none(aurocs),
            "average_precision": mean_or_none(aps),
            "ece": mean_or_none(eces),
            "f1_at_0.5": mean_or_none(f1_half),
            "f1_at_tuned_thresholds": mean_or_none(f1_tuned),
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


def infer_input_dim(embeddings_path: Path, token_pooling: str, l2_normalize_features: bool) -> int:
    sample_array = np.load(embeddings_path, mmap_mode="r")
    if sample_array.shape[0] == 0:
        raise SystemExit(f"No embedding rows available in {embeddings_path}")
    sample = np.asarray(sample_array[0], dtype=np.float32)
    feature = pool_feature_row(sample, token_pooling)
    if l2_normalize_features:
        norm = float(np.linalg.norm(feature))
        if norm > 0.0:
            feature = feature / norm
    return int(feature.shape[0])


def format_metric(value: float | None) -> str:
    return "null" if value is None else f"{float(value):.6f}"


def format_bash_command(argv: list[str]) -> str:
    return " \\\n  ".join(shlex.quote(part) for part in argv)


def build_evaluation_plan(*, split_profile: str, embedding_layout: str) -> EvaluationPlan:
    if split_profile == "source_transfer":
        split_specs: list[tuple[str, str, str]] = [
            ("d0_train", "d0_nih", "train"),
            ("d0_val", "d0_nih", "val"),
            ("d0_test", "d0_nih", "test"),
        ]
        output_name_map = {
            "d0_val": "d0_val_metrics.json",
            "d0_test": "d0_test_metrics.json",
        }
        if embedding_layout == "domain_split":
            split_specs.extend(
                [
                    ("d1_transfer", "d1_chexpert", "val"),
                    ("d2_transfer", "d2_mimic", "test"),
                ]
            )
            output_name_map.update(
                {
                    "d1_transfer": "d1_transfer_metrics.json",
                    "d2_transfer": "d2_transfer_metrics.json",
                }
            )
        return EvaluationPlan(
            name=split_profile,
            train_alias="d0_train",
            selection_alias="d0_val",
            primary_test_alias="d0_test",
            split_specs=tuple(split_specs),
            output_name_map=output_name_map,
            thresholds_filename="d0_val_f1_thresholds.json",
        )

    if split_profile in {"chexpert_target", "mimic_target"}:
        if embedding_layout != "domain_split":
            raise SystemExit(f"--split-profile {split_profile} requires --embedding-layout domain_split.")
        target_domain = "d1_chexpert" if split_profile == "chexpert_target" else "d2_mimic"
        return EvaluationPlan(
            name=split_profile,
            train_alias="target_train",
            selection_alias="target_val",
            primary_test_alias="target_test",
            split_specs=(
                ("target_train", target_domain, "train"),
                ("target_val", target_domain, "val"),
                ("target_test", target_domain, "test"),
            ),
            output_name_map={
                "target_val": "target_val_metrics.json",
                "target_test": "target_test_metrics.json",
            },
            thresholds_filename="target_val_f1_thresholds.json",
        )

    raise SystemExit(f"Unsupported --split-profile value: {split_profile}")


def render_recreation_report(
    *,
    experiment_dir: Path,
    config: dict[str, Any],
    split_data: dict[str, SplitData],
    metrics_by_alias: dict[str, dict[str, Any]],
) -> str:
    split_lines: list[str] = []
    for alias, current in split_data.items():
        split_lines.append(
            f"- `{alias}` -> `{current.split_dir}` with `{len(current.row_ids)}` rows and shape `{list(current.embeddings_shape)}`"
        )
    metric_lines: list[str] = []
    for alias in sorted(metrics_by_alias):
        summary = metrics_by_alias[alias]
        metric_lines.append(
            f"- `{alias}` macro AUROC `{format_metric(summary['macro']['auroc'])}`, "
            f"macro AP `{format_metric(summary['macro']['average_precision'])}`"
        )
    return "\n".join(
        [
            "# Domain Transfer Head Recreation Report",
            "",
            "## Scope",
            "",
            f"- Experiment directory: `{experiment_dir}`",
            f"- Embedding root: `{config['embedding_root']}`",
            f"- Manifest: `{config['manifest_csv']}`",
            f"- Embedding layout: `{config['embedding_layout']}`",
            f"- Token pooling: `{config['token_pooling']}`",
            f"- Head type: `{config['head_type']}`",
            f"- MLP hidden dims: `{config['mlp_hidden_dims']}`",
            f"- MLP dropout: `{config['mlp_dropout']}`",
            "",
            "## Recreation Command",
            "",
            "```bash",
            format_bash_command(["python", *config["argv"]]),
            "```",
            "",
            "## Split Inputs",
            "",
            *split_lines,
            "",
            "## Final Metrics",
            "",
            *metric_lines,
            "",
            "## Notes",
            "",
            f"- Training uses only `{config['train_alias']}` embeddings.",
            f"- Early stopping is driven by `{config['selection_alias']}` macro AUROC.",
            f"- `{config['selection_alias']}`-tuned thresholds are reused unchanged for later evaluation splits.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen image-only multilabel linear or MLP head on D0 embeddings and evaluate direct transfer to D1/D2."
        )
    )
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--split-profile",
        choices=("source_transfer", "chexpert_target", "mimic_target"),
        default="source_transfer",
        help="Which manifest split layout to train and evaluate.",
    )
    parser.add_argument(
        "--embedding-layout",
        choices=("domain_split", "source_only"),
        default="domain_split",
        help="`domain_split` expects root/domain/split directories. `source_only` maps root/train|val|test to D0 only.",
    )
    parser.add_argument(
        "--token-pooling",
        choices=("avg", "cls", "flatten"),
        default="avg",
        help="How to pool per-image token grids before the linear head. Ignored for already 1D rows.",
    )
    parser.add_argument("--l2-normalize-features", action="store_true")
    parser.add_argument("--head-type", choices=("linear", "mlp"), default=DEFAULT_HEAD_TYPE)
    parser.add_argument(
        "--mlp-hidden-dims",
        type=int,
        nargs="*",
        default=None,
        help="Hidden layer sizes for --head-type mlp. Leave unset for linear.",
    )
    parser.add_argument("--mlp-dropout", type=float, default=DEFAULT_MLP_DROPOUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--fp16-on-cuda", action="store_true")
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0.")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive.")
    if args.weight_decay < 0:
        raise SystemExit("--weight-decay must be >= 0.")
    if args.patience < 0:
        raise SystemExit("--patience must be >= 0.")
    if not (0.0 <= float(args.mlp_dropout) < 1.0):
        raise SystemExit("--mlp-dropout must be in [0.0, 1.0).")
    if args.max_rows_per_split is not None and args.max_rows_per_split <= 0:
        raise SystemExit("--max-rows-per-split must be positive when provided.")
    mlp_hidden_dims = normalize_mlp_hidden_dims(args.mlp_hidden_dims)
    if args.head_type == "linear" and mlp_hidden_dims:
        raise SystemExit("--mlp-hidden-dims is only valid with --head-type mlp.")
    if args.head_type == "mlp" and not mlp_hidden_dims:
        raise SystemExit("--head-type mlp requires at least one --mlp-hidden-dims value.")

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    manifest_csv = args.manifest_csv.resolve()
    embedding_root = args.embedding_root.resolve()
    label_names, manifest_by_key = load_manifest_records(manifest_csv)
    evaluation_plan = build_evaluation_plan(
        split_profile=args.split_profile,
        embedding_layout=args.embedding_layout,
    )

    split_data: dict[str, SplitData] = {}
    for alias, domain, split in evaluation_plan.split_specs:
        manifest_records = manifest_by_key.get((domain, split))
        if manifest_records is None:
            raise SystemExit(f"Manifest does not contain records for domain={domain} split={split}.")
        split_data[alias] = load_split_data(
            alias=alias,
            embedding_root=embedding_root,
            embedding_layout=args.embedding_layout,
            domain=domain,
            split=split,
            manifest_records=manifest_records,
            num_labels=len(label_names),
            max_rows=args.max_rows_per_split,
        )

    input_dim = infer_input_dim(
        split_data[evaluation_plan.train_alias].embeddings_path,
        token_pooling=args.token_pooling,
        l2_normalize_features=bool(args.l2_normalize_features),
    )
    generated_slug = "__".join(
        [
            DEFAULT_OPERATION_LABEL,
            slugify(embedding_root.name, fallback="embedding-root"),
            f"profile-{args.split_profile}",
            f"layout-{args.embedding_layout}",
            f"pool-{args.token_pooling}",
            f"head-{args.head_type}",
            f"hidden-{'x'.join(str(dim) for dim in mlp_hidden_dims) if mlp_hidden_dims else 'none'}",
            f"dropout-{format_float_slug(float(args.mlp_dropout))}",
            f"dim-{input_dim}",
        ]
    )
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=args.experiments_root.resolve(),
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=bool(args.overwrite),
    )

    train_loader = build_dataloader(
        split_data[evaluation_plan.train_alias],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=True,
        token_pooling=args.token_pooling,
        l2_normalize_features=bool(args.l2_normalize_features),
    )
    eval_loaders = {
        alias: build_dataloader(
            payload,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            shuffle=False,
            token_pooling=args.token_pooling,
            l2_normalize_features=bool(args.l2_normalize_features),
        )
        for alias, payload in split_data.items()
        if alias != evaluation_plan.train_alias
    }

    model = build_probe_model(
        head_type=args.head_type,
        input_dim=input_dim,
        num_labels=len(label_names),
        mlp_hidden_dims=mlp_hidden_dims,
        mlp_dropout=float(args.mlp_dropout),
    ).to(device)
    pos_weight = compute_pos_weight(split_data[evaluation_plan.train_alias].labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16_on_cuda and device.type == "cuda"))
    model_num_parameters = count_parameters(model)

    config = {
        "argv": list(sys.argv),
        "run_date_utc": utc_now_iso(),
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "embedding_root": str(embedding_root),
        "manifest_csv": str(manifest_csv),
        "split_profile": args.split_profile,
        "embedding_layout": args.embedding_layout,
        "token_pooling": args.token_pooling,
        "l2_normalize_features": bool(args.l2_normalize_features),
        "head_type": args.head_type,
        "head_description": describe_head(
            head_type=args.head_type,
            mlp_hidden_dims=mlp_hidden_dims,
            mlp_dropout=float(args.mlp_dropout),
        ),
        "mlp_hidden_dims": list(mlp_hidden_dims),
        "mlp_dropout": float(args.mlp_dropout),
        "label_names": label_names,
        "input_dim": input_dim,
        "model_num_parameters": model_num_parameters,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "device_requested": str(args.device),
        "device_resolved": str(device),
        "fp16_on_cuda": bool(args.fp16_on_cuda),
        "max_rows_per_split": int(args.max_rows_per_split) if args.max_rows_per_split is not None else None,
        "train_alias": evaluation_plan.train_alias,
        "selection_alias": evaluation_plan.selection_alias,
        "primary_test_alias": evaluation_plan.primary_test_alias,
        "split_inputs": {
            alias: {
                "domain": payload.domain,
                "split": payload.split,
                "embeddings_path": str(payload.embeddings_path),
                "num_rows": len(payload.row_ids),
                "shape": list(payload.embeddings_shape),
                "sidecar_path": str(payload.split_dir / payload.sidecar.relative_path),
                "sidecar_parser": payload.sidecar.parser,
            }
            for alias, payload in split_data.items()
        },
    }
    write_json(experiment_dir / "config.json", config)
    (experiment_dir / "train_log.jsonl").write_text("", encoding="utf-8")

    print(f"[info] experiment_dir={experiment_dir}")
    print(f"[info] embedding_root={embedding_root}")
    print(
        f"[info] input_dim={input_dim} labels={len(label_names)} device={device} "
        f"head={config['head_description']} params={model_num_parameters}"
    )

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
        selection_loss, selection_logits, selection_targets = evaluate_model(
            loader=eval_loaders[evaluation_plan.selection_alias],
            model=model,
            criterion=criterion,
            device=device,
            fp16_on_cuda=bool(args.fp16_on_cuda),
        )
        current_summary = summarize_split_metrics(
            split_alias=evaluation_plan.selection_alias,
            loss=selection_loss,
            targets=selection_targets,
            logits=selection_logits,
            label_names=label_names,
            tuned_thresholds=np.full((len(label_names),), 0.5, dtype=np.float32),
        )
        improved = best_summary is None or selection_tuple(current_summary) > selection_tuple(best_summary)
        if improved:
            best_epoch = epoch
            best_summary = current_summary
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "epoch": best_epoch,
                    "state_dict": best_state,
                    "best_summary": best_summary,
                    "label_names": label_names,
                    "head_type": args.head_type,
                    "mlp_hidden_dims": list(mlp_hidden_dims),
                    "mlp_dropout": float(args.mlp_dropout),
                    "input_dim": input_dim,
                    "num_labels": len(label_names),
                },
                experiment_dir / "best.ckpt",
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        append_jsonl(
            experiment_dir / "train_log.jsonl",
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "selection_alias": evaluation_plan.selection_alias,
                f"{evaluation_plan.selection_alias}_loss": float(selection_loss),
                f"{evaluation_plan.selection_alias}_macro_auroc": current_summary["macro"]["auroc"],
                f"{evaluation_plan.selection_alias}_macro_average_precision": current_summary["macro"]["average_precision"],
                "improved": bool(improved),
                "elapsed_sec": float(time.time() - epoch_started),
            },
        )
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"{evaluation_plan.selection_alias}_loss={selection_loss:.6f} "
            f"{evaluation_plan.selection_alias}_macro_auroc={format_metric(current_summary['macro']['auroc'])} "
            f"improved={str(improved).lower()}"
        )
        if epochs_without_improvement > args.patience:
            print(f"[early-stop] epoch={epoch} patience={args.patience}")
            break

    if best_state is None or best_summary is None:
        raise SystemExit("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    raw_eval_results: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for alias, loader in eval_loaders.items():
        raw_eval_results[alias] = evaluate_model(
            loader=loader,
            model=model,
            criterion=criterion,
            device=device,
            fp16_on_cuda=bool(args.fp16_on_cuda),
        )

    selection_loss, selection_logits, selection_targets = raw_eval_results[evaluation_plan.selection_alias]
    selection_probs = sigmoid_np(selection_logits.astype(np.float64))
    tuned_thresholds, threshold_payload = tune_thresholds(
        selection_targets,
        selection_probs,
        label_names,
        selection_split=evaluation_plan.selection_alias,
    )

    metrics_by_alias: dict[str, dict[str, Any]] = {}
    for alias, (loss, logits, targets) in raw_eval_results.items():
        metrics = summarize_split_metrics(
            split_alias=alias,
            loss=loss,
            targets=targets,
            logits=logits,
            label_names=label_names,
            tuned_thresholds=tuned_thresholds,
        )
        metrics_by_alias[alias] = metrics
        output_name = evaluation_plan.output_name_map.get(alias, f"{alias}_metrics.json")
        write_json(experiment_dir / output_name, metrics)

    write_json(experiment_dir / evaluation_plan.thresholds_filename, threshold_payload)
    torch.save(
        {
            "epoch": best_epoch,
            "state_dict": best_state,
            "best_summary": metrics_by_alias[evaluation_plan.selection_alias],
            "label_names": label_names,
            "tuned_thresholds": tuned_thresholds.tolist(),
            "token_pooling": args.token_pooling,
            "l2_normalize_features": bool(args.l2_normalize_features),
            "head_type": args.head_type,
            "mlp_hidden_dims": list(mlp_hidden_dims),
            "mlp_dropout": float(args.mlp_dropout),
            "input_dim": input_dim,
            "num_labels": len(label_names),
        },
        experiment_dir / "best.ckpt",
    )

    experiment_meta = {
        "run_date_utc": utc_now_iso(),
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "operation_label": DEFAULT_OPERATION_LABEL,
        "embedding_root": str(embedding_root),
        "manifest_csv": str(manifest_csv),
        "split_profile": args.split_profile,
        "embedding_layout": args.embedding_layout,
        "token_pooling": args.token_pooling,
        "l2_normalize_features": bool(args.l2_normalize_features),
        "head_type": args.head_type,
        "head_description": config["head_description"],
        "mlp_hidden_dims": list(mlp_hidden_dims),
        "mlp_dropout": float(args.mlp_dropout),
        "device_resolved": str(device),
        "best_epoch": int(best_epoch),
        "input_dim": int(input_dim),
        "num_labels": len(label_names),
        "model_num_parameters": model_num_parameters,
        "train_alias": evaluation_plan.train_alias,
        "selection_alias": evaluation_plan.selection_alias,
        "primary_test_alias": evaluation_plan.primary_test_alias,
        "split_inputs": config["split_inputs"],
        "macro_metrics": {alias: summary["macro"] for alias, summary in metrics_by_alias.items()},
        "thresholds_path": str(experiment_dir / evaluation_plan.thresholds_filename),
        "checkpoint_path": str(experiment_dir / "best.ckpt"),
    }
    write_json(experiment_dir / "experiment_meta.json", experiment_meta)

    recreation_report = render_recreation_report(
        experiment_dir=experiment_dir,
        config=config,
        split_data=split_data,
        metrics_by_alias=metrics_by_alias,
    )
    (experiment_dir / "recreation_report.md").write_text(recreation_report, encoding="utf-8")

    primary_test_fragment = ""
    if evaluation_plan.primary_test_alias and evaluation_plan.primary_test_alias in metrics_by_alias:
        primary_test_fragment = (
            f" {evaluation_plan.primary_test_alias}_macro_auroc="
            f"{format_metric(metrics_by_alias[evaluation_plan.primary_test_alias]['macro']['auroc'])}"
        )
    print(
        "[done] "
        f"best_epoch={best_epoch} "
        f"head={config['head_description']} "
        f"{evaluation_plan.selection_alias}_macro_auroc="
        f"{format_metric(metrics_by_alias[evaluation_plan.selection_alias]['macro']['auroc'])}"
        f"{primary_test_fragment} "
        f"output_dir={experiment_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
