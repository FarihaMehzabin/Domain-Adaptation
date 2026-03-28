#!/usr/bin/env python3
"""Train a frozen-embedding multilabel classifier on NIH CXR14 report embeddings."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'matplotlib'. Install: pip install -r requirements/frozen_embeddings_classifier.txt"
    ) from exc

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'numpy'. Install: pip install -r requirements/frozen_embeddings_classifier.txt"
    ) from exc

try:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'scikit-learn'. Install: pip install -r requirements/frozen_embeddings_classifier.txt"
    ) from exc

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'torch'. Install: pip install -r requirements/frozen_embeddings_classifier.txt"
    ) from exc


LABEL_SPECS = [
    ("label_atelectasis", "atelectasis"),
    ("label_cardiomegaly", "cardiomegaly"),
    ("label_consolidation", "consolidation"),
    ("label_edema", "edema"),
    ("label_pleural_effusion", "pleural_effusion"),
]
LABEL_COLUMNS = [column for column, _ in LABEL_SPECS]
LABEL_NAMES = [name for _, name in LABEL_SPECS]

DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_nih_cxr14 .csv")

DEFAULT_TRAIN_EMBEDDINGS = Path(
    "/workspace/report_embeddings/BiomedVLP-CXR-BERT-specialized/embeddings.npy"
)
DEFAULT_VAL_EMBEDDINGS = Path(
    "/workspace/report_embeddings/val/microsoft__BiomedVLP-CXR-BERT-specialized/embeddings.npy"
)
DEFAULT_TEST_EMBEDDINGS = Path(
    "/workspace/report_embeddings/test/microsoft__BiomedVLP-CXR-BERT-specialized/embeddings.npy"
)

DEFAULT_TRAIN_REPORT_IDS = Path(
    "/workspace/report_embeddings/train/BiomedVLP-CXR-BERT-specialized/report_ids.json"
)
DEFAULT_VAL_REPORT_IDS = Path(
    "/workspace/report_embeddings/val/microsoft__BiomedVLP-CXR-BERT-specialized/report_ids.json"
)
DEFAULT_TEST_REPORT_IDS = Path(
    "/workspace/report_embeddings/test/microsoft__BiomedVLP-CXR-BERT-specialized/report_ids.json"
)

DEFAULT_OUTPUT_ROOT = Path("/workspace/outputs/models/nih_cxr14/report_only/linear")


@dataclass
class SplitData:
    name: str
    features: torch.Tensor
    labels: torch.Tensor
    manifest_path: Path
    embeddings_path: Path
    report_ids_path: Path

    @property
    def num_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.features.shape[1])


@dataclass
class BatchPolicy:
    tuned_safe_micro_batch: int
    micro_batch: int
    effective_batch: int
    accum_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear multilabel classifier on frozen NIH CXR14 report embeddings."
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--train-embeddings", type=Path, default=DEFAULT_TRAIN_EMBEDDINGS)
    parser.add_argument("--val-embeddings", type=Path, default=DEFAULT_VAL_EMBEDDINGS)
    parser.add_argument("--test-embeddings", type=Path, default=DEFAULT_TEST_EMBEDDINGS)
    parser.add_argument("--train-report-ids", type=Path, default=DEFAULT_TRAIN_REPORT_IDS)
    parser.add_argument("--val-report-ids", type=Path, default=DEFAULT_VAL_REPORT_IDS)
    parser.add_argument("--test-report-ids", type=Path, default=DEFAULT_TEST_REPORT_IDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2, sort_keys=True), encoding="utf-8")


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, torch.Tensor):
        return to_serializable(value.detach().cpu().numpy())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def read_manifest_rows(manifest_csv: Path) -> list[dict[str, str]]:
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest CSV {manifest_csv} is missing a header row.")
        missing_columns = [column for column in LABEL_COLUMNS + ["image_path", "split"] if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"Manifest CSV {manifest_csv} is missing columns: {missing_columns}")
        return list(reader)


def read_report_ids(report_ids_path: Path) -> list[str]:
    if not report_ids_path.exists():
        raise FileNotFoundError(f"Report IDs file not found: {report_ids_path}")
    payload = json.loads(report_ids_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Report IDs file is empty or invalid: {report_ids_path}")
    return [str(item) for item in payload]


def load_embedding_array(embeddings_path: Path) -> np.ndarray:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    array = np.load(embeddings_path, allow_pickle=True)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array at {embeddings_path}, found shape {array.shape}.")
    array = np.asarray(array, dtype=np.float32)
    return np.ascontiguousarray(array)


def build_manifest_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        basename = Path(row["image_path"]).stem
        lookup[basename] = row
    return lookup


def validate_alignment(
    split_name: str,
    rows: list[dict[str, str]],
    embeddings: np.ndarray,
    report_ids: list[str],
    manifest_lookup: dict[str, dict[str, str]],
) -> None:
    if len(rows) != embeddings.shape[0]:
        raise ValueError(
            f"{split_name}: split row count ({len(rows)}) does not match embedding rows ({embeddings.shape[0]})."
        )
    if len(rows) != len(report_ids):
        raise ValueError(
            f"{split_name}: split row count ({len(rows)}) does not match report_ids entries ({len(report_ids)})."
        )

    missing_ids = [report_id for report_id in report_ids if report_id not in manifest_lookup]
    if missing_ids:
        preview = ", ".join(missing_ids[:3])
        raise ValueError(f"{split_name}: report_ids missing in manifest split: {preview}")


def build_labels(report_ids: list[str], manifest_lookup: dict[str, dict[str, str]]) -> torch.Tensor:
    labels = np.zeros((len(report_ids), len(LABEL_COLUMNS)), dtype=np.float32)
    for row_index, report_id in enumerate(report_ids):
        row = manifest_lookup[report_id]
        for label_index, column in enumerate(LABEL_COLUMNS):
            labels[row_index, label_index] = float(row[column])
    return torch.from_numpy(labels)


def load_split(
    split_name: str,
    manifest_rows: list[dict[str, str]],
    embeddings_path: Path,
    report_ids_path: Path,
) -> SplitData:
    embeddings = load_embedding_array(embeddings_path)
    report_ids = read_report_ids(report_ids_path)

    manifest_lookup = build_manifest_lookup(manifest_rows)
    validate_alignment(split_name, manifest_rows, embeddings, report_ids, manifest_lookup)

    features = torch.from_numpy(embeddings)
    labels = build_labels(report_ids, manifest_lookup)
    return SplitData(
        name=split_name,
        features=features,
        labels=labels,
        manifest_path=DEFAULT_MANIFEST_CSV,
        embeddings_path=embeddings_path,
        report_ids_path=report_ids_path,
    )


def compute_label_counts(labels: torch.Tensor) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    totals = labels.shape[0]
    positives = labels.sum(dim=0).to(torch.int64).tolist()
    for index, name in enumerate(LABEL_NAMES):
        positive_count = int(positives[index])
        counts[name] = {
            "positives": positive_count,
            "negatives": int(totals - positive_count),
        }
    return counts


def compute_pos_weight(labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    counts = compute_label_counts(labels)
    pos_weight_values: list[float] = []
    pos_weight_by_label: dict[str, float] = {}
    for name in LABEL_NAMES:
        positives = counts[name]["positives"]
        negatives = counts[name]["negatives"]
        if positives <= 0:
            raise ValueError(f"Cannot compute pos_weight for {name}: no positive examples in train split.")
        value = negatives / positives
        pos_weight_values.append(value)
        pos_weight_by_label[name] = value
    return torch.tensor(pos_weight_values, dtype=torch.float32), pos_weight_by_label


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message


def cleanup_after_trial(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def try_micro_batch(
    batch_size: int,
    train_split: SplitData,
    pos_weight: torch.Tensor,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
) -> bool:
    model: nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    criterion: nn.Module | None = None
    features: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None

    try:
        model = nn.Linear(train_split.embedding_dim, len(LABEL_NAMES))
        model.to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

        features = train_split.features[:batch_size].to(device=device, dtype=torch.float32)
        labels = train_split.labels[:batch_size].to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return True
    except Exception as exc:
        if is_oom_error(exc):
            return False
        raise
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        del loss
        del logits
        del labels
        del features
        del criterion
        del optimizer
        if model is not None:
            model.to("cpu")
        del model
        cleanup_after_trial(device)


def tune_max_safe_micro_batch(
    train_split: SplitData,
    pos_weight: torch.Tensor,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
) -> int:
    max_samples = train_split.num_samples
    initial = min(64, max_samples)
    current = initial
    last_good = 0
    first_bad: int | None = None

    while True:
        succeeded = try_micro_batch(
            batch_size=current,
            train_split=train_split,
            pos_weight=pos_weight,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        if succeeded:
            last_good = current
            if current >= max_samples:
                return last_good
            next_batch = min(current * 2, max_samples)
            if next_batch == current:
                return last_good
            current = next_batch
            continue

        first_bad = current
        break

    low = last_good + 1
    high = first_bad - 1
    if last_good == 0:
        low = 1

    while low <= high:
        mid = (low + high) // 2
        succeeded = try_micro_batch(
            batch_size=mid,
            train_split=train_split,
            pos_weight=pos_weight,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        if succeeded:
            last_good = mid
            low = mid + 1
        else:
            high = mid - 1

    if last_good <= 0:
        raise RuntimeError("Unable to find a safe micro-batch size, even for batch_size=1.")
    return last_good


def derive_batch_policy(tuned_safe_micro_batch: int) -> BatchPolicy:
    micro_batch = min(tuned_safe_micro_batch, 1024)
    if micro_batch >= 1024:
        effective_batch = 1024
        accum_steps = 1
    else:
        accum_steps = max(1, 1024 // micro_batch)
        effective_batch = micro_batch * accum_steps
    if not 512 <= effective_batch <= 1024:
        raise ValueError(
            f"Effective batch must be within [512, 1024], found {effective_batch} "
            f"(micro_batch={micro_batch}, accum_steps={accum_steps})."
        )
    return BatchPolicy(
        tuned_safe_micro_batch=tuned_safe_micro_batch,
        micro_batch=micro_batch,
        effective_batch=effective_batch,
        accum_steps=accum_steps,
    )


def build_dataloader(
    split: SplitData,
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(split.features, split.labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator,
    )


def compute_binary_metric(metric_name: str, targets: np.ndarray, probabilities: np.ndarray) -> float | None:
    if np.unique(targets).size < 2:
        return None
    try:
        if metric_name == "auroc":
            return float(roc_auc_score(targets, probabilities))
        if metric_name == "average_precision":
            return float(average_precision_score(targets, probabilities))
    except ValueError:
        return None
    raise ValueError(f"Unsupported metric: {metric_name}")


def summarize_predictions(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    per_label: dict[str, dict[str, float | None]] = {}
    macro_auroc_values: list[float] = []
    for label_index, label_name in enumerate(LABEL_NAMES):
        target_column = targets[:, label_index]
        probability_column = probabilities[:, label_index]
        auroc = compute_binary_metric("auroc", target_column, probability_column)
        average_precision = compute_binary_metric("average_precision", target_column, probability_column)
        if auroc is not None:
            macro_auroc_values.append(auroc)
        per_label[label_name] = {
            "auroc": auroc,
            "average_precision": average_precision,
        }

    macro_auroc = float(np.mean(macro_auroc_values)) if macro_auroc_values else None
    return {
        "macro_auroc": macro_auroc,
        "per_label": per_label,
    }


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    targets_list: list[np.ndarray] = []
    probabilities_list: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device=device, dtype=torch.float32, non_blocking=True)
            logits = model(features)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            targets = labels.cpu().numpy()
            probabilities_list.append(probabilities)
            targets_list.append(targets)

    all_targets = np.concatenate(targets_list, axis=0)
    all_probabilities = np.concatenate(probabilities_list, axis=0)
    return summarize_predictions(all_targets, all_probabilities), all_targets, all_probabilities


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    accum_steps: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_samples = 0
    micro_steps = 0

    for step_index, (features, labels) in enumerate(dataloader, start=1):
        features = features.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.float32, non_blocking=True)

        logits = model(features)
        raw_loss = criterion(logits, labels)
        (raw_loss / accum_steps).backward()

        batch_size = int(features.shape[0])
        total_loss += float(raw_loss.item()) * batch_size
        total_samples += batch_size
        micro_steps += 1

        if micro_steps == accum_steps or step_index == len(dataloader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_steps = 0

    return total_loss / max(total_samples, 1)


def tune_f1_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    thresholds_by_label: dict[str, float] = {}
    for label_index, label_name in enumerate(LABEL_NAMES):
        target_column = targets[:, label_index]
        probability_column = probabilities[:, label_index]

        if np.unique(target_column).size < 2:
            thresholds_by_label[label_name] = 0.5
            continue

        precision, recall, thresholds = precision_recall_curve(target_column, probability_column)
        if thresholds.size == 0:
            thresholds_by_label[label_name] = 0.5
            continue

        f1_values = (2.0 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        best_f1 = float(np.max(f1_values))
        best_indices = np.flatnonzero(np.isclose(f1_values, best_f1, atol=1e-12, rtol=0.0))
        thresholds_by_label[label_name] = float(np.min(thresholds[best_indices]))
    return thresholds_by_label


def save_calibration_artifacts(
    split_name: str,
    targets: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
) -> None:
    calibration_dir = output_dir / "calibration" / split_name
    calibration_dir.mkdir(parents=True, exist_ok=True)

    for label_index, label_name in enumerate(LABEL_NAMES):
        target_column = targets[:, label_index]
        probability_column = probabilities[:, label_index]
        prob_true, prob_pred = calibration_curve(target_column, probability_column, n_bins=10, strategy="quantile")

        payload = {
            "split": split_name,
            "label": label_name,
            "n_bins": 10,
            "strategy": "quantile",
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
        write_json(calibration_dir / f"{label_name}.json", payload)

        figure, axis = plt.subplots(figsize=(4.5, 4.5))
        axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="gray", label="Perfect")
        axis.plot(prob_pred, prob_true, marker="o", linewidth=1.5, color="#005f73", label=label_name)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Fraction of positives")
        axis.set_title(f"{split_name.title()} calibration: {label_name.replace('_', ' ')}")
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(calibration_dir / f"{label_name}.png", dpi=200)
        plt.close(figure)


def build_metrics_payload(
    split_name: str,
    metrics: dict[str, Any],
    thresholds_by_label: dict[str, float] | None,
) -> dict[str, Any]:
    payload = {
        "split": split_name,
        "macro_auroc": metrics["macro_auroc"],
        "per_label": {},
    }
    for label_name in LABEL_NAMES:
        per_label_payload = dict(metrics["per_label"][label_name])
        if thresholds_by_label is not None:
            per_label_payload["f1_threshold_from_val"] = thresholds_by_label[label_name]
        payload["per_label"][label_name] = per_label_payload
    return payload


def create_output_dir(output_root: Path, run_name: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    directory_name = run_name.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / directory_name
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_checkpoint(path: Path, model: nn.Module, epoch: int, val_macro_auroc: float | None) -> None:
    torch.save(
        {
            "epoch": epoch,
            "val_macro_auroc": val_macro_auroc,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    seed_everything(args.seed)

    manifest_rows = read_manifest_rows(args.manifest_csv)
    train_rows = [row for row in manifest_rows if row["split"] == "train"]
    val_rows = [row for row in manifest_rows if row["split"] == "val"]
    test_rows = [row for row in manifest_rows if row["split"] == "test"]

    train_split = load_split("train", train_rows, args.train_embeddings, args.train_report_ids)
    val_split = load_split("val", val_rows, args.val_embeddings, args.val_report_ids)
    test_split = load_split("test", test_rows, args.test_embeddings, args.test_report_ids)

    if train_split.embedding_dim != val_split.embedding_dim or train_split.embedding_dim != test_split.embedding_dim:
        raise ValueError(
            "All splits must share the same embedding dimension: "
            f"train={train_split.embedding_dim}, val={val_split.embedding_dim}, test={test_split.embedding_dim}"
        )

    pos_weight, pos_weight_by_label = compute_pos_weight(train_split.labels)

    tuned_safe_micro_batch = tune_max_safe_micro_batch(
        train_split=train_split,
        pos_weight=pos_weight,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    batch_policy = derive_batch_policy(tuned_safe_micro_batch)

    seed_everything(args.seed)

    output_dir = create_output_dir(args.output_root, args.run_name)
    checkpoint_path = output_dir / "best.ckpt"
    history_path = output_dir / "history.jsonl"

    config = {
        "run_date_utc": utc_now_iso(),
        "device_requested": args.device,
        "device_resolved": str(device),
        "seed": args.seed,
        "dtype": "float32",
        "label_names": LABEL_NAMES,
        "model": {
            "type": "linear",
            "input_dim": train_split.embedding_dim,
            "output_dim": len(LABEL_NAMES),
            "hidden_layers": [],
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "training": {
            "max_epochs": args.max_epochs,
            "early_stopping_patience": args.patience,
            "early_stopping_metric": "val_macro_auroc",
            "shuffle_train": True,
            "shuffle_val": False,
            "shuffle_test": False,
        },
        "paths": {
            "manifest_csv": args.manifest_csv,
            "train_embeddings": args.train_embeddings,
            "val_embeddings": args.val_embeddings,
            "test_embeddings": args.test_embeddings,
            "train_report_ids": args.train_report_ids,
            "val_report_ids": args.val_report_ids,
            "test_report_ids": args.test_report_ids,
            "output_dir": output_dir,
        },
        "split_sizes": {
            "train": train_split.num_samples,
            "val": val_split.num_samples,
            "test": test_split.num_samples,
        },
        "train_label_counts": compute_label_counts(train_split.labels),
        "pos_weight": pos_weight_by_label,
        "batch_policy": batch_policy.__dict__,
    }
    write_json(output_dir / "config.json", config)

    model = nn.Linear(train_split.embedding_dim, len(LABEL_NAMES))
    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = build_dataloader(
        train_split,
        batch_size=batch_policy.micro_batch,
        shuffle=True,
        generator=train_generator,
        pin_memory=pin_memory,
    )
    val_loader = build_dataloader(
        val_split,
        batch_size=batch_policy.micro_batch,
        shuffle=False,
        generator=None,
        pin_memory=pin_memory,
    )
    test_loader = build_dataloader(
        test_split,
        batch_size=batch_policy.micro_batch,
        shuffle=False,
        generator=None,
        pin_memory=pin_memory,
    )

    best_val_macro_auroc = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0

    with history_path.open("w", encoding="utf-8") as history_handle:
        for epoch in range(1, args.max_epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                accum_steps=batch_policy.accum_steps,
            )

            val_metrics, _, _ = evaluate_model(model, val_loader, device)
            val_macro_auroc = val_metrics["macro_auroc"]
            metric_for_selection = float("-inf") if val_macro_auroc is None else float(val_macro_auroc)
            improved = metric_for_selection > (best_val_macro_auroc + 1e-6)

            if improved:
                best_val_macro_auroc = metric_for_selection
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(checkpoint_path, model, epoch, val_macro_auroc)
            else:
                epochs_without_improvement += 1

            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_auroc": val_macro_auroc,
                "val_per_label": val_metrics["per_label"],
                "is_best": improved,
                "epochs_without_improvement": epochs_without_improvement,
            }
            history_handle.write(json.dumps(to_serializable(history_row), sort_keys=True) + "\n")
            history_handle.flush()

            print(
                f"epoch={epoch:02d} "
                f"train_loss={train_loss:.6f} "
                f"val_macro_auroc={val_macro_auroc if val_macro_auroc is not None else 'NA'} "
                f"best_epoch={best_epoch}"
            )

            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered after epoch {epoch}.")
                break

    if not checkpoint_path.exists():
        raise RuntimeError("Training completed without writing a best checkpoint.")

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device, dtype=torch.float32)

    val_metrics, val_targets, val_probabilities = evaluate_model(model, val_loader, device)
    val_thresholds = tune_f1_thresholds(val_targets, val_probabilities)
    save_calibration_artifacts("val", val_targets, val_probabilities, output_dir)

    test_metrics, test_targets, test_probabilities = evaluate_model(model, test_loader, device)
    save_calibration_artifacts("test", test_targets, test_probabilities, output_dir)

    val_payload = build_metrics_payload("val", val_metrics, val_thresholds)
    test_payload = build_metrics_payload("test", test_metrics, val_thresholds)
    test_payload["selected_checkpoint"] = {
        "epoch": int(checkpoint["epoch"]),
        "val_macro_auroc": checkpoint["val_macro_auroc"],
    }
    test_payload["evaluation_policy"] = {
        "evaluated_exactly_once_after_model_selection": True,
    }

    write_json(output_dir / "val_metrics.json", val_payload)
    write_json(output_dir / "test_metrics.json", test_payload)
    write_json(output_dir / "val_f1_thresholds.json", val_thresholds)

    print(f"best_epoch={best_epoch} best_val_macro_auroc={best_val_macro_auroc}")
    print(f"wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
