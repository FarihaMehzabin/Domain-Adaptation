#!/usr/bin/env python3
"""Shared helpers for hospital adapter experiments."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import densenet121

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_namespace import resolve_input_path  # noqa: E402
from scripts.masked_multilabel_utils import (  # noqa: E402
    POLICY_B_LABEL_POLICY,
    compute_masked_multilabel_metrics,
    extract_targets_and_masks,
)


DEFAULT_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
DEFAULT_RUNS_ROOT = ROOT / "experiments" / "hospital_adapters_v1" / "runs"
IMAGE_PATH_COLUMNS = ["abs_path", "image_path", "file_path", "filepath", "path", "rel_path"]
SUBJECT_ID_COLUMNS = ["subject_id", "patient_id", "Patient ID"]
STUDY_ID_COLUMNS = ["study_id", "Study ID"]
DICOM_ID_COLUMNS = ["dicom_id", "image_id", "Image Index"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class StageFailure(RuntimeError):
    """Raised when the experiment cannot safely continue."""


class CXRManifestDataset(Dataset):
    """Simple dataset backed by an already validated manifest dataframe."""

    def __init__(self, dataframe: pd.DataFrame, label_names: Sequence[str], image_size: int) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.label_names = list(label_names)
        self.mask_columns = [f"{label}_mask" for label in self.label_names]
        self.transform = build_transform(image_size)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        image = Image.open(row["resolved_path"]).convert("RGB")
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(row[self.label_names].values.astype(np.float32), dtype=torch.float32)
        mask_tensor = torch.tensor(row[self.mask_columns].values.astype(np.float32), dtype=torch.float32)
        return image_tensor, label_tensor, mask_tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if value is pd.NA:
        return None
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(data), handle, indent=2)


def choose_existing_column(columns: Sequence[str], candidates: Sequence[str], description: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return str(candidate)
    raise StageFailure(f"Missing {description}. Expected one of: {list(candidates)}")


def optional_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return str(candidate)
    return None


def resolve_project_or_absolute_path(raw_value: str | None) -> Path | None:
    if raw_value is None:
        return None
    resolved = resolve_input_path(raw_value)
    if resolved is None:
        return None
    return resolved.resolve()


def resolve_run_dir(out_dir: str | None, run_name: str) -> Path:
    if out_dir is None:
        base_dir = DEFAULT_RUNS_ROOT
    else:
        base_dir = Path(out_dir)
        if not base_dir.is_absolute():
            base_dir = (ROOT / base_dir).resolve()
        else:
            base_dir = base_dir.resolve()
    if base_dir.name == run_name:
        return base_dir
    return (base_dir / run_name).resolve()


def infer_split_name(manifest_path: Path, fallback: str) -> str:
    name = manifest_path.name.lower()
    if "support" in name:
        return "support"
    if "train" in name:
        return "train"
    if "val" in name:
        return "val"
    if "test" in name:
        return "test"
    return fallback


def read_csv_checked(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise StageFailure(f"Missing manifest: {manifest_path}")
    try:
        return pd.read_csv(manifest_path, low_memory=False)
    except Exception as exc:  # pragma: no cover - depends on local file corruption
        raise StageFailure(f"Could not read CSV {manifest_path}: {exc}") from exc


def resolve_image_path(raw_path: str | Path, manifest_path: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path.resolve()

    candidates = [
        manifest_path.parent / path,
        manifest_path.parent.parent / path,
        ROOT / path,
        ROOT / "data" / "mimic_cxr" / path,
        ROOT / "data" / "nih_chest_xray14" / path,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return (ROOT / path).resolve()


def infer_label_names(
    checkpoint_metadata: dict[str, Any] | None,
    adapter_metadata: dict[str, Any] | None = None,
    manifest_paths: Sequence[Path] | None = None,
) -> list[str]:
    for source in [checkpoint_metadata, adapter_metadata]:
        if source is None:
            continue
        label_names = source.get("label_names")
        if label_names is not None:
            return [str(label) for label in label_names]

    manifest_paths = list(manifest_paths or [])
    for manifest_path in manifest_paths:
        dataframe = read_csv_checked(manifest_path)
        if all(label in dataframe.columns for label in DEFAULT_LABELS):
            return list(DEFAULT_LABELS)

    raise StageFailure("Could not infer label names from checkpoint metadata or manifests.")


def validate_manifest(
    manifest_path: Path,
    split_name: str,
    label_names: Sequence[str],
    label_policy: str = POLICY_B_LABEL_POLICY,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataframe = read_csv_checked(manifest_path).copy()
    columns = list(dataframe.columns)

    path_column = choose_existing_column(columns, IMAGE_PATH_COLUMNS, "an image path column")
    subject_column = choose_existing_column(columns, SUBJECT_ID_COLUMNS, "a subject id column")
    study_column = choose_existing_column(columns, STUDY_ID_COLUMNS, "a study id column")
    dicom_column = optional_existing_column(columns, DICOM_ID_COLUMNS)

    try:
        targets_np, masks_np = extract_targets_and_masks(dataframe, label_names, label_policy)
    except ValueError as exc:
        raise StageFailure(f"{manifest_path}: {exc}") from exc

    mask_columns = [f"{label}_mask" for label in label_names]
    for label_index, label in enumerate(label_names):
        dataframe[label] = targets_np[:, label_index].astype(np.int64, copy=False)
        dataframe[mask_columns[label_index]] = masks_np[:, label_index].astype(np.int64, copy=False)

    dataframe["resolved_path"] = dataframe[path_column].map(lambda value: resolve_image_path(value, manifest_path))
    missing_paths = [str(path) for path in dataframe["resolved_path"] if not Path(path).exists()]
    if missing_paths:
        preview = missing_paths[:5]
        raise StageFailure(
            f"{manifest_path} has {len(missing_paths)} missing image paths. Examples: {preview}"
        )

    dataframe["_subject_id"] = dataframe[subject_column].astype("string").fillna("")
    dataframe["_study_id"] = dataframe[study_column].astype("string").fillna("")
    dataframe["_path_identity"] = dataframe["resolved_path"].astype("string").fillna("")
    if dicom_column is not None:
        dataframe["_dicom_id"] = dataframe[dicom_column].astype("string").fillna("")

    label_counts: dict[str, dict[str, int]] = {}
    total_rows = int(len(dataframe))
    for label_index, label in enumerate(label_names):
        n_valid = int(masks_np[:, label_index].sum())
        positives = int((targets_np[:, label_index] * masks_np[:, label_index]).sum())
        negatives = int(n_valid - positives)
        masked = int(total_rows - n_valid)
        label_counts[label] = {
            "positives": positives,
            "negatives": negatives,
            "masked": masked,
            "n_valid": n_valid,
        }

    summary = {
        "manifest_path": str(manifest_path.resolve()),
        "split_name": split_name,
        "label_policy": label_policy,
        "path_column": path_column,
        "subject_column": subject_column,
        "study_column": study_column,
        "dicom_column": dicom_column,
        "num_images": int(len(dataframe)),
        "num_subjects": int(dataframe["_subject_id"].nunique()),
        "num_studies": int(dataframe["_study_id"].nunique()),
        "num_dicoms": int(dataframe["_dicom_id"].nunique()) if dicom_column is not None else None,
        "label_counts": label_counts,
    }
    return dataframe, summary


def normalize_overlap_values(series: pd.Series) -> set[str]:
    normalized = series.dropna().astype("string").str.strip()
    normalized = normalized[normalized != ""]
    return set(normalized.tolist())


def summarize_overlap(left_values: set[str], right_values: set[str]) -> dict[str, Any]:
    overlap = sorted(left_values & right_values)
    return {"count": int(len(overlap)), "examples": overlap[:5]}


def check_leakage(split_frames: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {
        "subject_overlap": {},
        "study_overlap": {},
        "dicom_overlap": {},
        "path_overlap": {},
    }
    failures: list[str] = []
    split_names = list(split_frames.keys())

    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            pair_name = f"{left_name}_vs_{right_name}"
            left_frame = split_frames[left_name]
            right_frame = split_frames[right_name]

            subject_overlap = summarize_overlap(
                normalize_overlap_values(left_frame["_subject_id"]),
                normalize_overlap_values(right_frame["_subject_id"]),
            )
            checks["subject_overlap"][pair_name] = subject_overlap
            if subject_overlap["count"] > 0:
                failures.append(
                    f"subject_id leakage between {left_name} and {right_name}: {subject_overlap['examples']}"
                )

            study_overlap = summarize_overlap(
                normalize_overlap_values(left_frame["_study_id"]),
                normalize_overlap_values(right_frame["_study_id"]),
            )
            checks["study_overlap"][pair_name] = study_overlap
            if study_overlap["count"] > 0:
                failures.append(
                    f"study_id leakage between {left_name} and {right_name}: {study_overlap['examples']}"
                )

            path_overlap = summarize_overlap(
                normalize_overlap_values(left_frame["_path_identity"]),
                normalize_overlap_values(right_frame["_path_identity"]),
            )
            checks["path_overlap"][pair_name] = path_overlap
            if path_overlap["count"] > 0:
                failures.append(
                    f"image path leakage between {left_name} and {right_name}: {path_overlap['examples']}"
                )

            if "_dicom_id" in left_frame.columns and "_dicom_id" in right_frame.columns:
                dicom_overlap = summarize_overlap(
                    normalize_overlap_values(left_frame["_dicom_id"]),
                    normalize_overlap_values(right_frame["_dicom_id"]),
                )
            else:
                dicom_overlap = {"count": None, "examples": [], "note": "dicom id not available in both splits"}
            checks["dicom_overlap"][pair_name] = dicom_overlap
            if dicom_overlap["count"] not in (None, 0):
                failures.append(
                    f"dicom_id leakage between {left_name} and {right_name}: {dicom_overlap['examples']}"
                )

    return checks, failures


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_dataloader(
    dataframe: pd.DataFrame,
    label_names: Sequence[str],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = CXRManifestDataset(dataframe=dataframe, label_names=label_names, image_size=image_size)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        raise StageFailure("Checkpoint state dict is empty.")
    if not all(str(key).startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {str(key)[7:]: value for key, value in state_dict.items()}


def build_densenet121_model(num_labels: int) -> torch.nn.Module:
    model = densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_labels)
    return model


def load_base_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], list[str]]:
    if not checkpoint_path.exists():
        raise StageFailure(f"Missing checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        raise StageFailure(f"Could not load checkpoint {checkpoint_path}: {exc}") from exc

    metadata: dict[str, Any] = {}
    state_dict: dict[str, Any] | None = None
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            state_dict = checkpoint

        for key in [
            "label_names",
            "model_info",
            "num_classes",
            "architecture",
            "image_size",
            "epoch",
            "val_loss",
            "source_checkpoint",
            "adaptation_method",
            "best_metric_name",
            "best_metric_value",
            "pos_weight",
        ]:
            if key in checkpoint:
                metadata[key] = checkpoint[key]

    if state_dict is None:
        raise StageFailure(f"Could not find a model state dict inside checkpoint: {checkpoint_path}")

    state_dict = normalize_state_dict_keys(state_dict)
    classifier_weight = state_dict.get("classifier.weight")
    classifier_bias = state_dict.get("classifier.bias")
    if classifier_weight is None or classifier_bias is None:
        raise StageFailure(
            f"Checkpoint {checkpoint_path} is missing classifier weights required for DenseNet-121 loading."
        )

    num_labels = int(classifier_weight.shape[0])
    label_names = metadata.get("label_names")
    if label_names is None:
        label_names = list(DEFAULT_LABELS) if num_labels == len(DEFAULT_LABELS) else [f"label_{i}" for i in range(num_labels)]
        metadata["label_names"] = list(label_names)
    else:
        label_names = [str(label) for label in label_names]
        if len(label_names) != num_labels:
            raise StageFailure(
                "Checkpoint label count does not match classifier output size: "
                f"{len(label_names)} labels vs {num_labels} classifier outputs."
            )

    model = build_densenet121_model(num_labels=num_labels)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise StageFailure(f"Checkpoint weights could not be loaded into DenseNet-121: {exc}") from exc

    return model.to(device), metadata, label_names


def compute_pos_weight_from_dataframe(
    dataframe: pd.DataFrame,
    label_names: Sequence[str],
    label_policy: str,
    clamp_max: float = 10.0,
) -> torch.Tensor:
    targets_np, masks_np = extract_targets_and_masks(dataframe, label_names, label_policy)
    positives = (targets_np * masks_np).sum(axis=0)
    valid = masks_np.sum(axis=0)
    negatives = valid - positives

    values: list[float] = []
    for positive_count, negative_count in zip(positives.tolist(), negatives.tolist()):
        if positive_count <= 0:
            value = clamp_max
        else:
            value = float(negative_count / positive_count)
            if not math.isfinite(value):
                value = clamp_max
            value = min(max(value, 1.0), clamp_max)
        values.append(float(value))
    return torch.tensor(values, dtype=torch.float32)


def masked_bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape != targets.shape or logits.shape != masks.shape:
        raise ValueError(
            "logits, targets, and masks must share the same shape. "
            f"Got logits={tuple(logits.shape)}, targets={tuple(targets.shape)}, masks={tuple(masks.shape)}"
        )

    unreduced_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    masked_loss = unreduced_loss * masks
    valid_count = masks.sum()
    if torch.count_nonzero(valid_count).item() == 0:
        return logits.sum() * 0.0
    return masked_loss.sum() / valid_count


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_names: Sequence[str],
) -> dict[str, Any]:
    total_loss = 0.0
    total_valid = 0.0
    all_logits: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for images, labels, masks in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = masked_bce_with_logits_loss(logits, labels, masks)
            if not torch.isfinite(loss):
                raise StageFailure("Evaluation loss became NaN or infinite.")

            probabilities = torch.sigmoid(logits)
            valid_count = float(masks.sum().item())
            total_loss += float(loss.item()) * valid_count
            total_valid += valid_count

            all_logits.append(logits.detach().cpu().numpy())
            all_probabilities.append(probabilities.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())
            all_masks.append(masks.detach().cpu().numpy())

    if not all_targets:
        raise StageFailure("Evaluation split is empty. Cannot compute metrics.")

    logits_np = np.concatenate(all_logits, axis=0)
    probabilities_np = np.concatenate(all_probabilities, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    masks_np = np.concatenate(all_masks, axis=0)

    metrics = compute_masked_multilabel_metrics(targets_np, probabilities_np, masks_np, label_names)
    metrics["mean_ap"] = metrics["macro_auprc"]
    metrics["bce_loss"] = float(total_loss / total_valid) if total_valid > 0 else 0.0
    metrics["loss"] = metrics["bce_loss"]
    metrics["invalid_auroc_labels"] = [
        label for label in label_names if not metrics["per_label"][label]["auroc_defined"]
    ]
    metrics["invalid_auprc_labels"] = [
        label for label in label_names if not metrics["per_label"][label]["auprc_defined"]
    ]
    metrics["logits"] = logits_np
    metrics["probabilities"] = probabilities_np
    metrics["targets"] = targets_np
    metrics["masks"] = masks_np
    return metrics


def report_ready_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": float(metrics["loss"]),
        "bce_loss": float(metrics["bce_loss"]),
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "mean_ap": metrics["mean_ap"],
        "micro_auroc": metrics.get("micro_auroc"),
        "micro_auprc": metrics.get("micro_auprc"),
        "defined_auroc_labels": int(metrics["defined_auroc_labels"]),
        "defined_auprc_labels": int(metrics["defined_auprc_labels"]),
        "valid_label_count": int(metrics.get("valid_label_count", 0)),
        "invalid_auroc_labels": list(metrics["invalid_auroc_labels"]),
        "invalid_auprc_labels": list(metrics["invalid_auprc_labels"]),
        "per_label": metrics["per_label"],
    }


def build_split_report(
    *,
    run_name: str,
    model_name: str,
    split_summary: dict[str, Any],
    label_names: Sequence[str],
    metrics: dict[str, Any],
    checkpoint_path: Path,
    adapter_checkpoint_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "run_name": run_name,
        "model_name": model_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "adapter_checkpoint": str(adapter_checkpoint_path.resolve()) if adapter_checkpoint_path is not None else None,
        "manifest_path": split_summary["manifest_path"],
        "split_name": split_summary["split_name"],
        "label_policy": split_summary["label_policy"],
        "label_names": list(label_names),
        "num_images": int(split_summary["num_images"]),
        "num_subjects": int(split_summary["num_subjects"]),
        "num_studies": int(split_summary["num_studies"]),
        "num_dicoms": split_summary["num_dicoms"],
        "metrics": report_ready_metrics(metrics),
    }
    if extra:
        report.update(extra)
    return report


def save_predictions_csv(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    output_path: Path,
    label_names: Sequence[str],
) -> None:
    ensure_dir(output_path.parent)
    rows: list[dict[str, Any]] = []
    mask_columns = [f"{label}_mask" for label in label_names]

    for row_index, row in dataframe.reset_index(drop=True).iterrows():
        record: dict[str, Any] = {
            "image_path": str(row["resolved_path"]),
            "subject_id": row["_subject_id"],
            "study_id": row["_study_id"],
        }
        if "_dicom_id" in dataframe.columns:
            record["dicom_id"] = row["_dicom_id"]
        for column_name in ["image_id", "dicom_id", "abs_path", "rel_path", "path", "filepath", "image_path"]:
            if column_name in dataframe.columns:
                record[f"manifest_{column_name}"] = row[column_name]
        for label in label_names:
            record[f"true_{label}"] = int(row[label])
        for mask_column in mask_columns:
            record[mask_column] = int(row[mask_column])
        for label_index, label in enumerate(label_names):
            record[f"pred_{label}"] = float(probabilities[row_index, label_index])
        rows.append(record)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_train_log(history: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(history).to_csv(path, index=False)


def format_metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"

