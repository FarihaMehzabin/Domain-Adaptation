#!/usr/bin/env python3
"""Head-only few-shot adaptation from NIH to MIMIC."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import densenet121

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_namespace import (  # noqa: E402
    POLICYB_TARGET_MANIFEST_BASENAMES,
    build_named_run_paths,
    build_namespace_config,
    collect_missing_paths,
    default_policyb_source_report,
    enforce_policy_b_manifest_guard,
    infer_policyb_support_manifest,
    print_resolved_configuration,
    resolve_input_path,
    resolve_manifest_path,
    resolve_report_input,
)


LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
IMAGE_PATH_COLUMNS = ["abs_path", "image_path", "file_path", "filepath", "path", "rel_path"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class StageFailure(RuntimeError):
    """Raised when the stage cannot safely continue."""


class MIMICDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_size: int) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = build_transform(image_size)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        image = Image.open(row["resolved_path"]).convert("RGB")
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(row[LABELS].values.astype(np.float32), dtype=torch.float32)
        return image_tensor, label_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Head-only DenseNet-121 adaptation from NIH to MIMIC.")
    parser.add_argument("--checkpoint", "--source_checkpoint", dest="checkpoint", type=str, default=None)
    parser.add_argument("--support_csv", "--support_manifest", dest="support_csv", type=str, default=None)
    parser.add_argument("--val_csv", "--manifest_val", dest="val_csv", type=str, default=None)
    parser.add_argument("--test_csv", "--manifest_test", dest="test_csv", type=str, default=None)
    parser.add_argument("--source_only_report", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--label_policy", type=str, default="uignore_blankzero")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    namespace = build_namespace_config(args.base_dir, args.out_dir)
    if args.run_name is None:
        args.run_name = f"head_only_seed{args.seed}"

    args.base_dir = str(namespace.base_dir) if namespace.base_dir is not None else None
    args.out_dir = str(namespace.output_root)
    args.checkpoint = str(
        resolve_input_path(args.checkpoint, default_relative="checkpoints/nih_2k_densenet121_best.pt")
    )

    support_default = None
    if args.label_policy == "uignore_blankzero":
        support_default = infer_policyb_support_manifest(args.run_name, args.seed)

    support_manifest = resolve_manifest_path(
        args.support_csv,
        namespace=namespace,
        default_filename=support_default,
    )
    val_manifest = resolve_manifest_path(
        args.val_csv,
        namespace=namespace,
        default_filename=(
            POLICYB_TARGET_MANIFEST_BASENAMES["val"]
            if args.label_policy == "uignore_blankzero"
            else None
        ),
    )
    test_manifest = resolve_manifest_path(
        args.test_csv,
        namespace=namespace,
        default_filename=(
            POLICYB_TARGET_MANIFEST_BASENAMES["test"]
            if args.label_policy == "uignore_blankzero"
            else None
        ),
    )
    source_report_default = (
        default_policyb_source_report(args.seed)
        if namespace.base_dir is not None and args.label_policy == "uignore_blankzero"
        else "stage5_source_baseline.json"
    )
    source_only_report = resolve_report_input(
        args.source_only_report,
        namespace=namespace,
        default_filename=source_report_default,
    )

    args.support_csv = str(support_manifest) if support_manifest is not None else None
    args.val_csv = str(val_manifest) if val_manifest is not None else None
    args.test_csv = str(test_manifest) if test_manifest is not None else None
    args.source_only_report = str(source_only_report) if source_only_report is not None else None
    args._namespace_config = namespace
    args._artifact_paths = build_named_run_paths(
        namespace,
        args.run_name,
        include_checkpoints=True,
        include_loss_curve=True,
    )
    return args


def required_input_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    return {
        "checkpoint": Path(args.checkpoint) if args.checkpoint else None,
        "support_manifest": Path(args.support_csv) if args.support_csv else None,
        "val_manifest": Path(args.val_csv) if args.val_csv else None,
        "test_manifest": Path(args.test_csv) if args.test_csv else None,
        "source_only_report": Path(args.source_only_report) if args.source_only_report else None,
    }


def print_runtime_configuration(args: argparse.Namespace) -> None:
    artifact_paths = args._artifact_paths
    print_resolved_configuration(
        script_name=Path(__file__).name,
        base_dir=Path(args.base_dir) if args.base_dir else None,
        run_name=args.run_name,
        label_policy=args.label_policy,
        val_manifest=Path(args.val_csv) if args.val_csv else None,
        test_manifest=Path(args.test_csv) if args.test_csv else None,
        support_manifest=Path(args.support_csv) if args.support_csv else None,
        source_checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        source_only_report=Path(args.source_only_report) if args.source_only_report else None,
        checkpoint_output_path=artifact_paths["best_checkpoint"],
        prediction_val_output_path=artifact_paths["val_predictions"],
        prediction_test_output_path=artifact_paths["test_predictions"],
        report_output_path=artifact_paths["report_json"],
        report_markdown_path=artifact_paths["report_md"],
    )


def run_dry_run(args: argparse.Namespace) -> None:
    try:
        enforce_policy_b_manifest_guard(
            args.label_policy,
            val_manifest=Path(args.val_csv) if args.val_csv else None,
            test_manifest=Path(args.test_csv) if args.test_csv else None,
        )
    except ValueError as exc:
        raise StageFailure(str(exc)) from exc
    print_runtime_configuration(args)
    missing_files = collect_missing_paths(required_input_paths(args))
    if missing_files:
        raise StageFailure("Dry run failed:\n- " + "\n- ".join(missing_files))
    print("dry_run: configuration resolved and all required files exist")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is pd.NA:
        return None
    return value


def choose_existing_column(columns: list[str], candidates: list[str], description: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise StageFailure(f"Missing {description}. Expected one of: {candidates}")


def resolve_image_path(raw_path: str | Path, manifest_path: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path.resolve()

    candidates = [
        manifest_path.parent / path,
        manifest_path.parent.parent / path,
        Path.cwd() / path,
        Path.cwd() / "data" / "mimic_cxr" / path,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return (Path.cwd() / path).resolve()


def check_required_files(required_files: dict[str, Path]) -> list[str]:
    failures: list[str] = []
    for label, path in required_files.items():
        if not path.exists():
            failures.append(f"Missing required file for {label}: {path.resolve()}")
        elif not path.is_file():
            failures.append(f"Expected a file for {label}, but found something else: {path.resolve()}")
    return failures


def read_csv_checked(manifest_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(manifest_path, low_memory=False)
    except Exception as exc:  # pragma: no cover - depends on local file corruption
        raise StageFailure(f"Could not read CSV {manifest_path}: {exc}") from exc


def validate_manifest(manifest_path: Path, split_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not manifest_path.exists():
        raise StageFailure(f"Missing manifest: {manifest_path}")

    dataframe = read_csv_checked(manifest_path).copy()
    columns = list(dataframe.columns)

    path_column = choose_existing_column(columns, IMAGE_PATH_COLUMNS, "an image path column")
    if "subject_id" not in columns:
        raise StageFailure(f"{manifest_path} is missing required column: subject_id")
    if "study_id" not in columns:
        raise StageFailure(f"{manifest_path} is missing required column: study_id")

    dicom_id_available = "dicom_id" in columns

    for label in LABELS:
        if label not in columns:
            raise StageFailure(f"{manifest_path} is missing required label column: {label}")
        if dataframe[label].isna().any():
            raise StageFailure(f"{manifest_path} has NaN values in label column: {label}")

        numeric_labels = pd.to_numeric(dataframe[label], errors="coerce")
        invalid_mask = numeric_labels.isna() | ~numeric_labels.isin([0, 1])
        if invalid_mask.any():
            invalid_values = sorted(pd.unique(dataframe.loc[invalid_mask, label]).tolist())
            raise StageFailure(
                f"{manifest_path} has non-binary values in label column {label}: {invalid_values}"
            )
        dataframe[label] = numeric_labels.astype(np.int64)

    dataframe["resolved_path"] = dataframe[path_column].map(lambda value: resolve_image_path(value, manifest_path))
    missing_paths = [str(path) for path in dataframe["resolved_path"] if not Path(path).exists()]
    if missing_paths:
        preview = missing_paths[:5]
        raise StageFailure(
            f"{manifest_path} has {len(missing_paths)} missing image paths. Examples: {preview}"
        )

    label_counts = {
        label: {
            "positives": int(dataframe[label].sum()),
            "negatives": int(len(dataframe) - dataframe[label].sum()),
        }
        for label in LABELS
    }

    summary = {
        "manifest_path": str(manifest_path.resolve()),
        "split_name": split_name,
        "path_column": path_column,
        "dicom_id_available": dicom_id_available,
        "num_images": int(len(dataframe)),
        "num_subjects": int(dataframe["subject_id"].nunique()),
        "num_studies": int(dataframe["study_id"].nunique()),
        "num_dicoms": int(dataframe["dicom_id"].nunique()) if dicom_id_available else None,
        "label_counts": label_counts,
    }
    return dataframe, summary


def print_split_summary(summary: dict[str, Any]) -> None:
    print(f"[{summary['split_name']}] images: {summary['num_images']}")
    print(f"[{summary['split_name']}] subjects: {summary['num_subjects']}")
    print(f"[{summary['split_name']}] studies: {summary['num_studies']}")
    if summary["dicom_id_available"]:
        print(f"[{summary['split_name']}] dicom_id column: present")
    else:
        print(f"[{summary['split_name']}] dicom_id column: not available")
    for label in LABELS:
        counts = summary["label_counts"][label]
        print(
            f"[{summary['split_name']}] {label}: "
            f"positives={counts['positives']}, negatives={counts['negatives']}"
        )


def normalize_overlap_values(series: pd.Series) -> set[str]:
    normalized = series.dropna().astype("string").str.strip()
    normalized = normalized[normalized != ""]
    return set(normalized.tolist())


def summarize_overlap(left_values: set[str], right_values: set[str]) -> dict[str, Any]:
    overlap = sorted(left_values & right_values)
    return {"count": int(len(overlap)), "examples": overlap[:5]}


def choose_identity_column(dataframe: pd.DataFrame) -> str:
    if "resolved_path" in dataframe.columns:
        return "resolved_path"
    return choose_existing_column(list(dataframe.columns), IMAGE_PATH_COLUMNS, "an image path column")


def check_leakage(split_frames: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {
        "subject_overlap": {},
        "study_overlap": {},
        "dicom_overlap": {},
        "path_overlap": {},
    }
    failures: list[str] = []

    for left_name, right_name in combinations(split_frames.keys(), 2):
        pair_name = f"{left_name}_vs_{right_name}"
        left_frame = split_frames[left_name]
        right_frame = split_frames[right_name]

        subject_overlap = summarize_overlap(
            normalize_overlap_values(left_frame["subject_id"]),
            normalize_overlap_values(right_frame["subject_id"]),
        )
        checks["subject_overlap"][pair_name] = subject_overlap
        if subject_overlap["count"] > 0:
            failures.append(f"subject_id leakage between {left_name} and {right_name}: {subject_overlap['examples']}")

        study_overlap = summarize_overlap(
            normalize_overlap_values(left_frame["study_id"]),
            normalize_overlap_values(right_frame["study_id"]),
        )
        checks["study_overlap"][pair_name] = study_overlap
        if study_overlap["count"] > 0:
            failures.append(f"study_id leakage between {left_name} and {right_name}: {study_overlap['examples']}")

        left_path_column = choose_identity_column(left_frame)
        right_path_column = choose_identity_column(right_frame)
        path_overlap = summarize_overlap(
            normalize_overlap_values(left_frame[left_path_column]),
            normalize_overlap_values(right_frame[right_path_column]),
        )
        checks["path_overlap"][pair_name] = path_overlap
        if path_overlap["count"] > 0:
            failures.append(f"image path leakage between {left_name} and {right_name}: {path_overlap['examples']}")

        if "dicom_id" in left_frame.columns and "dicom_id" in right_frame.columns:
            dicom_overlap = summarize_overlap(
                normalize_overlap_values(left_frame["dicom_id"]),
                normalize_overlap_values(right_frame["dicom_id"]),
            )
        else:
            dicom_overlap = {"count": None, "examples": [], "note": "dicom_id not available in both splits"}
        checks["dicom_overlap"][pair_name] = dicom_overlap
        if dicom_overlap["count"] not in (None, 0):
            failures.append(f"dicom_id leakage between {left_name} and {right_name}: {dicom_overlap['examples']}")

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
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = MIMICDataset(dataframe=dataframe, image_size=image_size)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def build_model(num_labels: int = len(LABELS)) -> nn.Module:
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_labels)
    return model


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        raise StageFailure("Checkpoint state dict is empty.")
    if not all(str(key).startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {str(key)[7:]: value for key, value in state_dict.items()}


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    warnings_list: list[str],
) -> tuple[nn.Module, dict[str, Any]]:
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
        ]:
            if key in checkpoint:
                metadata[key] = checkpoint[key]

        if metadata.get("label_names") is not None and list(metadata["label_names"]) != LABELS:
            raise StageFailure(
                "Checkpoint label order does not match the required order. "
                f"Expected {LABELS}, found {list(metadata['label_names'])}"
            )
        if metadata.get("num_classes") is not None and int(metadata["num_classes"]) != len(LABELS):
            raise StageFailure(
                f"Checkpoint num_classes={metadata['num_classes']} does not match {len(LABELS)}"
            )
    elif hasattr(checkpoint, "keys"):
        state_dict = dict(checkpoint)

    if state_dict is None:
        raise StageFailure(f"Could not find a model state dict inside checkpoint: {checkpoint_path}")

    if not metadata:
        warnings_list.append(
            "Checkpoint metadata was not found. Continuing with the known DenseNet-121 architecture."
        )

    state_dict = normalize_state_dict_keys(state_dict)
    required_classifier_keys = {"classifier.weight", "classifier.bias"}
    missing_classifier_keys = sorted(key for key in required_classifier_keys if key not in state_dict)
    if missing_classifier_keys:
        raise StageFailure(
            "Checkpoint is missing classifier parameters. "
            f"Expected keys: {missing_classifier_keys}"
        )

    model = build_model(num_labels=len(LABELS))
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise StageFailure(f"Checkpoint weights could not be loaded into DenseNet-121: {exc}") from exc

    model = model.to(device)
    return model, metadata


def freeze_backbone(model: nn.Module) -> dict[str, Any]:
    total_parameters = 0
    trainable_parameters = 0
    trainable_parameter_names: list[str] = []
    backbone_trainable_names: list[str] = []
    classifier_frozen_names: list[str] = []

    for name, parameter in model.named_parameters():
        if name.startswith("classifier."):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False

        total_parameters += parameter.numel()
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()
            trainable_parameter_names.append(name)
            if not name.startswith("classifier."):
                backbone_trainable_names.append(name)
        elif name.startswith("classifier."):
            classifier_frozen_names.append(name)

    summary = {
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "trainable_parameter_names": trainable_parameter_names,
        "backbone_trainable_names": backbone_trainable_names,
        "classifier_frozen_names": classifier_frozen_names,
    }
    return summary


def verify_frozen_backbone(summary: dict[str, Any]) -> None:
    if summary["backbone_trainable_names"]:
        raise StageFailure(
            "Backbone parameters are still trainable: "
            f"{summary['backbone_trainable_names']}"
        )
    if summary["classifier_frozen_names"]:
        raise StageFailure(
            "Classifier parameters are frozen, but they must stay trainable: "
            f"{summary['classifier_frozen_names']}"
        )
    if not summary["trainable_parameter_names"]:
        raise StageFailure("No trainable parameters remain after freezing. Classifier should stay trainable.")
    if not all(name.startswith("classifier.") for name in summary["trainable_parameter_names"]):
        raise StageFailure(
            "Only classifier parameters should be trainable, but found: "
            f"{summary['trainable_parameter_names']}"
        )


def set_head_only_train_mode(model: nn.Module) -> None:
    model.eval()
    model.classifier.train()


def compute_binary_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "macro_auroc": None,
        "macro_auprc": None,
        "defined_auroc_labels": 0,
        "defined_auprc_labels": 0,
        "per_label": {},
    }

    macro_aurocs: list[float] = []
    macro_auprcs: list[float] = []

    for label_index, label_name in enumerate(LABELS):
        y_true = targets[:, label_index]
        y_prob = probabilities[:, label_index]
        label_metrics: dict[str, Any] = {
            "positives": int(y_true.sum()),
            "negatives": int((1 - y_true).sum()),
            "auroc": None,
            "auprc": None,
            "auroc_defined": False,
            "auprc_defined": False,
            "probability_mean": float(np.mean(y_prob)),
            "probability_std": float(np.std(y_prob)),
        }

        if len(np.unique(y_true)) < 2:
            label_metrics["reason"] = "only one class present in this split"
        else:
            auroc = float(roc_auc_score(y_true, y_prob))
            auprc = float(average_precision_score(y_true, y_prob))
            label_metrics["auroc"] = auroc
            label_metrics["auprc"] = auprc
            label_metrics["auroc_defined"] = True
            label_metrics["auprc_defined"] = True
            macro_aurocs.append(auroc)
            macro_auprcs.append(auprc)

        metrics["per_label"][label_name] = label_metrics

    if macro_aurocs:
        metrics["macro_auroc"] = float(np.mean(macro_aurocs))
        metrics["defined_auroc_labels"] = len(macro_aurocs)
    if macro_auprcs:
        metrics["macro_auprc"] = float(np.mean(macro_auprcs))
        metrics["defined_auprc_labels"] = len(macro_auprcs)

    return metrics


def evaluate_split(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, Any]:
    total_loss = 0.0
    total_items = 0
    all_logits: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise StageFailure("Validation or test loss became NaN or infinite.")

            probabilities = torch.sigmoid(logits)
            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

            all_logits.append(logits.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    if not all_targets:
        raise StageFailure("Evaluation split is empty. Cannot compute metrics.")

    logits_np = np.concatenate(all_logits, axis=0)
    probabilities_np = np.concatenate(all_probabilities, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    metrics = compute_binary_metrics(targets_np, probabilities_np)
    metrics["loss"] = float(total_loss / max(total_items, 1))
    metrics["logits"] = logits_np
    metrics["probabilities"] = probabilities_np
    metrics["targets"] = targets_np
    return metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    set_head_only_train_mode(model)
    total_loss = 0.0
    total_items = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise StageFailure("Training loss became NaN or infinite.")
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return float(total_loss / max(total_items, 1))


def metrics_for_report(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": float(metrics["loss"]),
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "defined_auroc_labels": int(metrics["defined_auroc_labels"]),
        "defined_auprc_labels": int(metrics["defined_auprc_labels"]),
        "per_label": metrics["per_label"],
    }


def save_predictions_csv(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    output_path: Path,
    path_column: str,
) -> str:
    ensure_dir(output_path.parent)
    rows: list[dict[str, Any]] = []
    for row_index, row in dataframe.reset_index(drop=True).iterrows():
        record: dict[str, Any] = {
            "image_path": str(row["resolved_path"]),
            "subject_id": row["subject_id"],
            "study_id": row["study_id"],
        }
        if "dicom_id" in dataframe.columns:
            record["dicom_id"] = row["dicom_id"]
        if path_column in dataframe.columns:
            record["manifest_image_path"] = row[path_column]

        for label in LABELS:
            record[f"true_{label}"] = int(row[label])
        for label_index, label in enumerate(LABELS):
            record[f"pred_{label}"] = float(probabilities[row_index, label_index])

        rows.append(record)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    return str(output_path.resolve())


def save_loss_curve(train_losses: list[float], val_losses: list[float], output_path: Path) -> str:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, marker="o", label="support train loss")
    plt.plot(epochs, val_losses, marker="o", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Head-only adaptation loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path.resolve())


def checkpoint_payload(
    model: nn.Module,
    epoch: int,
    train_loss: float,
    val_metrics: dict[str, Any],
    args: argparse.Namespace,
    source_checkpoint: str,
    parameter_summary: dict[str, Any],
    best_metric_name: str,
    best_metric_value: float,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_metrics["loss"]),
        "label_names": LABELS,
        "num_classes": len(LABELS),
        "architecture": "torchvision_densenet121",
        "image_size": int(args.image_size),
        "adaptation_method": "head_only",
        "source_checkpoint": source_checkpoint,
        "best_metric_name": best_metric_name,
        "best_metric_value": float(best_metric_value),
        "parameter_summary": parameter_summary,
        "model_state_dict": model.state_dict(),
    }


def load_source_only_report(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        raise StageFailure(f"Missing source-only report: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StageFailure(f"Could not read source-only report {report_path}: {exc}") from exc

    for key in ["val_metrics", "test_metrics"]:
        if key not in report:
            raise StageFailure(f"Source-only report is missing required section: {key}")
        if "macro_auroc" not in report[key] or "macro_auprc" not in report[key]:
            raise StageFailure(f"Source-only report section {key} is missing macro AUROC/AUPRC.")

    return report


def metric_delta(current_value: float | None, baseline_value: float | None) -> float | None:
    if current_value is None or baseline_value is None:
        return None
    return float(current_value - baseline_value)


def select_is_better(current_metrics: dict[str, Any], best_record: dict[str, Any] | None) -> tuple[bool, str, float]:
    current_auroc = current_metrics["macro_auroc"]
    current_loss = float(current_metrics["loss"])

    if current_auroc is not None:
        if best_record is None or best_record["macro_auroc"] is None:
            return True, "val_macro_auroc", float(current_auroc)
        if current_auroc > best_record["macro_auroc"] + 1e-12:
            return True, "val_macro_auroc", float(current_auroc)
        if math.isclose(current_auroc, best_record["macro_auroc"], abs_tol=1e-12):
            if current_loss < best_record["loss"] - 1e-12:
                return True, "val_macro_auroc", float(current_auroc)
        return False, "val_macro_auroc", float(current_auroc)

    if best_record is None or best_record["macro_auroc"] is not None:
        return True, "val_loss", float(current_loss)
    if current_loss < best_record["loss"] - 1e-12:
        return True, "val_loss", float(current_loss)
    return False, "val_loss", float(current_loss)


def format_metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def empty_split_summary(split_name: str) -> dict[str, Any]:
    return {
        "manifest_path": None,
        "split_name": split_name,
        "path_column": None,
        "dicom_id_available": False,
        "num_images": 0,
        "num_subjects": 0,
        "num_studies": 0,
        "num_dicoms": None,
        "label_counts": {
            label: {"positives": 0, "negatives": 0}
            for label in LABELS
        },
    }


def initial_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "FAILED",
        "safe_to_continue": False,
        "goal": "Adapt the NIH-trained DenseNet-121 model to MIMIC using head-only few-shot learning.",
        "run_name": args.run_name,
        "adaptation_method": "head_only",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "support_manifest": str(Path(args.support_csv).resolve()) if args.support_csv else None,
        "val_manifest": str(Path(args.val_csv).resolve()) if args.val_csv else None,
        "test_manifest": str(Path(args.test_csv).resolve()) if args.test_csv else None,
        "source_only_report": str(Path(args.source_only_report).resolve()) if args.source_only_report else None,
        "label_policy": args.label_policy,
        "base_dir": args.base_dir,
        "label_order": LABELS,
        "training_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "image_size": int(args.image_size),
            "patience": int(args.patience),
            "seed": int(args.seed),
        },
        "checkpoint_metadata": {},
        "support_counts": empty_split_summary("support"),
        "val_counts": empty_split_summary("val"),
        "test_counts": empty_split_summary("test"),
        "leakage_checks": {},
        "parameter_summary": {},
        "training_history": [],
        "best_epoch": None,
        "stopped_early": False,
        "best_checkpoint": None,
        "last_checkpoint": None,
        "prediction_files": {},
        "plot_files": {},
        "source_only_metrics": {},
        "metric_deltas": {},
        "val_metrics": {},
        "test_metrics": {},
        "device": None,
        "warnings": [],
        "failure_reasons": [],
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Mini-Stage F Head-only NIH to MIMIC Adaptation",
        "",
        "## Goal",
        "Adapt the NIH-trained DenseNet-121 model to MIMIC by freezing the DenseNet backbone and training only the classifier head on the small support set.",
        "",
        "## Inputs",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- support manifest: `{report['support_manifest']}`",
        f"- val manifest: `{report['val_manifest']}`",
        f"- test manifest: `{report['test_manifest']}`",
        f"- source-only report: `{report['source_only_report']}`",
        "",
        "## Training Setup",
        f"- run name: `{report['run_name']}`",
        f"- epochs: {report['training_config']['epochs']}",
        f"- batch size: {report['training_config']['batch_size']}",
        f"- learning rate: {report['training_config']['lr']}",
        f"- image size: {report['training_config']['image_size']}",
        f"- patience: {report['training_config']['patience']}",
        f"- seed: {report['training_config']['seed']}",
        f"- device: `{report['device']}`",
        "",
        "## Split Sizes",
    ]

    for split_name in ["support_counts", "val_counts", "test_counts"]:
        split = report[split_name]
        lines.append(f"### {split['split_name']}")
        lines.append(f"- images: {split['num_images']}")
        lines.append(f"- subjects: {split['num_subjects']}")
        lines.append(f"- studies: {split['num_studies']}")
        if split["dicom_id_available"]:
            lines.append(f"- dicoms: {split['num_dicoms']}")
        else:
            lines.append("- dicom_id: not available")
        for label in LABELS:
            counts = split["label_counts"][label]
            lines.append(
                f"- {label}: positives={counts['positives']}, negatives={counts['negatives']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Parameter Freeze Check",
            f"- total parameters: {report['parameter_summary'].get('total_parameters')}",
            f"- trainable parameters: {report['parameter_summary'].get('trainable_parameters')}",
            f"- trainable parameter names: {report['parameter_summary'].get('trainable_parameter_names')}",
            "",
            "## Training Outcome",
            f"- best epoch: {report['best_epoch']}",
            f"- stopped early: {'yes' if report['stopped_early'] else 'no'}",
            f"- best checkpoint: `{report['best_checkpoint']}`",
            f"- last checkpoint: `{report['last_checkpoint']}`",
        ]
    )

    if report["training_history"]:
        last_epoch = report["training_history"][-1]
        lines.append(f"- final train loss: {last_epoch['train_loss']:.4f}")
        lines.append(f"- final val loss: {last_epoch['val_loss']:.4f}")
        lines.append(f"- final val macro AUROC: {format_metric(last_epoch['val_macro_auroc'])}")
        lines.append(f"- final val macro AUPRC: {format_metric(last_epoch['val_macro_auprc'])}")

    for section_name in ["val_metrics", "test_metrics"]:
        metrics = report.get(section_name, {})
        lines.extend(["", f"## {section_name.replace('_', ' ').title()}"])
        if not metrics:
            lines.append("- not available")
            continue
        lines.append(f"- loss: {metrics['loss']:.4f}")
        lines.append(f"- macro AUROC: {format_metric(metrics['macro_auroc'])}")
        lines.append(f"- macro AUPRC: {format_metric(metrics['macro_auprc'])}")
        for label in LABELS:
            item = metrics["per_label"][label]
            lines.append(
                f"- {label}: "
                f"AUROC={format_metric(item['auroc'])}, "
                f"AUPRC={format_metric(item['auprc'])}, "
                f"prob_mean={item['probability_mean']:.4f}, "
                f"prob_std={item['probability_std']:.4f}"
            )
            if "reason" in item:
                lines.append(f"  note: {item['reason']}")

    if report["source_only_metrics"]:
        lines.extend(
            [
                "",
                "## Source-only Comparison",
                f"- source-only val macro AUROC: {format_metric(report['source_only_metrics']['val']['macro_auroc'])}",
                f"- source-only val macro AUPRC: {format_metric(report['source_only_metrics']['val']['macro_auprc'])}",
                f"- source-only test macro AUROC: {format_metric(report['source_only_metrics']['test']['macro_auroc'])}",
                f"- source-only test macro AUPRC: {format_metric(report['source_only_metrics']['test']['macro_auprc'])}",
                f"- val macro AUROC delta: {format_metric(report['metric_deltas'].get('val_macro_auroc_delta'))}",
                f"- val macro AUPRC delta: {format_metric(report['metric_deltas'].get('val_macro_auprc_delta'))}",
                f"- test macro AUROC delta: {format_metric(report['metric_deltas'].get('test_macro_auroc_delta'))}",
                f"- test macro AUPRC delta: {format_metric(report['metric_deltas'].get('test_macro_auprc_delta'))}",
            ]
        )

    lines.extend(["", "## Warnings"])
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")

    lines.extend(["", "## Final Decision"])
    lines.append(f"- status: {report['status']}")
    lines.append(f"- safe to continue: {'yes' if report['safe_to_continue'] else 'no'}")

    if report["failure_reasons"]:
        lines.extend(["", "## Failure Reasons"])
        for reason in report["failure_reasons"]:
            lines.append(f"- {reason}")

    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], report_json_path: Path, report_md_path: Path) -> None:
    save_json(json_ready(report), report_json_path)
    ensure_dir(report_md_path.parent)
    report_md_path.write_text(build_markdown_report(report), encoding="utf-8")


def run_adaptation(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    set_seed(args.seed)

    namespace = args._namespace_config
    artifact_paths = args._artifact_paths
    checkpoints_dir = namespace.checkpoints_dir
    outputs_dir = namespace.outputs_dir
    reports_dir = namespace.reports_dir
    ensure_dir(checkpoints_dir)
    ensure_dir(outputs_dir)
    ensure_dir(reports_dir)

    best_checkpoint_path = artifact_paths["best_checkpoint"]
    last_checkpoint_path = artifact_paths["last_checkpoint"]
    val_predictions_path = artifact_paths["val_predictions"]
    test_predictions_path = artifact_paths["test_predictions"]
    loss_curve_path = artifact_paths["loss_curve"]
    report_json_path = artifact_paths["report_json"]
    report_md_path = artifact_paths["report_md"]

    try:
        enforce_policy_b_manifest_guard(
            args.label_policy,
            val_manifest=Path(args.val_csv) if args.val_csv else None,
            test_manifest=Path(args.test_csv) if args.test_csv else None,
        )
    except ValueError as exc:
        raise StageFailure(str(exc)) from exc
    print_runtime_configuration(args)
    missing_files = collect_missing_paths(required_input_paths(args))
    if missing_files:
        report["failure_reasons"].extend(missing_files)
        raise StageFailure("Required files are missing.")

    source_only_report = load_source_only_report(Path(args.source_only_report))
    report["source_only_metrics"] = {
        "val": {
            "macro_auroc": source_only_report["val_metrics"].get("macro_auroc"),
            "macro_auprc": source_only_report["val_metrics"].get("macro_auprc"),
        },
        "test": {
            "macro_auroc": source_only_report["test_metrics"].get("macro_auroc"),
            "macro_auprc": source_only_report["test_metrics"].get("macro_auprc"),
        },
    }

    support_df, support_summary = validate_manifest(Path(args.support_csv), "support")
    val_df, val_summary = validate_manifest(Path(args.val_csv), "val")
    test_df, test_summary = validate_manifest(Path(args.test_csv), "test")
    report["support_counts"] = support_summary
    report["val_counts"] = val_summary
    report["test_counts"] = test_summary

    print_split_summary(support_summary)
    print_split_summary(val_summary)
    print_split_summary(test_summary)

    leakage_checks, leakage_failures = check_leakage(
        {"support": support_df, "val": val_df, "test": test_df}
    )
    report["leakage_checks"] = leakage_checks
    if leakage_failures:
        report["failure_reasons"].extend(leakage_failures)
        raise StageFailure("Leakage detected between support, val, and test.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report["device"] = str(device)
    if args.debug:
        print(f"device: {device}")

    model, checkpoint_metadata = load_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        warnings_list=report["warnings"],
    )
    report["checkpoint_metadata"] = checkpoint_metadata

    parameter_summary = freeze_backbone(model)
    report["parameter_summary"] = parameter_summary
    print(f"total parameters: {parameter_summary['total_parameters']}")
    print(f"trainable parameters: {parameter_summary['trainable_parameters']}")
    print(f"trainable parameter names: {parameter_summary['trainable_parameter_names']}")
    verify_frozen_backbone(parameter_summary)

    support_loader = build_dataloader(
        support_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = build_dataloader(
        val_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    best_record: dict[str, Any] | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, support_loader, optimizer, criterion, device)
        val_metrics = evaluate_split(model, val_loader, device, criterion)
        history_item = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
        }
        report["training_history"].append(history_item)

        is_better, metric_name, metric_value = select_is_better(val_metrics, best_record)
        if is_better:
            best_record = {
                "epoch": int(epoch),
                "loss": float(val_metrics["loss"]),
                "macro_auroc": val_metrics["macro_auroc"],
                "macro_auprc": val_metrics["macro_auprc"],
                "metric_name": metric_name,
                "metric_value": float(metric_value),
            }
            torch.save(
                checkpoint_payload(
                    model=model,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_metrics=val_metrics,
                    args=args,
                    source_checkpoint=str(Path(args.checkpoint).resolve()),
                    parameter_summary=parameter_summary,
                    best_metric_name=metric_name,
                    best_metric_value=float(metric_value),
                ),
                best_checkpoint_path,
            )
            report["best_epoch"] = int(epoch)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_macro_auroc={format_metric(val_metrics['macro_auroc'])} | "
            f"val_macro_auprc={format_metric(val_metrics['macro_auprc'])}"
        )

        if epochs_without_improvement >= args.patience:
            stopped_early = True
            if args.debug:
                print(f"early stopping at epoch {epoch} after {epochs_without_improvement} non-improving epochs")
            break

    report["stopped_early"] = stopped_early
    if best_record is None or not best_checkpoint_path.exists():
        raise StageFailure("Training did not produce a best checkpoint.")

    final_epoch = report["training_history"][-1]["epoch"]
    final_train_loss = report["training_history"][-1]["train_loss"]
    final_val_metrics = evaluate_split(model, val_loader, device, criterion)
    torch.save(
        checkpoint_payload(
            model=model,
            epoch=int(final_epoch),
            train_loss=float(final_train_loss),
            val_metrics=final_val_metrics,
            args=args,
            source_checkpoint=str(Path(args.checkpoint).resolve()),
            parameter_summary=parameter_summary,
            best_metric_name=best_record["metric_name"],
            best_metric_value=float(best_record["metric_value"]),
        ),
        last_checkpoint_path,
    )
    report["best_checkpoint"] = str(best_checkpoint_path.resolve())
    report["last_checkpoint"] = str(last_checkpoint_path.resolve())

    train_losses = [item["train_loss"] for item in report["training_history"]]
    val_losses = [item["val_loss"] for item in report["training_history"]]
    report["plot_files"]["loss_curve"] = save_loss_curve(train_losses, val_losses, loss_curve_path)

    best_model, _ = load_checkpoint(best_checkpoint_path, device, report["warnings"])
    reloaded_parameter_summary = freeze_backbone(best_model)
    verify_frozen_backbone(reloaded_parameter_summary)

    final_val_raw = evaluate_split(best_model, val_loader, device, criterion)
    test_loader = build_dataloader(
        test_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    final_test_raw = evaluate_split(best_model, test_loader, device, criterion)

    report["val_metrics"] = metrics_for_report(final_val_raw)
    report["test_metrics"] = metrics_for_report(final_test_raw)
    report["prediction_files"] = {
        "val": save_predictions_csv(
            dataframe=val_df,
            probabilities=final_val_raw["probabilities"],
            output_path=val_predictions_path,
            path_column=val_summary["path_column"],
        ),
        "test": save_predictions_csv(
            dataframe=test_df,
            probabilities=final_test_raw["probabilities"],
            output_path=test_predictions_path,
            path_column=test_summary["path_column"],
        ),
    }

    report["metric_deltas"] = {
        "val_macro_auroc_delta": metric_delta(
            report["val_metrics"]["macro_auroc"],
            report["source_only_metrics"]["val"]["macro_auroc"],
        ),
        "val_macro_auprc_delta": metric_delta(
            report["val_metrics"]["macro_auprc"],
            report["source_only_metrics"]["val"]["macro_auprc"],
        ),
        "test_macro_auroc_delta": metric_delta(
            report["test_metrics"]["macro_auroc"],
            report["source_only_metrics"]["test"]["macro_auroc"],
        ),
        "test_macro_auprc_delta": metric_delta(
            report["test_metrics"]["macro_auprc"],
            report["source_only_metrics"]["test"]["macro_auprc"],
        ),
    }

    write_reports(report, report_json_path, report_md_path)

    for output_path in [
        best_checkpoint_path,
        last_checkpoint_path,
        val_predictions_path,
        test_predictions_path,
        report_json_path,
        report_md_path,
        loss_curve_path,
    ]:
        if not output_path.exists():
            raise StageFailure(f"Expected output file was not created: {output_path}")

    report["status"] = "DONE"
    report["safe_to_continue"] = True
    write_reports(report, report_json_path, report_md_path)
    return report


def main() -> None:
    try:
        args = resolve_cli_args(parse_args())
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    if args.dry_run:
        try:
            run_dry_run(args)
        except StageFailure as exc:
            print(str(exc))
            sys.exit(1)
        print("DRY_RUN_OK")
        return

    report = initial_report(args)
    reports_dir = args._namespace_config.reports_dir
    ensure_dir(reports_dir)
    report_json_path = args._artifact_paths["report_json"]
    report_md_path = args._artifact_paths["report_md"]

    try:
        report = run_adaptation(args, report)
    except StageFailure as exc:
        if not report["failure_reasons"]:
            report["failure_reasons"].append(str(exc))
        if report["training_history"] or report["prediction_files"]:
            report["status"] = "PARTIAL"
        else:
            report["status"] = "FAILED"
        report["safe_to_continue"] = False
    except Exception as exc:  # pragma: no cover - depends on unexpected runtime failures
        report["failure_reasons"].append(f"Unexpected error: {exc}")
        report["warnings"].append("Unexpected traceback was captured in the report.")
        report["warnings"].append(traceback.format_exc())
        if report["training_history"] or report["prediction_files"]:
            report["status"] = "PARTIAL"
        else:
            report["status"] = "FAILED"
        report["safe_to_continue"] = False

    write_reports(report, report_json_path, report_md_path)
    print(report["status"])
    if report["status"] != "DONE":
        sys.exit(1)


if __name__ == "__main__":
    main()
