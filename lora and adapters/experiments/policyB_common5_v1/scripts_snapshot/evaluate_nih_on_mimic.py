#!/usr/bin/env python3
"""Evaluate the NIH 2k DenseNet-121 checkpoint on MIMIC common5 without adaptation."""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Any

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
    enforce_policy_b_manifest_guard,
    print_resolved_configuration,
    resolve_input_path,
    resolve_manifest_path,
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
    parser = argparse.ArgumentParser(
        description="Evaluate a trained NIH DenseNet-121 model on MIMIC common5 without adaptation."
    )
    parser.add_argument("--checkpoint", "--source_checkpoint", dest="checkpoint", type=str, default=None)
    parser.add_argument("--val_csv", "--manifest_val", dest="val_csv", type=str, default=None)
    parser.add_argument("--test_csv", "--manifest_test", dest="test_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--label_policy", type=str, default="uignore_blankzero")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    namespace = build_namespace_config(args.base_dir, args.out_dir)
    legacy_naming = args.run_name is None
    if args.run_name is None:
        args.run_name = "nih_to_mimic"

    args.base_dir = str(namespace.base_dir) if namespace.base_dir is not None else None
    args.out_dir = str(namespace.output_root)
    args.checkpoint = str(
        resolve_input_path(args.checkpoint, default_relative="checkpoints/nih_2k_densenet121_best.pt")
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
    args.val_csv = str(val_manifest) if val_manifest is not None else None
    args.test_csv = str(test_manifest) if test_manifest is not None else None
    args._namespace_config = namespace
    if legacy_naming:
        args._artifact_paths = {
            "report_json": (namespace.reports_dir / "mini_stage_d_nih_to_mimic.json").resolve(),
            "report_md": (namespace.reports_dir / "mini_stage_d_nih_to_mimic.md").resolve(),
            "val_predictions": (namespace.outputs_dir / "nih_to_mimic_val_predictions.csv").resolve(),
            "test_predictions": (namespace.outputs_dir / "nih_to_mimic_test_predictions.csv").resolve(),
        }
    else:
        args._artifact_paths = build_named_run_paths(namespace, args.run_name)
    return args


def required_input_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    return {
        "checkpoint": Path(args.checkpoint) if args.checkpoint else None,
        "val_manifest": Path(args.val_csv) if args.val_csv else None,
        "test_manifest": Path(args.test_csv) if args.test_csv else None,
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
        source_checkpoint=Path(args.checkpoint) if args.checkpoint else None,
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


def read_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise StageFailure(f"Missing manifest: {manifest_path}")
    try:
        return pd.read_csv(manifest_path)
    except Exception as exc:  # pragma: no cover - depends on local file corruption
        raise StageFailure(f"Could not read manifest {manifest_path}: {exc}") from exc


def validate_manifest(manifest_path: Path, split_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataframe = read_manifest(manifest_path).copy()
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
            bad_values = sorted(pd.unique(dataframe.loc[invalid_mask, label]))
            raise StageFailure(
                f"{manifest_path} has non-binary values in label column {label}: {bad_values}"
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


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
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

        for key in ["label_names", "model_info", "num_classes", "architecture", "image_size"]:
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
        raise StageFailure(
            f"Could not find a model state dict inside checkpoint: {checkpoint_path}"
        )

    if not metadata:
        warnings_list.append(
            "Checkpoint metadata was not found. Continuing with the known Mini-Stage B architecture."
        )

    model = build_model(num_labels=len(LABELS))
    state_dict = normalize_state_dict_keys(state_dict)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise StageFailure(f"Checkpoint weights could not be loaded into DenseNet-121: {exc}") from exc

    model = model.to(device)
    model.eval()
    return model, metadata


def build_dataloader(dataframe: pd.DataFrame, image_size: int, batch_size: int) -> DataLoader:
    dataset = MIMICDataset(dataframe=dataframe, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


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
) -> dict[str, Any]:
    criterion = nn.BCEWithLogitsLoss()
    all_logits: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_loss = 0.0
    total_items = 0

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
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


def save_predictions_csv(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    output_path: Path,
    path_column: str,
) -> str:
    ensure_dir(output_path.parent)
    records: list[dict[str, Any]] = []
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

        records.append(record)

    pd.DataFrame(records).to_csv(output_path, index=False)
    return str(output_path.resolve())


def metrics_for_report(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": float(metrics["loss"]),
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "defined_auroc_labels": int(metrics["defined_auroc_labels"]),
        "defined_auprc_labels": int(metrics["defined_auprc_labels"]),
        "per_label": metrics["per_label"],
    }


def format_metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


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


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Mini-Stage D NIH to MIMIC Evaluation",
        "",
        "## Goal",
        "Evaluate the trained NIH 2k DenseNet-121 model directly on MIMIC common5 without any target adaptation.",
        "",
        "## Checkpoint Used",
        f"- checkpoint: `{report['checkpoint']}`",
        "",
        "## MIMIC Manifests Used",
        f"- val: `{report['val_manifest']}`",
        f"- test: `{report['test_manifest']}`",
        "",
        "## Split Sizes",
        f"- val images: {report['val_counts'].get('num_images')}",
        f"- val subjects: {report['val_counts'].get('num_subjects')}",
        f"- val studies: {report['val_counts'].get('num_studies')}",
        f"- test images: {report['test_counts'].get('num_images')}",
        f"- test subjects: {report['test_counts'].get('num_subjects')}",
        f"- test studies: {report['test_counts'].get('num_studies')}",
        "",
        "## Label Counts",
        "### Val",
    ]

    for label in LABELS:
        counts = report["val_counts"]["label_counts"][label]
        lines.append(
            f"- {label}: positives={counts['positives']}, negatives={counts['negatives']}"
        )

    lines.extend(["", "### Test"])
    for label in LABELS:
        counts = report["test_counts"]["label_counts"][label]
        lines.append(
            f"- {label}: positives={counts['positives']}, negatives={counts['negatives']}"
        )

    for split_name in ["val_metrics", "test_metrics"]:
        split_metrics = report.get(split_name)
        lines.extend(["", f"## {split_name.replace('_', ' ').title()}"])
        if not split_metrics:
            lines.append("- not available")
            continue

        lines.append(f"- loss: {split_metrics['loss']:.4f}")
        lines.append(f"- macro AUROC: {format_metric(split_metrics['macro_auroc'])}")
        lines.append(f"- macro AUPRC: {format_metric(split_metrics['macro_auprc'])}")
        for label in LABELS:
            item = split_metrics["per_label"][label]
            lines.append(
                f"- {label}: "
                f"AUROC={format_metric(item['auroc'])}, "
                f"AUPRC={format_metric(item['auprc'])}, "
                f"prob_mean={item['probability_mean']:.4f}, "
                f"prob_std={item['probability_std']:.4f}"
            )
            if "reason" in item:
                lines.append(f"  note: {item['reason']}")

    lines.extend(["", "## Warnings"])
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")

    lines.extend(["", "## Final Decision"])
    lines.append(f"- status: {report['status']}")
    lines.append(
        f"- safe to continue: {'yes' if report['safe_to_continue'] else 'no'}"
    )

    if report["failure_reasons"]:
        lines.extend(["", "## Failure Reasons"])
        for reason in report["failure_reasons"]:
            lines.append(f"- {reason}")

    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], report_json_path: Path, report_md_path: Path) -> None:
    save_json(report, report_json_path)
    ensure_dir(report_md_path.parent)
    report_md_path.write_text(build_markdown_report(report), encoding="utf-8")


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
        "run_name": args.run_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "val_manifest": str(Path(args.val_csv).resolve()) if args.val_csv else None,
        "test_manifest": str(Path(args.test_csv).resolve()) if args.test_csv else None,
        "label_policy": args.label_policy,
        "base_dir": args.base_dir,
        "label_order": LABELS,
        "checkpoint_metadata": {},
        "val_counts": empty_split_summary("val"),
        "test_counts": empty_split_summary("test"),
        "val_metrics": {},
        "test_metrics": {},
        "prediction_files": {},
        "warnings": [],
        "failure_reasons": [],
    }


def run_evaluation(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    set_seed(args.seed)

    namespace = args._namespace_config
    artifact_paths = args._artifact_paths
    reports_dir = namespace.reports_dir
    outputs_dir = namespace.outputs_dir
    ensure_dir(reports_dir)
    ensure_dir(outputs_dir)

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
        raise StageFailure(report["failure_reasons"][-1])

    val_df, val_summary = validate_manifest(Path(args.val_csv), "val")
    test_df, test_summary = validate_manifest(Path(args.test_csv), "test")
    report["val_counts"] = val_summary
    report["test_counts"] = test_summary

    print_split_summary(val_summary)
    print_split_summary(test_summary)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.debug:
        print(f"device: {device}")

    model, checkpoint_metadata = load_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        warnings_list=report["warnings"],
    )
    report["checkpoint_metadata"] = checkpoint_metadata
    if checkpoint_metadata and args.debug:
        print(f"checkpoint metadata keys: {sorted(checkpoint_metadata.keys())}")

    val_loader = build_dataloader(val_df, args.image_size, args.batch_size)
    test_loader = build_dataloader(test_df, args.image_size, args.batch_size)

    val_raw = evaluate_split(model, val_loader, device)
    report["val_metrics"] = metrics_for_report(val_raw)
    report["prediction_files"]["val"] = save_predictions_csv(
        dataframe=val_df,
        probabilities=val_raw["probabilities"],
        output_path=artifact_paths["val_predictions"],
        path_column=val_summary["path_column"],
    )

    test_raw = evaluate_split(model, test_loader, device)
    report["test_metrics"] = metrics_for_report(test_raw)
    report["prediction_files"]["test"] = save_predictions_csv(
        dataframe=test_df,
        probabilities=test_raw["probabilities"],
        output_path=artifact_paths["test_predictions"],
        path_column=test_summary["path_column"],
    )

    report["status"] = "DONE"
    report["safe_to_continue"] = True
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

    reports_dir = args._namespace_config.reports_dir
    ensure_dir(reports_dir)
    report_json_path = args._artifact_paths["report_json"]
    report_md_path = args._artifact_paths["report_md"]

    report = initial_report(args)

    try:
        report = run_evaluation(args, report)
    except StageFailure as exc:
        if not report["failure_reasons"]:
            report["failure_reasons"].append(str(exc))
        if report["val_metrics"] or report["prediction_files"]:
            report["status"] = "PARTIAL"
        else:
            report["status"] = "FAILED"
        report["safe_to_continue"] = False
    except Exception as exc:  # pragma: no cover - depends on unexpected runtime failures
        report["failure_reasons"].append(f"Unexpected error: {exc}")
        report["warnings"].append("Unexpected traceback was captured in the report.")
        report["warnings"].append(traceback.format_exc())
        if report["val_metrics"] or report["prediction_files"]:
            report["status"] = "PARTIAL"
        else:
            report["status"] = "FAILED"
        report["safe_to_continue"] = False

    write_reports(report, report_json_path, report_md_path)
    print(report["status"])
    if report["status"] == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
