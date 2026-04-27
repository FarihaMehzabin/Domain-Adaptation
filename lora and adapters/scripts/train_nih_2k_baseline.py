#!/usr/bin/env python3
"""Simple NIH 2k DenseNet-121 baseline for mini-stage B."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import textwrap
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
from torchvision.models import DenseNet121_Weights, densenet121
from torchvision.utils import make_grid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_namespace import (  # noqa: E402
    build_namespace_config,
    collect_missing_paths,
    print_resolved_configuration,
    resolve_manifest_path,
)


LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
IMAGE_PATH_COLUMNS = ["abs_path", "image_path", "file_path", "filepath", "path", "rel_path"]
PATIENT_ID_COLUMNS = ["Patient ID", "patient_id"]
IMAGE_ID_COLUMNS = ["image_id", "Image Index"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple NIH 2k DenseNet-121 baseline.")
    parser.add_argument("--train_csv", "--manifest_train", dest="train_csv", type=str, default=None)
    parser.add_argument("--val_csv", "--manifest_val", dest="val_csv", type=str, default=None)
    parser.add_argument("--test_csv", "--manifest_test", dest="test_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--label_policy", type=str, default="uignore_blankzero")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--mode", choices=["check", "overfit32", "train"], required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    namespace = build_namespace_config(args.base_dir, args.out_dir)
    args.base_dir = str(namespace.base_dir) if namespace.base_dir is not None else None
    args.out_dir = str(namespace.output_root)
    train_manifest = resolve_manifest_path(
        args.train_csv,
        namespace=namespace,
        default_filename="nih_dev_2k_train.csv",
    )
    val_manifest = resolve_manifest_path(
        args.val_csv,
        namespace=namespace,
        default_filename="nih_dev_2k_val.csv",
    )
    test_manifest = resolve_manifest_path(
        args.test_csv,
        namespace=namespace,
        default_filename="nih_dev_2k_test.csv",
    )
    args.train_csv = str(train_manifest) if train_manifest is not None else None
    args.val_csv = str(val_manifest) if val_manifest is not None else None
    args.test_csv = str(test_manifest) if test_manifest is not None else None
    args._namespace_config = namespace

    if args.run_name:
        prefix = args.run_name
        args._artifact_paths = {
            "manifest_json": (namespace.reports_dir / f"{prefix}_manifest_check.json").resolve(),
            "manifest_md": (namespace.reports_dir / f"{prefix}_manifest_check.md").resolve(),
            "batch_png": (namespace.reports_dir / f"{prefix}_batch.png").resolve(),
            "check_json": (namespace.reports_dir / f"{prefix}_check.json").resolve(),
            "check_md": (namespace.reports_dir / f"{prefix}_check.md").resolve(),
            "overfit_png": (namespace.reports_dir / f"{prefix}_overfit32.png").resolve(),
            "overfit_json": (namespace.reports_dir / f"{prefix}_overfit32.json").resolve(),
            "best_checkpoint": (namespace.checkpoints_dir / f"{prefix}_best.pt").resolve(),
            "last_checkpoint": (namespace.checkpoints_dir / f"{prefix}_last.pt").resolve(),
            "val_predictions": (namespace.outputs_dir / f"{prefix}_val_predictions.csv").resolve(),
            "test_predictions": (namespace.outputs_dir / f"{prefix}_test_predictions.csv").resolve(),
            "loss_curve_png": (namespace.reports_dir / f"{prefix}_loss_curve.png").resolve(),
            "train_json": (namespace.reports_dir / f"{prefix}.json").resolve(),
            "train_md": (namespace.reports_dir / f"{prefix}.md").resolve(),
        }
    else:
        args._artifact_paths = {
            "manifest_json": (namespace.reports_dir / "nih_2k_check.json").resolve(),
            "manifest_md": (namespace.reports_dir / "nih_2k_check.md").resolve(),
            "batch_png": (namespace.reports_dir / "mini_stage_b_batch.png").resolve(),
            "check_json": (namespace.reports_dir / "mini_stage_b_check.json").resolve(),
            "check_md": (namespace.reports_dir / "mini_stage_b_check.md").resolve(),
            "overfit_png": (namespace.reports_dir / "mini_stage_b_overfit32.png").resolve(),
            "overfit_json": (namespace.reports_dir / "mini_stage_b_overfit32.json").resolve(),
            "best_checkpoint": (namespace.checkpoints_dir / "nih_2k_densenet121_best.pt").resolve(),
            "last_checkpoint": (namespace.checkpoints_dir / "nih_2k_densenet121_last.pt").resolve(),
            "val_predictions": (namespace.outputs_dir / "nih_2k_val_predictions.csv").resolve(),
            "test_predictions": (namespace.outputs_dir / "nih_2k_test_predictions.csv").resolve(),
            "loss_curve_png": (namespace.reports_dir / "mini_stage_b_loss_curve.png").resolve(),
            "train_json": (namespace.reports_dir / "mini_stage_b_train.json").resolve(),
            "train_md": (namespace.reports_dir / "mini_stage_b_train.md").resolve(),
        }
    return args


def required_input_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    return {
        "train_manifest": Path(args.train_csv) if args.train_csv else None,
        "val_manifest": Path(args.val_csv) if args.val_csv else None,
        "test_manifest": Path(args.test_csv) if args.test_csv else None,
    }


def selected_report_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    artifact_paths = args._artifact_paths
    if args.mode == "check":
        return artifact_paths["check_json"], artifact_paths["check_md"]
    if args.mode == "overfit32":
        return artifact_paths["overfit_json"], None
    return artifact_paths["train_json"], artifact_paths["train_md"]


def print_runtime_configuration(args: argparse.Namespace) -> None:
    artifact_paths = args._artifact_paths
    report_json_path, report_md_path = selected_report_paths(args)
    checkpoint_output_path = artifact_paths["best_checkpoint"] if args.mode == "train" else None
    prediction_val_output_path = artifact_paths["val_predictions"] if args.mode == "train" else None
    prediction_test_output_path = artifact_paths["test_predictions"] if args.mode == "train" else None
    print_resolved_configuration(
        script_name=Path(__file__).name,
        base_dir=Path(args.base_dir) if args.base_dir else None,
        run_name=args.run_name,
        label_policy=args.label_policy,
        train_manifest=Path(args.train_csv) if args.train_csv else None,
        val_manifest=Path(args.val_csv) if args.val_csv else None,
        test_manifest=Path(args.test_csv) if args.test_csv else None,
        checkpoint_output_path=checkpoint_output_path,
        prediction_val_output_path=prediction_val_output_path,
        prediction_test_output_path=prediction_test_output_path,
        report_output_path=report_json_path,
        report_markdown_path=report_md_path,
    )


def run_dry_run(args: argparse.Namespace) -> None:
    print_runtime_configuration(args)
    missing_files = collect_missing_paths(required_input_paths(args))
    if missing_files:
        raise FileNotFoundError("Dry run failed:\n- " + "\n- ".join(missing_files))
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


def read_manifest(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")
    return pd.read_csv(csv_path)


def choose_existing_column(columns: list[str], candidates: list[str], description: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find {description}. Expected one of: {candidates}")


def normalize_path(raw_path: str | Path, manifest_path: Path) -> str:
    path = Path(str(raw_path))
    if path.is_absolute():
        return str(path)
    return str((manifest_path.parent.parent / path).resolve())


def validate_manifest(df: pd.DataFrame, split_name: str, manifest_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"manifest_path": str(manifest_path), "split_name": split_name}
    columns = list(df.columns)

    path_column = choose_existing_column(columns, IMAGE_PATH_COLUMNS, "an image path column")
    patient_column = choose_existing_column(columns, PATIENT_ID_COLUMNS, "a patient ID column")
    image_id_column = choose_existing_column(columns, IMAGE_ID_COLUMNS, "an image ID column")

    for label in LABELS:
        if label not in df.columns:
            raise ValueError(f"Missing label column '{label}' in {manifest_path}")
        if df[label].isna().any():
            raise ValueError(f"Label column '{label}' contains NaN values in {manifest_path}")
        invalid_values = sorted(
            value for value in pd.unique(df[label]) if value not in (0, 1, 0.0, 1.0)
        )
        if invalid_values:
            raise ValueError(
                f"Label column '{label}' contains non-binary values in {manifest_path}: {invalid_values}"
            )

    df = df.copy()
    df["__resolved_path__"] = df[path_column].map(lambda value: normalize_path(value, manifest_path))
    missing_paths = [path for path in df["__resolved_path__"] if not Path(path).exists()]
    if missing_paths:
        preview = missing_paths[:5]
        raise FileNotFoundError(
            f"{split_name} manifest has {len(missing_paths)} missing image paths. Examples: {preview}"
        )

    summary["path_column"] = path_column
    summary["patient_column"] = patient_column
    summary["image_id_column"] = image_id_column
    summary["num_images"] = int(len(df))
    summary["num_patients"] = int(df[patient_column].nunique())
    summary["label_counts"] = {label: int(df[label].sum()) for label in LABELS}

    return summary


def validate_all_manifests(
    train_csv: str, val_csv: str, test_csv: str
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    split_paths = {
        "train": Path(train_csv),
        "val": Path(val_csv),
        "test": Path(test_csv),
    }
    dataframes: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}

    for split_name, path in split_paths.items():
        df = read_manifest(path)
        summaries[split_name] = validate_manifest(df, split_name, path)
        resolved_path_column = summaries[split_name]["path_column"]
        df = df.copy()
        df["resolved_path"] = df[resolved_path_column].map(lambda value: normalize_path(value, path))
        dataframes[split_name] = df

    overlap_summary: dict[str, dict[str, int]] = {}
    split_names = list(split_paths.keys())
    for i, left_name in enumerate(split_names):
        for right_name in split_names[i + 1 :]:
            left_df = dataframes[left_name]
            right_df = dataframes[right_name]

            left_patient_column = summaries[left_name]["patient_column"]
            right_patient_column = summaries[right_name]["patient_column"]
            left_image_column = summaries[left_name]["image_id_column"]
            right_image_column = summaries[right_name]["image_id_column"]

            patient_overlap = set(left_df[left_patient_column].astype(str)) & set(
                right_df[right_patient_column].astype(str)
            )
            image_overlap = set(left_df[left_image_column].astype(str)) & set(
                right_df[right_image_column].astype(str)
            )

            if patient_overlap:
                preview = sorted(patient_overlap)[:10]
                raise ValueError(
                    f"Patient leakage detected between {left_name} and {right_name}: {preview}"
                )
            if image_overlap:
                preview = sorted(image_overlap)[:10]
                raise ValueError(f"Image overlap detected between {left_name} and {right_name}: {preview}")

            overlap_summary[f"{left_name}_vs_{right_name}"] = {
                "patient_overlap": 0,
                "image_overlap": 0,
            }

    return dataframes, {"splits": summaries, "overlap": overlap_summary}


class NIH2KDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: int = 224,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.image_size = image_size
        self.transform = transform or build_transform(image_size)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        image = Image.open(row["resolved_path"]).convert("RGB")
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(row[LABELS].values.astype(np.float32), dtype=torch.float32)
        return image_tensor, label_tensor


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
    dataset = NIH2KDataset(dataframe=dataframe, image_size=image_size)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def compute_pos_weight(dataframe: pd.DataFrame) -> torch.Tensor:
    positives = dataframe[LABELS].sum(axis=0).astype(np.float32)
    negatives = len(dataframe) - positives
    safe_positives = positives.replace(0, 1.0)
    values = (negatives / safe_positives).astype(np.float32).values
    return torch.tensor(values, dtype=torch.float32)


def build_model(prefer_pretrained: bool = True) -> tuple[nn.Module, dict[str, Any]]:
    info = {"pretrained_requested": prefer_pretrained, "pretrained_used": False, "pretrained_error": None}
    weights = None
    if prefer_pretrained:
        try:
            weights = DenseNet121_Weights.DEFAULT
        except Exception as error:  # pragma: no cover - extremely unlikely
            info["pretrained_error"] = str(error)
            weights = None

    try:
        model = densenet121(weights=weights)
        info["pretrained_used"] = weights is not None
    except Exception as error:
        model = densenet121(weights=None)
        info["pretrained_used"] = False
        info["pretrained_error"] = str(error)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, len(LABELS))
    return model, info


def save_batch_visualization(images: torch.Tensor, path: Path) -> None:
    ensure_dir(path.parent)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    denormalized = torch.clamp(images.cpu() * std + mean, 0.0, 1.0)
    grid = make_grid(denormalized[: min(len(denormalized), 8)], nrow=4)
    np_grid = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np_grid)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_line_plot(
    values: list[float],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    second_values: list[float] | None = None,
    second_label: str | None = None,
) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(8, 5))
    steps = list(range(1, len(values) + 1))
    plt.plot(steps, values, marker="o", label="train")
    if second_values is not None:
        second_steps = list(range(1, len(second_values) + 1))
        plt.plot(second_steps, second_values, marker="o", label=second_label or "val")
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def compute_binary_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "per_label": {},
        "macro_auroc": None,
        "macro_auprc": None,
        "defined_auroc_labels": 0,
        "defined_auprc_labels": 0,
    }

    auroc_values: list[float] = []
    auprc_values: list[float] = []

    for index, label in enumerate(LABELS):
        y_true = targets[:, index]
        y_score = probabilities[:, index]
        label_metrics: dict[str, Any] = {
            "positives": int(y_true.sum()),
            "negatives": int((1 - y_true).sum()),
            "auroc": None,
            "auprc": None,
            "auroc_defined": False,
            "auprc_defined": False,
        }

        if len(np.unique(y_true)) < 2:
            label_metrics["reason"] = "only one class present in this split"
        else:
            auroc = float(roc_auc_score(y_true, y_score))
            auprc = float(average_precision_score(y_true, y_score))
            label_metrics["auroc"] = auroc
            label_metrics["auprc"] = auprc
            label_metrics["auroc_defined"] = True
            label_metrics["auprc_defined"] = True
            auroc_values.append(auroc)
            auprc_values.append(auprc)

        metrics["per_label"][label] = label_metrics

    if auroc_values:
        metrics["macro_auroc"] = float(np.mean(auroc_values))
        metrics["defined_auroc_labels"] = len(auroc_values)
    if auprc_values:
        metrics["macro_auprc"] = float(np.mean(auprc_values))
        metrics["defined_auprc_labels"] = len(auprc_values)

    return metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_logits: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

            probabilities = torch.sigmoid(logits)
            all_logits.append(logits.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    probabilities_np = np.concatenate(all_probabilities, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    results = compute_binary_metrics(targets_np, probabilities_np)
    results["loss"] = float(total_loss / max(total_items, 1))
    results["logits"] = logits_np
    results["probabilities"] = probabilities_np
    results["targets"] = targets_np
    return results


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if torch.isnan(loss):
            raise ValueError("Loss became NaN during training.")
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return float(total_loss / max(total_items, 1))


def select_overfit_subset(dataframe: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(dataframe) <= n:
        return dataframe.reset_index(drop=True).copy()

    for attempt in range(50):
        sample = dataframe.sample(n=n, random_state=seed + attempt).reset_index(drop=True)
        if all(sample[label].sum() > 0 for label in LABELS if dataframe[label].sum() > 0):
            return sample

    return dataframe.sample(n=n, random_state=seed).reset_index(drop=True)


def save_predictions_csv(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    logits: np.ndarray,
    output_path: Path,
    patient_column: str,
    image_id_column: str,
) -> None:
    ensure_dir(output_path.parent)
    rows = []
    for row_index, row in dataframe.reset_index(drop=True).iterrows():
        record: dict[str, Any] = {
            "image_id": row[image_id_column],
            "patient_id": row[patient_column],
            "image_path": row["resolved_path"],
        }
        for label_index, label in enumerate(LABELS):
            record[f"true_{label}"] = int(row[label])
            record[f"logit_{label}"] = float(logits[row_index, label_index])
            record[f"prob_{label}"] = float(probabilities[row_index, label_index])
        rows.append(record)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def checkpoint_payload(
    model: nn.Module,
    epoch: int,
    val_loss: float,
    pos_weight: torch.Tensor,
    model_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "val_loss": float(val_loss),
        "label_names": LABELS,
        "pos_weight": [float(x) for x in pos_weight.cpu().tolist()],
        "model_info": model_info,
        "model_state_dict": model.state_dict(),
    }


def make_manifest_markdown(report: dict[str, Any]) -> str:
    lines = ["# NIH 2k Manifest Check", ""]
    for split_name in ["train", "val", "test"]:
        split = report["splits"][split_name]
        lines.append(f"## {split_name}")
        lines.append(f"- images: {split['num_images']}")
        lines.append(f"- patients: {split['num_patients']}")
        lines.append(f"- image path column: `{split['path_column']}`")
        lines.append(f"- patient column: `{split['patient_column']}`")
        lines.append(f"- image id column: `{split['image_id_column']}`")
        lines.append("- label counts:")
        for label in LABELS:
            lines.append(f"  - {label}: {split['label_counts'][label]}")
        lines.append("")

    lines.append("## Overlap")
    for pair_name, overlap in report["overlap"].items():
        lines.append(f"- {pair_name}: patient_overlap={overlap['patient_overlap']}, image_overlap={overlap['image_overlap']}")
    lines.append("")
    return "\n".join(lines)


def save_manifest_reports(report: dict[str, Any], artifact_paths: dict[str, Path]) -> None:
    report_json = artifact_paths["manifest_json"]
    report_md = artifact_paths["manifest_md"]
    save_json(report, report_json)
    ensure_dir(report_md.parent)
    report_md.write_text(make_manifest_markdown(report), encoding="utf-8")


def make_train_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Mini-Stage B Train Report",
        "",
        f"- mode: `{report['mode']}`",
        f"- device: `{report['device']}`",
        f"- pretrained used: `{report['model_info']['pretrained_used']}`",
        "",
        "## Split Sizes",
        f"- train images: {report['manifest_check']['splits']['train']['num_images']}",
        f"- val images: {report['manifest_check']['splits']['val']['num_images']}",
        f"- test images: {report['manifest_check']['splits']['test']['num_images']}",
        "",
        "## Pos Weight",
    ]

    for label, value in zip(LABELS, report["pos_weight"]):
        lines.append(f"- {label}: {value:.4f}")

    lines.extend(["", "## Epoch History"])
    for epoch_row in report["history"]:
        val_macro = epoch_row["val_macro_auroc"]
        val_macro_text = "undefined" if val_macro is None else f"{val_macro:.4f}"
        lines.append(
            f"- epoch {epoch_row['epoch']}: train_loss={epoch_row['train_loss']:.4f}, "
            f"val_loss={epoch_row['val_loss']:.4f}, val_macro_auroc={val_macro_text}"
        )

    for split_name in ["val_metrics", "test_metrics"]:
        metrics = report[split_name]
        lines.extend(["", f"## {split_name.replace('_', ' ').title()}"])
        macro_auroc = "undefined" if metrics["macro_auroc"] is None else f"{metrics['macro_auroc']:.4f}"
        macro_auprc = "undefined" if metrics["macro_auprc"] is None else f"{metrics['macro_auprc']:.4f}"
        lines.append(f"- loss: {metrics['loss']:.4f}")
        lines.append(f"- macro AUROC: {macro_auroc}")
        lines.append(f"- macro AUPRC: {macro_auprc}")
        lines.append("- per-label metrics:")
        for label in LABELS:
            item = metrics["per_label"][label]
            auroc = "undefined" if item["auroc"] is None else f"{item['auroc']:.4f}"
            auprc = "undefined" if item["auprc"] is None else f"{item['auprc']:.4f}"
            lines.append(f"  - {label}: AUROC={auroc}, AUPRC={auprc}, positives={item['positives']}")

    lines.append("")
    return "\n".join(lines)


def sanitize_metrics_for_report(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": float(metrics["loss"]),
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "defined_auroc_labels": int(metrics["defined_auroc_labels"]),
        "defined_auprc_labels": int(metrics["defined_auprc_labels"]),
        "per_label": metrics["per_label"],
    }


def run_check_mode(
    train_df: pd.DataFrame,
    args: argparse.Namespace,
    manifest_report: dict[str, Any],
    out_dir: Path,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    dataloader = build_dataloader(
        dataframe=train_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    images, labels = next(iter(dataloader))
    batch_png = artifact_paths["batch_png"]
    save_batch_visualization(images, batch_png)

    result = {
        "mode": "check",
        "image_tensor_shape": list(images.shape),
        "label_tensor_shape": list(labels.shape),
        "batch_visualization": str(batch_png),
        "manifest_check": manifest_report,
        "pos_weight": [float(x) for x in compute_pos_weight(train_df).tolist()],
    }

    print(f"image tensor shape: {tuple(images.shape)}")
    print(f"label tensor shape: {tuple(labels.shape)}")
    return result


def run_overfit_mode(
    train_df: pd.DataFrame,
    args: argparse.Namespace,
    manifest_report: dict[str, Any],
    out_dir: Path,
    device: torch.device,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    subset_df = select_overfit_subset(train_df, n=32, seed=args.seed)
    pos_weight = compute_pos_weight(subset_df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dataloader = build_dataloader(
        dataframe=subset_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )

    model, model_info = build_model(prefer_pretrained=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses: list[float] = []
    iterator = iter(dataloader)
    max_steps = 100

    model.train()
    for step in range(max_steps):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            images, labels = next(iterator)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if torch.isnan(loss):
            raise ValueError("Loss became NaN during overfit32 mode.")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    success = final_loss < initial_loss * 0.8 or min_loss < initial_loss * 0.75

    plot_path = artifact_paths["overfit_png"]
    save_line_plot(losses, plot_path, "Overfit32 Loss Curve", "step", "loss")

    result = {
        "mode": "overfit32",
        "device": str(device),
        "subset_size": int(len(subset_df)),
        "steps": max_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "min_loss": min_loss,
        "success": success,
        "loss_curve_png": str(plot_path),
        "losses": losses,
        "pos_weight": [float(x) for x in pos_weight.cpu().tolist()],
        "model_info": model_info,
        "manifest_check": manifest_report,
    }

    report_path = artifact_paths["overfit_json"]
    save_json(result, report_path)

    if not success:
        raise RuntimeError(
            "overfit32 did not show a clear loss decrease. "
            f"initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}, min_loss={min_loss:.4f}"
        )

    return result


def run_train_mode(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    manifest_report: dict[str, Any],
    out_dir: Path,
    device: torch.device,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    train_loader = build_dataloader(train_df, args.image_size, args.batch_size, True, args.seed)
    val_loader = build_dataloader(val_df, args.image_size, args.batch_size, False, args.seed)
    test_loader = build_dataloader(test_df, args.image_size, args.batch_size, False, args.seed)

    pos_weight = compute_pos_weight(train_df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model, model_info = build_model(prefer_pretrained=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoints_dir = out_dir / "checkpoints"
    outputs_dir = out_dir / "outputs"
    reports_dir = out_dir / "reports"
    ensure_dir(checkpoints_dir)
    ensure_dir(outputs_dir)
    ensure_dir(reports_dir)

    best_checkpoint = artifact_paths["best_checkpoint"]
    last_checkpoint = artifact_paths["last_checkpoint"]

    history: list[dict[str, Any]] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = math.inf
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_eval = evaluate_model(model, val_loader, criterion, device)
        val_loss = float(val_eval["loss"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_auroc": val_eval["macro_auroc"],
                "val_macro_auprc": val_eval["macro_auprc"],
            }
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(checkpoint_payload(model, epoch, val_loss, pos_weight, model_info), best_checkpoint)

    torch.save(checkpoint_payload(model, args.epochs, val_losses[-1], pos_weight, model_info), last_checkpoint)

    best_state = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])

    val_metrics_raw = evaluate_model(model, val_loader, criterion, device)
    test_metrics_raw = evaluate_model(model, test_loader, criterion, device)

    val_output_csv = artifact_paths["val_predictions"]
    test_output_csv = artifact_paths["test_predictions"]
    save_predictions_csv(
        val_df,
        probabilities=val_metrics_raw["probabilities"],
        logits=val_metrics_raw["logits"],
        output_path=val_output_csv,
        patient_column=manifest_report["splits"]["val"]["patient_column"],
        image_id_column=manifest_report["splits"]["val"]["image_id_column"],
    )
    save_predictions_csv(
        test_df,
        probabilities=test_metrics_raw["probabilities"],
        logits=test_metrics_raw["logits"],
        output_path=test_output_csv,
        patient_column=manifest_report["splits"]["test"]["patient_column"],
        image_id_column=manifest_report["splits"]["test"]["image_id_column"],
    )

    loss_curve_png = artifact_paths["loss_curve_png"]
    save_line_plot(
        train_losses,
        path=loss_curve_png,
        title="Mini-Stage B Train/Val Loss",
        xlabel="epoch",
        ylabel="loss",
        second_values=val_losses,
        second_label="val",
    )

    val_metrics = sanitize_metrics_for_report(val_metrics_raw)
    test_metrics = sanitize_metrics_for_report(test_metrics_raw)

    report = {
        "mode": "train",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "image_size": args.image_size,
        "seed": args.seed,
        "model_info": model_info,
        "pos_weight": [float(x) for x in pos_weight.cpu().tolist()],
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "history": history,
        "manifest_check": manifest_report,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "artifacts": {
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "val_predictions_csv": str(val_output_csv),
            "test_predictions_csv": str(test_output_csv),
            "loss_curve_png": str(loss_curve_png),
        },
    }

    train_json = artifact_paths["train_json"]
    train_md = artifact_paths["train_md"]
    save_json(report, train_json)
    train_md.write_text(make_train_markdown(report), encoding="utf-8")

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
        except FileNotFoundError as exc:
            print(str(exc))
            sys.exit(1)
        print("DRY_RUN_OK")
        return

    set_seed(args.seed)

    out_dir = args._namespace_config.output_root
    ensure_dir(out_dir)
    ensure_dir(args._namespace_config.reports_dir)
    ensure_dir(args._namespace_config.outputs_dir)
    ensure_dir(args._namespace_config.checkpoints_dir)

    print_runtime_configuration(args)
    dataframes, manifest_report = validate_all_manifests(args.train_csv, args.val_csv, args.test_csv)
    save_manifest_reports(manifest_report, args._artifact_paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = dataframes["train"]
    val_df = dataframes["val"]
    test_df = dataframes["test"]

    if args.mode == "check":
        result = run_check_mode(train_df, args, manifest_report, out_dir, args._artifact_paths)
        check_json = args._artifact_paths["check_json"]
        check_md = args._artifact_paths["check_md"]
        save_json(result, check_json)
        check_md.write_text(
            textwrap.dedent(
                f"""\
                # Mini-Stage B Check Report

                - image tensor shape: `{tuple(result['image_tensor_shape'])}`
                - label tensor shape: `{tuple(result['label_tensor_shape'])}`
                - batch visualization: `{result['batch_visualization']}`
                """
            ),
            encoding="utf-8",
        )
        return

    if args.mode == "overfit32":
        run_overfit_mode(train_df, args, manifest_report, out_dir, device, args._artifact_paths)
        return

    run_train_mode(train_df, val_df, test_df, args, manifest_report, out_dir, device, args._artifact_paths)


if __name__ == "__main__":
    main()
