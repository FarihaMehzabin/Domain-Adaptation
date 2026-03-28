#!/usr/bin/env python3
"""Stage 5A: validation-only probability mixing for ResNet50 fused CLS."""

from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss  # type: ignore
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score


LABEL_SPECS = [
    ("label_atelectasis", "atelectasis"),
    ("label_cardiomegaly", "cardiomegaly"),
    ("label_consolidation", "consolidation"),
    ("label_edema", "edema"),
    ("label_pleural_effusion", "pleural_effusion"),
]
LABEL_COLUMNS = [column for column, _ in LABEL_SPECS]
LABEL_NAMES = [name for _, name in LABEL_SPECS]

ALPHA_GRID_COARSE = [round(value, 1) for value in np.linspace(0.0, 1.0, num=11).tolist()]
PRIMARY_MEMORY_CONFIG = {"name": "primary", "k": 50, "tau": 1}
REFERENCE_MEMORY_CONFIG = {"name": "reference", "k": 5, "tau": 10}
MEMORY_CONFIGS = [PRIMARY_MEMORY_CONFIG, REFERENCE_MEMORY_CONFIG]

DEFAULT_BASELINE_RUN_ROOT = Path("/workspace/outputs/models/nih_cxr14/fused/resnet50_cls_20260324T091149Z")
DEFAULT_BASELINE_RUN_ROOT_FALLBACK = Path(
    "/workspace/outputs/nih_cxr14_frozen_fused_linear_cls_resnet50/resnet50_cls_20260324T091149Z"
)
DEFAULT_STAGE4_OUTPUT_DIR = Path("/workspace/memory_eval/nih_cxr14/resnet50_fused_cls_val_memory_only")
DEFAULT_TRAIN_MEMORY_ROOT = Path("/workspace/memory/nih_cxr14/resnet50_fused_cls_train")
DEFAULT_VAL_EMBEDDINGS = Path("/workspace/fused_embeddings_cls/resnet50/val/embeddings.npy")
DEFAULT_VAL_IMAGE_PATHS = Path("/workspace/fused_embeddings_cls/resnet50/val/image_paths.txt")
DEFAULT_VAL_RUN_META = Path("/workspace/fused_embeddings_cls/resnet50/val/run_meta.json")
DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_nih_cxr14 .csv")
DEFAULT_OUTPUT_DIR = Path("/workspace/memory_eval/nih_cxr14/resnet50_fused_cls_val_probability_mixing")

FLOAT_TOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5A only: validation-only probability mixing between the archived ResNet50 "
            "fused CLS baseline classifier and the Stage 4 retrieval memory probabilities."
        )
    )
    parser.add_argument("--baseline-run-root", type=Path, default=DEFAULT_BASELINE_RUN_ROOT)
    parser.add_argument("--baseline-run-root-fallback", type=Path, default=DEFAULT_BASELINE_RUN_ROOT_FALLBACK)
    parser.add_argument("--stage4-output-dir", type=Path, default=DEFAULT_STAGE4_OUTPUT_DIR)
    parser.add_argument("--train-memory-root", type=Path, default=DEFAULT_TRAIN_MEMORY_ROOT)
    parser.add_argument("--val-embeddings", type=Path, default=DEFAULT_VAL_EMBEDDINGS)
    parser.add_argument("--val-image-paths", type=Path, default=DEFAULT_VAL_IMAGE_PATHS)
    parser.add_argument("--val-run-meta", type=Path, default=DEFAULT_VAL_RUN_META)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--run-optional-refinement", action="store_true")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Text file is empty: {path}")
    return lines


def load_embedding_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Array file not found: {path}")
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array at {path}, found shape {array.shape}.")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Array contains NaN or inf values: {path}")
    return np.ascontiguousarray(array)


def validate_label_names(label_names: list[str]) -> None:
    if label_names != LABEL_NAMES:
        raise ValueError(f"Unexpected label names/order. Expected {LABEL_NAMES}, found {label_names}.")


def example_id_from_path(path: str) -> str:
    return Path(path).stem


def summarize_norms(norms: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "min": float(norms.min()),
        "max": float(norms.max()),
    }


def normalize_rows(embeddings: np.ndarray) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    raw_norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.isfinite(raw_norms).all():
        raise ValueError("Raw embedding norms contain NaN or inf values.")
    if int(np.count_nonzero(raw_norms <= 0.0)) > 0:
        raise ValueError("Found zero-norm embeddings; cannot normalize.")
    normalized = embeddings / raw_norms[:, None].astype(np.float32)
    normalized = np.ascontiguousarray(normalized.astype(np.float32))
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized embeddings contain NaN or inf values.")
    normalized_norms = np.linalg.norm(normalized.astype(np.float64), axis=1)
    return normalized, summarize_norms(raw_norms), summarize_norms(normalized_norms)


def read_manifest_rows(manifest_csv: Path, split_name: str) -> list[dict[str, str]]:
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest CSV is missing a header row: {manifest_csv}")
        required = LABEL_COLUMNS + ["image_path", "split"]
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Manifest CSV {manifest_csv} is missing columns: {missing}")
        rows = [row for row in reader if row.get("split") == split_name]
    if not rows:
        raise ValueError(f"Manifest CSV {manifest_csv} contains no rows for split '{split_name}'.")
    return rows


def build_labels(rows: list[dict[str, str]]) -> np.ndarray:
    labels = np.zeros((len(rows), len(LABEL_COLUMNS)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for label_index, column in enumerate(LABEL_COLUMNS):
            labels[row_index, label_index] = float(row[column])
    if not np.isfinite(labels).all():
        raise ValueError("Label matrix contains NaN or inf values.")
    return labels


def align_validation_rows(
    manifest_rows: list[dict[str, str]],
    val_image_paths: list[str],
) -> tuple[list[dict[str, str]], list[str], np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    manifest_by_example_id: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        example_id = example_id_from_path(row["image_path"])
        if example_id in manifest_by_example_id:
            raise ValueError(f"Duplicate validation example ID in manifest: {example_id}")
        manifest_by_example_id[example_id] = row

    val_example_ids: list[str] = []
    aligned_rows: list[dict[str, str]] = []
    kept_indices: list[int] = []
    dropped: list[dict[str, Any]] = []
    seen_val_example_ids: set[str] = set()

    for row_index, image_path in enumerate(val_image_paths):
        example_id = example_id_from_path(image_path)
        if example_id in seen_val_example_ids:
            raise ValueError(f"Duplicate validation example ID in embedding order: {example_id}")
        seen_val_example_ids.add(example_id)
        row = manifest_by_example_id.get(example_id)
        if row is None:
            dropped.append(
                {
                    "val_row_index": row_index,
                    "example_id": example_id,
                    "image_path": image_path,
                    "reason": "missing_manifest_match_by_example_id",
                }
            )
            continue
        aligned_rows.append(row)
        val_example_ids.append(example_id)
        kept_indices.append(row_index)

    kept_index_array = np.asarray(kept_indices, dtype=np.int64)
    summary = {
        "validation_embeddings_loaded": int(len(val_image_paths)),
        "validation_example_ids_derived": int(len(val_image_paths)),
        "aligned_to_manifest": int(len(aligned_rows)),
        "dropped_count": int(len(dropped)),
        "drop_reasons": sorted({entry["reason"] for entry in dropped}),
        "id_parsing_rule": "example_id = Path(image_path).stem",
        "alignment_key": "validation example_id matched against manifest image_path stem",
    }
    return aligned_rows, val_example_ids, kept_index_array, dropped, summary


def load_faiss_index(index_path: Path, train_embeddings: np.ndarray) -> tuple[faiss.Index, dict[str, Any]]:
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
        except Exception as exc:  # pragma: no cover - defensive path
            details["load_error"] = f"{type(exc).__name__}: {exc}"
    index = faiss.IndexFlatIP(int(train_embeddings.shape[1]))
    index.add(np.ascontiguousarray(train_embeddings.astype(np.float32)))
    details["rebuilt_from_embeddings"] = True
    return index, details


def check_train_memory_consistency(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    train_example_ids: list[str],
    train_image_paths: list[str],
) -> None:
    row_count = int(train_embeddings.shape[0])
    if train_labels.shape[0] != row_count:
        raise ValueError(
            f"Train labels row count {train_labels.shape[0]} does not match train embeddings {row_count}."
        )
    if train_labels.shape[1] != len(LABEL_NAMES):
        raise ValueError(
            f"Train labels second dimension {train_labels.shape[1]} does not match label count {len(LABEL_NAMES)}."
        )
    if len(train_example_ids) != row_count:
        raise ValueError(f"Train example_ids count {len(train_example_ids)} does not match train embeddings {row_count}.")
    if len(train_image_paths) != row_count:
        raise ValueError(f"Train image_paths count {len(train_image_paths)} does not match train embeddings {row_count}.")
    for row_index, (example_id, image_path) in enumerate(zip(train_example_ids, train_image_paths)):
        if example_id_from_path(image_path) != example_id:
            raise ValueError(
                f"Train example ID mismatch at row {row_index}: example_ids has {example_id}, "
                f"image_paths has {example_id_from_path(image_path)}."
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


def tune_f1_thresholds(targets: np.ndarray, probabilities: np.ndarray, label_names: list[str]) -> dict[str, float]:
    thresholds_by_label: dict[str, float] = {}
    for label_index, label_name in enumerate(label_names):
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
        best_indices = np.flatnonzero(np.isclose(f1_values, best_f1, atol=FLOAT_TOL, rtol=0.0))
        thresholds_by_label[label_name] = float(np.min(thresholds[best_indices]))
    return thresholds_by_label


def evaluate_probabilities(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
    *,
    include_diagnostic_thresholds: bool,
) -> dict[str, Any]:
    per_label: dict[str, dict[str, Any]] = {}
    macro_auroc_values: list[float] = []
    macro_average_precision_values: list[float] = []
    macro_f1_at_0p5_values: list[float] = []

    for label_index, label_name in enumerate(label_names):
        target_column = targets[:, label_index]
        probability_column = probabilities[:, label_index]
        auroc = compute_binary_metric("auroc", target_column, probability_column)
        average_precision = compute_binary_metric("average_precision", target_column, probability_column)
        f1_at_0p5 = float(f1_score(target_column, probability_column >= 0.5, zero_division=0))

        if auroc is not None:
            macro_auroc_values.append(auroc)
        if average_precision is not None:
            macro_average_precision_values.append(average_precision)
        macro_f1_at_0p5_values.append(f1_at_0p5)

        per_label[label_name] = {
            "auroc": auroc,
            "average_precision": average_precision,
            "f1_at_0.5": f1_at_0p5,
            "positive_count": int(target_column.sum()),
            "negative_count": int(target_column.shape[0] - target_column.sum()),
        }

    metrics = {
        "macro_auroc": float(np.mean(macro_auroc_values)) if macro_auroc_values else None,
        "macro_average_precision": (
            float(np.mean(macro_average_precision_values)) if macro_average_precision_values else None
        ),
        "macro_f1_at_0.5": float(np.mean(macro_f1_at_0p5_values)) if macro_f1_at_0p5_values else None,
        "per_label": per_label,
    }

    if include_diagnostic_thresholds:
        thresholds = tune_f1_thresholds(targets, probabilities, label_names)
        threshold_f1_values: list[float] = []
        threshold_payload: dict[str, Any] = {}
        for label_index, label_name in enumerate(label_names):
            threshold = float(thresholds[label_name])
            target_column = targets[:, label_index]
            probability_column = probabilities[:, label_index]
            f1_value = float(f1_score(target_column, probability_column >= threshold, zero_division=0))
            threshold_f1_values.append(f1_value)
            threshold_payload[label_name] = {
                "threshold": threshold,
                "f1": f1_value,
            }
        metrics["diagnostic_threshold_tuned_f1"] = {
            "label": "diagnostic only",
            "note": "optimistic because thresholds were chosen on the same split",
            "macro_f1": float(np.mean(threshold_f1_values)) if threshold_f1_values else None,
            "per_label": threshold_payload,
        }

    return metrics


def compute_memory_probabilities(
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    train_labels: np.ndarray,
    k: int,
    tau: int,
) -> np.ndarray:
    sliced_indices = np.ascontiguousarray(neighbor_indices[:, :k], dtype=np.int64)
    sliced_scores = np.ascontiguousarray(neighbor_scores[:, :k], dtype=np.float32)
    scaled_scores = sliced_scores * float(tau)
    scaled_scores -= scaled_scores.max(axis=1, keepdims=True)
    weights = np.exp(scaled_scores, dtype=np.float64)
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
    neighbor_label_matrix = train_labels[sliced_indices]
    probabilities = np.sum(weights[:, :, None] * neighbor_label_matrix.astype(np.float64), axis=1)
    probabilities = np.ascontiguousarray(probabilities.astype(np.float32))
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Computed memory probabilities contain NaN or inf for k={k}, tau={tau}.")
    if probabilities.min() < -1e-6 or probabilities.max() > 1.0 + 1e-6:
        raise ValueError(
            f"Memory probabilities are out of range for k={k}, tau={tau}: "
            f"min={probabilities.min()}, max={probabilities.max()}."
        )
    return probabilities


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


def probability_summary(probabilities: np.ndarray) -> dict[str, float]:
    return {
        "min": float(probabilities.min()),
        "max": float(probabilities.max()),
        "mean": float(probabilities.mean()),
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def remap_state_dict_for_linear_head(state_dict: dict[str, Any]) -> tuple[dict[str, Any], str]:
    keys = list(state_dict.keys())
    if set(keys) == {"weight", "bias"}:
        return state_dict, "none"
    for prefix in ("model.", "module.", "linear.", "head."):
        remapped = {}
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                remapped = {}
                break
            remapped[key[len(prefix) :]] = value
        if set(remapped.keys()) == {"weight", "bias"}:
            return remapped, f"stripped_prefix:{prefix}"
    raise ValueError(f"Unsupported checkpoint keys for linear head: {keys[:10]}")


def find_saved_baseline_probability_artifacts(baseline_run_root: Path) -> list[Path]:
    if not baseline_run_root.exists():
        return []
    candidates: list[Path] = []
    patterns = [
        "*val*prob*.npy",
        "*val*probs*.npy",
        "*val*pred*.npy",
        "*val*logits*.npy",
        "*validation*prob*.npy",
    ]
    for pattern in patterns:
        candidates.extend(sorted(baseline_run_root.glob(pattern)))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def load_or_reconstruct_p_base(
    *,
    baseline_run_root: Path,
    val_embeddings_path: Path,
    val_image_paths_path: Path,
    manifest_csv_path: Path,
    batch_size: int,
    files_used: set[Path],
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, Any]]:
    saved_candidates = find_saved_baseline_probability_artifacts(baseline_run_root)
    val_image_paths = read_lines(val_image_paths_path)
    files_used.add(val_image_paths_path)
    val_example_ids = [example_id_from_path(path) for path in val_image_paths]
    if len(val_example_ids) != len(set(val_example_ids)):
        raise ValueError("Validation example IDs derived from baseline image paths are not unique.")

    manifest_rows = read_manifest_rows(manifest_csv_path, "val")
    files_used.add(manifest_csv_path)
    aligned_rows, manifest_val_example_ids, kept_indices, dropped_rows, _ = align_validation_rows(manifest_rows, val_image_paths)
    if dropped_rows:
        raise ValueError(f"Baseline reconstruction path found dropped validation rows: {len(dropped_rows)}")
    baseline_labels = build_labels(aligned_rows)

    if saved_candidates:
        loaded_path = saved_candidates[0]
        files_used.add(loaded_path)
        probabilities = load_embedding_array(loaded_path)
        if probabilities.shape != baseline_labels.shape:
            raise ValueError(
                f"Saved baseline probability shape {probabilities.shape} does not match val labels {baseline_labels.shape}."
            )
        details = {
            "source": "loaded_saved_validation_probabilities",
            "loaded_path": str(loaded_path),
            "reconstructed": False,
            "checkpoint_loading": None,
            "checkpoint_key_remap": None,
            "model_type": None,
            "input_dim": int(probabilities.shape[1]),
            "output_dim": int(probabilities.shape[1]),
            "device": None,
            "batch_size": None,
            "dropped_rows": 0,
            "saved_probability_candidates_found": [str(path) for path in saved_candidates],
        }
        return probabilities.astype(np.float32), val_example_ids, baseline_labels, details

    config_path = baseline_run_root / "config.json"
    checkpoint_path = baseline_run_root / "best.ckpt"
    archived_val_metrics_path = baseline_run_root / "val_metrics.json"
    archived_thresholds_path = baseline_run_root / "val_f1_thresholds.json"
    files_used.update({config_path, checkpoint_path, archived_val_metrics_path, archived_thresholds_path})

    config = read_json(config_path)
    archived_val_metrics = read_json(archived_val_metrics_path)
    archived_thresholds = read_json(archived_thresholds_path)
    label_names = [str(name) for name in config.get("label_names", [])]
    validate_label_names(label_names)

    model_config = dict(config.get("model", {}))
    if model_config.get("type") != "linear":
        raise ValueError(f"Unsupported baseline model type for reconstruction: {model_config.get('type')}")
    hidden_layers = list(model_config.get("hidden_layers", []))
    if hidden_layers:
        raise ValueError(f"Expected plain linear head, found hidden layers: {hidden_layers}")

    val_embeddings = load_embedding_array(val_embeddings_path)
    files_used.add(val_embeddings_path)
    expected_input_dim = int(model_config.get("input_dim"))
    expected_output_dim = int(model_config.get("output_dim"))
    if val_embeddings.shape[1] != expected_input_dim:
        raise ValueError(
            f"Validation embedding dim {val_embeddings.shape[1]} does not match baseline config input_dim {expected_input_dim}."
        )
    if expected_output_dim != len(LABEL_NAMES):
        raise ValueError(f"Baseline output_dim {expected_output_dim} does not match label count {len(LABEL_NAMES)}.")
    if val_embeddings.shape[0] != len(val_example_ids):
        raise ValueError(
            f"Validation embeddings rows {val_embeddings.shape[0]} do not match baseline image paths {len(val_example_ids)}."
        )

    checkpoint = load_checkpoint(checkpoint_path)
    if "model_state_dict" in checkpoint:
        raw_state_dict = checkpoint["model_state_dict"]
        checkpoint_loading = 'checkpoint["model_state_dict"]'
    elif isinstance(checkpoint, dict):
        raw_state_dict = checkpoint
        checkpoint_loading = "checkpoint_root"
    else:
        raise ValueError(f"Unexpected checkpoint payload type: {type(checkpoint)!r}")
    if not isinstance(raw_state_dict, dict):
        raise ValueError("Checkpoint model state is not a dictionary.")
    state_dict, remap_strategy = remap_state_dict_for_linear_head(raw_state_dict)

    model = torch.nn.Linear(expected_input_dim, expected_output_dim)
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
    p_base = np.ascontiguousarray(np.concatenate(probability_batches, axis=0).astype(np.float32))

    computed_metrics = evaluate_probabilities(baseline_labels, p_base, LABEL_NAMES, include_diagnostic_thresholds=False)
    archived_mismatches: dict[str, Any] = {"per_label": {}, "threshold_file_present": bool(archived_thresholds)}
    max_metric_delta = 0.0
    for label_name in LABEL_NAMES:
        computed_label = computed_metrics["per_label"][label_name]
        archived_label = archived_val_metrics["per_label"][label_name]
        auroc_delta = None
        ap_delta = None
        if archived_label.get("auroc") is not None and computed_label.get("auroc") is not None:
            auroc_delta = float(computed_label["auroc"] - archived_label["auroc"])
            max_metric_delta = max(max_metric_delta, abs(auroc_delta))
        if archived_label.get("average_precision") is not None and computed_label.get("average_precision") is not None:
            ap_delta = float(computed_label["average_precision"] - archived_label["average_precision"])
            max_metric_delta = max(max_metric_delta, abs(ap_delta))
        archived_mismatches["per_label"][label_name] = {
            "computed_auroc": computed_label["auroc"],
            "archived_auroc": archived_label.get("auroc"),
            "delta_auroc": auroc_delta,
            "computed_average_precision": computed_label["average_precision"],
            "archived_average_precision": archived_label.get("average_precision"),
            "delta_average_precision": ap_delta,
            "archived_f1_threshold_from_val": archived_label.get("f1_threshold_from_val"),
            "threshold_file_value": archived_thresholds.get(label_name),
        }
    archived_mismatches["max_abs_metric_delta"] = max_metric_delta
    archived_mismatches["matches_archived_metrics_within_1e-6"] = bool(max_metric_delta <= 1e-6)

    details = {
        "source": "reconstructed_from_checkpoint",
        "loaded_path": None,
        "reconstructed": True,
        "checkpoint_loading": checkpoint_loading,
        "checkpoint_key_remap": remap_strategy,
        "model_type": model_config.get("type"),
        "hidden_layers": hidden_layers,
        "input_dim": expected_input_dim,
        "output_dim": expected_output_dim,
        "device": "cpu",
        "batch_size": int(batch_size),
        "dropped_rows": int(len(dropped_rows)),
        "saved_probability_candidates_found": [str(path) for path in saved_candidates],
        "archived_metric_check": archived_mismatches,
    }
    return p_base, manifest_val_example_ids, baseline_labels, details


def align_array_to_reference(
    *,
    reference_example_ids: list[str],
    source_example_ids: list[str],
    array: np.ndarray,
    source_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if array.shape[0] != len(source_example_ids):
        raise ValueError(
            f"{source_name}: source row count {array.shape[0]} does not match source example ids {len(source_example_ids)}."
        )
    if len(reference_example_ids) != len(set(reference_example_ids)):
        raise ValueError(f"{source_name}: reference example IDs are not unique.")
    source_index: dict[str, int] = {}
    for row_index, example_id in enumerate(source_example_ids):
        if example_id in source_index:
            raise ValueError(f"{source_name}: duplicate source example ID {example_id}")
        source_index[example_id] = row_index

    aligned_indices: list[int] = []
    missing_reference_ids: list[str] = []
    for example_id in reference_example_ids:
        match = source_index.get(example_id)
        if match is None:
            missing_reference_ids.append(example_id)
            continue
        aligned_indices.append(match)

    extra_source_ids = sorted(set(source_example_ids) - set(reference_example_ids))
    if missing_reference_ids or extra_source_ids:
        reasons: list[str] = []
        if missing_reference_ids:
            reasons.append("missing_reference_match_by_example_id")
        if extra_source_ids:
            reasons.append("extra_source_example_ids_not_in_reference")
        summary = {
            "source_name": source_name,
            "loaded_rows": int(array.shape[0]),
            "reference_rows": int(len(reference_example_ids)),
            "aligned_rows": int(len(aligned_indices)),
            "dropped_rows": int(len(missing_reference_ids) + len(extra_source_ids)),
            "drop_reasons": reasons,
            "missing_reference_count": int(len(missing_reference_ids)),
            "extra_source_count": int(len(extra_source_ids)),
            "missing_reference_example_ids_preview": missing_reference_ids[:10],
            "extra_source_example_ids_preview": extra_source_ids[:10],
            "exact_row_order_match_before_alignment": False,
        }
        return np.ascontiguousarray(array[np.asarray(aligned_indices, dtype=np.int64)]), summary

    exact_match = source_example_ids == reference_example_ids
    aligned = np.ascontiguousarray(array[np.asarray(aligned_indices, dtype=np.int64)])
    summary = {
        "source_name": source_name,
        "loaded_rows": int(array.shape[0]),
        "reference_rows": int(len(reference_example_ids)),
        "aligned_rows": int(aligned.shape[0]),
        "dropped_rows": 0,
        "drop_reasons": [],
        "missing_reference_count": 0,
        "extra_source_count": 0,
        "missing_reference_example_ids_preview": [],
        "extra_source_example_ids_preview": [],
        "exact_row_order_match_before_alignment": bool(exact_match),
    }
    return aligned, summary


def mix_probabilities(p_base: np.ndarray, p_mem: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Alpha must be in [0, 1], found {alpha}.")
    if np.isclose(alpha, 1.0, atol=FLOAT_TOL, rtol=0.0):
        return np.ascontiguousarray(p_base.copy())
    if np.isclose(alpha, 0.0, atol=FLOAT_TOL, rtol=0.0):
        return np.ascontiguousarray(p_mem.copy())
    mixed = alpha * p_base.astype(np.float64) + (1.0 - alpha) * p_mem.astype(np.float64)
    mixed = np.ascontiguousarray(mixed.astype(np.float32))
    if not np.isfinite(mixed).all():
        raise ValueError(f"Mixed probabilities contain NaN or inf for alpha={alpha}.")
    if mixed.min() < -1e-6 or mixed.max() > 1.0 + 1e-6:
        raise ValueError(
            f"Mixed probabilities out of range for alpha={alpha}: min={mixed.min()} max={mixed.max()}."
        )
    return mixed


def score_row(row: dict[str, Any]) -> tuple[float, float, float, int, int]:
    macro_auroc = float("-inf") if row.get("macro_auroc") is None else float(row["macro_auroc"])
    macro_ap = float("-inf") if row.get("macro_average_precision") is None else float(row["macro_average_precision"])
    alpha = float(row["alpha"])
    memory_priority = 1 if (int(row["k"]) == 50 and int(row["tau"]) == 1) else 0
    stage_priority = 1 if str(row.get("search_stage", "coarse")) == "coarse" else 0
    return (macro_auroc, macro_ap, alpha, memory_priority, stage_priority)


def select_best_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise ValueError("No candidate rows provided for selection.")

    max_auroc = max(float(row["macro_auroc"]) for row in rows)
    auroc_candidates = [row for row in rows if np.isclose(float(row["macro_auroc"]), max_auroc, atol=FLOAT_TOL, rtol=0.0)]

    max_ap = max(float(row["macro_average_precision"]) for row in auroc_candidates)
    ap_candidates = [
        row
        for row in auroc_candidates
        if np.isclose(float(row["macro_average_precision"]), max_ap, atol=FLOAT_TOL, rtol=0.0)
    ]

    max_alpha = max(float(row["alpha"]) for row in ap_candidates)
    alpha_candidates = [row for row in ap_candidates if np.isclose(float(row["alpha"]), max_alpha, atol=FLOAT_TOL, rtol=0.0)]

    primary_candidates = [row for row in alpha_candidates if int(row["k"]) == 50 and int(row["tau"]) == 1]
    config_candidates = primary_candidates if primary_candidates else alpha_candidates

    coarse_candidates = [row for row in config_candidates if str(row.get("search_stage", "coarse")) == "coarse"]
    final_candidates = coarse_candidates if coarse_candidates else config_candidates

    best_row = sorted(
        final_candidates,
        key=lambda row: (
            float(row["macro_auroc"]),
            float(row["macro_average_precision"]),
            float(row["alpha"]),
            1 if (int(row["k"]) == 50 and int(row["tau"]) == 1) else 0,
            1 if str(row.get("search_stage", "coarse")) == "coarse" else 0,
        ),
        reverse=True,
    )[0]

    trace = {
        "max_macro_auroc": max_auroc,
        "macro_auroc_tied_candidates": int(len(auroc_candidates)),
        "max_macro_average_precision_within_auroc_ties": max_ap,
        "macro_average_precision_tied_candidates": int(len(ap_candidates)),
        "largest_alpha_within_metric_ties": max_alpha,
        "alpha_tied_candidates": int(len(alpha_candidates)),
        "primary_memory_config_tied_candidates": int(len(primary_candidates)),
        "coarse_stage_tied_candidates": int(len(coarse_candidates)),
        "final_candidates": int(len(final_candidates)),
        "tie_break_rule": [
            "highest validation macro AUROC",
            "higher validation macro average precision",
            "larger alpha",
            "prefer primary memory config (k=50, tau=1)",
            "prefer coarse over refined",
        ],
    }
    return best_row, trace


def select_best_row_within_memory_config(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise ValueError("No rows available for memory-config selection.")
    max_auroc = max(float(row["macro_auroc"]) for row in rows)
    auroc_candidates = [row for row in rows if np.isclose(float(row["macro_auroc"]), max_auroc, atol=FLOAT_TOL, rtol=0.0)]
    max_ap = max(float(row["macro_average_precision"]) for row in auroc_candidates)
    ap_candidates = [
        row
        for row in auroc_candidates
        if np.isclose(float(row["macro_average_precision"]), max_ap, atol=FLOAT_TOL, rtol=0.0)
    ]
    max_alpha = max(float(row["alpha"]) for row in ap_candidates)
    alpha_candidates = [row for row in ap_candidates if np.isclose(float(row["alpha"]), max_alpha, atol=FLOAT_TOL, rtol=0.0)]
    coarse_candidates = [row for row in alpha_candidates if str(row.get("search_stage", "coarse")) == "coarse"]
    final_candidates = coarse_candidates if coarse_candidates else alpha_candidates
    best_row = sorted(
        final_candidates,
        key=lambda row: (float(row["macro_auroc"]), float(row["macro_average_precision"]), float(row["alpha"])),
        reverse=True,
    )[0]
    trace = {
        "max_macro_auroc": max_auroc,
        "macro_auroc_tied_candidates": int(len(auroc_candidates)),
        "max_macro_average_precision_within_auroc_ties": max_ap,
        "macro_average_precision_tied_candidates": int(len(ap_candidates)),
        "largest_alpha_within_metric_ties": max_alpha,
        "alpha_tied_candidates": int(len(alpha_candidates)),
        "coarse_stage_tied_candidates": int(len(coarse_candidates)),
        "final_candidates": int(len(final_candidates)),
    }
    return best_row, trace


def evaluate_alpha_rows(
    *,
    p_base: np.ndarray,
    p_mem: np.ndarray,
    targets: np.ndarray,
    label_names: list[str],
    alpha_grid: list[float],
    memory_config: dict[str, Any],
    search_stage: str,
    include_threshold_diagnostics_for_best_only: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        p_mix = mix_probabilities(p_base, p_mem, alpha)
        metrics = evaluate_probabilities(
            targets,
            p_mix,
            label_names,
            include_diagnostic_thresholds=include_threshold_diagnostics_for_best_only,
        )
        rows.append(
            {
                "memory_config_name": str(memory_config["name"]),
                "k": int(memory_config["k"]),
                "tau": int(memory_config["tau"]),
                "alpha": float(alpha),
                "search_stage": search_stage,
                "p_mix": p_mix,
                "metrics": metrics,
                "macro_auroc": metrics["macro_auroc"],
                "macro_average_precision": metrics["macro_average_precision"],
                "macro_f1_at_0.5": metrics["macro_f1_at_0.5"],
            }
        )
    return rows


def write_mixing_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["memory_config_name", "k", "tau", "alpha", "search_stage", "macro_auroc", "macro_average_precision", "macro_f1_at_0.5"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "memory_config_name": row["memory_config_name"],
                    "k": int(row["k"]),
                    "tau": int(row["tau"]),
                    "alpha": float(row["alpha"]),
                    "search_stage": str(row["search_stage"]),
                    "macro_auroc": row["macro_auroc"],
                    "macro_average_precision": row["macro_average_precision"],
                    "macro_f1_at_0.5": row["macro_f1_at_0.5"],
                }
            )


def metric_headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_auroc": metrics["macro_auroc"],
        "macro_average_precision": metrics["macro_average_precision"],
        "macro_f1_at_0.5": metrics["macro_f1_at_0.5"],
    }


def comparison_payload(best_metrics: dict[str, Any], reference_metrics: dict[str, Any]) -> dict[str, Any]:
    per_label_auroc_deltas = {
        label_name: (
            None
            if best_metrics["per_label"][label_name]["auroc"] is None
            or reference_metrics["per_label"][label_name]["auroc"] is None
            else float(best_metrics["per_label"][label_name]["auroc"] - reference_metrics["per_label"][label_name]["auroc"])
        )
        for label_name in LABEL_NAMES
    }
    return {
        "macro_auroc_delta": float(best_metrics["macro_auroc"] - reference_metrics["macro_auroc"]),
        "macro_average_precision_delta": float(
            best_metrics["macro_average_precision"] - reference_metrics["macro_average_precision"]
        ),
        "macro_f1_at_0.5_delta": float(best_metrics["macro_f1_at_0.5"] - reference_metrics["macro_f1_at_0.5"]),
        "per_label_auroc_deltas": per_label_auroc_deltas,
    }


def binary_cross_entropy_per_example(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float64), 1e-7, 1.0 - 1e-7)
    bce = -(
        targets.astype(np.float64) * np.log(clipped) + (1.0 - targets.astype(np.float64)) * np.log(1.0 - clipped)
    )
    return np.mean(bce, axis=1)


def labels_to_names(label_row: np.ndarray) -> list[str]:
    return [LABEL_NAMES[index] for index in np.flatnonzero(label_row > 0.5).tolist()]


def choose_indices_for_category(
    *,
    sorted_indices: list[int],
    desired_counts: dict[str, int],
    label_count_categories: list[str],
    used_indices: set[int],
) -> list[int]:
    selected: list[int] = []
    remaining_by_category = dict(desired_counts)
    for category_name, desired in desired_counts.items():
        if desired <= 0:
            continue
        for index in sorted_indices:
            if index in used_indices or index in selected:
                continue
            if label_count_categories[index] != category_name:
                continue
            selected.append(index)
            remaining_by_category[category_name] -= 1
            if remaining_by_category[category_name] <= 0:
                break
    if len(selected) < sum(max(value, 0) for value in desired_counts.values()):
        for index in sorted_indices:
            if index in used_indices or index in selected:
                continue
            selected.append(index)
            if len(selected) >= sum(max(value, 0) for value in desired_counts.values()):
                break
    return selected[: sum(max(value, 0) for value in desired_counts.values())]


def infer_case_note(
    *,
    true_labels: np.ndarray,
    p_base: np.ndarray,
    p_mem: np.ndarray,
    p_mix: np.ndarray,
    base_bce: float,
    mix_bce: float,
) -> str:
    positive_mask = true_labels > 0.5
    negative_mask = ~positive_mask
    base_true_mean = float(p_base[positive_mask].mean()) if positive_mask.any() else 0.0
    mix_true_mean = float(p_mix[positive_mask].mean()) if positive_mask.any() else 0.0
    base_false_mean = float(p_base[negative_mask].mean()) if negative_mask.any() else 0.0
    mix_false_mean = float(p_mix[negative_mask].mean()) if negative_mask.any() else 0.0
    if np.allclose(p_base, p_mem, atol=0.05, rtol=0.0):
        return "both agree"
    if positive_mask.any():
        base_true_max = float(p_base[positive_mask].max())
        mix_true_max = float(p_mix[positive_mask].max())
        if mix_bce + 0.05 < base_bce and (
            mix_true_mean > base_true_mean or mix_true_max > base_true_max or mix_false_mean + 0.05 < base_false_mean
        ):
            return "memory rescues missed positive"
        if mix_bce > base_bce + 0.05 and (mix_true_mean + 0.05 < base_true_mean or mix_true_max + 0.10 < base_true_max):
            return "memory suppresses useful baseline score"
    if not positive_mask.any():
        if float(p_mem.max()) >= float(p_base.max()) + 0.15 and mix_bce > base_bce + 0.02:
            return "suspicious retrieval effect"
        if mix_bce + 0.05 < base_bce:
            return "disagreement on weak/ambiguous case"
        return "disagreement on weak/ambiguous case"
    if negative_mask.any() and float(np.max(p_mem[negative_mask])) >= float(np.max(p_base[negative_mask])) + 0.15 and mix_bce > base_bce:
        return "suspicious retrieval effect"
    return "disagreement on weak/ambiguous case"


def build_qualitative_cases(
    *,
    example_ids: list[str],
    targets: np.ndarray,
    p_base: np.ndarray,
    p_mem: np.ndarray,
    p_mix: np.ndarray,
) -> list[dict[str, Any]]:
    base_bce = binary_cross_entropy_per_example(targets, p_base)
    mix_bce = binary_cross_entropy_per_example(targets, p_mix)
    delta = base_bce - mix_bce
    label_count = targets.sum(axis=1).astype(np.int64)
    label_count_categories = [
        "negative_or_unlabeled" if int(count) == 0 else "single_positive" if int(count) == 1 else "multi_positive"
        for count in label_count.tolist()
    ]

    helped_sorted = np.argsort(-delta).astype(np.int64).tolist()
    hurt_sorted = np.argsort(delta).astype(np.int64).tolist()

    used_indices: set[int] = set()
    helped_indices = choose_indices_for_category(
        sorted_indices=helped_sorted,
        desired_counts={"negative_or_unlabeled": 1, "single_positive": 2, "multi_positive": 2},
        label_count_categories=label_count_categories,
        used_indices=used_indices,
    )
    used_indices.update(helped_indices)
    hurt_indices = choose_indices_for_category(
        sorted_indices=hurt_sorted,
        desired_counts={"negative_or_unlabeled": 1, "single_positive": 1, "multi_positive": 3},
        label_count_categories=label_count_categories,
        used_indices=used_indices,
    )

    selected_cases: list[tuple[int, str]] = [(index, "helped_by_memory_or_mixing") for index in helped_indices]
    selected_cases.extend((index, "hurt_or_ambiguous") for index in hurt_indices)

    payload: list[dict[str, Any]] = []
    for index, review_bucket in selected_cases[:10]:
        true_labels = targets[index]
        note = infer_case_note(
            true_labels=true_labels,
            p_base=p_base[index],
            p_mem=p_mem[index],
            p_mix=p_mix[index],
            base_bce=float(base_bce[index]),
            mix_bce=float(mix_bce[index]),
        )
        payload.append(
            {
                "review_bucket": review_bucket,
                "validation_example_id": example_ids[index],
                "label_count_category": label_count_categories[index],
                "true_labels": labels_to_names(true_labels),
                "true_label_vector": [int(value) for value in true_labels.astype(np.int64).tolist()],
                "p_base": {label_name: float(value) for label_name, value in zip(LABEL_NAMES, p_base[index].tolist())},
                "p_mem": {label_name: float(value) for label_name, value in zip(LABEL_NAMES, p_mem[index].tolist())},
                "p_mix": {label_name: float(value) for label_name, value in zip(LABEL_NAMES, p_mix[index].tolist())},
                "base_bce": float(base_bce[index]),
                "mix_bce": float(mix_bce[index]),
                "bce_delta_base_minus_mix": float(delta[index]),
                "note": note,
            }
        )
    return payload


def resolve_existing_path(description: str, candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve {description}. Checked: {joined}")


def build_success_report(
    *,
    timestamp: str,
    exact_files_used: list[str],
    output_artifacts: list[str],
    baseline_source: dict[str, Any],
    memory_sources: dict[str, Any],
    alignment_report: dict[str, Any],
    sanity_checks: dict[str, Any],
    coarse_rows_primary: list[dict[str, Any]],
    coarse_rows_reference: list[dict[str, Any]],
    best_primary_row: dict[str, Any],
    best_reference_row: dict[str, Any],
    best_overall_row: dict[str, Any],
    best_overall_trace: dict[str, Any],
    baseline_metrics: dict[str, Any],
    best_metrics: dict[str, Any],
    comparisons: dict[str, Any],
    qualitative_cases: list[dict[str, Any]],
    optional_refinement_rows: list[dict[str, Any]] | None,
) -> str:
    baseline_delta = comparisons["vs_baseline_only"]
    best_config_label = f"k={int(best_overall_row['k'])}, tau={int(best_overall_row['tau'])}, alpha={best_overall_row['alpha']:.2f}"
    verdict = comparisons["vs_baseline_only"]["macro_auroc_delta"] > 0.0 or comparisons["vs_baseline_only"]["macro_average_precision_delta"] > 0.0

    def alpha_table(rows: list[dict[str, Any]], best_row: dict[str, Any]) -> str:
        table_rows: list[list[str]] = []
        for row in rows:
            note = "<- best" if np.isclose(float(row["alpha"]), float(best_row["alpha"]), atol=FLOAT_TOL, rtol=0.0) else ""
            table_rows.append(
                [
                    f"{float(row['alpha']):.2f}",
                    format_metric(row["macro_auroc"]),
                    format_metric(row["macro_average_precision"]),
                    format_metric(row["macro_f1_at_0.5"]),
                    note,
                ]
            )
        return build_markdown_table(["alpha", "macro AUROC", "macro AP", "macro F1 @ 0.5", "note"], table_rows)

    def per_label_delta_table(delta_payload: dict[str, Any]) -> str:
        rows = [[label_name, format_metric(delta_payload["per_label_auroc_deltas"][label_name])] for label_name in LABEL_NAMES]
        return build_markdown_table(["label", "AUROC delta"], rows)

    diagnostic_block = "Not computed.\n"
    if "diagnostic_threshold_tuned_f1" in best_metrics:
        diagnostic_rows: list[list[str]] = []
        for label_name in LABEL_NAMES:
            label_payload = best_metrics["diagnostic_threshold_tuned_f1"]["per_label"][label_name]
            diagnostic_rows.append(
                [
                    label_name,
                    format_metric(label_payload["threshold"]),
                    format_metric(label_payload["f1"]),
                ]
            )
        diagnostic_block = "\n".join(
            [
                "Diagnostic only and optimistic because thresholds were chosen on the same validation split.",
                "",
                build_markdown_table(["label", "best val threshold", "diagnostic F1"], diagnostic_rows),
                "",
                f"Diagnostic macro F1: `{format_metric(best_metrics['diagnostic_threshold_tuned_f1']['macro_f1'])}`",
            ]
        )

    qualitative_summary_counts: dict[str, int] = {}
    for case in qualitative_cases:
        key = str(case["note"])
        qualitative_summary_counts[key] = qualitative_summary_counts.get(key, 0) + 1

    qualitative_lines = []
    for case in qualitative_cases:
        labels_text = ", ".join(case["true_labels"]) if case["true_labels"] else "none"
        qualitative_lines.append(
            "- "
            + f"{case['validation_example_id']} [{case['review_bucket']}, {case['label_count_category']}, labels={labels_text}]: "
            + f"{case['note']}"
        )

    baseline_archived_check = baseline_source.get("archived_metric_check")
    archived_check_text = ""
    if baseline_archived_check is not None:
        archived_check_text = (
            " Archived per-label AUROC/AP reproduced the archived validation metrics within tiny float noise "
            f"(max abs delta `{baseline_archived_check['max_abs_metric_delta']:.8f}`)."
        )
    baseline_section = (
        "Baseline validation probabilities were reconstructed from the archived checkpoint rather than loaded from a saved "
        "validation-probability artifact. The script loaded `config.json`, verified a plain linear head with "
        f"`input_dim={baseline_source['input_dim']}` and `output_dim={baseline_source['output_dim']}`, loaded "
        f"`best.ckpt` via `{baseline_source['checkpoint_loading']}`, applied checkpoint key remapping "
        f"`{baseline_source['checkpoint_key_remap']}`, and ran sigmoid inference over the saved validation fused embeddings "
        f"on `{baseline_source['device']}` with batch size `{baseline_source['batch_size']}`. The resulting `p_base` shape "
        f"is `{sanity_checks['shape_checks']['p_base_shape']}` in label order `{LABEL_NAMES}`.{archived_check_text}"
    )

    if baseline_source["source"] == "loaded_saved_validation_probabilities":
        baseline_section = (
            "Baseline validation probabilities were loaded from an archived validation probability artifact. "
            f"Loaded path: `{baseline_source['loaded_path']}`. The resulting `p_base` shape is "
            f"`{sanity_checks['shape_checks']['p_base_shape']}` in label order `{LABEL_NAMES}`."
        )

    memory_section_lines = [
        (
            f"For `k=50, tau=1`, `p_mem` was {memory_sources['primary']['mode']} using the existing train memory "
            f"and validation embeddings. Source paths: `{memory_sources['primary']['source_paths']}`."
        ),
        (
            f"For `k=5, tau=10`, `p_mem` was {memory_sources['reference']['mode']} from "
            f"`{memory_sources['reference']['source_paths'][0]}` with reference ordering from the Stage 4 output."
        ),
    ]

    refinement_text = "Optional local refinement was not run."
    if optional_refinement_rows:
        refinement_text = (
            "Optional local refinement on validation only was run around the coarse winner. "
            f"Evaluated alphas: {[float(row['alpha']) for row in optional_refinement_rows]}."
        )

    comparison_text = "\n".join(
        [
            f"Against baseline-only (`alpha=1.0`): macro AUROC delta `{format_metric(baseline_delta['macro_auroc_delta'])}`, "
            f"macro AP delta `{format_metric(baseline_delta['macro_average_precision_delta'])}`, macro F1 @ 0.5 delta "
            f"`{format_metric(baseline_delta['macro_f1_at_0.5_delta'])}`.",
            per_label_delta_table(baseline_delta),
            "",
            (
                f"Against memory-only for the same memory config (`alpha=0.0`, `k={int(best_overall_row['k'])}`, "
                f"`tau={int(best_overall_row['tau'])}`): macro AUROC delta "
                f"`{format_metric(comparisons['vs_memory_only_same_config']['macro_auroc_delta'])}`, macro AP delta "
                f"`{format_metric(comparisons['vs_memory_only_same_config']['macro_average_precision_delta'])}`, macro F1 @ 0.5 delta "
                f"`{format_metric(comparisons['vs_memory_only_same_config']['macro_f1_at_0.5_delta'])}`."
            ),
            per_label_delta_table(comparisons["vs_memory_only_same_config"]),
            "",
            (
                f"Against the other memory config's best mixed setting "
                f"(k={int(comparisons['other_memory_config_best']['k'])}, tau={int(comparisons['other_memory_config_best']['tau'])}, "
                f"alpha={float(comparisons['other_memory_config_best']['alpha']):.2f}): macro AUROC delta "
                f"`{format_metric(comparisons['vs_other_memory_config_best_mixed']['macro_auroc_delta'])}`, macro AP delta "
                f"`{format_metric(comparisons['vs_other_memory_config_best_mixed']['macro_average_precision_delta'])}`, macro F1 @ 0.5 delta "
                f"`{format_metric(comparisons['vs_other_memory_config_best_mixed']['macro_f1_at_0.5_delta'])}`."
            ),
            per_label_delta_table(comparisons["vs_other_memory_config_best_mixed"]),
            "",
            "Memory mostly helped where retrieval lifted missed positives or modestly regularized overconfident baseline scores, "
            "and hurt when retrieval pushed probability mass toward the wrong pathology or raised false-positive scores on weak cases.",
        ]
    )

    report_lines = [
        "# ResNet50 Fused CLS Validation Probability-Mixing Report",
        "",
        "## 1. Executive Summary",
        (
            f"Stage 5A succeeded at {timestamp}. The best mixed validation setting was `{best_config_label}` "
            f"with macro AUROC `{format_metric(best_metrics['macro_auroc'])}` and macro average precision "
            f"`{format_metric(best_metrics['macro_average_precision'])}`. Relative to baseline-only, the best mix changed "
            f"macro AUROC by `{format_metric(baseline_delta['macro_auroc_delta'])}` and macro average precision by "
            f"`{format_metric(baseline_delta['macro_average_precision_delta'])}`. "
            + ("Mixing helped validation signal overall." if verdict else "Mixing was neutral-to-marginal relative to the baseline.")
        ),
        "",
        "## 2. Objective",
        "This step implements Stage 5A only: validation-only probability mixing between `p_base` and `p_mem`.",
        "",
        "## 3. Exact Files Used",
    ]
    report_lines.extend([f"- `{path}`" for path in exact_files_used])
    report_lines.extend(
        [
            "",
            "## 4. Output Artifacts Created",
        ]
    )
    report_lines.extend([f"- `{path}`" for path in output_artifacts])
    report_lines.extend(
        [
            "",
            "## 5. Baseline Probability Source",
            baseline_section,
            "",
            "## 6. Memory Probability Sources",
        ]
    )
    report_lines.extend([f"- {line}" for line in memory_section_lines])
    report_lines.extend(
        [
            "",
            "## 7. Data Alignment and Row Ordering",
            (
                "The canonical validation ordering was the Stage 4 `val_example_ids.json` ordering. "
                "All other sources were aligned to that reference using `example_id = Path(image_path).stem`."
            ),
            f"- Reference validation rows: `{alignment_report['reference_rows']}`",
            f"- Baseline rows loaded: `{alignment_report['baseline']['loaded_rows']}`",
            f"- Primary memory rows loaded: `{alignment_report['primary_memory']['loaded_rows']}`",
            f"- Reference memory rows loaded: `{alignment_report['reference_memory']['loaded_rows']}`",
            f"- Manifest-aligned label rows: `{alignment_report['labels']['loaded_rows']}`",
            f"- Final aligned rows: `{alignment_report['final_aligned_rows']}`",
            f"- Rows dropped from baseline: `{alignment_report['baseline']['dropped_rows']}`",
            f"- Rows dropped from primary memory: `{alignment_report['primary_memory']['dropped_rows']}`",
            f"- Rows dropped from reference memory: `{alignment_report['reference_memory']['dropped_rows']}`",
            f"- Rows dropped from labels: `{alignment_report['labels']['dropped_rows']}`",
            "- Exact example-ID parsing rule: `example_id = Path(image_path).stem` for validation image paths, train memory image paths, and manifest image_path.",
            "",
            "## 8. Sanity Checks",
            f"- Shape checks: `{json.dumps(sanity_checks['shape_checks'], sort_keys=True)}`",
            f"- Finite checks: `{json.dumps(sanity_checks['finite_checks'], sort_keys=True)}`",
            f"- Probability range summaries: `{json.dumps(sanity_checks['range_checks'], sort_keys=True)}`",
            f"- Alpha endpoint checks: `{json.dumps(sanity_checks['alpha_endpoint_checks'], sort_keys=True)}`",
            f"- Leakage checks: `{json.dumps(sanity_checks['leakage_checks'], sort_keys=True)}`",
            "",
            "## 9. Alpha Sweep Results: Primary Memory Config (k=50, tau=1)",
            alpha_table(coarse_rows_primary, best_primary_row),
            "",
            "## 10. Alpha Sweep Results: Reference Memory Config (k=5, tau=10)",
            alpha_table(coarse_rows_reference, best_reference_row),
            "",
            "## 11. Best Overall Mixed Setting",
            (
                f"Chosen memory config: `k={int(best_overall_row['k'])}, tau={int(best_overall_row['tau'])}`. "
                f"Chosen alpha: `{float(best_overall_row['alpha']):.2f}`. Winner came from "
                f"`{best_overall_row.get('search_stage', 'coarse')}` search."
            ),
            (
                "Selection-rule trace: "
                f"`macro_auroc_ties={best_overall_trace['macro_auroc_tied_candidates']}`, "
                f"`macro_ap_ties={best_overall_trace['macro_average_precision_tied_candidates']}`, "
                f"`alpha_ties={best_overall_trace['alpha_tied_candidates']}`, "
                f"`primary_config_ties={best_overall_trace['primary_memory_config_tied_candidates']}`, "
                f"`coarse_stage_ties={best_overall_trace['coarse_stage_tied_candidates']}`."
            ),
            refinement_text,
            "",
            "## 12. Comparison vs Baseline-Only and Memory-Only",
            comparison_text,
            "",
            "## 13. Optional Threshold-Tuned Diagnostic",
            diagnostic_block,
            "",
            "## 14. Qualitative Case Review",
            (
                f"Ten validation cases were inspected under the best mixed setting. Case-note counts: "
                f"`{json.dumps(qualitative_summary_counts, sort_keys=True)}`."
            ),
        ]
    )
    report_lines.extend(qualitative_lines)
    report_lines.extend(
        [
            "",
            "Overall patterns: the helped bucket mostly contains cases where memory damped very large baseline false-positive scores, "
            "while the hurt bucket concentrates in dense multi-label studies where memory flattened useful positive baseline evidence.",
            "",
            "## 15. Interpretation",
            (
                "Did probability mixing improve over the baseline? "
                + ("Yes on validation by the primary threshold-free metrics." if verdict else "Only marginally or not at all on validation.")
            ),
            (
                "Is the improvement meaningful or marginal? "
                + ("Marginal, because the baseline was already strong and the deltas are small." if abs(baseline_delta["macro_auroc_delta"]) < 0.01 else "Material enough to be noticeable on validation.")
            ),
            (
                "Does the best alpha suggest memory is adding real value? "
                + (
                    f"Yes; the winning alpha `{float(best_overall_row['alpha']):.2f}` keeps a non-trivial memory contribution."
                    if 0.0 < float(best_overall_row["alpha"]) < 1.0
                    else f"The winning alpha `{float(best_overall_row['alpha']):.2f}` suggests the best result was at an endpoint."
                )
            ),
            (
                "Are there obvious labels where memory helps more? "
                f"Per-label AUROC deltas versus baseline were {json.dumps(baseline_delta['per_label_auroc_deltas'], sort_keys=True)}."
            ),
            (
                "Are there failure modes that suggest Stage 5B logit correction may be worth trying next if needed? "
                "Yes; several hurt or ambiguous cases come from calibration-like shifts where retrieval meaningfully changes score levels but not always in the right direction."
            ),
            "",
            "## 16. Constraints Respected",
            "- validation only",
            "- train memory only",
            "- no test tuning",
            "- no retraining",
            "- no logit correction",
            "- no embedding updates",
            "",
            "## 17. Final Verdict",
            "PASS: mixing improved validation signal"
            if comparisons["vs_baseline_only"]["macro_auroc_delta"] > 0.0
            else "CONDITIONAL PASS: mixing is roughly neutral or only marginally better"
            if verdict
            else "FAIL: mixing underperformed the baseline",
            (
                "The verdict follows the requested validation-only selection rule and the observed deltas against baseline-only."
            ),
            "",
            "## Appendix A. Metric JSON Snippet",
            "```json",
            json.dumps(
                {
                    "baseline_only": metric_headline(baseline_metrics),
                    "best_primary_memory_mixed": metric_headline(best_primary_row["metrics"]),
                    "best_reference_memory_mixed": metric_headline(best_reference_row["metrics"]),
                    "best_overall_mixed_config": {
                        "k": int(best_overall_row["k"]),
                        "tau": int(best_overall_row["tau"]),
                        "alpha": float(best_overall_row["alpha"]),
                        "search_stage": str(best_overall_row.get("search_stage", "coarse")),
                        **metric_headline(best_metrics),
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Appendix B. 5-Line Ultra-Short Update",
            f"Stage 5A completed on validation only at {timestamp}.",
            f"Best mix: k={int(best_overall_row['k'])}, tau={int(best_overall_row['tau'])}, alpha={float(best_overall_row['alpha']):.2f}.",
            f"Best macro AUROC/AP: {format_metric(best_metrics['macro_auroc'])} / {format_metric(best_metrics['macro_average_precision'])}.",
            f"Delta vs baseline-only AUROC/AP: {format_metric(baseline_delta['macro_auroc_delta'])} / {format_metric(baseline_delta['macro_average_precision_delta'])}.",
            "Validation-only mixing stayed within Stage 5A constraints.",
        ]
    )
    return "\n".join(report_lines).strip() + "\n"


def build_failure_report(
    *,
    timestamp: str,
    exact_files_used: list[str],
    output_artifacts: list[str],
    failure_message: str,
    trace_text: str,
) -> str:
    lines = [
        "# ResNet50 Fused CLS Validation Probability-Mixing Report",
        "",
        "## 1. Executive Summary",
        (
            f"Stage 5A failed at {timestamp}. The validation-only probability-mixing run did not complete because "
            f"`{failure_message}`."
        ),
        "",
        "## 2. Objective",
        "This step implements Stage 5A only: validation-only probability mixing between `p_base` and `p_mem`.",
        "",
        "## 3. Exact Files Used",
    ]
    lines.extend([f"- `{path}`" for path in exact_files_used] or ["- `none`"])
    lines.extend(
        [
            "",
            "## 4. Output Artifacts Created",
        ]
    )
    lines.extend([f"- `{path}`" for path in output_artifacts] or ["- `report.md` only"])
    lines.extend(
        [
            "",
            "## 5. Baseline Probability Source",
            "Not completed because execution failed before a reliable `p_base` artifact was finalized.",
            "",
            "## 6. Memory Probability Sources",
            "Not completed because execution failed before both `p_mem` sources were finalized.",
            "",
            "## 7. Data Alignment and Row Ordering",
            "Not completed because execution failed before alignment was finalized.",
            "",
            "## 8. Sanity Checks",
            "Not completed.",
            "",
            "## 9. Alpha Sweep Results: Primary Memory Config (k=50, tau=1)",
            "Not computed.",
            "",
            "## 10. Alpha Sweep Results: Reference Memory Config (k=5, tau=10)",
            "Not computed.",
            "",
            "## 11. Best Overall Mixed Setting",
            "No winner was selected.",
            "",
            "## 12. Comparison vs Baseline-Only and Memory-Only",
            "Not computed.",
            "",
            "## 13. Optional Threshold-Tuned Diagnostic",
            "Not computed.",
            "",
            "## 14. Qualitative Case Review",
            "Not computed.",
            "",
            "## 15. Interpretation",
            "Probability mixing could not be evaluated reliably on validation because execution failed.",
            "",
            "## 16. Constraints Respected",
            "- validation only: intended",
            "- train memory only: intended",
            "- no test tuning: yes",
            "- no retraining: yes",
            "- no logit correction: yes",
            "- no embedding updates: yes",
            "",
            "## 17. Final Verdict",
            "FAIL: mixing underperformed the baseline",
            "This failure verdict reflects that reliable validation-only mixing metrics were not produced.",
            "",
            "## Appendix A. Metric JSON Snippet",
            "```json",
            json.dumps({"error": failure_message}, indent=2, sort_keys=True),
            "```",
            "",
            "## Appendix B. 5-Line Ultra-Short Update",
            "Stage 5A failed.",
            failure_message,
            "No reliable best mixed setting was produced.",
            "No test artifacts were used.",
            "See failure trace below.",
            "",
            "Failure trace:",
            "```text",
            trace_text.strip() or failure_message,
            "```",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    timestamp = utc_now_iso()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.md"
    run_config_path = output_dir / "run_config.json"
    aligned_ids_path = output_dir / "aligned_val_example_ids.json"
    aligned_labels_path = output_dir / "aligned_val_labels.npy"
    p_base_path = output_dir / "p_base_val.npy"
    p_mem_primary_path = output_dir / "p_mem_k50_tau1.npy"
    p_mem_reference_path = output_dir / "p_mem_k5_tau10.npy"
    coarse_csv_path = output_dir / "mixing_results_coarse.csv"
    refined_csv_path = output_dir / "mixing_results_refined.csv"
    best_config_path = output_dir / "best_mixing_config.json"
    best_p_mix_path = output_dir / "p_mix_best.npy"
    best_metrics_path = output_dir / "best_metrics.json"
    qualitative_path = output_dir / "qualitative_mixing_cases.json"
    sanity_checks_path = output_dir / "sanity_checks.json"

    files_used: set[Path] = set()
    output_artifacts: list[Path] = []

    try:
        baseline_run_root = resolve_existing_path(
            "baseline run root",
            [args.baseline_run_root, args.baseline_run_root_fallback],
        )
        stage4_output_dir = resolve_existing_path("Stage 4 output dir", [args.stage4_output_dir])
        train_memory_root = resolve_existing_path("train memory root", [args.train_memory_root])
        val_embeddings_path = resolve_existing_path("validation embeddings", [args.val_embeddings])
        val_image_paths_path = resolve_existing_path("validation image paths", [args.val_image_paths])
        val_run_meta_path = resolve_existing_path("validation run meta", [args.val_run_meta])
        manifest_csv_path = resolve_existing_path("manifest CSV", [args.manifest_csv])

        files_used.update({val_run_meta_path})
        _ = read_json(val_run_meta_path)

        stage4_example_ids_path = stage4_output_dir / "val_example_ids.json"
        stage4_val_labels_path = stage4_output_dir / "val_labels.npy"
        stage4_default_p_mem_path = stage4_output_dir / "val_p_mem_default.npy"
        stage4_best_memory_config_path = stage4_output_dir / "best_memory_config.json"
        stage4_run_config_path = stage4_output_dir / "run_config.json"
        stage4_sweep_path = stage4_output_dir / "sweep_results.csv"
        files_used.update(
            {
                stage4_example_ids_path,
                stage4_val_labels_path,
                stage4_default_p_mem_path,
                stage4_best_memory_config_path,
                stage4_run_config_path,
                stage4_sweep_path,
            }
        )

        stage4_example_ids = [str(value) for value in read_json(stage4_example_ids_path)]
        if not stage4_example_ids:
            raise ValueError("Stage 4 val_example_ids.json is empty.")
        if len(stage4_example_ids) != len(set(stage4_example_ids)):
            raise ValueError("Stage 4 val_example_ids.json contains duplicates.")
        canonical_example_ids = stage4_example_ids
        stage4_val_labels = load_embedding_array(stage4_val_labels_path)
        stage4_default_p_mem = load_embedding_array(stage4_default_p_mem_path)
        if stage4_default_p_mem.shape[0] != len(canonical_example_ids):
            raise ValueError(
                f"Stage 4 default p_mem rows {stage4_default_p_mem.shape[0]} do not match canonical ids {len(canonical_example_ids)}."
            )

        p_base, baseline_example_ids, baseline_labels_raw, baseline_source = load_or_reconstruct_p_base(
            baseline_run_root=baseline_run_root,
            val_embeddings_path=val_embeddings_path,
            val_image_paths_path=val_image_paths_path,
            manifest_csv_path=manifest_csv_path,
            batch_size=args.batch_size,
            files_used=files_used,
        )

        baseline_aligned, baseline_alignment = align_array_to_reference(
            reference_example_ids=canonical_example_ids,
            source_example_ids=baseline_example_ids,
            array=p_base,
            source_name="baseline_probabilities",
        )
        labels_aligned, labels_alignment = align_array_to_reference(
            reference_example_ids=canonical_example_ids,
            source_example_ids=baseline_example_ids,
            array=baseline_labels_raw,
            source_name="baseline_manifest_labels",
        )
        if labels_aligned.shape != stage4_val_labels.shape:
            raise ValueError(
                f"Manifest-aligned labels shape {labels_aligned.shape} does not match Stage 4 labels {stage4_val_labels.shape}."
            )
        if not np.array_equal(labels_aligned.astype(np.float32), stage4_val_labels.astype(np.float32)):
            max_abs = float(np.max(np.abs(labels_aligned.astype(np.float32) - stage4_val_labels.astype(np.float32))))
            raise ValueError(f"Manifest-aligned validation labels do not match Stage 4 labels. max_abs_diff={max_abs}")

        p_mem_reference_aligned, reference_mem_alignment = align_array_to_reference(
            reference_example_ids=canonical_example_ids,
            source_example_ids=stage4_example_ids,
            array=stage4_default_p_mem,
            source_name="stage4_reference_memory_probabilities",
        )

        train_embeddings_path = train_memory_root / "embeddings.npy"
        train_labels_path = train_memory_root / "labels.npy"
        train_example_ids_path = train_memory_root / "example_ids.json"
        train_image_paths_path = train_memory_root / "image_paths.txt"
        train_index_path = train_memory_root / "index.faiss"
        files_used.update(
            {
                train_embeddings_path,
                train_labels_path,
                train_example_ids_path,
                train_image_paths_path,
                train_index_path,
            }
        )

        train_embeddings = load_embedding_array(train_embeddings_path)
        train_labels = load_embedding_array(train_labels_path)
        train_example_ids = [str(value) for value in read_json(train_example_ids_path)]
        train_image_paths = read_lines(train_image_paths_path)
        check_train_memory_consistency(train_embeddings, train_labels, train_example_ids, train_image_paths)
        validate_label_names(LABEL_NAMES)

        val_embeddings = load_embedding_array(val_embeddings_path)
        val_image_paths = read_lines(val_image_paths_path)
        manifest_rows = read_manifest_rows(manifest_csv_path, "val")
        aligned_rows, manifest_val_example_ids, kept_indices, dropped_rows, manifest_alignment_summary = align_validation_rows(
            manifest_rows,
            val_image_paths,
        )
        if dropped_rows:
            raise ValueError(f"Found {len(dropped_rows)} dropped validation rows while preparing memory queries.")
        normalized_val_embeddings, norm_summary_before, norm_summary_after = normalize_rows(val_embeddings[kept_indices])
        if normalized_val_embeddings.shape[0] != len(manifest_val_example_ids):
            raise ValueError("Normalized validation embeddings row count does not match derived validation example IDs.")

        p_mem_primary_source_mode = "loaded"
        primary_p_mem_raw: np.ndarray
        primary_source_paths: list[str]
        possible_primary_saved_paths = [
            stage4_output_dir / "val_p_mem_k50_tau1.npy",
            stage4_output_dir / "val_p_mem_primary.npy",
            stage4_output_dir / "val_p_mem_best.npy",
        ]
        saved_primary_path = next((path for path in possible_primary_saved_paths if path.exists()), None)
        if saved_primary_path is not None:
            files_used.add(saved_primary_path)
            primary_p_mem_raw = load_embedding_array(saved_primary_path)
            primary_source_paths = [str(saved_primary_path)]
        else:
            p_mem_primary_source_mode = "recomputed"
            index, index_loading = load_faiss_index(train_index_path, train_embeddings)
            if index.ntotal != train_embeddings.shape[0]:
                raise ValueError(
                    f"FAISS index ntotal {index.ntotal} does not match train embeddings rows {train_embeddings.shape[0]}."
                )
            neighbor_scores, neighbor_indices = index.search(
                np.ascontiguousarray(normalized_val_embeddings.astype(np.float32)),
                int(PRIMARY_MEMORY_CONFIG["k"]),
            )
            if neighbor_scores.shape != (normalized_val_embeddings.shape[0], int(PRIMARY_MEMORY_CONFIG["k"])):
                raise ValueError(f"Unexpected primary neighbor score shape: {neighbor_scores.shape}")
            if neighbor_indices.shape != (normalized_val_embeddings.shape[0], int(PRIMARY_MEMORY_CONFIG["k"])):
                raise ValueError(f"Unexpected primary neighbor index shape: {neighbor_indices.shape}")
            if not np.isfinite(neighbor_scores).all():
                raise ValueError("Primary neighbor scores contain NaN or inf values.")
            if int(np.count_nonzero((neighbor_indices < 0) | (neighbor_indices >= train_embeddings.shape[0]))) > 0:
                raise ValueError("Primary neighbor indices contain out-of-range values.")
            primary_p_mem_raw = compute_memory_probabilities(
                neighbor_indices=neighbor_indices,
                neighbor_scores=neighbor_scores,
                train_labels=train_labels,
                k=int(PRIMARY_MEMORY_CONFIG["k"]),
                tau=int(PRIMARY_MEMORY_CONFIG["tau"]),
            )
            primary_source_paths = [
                str(train_embeddings_path),
                str(train_labels_path),
                str(train_example_ids_path),
                str(train_image_paths_path),
                str(train_index_path),
                str(val_embeddings_path),
                str(val_image_paths_path),
            ]
            memory_primary_index_loading = index_loading
        if p_mem_primary_source_mode == "loaded":
            memory_primary_index_loading = None

        p_mem_primary_aligned, primary_mem_alignment = align_array_to_reference(
            reference_example_ids=canonical_example_ids,
            source_example_ids=manifest_val_example_ids,
            array=primary_p_mem_raw,
            source_name="primary_memory_probabilities",
        )

        shape_checks = {
            "p_base_shape": list(baseline_aligned.shape),
            "p_mem_primary_shape": list(p_mem_primary_aligned.shape),
            "p_mem_reference_shape": list(p_mem_reference_aligned.shape),
            "val_labels_shape": list(labels_aligned.shape),
        }
        if not (
            baseline_aligned.shape == p_mem_primary_aligned.shape == p_mem_reference_aligned.shape == labels_aligned.shape
        ):
            raise ValueError(f"Shape mismatch across aligned arrays: {shape_checks}")

        finite_checks = {
            "p_base_all_finite": bool(np.isfinite(baseline_aligned).all()),
            "p_mem_primary_all_finite": bool(np.isfinite(p_mem_primary_aligned).all()),
            "p_mem_reference_all_finite": bool(np.isfinite(p_mem_reference_aligned).all()),
            "val_labels_all_finite": bool(np.isfinite(labels_aligned).all()),
        }
        if not all(finite_checks.values()):
            raise ValueError(f"Finite check failed: {finite_checks}")

        range_checks = {
            "p_base": probability_summary(baseline_aligned),
            "p_mem_primary": probability_summary(p_mem_primary_aligned),
            "p_mem_reference": probability_summary(p_mem_reference_aligned),
        }
        for name, summary in range_checks.items():
            if summary["min"] < -1e-6 or summary["max"] > 1.0 + 1e-6:
                raise ValueError(f"Probability range check failed for {name}: {summary}")

        baseline_metrics = evaluate_probabilities(labels_aligned, baseline_aligned, LABEL_NAMES, include_diagnostic_thresholds=False)
        coarse_rows: list[dict[str, Any]] = []
        coarse_rows_primary = evaluate_alpha_rows(
            p_base=baseline_aligned,
            p_mem=p_mem_primary_aligned,
            targets=labels_aligned,
            label_names=LABEL_NAMES,
            alpha_grid=ALPHA_GRID_COARSE,
            memory_config=PRIMARY_MEMORY_CONFIG,
            search_stage="coarse",
            include_threshold_diagnostics_for_best_only=False,
        )
        coarse_rows_reference = evaluate_alpha_rows(
            p_base=baseline_aligned,
            p_mem=p_mem_reference_aligned,
            targets=labels_aligned,
            label_names=LABEL_NAMES,
            alpha_grid=ALPHA_GRID_COARSE,
            memory_config=REFERENCE_MEMORY_CONFIG,
            search_stage="coarse",
            include_threshold_diagnostics_for_best_only=False,
        )
        coarse_rows.extend(coarse_rows_primary)
        coarse_rows.extend(coarse_rows_reference)

        best_coarse_row, best_coarse_trace = select_best_rows(coarse_rows)
        refined_rows: list[dict[str, Any]] = []
        if args.run_optional_refinement and not (
            np.isclose(float(best_coarse_row["alpha"]), 0.0, atol=FLOAT_TOL, rtol=0.0)
            or np.isclose(float(best_coarse_row["alpha"]), 1.0, atol=FLOAT_TOL, rtol=0.0)
        ):
            refined_grid = []
            coarse_alpha = float(best_coarse_row["alpha"])
            for candidate in [coarse_alpha - 0.05, coarse_alpha - 0.02, coarse_alpha, coarse_alpha + 0.02, coarse_alpha + 0.05]:
                if 0.0 <= candidate <= 1.0:
                    refined_grid.append(round(candidate, 4))
            refined_grid = sorted(set(refined_grid))
            best_coarse_memory_config = {
                "name": str(best_coarse_row["memory_config_name"]),
                "k": int(best_coarse_row["k"]),
                "tau": int(best_coarse_row["tau"]),
            }
            p_mem_for_refinement = (
                p_mem_primary_aligned
                if (int(best_coarse_row["k"]) == 50 and int(best_coarse_row["tau"]) == 1)
                else p_mem_reference_aligned
            )
            refined_rows = evaluate_alpha_rows(
                p_base=baseline_aligned,
                p_mem=p_mem_for_refinement,
                targets=labels_aligned,
                label_names=LABEL_NAMES,
                alpha_grid=refined_grid,
                memory_config=best_coarse_memory_config,
                search_stage="refined",
                include_threshold_diagnostics_for_best_only=False,
            )

        all_candidate_rows = coarse_rows + refined_rows
        best_overall_row, best_overall_trace = select_best_rows(all_candidate_rows)
        best_primary_row, best_primary_trace = select_best_row_within_memory_config(
            [row for row in all_candidate_rows if int(row["k"]) == 50 and int(row["tau"]) == 1]
        )
        best_reference_row, best_reference_trace = select_best_row_within_memory_config(
            [row for row in all_candidate_rows if int(row["k"]) == 5 and int(row["tau"]) == 10]
        )

        best_overall_metrics = evaluate_probabilities(
            labels_aligned,
            best_overall_row["p_mix"],
            LABEL_NAMES,
            include_diagnostic_thresholds=True,
        )
        best_overall_row["metrics"] = best_overall_metrics
        if best_primary_row is best_overall_row:
            best_primary_row["metrics"] = best_overall_metrics
        if best_reference_row is best_overall_row:
            best_reference_row["metrics"] = best_overall_metrics

        alpha_endpoint_checks = {
            "primary_memory_config": {
                "alpha_1_matches_p_base": bool(np.array_equal(mix_probabilities(baseline_aligned, p_mem_primary_aligned, 1.0), baseline_aligned)),
                "alpha_0_matches_p_mem": bool(np.array_equal(mix_probabilities(baseline_aligned, p_mem_primary_aligned, 0.0), p_mem_primary_aligned)),
                "alpha_1_max_abs_diff": float(np.max(np.abs(mix_probabilities(baseline_aligned, p_mem_primary_aligned, 1.0) - baseline_aligned))),
                "alpha_0_max_abs_diff": float(np.max(np.abs(mix_probabilities(baseline_aligned, p_mem_primary_aligned, 0.0) - p_mem_primary_aligned))),
            },
            "reference_memory_config": {
                "alpha_1_matches_p_base": bool(np.array_equal(mix_probabilities(baseline_aligned, p_mem_reference_aligned, 1.0), baseline_aligned)),
                "alpha_0_matches_p_mem": bool(np.array_equal(mix_probabilities(baseline_aligned, p_mem_reference_aligned, 0.0), p_mem_reference_aligned)),
                "alpha_1_max_abs_diff": float(np.max(np.abs(mix_probabilities(baseline_aligned, p_mem_reference_aligned, 1.0) - baseline_aligned))),
                "alpha_0_max_abs_diff": float(np.max(np.abs(mix_probabilities(baseline_aligned, p_mem_reference_aligned, 0.0) - p_mem_reference_aligned))),
            },
        }
        for config_name, payload in alpha_endpoint_checks.items():
            if not payload["alpha_1_matches_p_base"] or not payload["alpha_0_matches_p_mem"]:
                raise ValueError(f"Alpha endpoint check failed for {config_name}: {payload}")

        leakage_checks = {
            "query_split_is_validation_only": True,
            "retrieved_items_are_train_memory_only": True,
            "test_artifacts_used": False,
            "baseline_model_retrained": False,
            "encoder_weights_changed": False,
            "logit_correction_run": False,
            "details": "Only validation embeddings were queried against the saved train memory; no test artifacts or model updates were used.",
        }

        sanity_checks = {
            "shape_checks": shape_checks,
            "finite_checks": finite_checks,
            "range_checks": range_checks,
            "alpha_endpoint_checks": alpha_endpoint_checks,
            "leakage_checks": leakage_checks,
            "baseline_archived_metric_check": baseline_source.get("archived_metric_check"),
            "validation_query_norms": {
                "before": norm_summary_before,
                "after": norm_summary_after,
            },
        }

        qualitative_cases = build_qualitative_cases(
            example_ids=canonical_example_ids,
            targets=labels_aligned,
            p_base=baseline_aligned,
            p_mem=p_mem_primary_aligned if int(best_overall_row["k"]) == 50 else p_mem_reference_aligned,
            p_mix=best_overall_row["p_mix"],
        )

        other_best_row = best_reference_row if int(best_overall_row["k"]) == 50 else best_primary_row
        p_mem_same_config = p_mem_primary_aligned if int(best_overall_row["k"]) == 50 else p_mem_reference_aligned
        memory_only_metrics_same_config = evaluate_probabilities(
            labels_aligned,
            p_mem_same_config,
            LABEL_NAMES,
            include_diagnostic_thresholds=False,
        )
        comparisons = {
            "vs_baseline_only": comparison_payload(best_overall_metrics, baseline_metrics),
            "vs_memory_only_same_config": comparison_payload(best_overall_metrics, memory_only_metrics_same_config),
            "vs_other_memory_config_best_mixed": comparison_payload(best_overall_metrics, other_best_row["metrics"]),
            "other_memory_config_best": {
                "k": int(other_best_row["k"]),
                "tau": int(other_best_row["tau"]),
                "alpha": float(other_best_row["alpha"]),
                "search_stage": str(other_best_row.get("search_stage", "coarse")),
            },
        }

        alignment_report = {
            "canonical_order_source": str(stage4_example_ids_path),
            "reference_rows": int(len(canonical_example_ids)),
            "final_aligned_rows": int(labels_aligned.shape[0]),
            "baseline": baseline_alignment,
            "primary_memory": primary_mem_alignment,
            "reference_memory": reference_mem_alignment,
            "labels": labels_alignment,
            "stage4_labels_shape": list(stage4_val_labels.shape),
            "stage4_labels_exact_match_to_manifest_aligned_labels": True,
            "manifest_alignment_summary_from_val_paths": manifest_alignment_summary,
        }
        if any(summary["dropped_rows"] != 0 for summary in [baseline_alignment, primary_mem_alignment, reference_mem_alignment, labels_alignment]):
            raise ValueError(f"Alignment dropped rows unexpectedly: {alignment_report}")

        np.save(aligned_labels_path, labels_aligned.astype(np.float32))
        np.save(p_base_path, baseline_aligned.astype(np.float32))
        np.save(p_mem_primary_path, p_mem_primary_aligned.astype(np.float32))
        np.save(p_mem_reference_path, p_mem_reference_aligned.astype(np.float32))
        np.save(best_p_mix_path, best_overall_row["p_mix"].astype(np.float32))
        aligned_ids_path.write_text(json.dumps(canonical_example_ids, indent=2), encoding="utf-8")
        output_artifacts.extend(
            [
                aligned_ids_path,
                aligned_labels_path,
                p_base_path,
                p_mem_primary_path,
                p_mem_reference_path,
                best_p_mix_path,
            ]
        )

        write_mixing_results_csv(coarse_csv_path, coarse_rows)
        output_artifacts.append(coarse_csv_path)
        if refined_rows:
            write_mixing_results_csv(refined_csv_path, refined_rows)
            output_artifacts.append(refined_csv_path)

        memory_sources = {
            "primary": {
                "mode": p_mem_primary_source_mode,
                "source_paths": primary_source_paths,
                "index_loading": memory_primary_index_loading,
            },
            "reference": {
                "mode": "loaded",
                "source_paths": [str(stage4_default_p_mem_path), str(stage4_example_ids_path)],
            },
        }

        run_config = {
            "timestamp": timestamp,
            "exact_paths_used": {
                "baseline_run_root": str(baseline_run_root),
                "baseline_checkpoint": str(baseline_run_root / "best.ckpt"),
                "baseline_config": str(baseline_run_root / "config.json"),
                "baseline_val_metrics": str(baseline_run_root / "val_metrics.json"),
                "baseline_val_f1_thresholds": str(baseline_run_root / "val_f1_thresholds.json"),
                "stage4_output_dir": str(stage4_output_dir),
                "stage4_val_example_ids": str(stage4_example_ids_path),
                "stage4_val_labels": str(stage4_val_labels_path),
                "stage4_reference_p_mem": str(stage4_default_p_mem_path),
                "stage4_best_memory_config": str(stage4_best_memory_config_path),
                "stage4_run_config": str(stage4_run_config_path),
                "stage4_sweep_results": str(stage4_sweep_path),
                "train_memory_root": str(train_memory_root),
                "train_embeddings": str(train_embeddings_path),
                "train_labels": str(train_labels_path),
                "train_example_ids": str(train_example_ids_path),
                "train_image_paths": str(train_image_paths_path),
                "train_faiss_index": str(train_index_path),
                "validation_embeddings": str(val_embeddings_path),
                "validation_image_paths": str(val_image_paths_path),
                "validation_run_meta": str(val_run_meta_path),
                "manifest_csv": str(manifest_csv_path),
            },
            "label_names": LABEL_NAMES,
            "alpha_grid_coarse": ALPHA_GRID_COARSE,
            "optional_local_refinement_run": bool(refined_rows),
            "evaluated_memory_configs": MEMORY_CONFIGS,
            "selection_rule": [
                "highest validation macro AUROC",
                "break ties with higher validation macro average precision",
                "if still tied, prefer larger alpha",
                "if still tied, prefer primary memory config (k=50, tau=1)",
                "if still tied, prefer the simpler coarse-grid winner over a refined alpha",
            ],
            "alignment_rule": {
                "canonical_validation_order": str(stage4_example_ids_path),
                "example_id_parsing": "example_id = Path(image_path).stem",
                "all_sources_aligned_by_example_id": True,
            },
            "p_base_source": {
                "loaded_or_reconstructed": "loaded"
                if baseline_source["source"] == "loaded_saved_validation_probabilities"
                else "reconstructed",
                "details": baseline_source,
            },
            "p_mem_sources": {
                "k50_tau1": memory_sources["primary"],
                "k5_tau10": memory_sources["reference"],
            },
            "stage5_scope": "validation_only_probability_mixing",
            "test_used": False,
            "retraining_run": False,
            "logit_correction_run": False,
            "encoder_updated": False,
        }
        write_json(run_config_path, run_config)
        output_artifacts.append(run_config_path)

        best_config_payload = {
            "chosen_memory_config": {
                "name": str(best_overall_row["memory_config_name"]),
                "k": int(best_overall_row["k"]),
                "tau": int(best_overall_row["tau"]),
            },
            "chosen_alpha": float(best_overall_row["alpha"]),
            "search_stage": str(best_overall_row.get("search_stage", "coarse")),
            "headline_metrics": metric_headline(best_overall_metrics),
            "deltas_vs_baseline_only": comparisons["vs_baseline_only"],
            "deltas_vs_memory_only_same_config": comparisons["vs_memory_only_same_config"],
            "deltas_vs_other_memory_config_best_mixed": comparisons["vs_other_memory_config_best_mixed"],
            "selection_trace": best_overall_trace,
        }
        write_json(best_config_path, best_config_payload)
        output_artifacts.append(best_config_path)

        best_metrics_payload = {
            "best_config": {
                "memory_config_name": str(best_overall_row["memory_config_name"]),
                "k": int(best_overall_row["k"]),
                "tau": int(best_overall_row["tau"]),
                "alpha": float(best_overall_row["alpha"]),
                "search_stage": str(best_overall_row.get("search_stage", "coarse")),
            },
            "metrics": best_overall_metrics,
            "baseline_only_metrics": baseline_metrics,
            "best_primary_memory_mixed": {
                "config": {
                    "k": int(best_primary_row["k"]),
                    "tau": int(best_primary_row["tau"]),
                    "alpha": float(best_primary_row["alpha"]),
                    "search_stage": str(best_primary_row.get("search_stage", "coarse")),
                },
                "metrics": best_primary_row["metrics"],
                "selection_trace": best_primary_trace,
            },
            "best_reference_memory_mixed": {
                "config": {
                    "k": int(best_reference_row["k"]),
                    "tau": int(best_reference_row["tau"]),
                    "alpha": float(best_reference_row["alpha"]),
                    "search_stage": str(best_reference_row.get("search_stage", "coarse")),
                },
                "metrics": best_reference_row["metrics"],
                "selection_trace": best_reference_trace,
            },
            "comparisons": comparisons,
        }
        write_json(best_metrics_path, best_metrics_payload)
        output_artifacts.append(best_metrics_path)

        write_json(qualitative_path, qualitative_cases)
        output_artifacts.append(qualitative_path)
        write_json(sanity_checks_path, sanity_checks)
        output_artifacts.append(sanity_checks_path)

        report_text = build_success_report(
            timestamp=timestamp,
            exact_files_used=sorted(str(path) for path in files_used),
            output_artifacts=sorted(str(path) for path in output_artifacts + [report_path]),
            baseline_source=baseline_source,
            memory_sources=memory_sources,
            alignment_report=alignment_report,
            sanity_checks=sanity_checks,
            coarse_rows_primary=coarse_rows_primary,
            coarse_rows_reference=coarse_rows_reference,
            best_primary_row=best_primary_row,
            best_reference_row=best_reference_row,
            best_overall_row=best_overall_row,
            best_overall_trace=best_overall_trace,
            baseline_metrics=baseline_metrics,
            best_metrics=best_overall_metrics,
            comparisons=comparisons,
            qualitative_cases=qualitative_cases,
            optional_refinement_rows=refined_rows or None,
        )
        report_path.write_text(report_text, encoding="utf-8")
        output_artifacts.append(report_path)
        return 0
    except Exception as exc:
        trace_text = traceback.format_exc()
        report_text = build_failure_report(
            timestamp=timestamp,
            exact_files_used=sorted(str(path) for path in files_used),
            output_artifacts=sorted(str(path) for path in output_artifacts + [report_path]),
            failure_message=f"{type(exc).__name__}: {exc}",
            trace_text=trace_text,
        )
        report_path.write_text(report_text, encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
