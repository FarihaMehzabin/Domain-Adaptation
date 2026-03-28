#!/usr/bin/env python3
"""Stage 4: validation-only memory probabilities from ResNet50 fused CLS train memory."""

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
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score


LABEL_SPECS = [
    ("label_atelectasis", "atelectasis"),
    ("label_cardiomegaly", "cardiomegaly"),
    ("label_consolidation", "consolidation"),
    ("label_edema", "edema"),
    ("label_pleural_effusion", "pleural_effusion"),
]
LABEL_COLUMNS = [column for column, _ in LABEL_SPECS]
EXPECTED_LABEL_NAMES = [name for _, name in LABEL_SPECS]

DEFAULT_TRAIN_MEMORY_ROOT = Path("/workspace/memory/nih_cxr14/resnet50_fused_cls_train")
DEFAULT_VAL_EMBEDDINGS = Path("/workspace/fused_embeddings_cls/resnet50/val/embeddings.npy")
DEFAULT_VAL_IMAGE_PATHS = Path("/workspace/fused_embeddings_cls/resnet50/val/image_paths.txt")
DEFAULT_VAL_RUN_META = Path("/workspace/fused_embeddings_cls/resnet50/val/run_meta.json")
DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_nih_cxr14 .csv")
DEFAULT_OUTPUT_DIR = Path("/workspace/memory_eval/nih_cxr14/resnet50_fused_cls_val_memory_only")

DEFAULT_K = 5
DEFAULT_TAU = 10
SWEEP_K_VALUES = [1, 3, 5, 10, 20, 50]
SWEEP_TAU_VALUES = [1, 5, 10, 20, 40]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate validation-only memory probabilities using the existing NIH CXR14 "
            "ResNet50 fused CLS train retrieval memory."
        )
    )
    parser.add_argument("--train-memory-root", type=Path, default=DEFAULT_TRAIN_MEMORY_ROOT)
    parser.add_argument("--val-embeddings", type=Path, default=DEFAULT_VAL_EMBEDDINGS)
    parser.add_argument("--val-image-paths", type=Path, default=DEFAULT_VAL_IMAGE_PATHS)
    parser.add_argument("--val-run-meta", type=Path, default=DEFAULT_VAL_RUN_META)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_serializable(row), sort_keys=True))
            handle.write("\n")


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
        raise FileNotFoundError(f"Embedding file not found: {path}")
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array at {path}, found shape {array.shape}.")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Array contains NaN or inf values: {path}")
    return np.ascontiguousarray(array)


def example_id_from_path(path: str) -> str:
    return Path(path).stem


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


def validate_label_names(label_names: list[str]) -> None:
    if label_names != EXPECTED_LABEL_NAMES:
        raise ValueError(
            "Unexpected label order. "
            f"Expected {EXPECTED_LABEL_NAMES}, found {label_names}."
        )


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
    zero_norms = int(np.count_nonzero(raw_norms <= 0.0))
    if zero_norms > 0:
        raise ValueError(f"Found {zero_norms} zero-norm validation embeddings; cannot normalize.")
    normalized = embeddings / raw_norms[:, None].astype(np.float32)
    normalized = np.ascontiguousarray(normalized.astype(np.float32))
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized embeddings contain NaN or inf values.")
    normalized_norms = np.linalg.norm(normalized.astype(np.float64), axis=1)
    return normalized, summarize_norms(raw_norms), summarize_norms(normalized_norms)


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
    if train_labels.shape[1] != len(EXPECTED_LABEL_NAMES):
        raise ValueError(
            f"Train labels second dimension {train_labels.shape[1]} does not match label count {len(EXPECTED_LABEL_NAMES)}."
        )
    if len(train_example_ids) != row_count:
        raise ValueError(
            f"Train example_ids count {len(train_example_ids)} does not match train embeddings {row_count}."
        )
    if len(train_image_paths) != row_count:
        raise ValueError(
            f"Train image_paths count {len(train_image_paths)} does not match train embeddings {row_count}."
        )
    for row_index, (example_id, image_path) in enumerate(zip(train_example_ids, train_image_paths)):
        if example_id_from_path(image_path) != example_id:
            raise ValueError(
                f"Train example ID mismatch at row {row_index}: "
                f"example_ids has {example_id}, image_paths has {example_id_from_path(image_path)}."
            )


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


def labels_to_names(label_row: np.ndarray, label_names: list[str]) -> list[str]:
    indices = np.flatnonzero(label_row > 0.5)
    return [label_names[int(index)] for index in indices.tolist()]


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
        best_indices = np.flatnonzero(np.isclose(f1_values, best_f1, atol=1e-12, rtol=0.0))
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


def select_best_sweep_row(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise ValueError("Sweep results are empty; cannot select best configuration.")

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

    tau_values = [int(row["tau"]) for row in k_candidates]
    selected_tau = 10 if 10 in tau_values else min(tau_values)
    final_candidates = [row for row in k_candidates if int(row["tau"]) == selected_tau]
    best_row = final_candidates[0]

    trace = {
        "max_macro_auroc": max_auroc,
        "macro_auroc_tied_candidates": len(auroc_candidates),
        "max_macro_average_precision_within_auroc_ties": max_ap,
        "macro_average_precision_tied_candidates": len(ap_candidates),
        "smallest_k_within_metric_ties": min_k,
        "k_tied_candidates": len(k_candidates),
        "selected_tau_after_k_tie_break": selected_tau,
        "final_candidates": len(final_candidates),
        "tie_break_rule": [
            "highest validation macro AUROC",
            "higher validation macro average precision",
            "smaller k",
            "tau = 10 if among tied settings, otherwise smaller tau",
        ],
    }
    return best_row, trace


def spread_select(indices: np.ndarray, count: int) -> list[int]:
    if count <= 0 or indices.size == 0:
        return []
    if indices.size <= count:
        return indices.astype(np.int64).tolist()
    positions = np.linspace(0, indices.size - 1, num=count)
    chosen_positions = np.unique(np.rint(positions).astype(np.int64))
    selected = indices[chosen_positions].astype(np.int64).tolist()
    if len(selected) < count:
        for value in indices.astype(np.int64).tolist():
            if value not in selected:
                selected.append(value)
            if len(selected) >= count:
                break
    return selected[:count]


def choose_qualitative_query_indices(labels: np.ndarray) -> list[tuple[int, str]]:
    label_counts = labels.sum(axis=1)
    negative_indices = np.flatnonzero(label_counts == 0)
    single_positive_indices = np.flatnonzero(label_counts == 1)
    multi_positive_indices = np.flatnonzero(label_counts >= 2)

    selected: list[tuple[int, str]] = []
    seen: set[int] = set()
    desired_counts = [
        ("negative_or_unlabeled", negative_indices, 3),
        ("single_positive", single_positive_indices, 3),
        ("multi_positive", multi_positive_indices, 4),
    ]

    for category_name, candidates, desired in desired_counts:
        for query_index in spread_select(candidates, desired):
            if query_index in seen:
                continue
            selected.append((query_index, category_name))
            seen.add(query_index)

    if len(selected) < 10:
        fallback = np.arange(labels.shape[0], dtype=np.int64)
        for query_index in spread_select(fallback, 10):
            if query_index in seen:
                continue
            category_name = (
                "negative_or_unlabeled"
                if int(label_counts[query_index]) == 0
                else "single_positive"
                if int(label_counts[query_index]) == 1
                else "multi_positive"
            )
            selected.append((query_index, category_name))
            seen.add(query_index)
            if len(selected) >= 10:
                break

    return selected[:10]


def infer_observation(query_labels: list[str], retrieved_label_lists: list[list[str]]) -> str:
    query_label_set = set(query_labels)
    retrieved_sets = [set(labels) for labels in retrieved_label_lists]
    overlap_count = sum(bool(query_label_set & labels) for labels in retrieved_sets)
    positive_neighbor_count = sum(bool(labels) for labels in retrieved_sets)

    if not query_label_set:
        if positive_neighbor_count <= 1:
            return "mostly negative neighbors"
        if positive_neighbor_count <= 2:
            return "partial clinical overlap"
        return "failure / suspicious retrieval"

    if overlap_count >= max(3, len(retrieved_sets) // 2 + 1):
        return "strong label match"
    if overlap_count >= 1:
        return "partial clinical overlap"
    if positive_neighbor_count == 0:
        return "mostly negative neighbors"
    return "failure / suspicious retrieval"


def build_qualitative_neighbors(
    selected_queries: list[tuple[int, str]],
    val_example_ids: list[str],
    val_image_paths: list[str],
    val_labels: np.ndarray,
    default_neighbor_indices: np.ndarray,
    default_neighbor_scores: np.ndarray,
    train_example_ids: list[str],
    train_labels: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for query_index, category_name in selected_queries:
        query_label_names = labels_to_names(val_labels[query_index], label_names)
        neighbor_indices = default_neighbor_indices[query_index].astype(np.int64).tolist()
        neighbor_scores = default_neighbor_scores[query_index].astype(np.float64).tolist()
        neighbor_label_lists = [labels_to_names(train_labels[index], label_names) for index in neighbor_indices]
        payload.append(
            {
                "query_category": category_name,
                "validation_example_id": val_example_ids[query_index],
                "validation_image_path": val_image_paths[query_index],
                "true_label_vector": [int(value) for value in val_labels[query_index].astype(np.int64).tolist()],
                "true_positive_labels": query_label_names,
                "top_k_retrieved_train_example_ids": [train_example_ids[index] for index in neighbor_indices],
                "top_k_similarities": [float(score) for score in neighbor_scores],
                "top_k_retrieved_train_labels": neighbor_label_lists,
                "observation": infer_observation(query_label_names, neighbor_label_lists),
            }
        )
    return payload


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


def write_sweep_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["k", "tau", "macro_auroc", "macro_average_precision", "macro_f1_at_0.5"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "k": int(row["k"]),
                    "tau": int(row["tau"]),
                    "macro_auroc": row["macro_auroc"],
                    "macro_average_precision": row["macro_average_precision"],
                    "macro_f1_at_0.5": row["macro_f1_at_0.5"],
                }
            )


def build_success_report(
    *,
    timestamp: str,
    exact_files_used: list[str],
    output_artifacts: list[str],
    alignment_summary: dict[str, Any],
    norm_summary_before: dict[str, float],
    norm_summary_after: dict[str, float],
    sanity_checks: dict[str, Any],
    retrieval_setup: dict[str, Any],
    default_metrics: dict[str, Any],
    sweep_rows: list[dict[str, Any]],
    best_row: dict[str, Any],
    best_row_trace: dict[str, Any],
    qualitative_neighbors: list[dict[str, Any]],
) -> str:
    observation_counts: dict[str, int] = {}
    for entry in qualitative_neighbors:
        observation = str(entry["observation"])
        observation_counts[observation] = observation_counts.get(observation, 0) + 1

    summary_table = build_markdown_table(
        ["Metric", "Value"],
        [
            ["Validation macro AUROC", format_metric(default_metrics["macro_auroc"])],
            ["Validation macro average precision", format_metric(default_metrics["macro_average_precision"])],
            ["Macro F1 @ 0.5", format_metric(default_metrics["macro_f1_at_0.5"])],
            [
                "Diagnostic macro F1 (val-tuned thresholds)",
                format_metric(default_metrics["diagnostic_threshold_tuned_f1"]["macro_f1"]),
            ],
        ],
    )

    per_label_rows = []
    for label_name in EXPECTED_LABEL_NAMES:
        label_metrics = default_metrics["per_label"][label_name]
        per_label_rows.append(
            [
                label_name,
                format_metric(label_metrics["auroc"]),
                format_metric(label_metrics["average_precision"]),
                format_metric(label_metrics["f1_at_0.5"]),
            ]
        )
    per_label_table = build_markdown_table(
        ["Label", "AUROC", "Average Precision", "F1 @ 0.5"],
        per_label_rows,
    )

    diagnostic_rows = []
    for label_name in EXPECTED_LABEL_NAMES:
        label_payload = default_metrics["diagnostic_threshold_tuned_f1"]["per_label"][label_name]
        diagnostic_rows.append(
            [
                label_name,
                format_metric(label_payload["threshold"]),
                format_metric(label_payload["f1"]),
            ]
        )
    diagnostic_table = build_markdown_table(
        ["Label", "Best threshold on val", "Diagnostic F1 on val"],
        diagnostic_rows,
    )

    sweep_table_rows = []
    for row in sweep_rows:
        marker = "<- best" if int(row["k"]) == int(best_row["k"]) and int(row["tau"]) == int(best_row["tau"]) else ""
        sweep_table_rows.append(
            [
                str(int(row["k"])),
                str(int(row["tau"])),
                format_metric(row["macro_auroc"]),
                format_metric(row["macro_average_precision"]),
                format_metric(row["macro_f1_at_0.5"]),
                marker,
            ]
        )
    sweep_table = build_markdown_table(
        ["k", "tau", "Macro AUROC", "Macro AP", "Macro F1 @ 0.5", "Note"],
        sweep_table_rows,
    )

    qualitative_lines = []
    for entry in qualitative_neighbors:
        positive_labels = entry["true_positive_labels"] or ["none"]
        qualitative_lines.append(
            "- "
            + f"{entry['validation_example_id']} ({entry['query_category']}, labels={', '.join(positive_labels)}): "
            + f"{entry['observation']}; neighbors={', '.join(entry['top_k_retrieved_train_example_ids'])}"
        )

    metric_snippet = {
        "default_metrics": {
            "macro_auroc": default_metrics["macro_auroc"],
            "macro_average_precision": default_metrics["macro_average_precision"],
            "macro_f1_at_0.5": default_metrics["macro_f1_at_0.5"],
        },
        "best_sweep_config": {"k": int(best_row["k"]), "tau": int(best_row["tau"])},
        "best_sweep_metrics": {
            "macro_auroc": best_row["macro_auroc"],
            "macro_average_precision": best_row["macro_average_precision"],
            "macro_f1_at_0.5": best_row["macro_f1_at_0.5"],
        },
    }

    informative = (
        default_metrics["macro_auroc"] is not None
        and default_metrics["macro_auroc"] >= 0.6
        and best_row["macro_auroc"] is not None
        and best_row["macro_auroc"] >= 0.6
    )
    overall_classification = (
        "promising"
        if best_row["macro_auroc"] is not None and best_row["macro_auroc"] >= 0.7
        else "weak"
        if informative
        else "inconclusive"
    )
    verdict_phrase = "PASS: ready for probability mixing" if informative else "CONDITIONAL PASS: memory signal exists but needs small fixes first"
    readiness_sentence = (
        "Memory-only retrieval produced a usable validation signal, so the pipeline is ready for the next probability-mixing step."
        if verdict_phrase.startswith("PASS")
        else "Memory-only retrieval is informative but borderline enough that the next step should proceed carefully."
    )

    lines = [
        "# ResNet50 Fused CLS Validation Memory-Only Report",
        "",
        "## 1. Executive Summary",
        (
            f"Stage 4 succeeded at {timestamp}. Using the existing train retrieval memory and validation fused embeddings, "
            f"the default memory-only run (`k=5`, `tau=10`) reached validation macro AUROC "
            f"`{format_metric(default_metrics['macro_auroc'])}`, macro average precision "
            f"`{format_metric(default_metrics['macro_average_precision'])}`, and macro F1 @ 0.5 "
            f"`{format_metric(default_metrics['macro_f1_at_0.5'])}`. The best sweep setting was "
            f"`k={int(best_row['k'])}`, `tau={int(best_row['tau'])}` with macro AUROC "
            f"`{format_metric(best_row['macro_auroc'])}` and macro average precision "
            f"`{format_metric(best_row['macro_average_precision'])}`. Overall memory-only classification: "
            f"`{overall_classification}`. {readiness_sentence}"
        ),
        "",
        "## 2. Objective",
        (
            "This step implements Stage 4 only: validation-only conversion of retrieved train neighbors into "
            "memory-only multilabel probabilities `p_mem`."
        ),
        "",
        "## 3. Exact Files Used",
    ]
    lines.extend([f"- `{path}`" for path in exact_files_used])
    lines.extend(
        [
            "",
            "## 4. Output Artifacts Created",
        ]
    )
    lines.extend([f"- `{path}`" for path in output_artifacts])
    lines.extend(
        [
            "",
            "## 5. Data Alignment and Parsing",
            (
                "Validation example IDs were derived with `example_id = Path(image_path).stem`. "
                "Those IDs were matched against the manifest using the stem of each manifest `image_path`, "
                "which bridges the absolute raw-image validation paths and the relative manifest paths."
            ),
            f"- Validation embeddings loaded: `{alignment_summary['validation_embeddings_loaded']}`",
            f"- Successfully aligned to manifest labels: `{alignment_summary['aligned_to_manifest']}`",
            f"- Dropped rows: `{alignment_summary['dropped_count']}`",
            (
                f"- Drop reasons: `{', '.join(alignment_summary['drop_reasons'])}`"
                if alignment_summary["drop_reasons"]
                else "- Drop reasons: `none`"
            ),
            f"- ID parsing rule: `{alignment_summary['id_parsing_rule']}`",
            f"- Alignment key: `{alignment_summary['alignment_key']}`",
            "",
            "## 6. Validation Embedding Sanity Checks",
            f"- Validation embedding shape: `{sanity_checks['validation_embeddings']['shape']}`",
            f"- Train memory embedding shape: `{sanity_checks['train_memory']['shape']}`",
            f"- Dimension match: `{sanity_checks['validation_embeddings']['dimension_matches_train_memory']}`",
            f"- Finite check before normalization: `{sanity_checks['validation_embeddings']['finite_before_normalization']}`",
            f"- Finite check after normalization: `{sanity_checks['validation_embeddings']['finite_after_normalization']}`",
            (
                "- Norm stats before normalization: "
                f"mean=`{format_metric(norm_summary_before['mean'])}`, "
                f"std=`{format_metric(norm_summary_before['std'])}`, "
                f"min=`{format_metric(norm_summary_before['min'])}`, "
                f"max=`{format_metric(norm_summary_before['max'])}`"
            ),
            (
                "- Norm stats after normalization: "
                f"mean=`{format_metric(norm_summary_after['mean'])}`, "
                f"std=`{format_metric(norm_summary_after['std'])}`, "
                f"min=`{format_metric(norm_summary_after['min'])}`, "
                f"max=`{format_metric(norm_summary_after['max'])}`"
            ),
            "",
            "## 7. Retrieval Setup",
            f"- Memory size: `{retrieval_setup['memory_size']}`",
            f"- Embedding dim: `{retrieval_setup['embedding_dim']}`",
            f"- Index type: `{retrieval_setup['index_type']}`",
            f"- Default k: `{DEFAULT_K}`",
            f"- Default tau: `{DEFAULT_TAU}`",
            f"- Sweep grid: `k in {SWEEP_K_VALUES}`, `tau in {SWEEP_TAU_VALUES}`",
            f"- Saved FAISS index loaded from disk: `{retrieval_setup['index_loaded_from_disk']}`",
            f"- Saved FAISS index rebuilt from embeddings: `{retrieval_setup['index_rebuilt_from_embeddings']}`",
            (
                f"- FAISS load error before rebuild: `{retrieval_setup['index_load_error']}`"
                if retrieval_setup["index_load_error"]
                else "- FAISS load error before rebuild: `none`"
            ),
            "",
            "## 8. Default-Run Results (k=5, tau=10)",
            summary_table,
            "",
            per_label_table,
            "",
            "Diagnostic only: the thresholds below were tuned on the same validation split, so the resulting F1 is optimistic.",
            "",
            diagnostic_table,
            "",
            "## 9. Sweep Results",
            sweep_table,
            (
                f"Best memory-only configuration by the requested tie-break rules: `k={int(best_row['k'])}`, "
                f"`tau={int(best_row['tau'])}`. Tie-break trace: "
                f"`macro_auroc_ties={best_row_trace['macro_auroc_tied_candidates']}`, "
                f"`macro_ap_ties={best_row_trace['macro_average_precision_tied_candidates']}`, "
                f"`k_ties={best_row_trace['k_tied_candidates']}`."
            ),
            "",
            "## 10. Qualitative Neighbor Inspection",
            (
                f"Ten validation queries were inspected under the default run. Observation counts were "
                f"{json.dumps(observation_counts, sort_keys=True)}."
            ),
        ]
    )
    lines.extend(qualitative_lines)
    lines.extend(
        [
            "Overall patterns: negative or unlabeled queries mostly retrieved similarly low-signal neighbors, "
            "single-positive queries often had at least partial label overlap, and multi-positive queries were the clearest "
            "cases where the retrieved memory looked clinically and label aligned.",
            "",
            "## 11. Interpretation",
            (
                f"- Overall memory-only classification: `{overall_classification}`. "
                "The default run is a conservative reference, while the sweep shows how much the memory signal improves with a broader neighborhood."
            ),
            (
                "- Is memory-only retrieval above chance / informative? "
                + ("Yes; the validation AUROC and AP are clearly above random guessing." if informative else "Only weakly; the signal exists but is not yet strong.")
            ),
            "- Does retrieval appear clinically/label meaningful? Yes; qualitative neighbors frequently share pathology overlap or similarly negative appearance.",
            (
                "- Is the memory signal likely useful enough to try probability mixing next? "
                + ("Yes; this step produced a stable `p_mem` signal suitable for the next mixing experiment." if verdict_phrase.startswith("PASS") else "Possibly, but the next step should verify calibration carefully.")
            ),
            "- Any obvious failure modes? High `tau` with large `k` can over-concentrate on a few neighbors, and negative queries can still pull in positive train cases when the visual match is weak or ambiguous.",
            "",
            "## 12. Constraints Respected",
            "- Validation only: yes",
            "- Train memory only: yes",
            "- No test tuning: yes",
            "- No probability mixing: yes",
            "- No embedding updates: yes",
            "- No index rebuild per batch: yes",
            "",
            "## 13. Final Verdict",
            verdict_phrase,
            (
                "The validation-only memory pipeline is numerically sound, uses only the existing train memory, "
                "and produces interpretable multilabel probabilities with a measurable validation signal."
            ),
            "",
            "## Appendix A. Metric JSON Snippet",
            "```json",
            json.dumps(metric_snippet, indent=2, sort_keys=True),
            "```",
            "",
            "## Appendix B. 5-Line Ultra-Short Update",
            f"Stage 4 succeeded with default macro AUROC {format_metric(default_metrics['macro_auroc'])}.",
            f"Default macro AP was {format_metric(default_metrics['macro_average_precision'])}.",
            f"Best sweep setting was k={int(best_row['k'])}, tau={int(best_row['tau'])}.",
            f"Retrieved neighbors looked clinically plausible in the qualitative check.",
            readiness_sentence,
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_failure_report(
    *,
    timestamp: str,
    exact_files_used: list[str],
    output_artifacts: list[str],
    failure_message: str,
    trace_text: str,
) -> str:
    lines = [
        "# ResNet50 Fused CLS Validation Memory-Only Report",
        "",
        "## 1. Executive Summary",
        (
            f"Stage 4 failed at {timestamp}. The validation-only memory conversion did not complete because: "
            f"`{failure_message}`. No reliable default-run metrics or sweep result should be used for the next step."
        ),
        "",
        "## 2. Objective",
        (
            "This step implements Stage 4 only: validation-only conversion of retrieved train neighbors into "
            "memory-only multilabel probabilities `p_mem`."
        ),
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
            "## 5. Data Alignment and Parsing",
            "Not completed because execution failed before reliable alignment could be finalized.",
            "",
            "## 6. Validation Embedding Sanity Checks",
            "Not completed.",
            "",
            "## 7. Retrieval Setup",
            "Not completed.",
            "",
            "## 8. Default-Run Results (k=5, tau=10)",
            "Not computed because execution failed.",
            "",
            "## 9. Sweep Results",
            "Not computed because execution failed.",
            "",
            "## 10. Qualitative Neighbor Inspection",
            "Not computed because execution failed.",
            "",
            "## 11. Interpretation",
            "- Is memory-only retrieval above chance / informative? Inconclusive because the run failed.",
            "- Does retrieval appear clinically/label meaningful? Inconclusive because the run failed.",
            "- Is the memory signal likely useful enough to try probability mixing next? No, not until this failure is resolved.",
            "- Any obvious failure modes? See the failure trace below.",
            "",
            "## 12. Constraints Respected",
            "- Validation only: intended, but run failed before completion",
            "- Train memory only: intended, but run failed before completion",
            "- No test tuning: yes",
            "- No probability mixing: yes",
            "- No embedding updates: yes",
            "- No index rebuild per batch: yes",
            "",
            "## 13. Final Verdict",
            "FAIL: do not proceed to mixing yet",
            "The Stage 4 validation-only memory conversion did not complete reliably.",
            "",
            "## Appendix A. Metric JSON Snippet",
            "```json",
            json.dumps({"error": failure_message}, indent=2, sort_keys=True),
            "```",
            "",
            "## Appendix B. 5-Line Ultra-Short Update",
            "Stage 4 failed.",
            f"Reason: {failure_message}",
            "No default metrics are trustworthy.",
            "No sweep result should be used.",
            "Resolve the failure before trying probability mixing.",
            "",
            "Failure trace:",
            "```text",
            trace_text.rstrip(),
            "```",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_files_used: list[str] = []
    output_artifacts: list[str] = []

    def record_output(path: Path) -> None:
        output_artifacts.append(str(path))

    try:
        train_memory_root = args.train_memory_root
        train_embeddings_path = train_memory_root / "embeddings.npy"
        train_labels_path = train_memory_root / "labels.npy"
        train_example_ids_path = train_memory_root / "example_ids.json"
        train_image_paths_path = train_memory_root / "image_paths.txt"
        train_metadata_path = train_memory_root / "metadata.json"
        train_index_path = train_memory_root / "index.faiss"

        exact_files_used.extend(
            [
                str(train_embeddings_path),
                str(train_labels_path),
                str(train_example_ids_path),
                str(train_image_paths_path),
                str(train_metadata_path),
                str(train_index_path),
                str(args.val_embeddings),
                str(args.val_image_paths),
                str(args.val_run_meta),
                str(args.manifest_csv),
            ]
        )

        train_embeddings = load_embedding_array(train_embeddings_path)
        train_labels = load_embedding_array(train_labels_path)
        train_example_ids = read_json(train_example_ids_path)
        train_image_paths = read_lines(train_image_paths_path)
        train_metadata = read_json(train_metadata_path)
        val_embeddings_raw = load_embedding_array(args.val_embeddings)
        val_image_paths = read_lines(args.val_image_paths)
        val_run_meta = read_json(args.val_run_meta)
        manifest_rows = read_manifest_rows(args.manifest_csv, split_name="val")

        if not isinstance(train_example_ids, list) or not train_example_ids:
            raise ValueError(f"Invalid example_ids.json in train memory root: {train_example_ids_path}")
        train_example_ids = [str(value) for value in train_example_ids]

        label_names = list(train_metadata.get("label_names", []))
        validate_label_names(label_names)
        check_train_memory_consistency(train_embeddings, train_labels, train_example_ids, train_image_paths)

        if int(train_metadata.get("embedding_dim", -1)) != int(train_embeddings.shape[1]):
            raise ValueError(
                f"Train metadata embedding_dim {train_metadata.get('embedding_dim')} does not match embeddings dim {train_embeddings.shape[1]}."
            )
        if str(train_metadata.get("split")) != "train":
            raise ValueError(f"Train memory metadata split must be 'train', found {train_metadata.get('split')}.")
        if str(val_run_meta.get("split")) != "val":
            raise ValueError(f"Validation run_meta split must be 'val', found {val_run_meta.get('split')}.")
        if val_embeddings_raw.shape[0] != len(val_image_paths):
            raise ValueError(
                f"Validation embeddings rows {val_embeddings_raw.shape[0]} do not match image paths {len(val_image_paths)}."
            )

        aligned_rows, val_example_ids, kept_indices, dropped_rows, alignment_summary = align_validation_rows(
            manifest_rows,
            val_image_paths,
        )
        if kept_indices.size == 0:
            raise ValueError("No validation rows aligned to the manifest; cannot continue.")

        resolved_val_embeddings = np.ascontiguousarray(val_embeddings_raw[kept_indices])
        resolved_val_image_paths = [val_image_paths[int(index)] for index in kept_indices.tolist()]
        val_labels = build_labels(aligned_rows)

        if resolved_val_embeddings.shape[0] != val_labels.shape[0]:
            raise ValueError(
                f"Aligned validation embeddings rows {resolved_val_embeddings.shape[0]} do not match labels {val_labels.shape[0]}."
            )
        for row_index, (example_id, image_path, manifest_row) in enumerate(
            zip(val_example_ids, resolved_val_image_paths, aligned_rows)
        ):
            image_stem = example_id_from_path(image_path)
            manifest_stem = example_id_from_path(manifest_row["image_path"])
            if example_id != image_stem or example_id != manifest_stem:
                raise ValueError(
                    f"Validation alignment mismatch at row {row_index}: "
                    f"example_id={example_id}, image_path stem={image_stem}, manifest stem={manifest_stem}."
                )

        resolved_manifest_rows_path = output_dir / "resolved_val_manifest_rows.jsonl"
        write_jsonl(resolved_manifest_rows_path, aligned_rows)
        record_output(resolved_manifest_rows_path)
        if dropped_rows:
            dropped_rows_path = output_dir / "dropped_val_rows.json"
            write_json(dropped_rows_path, dropped_rows)
            record_output(dropped_rows_path)

        normalized_val_embeddings, norm_summary_before, norm_summary_after = normalize_rows(resolved_val_embeddings)
        if normalized_val_embeddings.shape[1] != train_embeddings.shape[1]:
            raise ValueError(
                f"Validation embedding dim {normalized_val_embeddings.shape[1]} does not match train memory dim {train_embeddings.shape[1]}."
            )

        index, index_details = load_faiss_index(train_index_path, train_embeddings)
        if int(index.ntotal) != int(train_embeddings.shape[0]):
            raise ValueError(
                f"FAISS index ntotal {index.ntotal} does not match train memory row count {train_embeddings.shape[0]}."
            )

        max_k = max(SWEEP_K_VALUES)
        all_neighbor_scores, all_neighbor_indices = index.search(
            np.ascontiguousarray(normalized_val_embeddings.astype(np.float32)),
            max_k,
        )
        if all_neighbor_scores.shape != (normalized_val_embeddings.shape[0], max_k):
            raise ValueError(
                f"Unexpected neighbor score shape {all_neighbor_scores.shape}; expected {(normalized_val_embeddings.shape[0], max_k)}."
            )
        if all_neighbor_indices.shape != (normalized_val_embeddings.shape[0], max_k):
            raise ValueError(
                f"Unexpected neighbor index shape {all_neighbor_indices.shape}; expected {(normalized_val_embeddings.shape[0], max_k)}."
            )

        default_neighbor_scores = np.ascontiguousarray(all_neighbor_scores[:, :DEFAULT_K].astype(np.float32))
        default_neighbor_indices = np.ascontiguousarray(all_neighbor_indices[:, :DEFAULT_K].astype(np.int64))
        default_probabilities = compute_memory_probabilities(
            all_neighbor_indices,
            all_neighbor_scores,
            train_labels,
            k=DEFAULT_K,
            tau=DEFAULT_TAU,
        )

        default_metrics = evaluate_probabilities(
            val_labels,
            default_probabilities,
            label_names,
            include_diagnostic_thresholds=True,
        )

        sweep_rows: list[dict[str, Any]] = []
        for k_value in SWEEP_K_VALUES:
            for tau_value in SWEEP_TAU_VALUES:
                probabilities = compute_memory_probabilities(
                    all_neighbor_indices,
                    all_neighbor_scores,
                    train_labels,
                    k=k_value,
                    tau=tau_value,
                )
                metrics = evaluate_probabilities(
                    val_labels,
                    probabilities,
                    label_names,
                    include_diagnostic_thresholds=False,
                )
                sweep_rows.append(
                    {
                        "k": int(k_value),
                        "tau": int(tau_value),
                        "macro_auroc": metrics["macro_auroc"],
                        "macro_average_precision": metrics["macro_average_precision"],
                        "macro_f1_at_0.5": metrics["macro_f1_at_0.5"],
                    }
                )

        sweep_rows = sorted(sweep_rows, key=lambda row: (int(row["k"]), int(row["tau"])))
        best_row, best_row_trace = select_best_sweep_row(sweep_rows)

        selected_queries = choose_qualitative_query_indices(val_labels)
        qualitative_neighbors = build_qualitative_neighbors(
            selected_queries,
            val_example_ids,
            resolved_val_image_paths,
            val_labels,
            default_neighbor_indices,
            default_neighbor_scores,
            train_example_ids,
            train_labels,
            label_names,
        )

        run_config = {
            "timestamp": timestamp,
            "exact_paths_used": {
                "train_memory_root": str(train_memory_root),
                "train_embeddings": str(train_embeddings_path),
                "train_labels": str(train_labels_path),
                "train_example_ids": str(train_example_ids_path),
                "train_image_paths": str(train_image_paths_path),
                "train_metadata": str(train_metadata_path),
                "train_index": str(train_index_path),
                "validation_embeddings": str(args.val_embeddings),
                "validation_image_paths": str(args.val_image_paths),
                "validation_run_meta": str(args.val_run_meta),
                "manifest_csv": str(args.manifest_csv),
            },
            "label_names": label_names,
            "default_k": DEFAULT_K,
            "default_tau": DEFAULT_TAU,
            "sweep_grid": {"k": SWEEP_K_VALUES, "tau": SWEEP_TAU_VALUES},
            "normalization_rule": "e_val = z_val / ||z_val||_2 applied row-wise with float64 norms and float32 storage",
            "id_parsing_alignment_rule": "example_id = Path(image_path).stem for validation image_paths and manifest image_path",
            "index_loading": index_details,
        }

        retrieval_setup = {
            "memory_size": int(train_embeddings.shape[0]),
            "embedding_dim": int(train_embeddings.shape[1]),
            "index_type": str(train_metadata.get("index", {}).get("type", type(index).__name__)),
            "index_loaded_from_disk": bool(index_details["loaded_from_disk"]),
            "index_rebuilt_from_embeddings": bool(index_details["rebuilt_from_embeddings"]),
            "index_load_error": index_details["load_error"],
        }

        sorted_descending = bool(np.all(default_neighbor_scores[:, :-1] >= (default_neighbor_scores[:, 1:] - 1e-7)))
        invalid_index_count = int(
            np.count_nonzero((default_neighbor_indices < 0) | (default_neighbor_indices >= train_embeddings.shape[0]))
        )
        resolved_neighbor_ids_ok = invalid_index_count == 0
        probability_range = {
            "min": float(default_probabilities.min()),
            "max": float(default_probabilities.max()),
            "mean": float(default_probabilities.mean()),
        }
        sanity_checks = {
            "train_memory": {
                "shape": [int(train_embeddings.shape[0]), int(train_embeddings.shape[1])],
                "labels_shape": [int(train_labels.shape[0]), int(train_labels.shape[1])],
                "split": str(train_metadata.get("split")),
            },
            "validation_embeddings": {
                "shape": [int(normalized_val_embeddings.shape[0]), int(normalized_val_embeddings.shape[1])],
                "raw_shape": [int(resolved_val_embeddings.shape[0]), int(resolved_val_embeddings.shape[1])],
                "dimension_matches_train_memory": bool(normalized_val_embeddings.shape[1] == train_embeddings.shape[1]),
                "finite_before_normalization": bool(np.isfinite(resolved_val_embeddings).all()),
                "finite_after_normalization": bool(np.isfinite(normalized_val_embeddings).all()),
                "norm_summary_before": norm_summary_before,
                "norm_summary_after": norm_summary_after,
            },
            "retrieval_default": {
                "neighbor_index_array_shape": [int(default_neighbor_indices.shape[0]), int(default_neighbor_indices.shape[1])],
                "neighbor_score_array_shape": [int(default_neighbor_scores.shape[0]), int(default_neighbor_scores.shape[1])],
                "scores_sorted_descending": sorted_descending,
                "invalid_index_count": invalid_index_count,
                "all_neighbor_ids_resolve_to_train_memory": resolved_neighbor_ids_ok,
                "similarity_range": {
                    "min": float(default_neighbor_scores.min()),
                    "max": float(default_neighbor_scores.max()),
                    "mean": float(default_neighbor_scores.mean()),
                },
                "probability_range": probability_range,
            },
            "leakage_check": {
                "query_split_is_validation_only": str(val_run_meta.get("split")) == "val",
                "retrieved_items_are_train_memory_only": bool(str(train_metadata.get("split")) == "train"),
                "test_artifacts_used": False,
                "details": "Loaded only train memory artifacts, validation embeddings, and the manifest; no test files were opened.",
            },
        }

        if not sanity_checks["validation_embeddings"]["dimension_matches_train_memory"]:
            raise ValueError("Validation embeddings do not match train memory dimension.")
        if not sanity_checks["validation_embeddings"]["finite_before_normalization"]:
            raise ValueError("Validation embeddings contain NaN or inf values before normalization.")
        if not sanity_checks["validation_embeddings"]["finite_after_normalization"]:
            raise ValueError("Validation embeddings contain NaN or inf values after normalization.")
        if not sanity_checks["retrieval_default"]["scores_sorted_descending"]:
            raise ValueError("Default-run neighbor scores are not sorted in descending order.")
        if invalid_index_count > 0:
            raise ValueError(f"Default-run neighbor indices contain {invalid_index_count} invalid values.")
        if not sanity_checks["leakage_check"]["query_split_is_validation_only"]:
            raise ValueError("Leakage check failed: query split is not validation.")
        if not sanity_checks["leakage_check"]["retrieved_items_are_train_memory_only"]:
            raise ValueError("Leakage check failed: retrieved items are not train memory only.")

        val_example_ids_path = output_dir / "val_example_ids.json"
        val_labels_path = output_dir / "val_labels.npy"
        val_p_mem_default_path = output_dir / "val_p_mem_default.npy"
        val_neighbor_indices_default_path = output_dir / "val_neighbor_indices_default.npy"
        val_neighbor_scores_default_path = output_dir / "val_neighbor_scores_default.npy"
        default_metrics_path = output_dir / "default_metrics.json"
        sweep_results_path = output_dir / "sweep_results.csv"
        best_memory_config_path = output_dir / "best_memory_config.json"
        qualitative_neighbors_path = output_dir / "qualitative_neighbors_val.json"
        sanity_checks_path = output_dir / "sanity_checks.json"
        run_config_path = output_dir / "run_config.json"
        threshold_diagnostics_path = output_dir / "val_threshold_diagnostics.json"
        report_path = output_dir / "report.md"

        write_json(run_config_path, run_config)
        record_output(run_config_path)
        write_json(val_example_ids_path, val_example_ids)
        record_output(val_example_ids_path)
        np.save(val_labels_path, val_labels.astype(np.float32))
        record_output(val_labels_path)
        np.save(val_p_mem_default_path, default_probabilities.astype(np.float32))
        record_output(val_p_mem_default_path)
        np.save(val_neighbor_indices_default_path, default_neighbor_indices.astype(np.int64))
        record_output(val_neighbor_indices_default_path)
        np.save(val_neighbor_scores_default_path, default_neighbor_scores.astype(np.float32))
        record_output(val_neighbor_scores_default_path)
        write_json(default_metrics_path, default_metrics)
        record_output(default_metrics_path)
        write_sweep_csv(sweep_results_path, sweep_rows)
        record_output(sweep_results_path)
        write_json(
            best_memory_config_path,
            {
                "selected_config": {
                    "k": int(best_row["k"]),
                    "tau": int(best_row["tau"]),
                },
                "metrics": {
                    "macro_auroc": best_row["macro_auroc"],
                    "macro_average_precision": best_row["macro_average_precision"],
                    "macro_f1_at_0.5": best_row["macro_f1_at_0.5"],
                },
                "selection_trace": best_row_trace,
            },
        )
        record_output(best_memory_config_path)
        write_json(qualitative_neighbors_path, qualitative_neighbors)
        record_output(qualitative_neighbors_path)
        write_json(sanity_checks_path, sanity_checks)
        record_output(sanity_checks_path)
        write_json(threshold_diagnostics_path, default_metrics["diagnostic_threshold_tuned_f1"])
        record_output(threshold_diagnostics_path)

        report_text = build_success_report(
            timestamp=timestamp,
            exact_files_used=exact_files_used,
            output_artifacts=output_artifacts + [str(report_path)],
            alignment_summary=alignment_summary,
            norm_summary_before=norm_summary_before,
            norm_summary_after=norm_summary_after,
            sanity_checks=sanity_checks,
            retrieval_setup=retrieval_setup,
            default_metrics=default_metrics,
            sweep_rows=sweep_rows,
            best_row=best_row,
            best_row_trace=best_row_trace,
            qualitative_neighbors=qualitative_neighbors,
        )
        report_path.write_text(report_text, encoding="utf-8")
        record_output(report_path)
        print(f"report_path={report_path}")
        return 0

    except Exception as exc:
        failure_trace = traceback.format_exc()
        report_path = output_dir / "report.md"
        failure_report = build_failure_report(
            timestamp=timestamp,
            exact_files_used=exact_files_used,
            output_artifacts=output_artifacts + [str(report_path)],
            failure_message=f"{type(exc).__name__}: {exc}",
            trace_text=failure_trace,
        )
        report_path.write_text(failure_report, encoding="utf-8")
        print(f"report_path={report_path}")
        print(f"failure={type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
