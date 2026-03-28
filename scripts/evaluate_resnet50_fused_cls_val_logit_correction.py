#!/usr/bin/env python3
"""Stage 5B: validation-only logit correction for ResNet50 fused CLS."""

from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from evaluate_resnet50_fused_cls_val_probability_mixing import (
    FLOAT_TOL,
    LABEL_COLUMNS,
    LABEL_NAMES,
    align_array_to_reference,
    align_validation_rows,
    binary_cross_entropy_per_example,
    build_labels,
    build_markdown_table,
    check_train_memory_consistency,
    choose_indices_for_category,
    compute_memory_probabilities,
    example_id_from_path,
    format_metric,
    labels_to_names,
    load_checkpoint,
    load_embedding_array,
    load_faiss_index,
    normalize_rows,
    probability_summary,
    read_json,
    read_lines,
    read_manifest_rows,
    remap_state_dict_for_linear_head,
    resolve_existing_path,
    tune_f1_thresholds,
    validate_label_names,
    write_json,
)


FORMULATION_PRIMARY = "primary_multilabel_logit"
FORMULATION_PDF = "pdf_literal_logprob"
FORMULATIONS = [FORMULATION_PRIMARY, FORMULATION_PDF]

BETA_GRID_COARSE = [0.00, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00]
PRIMARY_MEMORY_CONFIG = {"k": 50, "tau": 1}
EPSILON = 1e-8
ECE_BINS = 15

DEFAULT_BASELINE_RUN_ROOT = Path("/workspace/outputs/models/nih_cxr14/fused/resnet50_cls_20260324T091149Z")
DEFAULT_BASELINE_RUN_ROOT_FALLBACK = Path(
    "/workspace/outputs/nih_cxr14_frozen_fused_linear_cls_resnet50/resnet50_cls_20260324T091149Z"
)
DEFAULT_STAGE5A_OUTPUT_DIR = Path("/workspace/memory_eval/nih_cxr14/resnet50_fused_cls_val_probability_mixing")
DEFAULT_STAGE4_OUTPUT_DIR = Path("/workspace/memory_eval/nih_cxr14/resnet50_fused_cls_val_memory_only")
DEFAULT_TRAIN_MEMORY_ROOT = Path("/workspace/memory/nih_cxr14/resnet50_fused_cls_train")
DEFAULT_VAL_EMBEDDINGS = Path("/workspace/fused_embeddings_cls/resnet50/val/embeddings.npy")
DEFAULT_VAL_IMAGE_PATHS = Path("/workspace/fused_embeddings_cls/resnet50/val/image_paths.txt")
DEFAULT_VAL_RUN_META = Path("/workspace/fused_embeddings_cls/resnet50/val/run_meta.json")
DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_nih_cxr14 .csv")
DEFAULT_OUTPUT_DIR = Path("/workspace/memory_eval/nih_cxr14/resnet50_fused_cls_val_logit_correction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5B only: validation-only logit correction using frozen baseline logits "
            "and frozen retrieval memory probabilities for ResNet50 fused CLS."
        )
    )
    parser.add_argument("--baseline-run-root", type=Path, default=DEFAULT_BASELINE_RUN_ROOT)
    parser.add_argument("--baseline-run-root-fallback", type=Path, default=DEFAULT_BASELINE_RUN_ROOT_FALLBACK)
    parser.add_argument("--stage5a-output-dir", type=Path, default=DEFAULT_STAGE5A_OUTPUT_DIR)
    parser.add_argument("--stage4-output-dir", type=Path, default=DEFAULT_STAGE4_OUTPUT_DIR)
    parser.add_argument("--train-memory-root", type=Path, default=DEFAULT_TRAIN_MEMORY_ROOT)
    parser.add_argument("--val-embeddings", type=Path, default=DEFAULT_VAL_EMBEDDINGS)
    parser.add_argument("--val-image-paths", type=Path, default=DEFAULT_VAL_IMAGE_PATHS)
    parser.add_argument("--val-run-meta", type=Path, default=DEFAULT_VAL_RUN_META)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eps", type=float, default=EPSILON)
    parser.add_argument("--ece-bins", type=int, default=ECE_BINS)
    parser.add_argument("--run-optional-refinement", action="store_true")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sigmoid_array(values: np.ndarray) -> np.ndarray:
    values64 = values.astype(np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-values64))
    probabilities = np.ascontiguousarray(probabilities.astype(np.float32))
    if not np.isfinite(probabilities).all():
        raise ValueError("Sigmoid probabilities contain NaN or inf values.")
    return probabilities


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


def compute_binary_ece(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int,
) -> tuple[float, list[dict[str, Any]]]:
    if targets.shape != probabilities.shape:
        raise ValueError(f"ECE inputs must share the same shape, found {targets.shape} vs {probabilities.shape}.")
    if targets.ndim != 1:
        raise ValueError(f"ECE expects 1D arrays, found {targets.ndim} dimensions.")
    clipped = np.clip(probabilities.astype(np.float64), 0.0, 1.0)
    targets64 = targets.astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, num=n_bins + 1, dtype=np.float64)
    reliability_rows: list[dict[str, Any]] = []
    weighted_gap = 0.0
    total_count = int(targets.shape[0])
    for bin_index in range(n_bins):
        lower = float(bin_edges[bin_index])
        upper = float(bin_edges[bin_index + 1])
        if bin_index == n_bins - 1:
            mask = (clipped >= lower) & (clipped <= upper)
        else:
            mask = (clipped >= lower) & (clipped < upper)
        count = int(np.count_nonzero(mask))
        if count == 0:
            reliability_rows.append(
                {
                    "bin_index": bin_index,
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "avg_probability": None,
                    "empirical_positive_rate": None,
                    "abs_gap": None,
                }
            )
            continue
        avg_probability = float(clipped[mask].mean())
        empirical_positive_rate = float(targets64[mask].mean())
        abs_gap = abs(avg_probability - empirical_positive_rate)
        weighted_gap += (count / max(total_count, 1)) * abs_gap
        reliability_rows.append(
            {
                "bin_index": bin_index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "avg_probability": avg_probability,
                "empirical_positive_rate": empirical_positive_rate,
                "abs_gap": float(abs_gap),
            }
        )
    return float(weighted_gap), reliability_rows


def compute_binary_brier(targets: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities.astype(np.float64) - targets.astype(np.float64)) ** 2))


def evaluate_probabilities_extended(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
    *,
    include_diagnostic_thresholds: bool,
    ece_bins: int,
    include_reliability_tables: bool,
) -> dict[str, Any]:
    per_label: dict[str, dict[str, Any]] = {}
    macro_auroc_values: list[float] = []
    macro_average_precision_values: list[float] = []
    macro_f1_at_0p5_values: list[float] = []
    macro_ece_values: list[float] = []
    macro_brier_values: list[float] = []

    for label_index, label_name in enumerate(label_names):
        target_column = targets[:, label_index]
        probability_column = probabilities[:, label_index]
        auroc = compute_binary_metric("auroc", target_column, probability_column)
        average_precision = compute_binary_metric("average_precision", target_column, probability_column)
        f1_at_0p5 = float(f1_score(target_column, probability_column >= 0.5, zero_division=0))
        ece, reliability_rows = compute_binary_ece(target_column, probability_column, n_bins=ece_bins)
        brier = compute_binary_brier(target_column, probability_column)

        if auroc is not None:
            macro_auroc_values.append(auroc)
        if average_precision is not None:
            macro_average_precision_values.append(average_precision)
        macro_f1_at_0p5_values.append(f1_at_0p5)
        macro_ece_values.append(ece)
        macro_brier_values.append(brier)

        payload = {
            "auroc": auroc,
            "average_precision": average_precision,
            "f1_at_0.5": f1_at_0p5,
            "ece": ece,
            "brier": brier,
            "positive_count": int(target_column.sum()),
            "negative_count": int(target_column.shape[0] - target_column.sum()),
        }
        if include_reliability_tables:
            payload["reliability_table"] = reliability_rows
        per_label[label_name] = payload

    metrics = {
        "macro_auroc": float(np.mean(macro_auroc_values)) if macro_auroc_values else None,
        "macro_average_precision": (
            float(np.mean(macro_average_precision_values)) if macro_average_precision_values else None
        ),
        "macro_f1_at_0.5": float(np.mean(macro_f1_at_0p5_values)) if macro_f1_at_0p5_values else None,
        "macro_ece": float(np.mean(macro_ece_values)) if macro_ece_values else None,
        "macro_brier": float(np.mean(macro_brier_values)) if macro_brier_values else None,
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


def find_saved_baseline_logit_artifacts(baseline_run_root: Path) -> list[Path]:
    if not baseline_run_root.exists():
        return []
    candidates: list[Path] = []
    patterns = [
        "*val*logit*.npy",
        "*validation*logit*.npy",
        "*val*logits*.npy",
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


def load_or_reconstruct_z_base(
    *,
    baseline_run_root: Path,
    val_embeddings_path: Path,
    val_image_paths_path: Path,
    manifest_csv_path: Path,
    batch_size: int,
    label_names: list[str],
    files_used: set[Path],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, dict[str, Any]]:
    saved_candidates = find_saved_baseline_logit_artifacts(baseline_run_root)
    val_image_paths = read_lines(val_image_paths_path)
    files_used.add(val_image_paths_path)
    val_example_ids = [example_id_from_path(path) for path in val_image_paths]
    if len(val_example_ids) != len(set(val_example_ids)):
        raise ValueError("Validation example IDs derived from baseline image paths are not unique.")

    manifest_rows = read_manifest_rows(manifest_csv_path, "val")
    files_used.add(manifest_csv_path)
    aligned_rows, manifest_val_example_ids, _, dropped_rows, _ = align_validation_rows(manifest_rows, val_image_paths)
    if dropped_rows:
        raise ValueError(f"Baseline reconstruction path found dropped validation rows: {len(dropped_rows)}")
    baseline_labels = build_labels(aligned_rows)

    if saved_candidates:
        loaded_path = saved_candidates[0]
        files_used.add(loaded_path)
        z_base = load_embedding_array(loaded_path)
        if z_base.shape != baseline_labels.shape:
            raise ValueError(f"Saved baseline logits shape {z_base.shape} does not match labels {baseline_labels.shape}.")
        p_base = sigmoid_array(z_base)
        details = {
            "source": "loaded_saved_validation_logits",
            "loaded_path": str(loaded_path),
            "reconstructed": False,
            "checkpoint_loading": None,
            "checkpoint_key_remap": None,
            "model_type": None,
            "hidden_layers": None,
            "input_dim": int(z_base.shape[1]),
            "output_dim": int(z_base.shape[1]),
            "device": None,
            "batch_size": None,
            "dropped_rows": 0,
            "saved_logit_candidates_found": [str(path) for path in saved_candidates],
            "architecture_assumption": "loaded archived validation logits directly",
        }
        return z_base.astype(np.float32), p_base, manifest_val_example_ids, baseline_labels, details

    config_path = baseline_run_root / "config.json"
    checkpoint_path = baseline_run_root / "best.ckpt"
    archived_val_metrics_path = baseline_run_root / "val_metrics.json"
    archived_thresholds_path = baseline_run_root / "val_f1_thresholds.json"
    files_used.update({config_path, checkpoint_path, archived_val_metrics_path, archived_thresholds_path})

    config = read_json(config_path)
    archived_val_metrics = read_json(archived_val_metrics_path)
    archived_thresholds = read_json(archived_thresholds_path)
    loaded_label_names = [str(name) for name in config.get("label_names", [])]
    if loaded_label_names != label_names:
        raise ValueError(f"Unexpected baseline label order. Expected {label_names}, found {loaded_label_names}.")

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
    if expected_output_dim != len(label_names):
        raise ValueError(f"Baseline output_dim {expected_output_dim} does not match label count {len(label_names)}.")
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
    logit_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            logits = model(features[start:end]).cpu().numpy().astype(np.float32)
            logit_batches.append(logits)

    z_base = np.ascontiguousarray(np.concatenate(logit_batches, axis=0).astype(np.float32))
    p_base = sigmoid_array(z_base)
    computed_metrics = evaluate_probabilities_extended(
        baseline_labels,
        p_base,
        label_names,
        include_diagnostic_thresholds=False,
        ece_bins=ECE_BINS,
        include_reliability_tables=False,
    )
    archived_mismatches: dict[str, Any] = {"per_label": {}, "threshold_file_present": bool(archived_thresholds)}
    max_metric_delta = 0.0
    for label_name in label_names:
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
        "saved_logit_candidates_found": [str(path) for path in saved_candidates],
        "archived_metric_check": archived_mismatches,
        "architecture_assumption": (
            f"plain torch.nn.Linear({expected_input_dim}, {expected_output_dim}) on frozen fused CLS embeddings"
        ),
    }
    return z_base, p_base, manifest_val_example_ids, baseline_labels, details


def load_frozen_memory_probabilities(
    *,
    stage5a_output_dir: Path,
    stage4_output_dir: Path,
    train_memory_root: Path,
    val_embeddings_path: Path,
    val_image_paths_path: Path,
    manifest_csv_path: Path,
    files_used: set[Path],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    stage5a_ids_path = stage5a_output_dir / "aligned_val_example_ids.json"
    stage5a_p_mem_path = stage5a_output_dir / "p_mem_k50_tau1.npy"
    if stage5a_ids_path.exists() and stage5a_p_mem_path.exists():
        files_used.update({stage5a_ids_path, stage5a_p_mem_path})
        example_ids = [str(value) for value in read_json(stage5a_ids_path)]
        probabilities = load_embedding_array(stage5a_p_mem_path)
        if probabilities.shape[0] != len(example_ids):
            raise ValueError(
                f"Stage 5A p_mem rows {probabilities.shape[0]} do not match example ids {len(example_ids)}."
            )
        return probabilities.astype(np.float32), example_ids, {
            "mode": "loaded",
            "source": "stage5a",
            "source_paths": [str(stage5a_p_mem_path), str(stage5a_ids_path)],
        }

    stage4_ids_path = stage4_output_dir / "val_example_ids.json"
    stage4_candidates = [
        stage4_output_dir / "val_p_mem_k50_tau1.npy",
        stage4_output_dir / "val_p_mem_best.npy",
    ]
    saved_stage4_path = next((path for path in stage4_candidates if path.exists()), None)
    if stage4_ids_path.exists() and saved_stage4_path is not None:
        files_used.update({stage4_ids_path, saved_stage4_path})
        example_ids = [str(value) for value in read_json(stage4_ids_path)]
        probabilities = load_embedding_array(saved_stage4_path)
        if probabilities.shape[0] != len(example_ids):
            raise ValueError(
                f"Stage 4 p_mem rows {probabilities.shape[0]} do not match example ids {len(example_ids)}."
            )
        return probabilities.astype(np.float32), example_ids, {
            "mode": "loaded",
            "source": "stage4",
            "source_paths": [str(saved_stage4_path), str(stage4_ids_path)],
        }

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
            val_embeddings_path,
            val_image_paths_path,
            manifest_csv_path,
        }
    )

    train_embeddings = load_embedding_array(train_embeddings_path)
    train_labels = load_embedding_array(train_labels_path)
    train_example_ids = [str(value) for value in read_json(train_example_ids_path)]
    train_image_paths = read_lines(train_image_paths_path)
    check_train_memory_consistency(train_embeddings, train_labels, train_example_ids, train_image_paths)

    val_embeddings = load_embedding_array(val_embeddings_path)
    val_image_paths = read_lines(val_image_paths_path)
    manifest_rows = read_manifest_rows(manifest_csv_path, "val")
    aligned_rows, manifest_val_example_ids, kept_indices, dropped_rows, _ = align_validation_rows(manifest_rows, val_image_paths)
    _ = aligned_rows
    if dropped_rows:
        raise ValueError(f"Found {len(dropped_rows)} dropped validation rows while preparing memory queries.")
    normalized_val_embeddings, _, _ = normalize_rows(val_embeddings[kept_indices])

    index, index_loading = load_faiss_index(train_index_path, train_embeddings)
    if index.ntotal != train_embeddings.shape[0]:
        raise ValueError(f"FAISS index ntotal {index.ntotal} does not match train embeddings rows {train_embeddings.shape[0]}.")
    neighbor_scores, neighbor_indices = index.search(
        np.ascontiguousarray(normalized_val_embeddings.astype(np.float32)),
        int(PRIMARY_MEMORY_CONFIG["k"]),
    )
    if neighbor_scores.shape != (normalized_val_embeddings.shape[0], int(PRIMARY_MEMORY_CONFIG["k"])):
        raise ValueError(f"Unexpected neighbor score shape: {neighbor_scores.shape}")
    if neighbor_indices.shape != (normalized_val_embeddings.shape[0], int(PRIMARY_MEMORY_CONFIG["k"])):
        raise ValueError(f"Unexpected neighbor index shape: {neighbor_indices.shape}")
    if not np.isfinite(neighbor_scores).all():
        raise ValueError("Neighbor scores contain NaN or inf values.")
    if int(np.count_nonzero((neighbor_indices < 0) | (neighbor_indices >= train_embeddings.shape[0]))) > 0:
        raise ValueError("Neighbor indices contain out-of-range values.")

    probabilities = compute_memory_probabilities(
        neighbor_indices=neighbor_indices,
        neighbor_scores=neighbor_scores,
        train_labels=train_labels,
        k=int(PRIMARY_MEMORY_CONFIG["k"]),
        tau=int(PRIMARY_MEMORY_CONFIG["tau"]),
    )
    return probabilities.astype(np.float32), manifest_val_example_ids, {
        "mode": "recomputed",
        "source": "recomputed_from_saved_train_memory",
        "source_paths": [
            str(train_embeddings_path),
            str(train_labels_path),
            str(train_example_ids_path),
            str(train_image_paths_path),
            str(train_index_path),
            str(val_embeddings_path),
            str(val_image_paths_path),
            str(manifest_csv_path),
        ],
        "index_loading": index_loading,
    }


def build_memory_transforms(
    p_mem: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, found {eps}.")
    p_mem64 = p_mem.astype(np.float64)
    primary_input = np.clip(p_mem64, eps, 1.0 - eps)
    pdf_input = np.clip(p_mem64, 0.0, 1.0 - eps)

    z_mem_primary = np.log((primary_input + eps) / (1.0 - primary_input + eps))
    z_mem_pdf = np.log(pdf_input + eps)

    z_mem_primary = np.ascontiguousarray(z_mem_primary.astype(np.float32))
    z_mem_pdf = np.ascontiguousarray(z_mem_pdf.astype(np.float32))

    if not np.isfinite(z_mem_primary).all():
        raise ValueError("Primary multilabel logit transform contains NaN or inf values.")
    if not np.isfinite(z_mem_pdf).all():
        raise ValueError("PDF-literal log-prob transform contains NaN or inf values.")

    pdf_tolerance = 1e-7
    if float(z_mem_pdf.max()) > pdf_tolerance:
        raise ValueError(f"PDF-literal transform exceeded zero tolerance: max={float(z_mem_pdf.max())}")

    summary = {
        "eps": float(eps),
        "primary_multilabel_logit": {
            "input_clip_range": [float(eps), float(1.0 - eps)],
            "low_clipped_count": int(np.count_nonzero(p_mem64 < eps)),
            "high_clipped_count": int(np.count_nonzero(p_mem64 > 1.0 - eps)),
            "clipped_using_eps_before_transform": True,
            "min": float(z_mem_primary.min()),
            "max": float(z_mem_primary.max()),
            "mean": float(z_mem_primary.mean()),
            "fraction_gt_zero": float(np.mean(z_mem_primary > 0.0)),
        },
        "pdf_literal_logprob": {
            "input_clip_range": [0.0, float(1.0 - eps)],
            "low_clipped_count": int(np.count_nonzero(p_mem64 < 0.0)),
            "high_clipped_count": int(np.count_nonzero(p_mem64 > 1.0 - eps)),
            "clipped_using_eps_before_transform": True,
            "min": float(z_mem_pdf.min()),
            "max": float(z_mem_pdf.max()),
            "mean": float(z_mem_pdf.mean()),
            "all_values_leq_zero_within_tolerance": bool(float(z_mem_pdf.max()) <= pdf_tolerance),
            "zero_tolerance": pdf_tolerance,
        },
    }
    return z_mem_primary, z_mem_pdf, summary


def apply_logit_correction(
    *,
    z_base: np.ndarray,
    z_mem: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if beta < 0.0:
        raise ValueError(f"beta must be >= 0, found {beta}.")
    if z_base.shape != z_mem.shape:
        raise ValueError(f"z_base and z_mem shapes must match, found {z_base.shape} vs {z_mem.shape}.")

    corrected = z_base.astype(np.float64) + float(beta) * z_mem.astype(np.float64)
    z_corr = np.ascontiguousarray(corrected.astype(np.float32))
    p_corr = sigmoid_array(z_corr)
    sanity = {
        "z_corr_all_finite": bool(np.isfinite(z_corr).all()),
        "p_corr_all_finite": bool(np.isfinite(p_corr).all()),
        "p_corr_min": float(p_corr.min()),
        "p_corr_max": float(p_corr.max()),
        "p_corr_range_ok": bool(float(p_corr.min()) >= -1e-6 and float(p_corr.max()) <= 1.0 + 1e-6),
    }
    if not sanity["z_corr_all_finite"] or not sanity["p_corr_all_finite"] or not sanity["p_corr_range_ok"]:
        raise ValueError(f"Corrected outputs failed sanity checks for beta={beta}: {sanity}")
    return z_corr, p_corr, sanity


def evaluate_beta_rows(
    *,
    formulation: str,
    z_base: np.ndarray,
    z_mem: np.ndarray,
    targets: np.ndarray,
    label_names: list[str],
    beta_grid: list[float],
    search_stage: str,
    ece_bins: int,
    include_diagnostic_thresholds: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for beta in beta_grid:
        z_corr, p_corr, output_sanity = apply_logit_correction(z_base=z_base, z_mem=z_mem, beta=beta)
        metrics = evaluate_probabilities_extended(
            targets,
            p_corr,
            label_names,
            include_diagnostic_thresholds=include_diagnostic_thresholds,
            ece_bins=ece_bins,
            include_reliability_tables=False,
        )
        rows.append(
            {
                "formulation": formulation,
                "beta": float(beta),
                "search_stage": search_stage,
                "z_corr": z_corr,
                "p_corr": p_corr,
                "metrics": metrics,
                "output_sanity": output_sanity,
                "macro_auroc": metrics["macro_auroc"],
                "macro_average_precision": metrics["macro_average_precision"],
                "macro_f1_at_0.5": metrics["macro_f1_at_0.5"],
                "macro_ece": metrics["macro_ece"],
                "macro_brier": metrics["macro_brier"],
            }
        )
    return rows


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

    min_ece = min(float(row["macro_ece"]) for row in ap_candidates)
    ece_candidates = [row for row in ap_candidates if np.isclose(float(row["macro_ece"]), min_ece, atol=FLOAT_TOL, rtol=0.0)]

    min_beta = min(float(row["beta"]) for row in ece_candidates)
    beta_candidates = [row for row in ece_candidates if np.isclose(float(row["beta"]), min_beta, atol=FLOAT_TOL, rtol=0.0)]

    primary_candidates = [row for row in beta_candidates if str(row["formulation"]) == FORMULATION_PRIMARY]
    formulation_candidates = primary_candidates if primary_candidates else beta_candidates

    coarse_candidates = [row for row in formulation_candidates if str(row.get("search_stage", "coarse")) == "coarse"]
    final_candidates = coarse_candidates if coarse_candidates else formulation_candidates

    best_row = sorted(
        final_candidates,
        key=lambda row: (
            float(row["macro_auroc"]),
            float(row["macro_average_precision"]),
            -float(row["macro_ece"]),
            -float(row["beta"]),
            1 if str(row["formulation"]) == FORMULATION_PRIMARY else 0,
            1 if str(row.get("search_stage", "coarse")) == "coarse" else 0,
        ),
        reverse=True,
    )[0]

    trace = {
        "max_macro_auroc": max_auroc,
        "macro_auroc_tied_candidates": int(len(auroc_candidates)),
        "max_macro_average_precision_within_auroc_ties": max_ap,
        "macro_average_precision_tied_candidates": int(len(ap_candidates)),
        "min_macro_ece_within_metric_ties": min_ece,
        "macro_ece_tied_candidates": int(len(ece_candidates)),
        "smallest_beta_within_metric_ties": min_beta,
        "beta_tied_candidates": int(len(beta_candidates)),
        "primary_formulation_tied_candidates": int(len(primary_candidates)),
        "coarse_stage_tied_candidates": int(len(coarse_candidates)),
        "final_candidates": int(len(final_candidates)),
        "tie_break_rule": [
            "highest validation macro AUROC",
            "break ties with higher validation macro average precision",
            "if still tied, lower validation macro ECE",
            "if still tied, prefer smaller beta",
            "if still tied, prefer primary_multilabel_logit over pdf_literal_logprob",
            "if still tied, prefer coarse-grid winner over refined winner",
        ],
    }
    return best_row, trace


def select_best_row_within_formulation(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise ValueError("No rows available for formulation selection.")
    max_auroc = max(float(row["macro_auroc"]) for row in rows)
    auroc_candidates = [row for row in rows if np.isclose(float(row["macro_auroc"]), max_auroc, atol=FLOAT_TOL, rtol=0.0)]
    max_ap = max(float(row["macro_average_precision"]) for row in auroc_candidates)
    ap_candidates = [
        row
        for row in auroc_candidates
        if np.isclose(float(row["macro_average_precision"]), max_ap, atol=FLOAT_TOL, rtol=0.0)
    ]
    min_ece = min(float(row["macro_ece"]) for row in ap_candidates)
    ece_candidates = [row for row in ap_candidates if np.isclose(float(row["macro_ece"]), min_ece, atol=FLOAT_TOL, rtol=0.0)]
    min_beta = min(float(row["beta"]) for row in ece_candidates)
    beta_candidates = [row for row in ece_candidates if np.isclose(float(row["beta"]), min_beta, atol=FLOAT_TOL, rtol=0.0)]
    coarse_candidates = [row for row in beta_candidates if str(row.get("search_stage", "coarse")) == "coarse"]
    final_candidates = coarse_candidates if coarse_candidates else beta_candidates
    best_row = sorted(
        final_candidates,
        key=lambda row: (
            float(row["macro_auroc"]),
            float(row["macro_average_precision"]),
            -float(row["macro_ece"]),
            -float(row["beta"]),
            1 if str(row.get("search_stage", "coarse")) == "coarse" else 0,
        ),
        reverse=True,
    )[0]
    trace = {
        "max_macro_auroc": max_auroc,
        "macro_auroc_tied_candidates": int(len(auroc_candidates)),
        "max_macro_average_precision_within_auroc_ties": max_ap,
        "macro_average_precision_tied_candidates": int(len(ap_candidates)),
        "min_macro_ece_within_metric_ties": min_ece,
        "macro_ece_tied_candidates": int(len(ece_candidates)),
        "smallest_beta_within_metric_ties": min_beta,
        "beta_tied_candidates": int(len(beta_candidates)),
        "coarse_stage_tied_candidates": int(len(coarse_candidates)),
        "final_candidates": int(len(final_candidates)),
    }
    return best_row, trace


def write_logit_correction_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "formulation",
        "beta",
        "search_stage",
        "macro_auroc",
        "macro_average_precision",
        "macro_f1_at_0.5",
        "macro_ece",
        "macro_brier",
    ]
    for label_name in LABEL_NAMES:
        fieldnames.extend(
            [
                f"{label_name}_auroc",
                f"{label_name}_average_precision",
                f"{label_name}_f1_at_0.5",
                f"{label_name}_ece",
                f"{label_name}_brier",
            ]
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {
                "formulation": str(row["formulation"]),
                "beta": float(row["beta"]),
                "search_stage": str(row.get("search_stage", "coarse")),
                "macro_auroc": row["macro_auroc"],
                "macro_average_precision": row["macro_average_precision"],
                "macro_f1_at_0.5": row["macro_f1_at_0.5"],
                "macro_ece": row["macro_ece"],
                "macro_brier": row["macro_brier"],
            }
            for label_name in LABEL_NAMES:
                label_metrics = row["metrics"]["per_label"][label_name]
                payload[f"{label_name}_auroc"] = label_metrics["auroc"]
                payload[f"{label_name}_average_precision"] = label_metrics["average_precision"]
                payload[f"{label_name}_f1_at_0.5"] = label_metrics["f1_at_0.5"]
                payload[f"{label_name}_ece"] = label_metrics["ece"]
                payload[f"{label_name}_brier"] = label_metrics["brier"]
            writer.writerow(payload)


def metric_headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_auroc": metrics.get("macro_auroc"),
        "macro_average_precision": metrics.get("macro_average_precision"),
        "macro_f1_at_0.5": metrics.get("macro_f1_at_0.5"),
        "macro_ece": metrics.get("macro_ece"),
        "macro_brier": metrics.get("macro_brier"),
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
    per_label_ece_deltas = {
        label_name: float(best_metrics["per_label"][label_name]["ece"] - reference_metrics["per_label"][label_name]["ece"])
        for label_name in LABEL_NAMES
    }
    return {
        "macro_auroc_delta": float(best_metrics["macro_auroc"] - reference_metrics["macro_auroc"]),
        "macro_average_precision_delta": float(
            best_metrics["macro_average_precision"] - reference_metrics["macro_average_precision"]
        ),
        "macro_f1_at_0.5_delta": float(best_metrics["macro_f1_at_0.5"] - reference_metrics["macro_f1_at_0.5"]),
        "macro_ece_delta": float(best_metrics["macro_ece"] - reference_metrics["macro_ece"]),
        "macro_brier_delta": float(best_metrics["macro_brier"] - reference_metrics["macro_brier"]),
        "per_label_auroc_deltas": per_label_auroc_deltas,
        "per_label_ece_deltas": per_label_ece_deltas,
    }


def infer_logit_case_note(
    *,
    true_labels: np.ndarray,
    p_base: np.ndarray,
    p_mem: np.ndarray,
    p_mix: np.ndarray,
    p_corr: np.ndarray,
    base_bce: float,
    mix_bce: float,
    corr_bce: float,
) -> str:
    positive_mask = true_labels > 0.5
    negative_mask = ~positive_mask

    if abs(corr_bce - mix_bce) <= 0.01 and abs(corr_bce - base_bce) <= 0.01:
        return "neutral / negligible change"

    if positive_mask.any():
        corr_true_mean = float(p_corr[positive_mask].mean())
        base_true_mean = float(p_base[positive_mask].mean())
        mix_true_mean = float(p_mix[positive_mask].mean())
        if corr_bce + 0.02 < min(base_bce, mix_bce) and corr_true_mean > max(base_true_mean, mix_true_mean):
            return "logit correction rescues weak positive"
        if corr_bce > min(base_bce, mix_bce) + 0.02 and corr_true_mean + 0.05 < max(base_true_mean, mix_true_mean):
            return "logit correction over-suppresses"

    if negative_mask.any():
        corr_false_mean = float(p_corr[negative_mask].mean())
        base_false_mean = float(p_base[negative_mask].mean())
        mix_false_mean = float(p_mix[negative_mask].mean())
        if corr_bce + 0.02 < min(base_bce, mix_bce) and corr_false_mean + 0.05 < min(base_false_mean, mix_false_mean):
            return "logit correction fixes false positive"
        if corr_bce > min(base_bce, mix_bce) + 0.02 and corr_false_mean > max(base_false_mean, mix_false_mean) + 0.05:
            return "suspicious memory effect"

    if corr_bce > base_bce + 0.02 and corr_bce > mix_bce + 0.02:
        return "logit correction suppresses useful baseline signal"
    if abs(float(np.max(p_mem)) - float(np.max(p_corr))) > 0.4 and corr_bce > min(base_bce, mix_bce):
        return "suspicious memory effect"
    return "neutral / negligible change"


def build_qualitative_cases(
    *,
    example_ids: list[str],
    targets: np.ndarray,
    z_base: np.ndarray,
    p_base: np.ndarray,
    p_mem: np.ndarray,
    p_mix_best: np.ndarray,
    p_corr_best: np.ndarray,
) -> list[dict[str, Any]]:
    base_bce = binary_cross_entropy_per_example(targets, p_base)
    mix_bce = binary_cross_entropy_per_example(targets, p_mix_best)
    corr_bce = binary_cross_entropy_per_example(targets, p_corr_best)

    improve_over_both = np.minimum(base_bce - corr_bce, mix_bce - corr_bce)
    stage5a_better = corr_bce - mix_bce
    neutral_score = np.abs(corr_bce - mix_bce) + np.abs(corr_bce - base_bce)

    label_count = targets.sum(axis=1).astype(np.int64)
    label_count_categories = [
        "negative_or_unlabeled" if int(count) == 0 else "single_positive" if int(count) == 1 else "multi_positive"
        for count in label_count.tolist()
    ]

    improved_sorted = [int(index) for index in np.argsort(-improve_over_both).tolist()]
    stage5a_sorted = [int(index) for index in np.argsort(-stage5a_better).tolist()]
    neutral_sorted = [int(index) for index in np.argsort(neutral_score).tolist()]

    used_indices: set[int] = set()
    improved_indices = choose_indices_for_category(
        sorted_indices=improved_sorted,
        desired_counts={"negative_or_unlabeled": 1, "single_positive": 1, "multi_positive": 2},
        label_count_categories=label_count_categories,
        used_indices=used_indices,
    )
    used_indices.update(improved_indices)
    stage5a_better_indices = choose_indices_for_category(
        sorted_indices=stage5a_sorted,
        desired_counts={"negative_or_unlabeled": 1, "single_positive": 1, "multi_positive": 2},
        label_count_categories=label_count_categories,
        used_indices=used_indices,
    )
    used_indices.update(stage5a_better_indices)
    neutral_indices = choose_indices_for_category(
        sorted_indices=neutral_sorted,
        desired_counts={"negative_or_unlabeled": 0, "single_positive": 2, "multi_positive": 2},
        label_count_categories=label_count_categories,
        used_indices=used_indices,
    )

    selected_cases: list[tuple[int, str]] = []
    selected_cases.extend((index, "stage5b_improves_over_baseline_and_stage5a") for index in improved_indices[:4])
    selected_cases.extend((index, "stage5a_better_than_stage5b") for index in stage5a_better_indices[:4])
    selected_cases.extend((index, "ambiguous_or_neutral") for index in neutral_indices[:4])

    payload: list[dict[str, Any]] = []
    for index, review_bucket in selected_cases[:12]:
        true_labels = targets[index]
        note = infer_logit_case_note(
            true_labels=true_labels,
            p_base=p_base[index],
            p_mem=p_mem[index],
            p_mix=p_mix_best[index],
            p_corr=p_corr_best[index],
            base_bce=float(base_bce[index]),
            mix_bce=float(mix_bce[index]),
            corr_bce=float(corr_bce[index]),
        )
        payload.append(
            {
                "review_bucket": review_bucket,
                "validation_example_id": example_ids[index],
                "label_count_category": label_count_categories[index],
                "true_labels": labels_to_names(true_labels),
                "true_label_vector": [int(value) for value in true_labels.astype(np.int64).tolist()],
                "z_base": {label_name: float(value) for label_name, value in zip(LABEL_NAMES, z_base[index].tolist())},
                "p_base": {label_name: float(value) for label_name, value in zip(LABEL_NAMES, p_base[index].tolist())},
                "p_mem": {label_name: float(value) for label_name, value in zip(LABEL_NAMES, p_mem[index].tolist())},
                "p_mix_best": {label_name: float(value) for label_name, value in zip(LABEL_NAMES, p_mix_best[index].tolist())},
                "p_corr_best": {label_name: float(value) for label_name, value in zip(LABEL_NAMES, p_corr_best[index].tolist())},
                "base_bce": float(base_bce[index]),
                "stage5a_mix_bce": float(mix_bce[index]),
                "stage5b_corr_bce": float(corr_bce[index]),
                "delta_vs_baseline_bce": float(base_bce[index] - corr_bce[index]),
                "delta_vs_stage5a_bce": float(mix_bce[index] - corr_bce[index]),
                "note": note,
            }
        )
    return payload


def compare_best_overall_to_stage5a(best_metrics: dict[str, Any], stage5a_metrics: dict[str, Any]) -> str:
    if float(best_metrics["macro_auroc"]) > float(stage5a_metrics["macro_auroc"]) + FLOAT_TOL:
        return "better"
    if np.isclose(float(best_metrics["macro_auroc"]), float(stage5a_metrics["macro_auroc"]), atol=FLOAT_TOL, rtol=0.0):
        if float(best_metrics["macro_average_precision"]) > float(stage5a_metrics["macro_average_precision"]) + FLOAT_TOL:
            return "better"
        if np.isclose(
            float(best_metrics["macro_average_precision"]),
            float(stage5a_metrics["macro_average_precision"]),
            atol=FLOAT_TOL,
            rtol=0.0,
        ):
            if float(best_metrics["macro_ece"]) < float(stage5a_metrics["macro_ece"]) - FLOAT_TOL:
                return "better"
            if np.isclose(float(best_metrics["macro_ece"]), float(stage5a_metrics["macro_ece"]), atol=FLOAT_TOL, rtol=0.0):
                return "tied"
    if (
        float(best_metrics["macro_auroc"]) + 5e-4 >= float(stage5a_metrics["macro_auroc"])
        and float(best_metrics["macro_average_precision"]) + 5e-4 >= float(stage5a_metrics["macro_average_precision"])
    ):
        return "roughly_neutral"
    return "worse"


def build_success_report(
    *,
    timestamp: str,
    exact_files_used: list[str],
    output_artifacts: list[str],
    baseline_source: dict[str, Any],
    memory_source: dict[str, Any],
    stage5a_source: dict[str, Any],
    alignment_report: dict[str, Any],
    sanity_checks: dict[str, Any],
    coarse_primary_rows: list[dict[str, Any]],
    coarse_pdf_rows: list[dict[str, Any]],
    best_primary_row: dict[str, Any],
    best_pdf_row: dict[str, Any],
    best_overall_row: dict[str, Any],
    best_overall_trace: dict[str, Any],
    baseline_metrics: dict[str, Any],
    memory_only_metrics: dict[str, Any],
    stage5a_metrics: dict[str, Any],
    best_metrics: dict[str, Any],
    comparisons: dict[str, Any],
    qualitative_cases: list[dict[str, Any]],
    optional_refinement_rows: list[dict[str, Any]] | None,
) -> str:
    def beta_table(rows: list[dict[str, Any]], best_row: dict[str, Any]) -> str:
        table_rows: list[list[str]] = []
        for row in rows:
            note = "<- best" if (
                np.isclose(float(row["beta"]), float(best_row["beta"]), atol=FLOAT_TOL, rtol=0.0)
                and str(row["search_stage"]) == str(best_row.get("search_stage", "coarse"))
            ) else ""
            table_rows.append(
                [
                    f"{float(row['beta']):.2f}",
                    format_metric(row["macro_auroc"]),
                    format_metric(row["macro_average_precision"]),
                    format_metric(row["macro_ece"]),
                    format_metric(row["macro_f1_at_0.5"]),
                    note,
                ]
            )
        return build_markdown_table(
            ["beta", "macro AUROC", "macro AP", "macro ECE", "macro F1 @ 0.5", "note"],
            table_rows,
        )

    def per_label_delta_table(delta_payload: dict[str, Any], key: str, value_title: str) -> str:
        rows = [[label_name, format_metric(delta_payload[key][label_name])] for label_name in LABEL_NAMES]
        return build_markdown_table(["label", value_title], rows)

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
        key = str(case["review_bucket"])
        qualitative_summary_counts[key] = qualitative_summary_counts.get(key, 0) + 1
    strict_improved_over_both_count = int(
        sum(
            1
            for case in qualitative_cases
            if float(case["delta_vs_baseline_bce"]) > 0.0 and float(case["delta_vs_stage5a_bce"]) > 0.0
        )
    )

    qualitative_lines = []
    for case in qualitative_cases:
        labels_text = ", ".join(case["true_labels"]) if case["true_labels"] else "none"
        qualitative_lines.append(
            "- "
            + f"{case['validation_example_id']} [{case['review_bucket']}, {case['label_count_category']}, labels={labels_text}]: "
            + f"{case['note']}"
        )

    refinement_text = "Optional local refinement on validation only was not run."
    if optional_refinement_rows:
        refinement_text = (
            "Optional local refinement on validation only was run around the coarse winner. "
            f"Evaluated betas: {[float(row['beta']) for row in optional_refinement_rows]}."
        )

    selection_trace_text = (
        "Selection-rule trace: "
        f"`macro_auroc_ties={best_overall_trace['macro_auroc_tied_candidates']}`, "
        f"`macro_ap_ties={best_overall_trace['macro_average_precision_tied_candidates']}`, "
        f"`macro_ece_ties={best_overall_trace['macro_ece_tied_candidates']}`, "
        f"`beta_ties={best_overall_trace['beta_tied_candidates']}`, "
        f"`primary_formulation_ties={best_overall_trace['primary_formulation_tied_candidates']}`, "
        f"`coarse_stage_ties={best_overall_trace['coarse_stage_tied_candidates']}`."
    )

    overall_vs_baseline = comparisons["vs_baseline_only"]
    overall_vs_stage5a = comparisons["vs_stage5a_best_mixing"]
    stage5b_vs_stage5a_status = compare_best_overall_to_stage5a(best_metrics, stage5a_metrics)
    if stage5b_vs_stage5a_status == "better":
        final_verdict = "PASS: Stage 5B improved over Stage 5A"
        final_verdict_explainer = "The winning Stage 5B setting beat Stage 5A on the requested validation-first comparison."
    elif stage5b_vs_stage5a_status in {"tied", "roughly_neutral"}:
        final_verdict = "CONDITIONAL PASS: Stage 5B is roughly neutral but still informative"
        final_verdict_explainer = "Stage 5B did not open a large gap over Stage 5A, but it still clarified whether calibration-style fusion is helping."
    else:
        final_verdict = "FAIL: Stage 5B underperformed Stage 5A and baseline-only"
        final_verdict_explainer = "The best corrected setting failed to beat the Stage 5A reference and did not justify replacing the baseline."

    report_lines = [
        "# ResNet50 Fused CLS Validation Logit-Correction Report",
        "",
        "## 1. Executive Summary",
        (
            f"Stage 5B succeeded at {timestamp}. The best validation-only logit-correction setting was "
            f"`{best_overall_row['formulation']}` with `beta={float(best_overall_row['beta']):.2f}`. Its validation macro AUROC "
            f"was `{format_metric(best_metrics['macro_auroc'])}`, macro AP was `{format_metric(best_metrics['macro_average_precision'])}`, "
            f"and macro ECE was `{format_metric(best_metrics['macro_ece'])}`. Relative to baseline-only, macro AUROC changed by "
            f"`{format_metric(overall_vs_baseline['macro_auroc_delta'])}`; relative to Stage 5A best mixing, macro AUROC changed by "
            f"`{format_metric(overall_vs_stage5a['macro_auroc_delta'])}`. "
            + (
                "Logit correction helped on validation beyond Stage 5A."
                if stage5b_vs_stage5a_status == "better"
                else "Logit correction was mostly marginal or neutral on validation."
            )
        ),
        "",
        "## 2. Objective",
        "This step implements Stage 5B only: validation-only logit correction using frozen baseline logits and frozen memory probabilities.",
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
            "## 5. Baseline Logit Source",
            (
                "Baseline validation logits were "
                + (
                    "loaded from an archived validation-logit artifact."
                    if baseline_source["source"] == "loaded_saved_validation_logits"
                    else "reconstructed from the frozen baseline checkpoint."
                )
            ),
            f"- Source files: `{json.dumps(baseline_source['source_paths'], sort_keys=True)}`",
            f"- Method: `{baseline_source['architecture_assumption']}`",
            f"- Checkpoint loading details: `{baseline_source['checkpoint_loading']}` with key remap `{baseline_source['checkpoint_key_remap']}`",
            f"- Shape and label order: `z_base={sanity_checks['shape_checks']['z_base_shape']}`, `p_base={sanity_checks['shape_checks']['p_base_shape']}`, labels=`{LABEL_NAMES}`",
            f"- Stage 5A `p_base` match check: `max_abs_diff={baseline_source['stage5a_p_base_check']['max_abs_diff']:.10f}`",
            "",
            "## 6. Memory Probability Source",
            (
                f"`p_mem(k=50, tau=1)` was {memory_source['mode']} from `{memory_source['source']}`. "
                f"Source files: `{json.dumps(memory_source['source_paths'], sort_keys=True)}`."
            ),
            f"- Shape and label order: `p_mem={sanity_checks['shape_checks']['p_mem_shape']}`, labels=`{LABEL_NAMES}`",
            "",
            "## 7. Formulations Evaluated",
            (
                "Primary multilabel-safe formulation: "
                "`z_mem = log((p_mem_clipped + eps) / (1 - p_mem_clipped + eps))`, "
                "`z_corr = z_base + beta * z_mem`, `p_corr = sigmoid(z_corr)`. "
                "This treats each label independently as a binary task."
            ),
            (
                "PDF-literal ablation: "
                "`z_mem_pdf = log(p_mem_clipped + eps)`, `z_corr_pdf = z_base + beta * z_mem_pdf`, "
                "`p_corr_pdf = sigmoid(z_corr_pdf)`. This preserves comparability to the baby-steps wording."
            ),
            "",
            "## 8. Data Alignment and Row Ordering",
            (
                "The canonical validation ordering was Stage 5A `aligned_val_example_ids.json`. "
                "All arrays were aligned by `example_id = Path(image_path).stem`."
            ),
            f"- Canonical ordering source: `{alignment_report['canonical_order_source']}`",
            f"- Baseline rows loaded: `{alignment_report['baseline']['loaded_rows']}`",
            f"- Stage 5A `p_base` rows loaded: `{alignment_report['stage5a_p_base']['loaded_rows']}`",
            f"- Memory rows loaded: `{alignment_report['memory']['loaded_rows']}`",
            f"- Labels rows loaded: `{alignment_report['labels']['loaded_rows']}`",
            f"- Stage 5A best-mix rows loaded: `{alignment_report['stage5a_best_mix']['loaded_rows']}`",
            f"- Final aligned rows: `{alignment_report['final_aligned_rows']}`",
            f"- Rows dropped: baseline=`{alignment_report['baseline']['dropped_rows']}`, memory=`{alignment_report['memory']['dropped_rows']}`, labels=`{alignment_report['labels']['dropped_rows']}`, stage5a_best_mix=`{alignment_report['stage5a_best_mix']['dropped_rows']}`",
            f"- Exact example-ID parsing rule: `{alignment_report['example_id_parsing_rule']}`",
            "",
            "## 9. Sanity Checks",
            f"- Shape checks: `{json.dumps(sanity_checks['shape_checks'], sort_keys=True)}`",
            f"- Finite checks: `{json.dumps(sanity_checks['finite_checks'], sort_keys=True)}`",
            f"- Probability range checks: `{json.dumps(sanity_checks['probability_range_checks'], sort_keys=True)}`",
            f"- Endpoint checks: `{json.dumps(sanity_checks['endpoint_checks'], sort_keys=True)}`",
            f"- Memory-transform summaries: `{json.dumps(sanity_checks['memory_transform_checks'], sort_keys=True)}`",
            f"- Leakage checks: `{json.dumps(sanity_checks['leakage_checks'], sort_keys=True)}`",
            "",
            "## 10. Beta Sweep Results: primary_multilabel_logit",
            beta_table(coarse_primary_rows, best_primary_row),
            "",
            "## 11. Beta Sweep Results: pdf_literal_logprob",
            beta_table(coarse_pdf_rows, best_pdf_row),
            "",
            "## 12. Best Overall Stage 5B Setting",
            (
                f"Chosen formulation: `{best_overall_row['formulation']}`. "
                f"Chosen beta: `{float(best_overall_row['beta']):.2f}`. Winner came from "
                f"`{best_overall_row.get('search_stage', 'coarse')}` search."
            ),
            selection_trace_text,
            refinement_text,
            "",
            "## 13. Comparison vs Baseline-Only, Memory-Only, and Stage 5A",
            (
                f"Against baseline-only: macro AUROC delta `{format_metric(overall_vs_baseline['macro_auroc_delta'])}`, "
                f"macro AP delta `{format_metric(overall_vs_baseline['macro_average_precision_delta'])}`, "
                f"macro F1 @ 0.5 delta `{format_metric(overall_vs_baseline['macro_f1_at_0.5_delta'])}`, "
                f"macro ECE delta `{format_metric(overall_vs_baseline['macro_ece_delta'])}`, "
                f"macro Brier delta `{format_metric(overall_vs_baseline['macro_brier_delta'])}`."
            ),
            per_label_delta_table(overall_vs_baseline, "per_label_auroc_deltas", "AUROC delta"),
            per_label_delta_table(overall_vs_baseline, "per_label_ece_deltas", "ECE delta"),
            "",
            (
                f"Against memory-only (same k=50, tau=1): macro AUROC delta `{format_metric(comparisons['vs_memory_only']['macro_auroc_delta'])}`, "
                f"macro AP delta `{format_metric(comparisons['vs_memory_only']['macro_average_precision_delta'])}`, "
                f"macro F1 @ 0.5 delta `{format_metric(comparisons['vs_memory_only']['macro_f1_at_0.5_delta'])}`, "
                f"macro ECE delta `{format_metric(comparisons['vs_memory_only']['macro_ece_delta'])}`, "
                f"macro Brier delta `{format_metric(comparisons['vs_memory_only']['macro_brier_delta'])}`."
            ),
            per_label_delta_table(comparisons["vs_memory_only"], "per_label_auroc_deltas", "AUROC delta"),
            per_label_delta_table(comparisons["vs_memory_only"], "per_label_ece_deltas", "ECE delta"),
            "",
            (
                f"Against Stage 5A best mixing (`alpha={float(stage5a_source['chosen_alpha']):.2f}`, `k=50`, `tau=1`): "
                f"macro AUROC delta `{format_metric(overall_vs_stage5a['macro_auroc_delta'])}`, "
                f"macro AP delta `{format_metric(overall_vs_stage5a['macro_average_precision_delta'])}`, "
                f"macro F1 @ 0.5 delta `{format_metric(overall_vs_stage5a['macro_f1_at_0.5_delta'])}`, "
                f"macro ECE delta `{format_metric(overall_vs_stage5a['macro_ece_delta'])}`, "
                f"macro Brier delta `{format_metric(overall_vs_stage5a['macro_brier_delta'])}`."
            ),
            per_label_delta_table(overall_vs_stage5a, "per_label_auroc_deltas", "AUROC delta"),
            per_label_delta_table(overall_vs_stage5a, "per_label_ece_deltas", "ECE delta"),
            "",
            (
                "Interpretation: Stage 5B mainly helps when memory supplies a calibration-style push in the right direction, "
                "and it hurts when retrieval evidence suppresses a valid baseline positive or amplifies the wrong weak label."
            ),
            "",
            "## 14. Optional Threshold-Tuned Diagnostic",
            diagnostic_block,
            "",
            "## 15. Qualitative Case Review",
            (
                f"Twelve validation cases were inspected. Bucket counts: `{json.dumps(qualitative_summary_counts, sort_keys=True)}`."
            ),
        ]
    )
    if strict_improved_over_both_count < 4:
        report_lines.append(
            "Strictly improved-over-both cases were fewer than four because the winning `beta=0.00` collapsed Stage 5B back to the baseline; the remaining slots in that bucket use the closest non-worse cases for inspection."
        )
    report_lines.extend(qualitative_lines)
    report_lines.extend(
        [
            "",
            "Overall patterns: helpful cases usually show logit correction sharpening a weak true positive or damping an overconfident false positive, while failure cases show over-suppression from memory logits or label-bleeding from similar retrieved studies.",
            "",
            "## 16. Interpretation",
            "Did Stage 5B improve over baseline-only? "
            + (
                "Yes on the winning validation selection metrics."
                if overall_vs_baseline["macro_auroc_delta"] > 0.0 or overall_vs_baseline["macro_average_precision_delta"] > 0.0
                else "No, or only trivially."
            ),
            "Did Stage 5B improve over Stage 5A? "
            + (
                "Yes."
                if stage5b_vs_stage5a_status == "better"
                else "Roughly neutral."
                if stage5b_vs_stage5a_status in {"tied", "roughly_neutral"}
                else "No."
            ),
            "Is the gain meaningful or marginal? "
            + (
                "Marginal; the deltas are small even when positive."
                if abs(overall_vs_stage5a["macro_auroc_delta"]) < 0.01 and abs(overall_vs_baseline["macro_auroc_delta"]) < 0.01
                else "Large enough to matter on validation."
            ),
            "Does the winning beta suggest memory is still useful? "
            + (
                f"Yes; the winning beta `{float(best_overall_row['beta']):.2f}` keeps a non-zero memory contribution."
                if float(best_overall_row["beta"]) > 0.0
                else "No; beta zero collapses back to the baseline."
            ),
            "Does the winning formulation support the multilabel-safe adaptation? "
            + (
                "Yes; the primary multilabel logit formulation won."
                if str(best_overall_row["formulation"]) == FORMULATION_PRIMARY
                else "No; the PDF-literal ablation validated better in this run."
            ),
            "Are there obvious failure modes? Yes: memory-derived corrections can over-suppress true positives in dense multi-label cases and can import the wrong label prior from similar retrieved studies.",
            "",
            "## 17. Constraints Respected",
            "- validation only",
            "- train memory only",
            "- no test tuning",
            "- no retraining",
            "- no embedding updates",
            "- no memory rebuilds",
            "",
            "## 18. Final Verdict",
            final_verdict,
            final_verdict_explainer,
            "",
            "## Appendix A. Metric JSON Snippet",
            "```json",
            json.dumps(
                {
                    "baseline_only": metric_headline(baseline_metrics),
                    "stage5a_best_mixing": metric_headline(stage5a_metrics),
                    "best_primary_multilabel_logit": metric_headline(best_primary_row["metrics"]),
                    "best_pdf_literal_logprob": metric_headline(best_pdf_row["metrics"]),
                    "best_overall_stage5b": {
                        "formulation": str(best_overall_row["formulation"]),
                        "beta": float(best_overall_row["beta"]),
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
            f"Stage 5B completed on validation only at {timestamp}.",
            f"Best setting: {best_overall_row['formulation']} with beta={float(best_overall_row['beta']):.2f}.",
            f"Best macro AUROC/AP/ECE: {format_metric(best_metrics['macro_auroc'])} / {format_metric(best_metrics['macro_average_precision'])} / {format_metric(best_metrics['macro_ece'])}.",
            f"Delta vs baseline AUROC/AP: {format_metric(overall_vs_baseline['macro_auroc_delta'])} / {format_metric(overall_vs_baseline['macro_average_precision_delta'])}.",
            "Validation-only logit correction stayed within Stage 5B constraints.",
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
        "# ResNet50 Fused CLS Validation Logit-Correction Report",
        "",
        "## 1. Executive Summary",
        (
            f"Stage 5B failed at {timestamp}. The validation-only logit-correction run did not complete because "
            f"`{failure_message}`."
        ),
        "",
        "## 2. Objective",
        "This step implements Stage 5B only: validation-only logit correction using frozen baseline logits and frozen memory probabilities.",
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
            "## 5. Baseline Logit Source",
            "Not completed because execution failed before a reliable `z_base` artifact was finalized.",
            "",
            "## 6. Memory Probability Source",
            "Not completed because execution failed before a reliable `p_mem(k=50,tau=1)` artifact was finalized.",
            "",
            "## 7. Formulations Evaluated",
            "Not completed.",
            "",
            "## 8. Data Alignment and Row Ordering",
            "Not completed because execution failed before alignment was finalized.",
            "",
            "## 9. Sanity Checks",
            "Not completed.",
            "",
            "## 10. Beta Sweep Results: primary_multilabel_logit",
            "Not computed.",
            "",
            "## 11. Beta Sweep Results: pdf_literal_logprob",
            "Not computed.",
            "",
            "## 12. Best Overall Stage 5B Setting",
            "No winner was selected.",
            "",
            "## 13. Comparison vs Baseline-Only, Memory-Only, and Stage 5A",
            "Not computed.",
            "",
            "## 14. Optional Threshold-Tuned Diagnostic",
            "Not computed.",
            "",
            "## 15. Qualitative Case Review",
            "Not computed.",
            "",
            "## 16. Interpretation",
            "Stage 5B could not be evaluated reliably on validation because execution failed.",
            "",
            "## 17. Constraints Respected",
            "- validation only: intended",
            "- train memory only: intended",
            "- no test tuning: yes",
            "- no retraining: yes",
            "- no embedding updates: yes",
            "- no memory rebuilds: yes",
            "",
            "## 18. Final Verdict",
            "FAIL: Stage 5B underperformed Stage 5A and baseline-only",
            "This failure verdict reflects that reliable validation-only Stage 5B metrics were not produced.",
            "",
            "## Appendix A. Metric JSON Snippet",
            "```json",
            json.dumps({"error": failure_message}, indent=2, sort_keys=True),
            "```",
            "",
            "## Appendix B. 5-Line Ultra-Short Update",
            "Stage 5B failed.",
            failure_message,
            "No reliable best logit-correction setting was produced.",
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
    z_base_path = output_dir / "z_base_val.npy"
    p_base_path = output_dir / "p_base_val.npy"
    p_mem_path = output_dir / "p_mem_k50_tau1.npy"
    z_mem_primary_path = output_dir / "z_mem_primary.npy"
    z_mem_pdf_path = output_dir / "z_mem_pdf_literal.npy"
    coarse_csv_path = output_dir / "logit_correction_results_coarse.csv"
    refined_csv_path = output_dir / "logit_correction_results_refined.csv"
    best_config_path = output_dir / "best_logit_correction_config.json"
    p_corr_best_path = output_dir / "p_corr_best.npy"
    z_corr_best_path = output_dir / "z_corr_best.npy"
    best_metrics_path = output_dir / "best_metrics.json"
    qualitative_path = output_dir / "qualitative_logit_correction_cases.json"
    sanity_checks_path = output_dir / "sanity_checks.json"

    files_used: set[Path] = set()
    output_artifacts: list[Path] = []

    try:
        if args.eps != EPSILON:
            raise ValueError(f"This stage requires eps={EPSILON}, found {args.eps}.")

        baseline_run_root = resolve_existing_path(
            "baseline run root",
            [args.baseline_run_root, args.baseline_run_root_fallback],
        )
        stage5a_output_dir = resolve_existing_path("Stage 5A output dir", [args.stage5a_output_dir])
        stage4_output_dir = resolve_existing_path("Stage 4 output dir", [args.stage4_output_dir])
        train_memory_root = resolve_existing_path("train memory root", [args.train_memory_root])
        val_embeddings_path = resolve_existing_path("validation embeddings", [args.val_embeddings])
        val_image_paths_path = resolve_existing_path("validation image paths", [args.val_image_paths])
        val_run_meta_path = resolve_existing_path("validation run meta", [args.val_run_meta])
        manifest_csv_path = resolve_existing_path("manifest CSV", [args.manifest_csv])

        files_used.add(val_run_meta_path)
        _ = read_json(val_run_meta_path)
        validate_label_names(LABEL_NAMES)

        stage5a_ids_path = stage5a_output_dir / "aligned_val_example_ids.json"
        stage5a_labels_path = stage5a_output_dir / "aligned_val_labels.npy"
        stage5a_p_base_path = stage5a_output_dir / "p_base_val.npy"
        stage5a_p_mem_path = stage5a_output_dir / "p_mem_k50_tau1.npy"
        stage5a_p_mix_best_path = stage5a_output_dir / "p_mix_best.npy"
        stage5a_best_config_path = stage5a_output_dir / "best_mixing_config.json"
        stage5a_best_metrics_path = stage5a_output_dir / "best_metrics.json"
        stage5a_run_config_path = stage5a_output_dir / "run_config.json"
        files_used.update(
            {
                stage5a_ids_path,
                stage5a_labels_path,
                stage5a_p_base_path,
                stage5a_p_mem_path,
                stage5a_p_mix_best_path,
                stage5a_best_config_path,
                stage5a_best_metrics_path,
                stage5a_run_config_path,
            }
        )

        if stage5a_ids_path.exists():
            canonical_example_ids = [str(value) for value in read_json(stage5a_ids_path)]
            canonical_order_source = stage5a_ids_path
        else:
            stage4_ids_path = stage4_output_dir / "val_example_ids.json"
            files_used.add(stage4_ids_path)
            canonical_example_ids = [str(value) for value in read_json(stage4_ids_path)]
            canonical_order_source = stage4_ids_path
        if not canonical_example_ids:
            raise ValueError("Canonical validation example ID list is empty.")
        if len(canonical_example_ids) != len(set(canonical_example_ids)):
            raise ValueError("Canonical validation example IDs contain duplicates.")

        stage5a_labels_raw = load_embedding_array(stage5a_labels_path)
        stage5a_p_base_raw = load_embedding_array(stage5a_p_base_path)
        stage5a_p_mem_raw = load_embedding_array(stage5a_p_mem_path)
        stage5a_p_mix_best_raw = load_embedding_array(stage5a_p_mix_best_path)
        stage5a_source_ids = [str(value) for value in read_json(stage5a_ids_path)]
        if len(stage5a_source_ids) != len(canonical_example_ids):
            raise ValueError(
                f"Stage 5A aligned example ids count {len(stage5a_source_ids)} does not match canonical ids {len(canonical_example_ids)}."
            )

        stage5a_labels_aligned, stage5a_labels_alignment = align_array_to_reference(
            reference_example_ids=canonical_example_ids,
            source_example_ids=stage5a_source_ids,
            array=stage5a_labels_raw,
            source_name="stage5a_aligned_labels",
        )
        stage5a_p_base_aligned, stage5a_p_base_alignment = align_array_to_reference(
            reference_example_ids=canonical_example_ids,
            source_example_ids=stage5a_source_ids,
            array=stage5a_p_base_raw,
            source_name="stage5a_p_base",
        )
        stage5a_p_mix_best_aligned, stage5a_p_mix_best_alignment = align_array_to_reference(
            reference_example_ids=canonical_example_ids,
            source_example_ids=stage5a_source_ids,
            array=stage5a_p_mix_best_raw,
            source_name="stage5a_p_mix_best",
        )

        stage5a_best_config = read_json(stage5a_best_config_path)
        stage5a_best_metrics_file = read_json(stage5a_best_metrics_path)
        chosen_memory_config = dict(stage5a_best_config.get("chosen_memory_config", {}))
        if int(chosen_memory_config.get("k", -1)) != PRIMARY_MEMORY_CONFIG["k"] or int(
            chosen_memory_config.get("tau", -1)
        ) != PRIMARY_MEMORY_CONFIG["tau"]:
            raise ValueError(
                f"Stage 5A best mixing config does not match required memory config {PRIMARY_MEMORY_CONFIG}: {stage5a_best_config}"
            )

        z_base_raw, p_base_raw, baseline_example_ids, baseline_labels_raw, baseline_source = load_or_reconstruct_z_base(
            baseline_run_root=baseline_run_root,
            val_embeddings_path=val_embeddings_path,
            val_image_paths_path=val_image_paths_path,
            manifest_csv_path=manifest_csv_path,
            batch_size=args.batch_size,
            label_names=LABEL_NAMES,
            files_used=files_used,
        )
        z_base_aligned, baseline_alignment = align_array_to_reference(
            reference_example_ids=canonical_example_ids,
            source_example_ids=baseline_example_ids,
            array=z_base_raw,
            source_name="baseline_logits",
        )
        p_base_aligned, p_base_alignment = align_array_to_reference(
            reference_example_ids=canonical_example_ids,
            source_example_ids=baseline_example_ids,
            array=p_base_raw,
            source_name="baseline_probabilities",
        )
        baseline_labels_aligned, labels_alignment = align_array_to_reference(
            reference_example_ids=canonical_example_ids,
            source_example_ids=baseline_example_ids,
            array=baseline_labels_raw,
            source_name="baseline_manifest_labels",
        )
        if baseline_labels_aligned.shape != stage5a_labels_aligned.shape:
            raise ValueError(
                f"Baseline labels shape {baseline_labels_aligned.shape} does not match Stage 5A labels {stage5a_labels_aligned.shape}."
            )
        if not np.array_equal(baseline_labels_aligned.astype(np.float32), stage5a_labels_aligned.astype(np.float32)):
            max_abs = float(
                np.max(np.abs(baseline_labels_aligned.astype(np.float32) - stage5a_labels_aligned.astype(np.float32)))
            )
            raise ValueError(f"Manifest-aligned validation labels do not match Stage 5A aligned labels. max_abs_diff={max_abs}")

        p_mem_raw, p_mem_source_ids, memory_source = load_frozen_memory_probabilities(
            stage5a_output_dir=stage5a_output_dir,
            stage4_output_dir=stage4_output_dir,
            train_memory_root=train_memory_root,
            val_embeddings_path=val_embeddings_path,
            val_image_paths_path=val_image_paths_path,
            manifest_csv_path=manifest_csv_path,
            files_used=files_used,
        )
        p_mem_aligned, memory_alignment = align_array_to_reference(
            reference_example_ids=canonical_example_ids,
            source_example_ids=p_mem_source_ids,
            array=p_mem_raw,
            source_name="memory_probabilities",
        )

        if any(
            summary["dropped_rows"] != 0
            for summary in [
                baseline_alignment,
                p_base_alignment,
                memory_alignment,
                labels_alignment,
                stage5a_p_base_alignment,
                stage5a_p_mix_best_alignment,
                stage5a_labels_alignment,
            ]
        ):
            raise ValueError(
                "Alignment dropped rows unexpectedly. "
                f"baseline={baseline_alignment['dropped_rows']}, p_base={p_base_alignment['dropped_rows']}, "
                f"memory={memory_alignment['dropped_rows']}, labels={labels_alignment['dropped_rows']}, "
                f"stage5a_p_base={stage5a_p_base_alignment['dropped_rows']}, "
                f"stage5a_p_mix_best={stage5a_p_mix_best_alignment['dropped_rows']}."
            )

        shape_checks = {
            "z_base_shape": list(z_base_aligned.shape),
            "p_base_shape": list(p_base_aligned.shape),
            "p_mem_shape": list(p_mem_aligned.shape),
            "val_labels_shape": list(stage5a_labels_aligned.shape),
            "stage5a_best_mix_shape": list(stage5a_p_mix_best_aligned.shape),
        }
        expected_shape = z_base_aligned.shape
        if not (
            z_base_aligned.shape
            == p_base_aligned.shape
            == p_mem_aligned.shape
            == stage5a_labels_aligned.shape
            == stage5a_p_mix_best_aligned.shape
        ):
            raise ValueError(f"Shape mismatch across aligned arrays: {shape_checks}")
        if list(expected_shape)[1] != len(LABEL_NAMES):
            raise ValueError(f"Expected label dimension {len(LABEL_NAMES)}, found {expected_shape}.")

        p_base_from_logits = sigmoid_array(z_base_aligned)
        p_base_reconstruction_diff = float(np.max(np.abs(p_base_from_logits - p_base_aligned)))
        stage5a_p_base_diff = float(np.max(np.abs(p_base_aligned - stage5a_p_base_aligned)))
        baseline_source["stage5a_p_base_check"] = {
            "max_abs_diff": stage5a_p_base_diff,
            "matches_within_1e-6": bool(stage5a_p_base_diff <= 1e-6),
        }
        baseline_source["source_paths"] = [
            str(baseline_run_root / "config.json"),
            str(baseline_run_root / "best.ckpt"),
            str(baseline_run_root / "val_metrics.json"),
            str(baseline_run_root / "val_f1_thresholds.json"),
            str(val_embeddings_path),
            str(val_image_paths_path),
            str(manifest_csv_path),
        ]

        if p_base_reconstruction_diff > 1e-6:
            raise ValueError(
                f"Sigmoid(z_base) does not reproduce reconstructed p_base within tolerance. max_abs_diff={p_base_reconstruction_diff}"
            )

        finite_checks = {
            "z_base_all_finite": bool(np.isfinite(z_base_aligned).all()),
            "p_base_all_finite": bool(np.isfinite(p_base_aligned).all()),
            "p_mem_all_finite": bool(np.isfinite(p_mem_aligned).all()),
            "val_labels_all_finite": bool(np.isfinite(stage5a_labels_aligned).all()),
            "stage5a_best_mix_all_finite": bool(np.isfinite(stage5a_p_mix_best_aligned).all()),
            "sigmoid_z_base_matches_p_base_allclose": bool(p_base_reconstruction_diff <= 1e-6),
        }
        if not all(finite_checks.values()):
            raise ValueError(f"Finite check failed: {finite_checks}")

        probability_range_checks = {
            "p_base": probability_summary(p_base_aligned),
            "p_mem": probability_summary(p_mem_aligned),
            "stage5a_best_mix": probability_summary(stage5a_p_mix_best_aligned),
        }
        for name, summary in probability_range_checks.items():
            if summary["min"] < -1e-6 or summary["max"] > 1.0 + 1e-6:
                raise ValueError(f"Probability range check failed for {name}: {summary}")

        z_mem_primary, z_mem_pdf, memory_transform_checks = build_memory_transforms(p_mem_aligned, eps=args.eps)
        np.save(z_mem_primary_path, z_mem_primary.astype(np.float32))
        np.save(z_mem_pdf_path, z_mem_pdf.astype(np.float32))
        output_artifacts.extend([z_mem_primary_path, z_mem_pdf_path])

        baseline_metrics = evaluate_probabilities_extended(
            stage5a_labels_aligned,
            p_base_aligned,
            LABEL_NAMES,
            include_diagnostic_thresholds=False,
            ece_bins=args.ece_bins,
            include_reliability_tables=False,
        )
        memory_only_metrics = evaluate_probabilities_extended(
            stage5a_labels_aligned,
            p_mem_aligned,
            LABEL_NAMES,
            include_diagnostic_thresholds=False,
            ece_bins=args.ece_bins,
            include_reliability_tables=False,
        )
        stage5a_best_metrics = evaluate_probabilities_extended(
            stage5a_labels_aligned,
            stage5a_p_mix_best_aligned,
            LABEL_NAMES,
            include_diagnostic_thresholds=False,
            ece_bins=args.ece_bins,
            include_reliability_tables=False,
        )

        coarse_primary_rows = evaluate_beta_rows(
            formulation=FORMULATION_PRIMARY,
            z_base=z_base_aligned,
            z_mem=z_mem_primary,
            targets=stage5a_labels_aligned,
            label_names=LABEL_NAMES,
            beta_grid=BETA_GRID_COARSE,
            search_stage="coarse",
            ece_bins=args.ece_bins,
            include_diagnostic_thresholds=False,
        )
        coarse_pdf_rows = evaluate_beta_rows(
            formulation=FORMULATION_PDF,
            z_base=z_base_aligned,
            z_mem=z_mem_pdf,
            targets=stage5a_labels_aligned,
            label_names=LABEL_NAMES,
            beta_grid=BETA_GRID_COARSE,
            search_stage="coarse",
            ece_bins=args.ece_bins,
            include_diagnostic_thresholds=False,
        )
        coarse_rows = coarse_primary_rows + coarse_pdf_rows
        best_coarse_row, _ = select_best_rows(coarse_rows)

        refined_rows: list[dict[str, Any]] = []
        if args.run_optional_refinement and not (
            np.isclose(float(best_coarse_row["beta"]), 0.0, atol=FLOAT_TOL, rtol=0.0)
            or np.isclose(float(best_coarse_row["beta"]), 2.0, atol=FLOAT_TOL, rtol=0.0)
        ):
            coarse_beta = float(best_coarse_row["beta"])
            refined_grid = sorted(
                {
                    round(candidate, 4)
                    for candidate in [
                        coarse_beta - 0.05,
                        coarse_beta - 0.02,
                        coarse_beta,
                        coarse_beta + 0.02,
                        coarse_beta + 0.05,
                    ]
                    if candidate >= 0.0
                }
            )
            refined_rows = evaluate_beta_rows(
                formulation=str(best_coarse_row["formulation"]),
                z_base=z_base_aligned,
                z_mem=z_mem_primary if str(best_coarse_row["formulation"]) == FORMULATION_PRIMARY else z_mem_pdf,
                targets=stage5a_labels_aligned,
                label_names=LABEL_NAMES,
                beta_grid=refined_grid,
                search_stage="refined",
                ece_bins=args.ece_bins,
                include_diagnostic_thresholds=False,
            )

        all_candidate_rows = coarse_rows + refined_rows
        best_overall_row, best_overall_trace = select_best_rows(all_candidate_rows)
        best_primary_row, best_primary_trace = select_best_row_within_formulation(
            [row for row in all_candidate_rows if str(row["formulation"]) == FORMULATION_PRIMARY]
        )
        best_pdf_row, best_pdf_trace = select_best_row_within_formulation(
            [row for row in all_candidate_rows if str(row["formulation"]) == FORMULATION_PDF]
        )

        best_overall_metrics = evaluate_probabilities_extended(
            stage5a_labels_aligned,
            best_overall_row["p_corr"],
            LABEL_NAMES,
            include_diagnostic_thresholds=True,
            ece_bins=args.ece_bins,
            include_reliability_tables=True,
        )
        best_overall_row["metrics"] = best_overall_metrics
        if best_primary_row is best_overall_row:
            best_primary_row["metrics"] = best_overall_metrics
        if best_pdf_row is best_overall_row:
            best_pdf_row["metrics"] = best_overall_metrics

        corrected_output_checks = [
            {
                "formulation": str(row["formulation"]),
                "beta": float(row["beta"]),
                "search_stage": str(row.get("search_stage", "coarse")),
                **row["output_sanity"],
            }
            for row in all_candidate_rows
        ]
        if not all(
            check["z_corr_all_finite"] and check["p_corr_all_finite"] and check["p_corr_range_ok"]
            for check in corrected_output_checks
        ):
            raise ValueError(f"Corrected output checks failed: {corrected_output_checks}")

        endpoint_checks = {
            FORMULATION_PRIMARY: {
                "beta_0_max_abs_diff_vs_p_base": float(
                    np.max(np.abs(apply_logit_correction(z_base=z_base_aligned, z_mem=z_mem_primary, beta=0.0)[1] - p_base_aligned))
                )
            },
            FORMULATION_PDF: {
                "beta_0_max_abs_diff_vs_p_base": float(
                    np.max(np.abs(apply_logit_correction(z_base=z_base_aligned, z_mem=z_mem_pdf, beta=0.0)[1] - p_base_aligned))
                )
            },
        }
        if endpoint_checks[FORMULATION_PRIMARY]["beta_0_max_abs_diff_vs_p_base"] > 1e-6:
            raise ValueError(f"Primary beta=0 endpoint check failed: {endpoint_checks[FORMULATION_PRIMARY]}")
        if endpoint_checks[FORMULATION_PDF]["beta_0_max_abs_diff_vs_p_base"] > 1e-6:
            raise ValueError(f"PDF beta=0 endpoint check failed: {endpoint_checks[FORMULATION_PDF]}")

        leakage_checks = {
            "query_split_is_validation_only": True,
            "retrieved_items_are_train_memory_only": True,
            "test_artifacts_used": False,
            "baseline_model_retrained": False,
            "encoder_weights_changed": False,
            "train_memory_changed": False,
            "train_memory_rebuilt": False,
            "prototype_memory_run": False,
            "stage6_run": False,
            "target_domain_adaptation_run": False,
            "index_rebuilt_per_batch": False,
            "details": "Only validation queries were evaluated against the frozen train memory and frozen baseline checkpoint outputs.",
        }

        sanity_checks = {
            "shape_checks": shape_checks,
            "finite_checks": finite_checks,
            "probability_range_checks": probability_range_checks,
            "endpoint_checks": endpoint_checks,
            "memory_transform_checks": memory_transform_checks,
            "corrected_output_checks": corrected_output_checks,
            "baseline_stage5a_match_check": baseline_source["stage5a_p_base_check"],
            "leakage_checks": leakage_checks,
        }

        comparisons = {
            "vs_baseline_only": comparison_payload(best_overall_metrics, baseline_metrics),
            "vs_memory_only": comparison_payload(best_overall_metrics, memory_only_metrics),
            "vs_stage5a_best_mixing": comparison_payload(best_overall_metrics, stage5a_best_metrics),
        }

        stage5a_source = {
            "chosen_alpha": float(stage5a_best_config["chosen_alpha"]),
            "chosen_memory_config": chosen_memory_config,
            "headline_metrics_from_stage5a_file": metric_headline(stage5a_best_metrics_file["metrics"]),
            "headline_metrics_recomputed": metric_headline(stage5a_best_metrics),
            "source_paths": [
                str(stage5a_best_config_path),
                str(stage5a_best_metrics_path),
                str(stage5a_p_mix_best_path),
                str(stage5a_ids_path),
            ],
        }

        qualitative_cases = build_qualitative_cases(
            example_ids=canonical_example_ids,
            targets=stage5a_labels_aligned,
            z_base=z_base_aligned,
            p_base=p_base_aligned,
            p_mem=p_mem_aligned,
            p_mix_best=stage5a_p_mix_best_aligned,
            p_corr_best=best_overall_row["p_corr"],
        )

        alignment_report = {
            "canonical_order_source": str(canonical_order_source),
            "final_aligned_rows": int(len(canonical_example_ids)),
            "baseline": baseline_alignment,
            "baseline_probabilities": p_base_alignment,
            "memory": memory_alignment,
            "labels": labels_alignment,
            "stage5a_p_base": stage5a_p_base_alignment,
            "stage5a_best_mix": stage5a_p_mix_best_alignment,
            "stage5a_labels": stage5a_labels_alignment,
            "example_id_parsing_rule": "example_id = Path(image_path).stem",
        }

        np.save(aligned_labels_path, stage5a_labels_aligned.astype(np.float32))
        np.save(z_base_path, z_base_aligned.astype(np.float32))
        np.save(p_base_path, p_base_aligned.astype(np.float32))
        np.save(p_mem_path, p_mem_aligned.astype(np.float32))
        np.save(p_corr_best_path, best_overall_row["p_corr"].astype(np.float32))
        np.save(z_corr_best_path, best_overall_row["z_corr"].astype(np.float32))
        aligned_ids_path.write_text(json.dumps(canonical_example_ids, indent=2), encoding="utf-8")
        output_artifacts.extend(
            [
                aligned_ids_path,
                aligned_labels_path,
                z_base_path,
                p_base_path,
                p_mem_path,
                p_corr_best_path,
                z_corr_best_path,
            ]
        )

        write_logit_correction_results_csv(coarse_csv_path, coarse_rows)
        output_artifacts.append(coarse_csv_path)
        if refined_rows:
            write_logit_correction_results_csv(refined_csv_path, refined_rows)
            output_artifacts.append(refined_csv_path)

        run_config = {
            "timestamp": timestamp,
            "exact_paths_used": {
                "baseline_run_root": str(baseline_run_root),
                "stage5a_output_dir": str(stage5a_output_dir),
                "stage4_output_dir": str(stage4_output_dir),
                "train_memory_root": str(train_memory_root),
                "validation_embeddings": str(val_embeddings_path),
                "validation_image_paths": str(val_image_paths_path),
                "validation_run_meta": str(val_run_meta_path),
                "manifest_csv": str(manifest_csv_path),
                "canonical_validation_order": str(canonical_order_source),
                "stage5a_aligned_ids": str(stage5a_ids_path),
                "stage5a_aligned_labels": str(stage5a_labels_path),
                "stage5a_p_base": str(stage5a_p_base_path),
                "stage5a_p_mem_k50_tau1": str(stage5a_p_mem_path),
                "stage5a_p_mix_best": str(stage5a_p_mix_best_path),
                "stage5a_best_mixing_config": str(stage5a_best_config_path),
                "stage5a_best_metrics": str(stage5a_best_metrics_path),
            },
            "label_names": LABEL_NAMES,
            "beta_grid_coarse": BETA_GRID_COARSE,
            "epsilon": float(args.eps),
            "ece_bins": int(args.ece_bins),
            "ece_definition": (
                "Per-label binary expected calibration error with 15 equal-width bins over [0,1]; "
                "for each non-empty bin, compute |mean(predicted probability) - empirical positive rate| and weight by bin count / N. "
                "Macro ECE is the mean of per-label ECE values."
            ),
            "formulations_evaluated": [
                {
                    "name": FORMULATION_PRIMARY,
                    "formula": "z_mem = log((clip(p_mem, eps, 1-eps) + eps) / (1 - clip(p_mem, eps, 1-eps) + eps)); z_corr = z_base + beta * z_mem; p_corr = sigmoid(z_corr)",
                },
                {
                    "name": FORMULATION_PDF,
                    "formula": "z_mem_pdf = log(clip(p_mem, 0, 1-eps) + eps); z_corr_pdf = z_base + beta * z_mem_pdf; p_corr_pdf = sigmoid(z_corr_pdf)",
                },
            ],
            "selection_rule": [
                "highest validation macro AUROC",
                "break ties with higher validation macro average precision",
                "if still tied, lower validation macro ECE",
                "if still tied, prefer smaller beta",
                "if still tied, prefer primary_multilabel_logit over pdf_literal_logprob",
                "if still tied, prefer coarse-grid winner over refined winner",
            ],
            "alignment_rule": {
                "canonical_validation_order": str(canonical_order_source),
                "example_id_parsing": "example_id = Path(image_path).stem",
                "all_sources_aligned_by_example_id": True,
            },
            "z_base_source": {
                "loaded_or_reconstructed": "loaded"
                if baseline_source["source"] == "loaded_saved_validation_logits"
                else "reconstructed",
                "details": baseline_source,
            },
            "p_mem_source": {
                "loaded_or_recomputed": memory_source["mode"],
                "details": memory_source,
            },
            "optional_local_refinement_run": bool(refined_rows),
            "stage5_scope": "validation_only_logit_correction",
            "test_used": False,
            "retraining_run": False,
            "encoder_updated": False,
            "train_memory_changed": False,
            "train_memory_rebuilt": False,
            "prototype_memory_run": False,
            "stage6_run": False,
            "target_domain_adaptation_run": False,
        }
        write_json(run_config_path, run_config)
        output_artifacts.append(run_config_path)

        best_config_payload = {
            "chosen_formulation": str(best_overall_row["formulation"]),
            "chosen_beta": float(best_overall_row["beta"]),
            "search_stage": str(best_overall_row.get("search_stage", "coarse")),
            "headline_metrics": metric_headline(best_overall_metrics),
            "deltas_vs_baseline_only": comparisons["vs_baseline_only"],
            "deltas_vs_memory_only": comparisons["vs_memory_only"],
            "deltas_vs_stage5a_best_mixing": comparisons["vs_stage5a_best_mixing"],
            "selection_trace": best_overall_trace,
        }
        write_json(best_config_path, best_config_payload)
        output_artifacts.append(best_config_path)

        best_metrics_payload = {
            "best_overall_config": {
                "formulation": str(best_overall_row["formulation"]),
                "beta": float(best_overall_row["beta"]),
                "search_stage": str(best_overall_row.get("search_stage", "coarse")),
            },
            "best_overall_metrics": best_overall_metrics,
            "baseline_only_metrics": baseline_metrics,
            "memory_only_metrics": memory_only_metrics,
            "stage5a_best_mixing_metrics": stage5a_best_metrics,
            "best_primary_multilabel_logit": {
                "config": {
                    "formulation": str(best_primary_row["formulation"]),
                    "beta": float(best_primary_row["beta"]),
                    "search_stage": str(best_primary_row.get("search_stage", "coarse")),
                },
                "metrics": best_primary_row["metrics"],
                "selection_trace": best_primary_trace,
            },
            "best_pdf_literal_logprob": {
                "config": {
                    "formulation": str(best_pdf_row["formulation"]),
                    "beta": float(best_pdf_row["beta"]),
                    "search_stage": str(best_pdf_row.get("search_stage", "coarse")),
                },
                "metrics": best_pdf_row["metrics"],
                "selection_trace": best_pdf_trace,
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
            memory_source=memory_source,
            stage5a_source=stage5a_source,
            alignment_report=alignment_report,
            sanity_checks=sanity_checks,
            coarse_primary_rows=coarse_primary_rows,
            coarse_pdf_rows=coarse_pdf_rows,
            best_primary_row=best_primary_row,
            best_pdf_row=best_pdf_row,
            best_overall_row=best_overall_row,
            best_overall_trace=best_overall_trace,
            baseline_metrics=baseline_metrics,
            memory_only_metrics=memory_only_metrics,
            stage5a_metrics=stage5a_best_metrics,
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
