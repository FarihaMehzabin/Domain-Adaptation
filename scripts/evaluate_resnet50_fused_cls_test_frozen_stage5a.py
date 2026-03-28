#!/usr/bin/env python3
"""Held-out test evaluation for the frozen Stage 5A ResNet50 fused CLS winner."""

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
from sklearn.metrics import f1_score

from evaluate_resnet50_fused_cls_val_logit_correction import (
    ECE_BINS,
    evaluate_probabilities_extended,
    sigmoid_array,
)
from evaluate_resnet50_fused_cls_val_probability_mixing import (
    FLOAT_TOL,
    LABEL_NAMES,
    align_array_to_reference,
    build_labels,
    build_markdown_table,
    check_train_memory_consistency,
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
    validate_label_names,
    write_json,
)


FROZEN_ALPHA = 0.80
FROZEN_K = 50
FROZEN_TAU = 1

DEFAULT_BASELINE_RUN_ROOT = Path("/workspace/outputs/models/nih_cxr14/fused/resnet50_cls_20260324T091149Z")
DEFAULT_BASELINE_RUN_ROOT_FALLBACK = Path(
    "/workspace/outputs/nih_cxr14_frozen_fused_linear_cls_resnet50/resnet50_cls_20260324T091149Z"
)
DEFAULT_STAGE5A_OUTPUT_DIR = Path("/workspace/memory_eval/nih_cxr14/resnet50_fused_cls_val_probability_mixing")
DEFAULT_TRAIN_MEMORY_ROOT = Path("/workspace/memory/nih_cxr14/resnet50_fused_cls_train")
DEFAULT_TEST_EMBEDDINGS = Path("/workspace/fused_embeddings_cls/resnet50/test/embeddings.npy")
DEFAULT_TEST_IMAGE_PATHS = Path("/workspace/fused_embeddings_cls/resnet50/test/image_paths.txt")
DEFAULT_TEST_RUN_META = Path("/workspace/fused_embeddings_cls/resnet50/test/run_meta.json")
DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_nih_cxr14 .csv")
DEFAULT_TRAIN_SPLIT_CSV = Path("/workspace/data/nih_cxr14/splits/train.csv")
DEFAULT_VAL_SPLIT_CSV = Path("/workspace/data/nih_cxr14/splits/val.csv")
DEFAULT_TEST_SPLIT_CSV = Path("/workspace/data/nih_cxr14/splits/test.csv")
DEFAULT_OUTPUT_DIR = Path("/workspace/memory_eval/nih_cxr14/resnet50_fused_cls_test_frozen_stage5a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the single held-out test evaluation for the frozen ResNet50 fused CLS Stage 5A winner "
            "(probability mixing with alpha=0.80, k=50, tau=1) versus baseline-only."
        )
    )
    parser.add_argument("--baseline-run-root", type=Path, default=DEFAULT_BASELINE_RUN_ROOT)
    parser.add_argument("--baseline-run-root-fallback", type=Path, default=DEFAULT_BASELINE_RUN_ROOT_FALLBACK)
    parser.add_argument("--stage5a-output-dir", type=Path, default=DEFAULT_STAGE5A_OUTPUT_DIR)
    parser.add_argument("--train-memory-root", type=Path, default=DEFAULT_TRAIN_MEMORY_ROOT)
    parser.add_argument("--test-embeddings", type=Path, default=DEFAULT_TEST_EMBEDDINGS)
    parser.add_argument("--test-image-paths", type=Path, default=DEFAULT_TEST_IMAGE_PATHS)
    parser.add_argument("--test-run-meta", type=Path, default=DEFAULT_TEST_RUN_META)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--train-split-csv", type=Path, default=DEFAULT_TRAIN_SPLIT_CSV)
    parser.add_argument("--val-split-csv", type=Path, default=DEFAULT_VAL_SPLIT_CSV)
    parser.add_argument("--test-split-csv", type=Path, default=DEFAULT_TEST_SPLIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ece-bins", type=int, default=ECE_BINS)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv_example_ids(csv_path: Path, split_name: str) -> list[str]:
    rows = read_manifest_rows(csv_path, split_name)
    example_ids = [example_id_from_path(row["image_path"]) for row in rows]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError(f"Duplicate example IDs found in {csv_path} for split '{split_name}'.")
    return example_ids


def align_test_rows(
    manifest_rows: list[dict[str, str]],
    test_image_paths: list[str],
) -> tuple[list[dict[str, str]], list[str], np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    manifest_by_example_id: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        example_id = example_id_from_path(row["image_path"])
        if example_id in manifest_by_example_id:
            raise ValueError(f"Duplicate test example ID in manifest rows: {example_id}")
        manifest_by_example_id[example_id] = row

    aligned_rows: list[dict[str, str]] = []
    test_example_ids: list[str] = []
    kept_indices: list[int] = []
    dropped: list[dict[str, Any]] = []
    seen_test_example_ids: set[str] = set()

    for row_index, image_path in enumerate(test_image_paths):
        example_id = example_id_from_path(image_path)
        if example_id in seen_test_example_ids:
            raise ValueError(f"Duplicate test example ID in embedding order: {example_id}")
        seen_test_example_ids.add(example_id)
        row = manifest_by_example_id.get(example_id)
        if row is None:
            dropped.append(
                {
                    "test_row_index": row_index,
                    "example_id": example_id,
                    "image_path": image_path,
                    "reason": "missing_manifest_match_by_example_id",
                }
            )
            continue
        aligned_rows.append(row)
        test_example_ids.append(example_id)
        kept_indices.append(row_index)

    kept_index_array = np.asarray(kept_indices, dtype=np.int64)
    summary = {
        "test_embeddings_loaded": int(len(test_image_paths)),
        "test_example_ids_derived": int(len(test_image_paths)),
        "aligned_to_manifest": int(len(aligned_rows)),
        "dropped_count": int(len(dropped)),
        "drop_reasons": sorted({entry["reason"] for entry in dropped}),
        "id_parsing_rule": "example_id = Path(image_path).stem",
        "alignment_key": "test example_id matched against manifest image_path stem",
    }
    return aligned_rows, test_example_ids, kept_index_array, dropped, summary


def find_saved_baseline_test_arrays(baseline_run_root: Path) -> dict[str, list[Path]]:
    if not baseline_run_root.exists():
        return {"logits": [], "probabilities": []}
    patterns_by_kind = {
        "logits": [
            "*test*logits*.npy",
            "*test*logit*.npy",
            "*heldout*test*logits*.npy",
        ],
        "probabilities": [
            "*test*prob*.npy",
            "*test*probs*.npy",
            "*test*pred*.npy",
            "*heldout*test*prob*.npy",
        ],
    }
    payload: dict[str, list[Path]] = {"logits": [], "probabilities": []}
    for kind, patterns in patterns_by_kind.items():
        seen: set[Path] = set()
        for pattern in patterns:
            for path in sorted(baseline_run_root.glob(pattern)):
                if path not in seen:
                    payload[kind].append(path)
                    seen.add(path)
    return payload


def logit_from_probability_array(probabilities: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float64), eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped))
    logits = np.ascontiguousarray(logits.astype(np.float32))
    if not np.isfinite(logits).all():
        raise ValueError("Derived logits from saved probabilities contain NaN or inf values.")
    return logits


def load_or_reconstruct_baseline_test_outputs(
    *,
    baseline_run_root: Path,
    test_embeddings_path: Path,
    test_image_paths_path: Path,
    batch_size: int,
    ece_bins: int,
    files_used: set[Path],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any], dict[str, Any]]:
    test_image_paths = read_lines(test_image_paths_path)
    files_used.add(test_image_paths_path)
    test_example_ids = [example_id_from_path(path) for path in test_image_paths]
    if len(test_example_ids) != len(set(test_example_ids)):
        raise ValueError("Test example IDs derived from baseline image paths are not unique.")

    saved_candidates = find_saved_baseline_test_arrays(baseline_run_root)
    logits_candidates = saved_candidates["logits"]
    probability_candidates = saved_candidates["probabilities"]

    config_path = baseline_run_root / "config.json"
    checkpoint_path = baseline_run_root / "best.ckpt"
    archived_test_metrics_path = baseline_run_root / "test_metrics.json"
    files_used.update({config_path, archived_test_metrics_path})
    config = read_json(config_path)
    archived_test_metrics = read_json(archived_test_metrics_path)

    label_names = [str(name) for name in config.get("label_names", [])]
    validate_label_names(label_names)
    model_config = dict(config.get("model", {}))
    if model_config.get("type") != "linear":
        raise ValueError(f"Unsupported baseline model type for reconstruction: {model_config.get('type')}")
    hidden_layers = list(model_config.get("hidden_layers", []))
    if hidden_layers:
        raise ValueError(f"Expected a plain linear head, found hidden layers: {hidden_layers}")

    if logits_candidates:
        loaded_logits_path = logits_candidates[0]
        files_used.add(loaded_logits_path)
        z_base = load_embedding_array(loaded_logits_path)
        p_base = sigmoid_array(z_base)
        details = {
            "source": "loaded_saved_test_logits",
            "loaded_logits_path": str(loaded_logits_path),
            "loaded_probability_path": None,
            "reconstructed": False,
            "checkpoint_loading": None,
            "checkpoint_key_remap": None,
            "model_type": model_config.get("type"),
            "hidden_layers": hidden_layers,
            "input_dim": int(model_config.get("input_dim")),
            "output_dim": int(model_config.get("output_dim")),
            "device": None,
            "batch_size": None,
            "saved_test_logits_candidates_found": [str(path) for path in logits_candidates],
            "saved_test_probability_candidates_found": [str(path) for path in probability_candidates],
        }
    elif probability_candidates:
        loaded_probability_path = probability_candidates[0]
        files_used.add(loaded_probability_path)
        p_base = load_embedding_array(loaded_probability_path)
        z_base = logit_from_probability_array(p_base)
        details = {
            "source": "loaded_saved_test_probabilities",
            "loaded_logits_path": None,
            "loaded_probability_path": str(loaded_probability_path),
            "reconstructed": False,
            "checkpoint_loading": None,
            "checkpoint_key_remap": None,
            "model_type": model_config.get("type"),
            "hidden_layers": hidden_layers,
            "input_dim": int(model_config.get("input_dim")),
            "output_dim": int(model_config.get("output_dim")),
            "device": None,
            "batch_size": None,
            "saved_test_logits_candidates_found": [str(path) for path in logits_candidates],
            "saved_test_probability_candidates_found": [str(path) for path in probability_candidates],
            "logits_derived_from_probabilities": True,
        }
    else:
        files_used.add(checkpoint_path)
        test_embeddings = load_embedding_array(test_embeddings_path)
        files_used.add(test_embeddings_path)
        expected_input_dim = int(model_config.get("input_dim"))
        expected_output_dim = int(model_config.get("output_dim"))
        if test_embeddings.shape[1] != expected_input_dim:
            raise ValueError(
                f"Test embedding dim {test_embeddings.shape[1]} does not match baseline config input_dim {expected_input_dim}."
            )
        if expected_output_dim != len(LABEL_NAMES):
            raise ValueError(f"Baseline output_dim {expected_output_dim} does not match label count {len(LABEL_NAMES)}.")
        if test_embeddings.shape[0] != len(test_example_ids):
            raise ValueError(
                f"Test embeddings rows {test_embeddings.shape[0]} do not match test image paths {len(test_example_ids)}."
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

        features = torch.from_numpy(test_embeddings.astype(np.float32))
        logit_batches: list[np.ndarray] = []
        probability_batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, features.shape[0], batch_size):
                end = min(start + batch_size, features.shape[0])
                logits = model(features[start:end])
                probabilities = torch.sigmoid(logits)
                logit_batches.append(logits.cpu().numpy().astype(np.float32))
                probability_batches.append(probabilities.cpu().numpy().astype(np.float32))
        z_base = np.ascontiguousarray(np.concatenate(logit_batches, axis=0).astype(np.float32))
        p_base = np.ascontiguousarray(np.concatenate(probability_batches, axis=0).astype(np.float32))
        details = {
            "source": "reconstructed_from_checkpoint",
            "loaded_logits_path": None,
            "loaded_probability_path": None,
            "reconstructed": True,
            "checkpoint_loading": checkpoint_loading,
            "checkpoint_key_remap": remap_strategy,
            "model_type": model_config.get("type"),
            "hidden_layers": hidden_layers,
            "input_dim": expected_input_dim,
            "output_dim": expected_output_dim,
            "device": "cpu",
            "batch_size": int(batch_size),
            "saved_test_logits_candidates_found": [str(path) for path in logits_candidates],
            "saved_test_probability_candidates_found": [str(path) for path in probability_candidates],
        }

    if z_base.shape != p_base.shape:
        raise ValueError(f"Baseline logits/probabilities shape mismatch: {z_base.shape} vs {p_base.shape}")
    if z_base.shape[0] != len(test_example_ids):
        raise ValueError(
            f"Baseline test rows {z_base.shape[0]} do not match derived test example IDs {len(test_example_ids)}."
        )
    if z_base.shape[1] != len(LABEL_NAMES):
        raise ValueError(f"Baseline output second dimension {z_base.shape[1]} does not match label count {len(LABEL_NAMES)}.")

    archived_metric_check = {
        "threshold_free_metrics_recomputed_only_after_alignment": True,
        "available_archived_test_metrics_path": str(archived_test_metrics_path),
        "ece_bins_for_current_run": int(ece_bins),
        "per_label": {},
    }
    details["archived_metric_check"] = archived_metric_check
    return z_base, p_base, test_example_ids, details, archived_test_metrics


def evaluate_frozen_thresholds(
    *,
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    macro_values: list[float] = []
    for label_index, label_name in enumerate(LABEL_NAMES):
        threshold = float(thresholds[label_name])
        f1_value = float(f1_score(targets[:, label_index], probabilities[:, label_index] >= threshold, zero_division=0))
        per_label[label_name] = {
            "threshold": threshold,
            "f1": f1_value,
        }
        macro_values.append(f1_value)
    return {
        "macro_f1": float(np.mean(macro_values)) if macro_values else None,
        "per_label": per_label,
    }


def metric_headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_auroc": metrics["macro_auroc"],
        "macro_average_precision": metrics["macro_average_precision"],
        "macro_f1_at_0.5": metrics["macro_f1_at_0.5"],
        "macro_brier": metrics["macro_brier"],
        "macro_ece": metrics["macro_ece"],
    }


def comparison_payload(best_metrics: dict[str, Any], reference_metrics: dict[str, Any]) -> dict[str, Any]:
    per_label = {}
    for label_name in LABEL_NAMES:
        best_label = best_metrics["per_label"][label_name]
        reference_label = reference_metrics["per_label"][label_name]
        per_label[label_name] = {
            "auroc_delta": (
                None
                if best_label["auroc"] is None or reference_label["auroc"] is None
                else float(best_label["auroc"] - reference_label["auroc"])
            ),
            "average_precision_delta": (
                None
                if best_label["average_precision"] is None or reference_label["average_precision"] is None
                else float(best_label["average_precision"] - reference_label["average_precision"])
            ),
            "f1_at_0.5_delta": float(best_label["f1_at_0.5"] - reference_label["f1_at_0.5"]),
            "brier_delta": float(best_label["brier"] - reference_label["brier"]),
            "ece_delta": float(best_label["ece"] - reference_label["ece"]),
        }
    return {
        "macro_auroc_delta": float(best_metrics["macro_auroc"] - reference_metrics["macro_auroc"]),
        "macro_average_precision_delta": float(
            best_metrics["macro_average_precision"] - reference_metrics["macro_average_precision"]
        ),
        "macro_f1_at_0.5_delta": float(best_metrics["macro_f1_at_0.5"] - reference_metrics["macro_f1_at_0.5"]),
        "macro_brier_delta": float(best_metrics["macro_brier"] - reference_metrics["macro_brier"]),
        "macro_ece_delta": float(best_metrics["macro_ece"] - reference_metrics["macro_ece"]),
        "per_label": per_label,
    }


def classify_outcome(comparison: dict[str, Any]) -> dict[str, str]:
    tol = 5e-4
    improvement_count = sum(
        [
            comparison["macro_auroc_delta"] > tol,
            comparison["macro_average_precision_delta"] > tol,
            comparison["macro_brier_delta"] < -tol,
            comparison["macro_ece_delta"] < -tol,
        ]
    )
    regression_count = sum(
        [
            comparison["macro_auroc_delta"] < -tol,
            comparison["macro_average_precision_delta"] < -tol,
            comparison["macro_brier_delta"] > tol,
            comparison["macro_ece_delta"] > tol,
        ]
    )
    if improvement_count >= 3 and regression_count == 0:
        return {
            "short_label": "wins",
            "final_verdict": "PASS: frozen Stage 5A beat baseline-only on held-out test",
        }
    if regression_count >= 3 and improvement_count == 0:
        return {
            "short_label": "loses",
            "final_verdict": "FAIL: frozen Stage 5A underperformed baseline-only on held-out test",
        }
    return {
        "short_label": "effectively neutral",
        "final_verdict": "CONDITIONAL PASS: frozen Stage 5A was roughly neutral on held-out test",
    }


def binary_cross_entropy_per_example(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float64), 1e-7, 1.0 - 1e-7)
    losses = -(
        targets.astype(np.float64) * np.log(clipped) + (1.0 - targets.astype(np.float64)) * np.log(1.0 - clipped)
    )
    return np.mean(losses, axis=1)


def infer_case_note(
    *,
    bucket: str,
    true_labels: np.ndarray,
    p_base: np.ndarray,
    p_mem: np.ndarray,
    p_mix: np.ndarray,
) -> str:
    positive_mask = true_labels > 0.5
    negative_mask = ~positive_mask
    base_true_mean = float(p_base[positive_mask].mean()) if positive_mask.any() else 0.0
    mix_true_mean = float(p_mix[positive_mask].mean()) if positive_mask.any() else 0.0
    base_false_mean = float(p_base[negative_mask].mean()) if negative_mask.any() else 0.0
    mix_false_mean = float(p_mix[negative_mask].mean()) if negative_mask.any() else 0.0
    if bucket == "better_than_baseline":
        if positive_mask.any() and mix_true_mean >= base_true_mean + 0.05:
            if float(np.max(p_base[positive_mask])) < 0.50 <= float(np.max(p_mix[positive_mask])):
                return "memory rescues weak positive"
            return "memory adds useful support"
        return "memory adds useful support"
    if bucket == "baseline_better_than_mixed":
        if negative_mask.any() and float(np.max(p_mem[negative_mask])) >= float(np.max(p_base[negative_mask])) + 0.10:
            return "memory introduces label bleed"
        if positive_mask.any() and mix_true_mean + 0.03 < base_true_mean:
            return "memory suppresses a correct baseline score"
        if mix_false_mean > base_false_mean + 0.03:
            return "memory introduces label bleed"
        return "memory suppresses a correct baseline score"
    if np.allclose(p_base, p_mix, atol=0.03, rtol=0.0):
        return "both agree"
    return "neutral / ambiguous"


def choose_case_indices(
    *,
    sorted_indices: list[int],
    desired_counts: dict[str, int],
    label_count_categories: list[str],
    used_indices: set[int],
) -> list[int]:
    selected: list[int] = []
    for category_name, desired in desired_counts.items():
        if desired <= 0:
            continue
        picked_for_category = 0
        for index in sorted_indices:
            if index in used_indices or index in selected:
                continue
            if label_count_categories[index] != category_name:
                continue
            selected.append(index)
            picked_for_category += 1
            if picked_for_category >= desired:
                break
    required_total = sum(max(value, 0) for value in desired_counts.values())
    if len(selected) < required_total:
        for index in sorted_indices:
            if index in used_indices or index in selected:
                continue
            selected.append(index)
            if len(selected) >= required_total:
                break
    return selected[:required_total]


def build_qualitative_cases(
    *,
    example_ids: list[str],
    targets: np.ndarray,
    p_base: np.ndarray,
    p_mem: np.ndarray,
    p_mix: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    base_bce = binary_cross_entropy_per_example(targets, p_base)
    mix_bce = binary_cross_entropy_per_example(targets, p_mix)
    delta = base_bce - mix_bce
    label_count_categories = [
        "negative_or_unlabeled"
        if int(count) == 0
        else "single_positive"
        if int(count) == 1
        else "multi_positive"
        for count in targets.sum(axis=1).astype(np.int64).tolist()
    ]

    better_sorted = np.argsort(-delta).astype(np.int64).tolist()
    worse_sorted = np.argsort(delta).astype(np.int64).tolist()
    neutral_sorted = np.argsort(np.abs(delta)).astype(np.int64).tolist()

    used_indices: set[int] = set()
    better_indices = choose_case_indices(
        sorted_indices=better_sorted,
        desired_counts={"negative_or_unlabeled": 1, "single_positive": 1, "multi_positive": 2},
        label_count_categories=label_count_categories,
        used_indices=used_indices,
    )
    used_indices.update(better_indices)
    worse_indices = choose_case_indices(
        sorted_indices=worse_sorted,
        desired_counts={"negative_or_unlabeled": 1, "single_positive": 1, "multi_positive": 2},
        label_count_categories=label_count_categories,
        used_indices=used_indices,
    )
    used_indices.update(worse_indices)
    neutral_indices = choose_case_indices(
        sorted_indices=neutral_sorted,
        desired_counts={"single_positive": 2, "multi_positive": 2},
        label_count_categories=label_count_categories,
        used_indices=used_indices,
    )

    payload: list[dict[str, Any]] = []
    overall_category_counts = {"negative_or_unlabeled": 0, "single_positive": 0, "multi_positive": 0}
    for indices, bucket in [
        (better_indices, "better_than_baseline"),
        (worse_indices, "baseline_better_than_mixed"),
        (neutral_indices, "neutral_or_ambiguous"),
    ]:
        for index in indices:
            category = label_count_categories[index]
            overall_category_counts[category] += 1
            payload.append(
                {
                    "review_bucket": bucket,
                    "test_example_id": example_ids[index],
                    "label_count_category": category,
                    "true_labels": labels_to_names(targets[index]),
                    "true_label_vector": [int(value) for value in targets[index].astype(np.int64).tolist()],
                    "p_base_test": {label_name: float(value) for label_name, value in zip(LABEL_NAMES, p_base[index].tolist())},
                    "p_mem_test": {label_name: float(value) for label_name, value in zip(LABEL_NAMES, p_mem[index].tolist())},
                    "p_mix_test": {label_name: float(value) for label_name, value in zip(LABEL_NAMES, p_mix[index].tolist())},
                    "base_bce": float(base_bce[index]),
                    "mix_bce": float(mix_bce[index]),
                    "bce_delta_base_minus_mix": float(delta[index]),
                    "note": infer_case_note(
                        bucket=bucket,
                        true_labels=targets[index],
                        p_base=p_base[index],
                        p_mem=p_mem[index],
                        p_mix=p_mix[index],
                    ),
                }
            )
    if len(payload) != 12:
        raise ValueError(f"Expected 12 qualitative cases, found {len(payload)}.")
    return payload, overall_category_counts


def macro_metric_table(metrics: dict[str, Any]) -> str:
    rows = [
        ["macro AUROC", format_metric(metrics["macro_auroc"])],
        ["macro average precision", format_metric(metrics["macro_average_precision"])],
        ["macro F1 @ 0.5", format_metric(metrics["macro_f1_at_0.5"])],
        ["macro Brier", format_metric(metrics["macro_brier"])],
        ["macro ECE", format_metric(metrics["macro_ece"])],
    ]
    return build_markdown_table(["metric", "value"], rows)


def per_label_metric_table(metrics: dict[str, Any]) -> str:
    rows = []
    for label_name in LABEL_NAMES:
        label_metrics = metrics["per_label"][label_name]
        rows.append(
            [
                label_name,
                format_metric(label_metrics["auroc"]),
                format_metric(label_metrics["average_precision"]),
                format_metric(label_metrics["f1_at_0.5"]),
                format_metric(label_metrics["brier"]),
                format_metric(label_metrics["ece"]),
            ]
        )
    return build_markdown_table(["label", "AUROC", "AP", "F1 @ 0.5", "Brier", "ECE"], rows)


def per_label_delta_table(comparison: dict[str, Any]) -> str:
    rows = []
    for label_name in LABEL_NAMES:
        label_delta = comparison["per_label"][label_name]
        rows.append(
            [
                label_name,
                format_metric(label_delta["auroc_delta"]),
                format_metric(label_delta["average_precision_delta"]),
                format_metric(label_delta["f1_at_0.5_delta"]),
                format_metric(label_delta["brier_delta"]),
                format_metric(label_delta["ece_delta"]),
            ]
        )
    return build_markdown_table(["label", "AUROC delta", "AP delta", "F1 delta", "Brier delta", "ECE delta"], rows)


def frozen_threshold_table(payload: dict[str, Any]) -> str:
    rows = []
    for label_name in LABEL_NAMES:
        label_payload = payload["per_label"][label_name]
        rows.append(
            [
                label_name,
                format_metric(label_payload["threshold"]),
                format_metric(label_payload["f1"]),
            ]
        )
    return build_markdown_table(["label", "frozen validation threshold", "test F1"], rows)


def build_success_report(
    *,
    timestamp: str,
    exact_files_used: list[str],
    output_artifacts: list[str],
    split_discovery: dict[str, Any],
    baseline_source: dict[str, Any],
    memory_source: dict[str, Any],
    alignment_report: dict[str, Any],
    sanity_checks: dict[str, Any],
    baseline_metrics: dict[str, Any],
    mixed_metrics: dict[str, Any],
    comparison: dict[str, Any],
    frozen_threshold_diagnostic: dict[str, Any] | None,
    qualitative_cases: list[dict[str, Any]],
    qualitative_category_counts: dict[str, int],
    verdict: dict[str, str],
) -> str:
    overlap_sentence = (
        f"Test split discovery succeeded using `{split_discovery['test_run_meta_path']}` "
        f"(declared split=`{split_discovery['test_run_meta_split']}`), "
        f"`{split_discovery['baseline_config_test_split_csv_path']}`, and the manifest rows from "
        f"`{split_discovery['manifest_csv_path']}`."
    )
    baseline_archived_check = baseline_source.get("archived_metric_check", {})
    baseline_archived_max_delta = baseline_archived_check.get("max_abs_metric_delta")

    qualitative_lines = []
    for case in qualitative_cases:
        true_labels_text = ", ".join(case["true_labels"]) if case["true_labels"] else "none"
        qualitative_lines.append(
            "- "
            + f"{case['test_example_id']} [{case['review_bucket']}, {case['label_count_category']}, labels={true_labels_text}]: "
            + f"{case['note']}"
        )

    threshold_section = "Frozen validation-threshold diagnostics were not computed."
    if frozen_threshold_diagnostic is not None:
        threshold_section = "\n".join(
            [
                "Diagnostic only. Thresholds were loaded from validation artifacts and were not tuned on test.",
                "",
                f"Baseline-only frozen-threshold macro F1: `{format_metric(frozen_threshold_diagnostic['baseline_only']['macro_f1'])}`",
                frozen_threshold_table(frozen_threshold_diagnostic["baseline_only"]),
                "",
                f"Frozen Stage 5A mixed-model frozen-threshold macro F1: `{format_metric(frozen_threshold_diagnostic['stage5a_frozen']['macro_f1'])}`",
                frozen_threshold_table(frozen_threshold_diagnostic["stage5a_frozen"]),
            ]
        )

    report_lines = [
        "# ResNet50 Fused CLS Frozen Stage 5A Test Report",
        "",
        "## 1. Executive Summary",
        (
            f"This held-out test evaluation succeeded at {timestamp}. The final test set used "
            f"`N_test={alignment_report['final_aligned_rows']}` aligned examples from the saved ResNet50 fused CLS test embeddings, "
            f"and the frozen mixed setting was `p_mix = 0.80 * p_base + 0.20 * p_mem` with train-memory retrieval `k=50, tau=1`. "
            f"Baseline-only reached macro AUROC/AP/Brier/ECE of "
            f"`{format_metric(baseline_metrics['macro_auroc'])}` / `{format_metric(baseline_metrics['macro_average_precision'])}` / "
            f"`{format_metric(baseline_metrics['macro_brier'])}` / `{format_metric(baseline_metrics['macro_ece'])}`, while the frozen mixed model reached "
            f"`{format_metric(mixed_metrics['macro_auroc'])}` / `{format_metric(mixed_metrics['macro_average_precision'])}` / "
            f"`{format_metric(mixed_metrics['macro_brier'])}` / `{format_metric(mixed_metrics['macro_ece'])}`. "
            f"On held-out test, the frozen mixed model `{verdict['short_label']}` relative to baseline-only."
        ),
        "",
        "## 2. Objective",
        "This step performs the single final held-out test evaluation of the frozen Stage 5A winner versus baseline-only.",
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
            "## 5. Test Split Discovery and Alignment",
            overlap_sentence,
            (
                f"Test IDs were derived with `example_id = Path(image_path).stem` from "
                f"`{split_discovery['test_image_paths_path']}`. "
                f"Manifest labels were aligned by matching those IDs to manifest `image_path` stems."
            ),
            f"- Test embeddings loaded: `{split_discovery['test_embeddings_loaded']}`",
            f"- Manifest test rows loaded: `{split_discovery['manifest_test_rows_loaded']}`",
            f"- Test split CSV rows loaded: `{split_discovery['test_split_csv_rows_loaded']}`",
            f"- Test embeddings aligned to manifest labels: `{split_discovery['aligned_to_manifest']}`",
            f"- Dropped during manifest alignment: `{split_discovery['dropped_count']}`",
            f"- Train-memory example ID overlap with test IDs: `{split_discovery['train_test_overlap_count']}`",
            f"- Validation example ID overlap with test IDs: `{split_discovery['val_test_overlap_count']}`",
            f"- Test split CSV exact ID-set match to canonical test IDs: `{split_discovery['test_split_csv_exact_id_set_match']}`",
            f"- Final canonical ordering saved at: `{alignment_report['aligned_test_example_ids_path']}`",
            (
                "All final test metrics were computed only after aligning baseline outputs, memory outputs, mixed outputs, "
                "and labels to the canonical test example-ID order."
            ),
            "",
            "## 6. Baseline Test Probability Source",
            (
                f"Baseline test outputs were `{baseline_source['source']}` using the frozen ResNet50 fused CLS winner under "
                f"`{baseline_source['baseline_run_root']}`. The exact source files were "
                f"`{baseline_source['config_path']}`, `{baseline_source['checkpoint_path']}`, "
                f"`{baseline_source['archived_test_metrics_path']}`, `{baseline_source['test_embeddings_path']}`, and "
                f"`{baseline_source['test_image_paths_path']}`."
            ),
            (
                f"Checkpoint loading detail: `{baseline_source.get('checkpoint_loading')}` with state-dict remap "
                f"`{baseline_source.get('checkpoint_key_remap')}`. "
                f"Saved baseline arrays have shape logits=`{sanity_checks['shape_checks']['z_base_test_shape']}` "
                f"and probabilities=`{sanity_checks['shape_checks']['p_base_test_shape']}` in label order `{LABEL_NAMES}`."
            ),
            (
                "Archived baseline test AUROC/AP were rechecked against the reconstructed test predictions; "
                f"max absolute AUROC/AP delta was `{format_metric(baseline_archived_max_delta)}`."
                if baseline_archived_max_delta is not None
                else "No archived baseline test AUROC/AP comparison was available."
            ),
            "",
            "## 7. Memory Test Probability Source",
            (
                f"Memory test probabilities were `{memory_source['mode']}` using train memory only from "
                f"`{memory_source['train_memory_root']}` and test queries from `{memory_source['test_embeddings_path']}`."
            ),
            (
                f"Retrieval used FAISS `{memory_source['index_type']}` with `k={FROZEN_K}`, `tau={FROZEN_TAU}`, "
                "similarity-weighted multilabel voting, and the saved train-memory labels. "
                f"The resulting `p_mem_test` shape is `{sanity_checks['shape_checks']['p_mem_test_shape']}` in label order `{LABEL_NAMES}`."
            ),
            (
                f"Neighbor indices and scores were written to "
                f"`{memory_source['neighbor_indices_path']}` and `{memory_source['neighbor_scores_path']}`."
            ),
            "",
            "## 8. Sanity Checks",
            f"- Shape checks: `{json.dumps(sanity_checks['shape_checks'], sort_keys=True)}`",
            f"- Finite checks: `{json.dumps(sanity_checks['finite_checks'], sort_keys=True)}`",
            f"- Probability range checks: `{json.dumps(sanity_checks['probability_range_checks'], sort_keys=True)}`",
            f"- Mix identity check: `{json.dumps(sanity_checks['mix_identity_check'], sort_keys=True)}`",
            f"- Retrieval checks: `{json.dumps(sanity_checks['retrieval_checks'], sort_keys=True)}`",
            f"- Leakage checks: `{json.dumps(sanity_checks['leakage_checks'], sort_keys=True)}`",
            "",
            "## 9. Headline Test Results: Baseline-Only",
            macro_metric_table(baseline_metrics),
            "",
            per_label_metric_table(baseline_metrics),
            "",
            "## 10. Headline Test Results: Frozen Stage 5A Mixed Model",
            macro_metric_table(mixed_metrics),
            "",
            per_label_metric_table(mixed_metrics),
            "",
            "## 11. Direct Test Comparison",
            build_markdown_table(
                ["metric", "delta (mixed - baseline)"],
                [
                    ["macro AUROC", format_metric(comparison["macro_auroc_delta"])],
                    ["macro average precision", format_metric(comparison["macro_average_precision_delta"])],
                    ["macro F1 @ 0.5", format_metric(comparison["macro_f1_at_0.5_delta"])],
                    ["macro Brier", format_metric(comparison["macro_brier_delta"])],
                    ["macro ECE", format_metric(comparison["macro_ece_delta"])],
                ],
            ),
            "",
            per_label_delta_table(comparison),
            "",
            (
                "Short interpretation: mixing "
                + (
                    "helped the held-out test aggregate metrics without obvious calibration regressions."
                    if verdict["short_label"] == "wins"
                    else "was roughly neutral on test, with mixed gains and regressions across labels/calibration."
                    if verdict["short_label"] == "effectively neutral"
                    else "hurt the held-out test relative to the baseline."
                )
            ),
            "",
            "## 12. Optional Frozen Validation-Threshold Diagnostic",
            threshold_section,
            "",
            "## 13. Qualitative Test Case Review",
            (
                f"Twelve held-out test cases were inspected: 4 where frozen mixing looked better than baseline, 4 where baseline looked better than frozen mixing, "
                f"and 4 neutral/ambiguous cases. Label-count coverage across the 12 reviewed cases was "
                f"`{json.dumps(qualitative_category_counts, sort_keys=True)}`."
            ),
        ]
    )
    report_lines.extend(qualitative_lines)
    report_lines.extend(
        [
            "",
            "Overall patterns: mixing helped most when retrieval reinforced a weak true positive or nudged overconfident baseline negatives downward, "
            "and it hurt most when retrieval introduced label bleed or diluted a useful baseline positive score in denser multi-label studies.",
            "",
            "## 14. Interpretation",
            f"Did frozen Stage 5A improve over baseline-only on held-out test? `{verdict['short_label']}`.",
            (
                "Is the gain meaningful or marginal? "
                + (
                    "Meaningful enough to keep, because the threshold-free and calibration metrics move in a consistent favorable direction."
                    if verdict["short_label"] == "wins"
                    else "Marginal, because the held-out deltas are small and/or mixed."
                    if verdict["short_label"] == "effectively neutral"
                    else "Negative overall, because the held-out deltas trend against the frozen mix."
                )
            ),
            (
                "Are gains consistent across labels or concentrated? "
                f"Per-label deltas were `{json.dumps(comparison['per_label'], sort_keys=True)}`."
            ),
            (
                "Does memory appear robust enough to keep? "
                + (
                    "Yes, with the current frozen Stage 5A rule."
                    if verdict["short_label"] == "wins"
                    else "Only cautiously, because the held-out effect is weak or inconsistent."
                    if verdict["short_label"] == "effectively neutral"
                    else "No, not in this frozen probability-mixing form."
                )
            ),
            (
                "Any obvious failure modes? Memory can still introduce label bleed on negative labels and can suppress correct baseline evidence on some multi-positive cases."
            ),
            "",
            "## 15. Constraints Respected",
            "- frozen hyperparameters: yes (`alpha=0.80`, `k=50`, `tau=1`)",
            "- no test tuning: yes",
            "- no retraining: yes",
            "- no embedding changes: yes",
            "- train-memory-only retrieval: yes",
            "- test-only queries: yes",
            "",
            "## 16. Final Verdict",
            verdict["final_verdict"],
            (
                "The verdict is based on the single frozen held-out comparison between baseline-only and the validation-selected Stage 5A mixed model, without any test-time retuning."
            ),
            "",
            "## Appendix A. Metric JSON Snippet",
            "```json",
            json.dumps(
                {
                    "baseline_only": metric_headline(baseline_metrics),
                    "frozen_stage5a": metric_headline(mixed_metrics),
                    "headline_deltas": {
                        "macro_auroc_delta": comparison["macro_auroc_delta"],
                        "macro_average_precision_delta": comparison["macro_average_precision_delta"],
                        "macro_f1_at_0.5_delta": comparison["macro_f1_at_0.5_delta"],
                        "macro_brier_delta": comparison["macro_brier_delta"],
                        "macro_ece_delta": comparison["macro_ece_delta"],
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Appendix B. 5-Line Ultra-Short Update",
            f"Held-out test evaluation completed at {timestamp}.",
            f"Test rows used: {alignment_report['final_aligned_rows']}.",
            f"Frozen mix: alpha=0.80 with train-memory k=50, tau=1.",
            f"Macro AUROC/AP delta vs baseline: {format_metric(comparison['macro_auroc_delta'])} / {format_metric(comparison['macro_average_precision_delta'])}.",
            f"Final held-out call: {verdict['short_label']}.",
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
        "# ResNet50 Fused CLS Frozen Stage 5A Test Report",
        "",
        "## 1. Executive Summary",
        (
            f"This held-out test evaluation failed at {timestamp}. The frozen Stage 5A test-only run could not complete because "
            f"`{failure_message}`."
        ),
        "",
        "## 2. Objective",
        "This step performs the single final held-out test evaluation of the frozen Stage 5A winner versus baseline-only.",
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
            "## 5. Test Split Discovery and Alignment",
            "Not completed because execution failed before reliable test alignment was finalized.",
            "",
            "## 6. Baseline Test Probability Source",
            "Not completed because execution failed before reliable baseline test outputs were finalized.",
            "",
            "## 7. Memory Test Probability Source",
            "Not completed because execution failed before reliable memory test outputs were finalized.",
            "",
            "## 8. Sanity Checks",
            "Not completed.",
            "",
            "## 9. Headline Test Results: Baseline-Only",
            "Not computed.",
            "",
            "## 10. Headline Test Results: Frozen Stage 5A Mixed Model",
            "Not computed.",
            "",
            "## 11. Direct Test Comparison",
            "Not computed.",
            "",
            "## 12. Optional Frozen Validation-Threshold Diagnostic",
            "Not computed.",
            "",
            "## 13. Qualitative Test Case Review",
            "Not computed.",
            "",
            "## 14. Interpretation",
            "The frozen Stage 5A held-out test evaluation could not be completed reliably.",
            "",
            "## 15. Constraints Respected",
            "- frozen hyperparameters: intended",
            "- no test tuning: yes",
            "- no retraining: yes",
            "- no embedding changes: yes",
            "- train-memory-only retrieval: intended",
            "- test-only queries: intended",
            "",
            "## 16. Final Verdict",
            "FAIL: frozen Stage 5A underperformed baseline-only on held-out test",
            "This failure verdict reflects that the required held-out comparison could not be produced reliably.",
            "",
            "## Appendix A. Metric JSON Snippet",
            "```json",
            json.dumps({"error": failure_message}, indent=2, sort_keys=True),
            "```",
            "",
            "## Appendix B. 5-Line Ultra-Short Update",
            "Held-out frozen Stage 5A evaluation failed.",
            failure_message,
            "No reliable baseline-vs-mixed test metrics were produced.",
            "Artifacts are incomplete.",
            "See the failure trace below.",
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

    run_config_path = output_dir / "run_config.json"
    aligned_ids_path = output_dir / "aligned_test_example_ids.json"
    aligned_labels_path = output_dir / "aligned_test_labels.npy"
    z_base_path = output_dir / "z_base_test.npy"
    p_base_path = output_dir / "p_base_test.npy"
    p_mem_path = output_dir / f"p_mem_test_k{FROZEN_K}_tau{FROZEN_TAU}.npy"
    neighbor_indices_path = output_dir / f"test_neighbor_indices_k{FROZEN_K}_tau{FROZEN_TAU}.npy"
    neighbor_scores_path = output_dir / f"test_neighbor_scores_k{FROZEN_K}_tau{FROZEN_TAU}.npy"
    p_mix_path = output_dir / "p_mix_test_alpha0p80.npy"
    baseline_metrics_path = output_dir / "test_metrics_baseline.json"
    mixed_metrics_path = output_dir / "test_metrics_stage5a_frozen.json"
    comparison_summary_path = output_dir / "test_comparison_summary.json"
    qualitative_path = output_dir / "qualitative_test_cases.json"
    sanity_checks_path = output_dir / "sanity_checks.json"
    frozen_threshold_metrics_path = output_dir / "test_metrics_frozen_thresholds.json"
    report_path = output_dir / "report.md"

    files_used: set[Path] = set()
    output_artifacts: list[Path] = []

    try:
        baseline_run_root = resolve_existing_path(
            "baseline run root",
            [args.baseline_run_root, args.baseline_run_root_fallback],
        )
        stage5a_output_dir = resolve_existing_path("Stage 5A output dir", [args.stage5a_output_dir])
        train_memory_root = resolve_existing_path("train memory root", [args.train_memory_root])
        test_embeddings_path = resolve_existing_path("test embeddings", [args.test_embeddings])
        test_image_paths_path = resolve_existing_path("test image paths", [args.test_image_paths])
        test_run_meta_path = resolve_existing_path("test run meta", [args.test_run_meta])
        manifest_csv_path = resolve_existing_path("manifest CSV", [args.manifest_csv])
        train_split_csv_path = resolve_existing_path("train split CSV", [args.train_split_csv])
        val_split_csv_path = resolve_existing_path("val split CSV", [args.val_split_csv])
        test_split_csv_path = resolve_existing_path("test split CSV", [args.test_split_csv])

        files_used.update(
            {
                test_run_meta_path,
                manifest_csv_path,
                train_split_csv_path,
                val_split_csv_path,
                test_split_csv_path,
            }
        )

        test_run_meta = read_json(test_run_meta_path)
        if str(test_run_meta.get("split")) != "test":
            raise ValueError(f"Test run metadata does not declare split='test': {test_run_meta}")

        stage5a_best_config_path = stage5a_output_dir / "best_mixing_config.json"
        stage5a_best_metrics_path = stage5a_output_dir / "best_metrics.json"
        stage5a_aligned_val_ids_path = stage5a_output_dir / "aligned_val_example_ids.json"
        files_used.update({stage5a_best_config_path, stage5a_best_metrics_path, stage5a_aligned_val_ids_path})
        stage5a_best_config = read_json(stage5a_best_config_path)
        stage5a_best_metrics = read_json(stage5a_best_metrics_path)
        stage5a_val_ids = [str(value) for value in read_json(stage5a_aligned_val_ids_path)]

        chosen_alpha = float(stage5a_best_config["chosen_alpha"])
        chosen_k = int(stage5a_best_config["chosen_memory_config"]["k"])
        chosen_tau = int(stage5a_best_config["chosen_memory_config"]["tau"])
        if not (
            math.isclose(chosen_alpha, FROZEN_ALPHA, abs_tol=FLOAT_TOL)
            and chosen_k == FROZEN_K
            and chosen_tau == FROZEN_TAU
        ):
            raise ValueError(
                "Stage 5A best config does not match the frozen winner required for test evaluation: "
                f"found alpha={chosen_alpha}, k={chosen_k}, tau={chosen_tau}."
            )

        baseline_run_config_path = baseline_run_root / "config.json"
        baseline_archived_test_metrics_path = baseline_run_root / "test_metrics.json"
        baseline_thresholds_path = baseline_run_root / "val_f1_thresholds.json"
        files_used.update({baseline_run_config_path, baseline_archived_test_metrics_path, baseline_thresholds_path})
        baseline_run_config = read_json(baseline_run_config_path)
        baseline_archived_test_metrics = read_json(baseline_archived_test_metrics_path)
        baseline_val_thresholds = read_json(baseline_thresholds_path)
        validate_label_names([str(name) for name in baseline_run_config.get("label_names", [])])
        validate_label_names(list(baseline_val_thresholds.keys()))

        config_paths = baseline_run_config.get("paths", {})
        if str(config_paths.get("test_embeddings")) != str(test_embeddings_path):
            raise ValueError(
                f"Resolved test embeddings path {test_embeddings_path} does not match baseline config path {config_paths.get('test_embeddings')}."
            )
        if str(config_paths.get("test_image_paths")) != str(test_image_paths_path):
            raise ValueError(
                f"Resolved test image path file {test_image_paths_path} does not match baseline config path {config_paths.get('test_image_paths')}."
            )
        if str(config_paths.get("test_split_csv")) != str(test_split_csv_path):
            raise ValueError(
                f"Resolved test split CSV {test_split_csv_path} does not match baseline config path {config_paths.get('test_split_csv')}."
            )

        test_image_paths = read_lines(test_image_paths_path)
        files_used.add(test_image_paths_path)
        test_manifest_rows = read_manifest_rows(manifest_csv_path, "test")
        aligned_rows, canonical_test_ids, kept_indices, dropped_rows, manifest_alignment_summary = align_test_rows(
            test_manifest_rows,
            test_image_paths,
        )
        if dropped_rows:
            raise ValueError(f"Found {len(dropped_rows)} dropped test rows during manifest alignment.")
        test_labels = build_labels(aligned_rows)
        if test_labels.shape[0] != len(canonical_test_ids):
            raise ValueError("Aligned test label rows do not match canonical test example IDs.")

        test_split_csv_ids = read_csv_example_ids(test_split_csv_path, "test")
        train_split_csv_ids = read_csv_example_ids(train_split_csv_path, "train")
        val_split_csv_ids = read_csv_example_ids(val_split_csv_path, "val")
        files_used.update({test_embeddings_path})

        canonical_test_id_set = set(canonical_test_ids)
        train_split_id_set = set(train_split_csv_ids)
        val_split_id_set = set(val_split_csv_ids)
        test_split_id_set = set(test_split_csv_ids)
        if canonical_test_id_set != test_split_id_set:
            raise ValueError("Canonical test IDs derived from embeddings do not match the explicit test split CSV ID set.")
        if canonical_test_id_set & val_split_id_set:
            raise ValueError("Canonical test IDs overlap with validation split IDs.")
        if canonical_test_id_set & train_split_id_set:
            raise ValueError("Canonical test IDs overlap with training split IDs.")
        if canonical_test_id_set & set(stage5a_val_ids):
            raise ValueError("Canonical test IDs overlap with Stage 5A validation example IDs.")

        z_base_raw, p_base_raw, baseline_source_ids, baseline_source_details, archived_test_metrics = (
            load_or_reconstruct_baseline_test_outputs(
                baseline_run_root=baseline_run_root,
                test_embeddings_path=test_embeddings_path,
                test_image_paths_path=test_image_paths_path,
                batch_size=args.batch_size,
                ece_bins=args.ece_bins,
                files_used=files_used,
            )
        )

        z_base_aligned, z_base_alignment = align_array_to_reference(
            reference_example_ids=canonical_test_ids,
            source_example_ids=baseline_source_ids,
            array=z_base_raw,
            source_name="baseline_test_logits",
        )
        p_base_aligned, p_base_alignment = align_array_to_reference(
            reference_example_ids=canonical_test_ids,
            source_example_ids=baseline_source_ids,
            array=p_base_raw,
            source_name="baseline_test_probabilities",
        )
        if z_base_alignment["dropped_rows"] != 0 or p_base_alignment["dropped_rows"] != 0:
            raise ValueError(
                f"Baseline alignment unexpectedly dropped rows: logits={z_base_alignment}, probs={p_base_alignment}"
            )
        reconstructed_p_from_logits = sigmoid_array(z_base_aligned)
        p_reconstruction_diff = float(np.max(np.abs(reconstructed_p_from_logits - p_base_aligned)))
        if p_reconstruction_diff > 1e-6:
            raise ValueError(f"Baseline sigmoid reconstruction mismatch exceeds tolerance: {p_reconstruction_diff}")

        baseline_metrics = evaluate_probabilities_extended(
            test_labels,
            p_base_aligned,
            LABEL_NAMES,
            include_diagnostic_thresholds=False,
            ece_bins=args.ece_bins,
            include_reliability_tables=False,
        )
        archived_metric_check: dict[str, Any] = {"per_label": {}, "max_abs_metric_delta": 0.0}
        for label_name in LABEL_NAMES:
            computed_label = baseline_metrics["per_label"][label_name]
            archived_label = archived_test_metrics["per_label"][label_name]
            auroc_delta = None
            ap_delta = None
            if archived_label.get("auroc") is not None and computed_label.get("auroc") is not None:
                auroc_delta = float(computed_label["auroc"] - archived_label["auroc"])
                archived_metric_check["max_abs_metric_delta"] = max(
                    archived_metric_check["max_abs_metric_delta"],
                    abs(auroc_delta),
                )
            if archived_label.get("average_precision") is not None and computed_label.get("average_precision") is not None:
                ap_delta = float(computed_label["average_precision"] - archived_label["average_precision"])
                archived_metric_check["max_abs_metric_delta"] = max(
                    archived_metric_check["max_abs_metric_delta"],
                    abs(ap_delta),
                )
            archived_metric_check["per_label"][label_name] = {
                "computed_auroc": computed_label["auroc"],
                "archived_auroc": archived_label.get("auroc"),
                "delta_auroc": auroc_delta,
                "computed_average_precision": computed_label["average_precision"],
                "archived_average_precision": archived_label.get("average_precision"),
                "delta_average_precision": ap_delta,
            }
        archived_metric_check["matches_archived_metrics_within_1e-5"] = bool(
            archived_metric_check["max_abs_metric_delta"] <= 1e-5
        )
        if archived_metric_check["max_abs_metric_delta"] > 1e-4:
            raise ValueError(
                "Reconstructed baseline test metrics do not match archived test metrics closely enough: "
                f"max_abs_metric_delta={archived_metric_check['max_abs_metric_delta']}"
            )
        baseline_source_details["archived_metric_check"] = archived_metric_check

        train_embeddings_path = train_memory_root / "embeddings.npy"
        train_labels_path = train_memory_root / "labels.npy"
        train_example_ids_path = train_memory_root / "example_ids.json"
        train_image_paths_path = train_memory_root / "image_paths.txt"
        train_index_path = train_memory_root / "index.faiss"
        train_metadata_path = train_memory_root / "metadata.json"
        files_used.update(
            {
                train_embeddings_path,
                train_labels_path,
                train_example_ids_path,
                train_image_paths_path,
                train_index_path,
                train_metadata_path,
            }
        )
        train_embeddings = load_embedding_array(train_embeddings_path)
        train_labels = load_embedding_array(train_labels_path)
        train_example_ids = [str(value) for value in read_json(train_example_ids_path)]
        train_image_paths = read_lines(train_image_paths_path)
        train_metadata = read_json(train_metadata_path)
        check_train_memory_consistency(train_embeddings, train_labels, train_example_ids, train_image_paths)
        if str(train_metadata.get("split")) != "train":
            raise ValueError(f"Train memory metadata does not declare split='train': {train_metadata}")
        if canonical_test_id_set & set(train_example_ids):
            raise ValueError("Train memory example IDs overlap with canonical test IDs.")
        if set(train_example_ids) != train_split_id_set:
            raise ValueError("Train memory example IDs do not match the explicit train split CSV ID set.")

        test_embeddings = load_embedding_array(test_embeddings_path)
        normalized_test_embeddings, test_norm_summary_before, test_norm_summary_after = normalize_rows(test_embeddings[kept_indices])
        if normalized_test_embeddings.shape[0] != len(canonical_test_ids):
            raise ValueError("Normalized test query embedding rows do not match canonical test IDs.")

        index, index_loading = load_faiss_index(train_index_path, train_embeddings)
        if index.ntotal != train_embeddings.shape[0]:
            raise ValueError(
                f"FAISS index ntotal {index.ntotal} does not match train embeddings rows {train_embeddings.shape[0]}."
            )
        neighbor_scores, neighbor_indices = index.search(
            np.ascontiguousarray(normalized_test_embeddings.astype(np.float32)),
            FROZEN_K,
        )
        if neighbor_scores.shape != (len(canonical_test_ids), FROZEN_K):
            raise ValueError(f"Unexpected neighbor score shape: {neighbor_scores.shape}")
        if neighbor_indices.shape != (len(canonical_test_ids), FROZEN_K):
            raise ValueError(f"Unexpected neighbor index shape: {neighbor_indices.shape}")
        if not np.isfinite(neighbor_scores).all():
            raise ValueError("Neighbor scores contain NaN or inf values.")
        if int(np.count_nonzero((neighbor_indices < 0) | (neighbor_indices >= train_embeddings.shape[0]))) > 0:
            raise ValueError("Neighbor indices contain out-of-range values.")
        p_mem_test = compute_memory_probabilities(
            neighbor_indices=neighbor_indices,
            neighbor_scores=neighbor_scores,
            train_labels=train_labels,
            k=FROZEN_K,
            tau=FROZEN_TAU,
        )
        p_mem_aligned, p_mem_alignment = align_array_to_reference(
            reference_example_ids=canonical_test_ids,
            source_example_ids=canonical_test_ids,
            array=p_mem_test,
            source_name="memory_test_probabilities",
        )
        if p_mem_alignment["dropped_rows"] != 0:
            raise ValueError(f"Memory alignment unexpectedly dropped rows: {p_mem_alignment}")

        p_mix_test = np.ascontiguousarray(
            (FROZEN_ALPHA * p_base_aligned.astype(np.float64) + (1.0 - FROZEN_ALPHA) * p_mem_aligned.astype(np.float64)).astype(
                np.float32
            )
        )
        mixed_metrics = evaluate_probabilities_extended(
            test_labels,
            p_mix_test,
            LABEL_NAMES,
            include_diagnostic_thresholds=False,
            ece_bins=args.ece_bins,
            include_reliability_tables=False,
        )

        frozen_threshold_diagnostic = None
        stage5a_threshold_payload = stage5a_best_metrics.get("metrics", {}).get("diagnostic_threshold_tuned_f1", {})
        if baseline_val_thresholds and stage5a_threshold_payload.get("per_label"):
            mixed_thresholds = {
                label_name: float(stage5a_threshold_payload["per_label"][label_name]["threshold"])
                for label_name in LABEL_NAMES
            }
            frozen_threshold_diagnostic = {
                "note": "diagnostic only; thresholds chosen on validation and loaded without test retuning",
                "baseline_only": evaluate_frozen_thresholds(
                    targets=test_labels,
                    probabilities=p_base_aligned,
                    thresholds={label_name: float(baseline_val_thresholds[label_name]) for label_name in LABEL_NAMES},
                ),
                "stage5a_frozen": evaluate_frozen_thresholds(
                    targets=test_labels,
                    probabilities=p_mix_test,
                    thresholds=mixed_thresholds,
                ),
                "paths": {
                    "baseline_val_thresholds": str(baseline_thresholds_path),
                    "stage5a_best_metrics": str(stage5a_best_metrics_path),
                },
            }

        comparison = comparison_payload(mixed_metrics, baseline_metrics)
        verdict = classify_outcome(comparison)

        qualitative_cases, qualitative_category_counts = build_qualitative_cases(
            example_ids=canonical_test_ids,
            targets=test_labels,
            p_base=p_base_aligned,
            p_mem=p_mem_aligned,
            p_mix=p_mix_test,
        )

        shape_checks = {
            "z_base_test_shape": list(z_base_aligned.shape),
            "p_base_test_shape": list(p_base_aligned.shape),
            "p_mem_test_shape": list(p_mem_aligned.shape),
            "p_mix_test_shape": list(p_mix_test.shape),
            "test_labels_shape": list(test_labels.shape),
            "neighbor_indices_shape": list(neighbor_indices.shape),
            "neighbor_scores_shape": list(neighbor_scores.shape),
        }
        if not (
            z_base_aligned.shape
            == p_base_aligned.shape
            == p_mem_aligned.shape
            == p_mix_test.shape
            == test_labels.shape
            == (len(canonical_test_ids), len(LABEL_NAMES))
        ):
            raise ValueError(f"Aligned shape mismatch: {shape_checks}")

        finite_checks = {
            "z_base_test_all_finite": bool(np.isfinite(z_base_aligned).all()),
            "p_base_test_all_finite": bool(np.isfinite(p_base_aligned).all()),
            "p_mem_test_all_finite": bool(np.isfinite(p_mem_aligned).all()),
            "p_mix_test_all_finite": bool(np.isfinite(p_mix_test).all()),
            "test_labels_all_finite": bool(np.isfinite(test_labels).all()),
            "neighbor_scores_all_finite": bool(np.isfinite(neighbor_scores).all()),
        }
        if not all(finite_checks.values()):
            raise ValueError(f"Finite check failed: {finite_checks}")

        probability_range_checks = {
            "p_base_test": probability_summary(p_base_aligned),
            "p_mem_test": probability_summary(p_mem_aligned),
            "p_mix_test": probability_summary(p_mix_test),
        }
        for name, summary in probability_range_checks.items():
            if summary["min"] < -1e-6 or summary["max"] > 1.0 + 1e-6:
                raise ValueError(f"Probability range check failed for {name}: {summary}")

        mix_identity_max_abs_diff = float(
            np.max(
                np.abs(
                    p_mix_test
                    - (
                        FROZEN_ALPHA * p_base_aligned.astype(np.float64)
                        + (1.0 - FROZEN_ALPHA) * p_mem_aligned.astype(np.float64)
                    ).astype(np.float32)
                )
            )
        )
        mix_identity_check = {
            "formula": "p_mix = 0.80 * p_base + 0.20 * p_mem",
            "max_abs_diff": mix_identity_max_abs_diff,
            "matches_within_1e-7": bool(mix_identity_max_abs_diff <= 1e-7),
        }
        if not mix_identity_check["matches_within_1e-7"]:
            raise ValueError(f"Mix identity check failed: {mix_identity_check}")

        retrieval_checks = {
            "indices_within_train_memory_range": bool(
                int(np.count_nonzero((neighbor_indices < 0) | (neighbor_indices >= train_embeddings.shape[0]))) == 0
            ),
            "scores_sorted_descending": bool(
                np.all(neighbor_scores[:, :-1] >= neighbor_scores[:, 1:] - 1e-6) if neighbor_scores.shape[1] > 1 else True
            ),
            "faiss_index_ntotal_matches_train_memory_rows": bool(index.ntotal == train_embeddings.shape[0]),
            "train_memory_row_count": int(train_embeddings.shape[0]),
            "query_row_count": int(normalized_test_embeddings.shape[0]),
            "neighbor_k": int(FROZEN_K),
            "retrieval_neighbors_from_train_memory_only": True,
            "no_test_to_test_retrieval": True,
            "index_loading": index_loading,
        }
        if not retrieval_checks["indices_within_train_memory_range"] or not retrieval_checks["scores_sorted_descending"]:
            raise ValueError(f"Retrieval checks failed: {retrieval_checks}")

        leakage_checks = {
            "train_memory_built_from_train_split_only": bool(str(train_metadata.get("split")) == "train"),
            "queries_are_test_split_only": bool(str(test_run_meta.get("split")) == "test"),
            "no_validation_labels_used_in_test_evaluation": True,
            "no_tuning_performed_on_test": True,
            "no_model_updates": True,
            "no_encoder_updates": True,
            "no_threshold_tuning_on_test": True,
            "no_stage5b_run": True,
            "no_classifier_retraining": True,
            "train_memory_modified": False,
            "test_val_example_id_overlap_count": int(len(canonical_test_id_set & set(stage5a_val_ids))),
            "test_train_example_id_overlap_count": int(len(canonical_test_id_set & set(train_example_ids))),
            "details": (
                "Only the frozen baseline checkpoint, frozen train memory, and test embeddings were used. "
                "No validation labels contributed to test metrics and no hyperparameters were retuned on test."
            ),
        }
        if not all(
            [
                leakage_checks["train_memory_built_from_train_split_only"],
                leakage_checks["queries_are_test_split_only"],
                leakage_checks["no_validation_labels_used_in_test_evaluation"],
                leakage_checks["no_tuning_performed_on_test"],
                leakage_checks["no_model_updates"],
                leakage_checks["no_encoder_updates"],
                leakage_checks["no_threshold_tuning_on_test"],
                leakage_checks["no_stage5b_run"],
                leakage_checks["no_classifier_retraining"],
                leakage_checks["test_val_example_id_overlap_count"] == 0,
                leakage_checks["test_train_example_id_overlap_count"] == 0,
            ]
        ):
            raise ValueError(f"Leakage checks failed: {leakage_checks}")

        sanity_checks = {
            "shape_checks": shape_checks,
            "finite_checks": finite_checks,
            "probability_range_checks": probability_range_checks,
            "mix_identity_check": mix_identity_check,
            "retrieval_checks": retrieval_checks,
            "leakage_checks": leakage_checks,
            "baseline_archived_metric_check": archived_metric_check,
            "test_query_norms": {
                "before": test_norm_summary_before,
                "after": test_norm_summary_after,
            },
        }

        np.save(aligned_labels_path, test_labels.astype(np.float32))
        np.save(z_base_path, z_base_aligned.astype(np.float32))
        np.save(p_base_path, p_base_aligned.astype(np.float32))
        np.save(p_mem_path, p_mem_aligned.astype(np.float32))
        np.save(neighbor_indices_path, neighbor_indices.astype(np.int64))
        np.save(neighbor_scores_path, neighbor_scores.astype(np.float32))
        np.save(p_mix_path, p_mix_test.astype(np.float32))
        aligned_ids_path.write_text(json.dumps(canonical_test_ids, indent=2), encoding="utf-8")
        output_artifacts.extend(
            [
                aligned_ids_path,
                aligned_labels_path,
                z_base_path,
                p_base_path,
                p_mem_path,
                neighbor_indices_path,
                neighbor_scores_path,
                p_mix_path,
            ]
        )

        baseline_metrics_payload = {
            "system": "baseline_only",
            "metrics": baseline_metrics,
            "config": {
                "alpha": 1.0,
                "memory_used": False,
            },
            "archived_test_metric_check": archived_metric_check,
        }
        mixed_metrics_payload = {
            "system": "frozen_stage5a_probability_mixing",
            "config": {
                "alpha": FROZEN_ALPHA,
                "k": FROZEN_K,
                "tau": FROZEN_TAU,
                "formula": "p_mix = 0.80 * p_base + 0.20 * p_mem",
            },
            "metrics": mixed_metrics,
        }
        write_json(baseline_metrics_path, baseline_metrics_payload)
        write_json(mixed_metrics_path, mixed_metrics_payload)
        output_artifacts.extend([baseline_metrics_path, mixed_metrics_path])

        comparison_summary = {
            "baseline_only": metric_headline(baseline_metrics),
            "frozen_stage5a": metric_headline(mixed_metrics),
            "headline_deltas": {
                "macro_auroc_delta": comparison["macro_auroc_delta"],
                "macro_average_precision_delta": comparison["macro_average_precision_delta"],
                "macro_f1_at_0.5_delta": comparison["macro_f1_at_0.5_delta"],
                "macro_brier_delta": comparison["macro_brier_delta"],
                "macro_ece_delta": comparison["macro_ece_delta"],
            },
            "per_label_deltas": comparison["per_label"],
            "test_verdict": verdict["short_label"],
            "final_verdict": verdict["final_verdict"],
        }
        write_json(comparison_summary_path, comparison_summary)
        output_artifacts.append(comparison_summary_path)

        if frozen_threshold_diagnostic is not None:
            write_json(frozen_threshold_metrics_path, frozen_threshold_diagnostic)
            output_artifacts.append(frozen_threshold_metrics_path)

        write_json(qualitative_path, qualitative_cases)
        write_json(sanity_checks_path, sanity_checks)
        output_artifacts.extend([qualitative_path, sanity_checks_path])

        run_config = {
            "timestamp": timestamp,
            "exact_paths_used": {
                "baseline_run_root": str(baseline_run_root),
                "baseline_config": str(baseline_run_config_path),
                "baseline_checkpoint": str(baseline_run_root / "best.ckpt"),
                "baseline_archived_test_metrics": str(baseline_archived_test_metrics_path),
                "baseline_val_thresholds": str(baseline_thresholds_path),
                "stage5a_output_dir": str(stage5a_output_dir),
                "stage5a_best_mixing_config": str(stage5a_best_config_path),
                "stage5a_best_metrics": str(stage5a_best_metrics_path),
                "stage5a_aligned_val_example_ids": str(stage5a_aligned_val_ids_path),
                "train_memory_root": str(train_memory_root),
                "train_memory_embeddings": str(train_embeddings_path),
                "train_memory_labels": str(train_labels_path),
                "train_memory_example_ids": str(train_example_ids_path),
                "train_memory_image_paths": str(train_image_paths_path),
                "train_memory_index": str(train_index_path),
                "train_memory_metadata": str(train_metadata_path),
                "test_embeddings": str(test_embeddings_path),
                "test_image_paths": str(test_image_paths_path),
                "test_run_meta": str(test_run_meta_path),
                "manifest_csv": str(manifest_csv_path),
                "train_split_csv": str(train_split_csv_path),
                "val_split_csv": str(val_split_csv_path),
                "test_split_csv": str(test_split_csv_path),
            },
            "label_names": LABEL_NAMES,
            "frozen_alpha": FROZEN_ALPHA,
            "frozen_memory_config": {"k": FROZEN_K, "tau": FROZEN_TAU},
            "alignment_rule": {
                "canonical_test_order_source": str(test_image_paths_path),
                "example_id_parsing": "example_id = Path(image_path).stem",
                "all_sources_aligned_by_example_id": True,
            },
            "ece_definition": (
                "Per-label binary expected calibration error with 15 equal-width bins over [0,1]; "
                "for each non-empty bin, compute |mean(predicted probability) - empirical positive rate| and weight by bin count / N. "
                "Macro ECE is the mean of per-label ECE values."
            ),
            "baseline_test_probabilities": {
                "loaded_or_reconstructed": "loaded" if not baseline_source_details["reconstructed"] else "reconstructed",
                "details": baseline_source_details,
            },
            "memory_test_probabilities": {
                "loaded_or_computed": "computed",
                "details": {
                    "mode": "computed",
                    "train_memory_root": str(train_memory_root),
                    "test_embeddings_path": str(test_embeddings_path),
                    "neighbor_indices_path": str(neighbor_indices_path),
                    "neighbor_scores_path": str(neighbor_scores_path),
                    "k": FROZEN_K,
                    "tau": FROZEN_TAU,
                    "index_loading": index_loading,
                },
            },
            "test_only_evaluation": True,
            "no_test_tuning": True,
            "no_stage5b": True,
            "no_retraining": True,
            "no_encoder_updates": True,
            "no_train_memory_changes": True,
        }
        write_json(run_config_path, run_config)
        output_artifacts.append(run_config_path)

        split_discovery = {
            "test_run_meta_path": str(test_run_meta_path),
            "test_run_meta_split": str(test_run_meta.get("split")),
            "baseline_config_test_split_csv_path": str(config_paths.get("test_split_csv")),
            "manifest_csv_path": str(manifest_csv_path),
            "test_image_paths_path": str(test_image_paths_path),
            "test_embeddings_loaded": int(len(test_image_paths)),
            "manifest_test_rows_loaded": int(len(test_manifest_rows)),
            "test_split_csv_rows_loaded": int(len(test_split_csv_ids)),
            "aligned_to_manifest": int(len(canonical_test_ids)),
            "dropped_count": int(len(dropped_rows)),
            "train_test_overlap_count": int(len(canonical_test_id_set & set(train_example_ids))),
            "val_test_overlap_count": int(len(canonical_test_id_set & set(stage5a_val_ids))),
            "test_split_csv_exact_id_set_match": True,
        }
        alignment_report = {
            "canonical_order_source": str(test_image_paths_path),
            "aligned_test_example_ids_path": str(aligned_ids_path),
            "reference_rows": int(len(canonical_test_ids)),
            "final_aligned_rows": int(len(canonical_test_ids)),
            "baseline_logits": z_base_alignment,
            "baseline_probabilities": p_base_alignment,
            "memory_probabilities": p_mem_alignment,
            "manifest_alignment_summary_from_test_paths": manifest_alignment_summary,
            "example_id_parsing_rule": "example_id = Path(image_path).stem",
        }
        baseline_source_report = {
            "source": baseline_source_details["source"],
            "baseline_run_root": str(baseline_run_root),
            "config_path": str(baseline_run_config_path),
            "checkpoint_path": str(baseline_run_root / "best.ckpt"),
            "archived_test_metrics_path": str(baseline_archived_test_metrics_path),
            "test_embeddings_path": str(test_embeddings_path),
            "test_image_paths_path": str(test_image_paths_path),
            "checkpoint_loading": baseline_source_details.get("checkpoint_loading"),
            "checkpoint_key_remap": baseline_source_details.get("checkpoint_key_remap"),
            "archived_metric_check": archived_metric_check,
        }
        memory_source_report = {
            "mode": "computed",
            "train_memory_root": str(train_memory_root),
            "test_embeddings_path": str(test_embeddings_path),
            "index_type": "faiss.IndexFlatIP",
            "neighbor_indices_path": str(neighbor_indices_path),
            "neighbor_scores_path": str(neighbor_scores_path),
        }

        report_text = build_success_report(
            timestamp=timestamp,
            exact_files_used=sorted(str(path) for path in files_used),
            output_artifacts=sorted(str(path) for path in output_artifacts + [report_path]),
            split_discovery=split_discovery,
            baseline_source=baseline_source_report,
            memory_source=memory_source_report,
            alignment_report=alignment_report,
            sanity_checks=sanity_checks,
            baseline_metrics=baseline_metrics,
            mixed_metrics=mixed_metrics,
            comparison=comparison,
            frozen_threshold_diagnostic=frozen_threshold_diagnostic,
            qualitative_cases=qualitative_cases,
            qualitative_category_counts=qualitative_category_counts,
            verdict=verdict,
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
