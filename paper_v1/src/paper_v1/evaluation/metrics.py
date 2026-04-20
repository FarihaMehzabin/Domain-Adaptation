"""Standalone multilabel metrics without sklearn."""

from __future__ import annotations

from typing import Any

import numpy as np


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = y_true.astype(np.int64)
    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    sorted_score = y_score[order]
    true_positives = np.cumsum(sorted_true == 1)
    false_positives = np.cumsum(sorted_true == 0)
    distinct = np.where(np.diff(sorted_score))[0]
    threshold_indices = np.r_[distinct, sorted_true.size - 1]
    tpr = true_positives[threshold_indices] / positives
    fpr = false_positives[threshold_indices] / negatives
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    return float(np.trapz(tpr, fpr))


def _binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = y_true.astype(np.int64)
    positives = int(y_true.sum())
    if positives == 0:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    true_positives = np.cumsum(sorted_true == 1)
    false_positives = np.cumsum(sorted_true == 0)
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / positives
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(precision * recall_delta))


def binary_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *, num_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total = len(y_true)
    if total == 0:
        return 0.0
    for bin_index in range(num_bins):
        lower = bins[bin_index]
        upper = bins[bin_index + 1]
        if bin_index == num_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += abs(confidence - accuracy) * (np.sum(mask) / total)
    return float(ece)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    num_bins: int = 10,
) -> dict[str, Any]:
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    if mask is None:
        mask = np.ones_like(y_true, dtype=bool)
    num_labels = y_true.shape[1]
    label_metrics: dict[str, dict[str, float | None]] = {}
    aurocs = []
    auprcs = []
    eces = []
    briers = []
    valid_label_counts = {
        "macro_auroc": 0,
        "macro_auprc": 0,
        "macro_ece": 0,
        "brier_score": 0,
    }
    flat_true = []
    flat_prob = []
    for label_index in range(num_labels):
        label_mask = mask[:, label_index].astype(bool)
        label_true = y_true[label_mask, label_index].astype(np.float32)
        label_prob = y_prob[label_mask, label_index].astype(np.float32)
        if label_true.size == 0:
            label_metrics[str(label_index)] = {
                "auroc": None,
                "auprc": None,
                "brier": None,
                "ece": None,
            }
            continue
        auroc = _binary_roc_auc(label_true, label_prob)
        auprc = _binary_average_precision(label_true, label_prob)
        brier = binary_brier_score(label_true, label_prob)
        ece = expected_calibration_error(label_true, label_prob, num_bins=num_bins)
        label_metrics[str(label_index)] = {
            "auroc": auroc,
            "auprc": auprc,
            "brier": brier,
            "ece": ece,
        }
        if auroc is not None:
            aurocs.append(auroc)
            valid_label_counts["macro_auroc"] += 1
        if auprc is not None:
            auprcs.append(auprc)
            valid_label_counts["macro_auprc"] += 1
        briers.append(brier)
        eces.append(ece)
        valid_label_counts["brier_score"] += 1
        valid_label_counts["macro_ece"] += 1
        flat_true.append(label_true)
        flat_prob.append(label_prob)

    micro_true = np.concatenate(flat_true) if flat_true else np.asarray([], dtype=np.float32)
    micro_prob = np.concatenate(flat_prob) if flat_prob else np.asarray([], dtype=np.float32)
    return {
        "label_metrics": label_metrics,
        "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
        "macro_auprc": float(np.mean(auprcs)) if auprcs else None,
        "micro_auroc": _binary_roc_auc(micro_true, micro_prob) if micro_true.size else None,
        "micro_auprc": _binary_average_precision(micro_true, micro_prob) if micro_true.size else None,
        "brier_score": float(np.mean(briers)) if briers else None,
        "macro_ece": float(np.mean(eces)) if eces else None,
        "num_examples": int(y_true.shape[0]),
        "valid_label_counts": valid_label_counts,
    }
