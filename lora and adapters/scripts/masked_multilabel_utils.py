#!/usr/bin/env python3
"""Mask-aware helpers for Policy B multilabel evaluation and loss."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


POLICY_B_LABEL_POLICY = "uignore_blankzero"


def get_label_columns(labels: Sequence[str]) -> list[str]:
    return [str(label) for label in labels]


def get_mask_columns(labels: Sequence[str]) -> list[str]:
    return [f"{label}_mask" for label in labels]


def _coerce_binary_series(dataframe: pd.DataFrame, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(dataframe[column_name], errors="coerce")
    invalid_mask = numeric.isna() | ~numeric.isin([0, 1])
    if invalid_mask.any():
        invalid_values = sorted(pd.unique(dataframe.loc[invalid_mask, column_name]).tolist())
        raise ValueError(f"{column_name} has non-binary values: {invalid_values}")
    return numeric.astype(np.int64)


def extract_targets_and_masks(
    df: pd.DataFrame,
    labels: Sequence[str],
    label_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    label_columns = get_label_columns(labels)
    missing_labels = [column for column in label_columns if column not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing required label columns: {missing_labels}")

    for column_name in label_columns:
        if df[column_name].isna().any():
            raise ValueError(f"{column_name} has NaN values")

    target_columns: list[np.ndarray] = []
    mask_columns: list[np.ndarray] = []
    required_mask_columns = get_mask_columns(labels)

    for label_name, mask_name in zip(label_columns, required_mask_columns):
        targets = _coerce_binary_series(df, label_name)
        target_columns.append(targets.to_numpy(dtype=np.float32))

        if label_policy == POLICY_B_LABEL_POLICY:
            if mask_name not in df.columns:
                raise ValueError(f"Missing required mask column: {mask_name}")
            if df[mask_name].isna().any():
                raise ValueError(f"{mask_name} has NaN values")
            masks = _coerce_binary_series(df, mask_name)
        else:
            masks = pd.Series(np.ones(len(df), dtype=np.int64), index=df.index)
        mask_columns.append(masks.to_numpy(dtype=np.float32))

    if label_columns:
        targets_np = np.stack(target_columns, axis=1).astype(np.float32, copy=False)
        masks_np = np.stack(mask_columns, axis=1).astype(np.float32, copy=False)
    else:
        targets_np = np.empty((len(df), 0), dtype=np.float32)
        masks_np = np.empty((len(df), 0), dtype=np.float32)
    return targets_np, masks_np


def masked_bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != targets.shape or logits.shape != masks.shape:
        raise ValueError(
            "logits, targets, and masks must share the same shape. "
            f"Got logits={tuple(logits.shape)}, targets={tuple(targets.shape)}, masks={tuple(masks.shape)}"
        )

    unreduced_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked_loss = unreduced_loss * masks
    valid_count = masks.sum()
    if torch.count_nonzero(valid_count).item() == 0:
        return logits.sum() * 0.0
    return masked_loss.sum() / valid_count


def compute_masked_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    masks: np.ndarray,
    labels: Sequence[str],
) -> dict[str, object]:
    if y_true.shape != y_prob.shape or y_true.shape != masks.shape:
        raise ValueError(
            "y_true, y_prob, and masks must share the same shape. "
            f"Got y_true={y_true.shape}, y_prob={y_prob.shape}, masks={masks.shape}"
        )

    total_rows = int(y_true.shape[0])
    metrics: dict[str, object] = {
        "macro_auroc": None,
        "macro_auprc": None,
        "micro_auroc": None,
        "micro_auprc": None,
        "defined_auroc_labels": 0,
        "defined_auprc_labels": 0,
        "valid_label_count": 0,
        "per_label": {},
    }

    macro_aurocs: list[float] = []
    macro_auprcs: list[float] = []

    for label_index, label_name in enumerate(get_label_columns(labels)):
        label_targets = y_true[:, label_index]
        label_probabilities = y_prob[:, label_index]
        label_masks = masks[:, label_index]

        invalid_masks = ~np.isin(label_masks, [0, 1])
        if invalid_masks.any():
            invalid_values = sorted(np.unique(label_masks[invalid_masks]).tolist())
            raise ValueError(f"{label_name}_mask has non-binary values: {invalid_values}")

        valid_selector = label_masks.astype(bool)
        valid_targets = label_targets[valid_selector]
        valid_probabilities = label_probabilities[valid_selector]

        n_valid = int(valid_selector.sum())
        positives = int(valid_targets.sum()) if n_valid > 0 else 0
        negatives = int(n_valid - positives)
        masked = int(total_rows - n_valid)
        if positives + negatives + masked != total_rows:
            raise ValueError(
                f"{label_name} count mismatch: positives={positives}, negatives={negatives}, "
                f"masked={masked}, rows={total_rows}"
            )

        label_metrics: dict[str, object] = {
            "positives": positives,
            "negatives": negatives,
            "masked": masked,
            "n_valid": n_valid,
            "valid": False,
            "auroc": None,
            "auprc": None,
            "auroc_defined": False,
            "auprc_defined": False,
            "probability_mean": float(np.mean(valid_probabilities)) if n_valid > 0 else None,
            "probability_std": float(np.std(valid_probabilities)) if n_valid > 0 else None,
        }

        if n_valid == 0:
            label_metrics["reason"] = "all rows masked"
        elif positives == 0 or negatives == 0:
            label_metrics["reason"] = "needs at least one positive and one negative after masking"
        else:
            auroc = float(roc_auc_score(valid_targets, valid_probabilities))
            auprc = float(average_precision_score(valid_targets, valid_probabilities))
            label_metrics["valid"] = True
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
    metrics["valid_label_count"] = len(macro_aurocs)
    return metrics
