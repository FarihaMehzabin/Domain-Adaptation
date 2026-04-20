"""Postmortem utilities for NIH -> CheXpert diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from paper_v1.data.label_space import DEFAULT_LABEL_SPACE
from paper_v1.evaluation.forgetting import compute_forgetting
from paper_v1.evaluation.metrics import _binary_roc_auc
from paper_v1.training.common import build_dataloader, move_batch_to_device
from paper_v1.training.losses import masked_bce_with_logits
from paper_v1.utils.io import read_json, write_csv, write_json


def prediction_dict_from_artifact(path: str | Path) -> dict[str, np.ndarray]:
    arrays = np.load(path)
    return {
        "logits": np.asarray(arrays["logits"]),
        "probabilities": np.asarray(arrays["probabilities"]),
        "targets": np.asarray(arrays["targets"]),
        "masks": np.asarray(arrays["masks"]),
    }


@torch.no_grad()
def collect_internal_outputs(model: torch.nn.Module, dataset, *, batch_size: int, device: torch.device) -> dict[str, Any]:
    dataloader = build_dataloader(dataset, batch_size, shuffle=False)
    model.eval()
    outputs_by_key: dict[str, list[np.ndarray]] = {
        "logits": [],
        "probabilities": [],
        "targets": [],
        "masks": [],
        "gate": [],
        "raw_gate": [],
        "residual_norm": [],
        "residual_abs": [],
        "retrieval_max_similarity": [],
        "retrieval_mean_similarity": [],
    }
    sample_ids: list[str] = []
    for batch in dataloader:
        moved = move_batch_to_device(batch, device)
        outputs = model(moved["embedding"])
        logits = outputs["logits"]
        probabilities = torch.sigmoid(logits)
        outputs_by_key["logits"].append(logits.detach().cpu().numpy())
        outputs_by_key["probabilities"].append(probabilities.detach().cpu().numpy())
        outputs_by_key["targets"].append(moved["target"].detach().cpu().numpy())
        outputs_by_key["masks"].append(moved["mask"].detach().cpu().numpy())
        if "gate" in outputs:
            outputs_by_key["gate"].append(outputs["gate"].detach().cpu().numpy())
        if "raw_gate" in outputs:
            outputs_by_key["raw_gate"].append(outputs["raw_gate"].detach().cpu().numpy())
        if "residual" in outputs:
            residual = outputs["residual"].detach()
            residual_norm = torch.linalg.norm(residual, dim=1, keepdim=True)
            outputs_by_key["residual_norm"].append(residual_norm.cpu().numpy())
            outputs_by_key["residual_abs"].append(residual.abs().cpu().numpy())
        if "retrieval_summary" in outputs:
            outputs_by_key["retrieval_max_similarity"].append(outputs["retrieval_summary"][:, :1].detach().cpu().numpy())
            outputs_by_key["retrieval_mean_similarity"].append(outputs["retrieval_summary"][:, 1:2].detach().cpu().numpy())
        sample_ids.extend(list(batch["sample_id"]))
    payload: dict[str, Any] = {"sample_ids": sample_ids}
    for key, chunks in outputs_by_key.items():
        payload[key] = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), dtype=np.float32)
    return payload


def broadcast_labelwise(values: np.ndarray, *, num_labels: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.shape[0] == 0:
        return np.zeros((0, num_labels), dtype=np.float32)
    if matrix.shape[1] == 1 and num_labels > 1:
        return np.repeat(matrix, num_labels, axis=1)
    if matrix.shape[1] != num_labels:
        raise ValueError(f"expected {num_labels} label columns, found {matrix.shape[1]}")
    return matrix


def write_histogram_csv(
    path: str | Path,
    values: np.ndarray,
    *,
    bins: int = 20,
    min_value: float | None = None,
    max_value: float | None = None,
) -> Path:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        rows = []
    else:
        range_arg = None if min_value is None or max_value is None else (min_value, max_value)
        counts, edges = np.histogram(flat, bins=bins, range=range_arg)
        rows = [
            {
                "bin_start": float(edges[index]),
                "bin_end": float(edges[index + 1]),
                "count": int(counts[index]),
            }
            for index in range(len(counts))
        ]
    fieldnames = ["bin_start", "bin_end", "count"]
    return write_csv(path, fieldnames, rows)


def write_histogram_summary(path: str | Path, values: np.ndarray) -> Path:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    summary = {
        "count": int(flat.size),
        "mean": float(np.mean(flat)) if flat.size else None,
        "std": float(np.std(flat)) if flat.size else None,
        "min": float(np.min(flat)) if flat.size else None,
        "max": float(np.max(flat)) if flat.size else None,
    }
    return write_json(path, summary)


def per_label_auroc_rows(predictions: dict[str, np.ndarray], *, label_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label_index, label_name in enumerate(label_names):
        label_mask = predictions["masks"][:, label_index].astype(bool)
        label_true = predictions["targets"][label_mask, label_index]
        label_prob = predictions["probabilities"][label_mask, label_index]
        rows.append(
            {
                "label_index": label_index,
                "label_name": label_name,
                "auroc": _binary_roc_auc(label_true, label_prob) if label_true.size else None,
            }
        )
    return rows


def write_per_label_auroc_delta(
    path: str | Path,
    *,
    source_predictions: dict[str, np.ndarray],
    candidate_predictions: dict[str, np.ndarray],
    label_names: list[str],
    domain_name: str,
    source_name: str,
    candidate_name: str,
) -> Path:
    source_rows = per_label_auroc_rows(source_predictions, label_names=label_names)
    candidate_rows = per_label_auroc_rows(candidate_predictions, label_names=label_names)
    merged_rows = []
    for source_row, candidate_row in zip(source_rows, candidate_rows):
        source_auroc = source_row["auroc"]
        candidate_auroc = candidate_row["auroc"]
        delta = None
        if source_auroc is not None and candidate_auroc is not None:
            delta = float(candidate_auroc - source_auroc)
        merged_rows.append(
            {
                "domain": domain_name,
                "label_index": source_row["label_index"],
                "label_name": source_row["label_name"],
                f"{source_name}_auroc": source_auroc,
                f"{candidate_name}_auroc": candidate_auroc,
                "delta_auroc": delta,
            }
        )
    fieldnames = list(merged_rows[0].keys()) if merged_rows else ["domain", "label_index", "label_name", "delta_auroc"]
    return write_csv(path, fieldnames, merged_rows)


def per_label_value_rows(
    values: np.ndarray,
    *,
    label_names: list[str],
    method: str,
    domain_name: str,
    metric_name: str,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    labelwise = broadcast_labelwise(values, num_labels=len(label_names))
    rows = []
    for label_index, label_name in enumerate(label_names):
        column = labelwise[:, label_index]
        rows.append(
            {
                "method": method,
                "seed": seed,
                "domain": domain_name,
                "metric": metric_name,
                "label_index": label_index,
                "label_name": label_name,
                "mean": float(np.mean(column)) if column.size else None,
                "std": float(np.std(column)) if column.size else None,
                "min": float(np.min(column)) if column.size else None,
                "max": float(np.max(column)) if column.size else None,
            }
        )
    return rows


def active_label_fraction_row(
    gate_values: np.ndarray,
    *,
    method: str,
    domain_name: str,
    threshold: float,
    seed: int | None = None,
) -> dict[str, Any]:
    gate_matrix = np.asarray(gate_values, dtype=np.float32)
    if gate_matrix.ndim == 1:
        gate_matrix = gate_matrix.reshape(-1, 1)
    active_fraction = (gate_matrix > threshold).mean(axis=1) if gate_matrix.size else np.zeros((0,), dtype=np.float32)
    return {
        "method": method,
        "seed": seed,
        "domain": domain_name,
        "threshold": float(threshold),
        "mean_active_fraction": float(np.mean(active_fraction)) if active_fraction.size else None,
        "std_active_fraction": float(np.std(active_fraction)) if active_fraction.size else None,
        "min_active_fraction": float(np.min(active_fraction)) if active_fraction.size else None,
        "max_active_fraction": float(np.max(active_fraction)) if active_fraction.size else None,
    }


def selection_count_distribution_rows(
    gate_values: np.ndarray,
    *,
    method: str,
    domain_name: str,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    gate_matrix = np.asarray(gate_values, dtype=np.float32)
    if gate_matrix.ndim == 1:
        gate_matrix = gate_matrix.reshape(-1, 1)
    if gate_matrix.size == 0:
        return []
    selected_counts = gate_matrix.sum(axis=1).astype(int)
    unique_counts, count_values = np.unique(selected_counts, return_counts=True)
    total = max(int(count_values.sum()), 1)
    return [
        {
            "method": method,
            "seed": seed,
            "domain": domain_name,
            "selected_label_count": int(selected_label_count),
            "num_samples": int(num_samples),
            "fraction": float(num_samples / total),
        }
        for selected_label_count, num_samples in zip(unique_counts, count_values, strict=True)
    ]


def aggregate_metric_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: list[str],
    value_keys: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)
    aggregated_rows: list[dict[str, Any]] = []
    for key, grouped_rows in grouped.items():
        aggregate = {group_key: value for group_key, value in zip(group_keys, key)}
        aggregate["num_rows"] = len(grouped_rows)
        for value_key in value_keys:
            filtered = [row[value_key] for row in grouped_rows if row.get(value_key) is not None]
            if not filtered:
                aggregate[f"{value_key}_mean"] = None
                aggregate[f"{value_key}_std"] = None
                aggregate[f"{value_key}_min"] = None
                aggregate[f"{value_key}_max"] = None
                continue
            values = np.asarray([float(value) for value in filtered], dtype=np.float64)
            aggregate[f"{value_key}_mean"] = float(np.mean(values))
            aggregate[f"{value_key}_std"] = float(np.std(values))
            aggregate[f"{value_key}_min"] = float(np.min(values))
            aggregate[f"{value_key}_max"] = float(np.max(values))
        aggregated_rows.append(aggregate)
    return aggregated_rows


def verify_main_method_invariants(
    *,
    model: torch.nn.Module,
    old_bank,
    train_dataset,
    device: torch.device,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    previous_requires_grad_false = all(
        not parameter.requires_grad for parameter in model.previous_model.parameters()
    )
    memory_parameters = list(model.memory_module.parameters())
    memory_trainable_parameter_count = int(
        sum(parameter.numel() for parameter in memory_parameters if parameter.requires_grad)
    )

    train_loader = build_dataloader(train_dataset, batch_size=min(32, max(len(train_dataset), 1)), shuffle=False)
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        batch = None

    previous_grad_abs_sum = 0.0
    if batch is not None:
        moved = move_batch_to_device(batch, device)
        model.zero_grad(set_to_none=True)
        outputs = model(moved["embedding"])
        loss = masked_bce_with_logits(outputs["logits"], moved["target"], mask=moved["mask"])
        loss.backward()
        for parameter in model.previous_model.parameters():
            if parameter.grad is not None:
                previous_grad_abs_sum += float(parameter.grad.detach().abs().sum().cpu())
        model.zero_grad(set_to_none=True)

    replay_active_each_epoch = bool(history) and all(float(row.get("replay_loss", 0.0)) > 0.0 for row in history)
    zero_penalty_active_each_epoch = bool(history) and all(
        float(row.get("zero_residual_penalty", 0.0)) > 0.0 for row in history
    )
    prototype_domain_counts: dict[str, int] = {}
    for domain in old_bank.source_domains:
        prototype_domain_counts[domain] = prototype_domain_counts.get(domain, 0) + 1

    expected_forgetting = 0.1
    toy_forgetting = compute_forgetting(
        [
            {"stage_name": "stage0", "domain_metrics": {"d0_nih": {"macro_auroc": 0.8}}},
            {"stage_name": "stage1", "domain_metrics": {"d0_nih": {"macro_auroc": 0.7}}},
        ]
    )
    forgetting_metric_correct = abs(float(toy_forgetting["final_average_forgetting"]) - expected_forgetting) < 1.0e-8

    return {
        "previous_requires_grad_false": previous_requires_grad_false,
        "previous_grad_abs_sum_after_backward": previous_grad_abs_sum,
        "previous_logits_frozen": previous_requires_grad_false and previous_grad_abs_sum == 0.0,
        "memory_trainable_parameter_count": memory_trainable_parameter_count,
        "old_bank_num_prototypes": int(old_bank.num_prototypes),
        "prototype_domain_counts": prototype_domain_counts,
        "current_domain_prototype_count": int(prototype_domain_counts.get("d1_chexpert", 0)),
        "retrieval_uses_current_domain_memory": bool(prototype_domain_counts.get("d1_chexpert", 0) > 0),
        "replay_active_each_epoch": replay_active_each_epoch,
        "zero_penalty_active_each_epoch": zero_penalty_active_each_epoch,
        "forgetting_metric_correct": forgetting_metric_correct,
        "toy_forgetting": toy_forgetting,
    }


def comparison_rows(
    result_rows: list[dict[str, Any]],
    *,
    rescue_methods: set[str],
) -> list[dict[str, Any]]:
    by_method = {row["method"]: row for row in result_rows}
    source_row = by_method["source_only"]
    lwf_row = by_method["lwf"]
    vq_row = by_method["vq_summary_replay"]
    rows = []
    for method_name in sorted(rescue_methods):
        row = by_method[method_name]
        rows.append(
            {
                "method": method_name,
                "seen_average_macro_auroc": row["seen_average_macro_auroc"],
                "nih_forgetting_macro_auroc": row["nih_forgetting_macro_auroc"],
                "d1_chexpert_test_macro_auroc": row["d1_chexpert_test_macro_auroc"],
                "average_macro_ece": row["average_macro_ece"],
                "seen_avg_delta_vs_source_only": float(row["seen_average_macro_auroc"] - source_row["seen_average_macro_auroc"]),
                "chexpert_delta_vs_source_only": float(
                    row["d1_chexpert_test_macro_auroc"] - source_row["d1_chexpert_test_macro_auroc"]
                ),
                "ece_delta_vs_source_only": float(row["average_macro_ece"] - source_row["average_macro_ece"]),
                "seen_avg_delta_vs_lwf": float(row["seen_average_macro_auroc"] - lwf_row["seen_average_macro_auroc"]),
                "seen_avg_delta_vs_vq_summary_replay": float(
                    row["seen_average_macro_auroc"] - vq_row["seen_average_macro_auroc"]
                ),
                "goal_preserve_source_only_seen_avg": bool(
                    row["seen_average_macro_auroc"] >= source_row["seen_average_macro_auroc"]
                ),
                "goal_gain_chexpert_or_calibration": bool(
                    row["d1_chexpert_test_macro_auroc"] > source_row["d1_chexpert_test_macro_auroc"]
                    or row["average_macro_ece"] < source_row["average_macro_ece"]
                ),
            }
        )
    return rows


def summarize_result_row(method: str, metrics: dict[str, Any], *, source_nih_auroc: float) -> dict[str, Any]:
    nih_metrics = metrics["d0_nih_test"]
    chex_metrics = metrics["d1_chexpert_test"]
    nih_auroc = float(nih_metrics["macro_auroc"])
    chex_auroc = float(chex_metrics["macro_auroc"])
    return {
        "method": method,
        "d0_nih_test_macro_auroc": nih_auroc,
        "d1_chexpert_test_macro_auroc": chex_auroc,
        "seen_average_macro_auroc": float((nih_auroc + chex_auroc) / 2.0),
        "nih_forgetting_macro_auroc": float(max(source_nih_auroc - nih_auroc, 0.0)),
        "average_macro_ece": float(
            (
                float(nih_metrics.get("macro_ece") or 0.0)
                + float(chex_metrics.get("macro_ece") or 0.0)
            )
            / 2.0
        ),
        "average_brier_score": float(
            (
                float(nih_metrics.get("brier_score") or 0.0)
                + float(chex_metrics.get("brier_score") or 0.0)
            )
            / 2.0
        ),
    }


def default_label_names() -> list[str]:
    return list(DEFAULT_LABEL_SPACE.names)
