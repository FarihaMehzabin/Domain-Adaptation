#!/usr/bin/env python3
"""Diagnostic checks for the official Policy B LoRA k20 experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adapt_head_only_mimic import (  # noqa: E402
    LABELS,
    StageFailure,
    build_dataloader,
    format_metric,
    json_ready,
    metrics_for_report,
    normalize_state_dict_keys,
    read_csv_checked,
    save_json,
    set_seed,
    validate_manifest,
)
from scripts.adapt_lora_mimic import build_lora_model_from_saved_state  # noqa: E402


ABS_DIFF_THRESHOLDS = [1e-6, 1e-4, 1e-3]
NEAR_ZERO_MAX_ABS = 1e-8
NEAR_ZERO_NORM = 1e-6
POSITIVE_HEAVY_FRACTION = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose the official Policy B LoRA k20 run.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="experiments/policyB_common5_v1",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="policyB_lora_k20_seed2027",
    )
    parser.add_argument(
        "--source_checkpoint",
        type=str,
        default="checkpoints/nih_2k_densenet121_best.pt",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--source_val_predictions",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--source_test_predictions",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_val_predictions",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_test_predictions",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--run_report_json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--support_manifest",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--diagnostic_md",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--diagnostic_json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prediction_diff_csv",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional override for last-checkpoint evaluation, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Allow overwriting existing diagnostic outputs.",
    )
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    base_dir = (ROOT / args.base_dir).resolve()
    if args.lora_checkpoint is None:
        args.lora_checkpoint = base_dir / "checkpoints" / f"{args.run_name}_best.pt"
    else:
        args.lora_checkpoint = Path(args.lora_checkpoint).resolve()

    if args.source_val_predictions is None:
        args.source_val_predictions = base_dir / "outputs" / "policyB_no_adaptation_eval_seed2027_val_predictions.csv"
    else:
        args.source_val_predictions = Path(args.source_val_predictions).resolve()

    if args.source_test_predictions is None:
        args.source_test_predictions = base_dir / "outputs" / "policyB_no_adaptation_eval_seed2027_test_predictions.csv"
    else:
        args.source_test_predictions = Path(args.source_test_predictions).resolve()

    if args.lora_val_predictions is None:
        args.lora_val_predictions = base_dir / "outputs" / f"{args.run_name}_val_predictions.csv"
    else:
        args.lora_val_predictions = Path(args.lora_val_predictions).resolve()

    if args.lora_test_predictions is None:
        args.lora_test_predictions = base_dir / "outputs" / f"{args.run_name}_test_predictions.csv"
    else:
        args.lora_test_predictions = Path(args.lora_test_predictions).resolve()

    if args.run_report_json is None:
        args.run_report_json = base_dir / "reports" / f"{args.run_name}.json"
    else:
        args.run_report_json = Path(args.run_report_json).resolve()

    if args.support_manifest is None:
        args.support_manifest = base_dir / "manifests" / "mimic_common5_policyB_support_k20_seed2027.csv"
    else:
        args.support_manifest = Path(args.support_manifest).resolve()

    if args.diagnostic_md is None:
        args.diagnostic_md = base_dir / "reports" / "policyB_lora_k20_diagnostic.md"
    else:
        args.diagnostic_md = Path(args.diagnostic_md).resolve()

    if args.diagnostic_json is None:
        args.diagnostic_json = base_dir / "reports" / "policyB_lora_k20_diagnostic.json"
    else:
        args.diagnostic_json = Path(args.diagnostic_json).resolve()

    if args.prediction_diff_csv is None:
        args.prediction_diff_csv = base_dir / "reports" / "policyB_lora_k20_prediction_diff.csv"
    else:
        args.prediction_diff_csv = Path(args.prediction_diff_csv).resolve()

    args.base_dir = base_dir
    args.source_checkpoint = (ROOT / args.source_checkpoint).resolve()
    return args


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def refuse_to_overwrite(paths: list[Path], allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise StageFailure(
            "Refusing to overwrite existing diagnostic output(s). "
            f"Re-run with --allow_overwrite if that is explicitly intended: {existing}"
        )


def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise StageFailure(f"Missing {description}: {path}")
    if not path.is_file():
        raise StageFailure(f"Expected a file for {description}, found something else: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    require_file(path, "JSON file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StageFailure(f"Could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StageFailure(f"Expected a JSON object in {path}")
    return payload


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    require_file(path, "checkpoint")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise StageFailure(f"Could not load checkpoint {path}: {exc}") from exc
    if isinstance(checkpoint, dict):
        return checkpoint
    if hasattr(checkpoint, "keys"):
        return dict(checkpoint)
    raise StageFailure(f"Unsupported checkpoint payload type for {path}: {type(checkpoint)!r}")


def extract_state_dict_from_checkpoint(checkpoint: dict[str, Any], path: Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, Any] | None = None
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        state_dict = checkpoint
    if state_dict is None:
        raise StageFailure(f"Could not find model_state_dict/state_dict in checkpoint {path}")
    state_dict = normalize_state_dict_keys(state_dict)
    return state_dict


def float_or_none(value: float | int | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def summarize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float().cpu()
    flat = values.reshape(-1)
    max_abs = float(flat.abs().max().item()) if flat.numel() else 0.0
    norm = float(torch.norm(flat, p=2).item()) if flat.numel() else 0.0
    return {
        "shape": list(values.shape),
        "numel": int(flat.numel()),
        "mean": float(values.mean().item()) if flat.numel() else 0.0,
        "std": float(values.std(unbiased=False).item()) if flat.numel() else 0.0,
        "min": float(values.min().item()) if flat.numel() else 0.0,
        "max": float(values.max().item()) if flat.numel() else 0.0,
        "max_abs": max_abs,
        "norm": norm,
        "all_zero": bool(torch.count_nonzero(flat).item() == 0),
        "near_zero": bool(max_abs <= NEAR_ZERO_MAX_ABS or norm <= NEAR_ZERO_NORM),
    }


def map_lora_state_key_to_source_key(key: str) -> str:
    return key.replace(".base_layer.", ".")


def inspect_lora_checkpoint(
    checkpoint_path: Path,
    source_state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    checkpoint = load_checkpoint_payload(checkpoint_path)
    state_dict = extract_state_dict_from_checkpoint(checkpoint, checkpoint_path)
    lora_keys = [
        key
        for key in state_dict
        if ".lora_down.weight" in key or ".lora_up.weight" in key
    ]
    lora_stats = {key: summarize_tensor(state_dict[key]) for key in sorted(lora_keys)}

    if lora_keys:
        flattened = torch.cat([state_dict[key].detach().float().reshape(-1).cpu() for key in sorted(lora_keys)])
        aggregate_lora = summarize_tensor(flattened)
    else:
        aggregate_lora = None

    all_zero_count = sum(1 for stats in lora_stats.values() if stats["all_zero"])
    near_zero_count = sum(1 for stats in lora_stats.values() if stats["near_zero"])

    matched_original = 0
    missing_original: list[str] = []
    changed_original: list[dict[str, Any]] = []
    unchanged_original = 0
    non_lora_keys = [key for key in state_dict if key not in lora_stats]

    for key in sorted(non_lora_keys):
        source_key = map_lora_state_key_to_source_key(key)
        if source_key not in source_state_dict:
            missing_original.append(key)
            continue
        matched_original += 1
        current = state_dict[key].detach().float().cpu()
        source = source_state_dict[source_key].detach().float().cpu()
        if current.shape != source.shape:
            changed_original.append(
                {
                    "key": key,
                    "source_key": source_key,
                    "reason": f"shape mismatch {list(current.shape)} vs {list(source.shape)}",
                }
            )
            continue
        diff = (current - source).abs()
        max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
        if max_abs_diff > 0.0:
            changed_original.append(
                {
                    "key": key,
                    "source_key": source_key,
                    "max_abs_diff": max_abs_diff,
                    "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
                }
            )
        else:
            unchanged_original += 1

    classifier_base_keys = [
        ("classifier.base_layer.weight", "classifier.weight"),
        ("classifier.base_layer.bias", "classifier.bias"),
        ("classifier.weight", "classifier.weight"),
        ("classifier.bias", "classifier.bias"),
    ]
    classifier_weight_comparison = []
    seen_classifier_keys: set[str] = set()
    for checkpoint_key, source_key in classifier_base_keys:
        if checkpoint_key not in state_dict or checkpoint_key in seen_classifier_keys:
            continue
        seen_classifier_keys.add(checkpoint_key)
        diff = (state_dict[checkpoint_key].detach().float().cpu() - source_state_dict[source_key].detach().float().cpu()).abs()
        classifier_weight_comparison.append(
            {
                "checkpoint_key": checkpoint_key,
                "source_key": source_key,
                "shape": list(state_dict[checkpoint_key].shape),
                "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
                "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
                "changed": bool(float(diff.max().item()) > 0.0 if diff.numel() else False),
            }
        )

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_keys": sorted(checkpoint.keys()),
        "state_dict_tensor_count": int(len(state_dict)),
        "has_lora_down_tensors": any(".lora_down.weight" in key for key in lora_keys),
        "has_lora_up_tensors": any(".lora_up.weight" in key for key in lora_keys),
        "lora_tensor_count": int(len(lora_keys)),
        "lora_tensor_names_sample": sorted(lora_keys)[:20],
        "lora_tensor_stats": lora_stats,
        "lora_tensor_stats_aggregate": aggregate_lora,
        "lora_all_zero_tensor_count": int(all_zero_count),
        "lora_near_zero_tensor_count": int(near_zero_count),
        "lora_nonzero": bool(len(lora_keys) > 0 and all_zero_count < len(lora_keys) and (aggregate_lora or {}).get("norm", 0.0) > 0.0),
        "classifier_weight_comparison": classifier_weight_comparison,
        "non_lora_original_weights_present": bool(matched_original > 0),
        "non_lora_original_tensor_count": int(len(non_lora_keys)),
        "non_lora_original_matched_count": int(matched_original),
        "non_lora_original_unchanged_count": int(unchanged_original),
        "non_lora_original_changed_count": int(len(changed_original)),
        "non_lora_original_changed_sample": changed_original[:20],
        "non_lora_original_missing_count": int(len(missing_original)),
        "non_lora_original_missing_sample": missing_original[:20],
        "parameter_summary_in_checkpoint": checkpoint.get("parameter_summary"),
        "epoch": checkpoint.get("epoch"),
        "train_loss": checkpoint.get("train_loss"),
        "val_loss": checkpoint.get("val_loss"),
        "best_metric_name": checkpoint.get("best_metric_name"),
        "best_metric_value": checkpoint.get("best_metric_value"),
    }


def choose_join_key(left: pd.DataFrame, right: pd.DataFrame) -> str:
    if "dicom_id" in left.columns and "dicom_id" in right.columns:
        return "dicom_id"
    for candidate in ["image_path", "manifest_image_path", "study_id", "subject_id"]:
        if candidate in left.columns and candidate in right.columns:
            return candidate
    raise StageFailure("Could not find a shared join key between prediction files.")


def safe_corr(left: pd.Series, right: pd.Series, method: str) -> float | None:
    if len(left) < 2:
        return None
    if left.nunique(dropna=False) <= 1 or right.nunique(dropna=False) <= 1:
        return None
    value = left.corr(right, method=method)
    return float_or_none(value)


def compare_prediction_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    split_name: str,
    comparison_name: str,
) -> dict[str, Any]:
    join_key = choose_join_key(left, right)
    left_duplicate_count = int(left.duplicated(subset=[join_key]).sum())
    right_duplicate_count = int(right.duplicated(subset=[join_key]).sum())

    left_predictions = left[[join_key] + [f"pred_{label}" for label in LABELS]].copy()
    right_predictions = right[[join_key] + [f"pred_{label}" for label in LABELS]].copy()
    merged = left_predictions.merge(right_predictions, on=join_key, how="inner", suffixes=("_left", "_right"))
    if merged.empty:
        raise StageFailure(f"No overlapping rows found for split={split_name} with join_key={join_key}")

    per_label: dict[str, Any] = {}
    diff_matrices: list[np.ndarray] = []

    for label in LABELS:
        left_col = f"pred_{label}_left"
        right_col = f"pred_{label}_right"
        diffs = (merged[left_col] - merged[right_col]).abs()
        diff_matrices.append(diffs.to_numpy(dtype=np.float64))
        per_label[label] = {
            "mean_abs_diff": float(diffs.mean()),
            "max_abs_diff": float(diffs.max()),
            "pearson": safe_corr(merged[left_col], merged[right_col], method="pearson"),
            "spearman": safe_corr(merged[left_col], merged[right_col], method="spearman"),
            "count_abs_diff_gt_1e-6": int((diffs > 1e-6).sum()),
            "count_abs_diff_gt_1e-4": int((diffs > 1e-4).sum()),
            "count_abs_diff_gt_1e-3": int((diffs > 1e-3).sum()),
        }

    all_diffs = np.column_stack(diff_matrices)
    overall_mean_abs_diff = float(np.mean(all_diffs))
    overall_max_abs_diff = float(np.max(all_diffs))
    overall = {
        "mean_abs_diff": overall_mean_abs_diff,
        "max_abs_diff": overall_max_abs_diff,
        "count_abs_diff_gt_1e-6": int(np.sum(all_diffs > 1e-6)),
        "count_abs_diff_gt_1e-4": int(np.sum(all_diffs > 1e-4)),
        "count_abs_diff_gt_1e-3": int(np.sum(all_diffs > 1e-3)),
        "n_elements": int(all_diffs.size),
    }
    effectively_source_only = bool(
        overall["count_abs_diff_gt_1e-6"] == 0
        or (overall_mean_abs_diff <= 1e-6 and overall["count_abs_diff_gt_1e-4"] == 0)
    )

    return {
        "comparison_name": comparison_name,
        "split_name": split_name,
        "join_key": join_key,
        "left_row_count": int(len(left)),
        "right_row_count": int(len(right)),
        "matched_row_count": int(len(merged)),
        "left_duplicate_join_keys": left_duplicate_count,
        "right_duplicate_join_keys": right_duplicate_count,
        "per_label": per_label,
        "overall": overall,
        "effectively_source_only": effectively_source_only,
        "flag_message": "LoRA predictions are effectively source-only." if effectively_source_only else None,
    }


def load_prediction_csv(path: Path) -> pd.DataFrame:
    return read_csv_checked(require_file(path, "prediction CSV"))


def build_prediction_frame_from_manifest(dataframe: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "image_path": dataframe["resolved_path"].astype(str),
            "subject_id": dataframe["subject_id"],
            "study_id": dataframe["study_id"],
        }
    )
    if "dicom_id" in dataframe.columns:
        frame["dicom_id"] = dataframe["dicom_id"]
    if "abs_path" in dataframe.columns:
        frame["manifest_image_path"] = dataframe["abs_path"]
    elif "resolved_path" in dataframe.columns:
        frame["manifest_image_path"] = dataframe["resolved_path"].astype(str)
    for label_index, label in enumerate(LABELS):
        frame[f"pred_{label}"] = probabilities[:, label_index]
    return frame


def evaluate_last_checkpoint_if_possible(
    last_checkpoint_path: Path | None,
    run_report: dict[str, Any] | None,
    device: torch.device,
) -> dict[str, Any] | None:
    if last_checkpoint_path is None or run_report is None:
        return None
    if not last_checkpoint_path.exists():
        return {
            "available": False,
            "reason": f"Last checkpoint not found: {last_checkpoint_path}",
        }

    training_config = run_report.get("training_config", {})
    label_policy = run_report.get("label_policy", "uignore_blankzero")
    val_manifest = Path(run_report["val_manifest"]).resolve()
    test_manifest = Path(run_report["test_manifest"]).resolve()
    image_size = int(training_config.get("image_size", 224))
    batch_size = int(training_config.get("batch_size", 8))
    seed = int(training_config.get("seed", 2027))
    rank = int(training_config.get("lora_rank", 4))
    alpha = float(training_config.get("lora_alpha", 4.0))
    dropout = float(training_config.get("lora_dropout", 0.0))

    set_seed(seed)
    checkpoint = load_checkpoint_payload(last_checkpoint_path)
    state_dict = extract_state_dict_from_checkpoint(checkpoint, last_checkpoint_path)
    model = build_lora_model_from_saved_state(
        state_dict=state_dict,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        device=device,
    )

    val_df, _ = validate_manifest(val_manifest, "val", label_policy)
    test_df, _ = validate_manifest(test_manifest, "test", label_policy)
    val_loader = build_dataloader(val_df, image_size=image_size, batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = build_dataloader(test_df, image_size=image_size, batch_size=batch_size, shuffle=False, seed=seed)

    from scripts.adapt_head_only_mimic import evaluate_split  # noqa: E402

    val_metrics_raw = evaluate_split(model, val_loader, device)
    test_metrics_raw = evaluate_split(model, test_loader, device)
    return {
        "available": True,
        "checkpoint_path": str(last_checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "val_metrics": metrics_for_report(val_metrics_raw),
        "test_metrics": metrics_for_report(test_metrics_raw),
        "val_predictions": build_prediction_frame_from_manifest(val_df, val_metrics_raw["probabilities"]),
        "test_predictions": build_prediction_frame_from_manifest(test_df, test_metrics_raw["probabilities"]),
    }


def analyze_training_history(
    run_report: dict[str, Any] | None,
    best_vs_source: dict[str, Any],
    last_eval: dict[str, Any] | None,
    source_predictions: dict[str, pd.DataFrame],
    best_predictions: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    if run_report is None:
        return {
            "available": False,
            "reason": "Run report JSON not found.",
        }

    history = run_report.get("training_history", [])
    epochs = [int(item["epoch"]) for item in history]
    train_losses = [float(item["train_loss"]) for item in history]
    val_losses = [float(item["val_loss"]) for item in history]
    val_macro_aurocs = [float_or_none(item.get("val_macro_auroc")) for item in history]
    val_macro_auprcs = [float_or_none(item.get("val_macro_auprc")) for item in history]
    best_epoch = run_report.get("best_epoch")
    stopped_early = bool(run_report.get("stopped_early", False))
    patience = run_report.get("training_config", {}).get("patience")

    train_loss_decreased = bool(len(train_losses) >= 2 and train_losses[-1] < train_losses[0] - 1e-12)
    val_auroc_improved_after_epoch1 = any(
        value is not None and val_macro_aurocs[0] is not None and value > val_macro_aurocs[0] + 1e-12
        for value in val_macro_aurocs[1:]
    ) if val_macro_aurocs else False
    val_auprc_improved_after_epoch1 = any(
        value is not None and val_macro_auprcs[0] is not None and value > val_macro_auprcs[0] + 1e-12
        for value in val_macro_auprcs[1:]
    ) if val_macro_auprcs else False

    early_stopping_reason = run_report.get("early_stopping_reason")
    if early_stopping_reason is None and stopped_early and patience is not None and best_epoch is not None:
        early_stopping_reason = (
            f"Inferred patience stop after {int(patience)} consecutive non-improving epochs "
            f"following best epoch {best_epoch}."
        )

    best_epoch_source_like = bool(best_epoch == 1 and best_vs_source["val"]["overall"]["mean_abs_diff"] < 0.01)

    later_epoch_comparison = None
    later_epochs_changed_predictions_not_selected = False
    if last_eval and last_eval.get("available"):
        last_vs_source = {
            "val": compare_prediction_frames(
                source_predictions["val"],
                last_eval["val_predictions"],
                split_name="val",
                comparison_name="source_vs_last",
            ),
            "test": compare_prediction_frames(
                source_predictions["test"],
                last_eval["test_predictions"],
                split_name="test",
                comparison_name="source_vs_last",
            ),
        }
        last_vs_best = {
            "val": compare_prediction_frames(
                best_predictions["val"],
                last_eval["val_predictions"],
                split_name="val",
                comparison_name="best_vs_last",
            ),
            "test": compare_prediction_frames(
                best_predictions["test"],
                last_eval["test_predictions"],
                split_name="test",
                comparison_name="best_vs_last",
            ),
        }
        later_epoch_comparison = {
            "last_vs_source": last_vs_source,
            "last_vs_best": last_vs_best,
        }
        later_epochs_changed_predictions_not_selected = bool(
            best_epoch == 1
            and (
                last_vs_best["val"]["overall"]["mean_abs_diff"] > 1e-3
                or last_vs_best["test"]["overall"]["mean_abs_diff"] > 1e-3
            )
            and (
                last_vs_source["val"]["overall"]["mean_abs_diff"] > best_vs_source["val"]["overall"]["mean_abs_diff"] + 1e-4
                or last_vs_source["test"]["overall"]["mean_abs_diff"] > best_vs_source["test"]["overall"]["mean_abs_diff"] + 1e-4
            )
        )

    return {
        "available": True,
        "history_length": int(len(history)),
        "epochs": epochs,
        "train_loss_per_epoch": train_losses,
        "val_loss_per_epoch": val_losses,
        "val_macro_auroc_per_epoch": val_macro_aurocs,
        "val_macro_auprc_per_epoch": val_macro_auprcs,
        "selected_best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "early_stopping_reason": early_stopping_reason,
        "train_loss_decreased": train_loss_decreased,
        "val_macro_auroc_improved_after_epoch1": val_auroc_improved_after_epoch1,
        "val_macro_auprc_improved_after_epoch1": val_auprc_improved_after_epoch1,
        "best_epoch_1_source_like": best_epoch_source_like,
        "later_epochs_changed_predictions_not_selected": later_epochs_changed_predictions_not_selected,
        "later_epoch_comparison": later_epoch_comparison,
    }


def audit_trainability(run_report: dict[str, Any] | None) -> dict[str, Any]:
    if run_report is None:
        return {
            "available": False,
            "reason": "Run report JSON not found.",
        }

    parameter_summary = run_report.get("parameter_summary", {})
    trainable_names = list(parameter_summary.get("trainable_parameter_names", []))
    expected_patterns_ok = all(
        (".lora_down.weight" in name or ".lora_up.weight" in name)
        for name in trainable_names
    )
    classifier_base_accidentally_trainable = any(
        name.startswith("classifier.") and ".lora_" not in name
        for name in trainable_names
    )
    denseblock_base_accidentally_trainable = any(
        name.startswith("features.denseblock4.") and ".lora_" not in name
        for name in trainable_names
    )

    return {
        "available": True,
        "target_module_count": parameter_summary.get("target_module_count"),
        "trainable_parameters": parameter_summary.get("trainable_parameters"),
        "trainable_parameter_tensors": parameter_summary.get("trainable_parameter_tensors"),
        "expected_target_module_count": 33,
        "expected_trainable_parameters": 87060,
        "expected_trainable_parameter_tensors": 66,
        "target_module_count_matches_expected": bool(parameter_summary.get("target_module_count") == 33),
        "trainable_parameters_match_expected": bool(parameter_summary.get("trainable_parameters") == 87060),
        "trainable_parameter_tensors_match_expected": bool(parameter_summary.get("trainable_parameter_tensors") == 66),
        "expected_trainable_names_include_lora_only": expected_patterns_ok,
        "classifier_base_accidentally_trainable": classifier_base_accidentally_trainable,
        "denseblock_base_accidentally_trainable": denseblock_base_accidentally_trainable,
        "unexpected_trainable_names": parameter_summary.get("unexpected_trainable_names", []),
        "missing_trainable_names": parameter_summary.get("missing_trainable_names", []),
        "trainable_parameter_names_sample": trainable_names[:20],
    }


def analyze_support_manifest(path: Path) -> dict[str, Any]:
    _, summary = validate_manifest(path, "support", "uignore_blankzero")
    per_label: dict[str, Any] = {}
    warnings: list[str] = []
    for label in LABELS:
        counts = summary["label_counts"][label]
        n_valid = int(counts["n_valid"])
        positives = int(counts["positives"])
        negatives = int(counts["negatives"])
        masked = int(counts["masked"])
        positive_fraction = None if n_valid == 0 else float(positives / n_valid)
        label_warnings = []
        if positives < 5:
            label_warnings.append("positives < 5")
        if negatives < 5:
            label_warnings.append("negatives < 5")
        if positive_fraction is not None and positive_fraction >= POSITIVE_HEAVY_FRACTION:
            label_warnings.append(f"positive fraction >= {POSITIVE_HEAVY_FRACTION:.1f}")
        if label_warnings:
            warnings.append(f"{label}: {', '.join(label_warnings)}")
        per_label[label] = {
            "positives": positives,
            "negatives": negatives,
            "masked": masked,
            "n_valid": n_valid,
            "positive_fraction": positive_fraction,
            "warnings": label_warnings,
        }
    severe_labels = [
        label
        for label, item in per_label.items()
        if item["positives"] < 5
        or item["negatives"] < 5
        or (item["positive_fraction"] is not None and item["positive_fraction"] >= POSITIVE_HEAVY_FRACTION)
    ]
    return {
        "manifest_path": str(path),
        "split_summary": summary,
        "per_label": per_label,
        "warnings": warnings,
        "severe_label_count": int(len(severe_labels)),
        "severe_labels": severe_labels,
    }


def build_prediction_diff_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_name = comparison["split_name"]
    join_key = comparison["join_key"]
    common = {
        "comparison_name": comparison["comparison_name"],
        "split_name": split_name,
        "join_key": join_key,
        "matched_row_count": comparison["matched_row_count"],
        "left_row_count": comparison["left_row_count"],
        "right_row_count": comparison["right_row_count"],
    }
    for label in LABELS:
        stats = comparison["per_label"][label]
        rows.append(
            {
                **common,
                "label": label,
                "mean_abs_diff": stats["mean_abs_diff"],
                "max_abs_diff": stats["max_abs_diff"],
                "pearson": stats["pearson"],
                "spearman": stats["spearman"],
                "count_abs_diff_gt_1e-6": stats["count_abs_diff_gt_1e-6"],
                "count_abs_diff_gt_1e-4": stats["count_abs_diff_gt_1e-4"],
                "count_abs_diff_gt_1e-3": stats["count_abs_diff_gt_1e-3"],
                "effectively_source_only": comparison["effectively_source_only"],
            }
        )
    overall = comparison["overall"]
    rows.append(
        {
            **common,
            "label": "__overall__",
            "mean_abs_diff": overall["mean_abs_diff"],
            "max_abs_diff": overall["max_abs_diff"],
            "pearson": None,
            "spearman": None,
            "count_abs_diff_gt_1e-6": overall["count_abs_diff_gt_1e-6"],
            "count_abs_diff_gt_1e-4": overall["count_abs_diff_gt_1e-4"],
            "count_abs_diff_gt_1e-3": overall["count_abs_diff_gt_1e-3"],
            "effectively_source_only": comparison["effectively_source_only"],
        }
    )
    return rows


def aggregate_best_vs_source(best_vs_source: dict[str, Any]) -> dict[str, Any]:
    means = [best_vs_source[split]["overall"]["mean_abs_diff"] for split in ["val", "test"]]
    maxima = [best_vs_source[split]["overall"]["max_abs_diff"] for split in ["val", "test"]]
    return {
        "overall_mean_abs_diff_across_splits": float(np.mean(means)),
        "overall_max_abs_diff_across_splits": float(np.max(maxima)),
        "all_splits_effectively_source_only": bool(
            best_vs_source["val"]["effectively_source_only"] and best_vs_source["test"]["effectively_source_only"]
        ),
    }


def choose_final_diagnosis(
    checkpoint_inspection: dict[str, Any],
    training_history: dict[str, Any],
    trainability_audit: dict[str, Any],
    support_analysis: dict[str, Any],
    best_vs_source_summary: dict[str, Any],
) -> tuple[str, str, list[str]]:
    reasons: list[str] = []

    implementation_bug_likely = any(
        [
            checkpoint_inspection["lora_tensor_count"] == 0,
            not checkpoint_inspection["has_lora_down_tensors"],
            not checkpoint_inspection["has_lora_up_tensors"],
            checkpoint_inspection["lora_all_zero_tensor_count"] == checkpoint_inspection["lora_tensor_count"]
            and checkpoint_inspection["lora_tensor_count"] > 0,
            checkpoint_inspection["non_lora_original_changed_count"] > 0,
            not trainability_audit.get("expected_trainable_names_include_lora_only", True),
            bool(trainability_audit.get("unexpected_trainable_names")),
            bool(trainability_audit.get("missing_trainable_names")),
        ]
    )
    if implementation_bug_likely:
        reasons.append("Checkpoint/trainability inspection found inconsistent or inactive LoRA state.")
        return "IMPLEMENTATION_BUG_LIKELY", "fix implementation bug first", reasons

    if training_history.get("available") and training_history.get("later_epochs_changed_predictions_not_selected"):
        reasons.append("Later epochs changed predictions materially, but validation selected epoch 1.")
        return "EFFECTIVE_NO_OP_DUE_TO_MODEL_SELECTION", "change LoRA learning rate/rank and rerun", reasons

    severe_support_bias = support_analysis["severe_label_count"] >= max(3, math.ceil(len(LABELS) / 2))
    if severe_support_bias and best_vs_source_summary["overall_mean_abs_diff_across_splits"] < 0.01:
        reasons.append("Support labels are heavily imbalanced and prediction movement is limited.")
        return "SUPPORT_SET_TOO_SMALL_OR_BIASED", "run full fine-tune k20 as upper bound", reasons

    learned_signals = (
        checkpoint_inspection["lora_nonzero"]
        and training_history.get("available")
        and training_history.get("train_loss_decreased")
        and best_vs_source_summary["overall_mean_abs_diff_across_splits"] > 1e-4
    )
    val_never_improved = training_history.get("available") and not (
        training_history.get("val_macro_auroc_improved_after_epoch1")
        or training_history.get("val_macro_auprc_improved_after_epoch1")
    )
    if learned_signals and val_never_improved:
        reasons.append("LoRA weights moved and train loss dropped, but validation metrics did not improve.")
        return "LEARNED_BUT_NO_GENERALIZATION", "run last-block fine-tune k20", reasons

    reasons.append("No single failure mode dominates the available evidence.")
    return "INCONCLUSIVE", "change LoRA learning rate/rank and rerun", reasons


def build_markdown_report(report: dict[str, Any]) -> str:
    best_vs_source = report["prediction_difference"]["best_vs_source"]
    checkpoint = report["checkpoint_inspection"]
    training = report["training_history_analysis"]
    trainability = report["trainability_audit"]
    support = report["support_set_analysis"]
    summary = report["summary"]

    lines = [
        "# Policy B LoRA k20 Diagnostic",
        "",
        "## Final Diagnosis",
        f"- final diagnosis: `{report['final_diagnosis']}`",
        f"- recommended next action: `{report['recommended_next_action']}`",
        f"- overall mean prediction diff vs source-only: {summary['overall_mean_abs_prediction_diff_vs_source_only']:.6f}",
        f"- overall max prediction diff vs source-only: {summary['overall_max_abs_prediction_diff_vs_source_only']:.6f}",
        f"- LoRA tensors nonzero: {'yes' if summary['lora_tensors_nonzero'] else 'no'}",
        f"- train loss decreased: {'yes' if summary['train_loss_decreased'] else 'no'}",
    ]

    for split_name in ["val", "test"]:
        split = best_vs_source[split_name]
        lines.extend(
            [
                "",
                f"## Prediction Diff: {split_name}",
                f"- join key: `{split['join_key']}`",
                f"- matched rows: {split['matched_row_count']}",
                f"- overall mean abs diff: {split['overall']['mean_abs_diff']:.6f}",
                f"- overall max abs diff: {split['overall']['max_abs_diff']:.6f}",
                f"- count abs diff > 1e-6: {split['overall']['count_abs_diff_gt_1e-6']}",
                f"- count abs diff > 1e-4: {split['overall']['count_abs_diff_gt_1e-4']}",
                f"- count abs diff > 1e-3: {split['overall']['count_abs_diff_gt_1e-3']}",
                f"- effectively source-only flag: {'yes' if split['effectively_source_only'] else 'no'}",
                "",
                "| Label | Mean Abs Diff | Max Abs Diff | Pearson | Spearman | >1e-6 | >1e-4 | >1e-3 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label in LABELS:
            item = split["per_label"][label]
            pearson = "n/a" if item["pearson"] is None else f"{item['pearson']:.6f}"
            spearman = "n/a" if item["spearman"] is None else f"{item['spearman']:.6f}"
            lines.append(
                f"| {label} | {item['mean_abs_diff']:.6f} | {item['max_abs_diff']:.6f} | "
                f"{pearson} | {spearman} | {item['count_abs_diff_gt_1e-6']} | "
                f"{item['count_abs_diff_gt_1e-4']} | {item['count_abs_diff_gt_1e-3']} |"
            )

    lines.extend(
        [
            "",
            "## Checkpoint Inspection",
            f"- checkpoint keys: {checkpoint['checkpoint_keys']}",
            f"- LoRA tensors present: {'yes' if checkpoint['lora_tensor_count'] > 0 else 'no'}",
            f"- LoRA tensor count: {checkpoint['lora_tensor_count']}",
            f"- LoRA all-zero tensor count: {checkpoint['lora_all_zero_tensor_count']}",
            f"- LoRA near-zero tensor count: {checkpoint['lora_near_zero_tensor_count']}",
            f"- LoRA aggregate norm: {0.0 if checkpoint['lora_tensor_stats_aggregate'] is None else checkpoint['lora_tensor_stats_aggregate']['norm']:.6f}",
            f"- non-LoRA original tensors changed vs source: {checkpoint['non_lora_original_changed_count']}",
            f"- classifier weight comparison: {checkpoint['classifier_weight_comparison']}",
        ]
    )

    lines.extend(
        [
            "",
            "## Training History",
            f"- history length: {training.get('history_length')}",
            f"- selected best epoch: {training.get('selected_best_epoch')}",
            f"- stopped early: {'yes' if training.get('stopped_early') else 'no'}",
            f"- early stopping reason: {training.get('early_stopping_reason')}",
            f"- train loss decreased: {'yes' if training.get('train_loss_decreased') else 'no'}",
            f"- val macro AUROC improved after epoch 1: {'yes' if training.get('val_macro_auroc_improved_after_epoch1') else 'no'}",
            f"- val macro AUPRC improved after epoch 1: {'yes' if training.get('val_macro_auprc_improved_after_epoch1') else 'no'}",
            f"- best epoch 1 source-like: {'yes' if training.get('best_epoch_1_source_like') else 'no'}",
            f"- later epochs changed predictions but were not selected: {'yes' if training.get('later_epochs_changed_predictions_not_selected') else 'no'}",
        ]
    )

    if training.get("later_epoch_comparison"):
        last_vs_source_val = training["later_epoch_comparison"]["last_vs_source"]["val"]["overall"]
        last_vs_source_test = training["later_epoch_comparison"]["last_vs_source"]["test"]["overall"]
        last_vs_best_val = training["later_epoch_comparison"]["last_vs_best"]["val"]["overall"]
        last_vs_best_test = training["later_epoch_comparison"]["last_vs_best"]["test"]["overall"]
        lines.extend(
            [
                f"- last vs source val mean/max abs diff: {last_vs_source_val['mean_abs_diff']:.6f} / {last_vs_source_val['max_abs_diff']:.6f}",
                f"- last vs source test mean/max abs diff: {last_vs_source_test['mean_abs_diff']:.6f} / {last_vs_source_test['max_abs_diff']:.6f}",
                f"- last vs best val mean/max abs diff: {last_vs_best_val['mean_abs_diff']:.6f} / {last_vs_best_val['max_abs_diff']:.6f}",
                f"- last vs best test mean/max abs diff: {last_vs_best_test['mean_abs_diff']:.6f} / {last_vs_best_test['max_abs_diff']:.6f}",
            ]
        )

    lines.extend(
        [
            "",
            "## Trainability Audit",
            f"- target module count: {trainability.get('target_module_count')} (expected 33)",
            f"- trainable parameters: {trainability.get('trainable_parameters')} (expected 87060)",
            f"- trainable parameter tensors: {trainability.get('trainable_parameter_tensors')} (expected 66)",
            f"- trainable names LoRA-only: {'yes' if trainability.get('expected_trainable_names_include_lora_only') else 'no'}",
            f"- classifier base accidentally trainable: {'yes' if trainability.get('classifier_base_accidentally_trainable') else 'no'}",
            f"- denseblock base accidentally trainable: {'yes' if trainability.get('denseblock_base_accidentally_trainable') else 'no'}",
        ]
    )

    lines.extend(["", "## Support Set"])
    lines.append("| Label | Positives | Negatives | Masked | n_valid | Positive Fraction | Warnings |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for label in LABELS:
        item = support["per_label"][label]
        positive_fraction = "n/a" if item["positive_fraction"] is None else f"{item['positive_fraction']:.3f}"
        warnings_text = "none" if not item["warnings"] else ", ".join(item["warnings"])
        lines.append(
            f"| {label} | {item['positives']} | {item['negatives']} | {item['masked']} | "
            f"{item['n_valid']} | {positive_fraction} | {warnings_text} |"
        )

    lines.extend(
        [
            "",
            "## Decision Notes",
        ]
    )
    for reason in report["decision_reasons"]:
        lines.append(f"- {reason}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = resolve_args(parse_args())

    refuse_to_overwrite(
        [args.diagnostic_md, args.diagnostic_json, args.prediction_diff_csv],
        allow_overwrite=args.allow_overwrite,
    )

    require_file(args.source_checkpoint, "source checkpoint")
    require_file(args.lora_checkpoint, "LoRA checkpoint")
    require_file(args.source_val_predictions, "source val predictions")
    require_file(args.source_test_predictions, "source test predictions")
    require_file(args.lora_val_predictions, "LoRA val predictions")
    require_file(args.lora_test_predictions, "LoRA test predictions")
    require_file(args.support_manifest, "support manifest")

    run_report = load_json(args.run_report_json) if args.run_report_json.exists() else None

    source_checkpoint = load_checkpoint_payload(args.source_checkpoint)
    source_state_dict = extract_state_dict_from_checkpoint(source_checkpoint, args.source_checkpoint)
    checkpoint_inspection = inspect_lora_checkpoint(args.lora_checkpoint, source_state_dict)

    source_predictions = {
        "val": load_prediction_csv(args.source_val_predictions),
        "test": load_prediction_csv(args.source_test_predictions),
    }
    best_predictions = {
        "val": load_prediction_csv(args.lora_val_predictions),
        "test": load_prediction_csv(args.lora_test_predictions),
    }

    best_vs_source = {
        "val": compare_prediction_frames(
            source_predictions["val"],
            best_predictions["val"],
            split_name="val",
            comparison_name="source_vs_best",
        ),
        "test": compare_prediction_frames(
            source_predictions["test"],
            best_predictions["test"],
            split_name="test",
            comparison_name="source_vs_best",
        ),
    }
    best_vs_source_summary = aggregate_best_vs_source(best_vs_source)

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    last_checkpoint_path = None
    if run_report is not None and run_report.get("last_checkpoint"):
        last_checkpoint_path = Path(run_report["last_checkpoint"]).resolve()
    last_eval = evaluate_last_checkpoint_if_possible(last_checkpoint_path, run_report, device)

    training_history = analyze_training_history(
        run_report=run_report,
        best_vs_source=best_vs_source,
        last_eval=last_eval,
        source_predictions=source_predictions,
        best_predictions=best_predictions,
    )
    trainability_audit = audit_trainability(run_report)
    support_analysis = analyze_support_manifest(args.support_manifest)

    final_diagnosis, recommended_next_action, decision_reasons = choose_final_diagnosis(
        checkpoint_inspection=checkpoint_inspection,
        training_history=training_history,
        trainability_audit=trainability_audit,
        support_analysis=support_analysis,
        best_vs_source_summary=best_vs_source_summary,
    )

    report = {
        "run_name": args.run_name,
        "base_dir": str(args.base_dir),
        "source_checkpoint": str(args.source_checkpoint),
        "lora_checkpoint": str(args.lora_checkpoint),
        "device_for_last_checkpoint_eval": str(device),
        "prediction_difference": {
            "best_vs_source": best_vs_source,
        },
        "checkpoint_inspection": checkpoint_inspection,
        "training_history_analysis": training_history,
        "trainability_audit": trainability_audit,
        "support_set_analysis": support_analysis,
        "last_checkpoint_evaluation": None if last_eval is None else {
            key: value
            for key, value in last_eval.items()
            if key not in {"val_predictions", "test_predictions"}
        },
        "final_diagnosis": final_diagnosis,
        "recommended_next_action": recommended_next_action,
        "decision_reasons": decision_reasons,
        "summary": {
            "overall_mean_abs_prediction_diff_vs_source_only": best_vs_source_summary["overall_mean_abs_diff_across_splits"],
            "overall_max_abs_prediction_diff_vs_source_only": best_vs_source_summary["overall_max_abs_diff_across_splits"],
            "lora_tensors_nonzero": checkpoint_inspection["lora_nonzero"],
            "train_loss_decreased": bool(training_history.get("train_loss_decreased", False)),
            "diagnostic_report_path": str(args.diagnostic_md),
        },
    }

    prediction_diff_rows = (
        build_prediction_diff_rows(best_vs_source["val"])
        + build_prediction_diff_rows(best_vs_source["test"])
    )
    prediction_diff_df = pd.DataFrame(prediction_diff_rows)

    ensure_parent(args.prediction_diff_csv)
    prediction_diff_df.to_csv(args.prediction_diff_csv, index=False)

    ensure_parent(args.diagnostic_json)
    ensure_parent(args.diagnostic_md)
    save_json(json_ready(report), args.diagnostic_json)
    args.diagnostic_md.write_text(build_markdown_report(report), encoding="utf-8")

    print(f"final diagnosis: {final_diagnosis}")
    print(
        "overall mean/max prediction difference vs source-only: "
        f"{report['summary']['overall_mean_abs_prediction_diff_vs_source_only']:.6f} / "
        f"{report['summary']['overall_max_abs_prediction_diff_vs_source_only']:.6f}"
    )
    print(f"LoRA tensors nonzero: {'yes' if report['summary']['lora_tensors_nonzero'] else 'no'}")
    print(f"train loss decreased: {'yes' if report['summary']['train_loss_decreased'] else 'no'}")
    print(f"recommended next single action: {recommended_next_action}")
    print(f"path to diagnostic report: {args.diagnostic_md}")


if __name__ == "__main__":
    main()
