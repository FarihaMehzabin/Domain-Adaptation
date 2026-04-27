#!/usr/bin/env python3
"""Audit Policy B manifests and reevaluate existing prediction files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
MANIFEST_FILES = {
    "train_pool": "manifests/mimic_common5_policyB_train_pool.csv",
    "val": "manifests/mimic_common5_policyB_val.csv",
    "test": "manifests/mimic_common5_policyB_test.csv",
}
SUPPORT_FILES = {
    "k5": "manifests/mimic_common5_policyB_support_k5_seed2027.csv",
    "k20": "manifests/mimic_common5_policyB_support_k20_seed2027.csv",
}
PREDICTION_SPECS = [
    {
        "run_id": "nih_to_mimic",
        "display_name": "no_adaptation",
        "retrain_required": False,
        "prediction_files": {
            "val": "outputs/nih_to_mimic_val_predictions.csv",
            "test": "outputs/nih_to_mimic_test_predictions.csv",
        },
    },
    {
        "run_id": "head_only_k5_seed2027",
        "display_name": "head_only_k5",
        "retrain_required": True,
        "prediction_files": {
            "val": "outputs/head_only_k5_seed2027_val_predictions.csv",
            "test": "outputs/head_only_k5_seed2027_test_predictions.csv",
        },
    },
    {
        "run_id": "head_only_k20_seed2027",
        "display_name": "head_only_k20",
        "retrain_required": True,
        "prediction_files": {
            "val": "outputs/head_only_k20_seed2027_val_predictions.csv",
            "test": "outputs/head_only_k20_seed2027_test_predictions.csv",
        },
    },
    {
        "run_id": "full_ft_k5_seed2027",
        "display_name": "full_ft_k5",
        "retrain_required": True,
        "prediction_files": {
            "val": "outputs/full_ft_k5_seed2027_val_predictions.csv",
            "test": "outputs/full_ft_k5_seed2027_test_predictions.csv",
        },
    },
    {
        "run_id": "full_ft_k20_seed2027",
        "display_name": "full_ft_k20",
        "retrain_required": True,
        "prediction_files": {
            "val": "outputs/full_ft_k20_seed2027_val_predictions.csv",
            "test": "outputs/full_ft_k20_seed2027_test_predictions.csv",
        },
    },
]
PATH_COLUMNS = ["abs_path", "rel_path", "path", "image_path", "manifest_image_path", "filepath", "file_path"]
PRED_COLUMN_CANDIDATES = {label: [f"pred_{label}", label] for label in LABELS}
TRUE_COLUMN_CANDIDATES = {label: [f"true_{label}", label] for label in LABELS}
REPORT_PATHS = {
    "audit_md": "reports/policyB_manifest_audit.md",
    "audit_json": "reports/policyB_manifest_audit.json",
    "eval_md": "reports/policyB_existing_predictions_eval.md",
    "eval_csv": "reports/policyB_existing_predictions_eval.csv",
    "eval_json": "reports/policyB_existing_predictions_eval.json",
}


class StageFailure(RuntimeError):
    """Raised when the Policy B audit cannot continue safely."""


@dataclass(frozen=True)
class MatchResult:
    prediction_indices: list[int]
    manifest_indices: list[int]
    method: str
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Policy B manifests and evaluate existing predictions.")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting Policy B report outputs.")
    return parser.parse_args()


def print_line(message: str) -> None:
    print(message, flush=True)


def read_csv_checked(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise StageFailure(f"Could not read CSV {path}: {exc}") from exc


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is pd.NA:
        return None
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(data), handle, indent=2, sort_keys=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise StageFailure("Refusing to write an empty evaluation CSV.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_ready(row.get(key)) for key in fieldnames})


def ensure_report_paths(root: Path, overwrite: bool) -> dict[str, Path]:
    paths = {key: root / relative for key, relative in REPORT_PATHS.items()}
    if overwrite:
        return paths
    existing = [str(path.resolve()) for path in paths.values() if path.exists()]
    if existing:
        raise StageFailure(
            "Refusing to overwrite existing Policy B reports. Pass --overwrite to replace: "
            f"{existing}"
        )
    return paths


def classify_raw_value(value: Any) -> str:
    if pd.isna(value):
        return "blank"
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return "blank"
        try:
            value = float(text)
        except ValueError as exc:
            raise StageFailure(f"Unexpected raw label string value: {value!r}") from exc
    if isinstance(value, (np.integer, int)):
        numeric_value = int(value)
    elif isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return "blank"
        if not float(value).is_integer():
            raise StageFailure(f"Unexpected non-integer raw label value: {value!r}")
        numeric_value = int(value)
    else:
        raise StageFailure(f"Unexpected raw label value type: {type(value).__name__} ({value!r})")
    if numeric_value == 1:
        return "1"
    if numeric_value == 0:
        return "0"
    if numeric_value == -1:
        return "-1"
    raise StageFailure(f"Unexpected raw label value: {numeric_value!r}")


def normalize_path_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.replace("\\", "/")


def normalize_key_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return None
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), ".15g")
    text = str(value).strip()
    return text or None


def choose_column(columns: list[str], candidates: list[str], description: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise StageFailure(f"Missing {description}. Expected one of: {candidates}")


def get_prediction_column(predictions: pd.DataFrame, label: str) -> str:
    return choose_column(list(predictions.columns), PRED_COLUMN_CANDIDATES[label], f"prediction column for {label}")


def get_true_column_if_present(predictions: pd.DataFrame, label: str) -> str | None:
    for candidate in TRUE_COLUMN_CANDIDATES[label]:
        if candidate in predictions.columns:
            return candidate
    return None


def add_key_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    enriched = dataframe.copy()
    if "dicom_id" in enriched.columns:
        enriched["_key_dicom_id"] = enriched["dicom_id"].map(normalize_key_value)
    if "study_id" in enriched.columns:
        enriched["_key_study_id"] = enriched["study_id"].map(normalize_key_value)
    for column in PATH_COLUMNS:
        if column in enriched.columns:
            enriched[f"_key_{column}"] = enriched[column].map(normalize_path_string)
    return enriched


def build_tuple_keys(dataframe: pd.DataFrame, columns: list[str]) -> pd.Series:
    tuple_values: list[tuple[str, ...] | None] = []
    for row_values in dataframe[columns].itertuples(index=False, name=None):
        normalized = [normalize_key_value(value) for value in row_values]
        if any(value is None for value in normalized):
            tuple_values.append(None)
        else:
            tuple_values.append(tuple(normalized))
    return pd.Series(tuple_values, index=dataframe.index, dtype="object")


def try_match_with_keys(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    prediction_keys: pd.Series,
    manifest_keys: pd.Series,
    method: str,
) -> MatchResult | None:
    if prediction_keys.isna().any() or manifest_keys.isna().any():
        return None
    if prediction_keys.duplicated().any() or manifest_keys.duplicated().any():
        return None
    left = pd.DataFrame({"_pred_index": predictions.index.to_numpy(), "_key": prediction_keys.to_numpy()})
    right = pd.DataFrame({"_manifest_index": manifest.index.to_numpy(), "_key": manifest_keys.to_numpy()})
    merged = left.merge(right, on="_key", how="outer", indicator=True, validate="1:1")
    if not merged["_merge"].eq("both").all():
        return None
    merged = merged.sort_values("_pred_index", kind="stable")
    return MatchResult(
        prediction_indices=merged["_pred_index"].astype(int).tolist(),
        manifest_indices=merged["_manifest_index"].astype(int).tolist(),
        method=method,
        warnings=[],
    )


def match_predictions_to_manifest(predictions: pd.DataFrame, manifest: pd.DataFrame, split: str) -> MatchResult:
    pred = add_key_columns(predictions)
    man = add_key_columns(manifest)

    if "_key_dicom_id" in pred.columns and "_key_dicom_id" in man.columns:
        match = try_match_with_keys(pred, man, pred["_key_dicom_id"], man["_key_dicom_id"], "dicom_id")
        if match is not None:
            return match

    if {"study_id", "dicom_id"}.issubset(pred.columns) and {"study_id", "dicom_id"}.issubset(man.columns):
        pred_tuple = build_tuple_keys(pred, ["study_id", "dicom_id"])
        man_tuple = build_tuple_keys(man, ["study_id", "dicom_id"])
        match = try_match_with_keys(pred, man, pred_tuple, man_tuple, "study_id+dicom_id")
        if match is not None:
            return match

    path_pair_candidates = [
        ("manifest_image_path", "abs_path", "manifest_image_path->abs_path"),
        ("image_path", "abs_path", "image_path->abs_path"),
        ("path", "abs_path", "path->abs_path"),
        ("manifest_image_path", "rel_path", "manifest_image_path->rel_path"),
        ("image_path", "rel_path", "image_path->rel_path"),
        ("path", "rel_path", "path->rel_path"),
    ]
    for pred_column, manifest_column, method in path_pair_candidates:
        pred_key = f"_key_{pred_column}"
        man_key = f"_key_{manifest_column}"
        if pred_key in pred.columns and man_key in man.columns:
            match = try_match_with_keys(pred, man, pred[pred_key], man[man_key], method)
            if match is not None:
                return match

    if len(predictions) != len(manifest):
        raise StageFailure(
            f"Could not match prediction rows for split={split} by stable keys, and row-order fallback is impossible "
            f"because prediction rows={len(predictions)} but manifest rows={len(manifest)}."
        )
    return MatchResult(
        prediction_indices=predictions.index.astype(int).tolist(),
        manifest_indices=manifest.index.astype(int).tolist(),
        method="row_order",
        warnings=["Used row-order fallback because no stable key produced an exact bijection."],
    )


def verify_prediction_truth_columns(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    match: MatchResult,
    run_id: str,
    split: str,
) -> None:
    aligned_predictions = predictions.loc[match.prediction_indices].reset_index(drop=True)
    aligned_manifest = manifest.loc[match.manifest_indices].reset_index(drop=True)
    mismatches: list[str] = []
    for label in LABELS:
        true_column = get_true_column_if_present(predictions, label)
        if true_column is None:
            continue
        prediction_truth = pd.to_numeric(aligned_predictions[true_column], errors="coerce")
        manifest_truth = pd.to_numeric(aligned_manifest[label], errors="coerce")
        unequal = prediction_truth.isna() | manifest_truth.isna() | (prediction_truth.astype(int) != manifest_truth.astype(int))
        if unequal.any():
            examples = pd.DataFrame(
                {
                    "dicom_id": aligned_manifest["dicom_id"],
                    "study_id": aligned_manifest["study_id"],
                    "prediction_value": aligned_predictions[true_column],
                    "manifest_value": aligned_manifest[label],
                }
            ).loc[unequal].head(5)
            mismatches.append(f"{label}: {examples.to_dict(orient='records')}")
    if mismatches:
        raise StageFailure(
            f"Prediction truth columns do not align with the Policy B manifest for run={run_id}, split={split}. "
            f"Match method={match.method}. Details: {mismatches}"
        )


def validate_manifest(dataframe: pd.DataFrame, split_name: str, warnings: list[str]) -> None:
    required = ["dicom_id", "subject_id", "study_id", "split", "label_policy", "label_set"]
    for label in LABELS:
        required.extend([label, f"{label}_mask", f"{label}_raw"])
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise StageFailure(f"Manifest split={split_name} is missing required columns: {missing}")
    if dataframe["label_policy"].astype("string").str.strip().nunique(dropna=False) != 1:
        raise StageFailure(f"Manifest split={split_name} has inconsistent label_policy values.")
    if dataframe["label_policy"].astype("string").str.strip().iloc[0] != "uignore_blankzero":
        raise StageFailure(f"Manifest split={split_name} is not tagged with label_policy=uignore_blankzero.")
    if dataframe["label_set"].astype("string").str.strip().nunique(dropna=False) != 1:
        raise StageFailure(f"Manifest split={split_name} has inconsistent label_set values.")
    if dataframe["label_set"].astype("string").str.strip().iloc[0] != "common5":
        raise StageFailure(f"Manifest split={split_name} is not tagged with label_set=common5.")
    split_values = dataframe["split"].astype("string").str.strip().unique().tolist()
    if split_values != [split_name]:
        raise StageFailure(f"Manifest split={split_name} contains unexpected split values: {split_values}")
    dicom_values = dataframe["dicom_id"].astype("string").str.strip()
    if dicom_values.isna().any() or (dicom_values == "").any():
        raise StageFailure(f"Manifest split={split_name} contains blank dicom_id values.")
    if dicom_values.duplicated().any():
        raise StageFailure(f"Manifest split={split_name} contains duplicated dicom_id values.")

    for label in LABELS:
        label_values = pd.to_numeric(dataframe[label], errors="coerce")
        mask_values = pd.to_numeric(dataframe[f"{label}_mask"], errors="coerce")
        if label_values.isna().any() or mask_values.isna().any():
            raise StageFailure(f"Manifest split={split_name} has NaN values in {label} or {label}_mask.")
        if (~label_values.isin([0, 1]) | ~mask_values.isin([0, 1])).any():
            raise StageFailure(f"Manifest split={split_name} has non-binary values in {label} or {label}_mask.")
        if ((label_values == 1) & (mask_values == 0)).any():
            raise StageFailure(f"Manifest split={split_name} has label==1 with mask==0 for {label}.")
        blank_rows = dataframe[f"{label}_raw"].map(classify_raw_value).eq("blank")
        if (blank_rows & (mask_values != 1)).any():
            warnings.append(
                f"Manifest split={split_name} has blank raw values that do not map to mask=1 for {label}."
            )


def count_overlap(series: pd.Series) -> set[str]:
    normalized = series.dropna().astype("string").str.strip()
    normalized = normalized[normalized != ""]
    return set(normalized.tolist())


def build_manifest_audit(manifests: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    split_audit: dict[str, Any] = {}
    for split_name, manifest in manifests.items():
        validate_manifest(manifest, split_name, warnings)
        raw_counts: dict[str, Any] = {}
        final_counts: dict[str, Any] = {}
        raw_frame = pd.DataFrame(index=manifest.index)
        mask_frame = pd.DataFrame(index=manifest.index)
        for label in LABELS:
            raw_categories = manifest[f"{label}_raw"].map(classify_raw_value)
            raw_frame[label] = raw_categories
            raw_counts[label] = {
                "1": int(raw_categories.eq("1").sum()),
                "0": int(raw_categories.eq("0").sum()),
                "-1": int(raw_categories.eq("-1").sum()),
                "blank": int(raw_categories.eq("blank").sum()),
            }
            label_values = pd.to_numeric(manifest[label], errors="coerce").astype(int)
            mask_values = pd.to_numeric(manifest[f"{label}_mask"], errors="coerce").astype(int)
            mask_frame[label] = mask_values
            final_counts[label] = {
                "positive": int(((label_values == 1) & (mask_values == 1)).sum()),
                "negative": int(((label_values == 0) & (mask_values == 1)).sum()),
                "masked": int((mask_values == 0).sum()),
            }

        split_audit[split_name] = {
            "rows": int(len(manifest)),
            "raw_counts": raw_counts,
            "final_counts": final_counts,
            "all_five_blank_rows": int(raw_frame.eq("blank").all(axis=1).sum()),
            "rows_with_no_valid_labels": int(mask_frame.eq(0).all(axis=1).sum()),
        }

    overlap_checks: dict[str, Any] = {}
    any_overlap = False
    for left_name, right_name in combinations(["train_pool", "val", "test"], 2):
        left = manifests[left_name]
        right = manifests[right_name]
        pair_name = f"{left_name}_vs_{right_name}"
        overlap_checks[pair_name] = {}
        for column in ["subject_id", "study_id", "dicom_id"]:
            overlap_values = sorted(count_overlap(left[column]) & count_overlap(right[column]))
            overlap_checks[pair_name][column] = {
                "count": int(len(overlap_values)),
                "examples": overlap_values[:5],
            }
            any_overlap = any_overlap or bool(overlap_values)
    return {"splits": split_audit, "overlap_checks": overlap_checks, "any_overlap": any_overlap}, warnings


def summarize_support_manifest(
    support_key: str,
    support_df: pd.DataFrame,
    train_pool_df: pd.DataFrame,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if support_df.empty:
        raise StageFailure(f"Support manifest {support_key} is empty.")
    if support_df.duplicated().any():
        warnings.append(f"Support manifest {support_key} contains duplicate full rows.")
    if support_df["dicom_id"].astype("string").str.strip().duplicated().any():
        warnings.append(f"Support manifest {support_key} contains duplicate dicom_id values.")
    train_pool_dicom = count_overlap(train_pool_df["dicom_id"])
    support_dicom = count_overlap(support_df["dicom_id"])
    if not support_dicom.issubset(train_pool_dicom):
        raise StageFailure(f"Support manifest {support_key} contains rows outside the Policy B train pool.")

    summary = {
        "rows": int(len(support_df)),
        "subject_count": int(support_df["subject_id"].nunique()),
        "study_count": int(support_df["study_id"].nunique()),
        "dicom_count": int(support_df["dicom_id"].nunique()),
        "per_label": {},
    }
    for label in LABELS:
        label_values = pd.to_numeric(support_df[label], errors="coerce").astype(int)
        mask_values = pd.to_numeric(support_df[f"{label}_mask"], errors="coerce").astype(int)
        positive = int(((label_values == 1) & (mask_values == 1)).sum())
        negative = int(((label_values == 0) & (mask_values == 1)).sum())
        masked = int((mask_values == 0).sum())
        summary["per_label"][label] = {
            "positive": positive,
            "negative": negative,
            "masked": masked,
        }
    return summary, warnings


def compute_label_metrics(
    aligned_manifest: pd.DataFrame,
    aligned_predictions: pd.DataFrame,
    run_id: str,
    split: str,
    match: MatchResult,
) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    macro_aurocs: list[float] = []
    macro_auprcs: list[float] = []
    total_masked = 0
    total_possible_instances = len(aligned_manifest) * len(LABELS)
    micro_truth_parts: list[np.ndarray] = []
    micro_pred_parts: list[np.ndarray] = []

    for label in LABELS:
        pred_column = get_prediction_column(aligned_predictions, label)
        pred_values = pd.to_numeric(aligned_predictions[pred_column], errors="coerce")
        if pred_values.isna().any():
            raise StageFailure(
                f"Prediction file for run={run_id}, split={split} has non-numeric or NaN values in {pred_column}."
            )
        truth = pd.to_numeric(aligned_manifest[label], errors="coerce").astype(int)
        mask = pd.to_numeric(aligned_manifest[f"{label}_mask"], errors="coerce").astype(int)
        valid = mask == 1
        valid_truth = truth[valid].to_numpy(dtype=np.int64)
        valid_pred = pred_values[valid].to_numpy(dtype=np.float64)
        n_valid = int(valid.sum())
        positives = int(valid_truth.sum())
        negatives = int(n_valid - positives)
        masked = int((mask == 0).sum())
        total_masked += masked

        auroc = None
        auprc = None
        if positives > 0 and negatives > 0:
            auroc = float(roc_auc_score(valid_truth, valid_pred))
            auprc = float(average_precision_score(valid_truth, valid_pred))
            macro_aurocs.append(auroc)
            macro_auprcs.append(auprc)

        micro_truth_parts.append(valid_truth)
        micro_pred_parts.append(valid_pred)
        per_label[label] = {
            "n_valid": n_valid,
            "positives": positives,
            "negatives": negatives,
            "masked": masked,
            "auroc": auroc,
            "auprc": auprc,
        }

    macro_auroc = float(np.mean(macro_aurocs)) if macro_aurocs else None
    macro_auprc = float(np.mean(macro_auprcs)) if macro_auprcs else None

    micro_auroc = None
    micro_auprc = None
    micro_note: str
    if total_masked > 0:
        micro_note = "Micro metrics omitted because Policy B masks some label instances, so global micro is not directly comparable."
    else:
        micro_truth = np.concatenate(micro_truth_parts)
        micro_pred = np.concatenate(micro_pred_parts)
        if len(np.unique(micro_truth)) > 1:
            micro_auroc = float(roc_auc_score(micro_truth, micro_pred))
            micro_auprc = float(average_precision_score(micro_truth, micro_pred))
            micro_note = "Micro metrics computed on the full unmasked label-instance pool."
        else:
            micro_note = "Micro metrics omitted because the global target vector has only one class."

    return {
        "run_id": run_id,
        "split": split,
        "match_method": match.method,
        "match_warnings": list(match.warnings),
        "n_rows": int(len(aligned_manifest)),
        "per_label": per_label,
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "micro_auroc": micro_auroc,
        "micro_auprc": micro_auprc,
        "micro_note": micro_note,
        "total_masked_label_instances": int(total_masked),
        "total_possible_label_instances": int(total_possible_instances),
    }


def evaluate_predictions(
    root: Path,
    manifests: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    evaluation: dict[str, Any] = {"runs": {}, "summary": {}}
    csv_rows: list[dict[str, Any]] = []

    for spec in PREDICTION_SPECS:
        run_data: dict[str, Any] = {
            "run_id": spec["run_id"],
            "display_name": spec["display_name"],
            "retrain_required": spec["retrain_required"],
            "prediction_files": {},
            "splits": {},
        }
        for split in ["val", "test"]:
            prediction_path = root / spec["prediction_files"][split]
            run_data["prediction_files"][split] = str(prediction_path.resolve())
            if not prediction_path.exists():
                warnings.append(f"Missing prediction file for run={spec['run_id']} split={split}: {prediction_path}")
                continue
            predictions = read_csv_checked(prediction_path)
            match = match_predictions_to_manifest(predictions, manifests[split], split)
            if match.warnings:
                warnings.extend([f"{spec['run_id']} split={split}: {warning}" for warning in match.warnings])
            verify_prediction_truth_columns(predictions, manifests[split], match, spec["run_id"], split)
            aligned_predictions = predictions.loc[match.prediction_indices].reset_index(drop=True)
            aligned_manifest = manifests[split].loc[match.manifest_indices].reset_index(drop=True)
            metrics = compute_label_metrics(
                aligned_manifest=aligned_manifest,
                aligned_predictions=aligned_predictions,
                run_id=spec["run_id"],
                split=split,
                match=match,
            )
            metrics["prediction_file"] = str(prediction_path.resolve())
            run_data["splits"][split] = metrics

            overall_row = {
                "row_type": "overall",
                "run_id": spec["run_id"],
                "display_name": spec["display_name"],
                "prediction_file": str(prediction_path.resolve()),
                "split": split,
                "match_method": metrics["match_method"],
                "match_warnings": " | ".join(metrics["match_warnings"]),
                "n_rows": metrics["n_rows"],
                "macro_auroc": metrics["macro_auroc"],
                "macro_auprc": metrics["macro_auprc"],
                "micro_auroc": metrics["micro_auroc"],
                "micro_auprc": metrics["micro_auprc"],
                "micro_note": metrics["micro_note"],
            }
            csv_rows.append(overall_row)
            for label in LABELS:
                label_metrics = metrics["per_label"][label]
                csv_rows.append(
                    {
                        **overall_row,
                        "row_type": "label",
                        "label": label,
                        "n_valid": label_metrics["n_valid"],
                        "positives": label_metrics["positives"],
                        "negatives": label_metrics["negatives"],
                        "masked": label_metrics["masked"],
                        "auroc": label_metrics["auroc"],
                        "auprc": label_metrics["auprc"],
                    }
                )
        evaluation["runs"][spec["run_id"]] = run_data

    no_adapt_test = evaluation["runs"]["nih_to_mimic"]["splits"]["test"]
    test_candidates = []
    for spec in PREDICTION_SPECS:
        run_result = evaluation["runs"][spec["run_id"]]["splits"].get("test")
        if not run_result or run_result["macro_auprc"] is None:
            continue
        test_candidates.append(
            {
                "run_id": spec["run_id"],
                "display_name": spec["display_name"],
                "prediction_file": run_result["prediction_file"],
                "macro_auprc": run_result["macro_auprc"],
                "macro_auroc": run_result["macro_auroc"],
            }
        )
    test_candidates.sort(
        key=lambda item: (
            item["macro_auprc"],
            item["macro_auroc"] if item["macro_auroc"] is not None else float("-inf"),
            item["run_id"],
        ),
        reverse=True,
    )
    best_existing = test_candidates[0] if test_candidates else None
    retrain_runs = [
        {
            "run_id": spec["run_id"],
            "display_name": spec["display_name"],
            "reason": "Support labels, adaptation training, and model selection were created under the old policy.",
        }
        for spec in PREDICTION_SPECS
        if spec["retrain_required"]
    ]

    evaluation["summary"] = {
        "no_adaptation_test": {
            "prediction_file": no_adapt_test["prediction_file"],
            "macro_auroc": no_adapt_test["macro_auroc"],
            "macro_auprc": no_adapt_test["macro_auprc"],
        },
        "best_existing_test_prediction": best_existing,
        "runs_requiring_retraining": retrain_runs,
        "runs_requiring_reevaluation_only": [
            {
                "run_id": "nih_to_mimic",
                "display_name": "no_adaptation",
                "reason": "The source-trained model does not need retraining, but the target-side metrics must be reevaluated under Policy B.",
            }
        ],
    }
    return evaluation, csv_rows, warnings


def build_audit_markdown(audit: dict[str, Any], support: dict[str, Any], warnings: list[str]) -> str:
    lines = [
        "# Policy B Manifest Audit",
        "",
        "Official policy: `uignore_blankzero`",
        "",
        "## Split Rows",
        "",
        "| Split | Rows | All-five-blank rows | Rows with no valid labels |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split in ["train_pool", "val", "test"]:
        split_data = audit["splits"][split]
        lines.append(
            f"| {split} | {split_data['rows']} | {split_data['all_five_blank_rows']} | {split_data['rows_with_no_valid_labels']} |"
        )

    lines.extend(
        [
            "",
            "## Raw Label Counts",
            "",
            "| Split | Label | Raw 1 | Raw 0 | Raw -1 | Raw blank |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in ["train_pool", "val", "test"]:
        for label in LABELS:
            counts = audit["splits"][split]["raw_counts"][label]
            lines.append(
                f"| {split} | {label} | {counts['1']} | {counts['0']} | {counts['-1']} | {counts['blank']} |"
            )

    lines.extend(
        [
            "",
            "## Final Label Counts",
            "",
            "| Split | Label | Positive | Negative | Masked |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for split in ["train_pool", "val", "test"]:
        for label in LABELS:
            counts = audit["splits"][split]["final_counts"][label]
            lines.append(
                f"| {split} | {label} | {counts['positive']} | {counts['negative']} | {counts['masked']} |"
            )

    lines.extend(
        [
            "",
            "## Split Overlap Checks",
            "",
            "| Pair | Key | Overlap count | Example values |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for pair_name, pair_data in audit["overlap_checks"].items():
        for key_name in ["subject_id", "study_id", "dicom_id"]:
            lines.append(
                f"| {pair_name} | {key_name} | {pair_data[key_name]['count']} | {pair_data[key_name]['examples']} |"
            )

    lines.extend(
        [
            "",
            "## Support Sets",
            "",
            "| Support | Rows | Subjects | Studies | DICOMs |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for support_key in ["k5", "k20"]:
        summary = support[support_key]
        lines.append(
            f"| {support_key} | {summary['rows']} | {summary['subject_count']} | {summary['study_count']} | {summary['dicom_count']} |"
        )

    lines.extend(
        [
            "",
            "## Support Label Counts",
            "",
            "| Support | Label | Positive | Negative | Masked |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for support_key in ["k5", "k20"]:
        for label in LABELS:
            counts = support[support_key]["per_label"][label]
            lines.append(
                f"| {support_key} | {label} | {counts['positive']} | {counts['negative']} | {counts['masked']} |"
            )

    lines.extend(["", "## Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def build_eval_markdown(evaluation: dict[str, Any], warnings: list[str]) -> str:
    summary = evaluation["summary"]
    lines = [
        "# Policy B Existing Predictions Evaluation",
        "",
        "Existing prediction files were reevaluated against the new Policy B manifests only. No model was retrained.",
        "",
        "## Headline",
        "",
        f"- no_adaptation test macro AUROC: {summary['no_adaptation_test']['macro_auroc']:.6f}",
        f"- no_adaptation test macro AUPRC: {summary['no_adaptation_test']['macro_auprc']:.6f}",
    ]
    best = summary["best_existing_test_prediction"]
    if best is not None:
        lines.append(f"- best existing test file: `{best['prediction_file']}`")
        lines.append(f"- best existing test macro AUROC: {best['macro_auroc']:.6f}")
        lines.append(f"- best existing test macro AUPRC: {best['macro_auprc']:.6f}")

    lines.extend(
        [
            "",
            "## Overall Metrics",
            "",
            "| Run | Split | Match | Macro AUROC | Macro AUPRC | Micro AUROC | Micro AUPRC |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for spec in PREDICTION_SPECS:
        run = evaluation["runs"][spec["run_id"]]
        for split in ["val", "test"]:
            metrics = run["splits"].get(split)
            if metrics is None:
                continue
            lines.append(
                "| "
                f"{spec['display_name']} | {split} | {metrics['match_method']} | "
                f"{format_metric(metrics['macro_auroc'])} | {format_metric(metrics['macro_auprc'])} | "
                f"{format_metric(metrics['micro_auroc'])} | {format_metric(metrics['micro_auprc'])} |"
            )

    lines.extend(
        [
            "",
            "## Per-Label Metrics",
            "",
            "| Run | Split | Label | N valid | Positives | Negatives | Masked | AUROC | AUPRC |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for spec in PREDICTION_SPECS:
        run = evaluation["runs"][spec["run_id"]]
        for split in ["val", "test"]:
            metrics = run["splits"].get(split)
            if metrics is None:
                continue
            for label in LABELS:
                label_metrics = metrics["per_label"][label]
                lines.append(
                    "| "
                    f"{spec['display_name']} | {split} | {label} | "
                    f"{label_metrics['n_valid']} | {label_metrics['positives']} | {label_metrics['negatives']} | "
                    f"{label_metrics['masked']} | {format_metric(label_metrics['auroc'])} | {format_metric(label_metrics['auprc'])} |"
                )

    lines.extend(["", "## Runs Requiring Retraining", ""])
    for item in summary["runs_requiring_retraining"]:
        lines.append(f"- {item['run_id']}: {item['reason']}")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    report_paths = ensure_report_paths(root, overwrite=args.overwrite)

    manifests = {
        split: read_csv_checked(root / relative_path)
        for split, relative_path in MANIFEST_FILES.items()
    }
    audit_report, audit_warnings = build_manifest_audit(manifests)

    support_summaries: dict[str, Any] = {}
    support_warnings: list[str] = []
    for support_key, relative_path in SUPPORT_FILES.items():
        support_df = read_csv_checked(root / relative_path)
        summary, warnings = summarize_support_manifest(support_key, support_df, manifests["train_pool"])
        support_summaries[support_key] = summary
        support_warnings.extend(warnings)

    evaluation_report, evaluation_rows, evaluation_warnings = evaluate_predictions(root, manifests)

    audit_payload = {
        "status": "DONE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_policy": "uignore_blankzero",
        **audit_report,
        "support_sets": support_summaries,
        "warnings": sorted(set(audit_warnings + support_warnings)),
    }
    evaluation_payload = {
        "status": "DONE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_policy": "uignore_blankzero",
        **evaluation_report,
        "warnings": sorted(set(evaluation_warnings)),
    }

    write_json(report_paths["audit_json"], audit_payload)
    write_text(report_paths["audit_md"], build_audit_markdown(audit_report, support_summaries, audit_payload["warnings"]))
    write_json(report_paths["eval_json"], evaluation_payload)
    write_csv(report_paths["eval_csv"], evaluation_rows)
    write_text(report_paths["eval_md"], build_eval_markdown(evaluation_report, evaluation_payload["warnings"]))

    no_adapt = evaluation_payload["summary"]["no_adaptation_test"]
    best_existing = evaluation_payload["summary"]["best_existing_test_prediction"]
    print_line("Policy B manifest files created:")
    for split in ["train_pool", "val", "test"]:
        print_line(f"- {split}: {root / MANIFEST_FILES[split]}")
    print_line("split row counts:")
    for split in ["train_pool", "val", "test"]:
        print_line(f"- {split}: {audit_payload['splits'][split]['rows']}")
    print_line("support sizes:")
    print_line(f"- k5: {support_summaries['k5']['rows']}")
    print_line(f"- k20: {support_summaries['k20']['rows']}")
    print_line(f"split overlap exists: {'yes' if audit_payload['any_overlap'] else 'no'}")
    print_line(
        "no-adaptation Policy B test: "
        f"macro AUROC={no_adapt['macro_auroc']:.6f}, macro AUPRC={no_adapt['macro_auprc']:.6f}"
    )
    if best_existing is not None:
        print_line(
            "best existing prediction file under Policy B: "
            f"{best_existing['prediction_file']} "
            f"(macro AUROC={best_existing['macro_auroc']:.6f}, macro AUPRC={best_existing['macro_auprc']:.6f})"
        )
    print_line("runs that must be retrained under Policy B:")
    for item in evaluation_payload["summary"]["runs_requiring_retraining"]:
        print_line(f"- {item['run_id']}: {item['reason']}")
    print_line("generated reports:")
    print_line(f"- {report_paths['audit_md']}")
    print_line(f"- {report_paths['audit_json']}")
    print_line(f"- {report_paths['eval_md']}")
    print_line(f"- {report_paths['eval_csv']}")
    print_line(f"- {report_paths['eval_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
