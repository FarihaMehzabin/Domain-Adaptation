#!/usr/bin/env python3
"""Audit MIMIC label-policy sensitivity using existing manifests and predictions only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
IMAGE_PATH_COLUMNS = ["abs_path", "image_path", "file_path", "filepath", "path", "rel_path"]
RAW_COLUMN_CANDIDATES = {
    "Atelectasis": ["raw_Atelectasis", "Atelectasis"],
    "Cardiomegaly": ["raw_Cardiomegaly", "Cardiomegaly"],
    "Consolidation": ["raw_Consolidation", "Consolidation"],
    "Edema": ["raw_Edema", "Edema"],
    "Effusion": ["raw_Pleural_Effusion", "Pleural Effusion", "raw_Effusion", "Effusion"],
}
PRED_COLUMN_CANDIDATES = {label: [f"pred_{label}", label] for label in LABELS}
TRUE_COLUMN_CANDIDATES = {label: [f"true_{label}", label] for label in LABELS}
SPLITS = ["val", "test"]
POLICIES = OrderedDict(
    [
        (
            "current_uzero_blankzero",
            {
                "short_name": "Policy A",
                "table_name": "A",
                "description": "1->1/m1, 0->0/m1, -1->0/m1, blank->0/m1",
                "uncertain_label": 0,
                "uncertain_mask": 1,
                "blank_label": 0,
                "blank_mask": 1,
            },
        ),
        (
            "uignore_blankzero",
            {
                "short_name": "Policy B",
                "table_name": "B",
                "description": "1->1/m1, 0->0/m1, -1->0/m0, blank->0/m1",
                "uncertain_label": 0,
                "uncertain_mask": 0,
                "blank_label": 0,
                "blank_mask": 1,
            },
        ),
        (
            "uignore_blankignore",
            {
                "short_name": "Policy C",
                "table_name": "C",
                "description": "1->1/m1, 0->0/m1, -1->0/m0, blank->0/m0",
                "uncertain_label": 0,
                "uncertain_mask": 0,
                "blank_label": 0,
                "blank_mask": 0,
            },
        ),
        (
            "uone_blankzero",
            {
                "short_name": "Policy D",
                "table_name": "D",
                "description": "1->1/m1, 0->0/m1, -1->1/m1, blank->0/m1",
                "uncertain_label": 1,
                "uncertain_mask": 1,
                "blank_label": 0,
                "blank_mask": 1,
            },
        ),
    ]
)
RUN_SPECS = [
    {
        "run_id": "nih_to_mimic",
        "display_name": "no adaptation",
        "prediction_files": {
            "val": "outputs/nih_to_mimic_val_predictions.csv",
            "test": "outputs/nih_to_mimic_test_predictions.csv",
        },
        "report_json": "reports/mini_stage_d_nih_to_mimic.json",
        "headline_eligible": True,
    },
    {
        "run_id": "head_only_k5_seed2027",
        "display_name": "head_only_k5",
        "prediction_files": {
            "val": "outputs/head_only_k5_seed2027_val_predictions.csv",
            "test": "outputs/head_only_k5_seed2027_test_predictions.csv",
        },
        "report_json": "reports/head_only_k5_seed2027.json",
        "headline_eligible": True,
    },
    {
        "run_id": "head_only_k20_seed2027",
        "display_name": "head_only_k20",
        "prediction_files": {
            "val": "outputs/head_only_k20_seed2027_val_predictions.csv",
            "test": "outputs/head_only_k20_seed2027_test_predictions.csv",
        },
        "report_json": "reports/head_only_k20_seed2027.json",
        "headline_eligible": True,
    },
    {
        "run_id": "full_ft_k5_seed2027",
        "display_name": "full_ft_k5",
        "prediction_files": {
            "val": "outputs/full_ft_k5_seed2027_val_predictions.csv",
            "test": "outputs/full_ft_k5_seed2027_test_predictions.csv",
        },
        "report_json": "reports/full_ft_k5_seed2027.json",
        "headline_eligible": True,
    },
    {
        "run_id": "full_ft_k20_seed2027",
        "display_name": "full_ft_k20",
        "prediction_files": {
            "val": "outputs/full_ft_k20_seed2027_val_predictions.csv",
            "test": "outputs/full_ft_k20_seed2027_test_predictions.csv",
        },
        "report_json": "reports/full_ft_k20_seed2027.json",
        "headline_eligible": False,
    },
]
OUTPUT_PATHS = {
    "sensitivity_md": "reports/label_policy_sensitivity.md",
    "sensitivity_csv": "reports/label_policy_sensitivity.csv",
    "sensitivity_json": "reports/label_policy_sensitivity.json",
    "verification_md": "reports/full_ft_k20_verification.md",
    "verification_json": "reports/full_ft_k20_verification.json",
}


class StageFailure(RuntimeError):
    """Raised when the audit cannot safely continue."""


@dataclass(frozen=True)
class MatchResult:
    prediction_indices: list[int]
    manifest_indices: list[int]
    method: str
    warnings: list[str]
    verification_note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit MIMIC label-policy sensitivity using existing manifests and predictions only."
    )
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    return parser.parse_args()


def read_csv_checked(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise StageFailure(f"Could not read CSV {path}: {exc}") from exc


def read_json_checked(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise StageFailure(f"Could not read JSON {path}: {exc}") from exc


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


def ensure_output_paths_do_not_exist(paths: list[Path]) -> None:
    existing = [str(path.resolve()) for path in paths if path.exists()]
    if existing:
        raise StageFailure(
            "Refusing to overwrite existing report files. Remove these paths first if you want to rerun: "
            f"{existing}"
        )


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(data), handle, indent=2)


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise StageFailure("No CSV rows were generated for the label-policy sensitivity report.")
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


def maybe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


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
        if not text:
            return None
        return text
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return None
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), ".15g")
    return str(value).strip()


def classify_raw_value(value: Any) -> str:
    if pd.isna(value):
        return "blank"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"nan", "none", "null"}:
            return "blank"
        try:
            numeric = float(stripped)
        except ValueError as exc:
            raise StageFailure(f"Unexpected raw label string value: {value!r}") from exc
        value = numeric
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
        return "positive"
    if numeric_value == 0:
        return "negative"
    if numeric_value == -1:
        return "uncertain"
    raise StageFailure(f"Unexpected raw label value: {numeric_value!r}")


def choose_column(columns: list[str], candidates: list[str], description: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise StageFailure(f"Missing {description}. Expected one of: {candidates}")


def get_prediction_column(predictions: pd.DataFrame, label: str) -> str:
    columns = list(predictions.columns)
    return choose_column(columns, PRED_COLUMN_CANDIDATES[label], f"prediction column for {label}")


def get_true_column_if_present(predictions: pd.DataFrame, label: str) -> str | None:
    columns = list(predictions.columns)
    for candidate in TRUE_COLUMN_CANDIDATES[label]:
        if candidate in columns:
            return candidate
    return None


def extract_raw_series(dataframe: pd.DataFrame, label: str) -> pd.Series:
    column = choose_column(list(dataframe.columns), RAW_COLUMN_CANDIDATES[label], f"raw label column for {label}")
    return dataframe[column]


def build_raw_table(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], int]:
    raw_categories: dict[str, pd.Series] = {}
    label_summary: dict[str, Any] = {}
    for label in LABELS:
        series = extract_raw_series(dataframe, label)
        categories = series.map(classify_raw_value)
        raw_categories[label] = categories
        label_summary[label] = {
            "positive_raw": int((categories == "positive").sum()),
            "negative_raw": int((categories == "negative").sum()),
            "uncertain_raw": int((categories == "uncertain").sum()),
            "blank_raw": int((categories == "blank").sum()),
        }
    raw_df = pd.DataFrame(raw_categories)
    all_blank_count = int(raw_df.eq("blank").all(axis=1).sum())
    return raw_df, label_summary, all_blank_count


def validate_current_binary_labels(dataframe: pd.DataFrame) -> None:
    for label in LABELS:
        if label not in dataframe.columns:
            raise StageFailure(f"Manifest is missing converted label column: {label}")
        numeric = pd.to_numeric(dataframe[label], errors="coerce")
        invalid_mask = numeric.isna() | ~numeric.isin([0, 1])
        if invalid_mask.any():
            bad_values = sorted(pd.unique(dataframe.loc[invalid_mask, label]).tolist())
            raise StageFailure(f"Manifest has non-binary converted values for {label}: {bad_values}")


def validate_required_manifest_keys(dataframe: pd.DataFrame, path: Path) -> None:
    required = ["subject_id", "study_id", "dicom_id"]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise StageFailure(f"Manifest {path} is missing required key columns: {missing}")


def add_key_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    enriched = dataframe.copy()
    enriched["_row_index"] = np.arange(len(enriched), dtype=np.int64)
    if "dicom_id" in enriched.columns:
        enriched["_key_dicom_id"] = enriched["dicom_id"].map(normalize_key_value)
    if "study_id" in enriched.columns:
        enriched["_key_study_id"] = enriched["study_id"].map(normalize_key_value)
    if "abs_path" in enriched.columns:
        enriched["_key_abs_path"] = enriched["abs_path"].map(normalize_path_string)
    if "rel_path" in enriched.columns:
        enriched["_key_rel_path"] = enriched["rel_path"].map(normalize_path_string)
    for column in IMAGE_PATH_COLUMNS:
        if column in enriched.columns and f"_key_path_{column}" not in enriched.columns:
            enriched[f"_key_path_{column}"] = enriched[column].map(normalize_path_string)
    return enriched


def verify_common5_against_target_raw(
    common5_df: pd.DataFrame,
    target_df: pd.DataFrame,
    common5_path: Path,
    target_path: Path,
) -> dict[str, Any]:
    common5 = add_key_columns(common5_df)
    target = add_key_columns(target_df)
    if "_key_dicom_id" not in common5.columns or "_key_dicom_id" not in target.columns:
        raise StageFailure("Could not cross-check raw manifests because dicom_id is missing.")
    if common5["_key_dicom_id"].isna().any() or target["_key_dicom_id"].isna().any():
        raise StageFailure("Could not cross-check raw manifests because some dicom_id values are blank.")
    if common5["_key_dicom_id"].duplicated().any():
        raise StageFailure(f"{common5_path} has duplicated dicom_id values; raw cross-check is ambiguous.")
    if target["_key_dicom_id"].duplicated().any():
        raise StageFailure(f"{target_path} has duplicated dicom_id values; raw cross-check is ambiguous.")

    common_raw_columns = {label: extract_raw_column_name(common5_df, label) for label in LABELS}
    target_raw_columns = {label: extract_raw_column_name(target_df, label) for label in LABELS}

    common_rows = common5[
        ["_key_dicom_id", "_key_study_id", "_row_index"] + list(common_raw_columns.values())
    ].rename(
        columns={
            "_row_index": "_common5_row_index",
            **{column: f"{label}_common5_raw" for label, column in common_raw_columns.items()},
        }
    )
    target_rows = target[
        ["_key_dicom_id", "_key_study_id", "_row_index"] + list(target_raw_columns.values())
    ].rename(
        columns={
            "_row_index": "_target_row_index",
            **{column: f"{label}_target_raw" for label, column in target_raw_columns.items()},
        }
    )
    merged = common_rows.merge(
        target_rows,
        on=["_key_dicom_id", "_key_study_id"],
        how="outer",
        indicator=True,
        validate="1:1",
    )
    if not merged["_merge"].eq("both").all():
        mismatch_counts = merged["_merge"].value_counts(dropna=False).to_dict()
        raise StageFailure(
            f"Common5 manifest {common5_path} and target raw manifest {target_path} do not contain the same rows: "
            f"{mismatch_counts}"
        )

    label_checks: dict[str, Any] = {}
    for label in LABELS:
        left_column = f"{label}_common5_raw"
        right_column = f"{label}_target_raw"
        left = merged[left_column].map(classify_raw_value)
        right = merged[right_column].map(classify_raw_value)
        mismatches = left != right
        label_checks[label] = {
            "common5_raw_column": common_raw_columns[label],
            "target_raw_column": target_raw_columns[label],
            "matches": bool((~mismatches).all()),
            "mismatch_count": int(mismatches.sum()),
        }
        if mismatches.any():
            examples = merged.loc[mismatches, ["_key_dicom_id", left_column, right_column]].head(5)
            raise StageFailure(
                f"Raw label mismatch between {common5_path} and {target_path} for {label}. "
                f"Examples: {examples.to_dict(orient='records')}"
            )

    return {
        "common5_manifest": str(common5_path.resolve()),
        "target_manifest": str(target_path.resolve()),
        "rows_checked": int(len(merged)),
        "labels": label_checks,
    }


def extract_raw_column_name(dataframe: pd.DataFrame, label: str) -> str:
    return choose_column(list(dataframe.columns), RAW_COLUMN_CANDIDATES[label], f"raw label column for {label}")


def prepare_manifest_bundle(root: Path, split: str) -> dict[str, Any]:
    common5_path = root / "manifests" / f"mimic_common5_{split}.csv"
    if not common5_path.exists():
        raise StageFailure(f"Missing required manifest: {common5_path}")

    common5_df = read_csv_checked(common5_path)
    validate_required_manifest_keys(common5_df, common5_path)
    validate_current_binary_labels(common5_df)
    common5_with_keys = add_key_columns(common5_df)
    raw_df, raw_summary, all_blank_count = build_raw_table(common5_df)

    target_candidates = []
    if split == "val":
        target_candidates.append(root / "manifests" / "mimic_target_val.csv")
        target_candidates.append(root / "manifests" / "mimic_target_query.csv")
    else:
        target_candidates.append(root / "manifests" / f"mimic_target_{split}.csv")
    target_path = next((path for path in target_candidates if path.exists()), None)

    cross_check = None
    if target_path is not None:
        target_df = read_csv_checked(target_path)
        cross_check = verify_common5_against_target_raw(common5_df, target_df, common5_path, target_path)

    all_blank_pct = float(all_blank_count / max(len(common5_df), 1))
    converted_positive_rows = int((common5_df[LABELS].sum(axis=1) > 0).sum())

    return {
        "split": split,
        "common5_path": common5_path,
        "target_raw_path": target_path,
        "manifest": common5_with_keys,
        "raw_categories": raw_df,
        "raw_summary": raw_summary,
        "all_blank_count": all_blank_count,
        "all_blank_pct": all_blank_pct,
        "converted_positive_rows": converted_positive_rows,
        "cross_check": cross_check,
    }


def build_tuple_keys(dataframe: pd.DataFrame, columns: list[str]) -> pd.Series:
    tuple_rows: list[tuple[str, ...] | None] = []
    for values in dataframe[columns].itertuples(index=False, name=None):
        normalized = [normalize_key_value(value) for value in values]
        if any(item is None for item in normalized):
            tuple_rows.append(None)
        else:
            tuple_rows.append(tuple(normalized))
    return pd.Series(tuple_rows, index=dataframe.index, dtype="object")


def build_path_key_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    return dataframe[column].map(normalize_path_string)


def try_match_with_keys(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    pred_key_series: pd.Series,
    manifest_key_series: pd.Series,
    method: str,
) -> MatchResult | None:
    if pred_key_series.isna().any() or manifest_key_series.isna().any():
        return None
    if pred_key_series.duplicated().any() or manifest_key_series.duplicated().any():
        return None
    pred_key_frame = pd.DataFrame(
        {"_pred_index": predictions.index.to_numpy(), "_match_key": pred_key_series.to_numpy()}
    )
    manifest_key_frame = pd.DataFrame(
        {"_manifest_index": manifest.index.to_numpy(), "_match_key": manifest_key_series.to_numpy()}
    )
    merged = pred_key_frame.merge(
        manifest_key_frame,
        on="_match_key",
        how="outer",
        indicator=True,
        validate="1:1",
    )
    if not merged["_merge"].eq("both").all():
        return None
    merged = merged.sort_values("_pred_index", kind="stable")
    return MatchResult(
        prediction_indices=merged["_pred_index"].astype(int).tolist(),
        manifest_indices=merged["_manifest_index"].astype(int).tolist(),
        method=method,
        warnings=[],
        verification_note=f"Matched {len(merged)} rows by {method}.",
    )


def match_predictions_to_manifest(predictions: pd.DataFrame, manifest: pd.DataFrame, split: str) -> MatchResult:
    pred = add_key_columns(predictions)
    man = manifest

    if "_key_dicom_id" in pred.columns and "_key_dicom_id" in man.columns:
        match = try_match_with_keys(pred, man, pred["_key_dicom_id"], man["_key_dicom_id"], "dicom_id")
        if match is not None:
            return match

    if (
        "_key_dicom_id" in pred.columns
        and "_key_dicom_id" in man.columns
        and "_key_study_id" in pred.columns
        and "_key_study_id" in man.columns
    ):
        pred_tuple = build_tuple_keys(pred, ["study_id", "dicom_id"])
        man_tuple = build_tuple_keys(man, ["study_id", "dicom_id"])
        match = try_match_with_keys(pred, man, pred_tuple, man_tuple, "study_id+dicom_id")
        if match is not None:
            return match

    path_candidates = [
        ("manifest_image_path", "_key_abs_path", "manifest_image_path->abs_path"),
        ("image_path", "_key_abs_path", "image_path->abs_path"),
        ("path", "_key_abs_path", "path->abs_path"),
        ("manifest_image_path", "_key_rel_path", "manifest_image_path->rel_path"),
        ("image_path", "_key_rel_path", "image_path->rel_path"),
        ("path", "_key_rel_path", "path->rel_path"),
    ]
    for pred_column, manifest_column, method in path_candidates:
        if pred_column in pred.columns and manifest_column in man.columns:
            pred_series = build_path_key_series(pred, pred_column)
            man_series = man[manifest_column]
            match = try_match_with_keys(pred, man, pred_series, man_series, method)
            if match is not None:
                return match

    if len(pred) != len(man):
        raise StageFailure(
            f"Could not match prediction rows for split={split} by stable keys, and row-order fallback is impossible "
            f"because prediction rows={len(pred)} but manifest rows={len(man)}."
        )

    return MatchResult(
        prediction_indices=pred.index.astype(int).tolist(),
        manifest_indices=man.index.astype(int).tolist(),
        method="row_order",
        warnings=["Used row-order fallback because no stable key produced an exact bijection."],
        verification_note=f"Matched {len(pred)} rows by row order only.",
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
        pred_true = pd.to_numeric(aligned_predictions[true_column], errors="coerce")
        man_true = pd.to_numeric(aligned_manifest[label], errors="coerce")
        unequal = pred_true.isna() | man_true.isna() | (pred_true.astype(int) != man_true.astype(int))
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
            f"Prediction truth columns do not align with the manifest for run={run_id}, split={split}. "
            f"Match method={match.method}. Details: {mismatches}"
        )


def apply_policy_to_label(raw_categories: pd.Series, policy_name: str) -> tuple[np.ndarray, np.ndarray]:
    policy = POLICIES[policy_name]
    y_true = np.zeros(len(raw_categories), dtype=np.int64)
    mask = np.zeros(len(raw_categories), dtype=bool)

    positive_mask = raw_categories.eq("positive").to_numpy()
    negative_mask = raw_categories.eq("negative").to_numpy()
    uncertain_mask = raw_categories.eq("uncertain").to_numpy()
    blank_mask = raw_categories.eq("blank").to_numpy()

    y_true[positive_mask] = 1
    mask[positive_mask] = True

    y_true[negative_mask] = 0
    mask[negative_mask] = True

    y_true[uncertain_mask] = int(policy["uncertain_label"])
    mask[uncertain_mask] = bool(policy["uncertain_mask"])

    y_true[blank_mask] = int(policy["blank_label"])
    mask[blank_mask] = bool(policy["blank_mask"])
    return y_true, mask


def compute_policy_metrics(
    aligned_manifest: pd.DataFrame,
    aligned_predictions: pd.DataFrame,
    raw_categories: pd.DataFrame,
    policy_name: str,
    match: MatchResult,
    run_id: str,
    split: str,
) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    valid_aurocs: list[float] = []
    valid_auprcs: list[float] = []
    valid_labels: list[str] = []
    valid_counts: dict[str, int] = {}
    micro_truth_parts: list[np.ndarray] = []
    micro_pred_parts: list[np.ndarray] = []
    row_has_valid = np.zeros(len(aligned_manifest), dtype=bool)
    row_has_positive = np.zeros(len(aligned_manifest), dtype=bool)
    total_valid_label_instances = 0

    for label in LABELS:
        pred_column = get_prediction_column(aligned_predictions, label)
        pred_values = pd.to_numeric(aligned_predictions[pred_column], errors="coerce")
        if pred_values.isna().any():
            raise StageFailure(
                f"Prediction file for run={run_id}, split={split} has NaN or non-numeric probabilities in {pred_column}."
            )
        y_true, y_mask = apply_policy_to_label(raw_categories[label], policy_name)
        valid_truth = y_true[y_mask]
        valid_pred = pred_values.to_numpy(dtype=np.float64)[y_mask]

        n_valid = int(y_mask.sum())
        n_positive = int(valid_truth.sum())
        n_negative = int(n_valid - n_positive)
        label_valid = n_positive > 0 and n_negative > 0
        auroc = float(roc_auc_score(valid_truth, valid_pred)) if label_valid else None
        auprc = float(average_precision_score(valid_truth, valid_pred)) if label_valid else None

        if label_valid:
            valid_aurocs.append(auroc)
            valid_auprcs.append(auprc)
            valid_labels.append(label)
            valid_counts[label] = n_valid
        total_valid_label_instances += n_valid
        if n_valid > 0:
            micro_truth_parts.append(valid_truth.astype(np.int64))
            micro_pred_parts.append(valid_pred.astype(np.float64))
        row_has_valid |= y_mask
        row_has_positive |= y_mask & (y_true == 1)

        per_label[label] = {
            "n_valid": n_valid,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_uncertain_raw": int((raw_categories[label] == "uncertain").sum()),
            "n_blank_raw": int((raw_categories[label] == "blank").sum()),
            "auroc": auroc,
            "auprc": auprc,
            "is_valid_label": label_valid,
            "mask_fraction": float(n_valid / max(len(aligned_manifest), 1)),
        }

    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else None
    macro_auprc = float(np.mean(valid_auprcs)) if valid_auprcs else None

    micro_auroc = None
    micro_auprc = None
    if micro_truth_parts:
        micro_truth = np.concatenate(micro_truth_parts)
        micro_pred = np.concatenate(micro_pred_parts)
        if len(np.unique(micro_truth)) > 1:
            micro_auroc = float(roc_auc_score(micro_truth, micro_pred))
            micro_auprc = float(average_precision_score(micro_truth, micro_pred))

    total_possible_label_instances = int(len(aligned_manifest) * len(LABELS))
    rows_with_no_valid_labels = int((~row_has_valid).sum())
    rows_with_at_least_one_positive = int(row_has_positive.sum())

    macro_comparable = len(valid_labels) == len(LABELS)
    micro_comparable = total_valid_label_instances == total_possible_label_instances and micro_auroc is not None
    if total_valid_label_instances == total_possible_label_instances:
        comparability_note = "All label instances remain in the evaluation set."
    elif len(valid_labels) == len(LABELS):
        comparability_note = "All five labels remain valid, but some label instances are masked."
    else:
        comparability_note = "At least one label drops out or some label instances are masked."

    return {
        "run_id": run_id,
        "split": split,
        "policy_name": policy_name,
        "policy_short_name": POLICIES[policy_name]["short_name"],
        "match_method": match.method,
        "match_warnings": list(match.warnings),
        "match_note": match.verification_note,
        "n_rows": int(len(aligned_manifest)),
        "per_label": per_label,
        "valid_labels": valid_labels,
        "valid_counts": valid_counts,
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "micro_auroc": micro_auroc,
        "micro_auprc": micro_auprc,
        "macro_auroc_defined": macro_auroc is not None,
        "macro_auprc_defined": macro_auprc is not None,
        "micro_auroc_defined": micro_auroc is not None,
        "micro_auprc_defined": micro_auprc is not None,
        "macro_auroc_comparable": macro_comparable and macro_auroc is not None,
        "macro_auprc_comparable": macro_comparable and macro_auprc is not None,
        "micro_auroc_comparable": micro_comparable,
        "micro_auprc_comparable": micro_comparable and micro_auprc is not None,
        "comparability_note": comparability_note,
        "rows_with_no_valid_labels": rows_with_no_valid_labels,
        "rows_with_no_valid_labels_pct": float(rows_with_no_valid_labels / max(len(aligned_manifest), 1)),
        "rows_with_at_least_one_positive": rows_with_at_least_one_positive,
        "rows_with_at_least_one_positive_pct": float(rows_with_at_least_one_positive / max(len(aligned_manifest), 1)),
        "total_valid_label_instances": total_valid_label_instances,
        "total_possible_label_instances": total_possible_label_instances,
        "valid_label_fraction": float(total_valid_label_instances / max(total_possible_label_instances, 1)),
    }


def compare_metric_values(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= tolerance


def compare_report_metrics(
    report_metrics: dict[str, Any],
    recomputed_metrics: dict[str, Any],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {
        "macro_auroc_matches": compare_metric_values(report_metrics.get("macro_auroc"), recomputed_metrics["macro_auroc"]),
        "macro_auprc_matches": compare_metric_values(report_metrics.get("macro_auprc"), recomputed_metrics["macro_auprc"]),
        "per_label": {},
    }
    all_match = comparisons["macro_auroc_matches"] and comparisons["macro_auprc_matches"]
    for label in LABELS:
        report_label_metrics = report_metrics.get("per_label", {}).get(label, {})
        recomputed_label_metrics = recomputed_metrics["per_label"][label]
        label_comparison = {
            "positives_match": int(report_label_metrics.get("positives", -1)) == recomputed_label_metrics["n_positive"],
            "negatives_match": int(report_label_metrics.get("negatives", -1)) == recomputed_label_metrics["n_negative"],
            "auroc_match": compare_metric_values(report_label_metrics.get("auroc"), recomputed_label_metrics["auroc"]),
            "auprc_match": compare_metric_values(report_label_metrics.get("auprc"), recomputed_label_metrics["auprc"]),
        }
        label_ok = all(label_comparison.values())
        label_comparison["all_match"] = label_ok
        comparisons["per_label"][label] = label_comparison
        all_match = all_match and label_ok
    comparisons["all_match"] = all_match
    return comparisons


def verify_full_ft_k20(
    root: Path,
    run_results: dict[str, Any],
) -> dict[str, Any]:
    json_path = root / "reports" / "full_ft_k20_seed2027.json"
    md_path = root / "reports" / "full_ft_k20_seed2027.md"
    prediction_paths = {
        "val": root / "outputs" / "full_ft_k20_seed2027_val_predictions.csv",
        "test": root / "outputs" / "full_ft_k20_seed2027_test_predictions.csv",
    }
    result: dict[str, Any] = {
        "status": "INCOMPLETE",
        "json_report_path": str(json_path.resolve()),
        "markdown_report_path": str(md_path.resolve()),
        "prediction_files": {split: str(path.resolve()) for split, path in prediction_paths.items()},
        "json_report_exists": json_path.exists(),
        "markdown_report_exists": md_path.exists(),
        "prediction_file_exists": {split: path.exists() for split, path in prediction_paths.items()},
        "recomputed_metrics_available": "full_ft_k20_seed2027" in run_results,
        "checks": {},
        "reasons": [],
    }

    missing = []
    if not json_path.exists():
        missing.append(str(json_path.resolve()))
    if not md_path.exists():
        missing.append(str(md_path.resolve()))
    for split, path in prediction_paths.items():
        if not path.exists():
            missing.append(f"{split}: {path.resolve()}")
    if "full_ft_k20_seed2027" not in run_results:
        missing.append("recomputed metrics for full_ft_k20_seed2027")
    if missing:
        result["reasons"].append(f"Missing required artifacts: {missing}")
        return result

    report_json = read_json_checked(json_path)
    report_md = md_path.read_text(encoding="utf-8")
    recomputed = run_results["full_ft_k20_seed2027"]

    markdown_checks = {
        "contains_run_name": "full_ft_k20_seed2027" in report_md,
        "contains_status_done": "- status: DONE" in report_md,
        "contains_safe_to_continue_yes": "- safe to continue: yes" in report_md.lower(),
    }
    json_checks = {
        "status_done": report_json.get("status") == "DONE",
        "safe_to_continue": bool(report_json.get("safe_to_continue")) is True,
        "prediction_file_paths_match": all(
            str(prediction_paths[split].resolve()) == str(report_json.get("prediction_files", {}).get(split))
            for split in SPLITS
        ),
        "manifest_paths_match": (
            str((root / "manifests" / "mimic_common5_val.csv").resolve()) == str(report_json.get("val_manifest"))
            and str((root / "manifests" / "mimic_common5_test.csv").resolve()) == str(report_json.get("test_manifest"))
        ),
        "label_order_match": report_json.get("label_order") == LABELS,
    }
    split_checks = {}
    all_match = all(markdown_checks.values()) and all(json_checks.values())
    for split in SPLITS:
        report_metrics = report_json.get(f"{split}_metrics")
        if not isinstance(report_metrics, dict):
            split_checks[split] = {"present": False, "all_match": False}
            all_match = False
            continue
        metrics_comparison = compare_report_metrics(report_metrics, recomputed[split]["current_uzero_blankzero"])
        split_checks[split] = {
            "present": True,
            "metrics_comparison": metrics_comparison,
            "match_method": recomputed[split]["current_uzero_blankzero"]["match_method"],
        }
        all_match = all_match and metrics_comparison["all_match"]

    result["checks"] = {
        "markdown": markdown_checks,
        "json": json_checks,
        "splits": split_checks,
    }
    result["status"] = "VERIFIED_COMPLETE" if all_match else "ARTIFACTS_PRESENT_BUT_UNTRUSTED"
    if not all_match:
        result["reasons"].append("At least one report/prediction/metric consistency check failed.")
    return result


def choose_recommended_policy(policy_rollup: dict[str, Any]) -> tuple[str, str]:
    candidate_b = policy_rollup["uignore_blankzero"]
    candidate_c = policy_rollup["uignore_blankignore"]

    if (
        candidate_b["all_splits_have_all_labels"]
        and candidate_b["max_rows_no_valid_pct"] <= 0.10
        and candidate_b["min_valid_label_fraction"] >= 0.80
    ):
        reason = (
            "Policy B is the conservative default here: it stops forcing uncertain `-1` targets to negative, "
            "preserves all five labels on both splits, and keeps the evaluation set substantially intact. "
            "Blank-zero remains an assumption and must be documented explicitly."
        )
        return "uignore_blankzero", reason

    if (
        candidate_c["all_splits_have_all_labels"]
        and candidate_c["max_rows_no_valid_pct"] <= 0.10
        and candidate_c["min_valid_label_fraction"] >= 0.80
    ):
        reason = (
            "Policy C is conservative on both uncertainty and blanks while still preserving enough valid supervision "
            "to keep the evaluation usable."
        )
        return "uignore_blankignore", reason

    reason = (
        "Policies that ignore blanks remove too much evaluable signal here, so the least risky fallback is Policy A. "
        "If kept, the blank-zero and uncertain-zero assumptions must be documented explicitly."
    )
    return "current_uzero_blankzero", reason


def build_policy_rollup(split_policy_summary: dict[str, Any]) -> dict[str, Any]:
    rollup: dict[str, Any] = {}
    for policy_name in POLICIES:
        split_summaries = [split_policy_summary[split][policy_name] for split in SPLITS]
        rollup[policy_name] = {
            "all_splits_have_all_labels": all(summary["valid_label_count"] == len(LABELS) for summary in split_summaries),
            "max_rows_no_valid_pct": max(summary["rows_with_no_valid_labels_pct"] for summary in split_summaries),
            "min_valid_label_fraction": min(summary["valid_label_fraction"] for summary in split_summaries),
            "split_summaries": split_summaries,
        }
    return rollup


def summarize_split_policy(manifest_bundle: dict[str, Any], policy_name: str) -> dict[str, Any]:
    raw_categories = manifest_bundle["raw_categories"]
    split = manifest_bundle["split"]
    rows_with_no_valid_labels = np.zeros(len(raw_categories), dtype=bool)
    row_has_positive = np.zeros(len(raw_categories), dtype=bool)
    valid_label_count = 0
    total_valid_label_instances = 0
    per_label: dict[str, Any] = {}
    for label in LABELS:
        y_true, y_mask = apply_policy_to_label(raw_categories[label], policy_name)
        valid_truth = y_true[y_mask]
        n_valid = int(y_mask.sum())
        n_positive = int(valid_truth.sum())
        n_negative = int(n_valid - n_positive)
        label_valid = n_positive > 0 and n_negative > 0
        if label_valid:
            valid_label_count += 1
        total_valid_label_instances += n_valid
        rows_with_no_valid_labels |= ~y_mask
        row_has_positive |= y_mask & (y_true == 1)
        per_label[label] = {
            "n_valid": n_valid,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "is_valid_label": label_valid,
            "n_uncertain_raw": manifest_bundle["raw_summary"][label]["uncertain_raw"],
            "n_blank_raw": manifest_bundle["raw_summary"][label]["blank_raw"],
        }
    any_valid_row = np.zeros(len(raw_categories), dtype=bool)
    for label in LABELS:
        _, y_mask = apply_policy_to_label(raw_categories[label], policy_name)
        any_valid_row |= y_mask
    rows_no_valid = int((~any_valid_row).sum())
    total_possible = int(len(raw_categories) * len(LABELS))
    return {
        "split": split,
        "policy_name": policy_name,
        "policy_short_name": POLICIES[policy_name]["short_name"],
        "rows_with_no_valid_labels": rows_no_valid,
        "rows_with_no_valid_labels_pct": float(rows_no_valid / max(len(raw_categories), 1)),
        "rows_with_at_least_one_positive": int(row_has_positive.sum()),
        "rows_with_at_least_one_positive_pct": float(row_has_positive.sum() / max(len(raw_categories), 1)),
        "valid_label_count": valid_label_count,
        "total_valid_label_instances": total_valid_label_instances,
        "total_possible_label_instances": total_possible,
        "valid_label_fraction": float(total_valid_label_instances / max(total_possible, 1)),
        "per_label": per_label,
    }


def run_sensitivity_audit(root: Path) -> dict[str, Any]:
    manifest_bundles = {split: prepare_manifest_bundle(root, split) for split in SPLITS}
    split_policy_summary = {
        split: {policy_name: summarize_split_policy(bundle, policy_name) for policy_name in POLICIES}
        for split, bundle in manifest_bundles.items()
    }
    policy_rollup = build_policy_rollup(split_policy_summary)

    run_results: dict[str, Any] = {}
    available_runs: list[dict[str, Any]] = []
    for run_spec in RUN_SPECS:
        split_results: dict[str, Any] = {}
        split_artifacts: dict[str, Any] = {}
        missing_prediction_files = []
        for split in SPLITS:
            prediction_path = root / run_spec["prediction_files"][split]
            split_artifacts[split] = {
                "prediction_path": str(prediction_path.resolve()),
                "prediction_exists": prediction_path.exists(),
            }
            if not prediction_path.exists():
                missing_prediction_files.append(str(prediction_path.resolve()))
                continue

            predictions = read_csv_checked(prediction_path)
            match = match_predictions_to_manifest(predictions, manifest_bundles[split]["manifest"], split)
            verify_prediction_truth_columns(
                predictions=predictions,
                manifest=manifest_bundles[split]["manifest"],
                match=match,
                run_id=run_spec["run_id"],
                split=split,
            )

            aligned_predictions = predictions.loc[match.prediction_indices].reset_index(drop=True)
            aligned_manifest = manifest_bundles[split]["manifest"].loc[match.manifest_indices].reset_index(drop=True)
            aligned_raw_categories = (
                manifest_bundles[split]["raw_categories"].loc[match.manifest_indices].reset_index(drop=True)
            )

            policy_metrics = {
                policy_name: compute_policy_metrics(
                    aligned_manifest=aligned_manifest,
                    aligned_predictions=aligned_predictions,
                    raw_categories=aligned_raw_categories,
                    policy_name=policy_name,
                    match=match,
                    run_id=run_spec["run_id"],
                    split=split,
                )
                for policy_name in POLICIES
            }
            split_results[split] = policy_metrics
            split_artifacts[split]["match_method"] = match.method
            split_artifacts[split]["match_warnings"] = list(match.warnings)

        run_results[run_spec["run_id"]] = {
            "run_id": run_spec["run_id"],
            "display_name": run_spec["display_name"],
            "report_json_path": str((root / run_spec["report_json"]).resolve()),
            "headline_eligible": bool(run_spec["headline_eligible"]),
            "artifacts": split_artifacts,
            "missing_prediction_files": missing_prediction_files,
            **split_results,
        }
        if split_results:
            available_runs.append(run_spec)

    full_ft_k20_verification = verify_full_ft_k20(root, run_results)
    if full_ft_k20_verification["status"] == "VERIFIED_COMPLETE":
        run_results["full_ft_k20_seed2027"]["headline_eligible"] = True

    recommended_policy, recommendation_reason = choose_recommended_policy(policy_rollup)
    included_runs = [
        run_results[run_spec["run_id"]]
        for run_spec in RUN_SPECS
        if run_spec["run_id"] in run_results and run_results[run_spec["run_id"]].get("headline_eligible", False)
    ]

    best_method_by_policy: dict[str, Any] = {}
    best_method_names: list[str] = []
    for policy_name in POLICIES:
        test_candidates = []
        for run in included_runs:
            if "test" not in run:
                continue
            metrics = run["test"][policy_name]
            if metrics["macro_auprc"] is None:
                continue
            test_candidates.append(
                {
                    "run_id": run["run_id"],
                    "display_name": run["display_name"],
                    "macro_auprc": metrics["macro_auprc"],
                    "macro_auroc": metrics["macro_auroc"],
                }
            )
        if not test_candidates:
            best_method_by_policy[policy_name] = None
            continue
        test_candidates.sort(
            key=lambda item: (
                item["macro_auprc"],
                item["macro_auroc"] if item["macro_auroc"] is not None else float("-inf"),
                item["display_name"],
            ),
            reverse=True,
        )
        best = test_candidates[0]
        best_method_by_policy[policy_name] = best
        best_method_names.append(best["display_name"])

    unique_best_methods = sorted({name for name in best_method_names if name is not None})
    ranking_changes = len(unique_best_methods) > 1

    runs_that_need_repeating: list[str] = []
    if recommended_policy != "current_uzero_blankzero":
        runs_that_need_repeating.append(
            "no adaptation: reevaluate existing NIH->MIMIC predictions under the new policy before citing target-side metrics"
        )
        runs_that_need_repeating.append(
            "head_only_k5: rerun training and evaluation because support labels and model selection both used the current policy"
        )
        runs_that_need_repeating.append(
            "head_only_k20: rerun training and evaluation because support labels and model selection both used the current policy"
        )
        runs_that_need_repeating.append(
            "full_ft_k5: rerun training and evaluation because support labels and model selection both used the current policy"
        )
        if full_ft_k20_verification["status"] == "VERIFIED_COMPLETE":
            runs_that_need_repeating.append(
                "full_ft_k20: rerun training and evaluation because support labels and model selection both used the current policy"
            )
        else:
            runs_that_need_repeating.append(
                "full_ft_k20: do not cite the old run; if this setting matters, rerun it cleanly under the new policy"
            )

    headline = {
        "recommended_policy": recommended_policy,
        "recommended_policy_short_name": POLICIES[recommended_policy]["short_name"],
        "recommendation_reason": recommendation_reason,
        "best_method_by_policy": best_method_by_policy,
        "ranking_changes": ranking_changes,
        "ranking_metric": "test macro AUPRC",
        "runs_that_need_repeating": runs_that_need_repeating,
        "full_ft_k20_verification_status": full_ft_k20_verification["status"],
    }

    return {
        "status": "DONE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "labels": LABELS,
        "policies": POLICIES,
        "manifests": {
            split: {
                "common5_path": str(bundle["common5_path"].resolve()),
                "target_raw_path": str(bundle["target_raw_path"].resolve()) if bundle["target_raw_path"] else None,
                "n_rows": int(len(bundle["manifest"])),
                "raw_summary": bundle["raw_summary"],
                "all_blank_count": bundle["all_blank_count"],
                "all_blank_pct": bundle["all_blank_pct"],
                "converted_positive_rows": bundle["converted_positive_rows"],
                "cross_check": bundle["cross_check"],
            }
            for split, bundle in manifest_bundles.items()
        },
        "split_policy_summary": split_policy_summary,
        "runs": run_results,
        "headline": headline,
        "full_ft_k20_verification": full_ft_k20_verification,
    }


def build_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifests = report["manifests"]
    for run_id, run_data in report["runs"].items():
        for split in SPLITS:
            if split not in run_data:
                continue
            split_manifest = manifests[split]
            for policy_name, metrics in run_data[split].items():
                base_row = {
                    "row_type": "overall",
                    "run_id": run_id,
                    "run_display_name": run_data["display_name"],
                    "split": split,
                    "policy_name": policy_name,
                    "policy_short_name": POLICIES[policy_name]["short_name"],
                    "match_method": metrics["match_method"],
                    "match_warnings": " | ".join(metrics["match_warnings"]),
                    "n_rows": metrics["n_rows"],
                    "all_five_blank_rows": split_manifest["all_blank_count"],
                    "all_five_blank_pct": split_manifest["all_blank_pct"],
                    "rows_with_at_least_one_positive": metrics["rows_with_at_least_one_positive"],
                    "rows_with_at_least_one_positive_pct": metrics["rows_with_at_least_one_positive_pct"],
                    "rows_with_no_valid_labels": metrics["rows_with_no_valid_labels"],
                    "rows_with_no_valid_labels_pct": metrics["rows_with_no_valid_labels_pct"],
                    "total_valid_label_instances": metrics["total_valid_label_instances"],
                    "total_possible_label_instances": metrics["total_possible_label_instances"],
                    "valid_label_fraction": metrics["valid_label_fraction"],
                    "valid_label_count": len(metrics["valid_labels"]),
                    "macro_auroc": metrics["macro_auroc"],
                    "macro_auprc": metrics["macro_auprc"],
                    "micro_auroc": metrics["micro_auroc"],
                    "micro_auprc": metrics["micro_auprc"],
                    "macro_auroc_defined": metrics["macro_auroc_defined"],
                    "macro_auprc_defined": metrics["macro_auprc_defined"],
                    "micro_auroc_defined": metrics["micro_auroc_defined"],
                    "micro_auprc_defined": metrics["micro_auprc_defined"],
                    "macro_auroc_comparable": metrics["macro_auroc_comparable"],
                    "macro_auprc_comparable": metrics["macro_auprc_comparable"],
                    "micro_auroc_comparable": metrics["micro_auroc_comparable"],
                    "micro_auprc_comparable": metrics["micro_auprc_comparable"],
                    "comparability_note": metrics["comparability_note"],
                }
                rows.append(base_row)
                for label in LABELS:
                    label_metrics = metrics["per_label"][label]
                    rows.append(
                        {
                            **base_row,
                            "row_type": "label",
                            "label": label,
                            "n_valid": label_metrics["n_valid"],
                            "n_positive": label_metrics["n_positive"],
                            "n_negative": label_metrics["n_negative"],
                            "n_uncertain_raw": label_metrics["n_uncertain_raw"],
                            "n_blank_raw": label_metrics["n_blank_raw"],
                            "label_valid": label_metrics["is_valid_label"],
                            "auroc": label_metrics["auroc"],
                            "auprc": label_metrics["auprc"],
                        }
                    )
    return rows


def format_float(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_raw_missingness_table(report: dict[str, Any]) -> str:
    lines = [
        "| Split | Label | Raw +1 | Raw 0 | Raw -1 | Raw blank |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        raw_summary = report["manifests"][split]["raw_summary"]
        for label in LABELS:
            item = raw_summary[label]
            lines.append(
                f"| {split} | {label} | {item['positive_raw']} | {item['negative_raw']} | "
                f"{item['uncertain_raw']} | {item['blank_raw']} |"
            )
    return "\n".join(lines)


def build_all_blank_table(report: dict[str, Any]) -> str:
    lines = [
        "| Split | Rows | All five blank | Percent | Rows with >=1 converted positive under current manifest |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        manifest = report["manifests"][split]
        lines.append(
            f"| {split} | {manifest['n_rows']} | {manifest['all_blank_count']} | "
            f"{format_float(manifest['all_blank_pct'] * 100.0, 2)}% | {manifest['converted_positive_rows']} |"
        )
    return "\n".join(lines)


def build_policy_definition_table() -> str:
    lines = [
        "| Policy | Name | Mapping |",
        "| --- | --- | --- |",
    ]
    for policy_name, policy in POLICIES.items():
        lines.append(f"| {policy['table_name']} | `{policy_name}` | {policy['description']} |")
    return "\n".join(lines)


def build_metric_table_for_split(report: dict[str, Any], split: str) -> str:
    lines = [
        "| Run | A macro AUROC | A macro AUPRC | B macro AUROC | B macro AUPRC | C macro AUROC | C macro AUPRC | D macro AUROC | D macro AUPRC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_spec in RUN_SPECS:
        run_data = report["runs"].get(run_spec["run_id"])
        if run_data is None or split not in run_data:
            continue
        if run_spec["run_id"] == "full_ft_k20_seed2027" and report["full_ft_k20_verification"]["status"] != "VERIFIED_COMPLETE":
            continue
        row = [run_data["display_name"]]
        for policy_name in POLICIES:
            metrics = run_data[split][policy_name]
            row.append(format_float(metrics["macro_auroc"]))
            row.append(format_float(metrics["macro_auprc"]))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_ranking_table(report: dict[str, Any]) -> str:
    lines = [
        "| Policy | Best method on test macro AUPRC | Test macro AUPRC | Test macro AUROC |",
        "| --- | --- | ---: | ---: |",
    ]
    for policy_name, best in report["headline"]["best_method_by_policy"].items():
        policy_short = POLICIES[policy_name]["short_name"]
        if best is None:
            lines.append(f"| {policy_short} | NA | NA | NA |")
        else:
            lines.append(
                f"| {policy_short} | {best['display_name']} | {format_float(best['macro_auprc'])} | "
                f"{format_float(best['macro_auroc'])} |"
            )
    return "\n".join(lines)


def build_policy_coverage_table(report: dict[str, Any]) -> str:
    lines = [
        "| Split | Policy | Valid labels | Valid label fraction | Rows with no valid labels | Macro comparable | Micro comparable |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for split in SPLITS:
        for policy_name, summary in report["split_policy_summary"][split].items():
            lines.append(
                f"| {split} | {POLICIES[policy_name]['short_name']} | {summary['valid_label_count']} / {len(LABELS)} | "
                f"{format_float(summary['valid_label_fraction'] * 100.0, 2)}% | "
                f"{format_float(summary['rows_with_no_valid_labels_pct'] * 100.0, 2)}% | "
                f"{'yes' if summary['valid_label_count'] == len(LABELS) else 'no'} | "
                f"{'yes' if summary['valid_label_fraction'] == 1.0 else 'no'} |"
            )
    return "\n".join(lines)


def build_repeating_runs_section(report: dict[str, Any]) -> str:
    runs = report["headline"]["runs_that_need_repeating"]
    if not runs:
        return "No old target-side runs need repeating if Policy A is retained."
    return "\n".join(f"- {item}" for item in runs)


def build_verification_markdown(verification: dict[str, Any]) -> str:
    lines = [
        "# full_ft_k20 Verification",
        "",
        f"- status: `{verification['status']}`",
        f"- json report exists: {'yes' if verification['json_report_exists'] else 'no'}",
        f"- markdown report exists: {'yes' if verification['markdown_report_exists'] else 'no'}",
        f"- recomputed metrics available: {'yes' if verification['recomputed_metrics_available'] else 'no'}",
        "",
        "## Artifact Presence",
        "",
        "| Artifact | Present | Path |",
        "| --- | --- | --- |",
        f"| JSON report | {'yes' if verification['json_report_exists'] else 'no'} | `{verification['json_report_path']}` |",
        f"| Markdown report | {'yes' if verification['markdown_report_exists'] else 'no'} | `{verification['markdown_report_path']}` |",
    ]
    for split in SPLITS:
        lines.append(
            f"| {split} predictions | {'yes' if verification['prediction_file_exists'][split] else 'no'} | "
            f"`{verification['prediction_files'][split]}` |"
        )

    if verification["checks"]:
        lines.extend(
            [
                "",
                "## Consistency Checks",
                "",
                f"- markdown checks: {verification['checks']['markdown']}",
                f"- json checks: {verification['checks']['json']}",
            ]
        )
        for split in SPLITS:
            split_checks = verification["checks"]["splits"].get(split, {})
            lines.append(f"- {split} metric checks: {split_checks}")

    if verification["reasons"]:
        lines.extend(["", "## Reasons", ""])
        lines.extend(f"- {reason}" for reason in verification["reasons"])
    return "\n".join(lines) + "\n"


def build_sensitivity_markdown(report: dict[str, Any], root: Path) -> str:
    recommended_policy = report["headline"]["recommended_policy"]
    recommended_short = report["headline"]["recommended_policy_short_name"]
    full_ft_k20_status = report["full_ft_k20_verification"]["status"]
    ranking_changes = report["headline"]["ranking_changes"]

    section7_notes = [report["headline"]["recommendation_reason"]]
    if recommended_policy in {"current_uzero_blankzero", "uignore_blankzero", "uone_blankzero"}:
        section7_notes.append(
            "This recommendation still treats blank/NaN as zero, so that assumption must be stated explicitly in the paper."
        )
    if report["split_policy_summary"]["val"]["uignore_blankignore"]["rows_with_no_valid_labels"] > 0 or report["split_policy_summary"]["test"]["uignore_blankignore"]["rows_with_no_valid_labels"] > 0:
        section7_notes.append(
            "Policy C drops blank labels entirely. In this dataset that creates rows with no evaluable common5 labels, so blank-ignore is not a safe default unless that loss of supervision is intentional."
        )

    lines = [
        "# MIMIC Label Policy Sensitivity Audit",
        "",
        "## 1. Why This Audit Was Needed",
        "Current MIMIC common5 target manifests were built with the U-zero plus blank-zero policy. That means existing target-side metrics implicitly treat both uncertain `-1` labels and blank/NaN labels as negatives. Because the completed MIMIC results were evaluated on those converted manifests, reported target-side performance can change when the label policy changes even if the predictions stay fixed.",
        "",
        "## 2. Raw Label Missingness Summary",
        build_raw_missingness_table(report),
        "",
        "Cross-check note:",
        f"- val raw labels were cross-checked against `{maybe_rel(Path(report['manifests']['val']['target_raw_path']), root) if report['manifests']['val']['target_raw_path'] else 'none'}` because `mimic_target_val.csv` is not present here.",
        f"- test raw labels were cross-checked against `{maybe_rel(Path(report['manifests']['test']['target_raw_path']), root) if report['manifests']['test']['target_raw_path'] else 'none'}`.",
        "",
        "## 3. All-Blank Row Summary",
        build_all_blank_table(report),
        "",
        "## 4. Policy Definitions",
        build_policy_definition_table(),
        "",
        "## 5. Metric Sensitivity Results",
        "Validation split:",
        build_metric_table_for_split(report, "val"),
        "",
        "Test split:",
        build_metric_table_for_split(report, "test"),
        "",
        "Coverage and comparability:",
        build_policy_coverage_table(report),
        "",
        "## 6. Ranking Stability",
        build_ranking_table(report),
        "",
        (
            "The best method does change across policies."
            if ranking_changes
            else "The best method is stable across policies under the primary ranking metric."
        ),
        f"Ranking was evaluated on `{report['headline']['ranking_metric']}`.",
        "",
        "## 7. Recommended Primary Policy",
        f"Recommend `{recommended_policy}` ({recommended_short}).",
        "",
    ]
    lines.extend(f"- {note}" for note in section7_notes)
    lines.extend(
        [
            "",
            "## 8. Runs That Need Repeating",
            build_repeating_runs_section(report),
            "",
            "## 9. Final Decision Needed",
            f"Before new training, accept policy {recommended_short} (`{recommended_policy}`) or revise it.",
            "",
            "## Appendices",
            f"- `full_ft_k20_seed2027` verification status: `{full_ft_k20_status}`",
            f"- generated sensitivity JSON: `{maybe_rel(root / OUTPUT_PATHS['sensitivity_json'], root)}`",
            f"- generated sensitivity CSV: `{maybe_rel(root / OUTPUT_PATHS['sensitivity_csv'], root)}`",
            f"- generated full_ft_k20 verification JSON: `{maybe_rel(root / OUTPUT_PATHS['verification_json'], root)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def print_final_summary(report: dict[str, Any], root: Path) -> None:
    best_methods = []
    for policy_name, best in report["headline"]["best_method_by_policy"].items():
        if best is None:
            best_methods.append(f"{POLICIES[policy_name]['short_name']}=NA")
        else:
            best_methods.append(f"{POLICIES[policy_name]['short_name']}={best['display_name']}")
    print(f"recommended policy: {report['headline']['recommended_policy']}")
    print(f"full_ft_k20 verified: {report['full_ft_k20_verification']['status']}")
    print(f"best method under each policy: {', '.join(best_methods)}")
    print(f"method ranking changes: {'yes' if report['headline']['ranking_changes'] else 'no'}")
    print("generated reports:")
    for key in OUTPUT_PATHS:
        print(f"- {maybe_rel(root / OUTPUT_PATHS[key], root)}")


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output_paths = [root / relative_path for relative_path in OUTPUT_PATHS.values()]
    ensure_output_paths_do_not_exist(output_paths)

    report = run_sensitivity_audit(root)
    csv_rows = build_csv_rows(report)
    sensitivity_md = build_sensitivity_markdown(report, root)
    verification_md = build_verification_markdown(report["full_ft_k20_verification"])

    write_json(report, root / OUTPUT_PATHS["sensitivity_json"])
    write_csv(csv_rows, root / OUTPUT_PATHS["sensitivity_csv"])
    write_text(sensitivity_md, root / OUTPUT_PATHS["sensitivity_md"])
    write_json(report["full_ft_k20_verification"], root / OUTPUT_PATHS["verification_json"])
    write_text(verification_md, root / OUTPUT_PATHS["verification_md"])

    print_final_summary(report, root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StageFailure as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception:  # pragma: no cover
        traceback.print_exc()
        raise SystemExit(1)
