#!/usr/bin/env python3
"""Build official MIMIC common5 manifests under Policy B."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


POLICY_NAME = "uignore_blankzero"
LABEL_SET = "common5"
FINAL_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
RAW_COLUMN_CANDIDATES = {
    "Atelectasis": ["Atelectasis", "raw_Atelectasis"],
    "Cardiomegaly": ["Cardiomegaly", "raw_Cardiomegaly"],
    "Consolidation": ["Consolidation", "raw_Consolidation"],
    "Edema": ["Edema", "raw_Edema"],
    "Effusion": ["Pleural Effusion", "Effusion", "raw_Pleural_Effusion", "raw_Effusion"],
}
SPLIT_SPECS = [
    {
        "output_split": "train_pool",
        "output_filename": "mimic_common5_policyB_train_pool.csv",
        "source_candidates": [
            "manifests/mimic_target_train_pool.csv",
            "manifests/mimic_target_train_pool_common7_raw.csv",
            "manifests/mimic_common5_train_pool.csv",
        ],
    },
    {
        "output_split": "val",
        "output_filename": "mimic_common5_policyB_val.csv",
        "source_candidates": [
            "manifests/mimic_target_query.csv",
            "manifests/mimic_target_val.csv",
            "manifests/mimic_target_query_common7_raw.csv",
            "manifests/mimic_common5_val.csv",
        ],
    },
    {
        "output_split": "test",
        "output_filename": "mimic_common5_policyB_test.csv",
        "source_candidates": [
            "manifests/mimic_target_test.csv",
            "manifests/mimic_target_test_common7_raw.csv",
            "manifests/mimic_common5_test.csv",
        ],
    },
]
PRESERVE_COLUMNS = [
    "dicom_id",
    "subject_id",
    "study_id",
    "abs_path",
    "rel_path",
    "path",
    "image_path",
    "filepath",
    "file_path",
    "PerformedProcedureStepDescription",
    "ViewPosition",
    "Rows",
    "Columns",
    "StudyDate",
    "StudyTime",
    "ProcedureCodeSequence_CodeMeaning",
    "ViewCodeSequence_CodeMeaning",
    "PatientOrientationCodeSequence_CodeMeaning",
    "split_source",
    "image_id",
    "label_source",
    "image_format",
    "path_exists",
]


class StageFailure(RuntimeError):
    """Raised when manifest generation cannot continue safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create official MIMIC common5 Policy B manifests.")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting Policy B manifest outputs.")
    return parser.parse_args()


def print_line(message: str) -> None:
    print(message, flush=True)


def read_csv_checked(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise StageFailure(f"Could not read CSV {path}: {exc}") from exc


def choose_column(columns: list[str], candidates: list[str], description: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise StageFailure(f"Missing {description}. Expected one of: {candidates}")


def contains_raw_columns(dataframe: pd.DataFrame) -> bool:
    columns = list(dataframe.columns)
    try:
        for label in FINAL_LABELS:
            choose_column(columns, RAW_COLUMN_CANDIDATES[label], f"raw label column for {label}")
    except StageFailure:
        return False
    return True


def find_source_manifest(root: Path, spec: dict[str, Any]) -> tuple[Path, pd.DataFrame]:
    checked_paths: list[str] = []
    for relative_path in spec["source_candidates"]:
        candidate = root / relative_path
        checked_paths.append(str(candidate.resolve()))
        if not candidate.exists():
            continue
        dataframe = read_csv_checked(candidate)
        if contains_raw_columns(dataframe):
            return candidate, dataframe
    raise StageFailure(
        f"Could not find a raw or least-converted source manifest for split={spec['output_split']}. "
        f"Checked: {checked_paths}"
    )


def normalize_raw_value(value: Any) -> int | pd.NA:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return pd.NA
        try:
            value = float(text)
        except ValueError as exc:
            raise StageFailure(f"Unexpected raw label string value: {value!r}") from exc
    if isinstance(value, (np.integer, int)):
        numeric_value = int(value)
    elif isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return pd.NA
        if not float(value).is_integer():
            raise StageFailure(f"Unexpected non-integer raw label value: {value!r}")
        numeric_value = int(value)
    else:
        raise StageFailure(f"Unexpected raw label value type: {type(value).__name__} ({value!r})")
    if numeric_value not in {1, 0, -1}:
        raise StageFailure(f"Unexpected raw label value: {numeric_value!r}")
    return numeric_value


def normalize_raw_series(series: pd.Series) -> pd.Series:
    normalized = series.map(normalize_raw_value)
    return normalized.astype("Int64")


def convert_policy_b(raw_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    positive = raw_series.eq(1)
    negative = raw_series.eq(0)
    uncertain = raw_series.eq(-1)
    blank = raw_series.isna()

    label_values = pd.Series(np.zeros(len(raw_series), dtype=np.int64), index=raw_series.index)
    mask_values = pd.Series(np.zeros(len(raw_series), dtype=np.int64), index=raw_series.index)

    label_values.loc[positive] = 1
    mask_values.loc[positive | negative | blank] = 1
    label_values.loc[negative | uncertain | blank] = 0
    mask_values.loc[uncertain] = 0
    return label_values.astype(int), mask_values.astype(int)


def build_policy_b_manifest(source_path: Path, source_df: pd.DataFrame, output_split: str) -> pd.DataFrame:
    columns = list(source_df.columns)
    raw_columns = {
        label: choose_column(columns, RAW_COLUMN_CANDIDATES[label], f"raw label column for {label}")
        for label in FINAL_LABELS
    }
    result = pd.DataFrame(index=source_df.index)

    for column in PRESERVE_COLUMNS:
        if column in source_df.columns:
            result[column] = source_df[column]

    if "split" in source_df.columns:
        result["source_split"] = source_df["split"]

    for label in FINAL_LABELS:
        raw_series = normalize_raw_series(source_df[raw_columns[label]])
        label_values, mask_values = convert_policy_b(raw_series)
        result[f"{label}_raw"] = raw_series
        result[label] = label_values
        result[f"{label}_mask"] = mask_values

    result["label_policy"] = POLICY_NAME
    result["label_set"] = LABEL_SET
    result["split"] = output_split
    result["source_manifest"] = source_path.name

    ordered_columns: list[str] = []
    for column in [
        "dicom_id",
        "subject_id",
        "study_id",
        "abs_path",
        "rel_path",
        "path",
        "image_path",
        "filepath",
        "file_path",
        "source_split",
        "split_source",
        "split",
        "label_policy",
        "label_set",
        "source_manifest",
        "PerformedProcedureStepDescription",
        "ViewPosition",
        "Rows",
        "Columns",
        "StudyDate",
        "StudyTime",
        "ProcedureCodeSequence_CodeMeaning",
        "ViewCodeSequence_CodeMeaning",
        "PatientOrientationCodeSequence_CodeMeaning",
        "image_id",
        "label_source",
        "image_format",
        "path_exists",
    ]:
        if column in result.columns and column not in ordered_columns:
            ordered_columns.append(column)
    for label in FINAL_LABELS:
        ordered_columns.append(f"{label}_raw")
    for label in FINAL_LABELS:
        ordered_columns.append(label)
    for label in FINAL_LABELS:
        ordered_columns.append(f"{label}_mask")
    for column in result.columns:
        if column not in ordered_columns:
            ordered_columns.append(column)
    result = result.loc[:, ordered_columns]

    if "dicom_id" in result.columns:
        normalized_dicom = result["dicom_id"].astype("string").str.strip()
        if normalized_dicom.isna().any() or (normalized_dicom == "").any():
            raise StageFailure(f"{source_path} contains blank dicom_id values.")
        if normalized_dicom.duplicated().any():
            raise StageFailure(f"{source_path} contains duplicated dicom_id values.")

    return result


def ensure_output_paths(root: Path, overwrite: bool) -> dict[str, Path]:
    outputs = {
        spec["output_split"]: root / "manifests" / spec["output_filename"]
        for spec in SPLIT_SPECS
    }
    if overwrite:
        return outputs
    existing = [str(path.resolve()) for path in outputs.values() if path.exists()]
    if existing:
        raise StageFailure(
            "Refusing to overwrite existing Policy B manifests. Pass --overwrite to replace: "
            f"{existing}"
        )
    return outputs


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    outputs = ensure_output_paths(root, overwrite=args.overwrite)

    created: dict[str, Path] = {}
    for spec in SPLIT_SPECS:
        source_path, source_df = find_source_manifest(root, spec)
        manifest_df = build_policy_b_manifest(source_path, source_df, spec["output_split"])
        output_path = outputs[spec["output_split"]]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(output_path, index=False)
        created[spec["output_split"]] = output_path
        print_line(
            f"created split={spec['output_split']} rows={len(manifest_df)} "
            f"source={source_path.name} output={output_path}"
        )

    print_line("Policy B manifest files created:")
    for split in ["train_pool", "val", "test"]:
        print_line(f"- {split}: {created[split]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
