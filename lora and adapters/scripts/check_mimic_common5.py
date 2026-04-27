#!/usr/bin/env python3
"""Verify a local MIMIC-CXR-JPG subset for NIH-to-MIMIC transfer."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FINAL_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
MIMIC_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
MIMIC_TO_FINAL = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Pleural Effusion": "Effusion",
}
EXPECTED_SPLITS = {"train", "validate", "test"}
FRONTAL_VIEWS = {"AP", "PA"}
PATH_COLUMN_CANDIDATES = ["path", "image_path", "filepath", "file_path", "rel_path", "abs_path"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class StageFailure(RuntimeError):
    """Raised when the verification cannot safely continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a local MIMIC subset for the common 5 labels.")
    parser.add_argument("--mimic_root", type=str, default=None)
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--labels_csv", type=str, default=None)
    parser.add_argument("--split_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="reports")
    parser.add_argument("--manifest_dir", type=str, default="manifests")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def clean_optional_path(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    return value or None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def json_block(data: Any) -> str:
    return "```json\n" + json.dumps(json_ready(data), indent=2, sort_keys=True) + "\n```"


def print_line(message: str) -> None:
    print(message, flush=True)


def choose_mimic_root(raw_root: str | None, warnings: list[str]) -> Path:
    explicit_root = clean_optional_path(raw_root)
    if explicit_root is not None:
        path = Path(explicit_root).expanduser().resolve()
        if not path.exists():
            raise StageFailure(f"Missing MIMIC root directory: {path}")
        if not path.is_dir():
            raise StageFailure(f"MIMIC root is not a directory: {path}")
        return path

    env_root = clean_optional_path(os.getenv("MIMIC_ROOT"))
    if env_root is not None:
        path = Path(env_root).expanduser().resolve()
        if not path.exists():
            raise StageFailure(f"MIMIC_ROOT points to a missing directory: {path}")
        if not path.is_dir():
            raise StageFailure(f"MIMIC_ROOT is not a directory: {path}")
        return path

    candidate_roots: list[Path] = []
    for candidate in [Path.cwd() / "data" / "mimic_cxr", Path("/workspace/data/mimic_cxr")]:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_dir() and resolved not in candidate_roots:
            candidate_roots.append(resolved)

    if len(candidate_roots) == 1:
        warnings.append(f"Inferred mimic_root from verified local path: {candidate_roots[0]}")
        return candidate_roots[0]

    if len(candidate_roots) > 1:
        raise StageFailure(
            "Multiple possible MIMIC roots were found. Please pass --mimic_root explicitly."
        )

    raise StageFailure("Could not infer MIMIC root. Pass --mimic_root or set MIMIC_ROOT.")


def find_required_file(
    explicit_path: str | None,
    mimic_root: Path,
    expected_names: list[str],
    description: str,
) -> Path:
    raw_explicit = clean_optional_path(explicit_path)
    if raw_explicit is not None:
        path = Path(raw_explicit).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        if not path.exists():
            raise StageFailure(f"Missing {description}: {path}")
        if not path.is_file():
            raise StageFailure(f"{description} is not a file: {path}")
        return path

    for filename in expected_names:
        matches = sorted(path.resolve() for path in mimic_root.rglob(filename) if path.is_file())
        if matches:
            return matches[0]

    expected_text = ", ".join(expected_names)
    raise StageFailure(f"Could not find {description} under {mimic_root}. Expected one of: {expected_text}")


def read_csv_checked(path: Path, description: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, compression="infer", low_memory=False)
    except Exception as exc:  # pragma: no cover - exercised only on bad local files
        raise StageFailure(f"Could not read {description} at {path}: {exc}") from exc


def require_columns(df: pd.DataFrame, required: list[str], description: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise StageFailure(f"Missing required columns in {description}: {missing}")


def normalize_dicom_ids(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_int_ids(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def raw_label_column_name(label_name: str) -> str:
    return "raw_" + label_name.replace(" ", "_")


def uzero_value(value: Any) -> int:
    if pd.isna(value):
        return 0
    numeric_value = float(value)
    if numeric_value == 1.0:
        return 1
    if numeric_value in {0.0, -1.0}:
        return 0
    raise ValueError(f"Unsupported label value for U-zero: {value}")


def apply_uzero(series: pd.Series) -> pd.Series:
    return series.map(uzero_value).astype(int)


def validate_and_summarize_label_values(labels_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    summaries: dict[str, dict[str, int]] = {}
    for label_name in MIMIC_LABELS:
        numeric_values = pd.to_numeric(labels_df[label_name], errors="coerce")
        invalid_non_numeric = labels_df[label_name].notna() & numeric_values.isna()
        invalid_numeric = numeric_values.notna() & ~numeric_values.isin([1.0, 0.0, -1.0])

        if invalid_non_numeric.any() or invalid_numeric.any():
            bad_values = sorted(pd.unique(labels_df.loc[invalid_non_numeric | invalid_numeric, label_name]))
            raise StageFailure(f"Invalid values found in label column '{label_name}': {bad_values}")

        labels_df[label_name] = numeric_values
        summaries[label_name] = {
            "1": int((numeric_values == 1.0).sum()),
            "0": int((numeric_values == 0.0).sum()),
            "-1": int((numeric_values == -1.0).sum()),
            "missing": int(numeric_values.isna().sum()),
        }

    return summaries


def find_path_column(columns: list[str]) -> str | None:
    for candidate in PATH_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def build_local_image_index(mimic_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    image_paths = sorted(
        path.resolve()
        for path in mimic_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise StageFailure(f"No .jpg/.jpeg/.png files found under {mimic_root}")

    first_seen: dict[str, Path] = {}
    duplicate_paths: dict[str, list[str]] = {}
    records: list[dict[str, str]] = []

    for image_path in image_paths:
        dicom_id = image_path.stem
        if dicom_id in first_seen:
            duplicate_paths.setdefault(dicom_id, [str(first_seen[dicom_id])]).append(str(image_path))
            continue
        first_seen[dicom_id] = image_path
        try:
            rel_path = image_path.relative_to(mimic_root)
            rel_text = str(rel_path)
        except ValueError:
            rel_text = image_path.name
        records.append(
            {
                "dicom_id": dicom_id,
                "indexed_abs_path": str(image_path),
                "indexed_rel_path": rel_text,
            }
        )

    local_df = pd.DataFrame.from_records(records)
    local_df["dicom_id"] = normalize_dicom_ids(local_df["dicom_id"])

    summary = {
        "local_image_files_found": int(len(image_paths)),
        "unique_local_dicoms": int(len(local_df)),
        "duplicate_local_dicom_ids": int(len(duplicate_paths)),
        "duplicate_local_examples": {key: value[:3] for key, value in sorted(duplicate_paths.items())[:5]},
        "image_extensions": sorted({path.suffix.lower() for path in image_paths}),
    }
    return local_df, summary


def coalesce_id_columns(
    merged_df: pd.DataFrame,
    base_name: str,
    alternate_name: str,
    warnings: list[str],
) -> None:
    if base_name in merged_df.columns:
        merged_df[base_name] = normalize_int_ids(merged_df[base_name])
    if alternate_name in merged_df.columns:
        merged_df[alternate_name] = normalize_int_ids(merged_df[alternate_name])

    if base_name in merged_df.columns and alternate_name in merged_df.columns:
        mismatch_mask = (
            merged_df[base_name].notna()
            & merged_df[alternate_name].notna()
            & (merged_df[base_name] != merged_df[alternate_name])
        )
        mismatch_count = int(mismatch_mask.sum())
        if mismatch_count > 0:
            preview = merged_df.loc[mismatch_mask, ["dicom_id", base_name, alternate_name]].head(5)
            raise StageFailure(
                f"Found {mismatch_count} mismatched {base_name} values between metadata and split. "
                f"Examples: {preview.to_dict(orient='records')}"
            )
        merged_df[base_name] = merged_df[base_name].fillna(merged_df[alternate_name])
        merged_df.drop(columns=[alternate_name], inplace=True)
    elif alternate_name in merged_df.columns and base_name not in merged_df.columns:
        merged_df.rename(columns={alternate_name: base_name}, inplace=True)
        warnings.append(f"Used {alternate_name} as {base_name} because metadata did not provide it.")


def merge_local_subset(
    local_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    split_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata_merge = local_df.merge(
        metadata_df,
        on="dicom_id",
        how="left",
        indicator="_metadata_merge",
    )
    merged = metadata_merge.merge(
        split_df,
        on="dicom_id",
        how="left",
        suffixes=("", "_split"),
        indicator="_split_merge",
    )

    coalesce_id_columns(merged, "subject_id", "subject_id_split", warnings)
    coalesce_id_columns(merged, "study_id", "study_id_split", warnings)

    if "split" not in merged.columns:
        raise StageFailure("Split column is missing after joining metadata and split tables.")

    merged["split"] = merged["split"].astype("string").str.strip()

    merged = merged.merge(
        labels_df[["subject_id", "study_id", *MIMIC_LABELS]],
        on=["subject_id", "study_id"],
        how="left",
        indicator="_labels_merge",
    )

    rows_missing_metadata = int((merged["_metadata_merge"] != "both").sum())
    rows_missing_split = int((merged["_split_merge"] != "both").sum())
    rows_missing_label_row = int((merged["_labels_merge"] != "both").sum())
    rows_all_target_labels_missing = int(merged[MIMIC_LABELS].isna().all(axis=1).sum())

    summary = {
        "joined_image_rows": int(len(merged)),
        "unique_subjects": int(merged["subject_id"].dropna().nunique()),
        "unique_studies": int(merged["study_id"].dropna().nunique()),
        "rows_missing_metadata": rows_missing_metadata,
        "rows_missing_split": rows_missing_split,
        "rows_missing_label_row": rows_missing_label_row,
        "rows_all_target_labels_missing": rows_all_target_labels_missing,
        "metadata_join_rate": round(1.0 - (rows_missing_metadata / max(len(merged), 1)), 6),
        "split_join_rate": round(1.0 - (rows_missing_split / max(len(merged), 1)), 6),
        "label_join_rate": round(1.0 - (rows_missing_label_row / max(len(merged), 1)), 6),
    }
    return merged, summary


def build_common5_labels(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for raw_label in MIMIC_LABELS:
        result[raw_label] = pd.to_numeric(result[raw_label], errors="coerce")
        result[raw_label_column_name(raw_label)] = result[raw_label]

    for raw_label, final_label in MIMIC_TO_FINAL.items():
        result[final_label] = apply_uzero(result[raw_label])

    return result


def normalize_candidate_path(raw_path: Any, mimic_root: Path) -> Path | None:
    if pd.isna(raw_path):
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (mimic_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def relative_to_root(path: Path, mimic_root: Path) -> str:
    try:
        return str(path.relative_to(mimic_root))
    except ValueError:
        return path.name


def standard_mimic_paths(row: pd.Series, mimic_root: Path) -> list[Path]:
    if pd.isna(row.get("subject_id")) or pd.isna(row.get("study_id")) or pd.isna(row.get("dicom_id")):
        return []

    subject_id = str(int(row["subject_id"]))
    study_id = str(int(row["study_id"]))
    dicom_id = str(row["dicom_id"])
    subject_prefix = subject_id[:2]
    split_value = str(row.get("split", "")).strip()
    split_aliases = [split_value]
    if split_value == "validate":
        split_aliases.append("val")

    candidates: list[Path] = []
    for extension in [".jpg", ".jpeg", ".png"]:
        candidates.append(
            mimic_root / "files" / f"p{subject_prefix}" / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}{extension}"
        )
        for split_alias in split_aliases:
            if split_alias:
                candidates.append(
                    mimic_root
                    / "raw"
                    / split_alias
                    / f"p{subject_prefix}"
                    / f"p{subject_id}"
                    / f"s{study_id}"
                    / f"{dicom_id}{extension}"
                )
    return candidates


def resolve_candidate_paths(
    df: pd.DataFrame,
    mimic_root: Path,
    metadata_path_column: str | None,
    split_path_column: str | None,
) -> pd.DataFrame:
    resolved = df.copy()

    metadata_column_in_merged = metadata_path_column if metadata_path_column in resolved.columns else None
    split_column_in_merged = None
    if split_path_column is not None:
        for candidate_name in [split_path_column, f"{split_path_column}_split"]:
            if candidate_name in resolved.columns:
                split_column_in_merged = candidate_name
                break

    source_renames: dict[str, str] = {}
    for column_name in [metadata_column_in_merged, split_column_in_merged]:
        if column_name in {"abs_path", "rel_path", "path_method"} and column_name not in source_renames:
            source_renames[column_name] = f"source_{column_name}"
    if source_renames:
        resolved = resolved.rename(columns=source_renames)
        metadata_column_in_merged = source_renames.get(metadata_column_in_merged, metadata_column_in_merged)
        split_column_in_merged = source_renames.get(split_column_in_merged, split_column_in_merged)

    resolved["abs_path"] = None
    resolved["rel_path"] = None
    resolved["path_method"] = "missing"

    for index, row in resolved.iterrows():
        chosen_path: Path | None = None
        chosen_method = "missing"

        for column_name in [metadata_column_in_merged, split_column_in_merged]:
            if column_name is None:
                continue
            candidate = normalize_candidate_path(row[column_name], mimic_root)
            if candidate is not None and candidate.exists():
                chosen_path = candidate
                chosen_method = f"path_column:{column_name}"
                break

        if chosen_path is None:
            indexed_path = normalize_candidate_path(row.get("indexed_abs_path"), mimic_root)
            if indexed_path is not None and indexed_path.exists():
                chosen_path = indexed_path
                chosen_method = "recursive_search"

        if chosen_path is None:
            for candidate in standard_mimic_paths(row, mimic_root):
                if candidate.exists():
                    chosen_path = candidate.resolve()
                    chosen_method = "standard_pattern"
                    break

        if chosen_path is not None:
            resolved.at[index, "abs_path"] = str(chosen_path)
            resolved.at[index, "rel_path"] = relative_to_root(chosen_path, mimic_root)
            resolved.at[index, "path_method"] = chosen_method

    return resolved


def get_final_manifest_columns(df: pd.DataFrame) -> list[str]:
    columns = ["dicom_id", "subject_id", "study_id", "split", "abs_path", "rel_path", "path_method"]
    for optional_column in ["ViewPosition", "Rows", "Columns"]:
        if optional_column in df.columns:
            columns.append(optional_column)
    columns.extend(raw_label_column_name(label_name) for label_name in MIMIC_LABELS)
    columns.extend(FINAL_LABELS)
    return columns


def prepare_final_manifest(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    manifest = df.copy()
    manifest["split"] = split_name
    manifest_columns = get_final_manifest_columns(manifest)
    return manifest.loc[:, manifest_columns].sort_values(["subject_id", "study_id", "dicom_id"]).reset_index(drop=True)


def make_subjectwise_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if df["subject_id"].isna().any():
        raise StageFailure("Cannot create a fallback subject-wise split because some subject_id values are missing.")

    subjects = sorted(pd.unique(df["subject_id"].astype(int)))
    if len(subjects) < 3:
        raise StageFailure("Need at least 3 unique subjects to create a 60/20/20 fallback split.")

    rng = random.Random(seed)
    rng.shuffle(subjects)

    num_subjects = len(subjects)
    train_end = max(1, int(round(num_subjects * 0.6)))
    val_end = max(train_end + 1, int(round(num_subjects * 0.8)))
    if val_end >= num_subjects:
        val_end = num_subjects - 1

    train_subjects = set(subjects[:train_end])
    val_subjects = set(subjects[train_end:val_end])
    test_subjects = set(subjects[val_end:])

    if not train_subjects or not val_subjects or not test_subjects:
        raise StageFailure("Fallback subject-wise split produced an empty split.")

    result = df.copy()
    result["split"] = "test"
    result.loc[result["subject_id"].astype(int).isin(train_subjects), "split"] = "train"
    result.loc[result["subject_id"].astype(int).isin(val_subjects), "split"] = "validate"
    return result


def check_split_leakage(split_frames: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {
        "subject_overlap": {},
        "study_overlap": {},
        "dicom_overlap": {},
        "path_exists": {},
        "label_binary": {},
    }
    failures: list[str] = []

    for left_name, right_name in combinations(split_frames.keys(), 2):
        pair_name = f"{left_name}_vs_{right_name}"
        for column_name, bucket in [
            ("subject_id", "subject_overlap"),
            ("study_id", "study_overlap"),
            ("dicom_id", "dicom_overlap"),
        ]:
            left_values = set(split_frames[left_name][column_name].dropna().astype(str))
            right_values = set(split_frames[right_name][column_name].dropna().astype(str))
            overlap = sorted(left_values & right_values)
            checks[bucket][pair_name] = {"count": int(len(overlap)), "examples": overlap[:5]}
            if overlap:
                failures.append(f"{column_name} leakage between {left_name} and {right_name}: {overlap[:5]}")

    for split_name, frame in split_frames.items():
        if frame.empty:
            failures.append(f"{split_name} manifest is empty.")
        missing_paths = [path for path in frame["abs_path"].astype(str) if not Path(path).exists()]
        checks["path_exists"][split_name] = {
            "missing_count": int(len(missing_paths)),
            "examples": missing_paths[:5],
        }
        if missing_paths:
            failures.append(f"{split_name} manifest has missing image paths.")

        label_state: dict[str, Any] = {}
        for label_name in FINAL_LABELS:
            values = pd.to_numeric(frame[label_name], errors="coerce")
            invalid_mask = values.isna() | ~values.isin([0, 1])
            invalid_values = sorted(pd.unique(frame.loc[invalid_mask, label_name]))
            label_state[label_name] = {
                "invalid_count": int(invalid_mask.sum()),
                "invalid_values": invalid_values[:5],
            }
            if invalid_mask.any():
                failures.append(f"{split_name} label column '{label_name}' contains non-binary values.")
        checks["label_binary"][split_name] = label_state

    return checks, failures


def summarize_final_splits(split_frames: dict[str, pd.DataFrame], warnings: list[str], failures: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    test_frame = split_frames["test"]

    for split_name, frame in split_frames.items():
        split_summary: dict[str, Any] = {
            "image_count": int(len(frame)),
            "subject_count": int(frame["subject_id"].nunique()),
            "study_count": int(frame["study_id"].nunique()),
            "labels": {},
        }
        for label_name in FINAL_LABELS:
            positives = int(frame[label_name].sum())
            negatives = int(len(frame) - positives)
            prevalence = round(positives / len(frame), 6) if len(frame) else None
            split_summary["labels"][label_name] = {
                "positive_count": positives,
                "negative_count": negatives,
                "prevalence": prevalence,
            }
        summary[split_name] = split_summary

    for label_name in FINAL_LABELS:
        positives = int(test_frame[label_name].sum())
        negatives = int(len(test_frame) - positives)
        if positives == 0:
            failures.append(f"Test split has zero positives for {label_name}.")
        if negatives == 0:
            failures.append(f"Test split has zero negatives for {label_name}.")
        if positives < 10:
            warnings.append(f"Test split has fewer than 10 positives for {label_name}: {positives}")

    return summary


def format_percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 4)


def build_markdown_report(report: dict[str, Any]) -> str:
    sections = [
        "# Mini-Stage C MIMIC Check",
        "",
        "## Summary",
        "",
        json_block(
            {
                "status": report["status"],
                "safe_to_continue": report["safe_to_continue"],
            }
        ),
        "",
        "## Files Used",
        "",
        json_block(report["files_used"]),
        "",
        "## Counts",
        "",
        json_block(report["counts"]),
        "",
        "## Path Resolution",
        "",
        json_block(report["path_resolution"]),
        "",
        "## View Counts",
        "",
        json_block(report["view_counts"]),
        "",
        "## Label Value Counts",
        "",
        json_block(report["raw_label_value_counts"]),
        "",
        "## Final Label Counts By Split",
        "",
        json_block(report["label_counts_by_split"]),
        "",
        "## Leakage Checks",
        "",
        json_block(report["leakage_checks"]),
        "",
        "## Warnings",
        "",
        json_block(report["warnings"]),
        "",
        "## Failure Reasons",
        "",
        json_block(report["failure_reasons"]),
        "",
        "## Safe To Continue",
        "",
        json_block({"safe_to_continue": report["safe_to_continue"]}),
        "",
    ]
    return "\n".join(sections)


def save_reports(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    ensure_dir(out_dir)
    markdown_path = out_dir / "mini_stage_c_mimic_check.md"
    json_path = out_dir / "mini_stage_c_mimic_check.json"

    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")
    json_path.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True), encoding="utf-8")
    return markdown_path, json_path


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    warnings: list[str] = []
    partial_reasons: list[str] = []
    failures: list[str] = []

    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    ensure_dir(out_dir)
    ensure_dir(manifest_dir)

    mimic_root = choose_mimic_root(args.mimic_root, warnings)
    metadata_csv = find_required_file(
        args.metadata_csv,
        mimic_root,
        ["mimic-cxr-2.0.0-metadata.csv.gz", "mimic-cxr-2.0.0-metadata.csv"],
        "metadata CSV",
    )
    labels_csv = find_required_file(
        args.labels_csv,
        mimic_root,
        ["mimic-cxr-2.0.0-chexpert.csv.gz", "mimic-cxr-2.0.0-chexpert.csv"],
        "CheXpert labels CSV",
    )
    split_csv = find_required_file(
        args.split_csv,
        mimic_root,
        ["mimic-cxr-2.0.0-split.csv.gz", "mimic-cxr-2.0.0-split.csv"],
        "split CSV",
    )

    print_line(f"MIMIC root: {mimic_root}")
    print_line(f"Metadata CSV: {metadata_csv}")
    print_line(f"Labels CSV: {labels_csv}")
    print_line(f"Split CSV: {split_csv}")

    metadata_df = read_csv_checked(metadata_csv, "metadata CSV")
    split_df = read_csv_checked(split_csv, "split CSV")
    labels_df = read_csv_checked(labels_csv, "labels CSV")

    metadata_path_column = find_path_column(list(metadata_df.columns))
    split_path_column = find_path_column(list(split_df.columns))

    require_columns(metadata_df, ["dicom_id"], "metadata CSV")
    require_columns(split_df, ["dicom_id", "study_id", "subject_id", "split"], "split CSV")
    require_columns(labels_df, ["subject_id", "study_id", *MIMIC_LABELS], "labels CSV")

    metadata_df["dicom_id"] = normalize_dicom_ids(metadata_df["dicom_id"])
    if "subject_id" in metadata_df.columns:
        metadata_df["subject_id"] = normalize_int_ids(metadata_df["subject_id"])
    if "study_id" in metadata_df.columns:
        metadata_df["study_id"] = normalize_int_ids(metadata_df["study_id"])
    if "Rows" not in metadata_df.columns:
        warnings.append("Metadata column 'Rows' is not available.")
    if "Columns" not in metadata_df.columns:
        warnings.append("Metadata column 'Columns' is not available.")
    if "ViewPosition" not in metadata_df.columns:
        warnings.append("Metadata column 'ViewPosition' is not available. Frontal filtering cannot be applied.")

    split_df["dicom_id"] = normalize_dicom_ids(split_df["dicom_id"])
    split_df["subject_id"] = normalize_int_ids(split_df["subject_id"])
    split_df["study_id"] = normalize_int_ids(split_df["study_id"])
    split_df["split"] = split_df["split"].astype("string").str.strip()

    labels_df["subject_id"] = normalize_int_ids(labels_df["subject_id"])
    labels_df["study_id"] = normalize_int_ids(labels_df["study_id"])
    raw_label_value_counts = validate_and_summarize_label_values(labels_df)

    metadata_summary = {
        "rows": int(len(metadata_df)),
        "unique_dicom_id": int(metadata_df["dicom_id"].nunique(dropna=True)),
        "duplicate_dicom_id_count": int(metadata_df["dicom_id"].duplicated().sum()),
        "view_position_counts": (
            metadata_df["ViewPosition"].fillna("MISSING").value_counts(dropna=False).to_dict()
            if "ViewPosition" in metadata_df.columns
            else {}
        ),
    }
    split_value_counts = split_df["split"].fillna("MISSING").value_counts(dropna=False).to_dict()
    split_summary = {
        "rows_per_split": {str(key): int(value) for key, value in split_value_counts.items()},
        "subjects_per_split": {
            str(split_name): int(split_df.loc[split_df["split"] == split_name, "subject_id"].nunique())
            for split_name in sorted(pd.unique(split_df["split"].dropna()))
        },
        "studies_per_split": {
            str(split_name): int(split_df.loc[split_df["split"] == split_name, "study_id"].nunique())
            for split_name in sorted(pd.unique(split_df["split"].dropna()))
        },
    }

    split_values = set(split_df["split"].dropna().astype(str))
    unexpected_split_values = sorted(split_values - EXPECTED_SPLITS)
    missing_expected_split_values = sorted(EXPECTED_SPLITS - split_values)
    if unexpected_split_values:
        warnings.append(f"Unexpected split values found: {unexpected_split_values}")
    if missing_expected_split_values:
        warnings.append(f"Missing expected split values: {missing_expected_split_values}")

    local_df, local_image_summary = build_local_image_index(mimic_root)
    if local_image_summary["duplicate_local_dicom_ids"] > 0:
        warnings.append(
            f"Found duplicate local dicom_id image files: {local_image_summary['duplicate_local_dicom_ids']}"
        )

    merged_df, join_summary = merge_local_subset(local_df, metadata_df, split_df, labels_df, warnings)

    if join_summary["metadata_join_rate"] < 0.95:
        partial_reasons.append(
            f"Only {join_summary['metadata_join_rate']:.2%} of local images matched metadata."
        )
    if join_summary["split_join_rate"] < 0.95:
        partial_reasons.append(f"Only {join_summary['split_join_rate']:.2%} of local images matched split data.")
    if join_summary["rows_missing_metadata"] > 0:
        warnings.append(f"Local images missing metadata rows: {join_summary['rows_missing_metadata']}")
    if join_summary["rows_missing_split"] > 0:
        warnings.append(f"Local images missing split rows: {join_summary['rows_missing_split']}")
    if join_summary["rows_missing_label_row"] > 0:
        warnings.append(f"Rows missing a CheXpert label row: {join_summary['rows_missing_label_row']}")
    if join_summary["rows_all_target_labels_missing"] > 0:
        warnings.append(
            "Rows with all 5 raw target labels missing before U-zero: "
            f"{join_summary['rows_all_target_labels_missing']}"
        )

    candidate_df = merged_df.copy()
    candidate_df = candidate_df.loc[candidate_df["_metadata_merge"] == "both"].copy()
    candidate_df = candidate_df.loc[candidate_df["_split_merge"] == "both"].copy()
    candidate_df = resolve_candidate_paths(candidate_df, mimic_root, metadata_path_column, split_path_column)
    candidate_df["path_exists"] = candidate_df["abs_path"].map(lambda value: Path(str(value)).exists())

    resolved_paths = int(candidate_df["path_exists"].sum())
    missing_paths = int((~candidate_df["path_exists"]).sum())
    candidate_count = int(len(candidate_df))
    path_resolution = {
        "candidate_rows": candidate_count,
        "resolved_paths": resolved_paths,
        "missing_paths": missing_paths,
        "percent_resolved": format_percent(resolved_paths, candidate_count),
        "path_method_counts": candidate_df["path_method"].fillna("MISSING").value_counts(dropna=False).to_dict(),
        "metadata_path_column": metadata_path_column,
        "split_path_column": split_path_column,
    }
    if candidate_count == 0:
        failures.append("No clean candidate rows remained after metadata and split joins.")
    elif path_resolution["percent_resolved"] < 95.0:
        partial_reasons.append(
            f"Only {path_resolution['percent_resolved']}% of candidate image paths were resolved."
        )

    view_counts = {}
    excluded_non_frontal = 0
    if "ViewPosition" in candidate_df.columns:
        view_counts = {
            str(key): int(value)
            for key, value in candidate_df["ViewPosition"].fillna("MISSING").value_counts(dropna=False).to_dict().items()
        }
        frontal_df = candidate_df.loc[candidate_df["ViewPosition"].isin(FRONTAL_VIEWS)].copy()
        excluded_non_frontal = int(len(candidate_df) - len(frontal_df))
        if excluded_non_frontal > 0:
            warnings.append(f"Excluded non-frontal or missing-view images from primary manifests: {excluded_non_frontal}")
    else:
        frontal_df = candidate_df.copy()
        warnings.append("ViewPosition is unavailable, so frontal filtering was skipped.")

    if frontal_df.empty:
        failures.append("No rows remained after frontal filtering.")

    frontal_df = build_common5_labels(frontal_df)

    has_official_split = EXPECTED_SPLITS.issubset(set(frontal_df["split"].dropna().astype(str)))
    if has_official_split:
        split_ready_df = frontal_df.copy()
    else:
        warnings.append("Official split values were incomplete in the local subset. Created a subject-wise fallback split.")
        split_ready_df = make_subjectwise_split(frontal_df, args.seed)

    split_frames = {
        "train_pool": prepare_final_manifest(split_ready_df.loc[split_ready_df["split"] == "train"].copy(), "train_pool"),
        "val": prepare_final_manifest(split_ready_df.loc[split_ready_df["split"] == "validate"].copy(), "val"),
        "test": prepare_final_manifest(split_ready_df.loc[split_ready_df["split"] == "test"].copy(), "test"),
    }

    for split_name, frame in split_frames.items():
        if frame["abs_path"].isna().any():
            failures.append(f"{split_name} manifest contains missing abs_path values.")

    leakage_checks, leakage_failures = check_split_leakage(split_frames)
    failures.extend(leakage_failures)

    label_counts_by_split = summarize_final_splits(split_frames, warnings, failures)

    manifest_paths = {
        "train_pool": manifest_dir / "mimic_common5_train_pool.csv",
        "val": manifest_dir / "mimic_common5_val.csv",
        "test": manifest_dir / "mimic_common5_test.csv",
    }
    for split_name, manifest_path in manifest_paths.items():
        ensure_dir(manifest_path.parent)
        split_frames[split_name].to_csv(manifest_path, index=False)

    final_missing_paths = {
        split_name: int((~frame["abs_path"].map(lambda value: Path(str(value)).exists())).sum())
        for split_name, frame in split_frames.items()
    }
    if any(count > 0 for count in final_missing_paths.values()):
        failures.append(f"Final manifests contain missing paths: {final_missing_paths}")

    counts = {
        "metadata": metadata_summary,
        "split": split_summary,
        "labels_rows": int(len(labels_df)),
        "local_images": local_image_summary,
        "join": join_summary,
        "frontal_candidate_rows": int(len(frontal_df)),
        "excluded_non_frontal": excluded_non_frontal,
        "manifest_rows": {split_name: int(len(frame)) for split_name, frame in split_frames.items()},
    }

    status = "DONE"
    if failures:
        status = "FAILED"
    elif partial_reasons:
        status = "PARTIAL"

    report = {
        "status": status,
        "safe_to_continue": status == "DONE",
        "files_used": {
            "mimic_root": str(mimic_root),
            "metadata_csv": str(metadata_csv),
            "labels_csv": str(labels_csv),
            "split_csv": str(split_csv),
            "train_manifest": str(manifest_paths["train_pool"]),
            "val_manifest": str(manifest_paths["val"]),
            "test_manifest": str(manifest_paths["test"]),
        },
        "counts": counts,
        "path_resolution": path_resolution,
        "view_counts": view_counts,
        "raw_label_value_counts": raw_label_value_counts,
        "label_counts_by_split": label_counts_by_split,
        "leakage_checks": leakage_checks,
        "warnings": warnings,
        "failure_reasons": failures + partial_reasons,
    }

    return report


def print_console_summary(report: dict[str, Any]) -> None:
    print_line(f"Metadata rows: {report['counts']['metadata']['rows']}")
    print_line(f"Metadata unique dicom_id: {report['counts']['metadata']['unique_dicom_id']}")
    print_line(f"Metadata duplicate dicom_id count: {report['counts']['metadata']['duplicate_dicom_id_count']}")

    rows_per_split = report["counts"]["split"]["rows_per_split"]
    print_line(f"Split rows per split: {rows_per_split}")
    print_line(f"Split subjects per split: {report['counts']['split']['subjects_per_split']}")
    print_line(f"Split studies per split: {report['counts']['split']['studies_per_split']}")

    print_line(f"Joined image rows: {report['counts']['join']['joined_image_rows']}")
    print_line(f"Joined unique subjects: {report['counts']['join']['unique_subjects']}")
    print_line(f"Joined unique studies: {report['counts']['join']['unique_studies']}")
    print_line(f"Rows missing metadata: {report['counts']['join']['rows_missing_metadata']}")
    print_line(f"Rows missing label rows: {report['counts']['join']['rows_missing_label_row']}")

    path_summary = report["path_resolution"]
    print_line(
        "Path resolution: "
        f"{path_summary['resolved_paths']}/{path_summary['candidate_rows']} "
        f"({path_summary['percent_resolved']}%)"
    )
    if report["view_counts"]:
        print_line(f"View counts: {report['view_counts']}")

    for split_name, split_summary in report["label_counts_by_split"].items():
        label_bits = ", ".join(
            f"{label_name}={values['positive_count']}"
            for label_name, values in split_summary["labels"].items()
        )
        print_line(
            f"{split_name}: images={split_summary['image_count']}, "
            f"subjects={split_summary['subject_count']}, studies={split_summary['study_count']}, {label_bits}"
        )

    if report["warnings"]:
        print_line("Warnings:")
        for warning in report["warnings"]:
            print_line(f"- {warning}")

    if report["failure_reasons"]:
        print_line("Failure reasons:")
        for reason in report["failure_reasons"]:
            print_line(f"- {reason}")

    print_line(f"Safe to continue: {report['safe_to_continue']}")
    print_line(report["status"])


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    try:
        report = run_stage(args)
    except StageFailure as exc:
        failure_report = {
            "status": "FAILED",
            "safe_to_continue": False,
            "files_used": {
                "mimic_root": clean_optional_path(args.mimic_root),
                "metadata_csv": clean_optional_path(args.metadata_csv),
                "labels_csv": clean_optional_path(args.labels_csv),
                "split_csv": clean_optional_path(args.split_csv),
            },
            "counts": {},
            "path_resolution": {},
            "view_counts": {},
            "raw_label_value_counts": {},
            "label_counts_by_split": {},
            "leakage_checks": {},
            "warnings": [],
            "failure_reasons": [str(exc)],
        }
        save_reports(failure_report, out_dir)
        print_line(f"FAILED: {exc}")
        print_line("FAILED")
        sys.exit(1)

    save_reports(report, out_dir)
    print_console_summary(report)

    if report["status"] == "FAILED":
        sys.exit(1)
    if report["status"] == "PARTIAL":
        sys.exit(2)


if __name__ == "__main__":
    main()
