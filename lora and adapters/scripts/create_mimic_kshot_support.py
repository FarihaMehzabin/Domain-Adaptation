#!/usr/bin/env python3
"""Create small multi-label MIMIC K-shot support sets from a clean train pool."""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
IMAGE_PATH_COLUMN_CANDIDATES = ["abs_path", "image_path", "filepath", "file_path", "path", "rel_path"]
ID_COLUMNS = ["subject_id", "study_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create clean multi-label K-shot support sets from MIMIC train_pool.")
    parser.add_argument(
        "--train_pool_csv",
        type=str,
        default="manifests/mimic_common5_train_pool.csv",
        help="CSV used as the only source for support samples.",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="manifests/mimic_common5_val.csv",
        help="Existing validation CSV used only for leakage checks and future adaptation validation.",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="manifests/mimic_common5_test.csv",
        help="Existing test CSV used only for leakage checks.",
    )
    parser.add_argument("--out_dir", type=str, default="reports", help="Directory for Markdown and JSON reports.")
    parser.add_argument("--manifest_dir", type=str, default="manifests", help="Directory for support manifests.")
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[5, 20],
        help="K values to create. Example: --k_values 5 20",
    )
    parser.add_argument("--seed", type=int, default=2027, help="Random seed for deterministic tie-breaking.")
    parser.add_argument("--debug", action="store_true", help="Print extra details while running.")
    return parser.parse_args()


def print_line(message: str) -> None:
    print(message, flush=True)


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
    if value is pd.NA:
        return None
    return value


def json_block(data: Any) -> str:
    return "```json\n" + json.dumps(json_ready(data), indent=2, sort_keys=True) + "\n```"


def serialize_examples(values: list[Any], limit: int = 5) -> list[Any]:
    return [json_ready(value) for value in values[:limit]]


def build_empty_report(input_files: dict[str, str], out_dir: Path, manifest_dir: Path, k_values: list[int], seed: int) -> dict[str, Any]:
    return {
        "status": "FAILED",
        "safe_to_continue": False,
        "goal": "Create clean K-shot support sets from MIMIC train_pool.",
        "input_files": input_files,
        "adaptation_val_csv": input_files["val_csv"],
        "out_dir": str(out_dir),
        "manifest_dir": str(manifest_dir),
        "k_values": k_values,
        "seed": seed,
        "label_order": LABELS,
        "support_files": {},
        "achieved_positives_by_k": {},
        "support_sizes_by_k": {},
        "leakage_checks": {
            "subject_overlap": {},
            "study_overlap": {},
            "image_overlap": {},
            "dicom_overlap": {},
            "label_binary": {},
            "image_path_columns": {},
        },
        "warnings": [],
        "failure_reasons": [],
    }


def resolve_path(raw_path: str) -> Path:
    return Path(raw_path).expanduser().resolve()


def check_required_files(paths: dict[str, Path]) -> list[str]:
    failures: list[str] = []
    for label, path in paths.items():
        if not path.exists():
            failures.append(f"Missing required file for {label}: {path}")
        elif not path.is_file():
            failures.append(f"Expected a file for {label}, but found something else: {path}")
    return failures


def read_csv_checked(path: Path, manifest_name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pragma: no cover - exercised only on bad local files
        raise RuntimeError(f"Could not read {manifest_name} at {path}: {exc}") from exc


def find_image_path_column(columns: list[str]) -> str | None:
    for column in IMAGE_PATH_COLUMN_CANDIDATES:
        if column in columns:
            return column
    return None


def validate_required_columns(df: pd.DataFrame, manifest_name: str) -> tuple[str | None, list[str]]:
    failures: list[str] = []
    path_column = find_image_path_column(list(df.columns))
    if path_column is None:
        failures.append(
            f"{manifest_name} is missing an image path column. Expected one of: {IMAGE_PATH_COLUMN_CANDIDATES}"
        )

    required_columns = [*ID_COLUMNS, *LABELS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        failures.append(f"{manifest_name} is missing required columns: {missing_columns}")

    if "dicom_id" not in df.columns:
        failures.append(f"{manifest_name} is missing required column: dicom_id")

    return path_column, failures


def validate_non_empty_paths(df: pd.DataFrame, manifest_name: str, path_column: str) -> list[str]:
    values = df[path_column].astype("string").str.strip()
    missing_mask = values.isna() | (values == "")
    if not missing_mask.any():
        return []
    return [f"{manifest_name} has {int(missing_mask.sum())} missing values in image path column '{path_column}'."]


def validate_binary_labels(df: pd.DataFrame, manifest_name: str) -> tuple[dict[str, Any], list[str]]:
    label_summary: dict[str, Any] = {}
    failures: list[str] = []

    for label in LABELS:
        values = pd.to_numeric(df[label], errors="coerce")
        invalid_mask = values.isna() | ~values.isin([0, 1])
        invalid_values = pd.unique(df.loc[invalid_mask, label]).tolist()
        label_summary[label] = {
            "invalid_count": int(invalid_mask.sum()),
            "invalid_values": serialize_examples(invalid_values),
        }
        if invalid_mask.any():
            failures.append(
                f"{manifest_name} label column '{label}' is not binary 0/1. "
                f"Found invalid values: {serialize_examples(invalid_values)}"
            )
        else:
            df[label] = values.astype(int)

    return label_summary, failures


def normalize_overlap_values(series: pd.Series) -> set[str]:
    normalized = series.dropna().astype("string").str.strip()
    normalized = normalized[normalized != ""]
    return set(normalized.tolist())


def summarize_overlap(left_values: set[str], right_values: set[str]) -> dict[str, Any]:
    overlap = sorted(left_values & right_values)
    return {"count": int(len(overlap)), "examples": overlap[:5]}


def check_leakage(split_frames: dict[str, pd.DataFrame], image_path_columns: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {
        "subject_overlap": {},
        "study_overlap": {},
        "image_overlap": {},
        "dicom_overlap": {},
        "label_binary": {},
        "image_path_columns": image_path_columns,
    }
    failures: list[str] = []

    for split_name, frame in split_frames.items():
        label_state, label_failures = validate_binary_labels(frame, split_name)
        checks["label_binary"][split_name] = label_state
        failures.extend(label_failures)

    for left_name, right_name in combinations(split_frames.keys(), 2):
        pair_name = f"{left_name}_vs_{right_name}"
        left_frame = split_frames[left_name]
        right_frame = split_frames[right_name]

        subject_overlap = summarize_overlap(
            normalize_overlap_values(left_frame["subject_id"]),
            normalize_overlap_values(right_frame["subject_id"]),
        )
        checks["subject_overlap"][pair_name] = subject_overlap
        if subject_overlap["count"] > 0:
            failures.append(f"subject_id leakage between {left_name} and {right_name}: {subject_overlap['examples']}")

        study_overlap = summarize_overlap(
            normalize_overlap_values(left_frame["study_id"]),
            normalize_overlap_values(right_frame["study_id"]),
        )
        checks["study_overlap"][pair_name] = study_overlap
        if study_overlap["count"] > 0:
            failures.append(f"study_id leakage between {left_name} and {right_name}: {study_overlap['examples']}")

        image_overlap = summarize_overlap(
            normalize_overlap_values(left_frame[image_path_columns[left_name]]),
            normalize_overlap_values(right_frame[image_path_columns[right_name]]),
        )
        checks["image_overlap"][pair_name] = image_overlap
        if image_overlap["count"] > 0:
            failures.append(
                f"image path leakage between {left_name} and {right_name}: {image_overlap['examples']}"
            )

        dicom_overlap = summarize_overlap(
            normalize_overlap_values(left_frame["dicom_id"]),
            normalize_overlap_values(right_frame["dicom_id"]),
        )
        checks["dicom_overlap"][pair_name] = dicom_overlap
        if dicom_overlap["count"] > 0:
            failures.append(f"dicom_id leakage between {left_name} and {right_name}: {dicom_overlap['examples']}")

    return checks, failures


def compute_positive_counts(df: pd.DataFrame) -> dict[str, int]:
    return {label: int(df[label].sum()) for label in LABELS}


def choose_best_candidate(
    train_pool_df: pd.DataFrame,
    selected_indices: set[int],
    remaining_by_label: dict[str, int],
    random_priority: dict[int, int],
) -> int | None:
    unmet_labels = [label for label, remaining in remaining_by_label.items() if remaining > 0]
    if not unmet_labels:
        return None

    best_index: int | None = None
    best_score: tuple[int, int, int, int] | None = None

    for row_index, row in train_pool_df.iterrows():
        if row_index in selected_indices:
            continue

        gain_labels = sum(int(row[label]) for label in unmet_labels)
        if gain_labels == 0:
            continue

        weighted_gain = sum(remaining_by_label[label] * int(row[label]) for label in unmet_labels)
        total_positive_labels = sum(int(row[label]) for label in LABELS)
        score = (weighted_gain, gain_labels, total_positive_labels, -random_priority[row_index])

        if best_score is None or score > best_score:
            best_score = score
            best_index = row_index

    return best_index


def prune_selected_rows(selected_df: pd.DataFrame, required_by_label: dict[str, int]) -> pd.DataFrame:
    if selected_df.empty:
        return selected_df.copy()

    keep_indices = list(selected_df.index)
    current_counts = compute_positive_counts(selected_df)

    for row_index in reversed(keep_indices.copy()):
        row = selected_df.loc[row_index]
        can_remove = True
        for label in LABELS:
            if current_counts[label] - int(row[label]) < required_by_label[label]:
                can_remove = False
                break
        if not can_remove:
            continue
        keep_indices.remove(row_index)
        for label in LABELS:
            current_counts[label] -= int(row[label])

    return selected_df.loc[keep_indices].copy()


def select_kshot_support(train_pool_df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    if k <= 0:
        raise ValueError(f"K must be positive. Received: {k}")

    available_by_label = compute_positive_counts(train_pool_df)
    required_by_label = {label: min(k, available_by_label[label]) for label in LABELS}

    if sum(required_by_label.values()) == 0:
        return train_pool_df.iloc[0:0].copy()

    rng = np.random.default_rng(seed)
    random_order = rng.permutation(len(train_pool_df))
    random_priority = {row_index: int(priority) for row_index, priority in zip(train_pool_df.index, random_order)}

    selected_indices: list[int] = []
    selected_index_set: set[int] = set()
    achieved_by_label = {label: 0 for label in LABELS}

    while True:
        remaining_by_label = {
            label: max(required_by_label[label] - achieved_by_label[label], 0)
            for label in LABELS
        }
        if all(remaining == 0 for remaining in remaining_by_label.values()):
            break

        best_index = choose_best_candidate(train_pool_df, selected_index_set, remaining_by_label, random_priority)
        if best_index is None:
            break

        selected_indices.append(best_index)
        selected_index_set.add(best_index)
        selected_row = train_pool_df.loc[best_index]
        for label in LABELS:
            achieved_by_label[label] += int(selected_row[label])

    selected_df = train_pool_df.loc[selected_indices].copy()
    selected_df = prune_selected_rows(selected_df, required_by_label)
    return selected_df


def validate_support_manifest(
    support_df: pd.DataFrame,
    train_pool_df: pd.DataFrame,
    path_column: str,
) -> list[str]:
    failures: list[str] = []

    if support_df.duplicated().any():
        failures.append("Support manifest contains duplicate rows.")

    support_paths = normalize_overlap_values(support_df[path_column])
    train_pool_paths = normalize_overlap_values(train_pool_df[path_column])
    if not support_paths.issubset(train_pool_paths):
        failures.append("Support manifest contains rows outside the train_pool image set.")

    if "dicom_id" in support_df.columns:
        duplicate_dicom_mask = support_df["dicom_id"].astype("string").str.strip().duplicated()
        if duplicate_dicom_mask.any():
            failures.append("Support manifest contains duplicate dicom_id values.")

    duplicate_path_mask = support_df[path_column].astype("string").str.strip().duplicated()
    if duplicate_path_mask.any():
        failures.append(f"Support manifest contains duplicate image paths in column '{path_column}'.")

    support_path_column, column_failures = validate_required_columns(support_df, "support_manifest")
    failures.extend(column_failures)
    if support_path_column is not None:
        failures.extend(validate_non_empty_paths(support_df, "support_manifest", support_path_column))

    label_summary, label_failures = validate_binary_labels(support_df, "support_manifest")
    del label_summary  # not needed here, but validation keeps the function symmetric with input checks
    failures.extend(label_failures)

    return failures


def build_support_summary(support_df: pd.DataFrame, k: int) -> dict[str, Any]:
    positive_counts = compute_positive_counts(support_df) if not support_df.empty else {label: 0 for label in LABELS}
    return {
        "k": int(k),
        "total_images_selected": int(len(support_df)),
        "total_subjects_selected": int(support_df["subject_id"].nunique()) if "subject_id" in support_df.columns else 0,
        "total_studies_selected": int(support_df["study_id"].nunique()) if "study_id" in support_df.columns else 0,
        "positive_counts": positive_counts,
        "reached_k": {label: bool(count >= k) for label, count in positive_counts.items()},
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    sections = [
        "# Mini-Stage E K-shot Support",
        "",
        "## Goal",
        "",
        report["goal"],
        "",
        "## Input Files",
        "",
        json_block(report["input_files"]),
        "",
        "## Label Names",
        "",
        json_block(report["label_order"]),
        "",
        "## K-shot Definition",
        "",
        "K-shot means we try to select at least K positive examples for each disease. "
        "This is multi-label data, so one image may count for more than one disease.",
        "",
        "## Leakage Checks",
        "",
        json_block(report["leakage_checks"]),
        "",
        "## Support Set Summary: K=5",
        "",
        json_block(report["support_sizes_by_k"].get("5", {})),
        "",
        json_block(report["achieved_positives_by_k"].get("5", {})),
        "",
        "## Support Set Summary: K=20",
        "",
        json_block(report["support_sizes_by_k"].get("20", {})),
        "",
        json_block(report["achieved_positives_by_k"].get("20", {})),
        "",
        "## Adaptation Validation Set",
        "",
        f"Use the existing validation manifest only: `{report['adaptation_val_csv']}`",
        "",
        "## Warnings",
        "",
        json_block(report["warnings"]),
        "",
        "## Final Decision",
        "",
        json_block(
            {
                "status": report["status"],
                "safe_to_continue": report["safe_to_continue"],
                "failure_reasons": report["failure_reasons"],
            }
        ),
        "",
    ]
    return "\n".join(sections)


def save_reports(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    ensure_dir(out_dir)
    markdown_path = out_dir / "mini_stage_e_kshot_support.md"
    json_path = out_dir / "mini_stage_e_kshot_support.json"
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")
    json_path.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True), encoding="utf-8")
    return markdown_path, json_path


def determine_status(failures: list[str], warnings: list[str]) -> str:
    if failures:
        return "FAILED"
    if any("could not reach" in warning for warning in warnings):
        return "PARTIAL"
    return "DONE"


def run_stage(args: argparse.Namespace) -> tuple[dict[str, Any], tuple[Path, Path]]:
    out_dir = resolve_path(args.out_dir)
    manifest_dir = resolve_path(args.manifest_dir)
    ensure_dir(out_dir)
    ensure_dir(manifest_dir)

    input_files = {
        "train_pool_csv": str(resolve_path(args.train_pool_csv)),
        "val_csv": str(resolve_path(args.val_csv)),
        "test_csv": str(resolve_path(args.test_csv)),
    }
    report = build_empty_report(input_files, out_dir, manifest_dir, list(args.k_values), args.seed)
    warnings: list[str] = []
    failures: list[str] = []

    csv_paths = {key: Path(value) for key, value in input_files.items()}
    failures.extend(check_required_files(csv_paths))
    if failures:
        report["warnings"] = warnings
        report["failure_reasons"] = failures
        report["status"] = "FAILED"
        report["safe_to_continue"] = False
        report_paths = save_reports(report, out_dir)
        return report, report_paths

    manifest_names = {
        "train_pool": "train_pool_csv",
        "val": "val_csv",
        "test": "test_csv",
    }
    split_frames: dict[str, pd.DataFrame] = {}
    image_path_columns: dict[str, str] = {}

    for split_name, file_key in manifest_names.items():
        manifest_path = csv_paths[file_key]
        try:
            frame = read_csv_checked(manifest_path, split_name)
        except RuntimeError as exc:
            failures.append(str(exc))
            continue

        path_column, column_failures = validate_required_columns(frame, split_name)
        failures.extend(column_failures)
        if path_column is None:
            continue

        image_path_columns[split_name] = path_column
        failures.extend(validate_non_empty_paths(frame, split_name, path_column))
        split_frames[split_name] = frame

    if failures:
        report["leakage_checks"]["image_path_columns"] = image_path_columns
        report["warnings"] = warnings
        report["failure_reasons"] = failures
        report["status"] = "FAILED"
        report["safe_to_continue"] = False
        report_paths = save_reports(report, out_dir)
        return report, report_paths

    leakage_checks, leakage_failures = check_leakage(split_frames, image_path_columns)
    report["leakage_checks"] = leakage_checks
    failures.extend(leakage_failures)

    if failures:
        report["warnings"] = warnings
        report["failure_reasons"] = failures
        report["status"] = "FAILED"
        report["safe_to_continue"] = False
        report_paths = save_reports(report, out_dir)
        return report, report_paths

    train_pool_df = split_frames["train_pool"]
    train_pool_path_column = image_path_columns["train_pool"]

    support_files: dict[str, str] = {}
    achieved_positives_by_k: dict[str, Any] = {}
    support_sizes_by_k: dict[str, Any] = {}

    for k in args.k_values:
        support_df = select_kshot_support(train_pool_df, k=k, seed=args.seed)
        support_failures = validate_support_manifest(support_df, train_pool_df, train_pool_path_column)
        failures.extend(support_failures)

        summary = build_support_summary(support_df, k)
        k_key = str(k)
        achieved_positives_by_k[k_key] = summary["positive_counts"]
        support_sizes_by_k[k_key] = {
            "total_images_selected": summary["total_images_selected"],
            "total_subjects_selected": summary["total_subjects_selected"],
            "total_studies_selected": summary["total_studies_selected"],
            "reached_k": summary["reached_k"],
        }

        for label, achieved in summary["positive_counts"].items():
            if achieved < k:
                warnings.append(f"K={k}: {label} could not reach {k} positives. Achieved {achieved}.")

        output_path = manifest_dir / f"mimic_support_k{k}_seed{args.seed}.csv"
        support_df.to_csv(output_path, index=False)
        support_files[k_key] = str(output_path)

    report["support_files"] = support_files
    report["achieved_positives_by_k"] = achieved_positives_by_k
    report["support_sizes_by_k"] = support_sizes_by_k
    report["warnings"] = warnings
    report["failure_reasons"] = failures
    report["status"] = determine_status(failures, warnings)
    report["safe_to_continue"] = report["status"] == "DONE"

    report_paths = save_reports(report, out_dir)
    return report, report_paths


def print_console_summary(report: dict[str, Any], report_paths: tuple[Path, Path], debug: bool) -> None:
    print_line(f"Goal: {report['goal']}")
    print_line(f"Train pool CSV: {report['input_files']['train_pool_csv']}")
    print_line(f"Validation CSV: {report['input_files']['val_csv']}")
    print_line(f"Test CSV: {report['input_files']['test_csv']}")
    print_line(f"Adaptation validation set: {report['adaptation_val_csv']}")

    image_path_columns = report["leakage_checks"].get("image_path_columns", {})
    if image_path_columns:
        print_line(f"Image path columns: {image_path_columns}")

    if debug:
        print_line("Leakage checks:")
        print_line(json.dumps(json_ready(report["leakage_checks"]), indent=2, sort_keys=True))

    for k in report["k_values"]:
        k_key = str(k)
        size_summary = report["support_sizes_by_k"].get(k_key, {})
        positive_counts = report["achieved_positives_by_k"].get(k_key, {})
        support_file = report["support_files"].get(k_key)

        print_line(f"K={k}:")
        if size_summary:
            print_line(f"- total images selected: {size_summary['total_images_selected']}")
            print_line(f"- total subjects selected: {size_summary['total_subjects_selected']}")
            print_line(f"- total studies selected: {size_summary['total_studies_selected']}")
        for label in LABELS:
            achieved = int(positive_counts.get(label, 0))
            print_line(f"- {label}: achieved {achieved} / required {k}")
        if support_file:
            print_line(f"- support file: {support_file}")

    if report["warnings"]:
        print_line("Warnings:")
        for warning in report["warnings"]:
            print_line(f"- {warning}")

    if report["failure_reasons"]:
        print_line("Failure reasons:")
        for reason in report["failure_reasons"]:
            print_line(f"- {reason}")

    print_line(f"Markdown report: {report_paths[0]}")
    print_line(f"JSON report: {report_paths[1]}")
    print_line(f"Safe to continue: {report['safe_to_continue']}")
    print_line(report["status"])


def main() -> int:
    args = parse_args()
    report, report_paths = run_stage(args)
    print_console_summary(report, report_paths, debug=args.debug)
    if report["status"] == "FAILED":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
