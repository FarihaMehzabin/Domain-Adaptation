#!/usr/bin/env python3
"""Create Policy B K-shot support manifests from the Policy B train pool."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]


class StageFailure(RuntimeError):
    """Raised when support creation cannot continue safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Policy B MIMIC common5 support manifests.")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--train_pool_csv",
        type=str,
        default="manifests/mimic_common5_policyB_train_pool.csv",
        help="Policy B train-pool manifest used as the only support source.",
    )
    parser.add_argument("--k_values", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting Policy B support outputs.")
    return parser.parse_args()


def print_line(message: str) -> None:
    print(message, flush=True)


def read_csv_checked(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise StageFailure(f"Could not read CSV {path}: {exc}") from exc


def validate_train_pool(train_pool: pd.DataFrame, path: Path) -> None:
    required_columns = ["dicom_id", "subject_id", "study_id", "split", "label_policy", "label_set"]
    for label in LABELS:
        required_columns.append(label)
        required_columns.append(f"{label}_mask")
    missing = [column for column in required_columns if column not in train_pool.columns]
    if missing:
        raise StageFailure(f"Train pool {path} is missing required columns: {missing}")
    if train_pool["label_policy"].astype("string").str.strip().nunique(dropna=False) != 1:
        raise StageFailure("Train pool has inconsistent label_policy values.")
    if train_pool["label_policy"].astype("string").str.strip().iloc[0] != "uignore_blankzero":
        raise StageFailure("Train pool is not tagged with label_policy=uignore_blankzero.")
    if train_pool["split"].astype("string").str.strip().nunique(dropna=False) != 1:
        raise StageFailure("Train pool has inconsistent split values.")
    if train_pool["split"].astype("string").str.strip().iloc[0] != "train_pool":
        raise StageFailure("Train pool split must be train_pool.")
    dicom_values = train_pool["dicom_id"].astype("string").str.strip()
    if dicom_values.isna().any() or (dicom_values == "").any():
        raise StageFailure("Train pool has blank dicom_id values.")


def positive_indicator_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    indicators = {}
    for label in LABELS:
        label_values = pd.to_numeric(dataframe[label], errors="coerce")
        mask_values = pd.to_numeric(dataframe[f"{label}_mask"], errors="coerce")
        if label_values.isna().any() or mask_values.isna().any():
            raise StageFailure(f"Found NaN values in {label} or {label}_mask.")
        invalid_mask = ~mask_values.isin([0, 1]) | ~label_values.isin([0, 1])
        if invalid_mask.any():
            raise StageFailure(f"Found non-binary values in {label} or {label}_mask.")
        indicators[label] = ((label_values == 1) & (mask_values == 1)).astype(int)
    return pd.DataFrame(indicators, index=dataframe.index)


def choose_best_candidate(
    positives: pd.DataFrame,
    selected_indices: set[int],
    selected_dicom_ids: set[str],
    dicom_by_index: dict[int, str | None],
    remaining_by_label: dict[str, int],
    random_priority: dict[int, int],
    avoid_duplicate_dicom: bool,
) -> int | None:
    unmet = [label for label, remaining in remaining_by_label.items() if remaining > 0]
    if not unmet:
        return None

    best_index: int | None = None
    best_score: tuple[int, int, int, int, int] | None = None
    for row_index, row in positives.iterrows():
        if row_index in selected_indices:
            continue
        dicom_id = dicom_by_index.get(row_index)
        duplicate_dicom = dicom_id is not None and dicom_id in selected_dicom_ids
        if avoid_duplicate_dicom and duplicate_dicom:
            continue

        gain_labels = sum(int(row[label]) for label in unmet)
        if gain_labels == 0:
            continue

        weighted_gain = sum(remaining_by_label[label] * int(row[label]) for label in unmet)
        total_positive_labels = sum(int(row[label]) for label in LABELS)
        duplicate_penalty = -1 if duplicate_dicom else 0
        score = (weighted_gain, gain_labels, total_positive_labels, duplicate_penalty, -random_priority[row_index])
        if best_score is None or score > best_score:
            best_score = score
            best_index = row_index
    return best_index


def prune_selected_rows(
    selected_indices: list[int],
    positives: pd.DataFrame,
    required_by_label: dict[str, int],
) -> list[int]:
    keep = list(selected_indices)
    selected_frame = positives.loc[keep]
    current_counts = {label: int(selected_frame[label].sum()) for label in LABELS}

    for row_index in reversed(keep.copy()):
        row = positives.loc[row_index]
        if any(current_counts[label] - int(row[label]) < required_by_label[label] for label in LABELS):
            continue
        keep.remove(row_index)
        for label in LABELS:
            current_counts[label] -= int(row[label])
    return keep


def select_support_indices(train_pool: pd.DataFrame, positives: pd.DataFrame, k: int, seed: int) -> tuple[list[int], bool]:
    available = {label: int(positives[label].sum()) for label in LABELS}
    insufficient = {label: count for label, count in available.items() if count < k}
    if insufficient:
        raise StageFailure(f"Could not satisfy K={k}. Available positives: {insufficient}")

    rng = np.random.default_rng(seed)
    random_order = rng.permutation(len(train_pool))
    random_priority = {row_index: int(priority) for row_index, priority in zip(train_pool.index, random_order)}
    dicom_by_index = {}
    if "dicom_id" in train_pool.columns:
        dicom_by_index = {
            row_index: (None if pd.isna(value) or str(value).strip() == "" else str(value).strip())
            for row_index, value in train_pool["dicom_id"].items()
        }

    for avoid_duplicate_dicom in [True, False]:
        selected_indices: list[int] = []
        selected_index_set: set[int] = set()
        selected_dicom_ids: set[str] = set()
        achieved = {label: 0 for label in LABELS}

        while True:
            remaining = {label: max(k - achieved[label], 0) for label in LABELS}
            if all(value == 0 for value in remaining.values()):
                pruned = prune_selected_rows(selected_indices, positives, {label: k for label in LABELS})
                return pruned, not avoid_duplicate_dicom

            best_index = choose_best_candidate(
                positives=positives,
                selected_indices=selected_index_set,
                selected_dicom_ids=selected_dicom_ids,
                dicom_by_index=dicom_by_index,
                remaining_by_label=remaining,
                random_priority=random_priority,
                avoid_duplicate_dicom=avoid_duplicate_dicom,
            )
            if best_index is None:
                break

            selected_indices.append(best_index)
            selected_index_set.add(best_index)
            dicom_id = dicom_by_index.get(best_index)
            if dicom_id is not None:
                selected_dicom_ids.add(dicom_id)
            for label in LABELS:
                achieved[label] += int(positives.at[best_index, label])

    raise StageFailure(f"Could not satisfy K={k} with the available train_pool rows.")


def summarize_support(dataframe: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "support_size": int(len(dataframe)),
        "subject_count": int(dataframe["subject_id"].nunique()) if "subject_id" in dataframe.columns else 0,
        "study_count": int(dataframe["study_id"].nunique()) if "study_id" in dataframe.columns else 0,
        "dicom_count": int(dataframe["dicom_id"].nunique()) if "dicom_id" in dataframe.columns else 0,
        "per_label": {},
    }
    for label in LABELS:
        label_values = pd.to_numeric(dataframe[label], errors="coerce").astype(int)
        mask_values = pd.to_numeric(dataframe[f"{label}_mask"], errors="coerce").astype(int)
        positive = int(((label_values == 1) & (mask_values == 1)).sum())
        negative = int(((label_values == 0) & (mask_values == 1)).sum())
        masked = int((mask_values == 0).sum())
        summary["per_label"][label] = {
            "positive": positive,
            "negative": negative,
            "masked": masked,
        }
    return summary


def ensure_output_paths(root: Path, k_values: list[int], seed: int, overwrite: bool) -> dict[int, Path]:
    outputs = {
        k: root / "manifests" / f"mimic_common5_policyB_support_k{k}_seed{seed}.csv"
        for k in k_values
    }
    if overwrite:
        return outputs
    existing = [str(path.resolve()) for path in outputs.values() if path.exists()]
    if existing:
        raise StageFailure(
            "Refusing to overwrite existing Policy B support manifests. Pass --overwrite to replace: "
            f"{existing}"
        )
    return outputs


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    train_pool_path = (root / args.train_pool_csv).resolve()
    outputs = ensure_output_paths(root, args.k_values, args.seed, overwrite=args.overwrite)

    train_pool = read_csv_checked(train_pool_path)
    validate_train_pool(train_pool, train_pool_path)
    positives = positive_indicator_frame(train_pool)

    for k in args.k_values:
        selected_indices, used_duplicate_fallback = select_support_indices(train_pool, positives, k=k, seed=args.seed)
        support_df = train_pool.loc[selected_indices].copy()
        if "dicom_id" in support_df.columns and support_df["dicom_id"].astype("string").str.strip().duplicated().any():
            raise StageFailure(f"Support manifest for K={k} still contains duplicate dicom_id values.")
        summary = summarize_support(support_df)
        output_path = outputs[k]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        support_df.to_csv(output_path, index=False)
        print_line(f"created K={k} support_size={summary['support_size']} output={output_path}")
        for label in LABELS:
            counts = summary["per_label"][label]
            print_line(
                f"- {label}: positive={counts['positive']} negative={counts['negative']} masked={counts['masked']}"
            )
        if used_duplicate_fallback:
            print_line(f"- warning: duplicate dicom_id avoidance could not be maintained for K={k}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
