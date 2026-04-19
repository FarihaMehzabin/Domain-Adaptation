#!/usr/bin/env python3
"""Build a PhysioNet-backed MIMIC-CXR-JPG subset manifest and download plan."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

LABEL_COLUMNS = (
    "label_atelectasis",
    "label_cardiomegaly",
    "label_consolidation",
    "label_edema",
    "label_pleural_effusion",
    "label_pneumonia",
    "label_pneumothorax",
)

LABEL_NAME_MAP = {
    "label_atelectasis": "Atelectasis",
    "label_cardiomegaly": "Cardiomegaly",
    "label_consolidation": "Consolidation",
    "label_edema": "Edema",
    "label_pleural_effusion": "Pleural Effusion",
    "label_pneumonia": "Pneumonia",
    "label_pneumothorax": "Pneumothorax",
}

DEFAULT_METADATA_CSV = Path("/workspace/mimic_cxr/metadata/mimic-cxr-2.0.0-metadata.csv.gz")
DEFAULT_SPLIT_CSV = Path("/workspace/mimic_cxr/metadata/mimic-cxr-2.0.0-split.csv.gz")
DEFAULT_CHEXPERT_CSV = Path("/workspace/mimic_cxr/metadata/mimic-cxr-2.0.0-chexpert.csv.gz")
DEFAULT_TEST_LABELS_CSV = Path("/workspace/mimic_cxr/metadata/mimic-cxr-2.1.0-test-set-labeled.csv")
DEFAULT_DOWNLOAD_ROOT = Path("/workspace/mimic_cxr/raw")
DEFAULT_MANIFEST_CSV = Path("/workspace/manifest/manifest_mimic_target.csv")
DEFAULT_SUMMARY_JSON = Path("/workspace/manifest/manifest_mimic_target.summary.json")
DEFAULT_PLAN_TSV = Path("/workspace/mimic_cxr/download_plan_target.tsv")
DEFAULT_SELECTED_URLS = Path("/workspace/mimic_cxr/selected_urls_target.txt")
PHYSIONET_BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a MIMIC-CXR-JPG subset manifest from official PhysioNet tables.")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--chexpert-csv", type=Path, default=DEFAULT_CHEXPERT_CSV)
    parser.add_argument("--test-labels-csv", type=Path, default=DEFAULT_TEST_LABELS_CSV)
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--download-plan-tsv", type=Path, default=DEFAULT_PLAN_TSV)
    parser.add_argument("--selected-urls-txt", type=Path, default=DEFAULT_SELECTED_URLS)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def normalize_float_label(raw_value: str) -> str:
    value = (raw_value or "").strip()
    if not value:
        return "0"
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def prefixed_subject_id(raw_value: str) -> str:
    value = str(raw_value).strip()
    return value if value.startswith("p") else f"p{value}"


def prefixed_study_id(raw_value: str) -> str:
    value = str(raw_value).strip()
    return value if value.startswith("s") else f"s{value}"


def load_gzip_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_plain_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def label_signature(row: dict[str, str]) -> str:
    return "|".join(str(row.get(column) or "0").strip() for column in LABEL_COLUMNS)


def allocate_group_counts(group_sizes: dict[str, int], target_count: int) -> dict[str, int]:
    total_count = sum(group_sizes.values())
    if target_count < 0 or target_count >= total_count:
        return dict(group_sizes)
    if target_count == 0:
        return {key: 0 for key in group_sizes}

    allocated = {key: 0 for key in group_sizes}
    remainders: list[tuple[float, str]] = []

    for key, size in group_sizes.items():
        exact = (target_count * size) / total_count
        base = min(size, int(math.floor(exact)))
        allocated[key] = base
        remainders.append((exact - base, key))

    remaining = target_count - sum(allocated.values())

    if remaining > 0:
        for _, key in sorted(remainders, reverse=True):
            if remaining <= 0:
                break
            if group_sizes[key] > 0 and allocated[key] == 0:
                allocated[key] = 1
                remaining -= 1

    if remaining > 0:
        for _, key in sorted(remainders, reverse=True):
            if remaining <= 0:
                break
            room = group_sizes[key] - allocated[key]
            if room <= 0:
                continue
            take = min(room, remaining)
            allocated[key] += take
            remaining -= take

    return allocated


def sample_grouped_rows(rows: list[dict[str, str]], target_count: int, *, seed: int) -> list[dict[str, str]]:
    if target_count < 0 or target_count >= len(rows):
        return list(rows)
    if target_count == 0:
        return []

    rng = random.Random(seed)
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[label_signature(row)].append((index, row))

    allocations = allocate_group_counts({key: len(items) for key, items in grouped.items()}, target_count)
    selected: list[tuple[int, dict[str, str]]] = []
    for key, items in grouped.items():
        take = allocations[key]
        if take <= 0:
            continue
        local_items = list(items)
        rng.shuffle(local_items)
        selected.extend(local_items[:take])

    selected.sort(key=lambda item: item[0])
    return [row for _, row in selected]


def view_group_from_position(view_position: str) -> str:
    view = (view_position or "").strip().upper()
    if view in {"PA", "AP"}:
        return "FRONTAL"
    if "LAT" in view or view in {"LL", "RL"}:
        return "LATERAL"
    return "UNKNOWN"


def rank_metadata_row(row: dict[str, str]) -> tuple[int, str]:
    view = (row.get("ViewPosition") or "").strip().upper()
    if view in {"PA", "AP"}:
        priority = 0
    elif "LAT" in view or view in {"LL", "RL"}:
        priority = 1
    else:
        priority = 2
    return priority, row["dicom_id"]


def build_label_lookup(
    *,
    chexpert_rows: list[dict[str, str]],
    test_label_rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, str]]]:
    train_validate_labels: dict[tuple[str, str], dict[str, str]] = {}
    for row in chexpert_rows:
        key = (row["subject_id"].strip(), row["study_id"].strip())
        train_validate_labels[key] = {
            column: normalize_float_label(row.get(source_name) or "")
            for column, source_name in LABEL_NAME_MAP.items()
        }

    test_labels: dict[str, dict[str, str]] = {}
    for row in test_label_rows:
        study_id = row["study_id"].strip()
        test_labels[study_id] = {
            column: normalize_float_label(row.get(source_name) or "")
            for column, source_name in LABEL_NAME_MAP.items()
        }
    return train_validate_labels, test_labels


def build_candidate_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    metadata_rows = load_gzip_rows(args.metadata_csv)
    split_rows = load_gzip_rows(args.split_csv)
    chexpert_rows = load_gzip_rows(args.chexpert_csv)
    test_label_rows = load_plain_rows(args.test_labels_csv)
    train_validate_labels, test_labels = build_label_lookup(
        chexpert_rows=chexpert_rows,
        test_label_rows=test_label_rows,
    )

    split_lookup = {
        (row["subject_id"].strip(), row["study_id"].strip(), row["dicom_id"].strip()): row["split"].strip()
        for row in split_rows
    }

    by_study: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in metadata_rows:
        by_study[(row["subject_id"].strip(), row["study_id"].strip())].append(row)

    candidates: list[dict[str, str]] = []
    for (subject_id_raw, study_id_raw), rows in by_study.items():
        chosen = sorted(rows, key=rank_metadata_row)[0]
        dicom_id = chosen["dicom_id"].strip()
        source_split = split_lookup[(subject_id_raw, study_id_raw, dicom_id)]

        if source_split == "train":
            labels = train_validate_labels.get((subject_id_raw, study_id_raw))
        elif source_split == "validate":
            labels = train_validate_labels.get((subject_id_raw, study_id_raw))
        elif source_split == "test":
            labels = test_labels.get(study_id_raw)
        else:
            raise SystemExit(f"Unexpected split: {source_split}")

        if labels is None:
            continue

        patient_id = prefixed_subject_id(subject_id_raw)
        study_id = prefixed_study_id(study_id_raw)
        subject_prefix = patient_id[:3]
        relative_physionet_path = f"files/{subject_prefix}/{patient_id}/{study_id}/{dicom_id}.jpg"
        split_name = "val" if source_split == "validate" else source_split
        image_path = str(Path("mimic_cxr/raw") / split_name / subject_prefix / patient_id / study_id / f"{dicom_id}.jpg")

        candidate = {
            "domain": "d2_mimic",
            "dataset": "mimic_cxr",
            "split": split_name,
            "source_split": source_split,
            "row_id": f"mimic_cxr__{patient_id}__{study_id}__{dicom_id}",
            "image_path": image_path,
            "patient_id": patient_id,
            "study_id": study_id,
            "view_raw": (chosen.get("ViewPosition") or "NA").strip() or "NA",
            "view_group": view_group_from_position(chosen.get("ViewPosition") or ""),
            "sex": "NA",
            "age": "NA",
            "dicom_id": dicom_id,
            "physionet_relative_path": relative_physionet_path,
            "download_url": PHYSIONET_BASE_URL + relative_physionet_path,
            "local_download_path": str(args.download_root / split_name / subject_prefix / patient_id / study_id / f"{dicom_id}.jpg"),
        }
        candidate.update(labels)
        candidates.append(candidate)

    return candidates


def positive_count(rows: list[dict[str, str]], column: str) -> int:
    return sum(1 for row in rows if str(row.get(column) or "0").strip() == "1")


def split_report(source_rows: list[dict[str, str]], selected_rows: list[dict[str, str]], requested: int) -> dict[str, object]:
    return {
        "available_rows": len(source_rows),
        "selected_rows": len(selected_rows),
        "requested_rows": requested,
        "selection_mode": "all" if requested < 0 or requested >= len(source_rows) else "sampled",
        "label_positive_counts": {
            column: {
                "available": positive_count(source_rows, column),
                "selected": positive_count(selected_rows, column),
            }
            for column in LABEL_COLUMNS
        },
        "label_signature_counts_selected": dict(Counter(label_signature(row) for row in selected_rows)),
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "domain",
        "dataset",
        "split",
        "source_split",
        "row_id",
        "image_path",
        "patient_id",
        "study_id",
        "view_raw",
        "view_group",
        "sex",
        "age",
        *LABEL_COLUMNS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_download_plan(tsv_path: Path, urls_txt_path: Path, rows: list[dict[str, str]]) -> None:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["url", "local_path", "split", "study_id", "row_id"])
        for row in rows:
            writer.writerow(
                [
                    row["download_url"],
                    row["local_download_path"],
                    row["split"],
                    row["study_id"],
                    row["row_id"],
                ]
            )
    urls_txt_path.parent.mkdir(parents=True, exist_ok=True)
    urls_txt_path.write_text("\n".join(row["download_url"] for row in rows) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    candidate_rows = build_candidate_rows(args)

    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        by_split[row["source_split"]].append(row)

    selected_train = sample_grouped_rows(by_split["train"], args.train_count, seed=args.seed)
    selected_val = sample_grouped_rows(by_split["validate"], args.val_count, seed=args.seed + 1)
    selected_test = sample_grouped_rows(by_split["test"], args.test_count, seed=args.seed + 2)

    for row in selected_train:
        row["split"] = "train"
    for row in selected_val:
        row["split"] = "val"
    for row in selected_test:
        row["split"] = "test"

    selected_rows = selected_train + selected_val + selected_test
    write_manifest(args.manifest_csv, selected_rows)
    write_download_plan(args.download_plan_tsv, args.selected_urls_txt, selected_rows)

    summary = {
        "manifest_csv": str(args.manifest_csv),
        "download_plan_tsv": str(args.download_plan_tsv),
        "selected_urls_txt": str(args.selected_urls_txt),
        "download_root": str(args.download_root),
        "seed": args.seed,
        "requested_counts": {
            "train": args.train_count,
            "val": args.val_count,
            "test": args.test_count,
        },
        "selected_counts": dict(Counter(row["split"] for row in selected_rows)),
        "source_split_counts": {
            "train": split_report(by_split["train"], selected_train, args.train_count),
            "validate": split_report(by_split["validate"], selected_val, args.val_count),
            "test": split_report(by_split["test"], selected_test, args.test_count),
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] manifest={args.manifest_csv}")
    print(f"[done] rows={len(selected_rows)} split_counts={dict(Counter(row['split'] for row in selected_rows))}")
    print(f"[done] plan={args.download_plan_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
