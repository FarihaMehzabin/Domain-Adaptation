#!/usr/bin/env python3
"""Build a common 7-label manifest for NIH, CheXpert, and MIMIC-CXR transfer runs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


COMMON_LABELS = (
    ("atelectasis", "Atelectasis", "Atelectasis"),
    ("cardiomegaly", "Cardiomegaly", "Cardiomegaly"),
    ("consolidation", "Consolidation", "Consolidation"),
    ("edema", "Edema", "Edema"),
    ("pleural_effusion", "Pleural Effusion", "Pleural Effusion"),
    ("pneumonia", "Pneumonia", "Pneumonia"),
    ("pneumothorax", "Pneumothorax", "Pneumothorax"),
)

DOMAIN_MAP = {
    "nih_cxr14": "d0_nih",
    "chexpert": "d1_chexpert",
    "mimic_cxr": "d2_mimic",
}

OUTPUT_FIELDS = [
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
]
OUTPUT_FIELDS.extend([f"label_{label_name}" for label_name, _, _ in COMMON_LABELS])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a common 7-label manifest across NIH CXR14, CheXpert, and MIMIC-CXR "
            "for frozen image-embedding transfer experiments."
        )
    )
    parser.add_argument(
        "--nih-manifest",
        type=Path,
        default=Path("/workspace/manifest_nih_cxr14_all14.csv"),
    )
    parser.add_argument(
        "--chexpert-train-csv",
        type=Path,
        default=Path("/workspace/data/chexpert_small/raw/train.csv"),
    )
    parser.add_argument(
        "--chexpert-valid-csv",
        type=Path,
        default=Path("/workspace/data/chexpert_small/raw/valid.csv"),
    )
    parser.add_argument(
        "--mimic-labeled-csv",
        type=Path,
        default=Path("/workspace/data/mimic_cxr/raw/mimic-cxr.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/workspace/manifest_common_labels_nih_train_val_test_chexpert_mimic.csv"),
    )
    return parser.parse_args()


def require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")


def normalize_binary_label(raw_value: str) -> str:
    value = float((raw_value or "0").strip())
    return "1" if value > 0.0 else "0"


def build_nih_rows(manifest_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = (row.get("dataset") or "").strip()
            if dataset != "nih_cxr14":
                continue
            split = (row.get("split") or "").strip().lower()
            if split not in {"train", "val", "test"}:
                continue
            image_path = (row.get("image_path") or "").strip()
            if not image_path:
                continue
            row_id = Path(image_path).stem
            output_row = {
                "domain": DOMAIN_MAP["nih_cxr14"],
                "dataset": "nih_cxr14",
                "split": split,
                "source_split": split,
                "row_id": row_id,
                "image_path": image_path,
                "patient_id": (row.get("patient_id") or "NA").strip() or "NA",
                "study_id": (row.get("study_id") or "NA").strip() or "NA",
                "view_raw": (row.get("view_raw") or "NA").strip() or "NA",
                "view_group": (row.get("view_group") or "UNKNOWN").strip() or "UNKNOWN",
                "sex": (row.get("sex") or "NA").strip() or "NA",
                "age": (row.get("age") or "NA").strip() or "NA",
            }
            for label_name, _, _ in COMMON_LABELS:
                output_row[f"label_{label_name}"] = normalize_binary_label(
                    row.get(f"label_{label_name}") or "0"
                )
            rows.append(output_row)
    return rows


def chexpert_view_fields(row: dict[str, str]) -> tuple[str, str]:
    frontal_lateral = (row.get("Frontal/Lateral") or "").strip()
    ap_pa = (row.get("AP/PA") or "").strip()
    view_raw = ap_pa or frontal_lateral or "NA"
    view_group = frontal_lateral.upper() if frontal_lateral else "UNKNOWN"
    return view_raw, view_group


def normalize_chexpert_image_path(source_image_path: str) -> Path:
    source_path = Path(source_image_path)
    if source_path.parts and source_path.parts[0] == "CheXpert-v1.0-small":
        source_path = Path(*source_path.parts[1:])
    return Path("chexpert_small/raw") / source_path


def build_chexpert_rows(
    csv_path: Path,
    *,
    split: str,
    source_split: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_image_path = (row.get("Path") or "").strip()
            if not source_image_path:
                continue
            image_path = str(normalize_chexpert_image_path(source_image_path))
            image_parts = Path(source_image_path).parts
            patient_id = image_parts[-3] if len(image_parts) >= 3 else "NA"
            study_id = image_parts[-2] if len(image_parts) >= 2 else "NA"
            image_stem = Path(source_image_path).stem
            row_id = f"chexpert__{patient_id}__{study_id}__{image_stem}"
            view_raw, view_group = chexpert_view_fields(row)
            output_row = {
                "domain": DOMAIN_MAP["chexpert"],
                "dataset": "chexpert",
                "split": split,
                "source_split": source_split,
                "row_id": row_id,
                "image_path": image_path,
                "patient_id": patient_id,
                "study_id": study_id,
                "view_raw": view_raw,
                "view_group": view_group,
                "sex": (row.get("Sex") or "NA").strip() or "NA",
                "age": (row.get("Age") or "NA").strip() or "NA",
            }
            for label_name, chexpert_label, _ in COMMON_LABELS:
                output_row[f"label_{label_name}"] = normalize_binary_label(
                    row.get(chexpert_label) or "0"
                )
            rows.append(output_row)
    return rows


def build_mimic_rows(labeled_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with labeled_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_split = (row.get("split") or "").strip().lower()
            if source_split not in {"valid", "test"}:
                continue
            split = "val" if source_split == "valid" else "test"
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            image_path = str(Path("mimic_cxr/raw") / source_split / filename)
            image_stem = Path(filename).stem
            output_row = {
                "domain": DOMAIN_MAP["mimic_cxr"],
                "dataset": "mimic_cxr",
                "split": split,
                "source_split": source_split,
                "row_id": f"mimic_cxr__{image_stem}",
                "image_path": image_path,
                "patient_id": "NA",
                "study_id": image_stem,
                "view_raw": "NA",
                "view_group": "UNKNOWN",
                "sex": "NA",
                "age": "NA",
            }
            for label_name, _, mimic_label in COMMON_LABELS:
                output_row[f"label_{label_name}"] = normalize_binary_label(
                    row.get(mimic_label) or "0"
                )
            rows.append(output_row)
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    require_exists(args.nih_manifest, "NIH manifest")
    require_exists(args.chexpert_train_csv, "CheXpert train CSV")
    require_exists(args.chexpert_valid_csv, "CheXpert valid CSV")
    require_exists(args.mimic_labeled_csv, "Labeled MIMIC CSV")

    rows: list[dict[str, str]] = []
    rows.extend(build_nih_rows(args.nih_manifest))
    rows.extend(
        build_chexpert_rows(
            args.chexpert_train_csv,
            split="train",
            source_split="train",
        )
    )
    rows.extend(
        build_chexpert_rows(
            args.chexpert_valid_csv,
            split="val",
            source_split="valid",
        )
    )
    rows.extend(build_mimic_rows(args.mimic_labeled_csv))
    write_manifest(args.output_csv, rows)

    counts = Counter((row["domain"], row["dataset"], row["split"]) for row in rows)
    print(f"[done] wrote {len(rows)} rows to {args.output_csv}")
    for domain, dataset, split in sorted(counts):
        print(
            f"[count] domain={domain} dataset={dataset} split={split} rows={counts[(domain, dataset, split)]}"
        )


if __name__ == "__main__":
    main()
