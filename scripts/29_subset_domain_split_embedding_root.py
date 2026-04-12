#!/usr/bin/env python3
"""Materialize a manifest-aligned subset of a domain/split embedding root."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


ROOT_FILES = ("config.json", "label_names.json")
SPLIT_FILES = ("image_manifest.csv", "image_paths.txt", "row_ids.json", "study_ids.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new embedding root by subsetting one domain's split directories to the exact "
            "row_ids referenced by a manifest CSV."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        help="Split names to materialize from the source root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest_rows(manifest_csv: Path, *, domain: str) -> dict[str, list[dict[str, str]]]:
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest CSV has no header: {manifest_csv}")
        for row in reader:
            if (row.get("domain") or "").strip() != domain:
                continue
            split = (row.get("split") or "").strip()
            rows_by_split.setdefault(split, []).append(dict(row))
    return rows_by_split


def read_json_list(path: Path) -> list[str]:
    return [str(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_root_files(source_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ROOT_FILES:
        source_path = source_root / name
        if source_path.exists():
            shutil.copy2(source_path, output_root / name)


def copy_if_same(source_dir: Path, output_dir: Path, *, row_ids: list[str]) -> bool:
    source_row_ids_path = source_dir / "row_ids.json"
    if not source_row_ids_path.exists():
        return False
    source_row_ids = read_json_list(source_row_ids_path)
    if source_row_ids != row_ids:
        return False
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)
    return True


def subset_text_lines(path: Path, indices: list[int]) -> list[str]:
    return [line for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()) if idx in set(indices)]


def subset_csv_rows(path: Path, indices: list[int]) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {path}")
        rows = list(reader)
        selected = [rows[idx] for idx in indices]
        return list(reader.fieldnames), selected


def write_csv(path: Path, *, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_subset(
    *,
    source_root: Path,
    output_root: Path,
    domain: str,
    split: str,
    manifest_rows: list[dict[str, str]],
) -> None:
    source_dir = source_root / domain / split
    if not source_dir.exists():
        raise SystemExit(f"Source split directory not found: {source_dir}")

    output_dir = output_root / domain / split
    output_dir.mkdir(parents=True, exist_ok=True)

    target_row_ids = [str(row["row_id"]) for row in manifest_rows]
    if copy_if_same(source_dir, output_dir, row_ids=target_row_ids):
        print(f"[copy] {domain}/{split} rows={len(target_row_ids)} mode=full")
        return

    source_row_ids = read_json_list(source_dir / "row_ids.json")
    row_to_index = {row_id: idx for idx, row_id in enumerate(source_row_ids)}
    missing = [row_id for row_id in target_row_ids if row_id not in row_to_index]
    if missing:
        raise SystemExit(
            f"Source root is missing {len(missing)} row_ids for {domain}/{split}. "
            f"Examples: {missing[:10]}"
        )
    indices = [row_to_index[row_id] for row_id in target_row_ids]

    embeddings = np.load(source_dir / "embeddings.npy", mmap_mode="r")
    subset_embeddings = np.asarray(embeddings[indices], dtype=np.float32)
    np.save(output_dir / "embeddings.npy", subset_embeddings)

    for name in ("row_ids.json", "study_ids.json"):
        values = read_json_list(source_dir / name)
        selected = [values[idx] for idx in indices]
        write_json(output_dir / name, selected)

    image_paths = (source_dir / "image_paths.txt").read_text(encoding="utf-8").splitlines()
    selected_paths = [image_paths[idx] for idx in indices]
    (output_dir / "image_paths.txt").write_text("\n".join(selected_paths) + "\n", encoding="utf-8")

    fieldnames, rows = subset_csv_rows(source_dir / "image_manifest.csv", indices)
    write_csv(output_dir / "image_manifest.csv", fieldnames=fieldnames, rows=rows)

    write_json(
        output_dir / "run_meta.json",
        {
            "source_root": str(source_root.resolve()),
            "domain": domain,
            "split": split,
            "rows": len(target_row_ids),
            "status": "subset_from_source_root",
        },
    )
    write_json(
        output_dir / "split_progress.json",
        {
            "domain": domain,
            "split": split,
            "status": "completed_from_subset",
            "total_rows": len(target_row_ids),
            "embedded_rows": len(target_row_ids),
            "completed_batches": None,
            "failed_rows": 0,
            "final_embedding_path": str((output_dir / "embeddings.npy").resolve()),
        },
    )
    print(f"[copy] {domain}/{split} rows={len(target_row_ids)} mode=subset")


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    manifest_rows_by_split = read_manifest_rows(args.manifest_csv.resolve(), domain=str(args.domain))

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output root already exists and is non-empty: {output_root}")
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)

    copy_root_files(source_root, output_root)
    for split in args.splits:
        rows = manifest_rows_by_split.get(split, [])
        if not rows:
            continue
        materialize_subset(
            source_root=source_root,
            output_root=output_root,
            domain=str(args.domain),
            split=str(split),
            manifest_rows=rows,
        )

    print(f"[done] output_root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
