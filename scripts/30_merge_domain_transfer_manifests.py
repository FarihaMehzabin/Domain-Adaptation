#!/usr/bin/env python3
"""Merge multiple domain-transfer manifests that share the same schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple manifest CSV files into one combined manifest."
    )
    parser.add_argument("--input-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--prefer-last-domain-split",
        nargs=2,
        action="append",
        metavar=("DOMAIN", "SPLIT"),
        default=None,
        help=(
            "For the listed domain/split pairs, rows from later input manifests replace rows from earlier ones."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    return list(reader.fieldnames), rows


def main() -> int:
    args = parse_args()
    output_csv = args.output_csv.resolve()
    if output_csv.exists() and not args.overwrite:
        raise SystemExit(f"Output CSV already exists: {output_csv}")

    expected_fields: list[str] | None = None
    merged_rows: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    prefer_last_domain_splits = {
        (str(domain).strip(), str(split).strip())
        for domain, split in (args.prefer_last_domain_split or [])
    }

    for input_csv in args.input_csv:
        fields, rows = read_csv_rows(input_csv.resolve())
        if expected_fields is None:
            expected_fields = fields
        elif fields != expected_fields:
            raise SystemExit(
                f"Manifest schema mismatch for {input_csv}.\n"
                f"expected={expected_fields}\n"
                f"found={fields}"
            )
        present_domain_splits = {
            ((row.get("domain") or "").strip(), (row.get("split") or "").strip())
            for row in rows
        }
        replace_now = present_domain_splits & prefer_last_domain_splits
        if replace_now:
            merged_rows = [
                row
                for row in merged_rows
                if ((row.get("domain") or "").strip(), (row.get("split") or "").strip()) not in replace_now
            ]
            seen_keys = {
                key for key in seen_keys if (key[0], key[1]) not in replace_now
            }
        for row in rows:
            key = (
                (row.get("domain") or "").strip(),
                (row.get("split") or "").strip(),
                (row.get("row_id") or "").strip(),
            )
            if key in seen_keys:
                raise SystemExit(f"Duplicate domain/split/row_id encountered while merging manifests: {key}")
            seen_keys.add(key)
            merged_rows.append(row)

    if expected_fields is None:
        raise SystemExit("No input manifests were provided.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=expected_fields)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"[done] wrote {len(merged_rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
