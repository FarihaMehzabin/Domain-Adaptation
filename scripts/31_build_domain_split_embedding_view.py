#!/usr/bin/env python3
"""Build a lightweight domain/split embedding-root view from existing roots."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a domain/split embedding root by symlinking explicit source root/domain/split mappings."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label-source-root", type=Path, required=True)
    parser.add_argument(
        "--mapping",
        nargs=5,
        action="append",
        metavar=("OUT_DOMAIN", "OUT_SPLIT", "SRC_ROOT", "SRC_DOMAIN", "SRC_SPLIT"),
        required=True,
        help="Map one output domain/split directory to one source root/domain/split directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    label_source_root = args.label_source_root.resolve()

    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output root already exists: {output_root}")
        if output_root.is_symlink() or output_root.is_file():
            output_root.unlink()
        else:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    label_names_path = label_source_root / "label_names.json"
    if label_names_path.exists():
        shutil.copy2(label_names_path, output_root / "label_names.json")

    mappings_payload: list[dict[str, str]] = []
    seen_outputs: set[tuple[str, str]] = set()
    for out_domain, out_split, src_root_raw, src_domain, src_split in args.mapping:
        source_root = Path(src_root_raw).resolve()
        source_dir = source_root / src_domain / src_split
        if not source_dir.exists():
            raise SystemExit(f"Source split directory not found: {source_dir}")
        output_key = (out_domain, out_split)
        if output_key in seen_outputs:
            raise SystemExit(f"Duplicate output mapping declared: {output_key}")
        seen_outputs.add(output_key)

        destination_dir = output_root / out_domain
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / out_split
        destination_path.symlink_to(source_dir)
        mappings_payload.append(
            {
                "output_domain": out_domain,
                "output_split": out_split,
                "source_root": str(source_root),
                "source_domain": src_domain,
                "source_split": src_split,
                "source_dir": str(source_dir),
            }
        )

    meta_path = output_root / "view_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "label_source_root": str(label_source_root),
                "mappings": mappings_payload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] output_root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
