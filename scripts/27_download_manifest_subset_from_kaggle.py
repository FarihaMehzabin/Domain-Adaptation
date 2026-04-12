#!/usr/bin/env python3
"""Download only manifest-listed files from Kaggle-backed datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_common_labels_pilot5h.csv")
DEFAULT_WORKSPACE_ROOT = Path("/workspace")
DEFAULT_KAGGLE_CONFIG = Path("/workspace/kaggle.json")
DEFAULT_NIH_DATASET = "nih-chest-xrays/data"
DEFAULT_CHEXPERT_DATASET = "ashery/chexpert"
DEFAULT_MIMIC_DATASET = "itsanmol124/mimic-cxr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download only the manifest-listed subset files from Kaggle.")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--kaggle-config", type=Path, default=DEFAULT_KAGGLE_CONFIG)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nih_cxr14", "chexpert"],
        choices=["nih_cxr14", "chexpert", "mimic_cxr"],
        help="Datasets to fetch from the manifest.",
    )
    parser.add_argument("--nih-dataset-ref", type=str, default=DEFAULT_NIH_DATASET)
    parser.add_argument("--chexpert-dataset-ref", type=str, default=DEFAULT_CHEXPERT_DATASET)
    parser.add_argument(
        "--mimic-image-dataset-ref",
        type=str,
        default=DEFAULT_MIMIC_DATASET,
        help=(
            "Kaggle dataset ref for MIMIC images. Defaults to the manifest-matching layout "
            f"({DEFAULT_MIMIC_DATASET})."
        ),
    )
    parser.add_argument(
        "--mimic-match-mode",
        choices=["exact", "basename"],
        default="exact",
        help="How to map manifest MIMIC paths to Kaggle files when a MIMIC dataset ref is provided.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser.parse_args()


def ensure_kaggle_env(kaggle_config: Path) -> None:
    if not kaggle_config.exists():
        raise SystemExit(f"kaggle.json not found: {kaggle_config}")
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_config.parent)


def authenticate_api() -> KaggleApi:
    api = KaggleApi()
    api.authenticate()
    return api


def list_dataset_files(api: KaggleApi, dataset_ref: str) -> list[str]:
    names: list[str] = []
    page_token: str | None = None
    while True:
        response = api.dataset_list_files(dataset_ref, page_token=page_token, page_size=500)
        names.extend(file.name for file in response.files)
        page_token = response.next_page_token
        if not page_token:
            break
    return names


def read_manifest_rows(manifest_csv: Path) -> list[dict[str, str]]:
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV not found: {manifest_csv}")
    text = manifest_csv.read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise SystemExit(f"Manifest CSV has no header: {manifest_csv}")
    return [{key: (value or "") for key, value in row.items()} for row in reader]


def build_requested_files(rows: list[dict[str, str]], datasets: set[str]) -> dict[str, set[str]]:
    requested: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        dataset = (row.get("dataset") or "").strip()
        image_path = (row.get("image_path") or "").strip()
        if not image_path:
            continue
        if dataset == "nih_cxr14" and "nih_cxr14" in datasets:
            prefix = "nih_cxr14/raw/"
            if not image_path.startswith(prefix):
                raise SystemExit(f"Unexpected NIH image path: {image_path}")
            requested["nih_cxr14"].add(image_path.removeprefix(prefix))
        elif dataset == "chexpert" and "chexpert" in datasets:
            prefix = "chexpert_small/raw/"
            if not image_path.startswith(prefix):
                raise SystemExit(f"Unexpected CheXpert image path: {image_path}")
            requested["chexpert"].add(image_path.removeprefix(prefix))
        elif dataset == "mimic_cxr" and "mimic_cxr" in datasets:
            prefix = "mimic_cxr/raw/"
            if not image_path.startswith(prefix):
                raise SystemExit(f"Unexpected MIMIC image path: {image_path}")
            requested["mimic_cxr"].add(image_path.removeprefix(prefix))
    if "nih_cxr14" in datasets:
        requested["nih_cxr14"].add("Data_Entry_2017.csv")
    if "chexpert" in datasets:
        requested["chexpert"].add("valid.csv")
    return requested


def resolve_remote_targets(
    *,
    requested_files: set[str],
    available_files: list[str],
    match_mode: str,
) -> tuple[dict[str, str], list[str]]:
    matched: dict[str, str] = {}
    unresolved: list[str] = []
    available_set = set(available_files)
    by_basename: dict[str, list[str]] = defaultdict(list)
    for name in available_files:
        by_basename[Path(name).name].append(name)

    for requested in sorted(requested_files):
        if requested in available_set:
            matched[requested] = requested
            continue
        if match_mode == "basename":
            basename_matches = by_basename.get(Path(requested).name, [])
            if len(basename_matches) == 1:
                matched[requested] = basename_matches[0]
                continue
        unresolved.append(requested)
    return matched, unresolved


def download_one_file(
    *,
    api: KaggleApi,
    dataset_ref: str,
    remote_name: str,
    destination: Path,
    force: bool,
    dry_run: bool,
) -> str:
    if destination.exists() and not force:
        return "skipped_existing"
    if dry_run:
        return "planned"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kaggle_subset_") as tmp_dir:
        api.dataset_download_file(dataset_ref, remote_name, path=tmp_dir, force=True, quiet=True)
        downloaded = Path(tmp_dir) / Path(remote_name).name
        if not downloaded.exists():
            raise SystemExit(f"Kaggle download did not produce expected file: {downloaded}")
        shutil.move(str(downloaded), str(destination))
    return "downloaded"


def main() -> int:
    args = parse_args()
    ensure_kaggle_env(args.kaggle_config.resolve())
    api = authenticate_api()

    manifest_rows = read_manifest_rows(args.manifest_csv.resolve())
    dataset_names = set(args.datasets)
    requested_files = build_requested_files(manifest_rows, dataset_names)

    dataset_configs: dict[str, dict[str, Any]] = {
        "nih_cxr14": {
            "dataset_ref": args.nih_dataset_ref,
            "local_prefix": "nih_cxr14/raw",
            "match_mode": "exact",
        },
        "chexpert": {
            "dataset_ref": args.chexpert_dataset_ref,
            "local_prefix": "chexpert_small/raw",
            "match_mode": "exact",
        },
    }
    if "mimic_cxr" in dataset_names:
        if not args.mimic_image_dataset_ref.strip():
            raise SystemExit(
                "A MIMIC image dataset ref is required for --datasets mimic_cxr."
            )
        dataset_configs["mimic_cxr"] = {
            "dataset_ref": args.mimic_image_dataset_ref.strip(),
            "local_prefix": "mimic_cxr/raw",
            "match_mode": args.mimic_match_mode,
        }

    summary: dict[str, Any] = {
        "manifest_csv": str(args.manifest_csv.resolve()),
        "workspace_root": str(args.workspace_root.resolve()),
        "dry_run": bool(args.dry_run),
        "datasets": {},
    }

    workspace_root = args.workspace_root.resolve()
    for dataset_name, config in dataset_configs.items():
        requested = requested_files.get(dataset_name, set())
        if not requested:
            continue
        available = list_dataset_files(api, str(config["dataset_ref"]))
        matched, unresolved = resolve_remote_targets(
            requested_files=requested,
            available_files=available,
            match_mode=str(config["match_mode"]),
        )

        actions: list[dict[str, str]] = []
        for local_relative, remote_name in sorted(matched.items()):
            destination = workspace_root / str(config["local_prefix"]) / local_relative
            status = download_one_file(
                api=api,
                dataset_ref=str(config["dataset_ref"]),
                remote_name=remote_name,
                destination=destination,
                force=bool(args.force),
                dry_run=bool(args.dry_run),
            )
            actions.append(
                {
                    "remote_name": remote_name,
                    "local_relative": str(Path(config["local_prefix"]) / local_relative),
                    "status": status,
                }
            )

        summary["datasets"][dataset_name] = {
            "dataset_ref": str(config["dataset_ref"]),
            "requested_count": len(requested),
            "matched_count": len(matched),
            "unresolved_count": len(unresolved),
            "unresolved_examples": unresolved[:20],
            "actions": actions,
        }

    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_path is not None:
        args.summary_path.resolve().write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
