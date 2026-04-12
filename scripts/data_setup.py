

#!/usr/bin/env python3
# Instructions (with meanings):
#   python scripts/setup_data.py --choice 1 --clean-raw
#     - Downloads all datasets and clears each raw/ folder first.
#   python scripts/setup_data.py --choice 2
#     - Downloads only NIH CXR14 into data/nih_cxr14/raw.
#   python scripts/setup_data.py --choice 3
#     - Downloads only PadChest-small into data/padchest_small/raw.
#   python scripts/setup_data.py --choice 4
#     - Downloads only CheXpert-small into data/chexpert_small/raw.
#   python scripts/setup_data.py --datasets nih_cxr14 chexpert_small
#     - Downloads the listed datasets (names are exact).
#   python scripts/setup_data.py --datasets all --dry-run
#     - Prints what would be downloaded without changing files.
# Flags:
#   --clean-raw     Remove existing raw/ contents before download.
#   --force         Download even if ready markers exist.
#   --keep-archives Keep zip/tar archives after extraction.
#   --dry-run       Show actions only; skip download/extraction.
# Notes:
# - Requires Kaggle CLI and kaggle.json credentials.
# - Keeps manifests/splits/docs intact and only manages raw/ subfolders.

"""
Prepare dataset raw folders with on-demand downloads.

Usage examples:
  python scripts/setup_data.py --datasets all --clean-raw
  python scripts/setup_data.py --datasets nih_cxr14 chexpert_small

Notes:
- Requires Kaggle CLI and kaggle.json credentials.
- Keeps manifests/splits/docs intact and only manages raw/ subfolders.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import requests

NIH_PARALLEL_DOWNLOAD_WORKERS = 8
NIH_PARALLEL_EXTRACT_WORKERS = 8
RANGED_DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024

DATASETS = {
    "nih_cxr14": {
        "kaggle_id": "nih-chest-xrays/data",
        "raw_rel": "nih_cxr14/raw",
        "ready_markers": [
            "Data_Entry_2017.csv",
            "images_001",
            "images_012",
        ],
    },
    "chexpert_small": {
        "kaggle_id": "ashery/chexpert",
        "raw_rel": "chexpert_small/raw",
        "ready_markers": [
            "train.csv",
            "valid.csv",
            "train",
            "valid",
        ],
    },
    "padchest_small": {
        "kaggle_id": "seoyunje/padchest-small-dataset",
        "raw_rel": "padchest_small/raw",
        "ready_markers": [
            "PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
            "PC/images-224/images-224",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup raw datasets with Kaggle downloads.")
    parser.add_argument(
        "--choice",
        type=int,
        choices=[1, 2, 3, 4],
        help="Quick selection: 1=all, 2=nih_cxr14, 3=padchest_small, 4=chexpert_small",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to prepare: all, nih_cxr14, chexpert_small, padchest_small",
    )
    parser.add_argument(
        "--data-root",
        default="/workspace/data",
        help="Root data directory.",
    )
    parser.add_argument(
        "--kaggle-config",
        default="/workspace/kaggle.json",
        help="Path to kaggle.json credentials.",
    )
    parser.add_argument(
        "--clean-raw",
        action="store_true",
        help="Remove existing raw/ contents before download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if ready markers exist.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep zip/tar archives after extraction (default deletes).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without downloading or extracting.",
    )
    parser.add_argument(
        "--nih-manifest",
        default="",
        help="Optional path to the NIH manifest CSV used to write train/val/test split files.",
    )
    parser.add_argument(
        "--split-output-dir",
        default="",
        help="Optional output directory for generated split CSVs. Defaults to <data-root>/nih_cxr14/splits.",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def ensure_kaggle_env(kaggle_json: Path) -> dict[str, str]:
    env = os.environ.copy()
    if not kaggle_json.exists():
        raise FileNotFoundError(f"kaggle.json not found at {kaggle_json}")
    kaggle_json.chmod(0o600)
    env["KAGGLE_CONFIG_DIR"] = str(kaggle_json.parent)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_json.parent)
    return env


def kaggle_dataset_archive_name(kaggle_id: str) -> str:
    return f"{kaggle_id.rsplit('/', 1)[1]}.zip"


def kaggle_dataset_download_url(kaggle_id: str) -> tuple[str, str]:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

    api = KaggleApi()
    api.authenticate()
    owner_slug, dataset_slug, dataset_version_number = api.split_dataset_string(kaggle_id)
    with api.build_kaggle_client() as kaggle_client:
        request = ApiDownloadDatasetRequest()
        request.owner_slug = owner_slug
        request.dataset_slug = dataset_slug
        request.dataset_version_number = (
            int(dataset_version_number) if dataset_version_number else None
        )
        response = kaggle_client.datasets.dataset_api_client.download_dataset(request)
    return response.request.url, f"{dataset_slug}.zip"


def ranged_download_size(url: str) -> int:
    response = requests.get(
        url,
        headers={"Range": "bytes=0-0"},
        stream=True,
        timeout=(30, 60),
    )
    try:
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError("Remote server does not support ranged downloads.")
        content_range = response.headers.get("content-range")
        if not content_range or "/" not in content_range:
            raise RuntimeError("Missing content-range header for ranged download.")
        return int(content_range.rsplit("/", 1)[1])
    finally:
        response.close()


def download_range(url: str, destination: Path, start: int, end: int) -> None:
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(3):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Expected partial content for bytes {start}-{end}, got {response.status_code}."
                    )
                with destination.open("r+b") as handle:
                    handle.seek(start)
                    for chunk in response.iter_content(chunk_size=RANGED_DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            handle.write(chunk)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def download_nih_dataset_parallel(kaggle_id: str, raw_dir: Path) -> None:
    url, archive_name = kaggle_dataset_download_url(kaggle_id)
    archive_path = raw_dir / archive_name
    partial_path = raw_dir / f"{archive_name}.partial"
    total_bytes = ranged_download_size(url)
    worker_count = min(NIH_PARALLEL_DOWNLOAD_WORKERS, total_bytes) or 1
    byte_span = (total_bytes + worker_count - 1) // worker_count
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_bytes:
        end = min(total_bytes - 1, start + byte_span - 1)
        ranges.append((start, end))
        start = end + 1

    if archive_path.exists():
        archive_path.unlink()
    if partial_path.exists():
        partial_path.unlink()

    with partial_path.open("wb") as handle:
        handle.truncate(total_bytes)

    print(
        f"[nih_cxr14] downloading {archive_name} with {len(ranges)} parallel ranges "
        f"({total_bytes} bytes)"
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as executor:
            futures = [
                executor.submit(download_range, url, partial_path, start, end)
                for start, end in ranges
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise

    partial_path.replace(archive_path)


def clean_raw_dir(raw_dir: Path) -> None:
    if raw_dir.exists():
        for item in raw_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def extract_zip_member_batch(archive_path: Path, raw_dir: Path, member_names: list[str]) -> None:
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member_name in member_names:
            info = zf.getinfo(member_name)
            target_path = raw_dir / member_name
            if info.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=RANGED_DOWNLOAD_CHUNK_SIZE)


def extract_zip_parallel(archive_path: Path, raw_dir: Path, worker_count: int) -> None:
    with zipfile.ZipFile(archive_path, "r") as zf:
        members_by_group: dict[str, list[str]] = {}
        for info in zf.infolist():
            member_name = info.filename
            stripped_name = member_name.rstrip("/")
            if not stripped_name:
                continue
            group_name = stripped_name.split("/", 1)[0]
            members_by_group.setdefault(group_name, []).append(member_name)

    batches = list(members_by_group.values())
    if not batches:
        return

    print(f"[nih_cxr14] extracting {archive_path.name} with {min(worker_count, len(batches))} workers")
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(worker_count, len(batches))) as executor:
        futures = [
            executor.submit(extract_zip_member_batch, archive_path, raw_dir, batch)
            for batch in batches
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def extract_archives(raw_dir: Path, keep_archives: bool, parallel_zip_workers: int = 1) -> None:
    archives = list(raw_dir.glob("*.zip")) + list(raw_dir.glob("*.tar.gz")) + list(raw_dir.glob("*.tgz"))
    for archive in archives:
        if archive.suffix == ".zip":
            if parallel_zip_workers > 1:
                extract_zip_parallel(archive, raw_dir, parallel_zip_workers)
            else:
                with zipfile.ZipFile(archive, "r") as zf:
                    zf.extractall(raw_dir)
        else:
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(raw_dir)
        if not keep_archives:
            archive.unlink()



def cleanup_archives(raw_dir: Path) -> None:
    for archive in raw_dir.glob("*.zip"):
        archive.unlink()
    for archive in raw_dir.glob("*.tar.gz"):
        archive.unlink()
    for archive in raw_dir.glob("*.tgz"):
        archive.unlink()


def extract_nested_archives(root_dir: Path, keep_archives: bool) -> None:
    nested = list(root_dir.rglob("*.zip")) + list(root_dir.rglob("*.tar.gz")) + list(root_dir.rglob("*.tgz"))
    for archive in nested:
        if archive.parent == root_dir:
            continue
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(archive.parent)
        else:
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(archive.parent)
        if not keep_archives:
            archive.unlink()


def chexpert_postprocess(raw_dir: Path) -> None:
    nested_root = raw_dir / "CheXpert-v1.0-small"
    if nested_root.exists() and nested_root.is_dir():
        for item in nested_root.iterdir():
            target = raw_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        shutil.rmtree(nested_root)


def padchest_postprocess(raw_dir: Path, keep_archives: bool) -> None:
    pc_dir = raw_dir / "PC"
    if pc_dir.exists():
        extract_nested_archives(pc_dir, keep_archives)


def is_ready(raw_dir: Path, markers: list[str]) -> bool:
    for marker in markers:
        if not (raw_dir / marker).exists():
            return False
    return True


def resolve_nih_manifest(manifest_arg: str) -> Path:
    if manifest_arg:
        manifest_path = Path(manifest_arg)
        if not manifest_path.exists():
            raise FileNotFoundError(f"NIH manifest not found at {manifest_path}")
        return manifest_path

    candidates = [
        Path("/workspace/manifest/manifest_nih_cxr14.csv"),
        Path("/workspace/manifest/manifest_nih_cxr14 .csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find NIH manifest CSV. Pass --nih-manifest with the manifest path."
    )


def write_nih_split_files(manifest_path: Path, split_output_dir: Path, dry_run: bool) -> None:
    split_output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest {manifest_path} is missing a header row.")
        if "split" not in reader.fieldnames:
            raise ValueError(f"Manifest {manifest_path} does not contain a 'split' column.")

        fieldnames = reader.fieldnames
        split_names = ["train", "val", "test"]
        split_rows: dict[str, list[dict[str, str]]] = {name: [] for name in split_names}
        unexpected_splits: set[str] = set()

        for row in reader:
            split_name = row["split"].strip().lower()
            if split_name in split_rows:
                split_rows[split_name].append(row)
            else:
                unexpected_splits.add(split_name)

    if unexpected_splits:
        raise ValueError(
            f"Unexpected split values in {manifest_path}: {sorted(unexpected_splits)}"
        )

    if dry_run:
        for split_name in split_names:
            output_path = split_output_dir / f"{split_name}.csv"
            print(f"[nih_cxr14] would write {len(split_rows[split_name])} rows to {output_path}")
        return

    for split_name in split_names:
        output_path = split_output_dir / f"{split_name}.csv"
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows[split_name])
        print(f"[nih_cxr14] wrote {len(split_rows[split_name])} rows to {output_path}")


def download_dataset(
    name: str,
    cfg: dict,
    data_root: Path,
    env: dict[str, str],
    clean_raw: bool,
    force: bool,
    keep_archives: bool,
    dry_run: bool,
) -> None:
    raw_dir = data_root / cfg["raw_rel"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    if clean_raw and not dry_run:
        clean_raw_dir(raw_dir)

    if not force and is_ready(raw_dir, cfg["ready_markers"]):
        print(f"[{name}] ready markers present. Skipping download.")
        return

    if dry_run:
        print(f"[{name}] would download {cfg['kaggle_id']} to {raw_dir}")
        return

    if name == "nih_cxr14":
        archive_path = raw_dir / kaggle_dataset_archive_name(cfg["kaggle_id"])
        if archive_path.exists() and not force:
            print(f"[{name}] using existing archive {archive_path}")
        else:
            download_nih_dataset_parallel(cfg["kaggle_id"], raw_dir)
    else:
        run_cmd(["kaggle", "datasets", "download", "-d", cfg["kaggle_id"], "-p", str(raw_dir)], env=env)

    extract_archives(
        raw_dir,
        keep_archives,
        parallel_zip_workers=NIH_PARALLEL_EXTRACT_WORKERS if name == "nih_cxr14" else 1,
    )

    if not keep_archives:
        cleanup_archives(raw_dir)

    if name == "chexpert_small":
        chexpert_postprocess(raw_dir)
    elif name == "padchest_small":
        padchest_postprocess(raw_dir, keep_archives)

    if not is_ready(raw_dir, cfg["ready_markers"]):
        raise RuntimeError(f"[{name}] expected files not found after setup.")

    print(f"[{name}] ready.")


def normalize_datasets(requested: list[str], choice: int | None) -> list[str]:
    if choice is not None:
        mapping = {
            1: list(DATASETS.keys()),
            2: ["nih_cxr14"],
            3: ["padchest_small"],
            4: ["chexpert_small"],
        }
        return mapping[choice]
    if "all" in requested:
        return list(DATASETS.keys())
    unknown = [d for d in requested if d not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return requested


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    kaggle_json = Path(args.kaggle_config)

    datasets = normalize_datasets(args.datasets, args.choice)
    env = ensure_kaggle_env(kaggle_json)

    for name in datasets:
        cfg = DATASETS[name]
        download_dataset(
            name=name,
            cfg=cfg,
            data_root=data_root,
            env=env,
            clean_raw=args.clean_raw,
            force=args.force,
            keep_archives=args.keep_archives,
            dry_run=args.dry_run,
        )
        if name == "nih_cxr14":
            manifest_path = resolve_nih_manifest(args.nih_manifest)
            split_output_dir = (
                Path(args.split_output_dir)
                if args.split_output_dir
                else data_root / "nih_cxr14" / "splits"
            )
            write_nih_split_files(
                manifest_path=manifest_path,
                split_output_dir=split_output_dir,
                dry_run=args.dry_run,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
