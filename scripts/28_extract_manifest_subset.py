#!/usr/bin/env python3
"""Extract only manifest-listed image files from a local directory or zip archive."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from pathlib import Path


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest/manifest_mimic_target_1000.csv")
DEFAULT_OUTPUT_ROOT = Path("./subset_export")
DEFAULT_DATASET = "mimic_cxr"
DEFAULT_IMAGE_PREFIX = "mimic_cxr/raw/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy or extract only the image files referenced by a manifest or plain path list. "
            "The output preserves the workspace layout expected by the training scripts."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help=(
            "Local source directory or zip archive. For the MIMIC Kaggle mirror, this should "
            "contain paths like train/s50009377.jpg and test/s50010747.jpg."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination root. Files are written under <output-root>/mimic_cxr/raw/...",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Manifest CSV with image_path entries. Defaults to none if --path-list is provided.",
    )
    parser.add_argument(
        "--path-list",
        type=Path,
        default=None,
        help=(
            "Optional plain-text list of relative image paths such as train/s50009377.jpg. "
            "If provided, this is used instead of --manifest-csv."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset name filter when reading a manifest CSV.",
    )
    parser.add_argument(
        "--image-prefix",
        type=str,
        default=DEFAULT_IMAGE_PREFIX,
        help="Manifest image_path prefix to strip when building relative paths.",
    )
    parser.add_argument(
        "--bundle-zip",
        type=Path,
        default=None,
        help="Optional output zip to create after extraction.",
    )
    parser.add_argument(
        "--recursive-fallback",
        action="store_true",
        help=(
            "If exact relative-path lookup fails for a directory source, fall back to a recursive "
            "basename search. Use this only when the source layout differs from train/test/*.jpg."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()
    if args.path_list is None and args.manifest_csv is None:
        args.manifest_csv = DEFAULT_MANIFEST_CSV
    if args.path_list is not None and args.manifest_csv is not None:
        raise SystemExit("Pass either --manifest-csv or --path-list, not both.")
    return args


def normalize_relative_path(value: str) -> str:
    return Path(value.strip()).as_posix().lstrip("./")


def load_requested_paths_from_manifest(
    manifest_csv: Path,
    *,
    dataset: str,
    image_prefix: str,
) -> list[str]:
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV not found: {manifest_csv}")
    rows: list[str] = []
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest CSV has no header: {manifest_csv}")
        for row in reader:
            current_dataset = (row.get("dataset") or "").strip()
            image_path = (row.get("image_path") or "").strip()
            if current_dataset != dataset or not image_path:
                continue
            if not image_path.startswith(image_prefix):
                raise SystemExit(
                    f"Unexpected image_path for dataset {dataset!r}: {image_path!r} "
                    f"(expected prefix {image_prefix!r})"
                )
            rows.append(normalize_relative_path(image_path.removeprefix(image_prefix)))
    return sorted(set(rows))


def load_requested_paths_from_list(path_list: Path) -> list[str]:
    if not path_list.exists():
        raise SystemExit(f"Path list not found: {path_list}")
    rows = [
        normalize_relative_path(line)
        for line in path_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted(set(rows))


def build_recursive_index(source_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        index.setdefault(path.name, []).append(path)
    return index


def resolve_from_directory(
    *,
    source_dir: Path,
    relative_paths: list[str],
    recursive_fallback: bool,
) -> tuple[dict[str, Path], list[str]]:
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    basename_index: dict[str, list[Path]] | None = None

    for relative in relative_paths:
        exact = source_dir / relative
        if exact.exists():
            resolved[relative] = exact
            continue
        if not recursive_fallback:
            missing.append(relative)
            continue
        if basename_index is None:
            basename_index = build_recursive_index(source_dir)
        matches = basename_index.get(Path(relative).name, [])
        if len(matches) == 1:
            resolved[relative] = matches[0]
            continue
        missing.append(relative)
    return resolved, missing


def resolve_from_zip(
    *,
    source_zip: Path,
    relative_paths: list[str],
) -> tuple[dict[str, str], list[str]]:
    resolved: dict[str, str] = {}
    missing: list[str] = []
    with zipfile.ZipFile(source_zip, "r") as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        exact_map = {normalize_relative_path(name): name for name in names}
        suffix_map: dict[str, list[str]] = {}
        for name in names:
            normalized = normalize_relative_path(name)
            suffix_map.setdefault(Path(normalized).name, []).append(name)

        for relative in relative_paths:
            normalized = normalize_relative_path(relative)
            if normalized in exact_map:
                resolved[relative] = exact_map[normalized]
                continue

            slash_suffix = f"/{normalized}"
            suffix_matches = [name for name in names if normalize_relative_path(name).endswith(slash_suffix)]
            if len(suffix_matches) == 1:
                resolved[relative] = suffix_matches[0]
                continue

            basename_matches = suffix_map.get(Path(normalized).name, [])
            if len(basename_matches) == 1:
                resolved[relative] = basename_matches[0]
                continue

            missing.append(relative)
    return resolved, missing


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_from_directory(
    *,
    resolved_paths: dict[str, Path],
    output_root: Path,
    overwrite: bool,
) -> int:
    copied = 0
    for relative, source_path in sorted(resolved_paths.items()):
        destination = output_root / DEFAULT_IMAGE_PREFIX / relative
        if destination.exists() and not overwrite:
            continue
        ensure_parent(destination)
        shutil.copy2(source_path, destination)
        copied += 1
    return copied


def copy_from_zip(
    *,
    source_zip: Path,
    resolved_paths: dict[str, str],
    output_root: Path,
    overwrite: bool,
) -> int:
    copied = 0
    with zipfile.ZipFile(source_zip, "r") as archive:
        for relative, archive_name in sorted(resolved_paths.items()):
            destination = output_root / DEFAULT_IMAGE_PREFIX / relative
            if destination.exists() and not overwrite:
                continue
            ensure_parent(destination)
            with archive.open(archive_name, "r") as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            copied += 1
    return copied


def write_bundle_zip(bundle_zip: Path, output_root: Path) -> None:
    bundle_zip.parent.mkdir(parents=True, exist_ok=True)
    base_dir = output_root / "mimic_cxr"
    if not base_dir.exists():
        raise SystemExit(f"Nothing to bundle under {base_dir}")
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(base_dir.rglob("*")):
            if not path.is_file():
                continue
            archive.write(path, arcname=path.relative_to(output_root))


def main() -> int:
    args = parse_args()
    if args.path_list is not None:
        relative_paths = load_requested_paths_from_list(args.path_list)
    else:
        relative_paths = load_requested_paths_from_manifest(
            args.manifest_csv,
            dataset=args.dataset,
            image_prefix=args.image_prefix,
        )

    if not relative_paths:
        raise SystemExit("No matching image paths were found.")

    source = args.source
    if not source.exists():
        raise SystemExit(f"Source path not found: {source}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        resolved_paths, missing = resolve_from_directory(
            source_dir=source.resolve(),
            relative_paths=relative_paths,
            recursive_fallback=bool(args.recursive_fallback),
        )
        copied = copy_from_directory(
            resolved_paths=resolved_paths,
            output_root=output_root,
            overwrite=bool(args.overwrite),
        )
    else:
        resolved_paths, missing = resolve_from_zip(
            source_zip=source.resolve(),
            relative_paths=relative_paths,
        )
        copied = copy_from_zip(
            source_zip=source.resolve(),
            resolved_paths=resolved_paths,
            output_root=output_root,
            overwrite=bool(args.overwrite),
        )

    print(f"[done] requested={len(relative_paths)} resolved={len(resolved_paths)} copied={copied}")
    print(f"[done] output_root={output_root}")
    if args.bundle_zip is not None:
        write_bundle_zip(args.bundle_zip.resolve(), output_root)
        print(f"[done] bundle_zip={args.bundle_zip.resolve()}")

    if missing:
        print(f"[missing] count={len(missing)}", file=sys.stderr)
        for path in missing[:50]:
            print(f"[missing] {path}", file=sys.stderr)
        if len(missing) > 50:
            print(f"[missing] ... {len(missing) - 50} more", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
