#!/usr/bin/env python3
"""Create the isolated Policy B common5 workspace without touching legacy outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_namespace import DEFAULT_POLICYB_BASE_DIR  # noqa: E402


WORKSPACE_DIRS = [
    "configs",
    "manifests",
    "checkpoints",
    "outputs",
    "reports",
    "logs",
    "scripts_snapshot",
]

POLICYB_MANIFEST_SOURCES = [
    "manifests/mimic_common5_policyB_train_pool.csv",
    "manifests/mimic_common5_policyB_val.csv",
    "manifests/mimic_common5_policyB_test.csv",
    "manifests/mimic_common5_policyB_support_k5_seed2027.csv",
    "manifests/mimic_common5_policyB_support_k20_seed2027.csv",
    "manifests/nih_dev_2k_train.csv",
    "manifests/nih_dev_2k_val.csv",
    "manifests/nih_dev_2k_test.csv",
]

POLICYB_REPORT_SOURCES = [
    "reports/official_label_policy.md",
    "reports/official_label_policy.json",
    "reports/policyB_manifest_audit.md",
    "reports/policyB_existing_predictions_eval.md",
]

SCRIPT_SNAPSHOT_SOURCES = [
    "scripts/experiment_namespace.py",
    "scripts/prepare_policyB_workspace.py",
    "scripts/evaluate_nih_on_mimic.py",
    "scripts/adapt_head_only_mimic.py",
    "scripts/adapt_full_finetune_mimic.py",
    "scripts/adapt_lastblock_mimic.py",
    "scripts/adapt_lora_mimic.py",
    "scripts/train_nih_2k_baseline.py",
]

EXPERIMENT_DEFAULTS = {
    "experiment_namespace": "policyB_common5_v1",
    "base_dir": "experiments/policyB_common5_v1",
    "label_policy": "uignore_blankzero",
    "label_set": "common5",
    "labels": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"],
    "primary_metric": "macro_auprc",
    "secondary_metric": "macro_auroc",
    "selection_split": "val",
    "test_split": "test",
    "seed": 2027,
    "notes": "All official Policy B reruns must write only inside experiments/policyB_common5_v1.",
}

README_TEXT = """# Policy B Common5 Workspace

This folder contains the official Policy B common5 experiments.

The legacy root-level folders such as `checkpoints/`, `outputs/`, `reports/`, `logs/`, and `manifests/` are provisional or historical and must remain untouched.

All new official checkpoints, outputs, logs, and reports must be written inside this namespace: `experiments/policyB_common5_v1/`.

Adaptation runs from the old policy must be rerun here before they are treated as official Policy B results.

The no-adaptation NIH source model can be reevaluated for Policy B comparison, but final adaptation comparisons require Policy B-trained adapters or Policy B full fine-tunes produced inside this namespace.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the isolated Policy B common5 experiment workspace.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=str(DEFAULT_POLICYB_BASE_DIR),
        help="Workspace directory to create.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files when contents differ.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def file_bytes(path: Path) -> bytes:
    return path.read_bytes()


def write_text_file(
    destination: Path,
    text: str,
    *,
    overwrite: bool,
    created_files: list[str],
    skipped_existing: list[str],
    blocked_overwrites: list[str],
) -> None:
    ensure_dir(destination.parent)
    encoded = text.encode("utf-8")
    if destination.exists():
        if destination.is_dir():
            blocked_overwrites.append(f"Expected file but found directory: {relative_to_root(destination)}")
            print(f"BLOCKED {relative_to_root(destination)}")
            return
        if file_bytes(destination) == encoded:
            skipped_existing.append(relative_to_root(destination))
            print(f"UNCHANGED {relative_to_root(destination)}")
            return
        if not overwrite:
            blocked_overwrites.append(relative_to_root(destination))
            print(f"BLOCKED {relative_to_root(destination)}")
            return
    destination.write_bytes(encoded)
    created_files.append(relative_to_root(destination))
    print(f"WROTE {relative_to_root(destination)}")


def write_json_file(
    destination: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
    created_files: list[str],
    skipped_existing: list[str],
    blocked_overwrites: list[str],
) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    write_text_file(
        destination,
        text,
        overwrite=overwrite,
        created_files=created_files,
        skipped_existing=skipped_existing,
        blocked_overwrites=blocked_overwrites,
    )


def copy_file(
    source: Path,
    destination: Path,
    *,
    overwrite: bool,
    copied_files: list[str],
    missing_sources: list[str],
    skipped_existing: list[str],
    blocked_overwrites: list[str],
) -> None:
    if not source.exists():
        missing_sources.append(relative_to_root(source))
        print(f"MISSING {relative_to_root(source)}")
        return

    ensure_dir(destination.parent)
    source_bytes = source.read_bytes()
    if destination.exists():
        if destination.is_dir():
            blocked_overwrites.append(f"Expected file but found directory: {relative_to_root(destination)}")
            print(f"BLOCKED {relative_to_root(destination)}")
            return
        if file_bytes(destination) == source_bytes:
            skipped_existing.append(relative_to_root(destination))
            print(f"UNCHANGED {relative_to_root(destination)}")
            return
        if not overwrite:
            blocked_overwrites.append(relative_to_root(destination))
            print(f"BLOCKED {relative_to_root(destination)}")
            return

    shutil.copy2(source, destination)
    copied_files.append(relative_to_root(destination))
    print(f"COPIED {relative_to_root(destination)}")


def build_workspace_manifest(
    *,
    base_dir: Path,
    created_dirs: list[str],
    created_files: list[str],
    copied_files: list[str],
    skipped_existing: list[str],
    blocked_overwrites: list[str],
    missing_sources: list[str],
) -> dict[str, Any]:
    workspace_directories = sorted(
        relative_to_root(path)
        for path in base_dir.rglob("*")
        if path.is_dir()
    )
    workspace_files = sorted(
        relative_to_root(path)
        for path in base_dir.rglob("*")
        if path.is_file()
    )
    return {
        "workspace": relative_to_root(base_dir),
        "workspace_directories": workspace_directories,
        "workspace_files": workspace_files,
        "created_directories": sorted(created_dirs),
        "created_files": sorted(created_files),
        "copied_files": sorted(copied_files),
        "skipped_existing": sorted(skipped_existing),
        "blocked_overwrites": sorted(blocked_overwrites),
        "missing_sources": sorted(missing_sources),
    }


def main() -> int:
    args = parse_args()
    base_dir = (ROOT / args.base_dir).resolve() if not Path(args.base_dir).is_absolute() else Path(args.base_dir).resolve()

    created_dirs: list[str] = []
    created_files: list[str] = []
    copied_files: list[str] = []
    skipped_existing: list[str] = []
    blocked_overwrites: list[str] = []
    missing_sources: list[str] = []

    ensure_dir(base_dir)
    created_dirs.append(relative_to_root(base_dir))
    print(f"DIR {relative_to_root(base_dir)}")

    for directory_name in WORKSPACE_DIRS:
        directory_path = base_dir / directory_name
        if not directory_path.exists():
            ensure_dir(directory_path)
            created_dirs.append(relative_to_root(directory_path))
            print(f"DIR {relative_to_root(directory_path)}")
        else:
            print(f"DIR EXISTS {relative_to_root(directory_path)}")

    for source_relative in POLICYB_MANIFEST_SOURCES:
        source_path = ROOT / source_relative
        destination_path = base_dir / "manifests" / source_path.name
        copy_file(
            source_path,
            destination_path,
            overwrite=args.overwrite,
            copied_files=copied_files,
            missing_sources=missing_sources,
            skipped_existing=skipped_existing,
            blocked_overwrites=blocked_overwrites,
        )

    for source_relative in POLICYB_REPORT_SOURCES:
        source_path = ROOT / source_relative
        destination_path = base_dir / "reports" / source_path.name
        copy_file(
            source_path,
            destination_path,
            overwrite=args.overwrite,
            copied_files=copied_files,
            missing_sources=missing_sources,
            skipped_existing=skipped_existing,
            blocked_overwrites=blocked_overwrites,
        )

    for source_relative in SCRIPT_SNAPSHOT_SOURCES:
        source_path = ROOT / source_relative
        destination_path = base_dir / "scripts_snapshot" / source_path.name
        copy_file(
            source_path,
            destination_path,
            overwrite=args.overwrite,
            copied_files=copied_files,
            missing_sources=missing_sources,
            skipped_existing=skipped_existing,
            blocked_overwrites=blocked_overwrites,
        )

    write_json_file(
        base_dir / "configs" / "experiment_defaults.json",
        EXPERIMENT_DEFAULTS,
        overwrite=args.overwrite,
        created_files=created_files,
        skipped_existing=skipped_existing,
        blocked_overwrites=blocked_overwrites,
    )
    write_text_file(
        base_dir / "README.md",
        README_TEXT,
        overwrite=args.overwrite,
        created_files=created_files,
        skipped_existing=skipped_existing,
        blocked_overwrites=blocked_overwrites,
    )

    workspace_manifest = build_workspace_manifest(
        base_dir=base_dir,
        created_dirs=created_dirs,
        created_files=created_files,
        copied_files=copied_files,
        skipped_existing=skipped_existing,
        blocked_overwrites=blocked_overwrites,
        missing_sources=missing_sources,
    )
    write_json_file(
        base_dir / "workspace_manifest.json",
        workspace_manifest,
        overwrite=args.overwrite,
        created_files=created_files,
        skipped_existing=skipped_existing,
        blocked_overwrites=blocked_overwrites,
    )

    print(f"workspace: {relative_to_root(base_dir)}")
    print("copied files:")
    for item in copied_files:
        print(f"- {item}")
    print("missing files:")
    for item in missing_sources:
        print(f"- {item}")

    if missing_sources or blocked_overwrites:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
