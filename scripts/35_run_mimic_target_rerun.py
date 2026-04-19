#!/usr/bin/env python3
"""Run the current reproducible MIMIC target-only rerun suite."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import experiment_layout


DEFAULT_WORKSPACE = Path("/workspace")
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments/by_id")
DEFAULT_MANIFEST_DIR = Path("/workspace/manifest")
DEFAULT_MIMIC_ROOT = Path("/workspace/mimic_cxr")
DEFAULT_METADATA_CSV = DEFAULT_MIMIC_ROOT / "metadata/mimic-cxr-2.0.0-metadata.csv.gz"
DEFAULT_SPLIT_CSV = DEFAULT_MIMIC_ROOT / "metadata/mimic-cxr-2.0.0-split.csv.gz"
DEFAULT_CHEXPERT_CSV = DEFAULT_MIMIC_ROOT / "metadata/mimic-cxr-2.0.0-chexpert.csv.gz"
DEFAULT_TEST_LABELS_CSV = DEFAULT_MIMIC_ROOT / "metadata/mimic-cxr-2.1.0-test-set-labeled.csv"
DEFAULT_DOWNLOAD_ROOT = DEFAULT_MIMIC_ROOT / "raw"
DEFAULT_MODEL_DIR = Path("/workspace/.cache/cxr_foundation")

BUILD_MANIFEST_SCRIPT = Path("/workspace/scripts/32_build_mimic_jpg_subset.py")
EXPORT_SCRIPT = Path("/workspace/scripts/14_generate_cxr_foundation_embeddings.py")
TRAIN_HEAD_SCRIPT = Path("/workspace/scripts/15_train_domain_transfer_linear_probe.py")
PARTIAL_FINETUNE_SCRIPT = Path("/workspace/scripts/29_train_image_only_partial_finetune.py")


def format_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def slug_from_counts(train_count: int, val_count: int, test_count: int) -> str:
    return f"mimic_target_t{train_count}_v{val_count}_te{test_count}"


def default_output_path(base_dir: Path, stem: str, suffix: str) -> Path:
    return base_dir / f"{stem}{suffix}"


def reserve_experiment_names(
    *,
    experiments_root: Path,
    include_export: bool,
    include_linear: bool,
    include_mlp256: bool,
    include_mlp512: bool,
    include_partial_finetune: bool,
    slug_prefix: str,
    embedding_batch_size: int,
) -> dict[str, str]:
    next_number = experiment_layout.next_experiment_number(experiments_root)
    reserved: dict[str, str] = {}

    def allocate(key: str, slug: str) -> None:
        nonlocal next_number
        reserved[key] = f"exp{next_number:04d}__{slug}"
        next_number += 1

    if include_export:
        allocate(
            "export",
            f"cxr_foundation_embedding_export__{slug_prefix}__general_avg_batch{embedding_batch_size}",
        )
    if include_linear:
        allocate("linear", f"domain_transfer_head_training__{slug_prefix}__cxr_foundation_linear")
    if include_mlp256:
        allocate("mlp256", f"domain_transfer_head_training__{slug_prefix}__cxr_foundation_mlp256")
    if include_mlp512:
        allocate("mlp512", f"domain_transfer_head_training__{slug_prefix}__cxr_foundation_mlp512")
    if include_partial_finetune:
        allocate(
            "partial_finetune",
            f"domain_transfer_partial_finetune_training__{slug_prefix}__vit_b16_lastblock",
        )
    return reserved


def run_command(argv: list[str], *, dry_run: bool) -> None:
    print(format_command(argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fresh official MIMIC target manifest, export CXR Foundation embeddings, "
            "and rerun the current target-only head suite."
        )
    )
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)

    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--chexpert-csv", type=Path, default=DEFAULT_CHEXPERT_CSV)
    parser.add_argument("--test-labels-csv", type=Path, default=DEFAULT_TEST_LABELS_CSV)
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)

    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument(
        "--test-count",
        type=int,
        default=676,
        help="Official labeled MIMIC test rows currently cap at 676 with the bundled builder inputs.",
    )
    parser.add_argument("--subset-seed", type=int, default=13)
    parser.add_argument("--train-seed", type=int, default=1337)

    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--download-plan-tsv", type=Path, default=None)
    parser.add_argument("--selected-urls-txt", type=Path, default=None)

    parser.add_argument("--embedding-root", type=Path, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--token-pooling", choices=("avg", "cls", "flatten", "none"), default="avg")
    parser.add_argument("--hf-token-env-var", type=str, default="HF_TOKEN")

    parser.add_argument("--head-batch-size", type=int, default=512)
    parser.add_argument("--head-num-workers", type=int, default=0)
    parser.add_argument("--head-epochs", type=int, default=50)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--head-weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fp16-on-cuda", dest="fp16_on_cuda", action="store_true")
    parser.add_argument("--no-fp16-on-cuda", dest="fp16_on_cuda", action="store_false")
    parser.set_defaults(fp16_on_cuda=True)

    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--skip-embedding-export", action="store_true")
    parser.add_argument("--skip-linear", action="store_true")
    parser.add_argument("--skip-mlp256", action="store_true")
    parser.add_argument("--skip-mlp512", action="store_true")
    parser.add_argument("--run-partial-finetune", action="store_true")

    parser.add_argument("--partial-batch-size", type=int, default=8)
    parser.add_argument("--partial-num-workers", type=int, default=4)
    parser.add_argument("--partial-epochs", type=int, default=10)
    parser.add_argument("--partial-lr", type=float, default=5e-5)
    parser.add_argument("--partial-weight-decay", type=float, default=1e-4)
    parser.add_argument("--partial-patience", type=int, default=3)
    parser.add_argument("--partial-trainable-blocks", type=int, default=1)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.train_count <= 0 or args.val_count <= 0 or args.test_count <= 0:
        raise SystemExit("train/val/test counts must be positive.")

    experiments_root = args.experiments_root.resolve()
    manifest_dir = DEFAULT_MANIFEST_DIR.resolve()
    mimic_root = DEFAULT_MIMIC_ROOT.resolve()
    slug_prefix = slug_from_counts(args.train_count, args.val_count, args.test_count)
    manifest_stem = f"manifest_{slug_prefix}"

    manifest_csv = (args.manifest_csv or default_output_path(manifest_dir, manifest_stem, ".csv")).resolve()
    summary_json = (
        args.summary_json or default_output_path(manifest_dir, manifest_stem, ".summary.json")
    ).resolve()
    download_plan_tsv = (
        args.download_plan_tsv or default_output_path(mimic_root, f"download_plan_{slug_prefix}", ".tsv")
    ).resolve()
    selected_urls_txt = (
        args.selected_urls_txt or default_output_path(mimic_root, f"selected_urls_{slug_prefix}", ".txt")
    ).resolve()

    include_export = not args.skip_embedding_export
    include_linear = not args.skip_linear
    include_mlp256 = not args.skip_mlp256
    include_mlp512 = not args.skip_mlp512
    include_partial_finetune = bool(args.run_partial_finetune)

    if args.skip_embedding_export and args.embedding_root is None:
        raise SystemExit("--skip-embedding-export requires --embedding-root.")
    if args.skip_manifest and not manifest_csv.exists():
        raise SystemExit(f"--skip-manifest requested, but manifest does not exist: {manifest_csv}")

    reserved_names = reserve_experiment_names(
        experiments_root=experiments_root,
        include_export=include_export,
        include_linear=include_linear,
        include_mlp256=include_mlp256,
        include_mlp512=include_mlp512,
        include_partial_finetune=include_partial_finetune,
        slug_prefix=slug_prefix,
        embedding_batch_size=int(args.embedding_batch_size),
    )

    if include_export:
        embedding_root = experiments_root / reserved_names["export"]
    else:
        embedding_root = args.embedding_root.resolve()

    commands: list[list[str]] = []

    if not args.skip_manifest:
        commands.append(
            [
                sys.executable,
                str(BUILD_MANIFEST_SCRIPT),
                "--metadata-csv",
                str(args.metadata_csv.resolve()),
                "--split-csv",
                str(args.split_csv.resolve()),
                "--chexpert-csv",
                str(args.chexpert_csv.resolve()),
                "--test-labels-csv",
                str(args.test_labels_csv.resolve()),
                "--download-root",
                str(args.download_root.resolve()),
                "--manifest-csv",
                str(manifest_csv),
                "--summary-json",
                str(summary_json),
                "--download-plan-tsv",
                str(download_plan_tsv),
                "--selected-urls-txt",
                str(selected_urls_txt),
                "--train-count",
                str(args.train_count),
                "--val-count",
                str(args.val_count),
                "--test-count",
                str(args.test_count),
                "--seed",
                str(args.subset_seed),
            ]
        )

    if include_export:
        export_cmd = [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--manifest-csv",
            str(manifest_csv),
            "--data-root",
            str(args.data_root.resolve()),
            "--experiments-root",
            str(experiments_root),
            "--model-dir",
            str(args.model_dir.resolve()),
            "--experiment-name",
            reserved_names["export"],
            "--batch-size",
            str(args.embedding_batch_size),
            "--token-pooling",
            args.token_pooling,
            "--hf-token-env-var",
            str(args.hf_token_env_var),
        ]
        if args.overwrite:
            export_cmd.append("--overwrite")
        commands.append(export_cmd)

    common_head_args = [
        "--embedding-root",
        str(embedding_root),
        "--manifest-csv",
        str(manifest_csv),
        "--experiments-root",
        str(experiments_root),
        "--split-profile",
        "mimic_target",
        "--embedding-layout",
        "domain_split",
        "--token-pooling",
        args.token_pooling if args.token_pooling in {"avg", "cls", "flatten"} else "avg",
        "--batch-size",
        str(args.head_batch_size),
        "--num-workers",
        str(args.head_num_workers),
        "--epochs",
        str(args.head_epochs),
        "--lr",
        str(args.head_lr),
        "--weight-decay",
        str(args.head_weight_decay),
        "--patience",
        str(args.head_patience),
        "--seed",
        str(args.train_seed),
        "--device",
        str(args.device),
    ]
    if args.fp16_on_cuda:
        common_head_args.append("--fp16-on-cuda")
    if args.overwrite:
        common_head_args.append("--overwrite")

    if include_linear:
        commands.append(
            [
                sys.executable,
                str(TRAIN_HEAD_SCRIPT),
                "--experiment-name",
                reserved_names["linear"],
                "--head-type",
                "linear",
                *common_head_args,
            ]
        )

    if include_mlp256:
        commands.append(
            [
                sys.executable,
                str(TRAIN_HEAD_SCRIPT),
                "--experiment-name",
                reserved_names["mlp256"],
                "--head-type",
                "mlp",
                "--mlp-hidden-dims",
                "256",
                *common_head_args,
            ]
        )

    if include_mlp512:
        commands.append(
            [
                sys.executable,
                str(TRAIN_HEAD_SCRIPT),
                "--experiment-name",
                reserved_names["mlp512"],
                "--head-type",
                "mlp",
                "--mlp-hidden-dims",
                "512",
                *common_head_args,
            ]
        )

    if include_partial_finetune:
        partial_cmd = [
            sys.executable,
            str(PARTIAL_FINETUNE_SCRIPT),
            "--manifest-csv",
            str(manifest_csv),
            "--data-root",
            str(args.data_root.resolve()),
            "--experiments-root",
            str(experiments_root),
            "--experiment-name",
            reserved_names["partial_finetune"],
            "--split-profile",
            "mimic_target",
            "--batch-size",
            str(args.partial_batch_size),
            "--num-workers",
            str(args.partial_num_workers),
            "--epochs",
            str(args.partial_epochs),
            "--lr",
            str(args.partial_lr),
            "--weight-decay",
            str(args.partial_weight_decay),
            "--patience",
            str(args.partial_patience),
            "--seed",
            str(args.train_seed),
            "--device",
            str(args.device),
            "--trainable-blocks",
            str(args.partial_trainable_blocks),
        ]
        if args.fp16_on_cuda:
            partial_cmd.append("--fp16-on-cuda")
        if args.overwrite:
            partial_cmd.append("--overwrite")
        commands.append(partial_cmd)

    print(f"[plan] manifest_csv={manifest_csv}", flush=True)
    print(f"[plan] embedding_root={embedding_root}", flush=True)
    for key in ("export", "linear", "mlp256", "mlp512", "partial_finetune"):
        if key in reserved_names:
            print(f"[plan] {key}_experiment={experiments_root / reserved_names[key]}", flush=True)

    for command in commands:
        run_command(command, dry_run=bool(args.dry_run))

    print("[done] MIMIC rerun suite finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
