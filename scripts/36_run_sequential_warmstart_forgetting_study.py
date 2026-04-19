#!/usr/bin/env python3
"""Run the NIH -> CheXpert -> MIMIC warm-start forgetting study."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import experiment_layout


DEFAULT_OUTPUT_ROOT = Path("/tmp/cxr_sequential_forgetting_study")
DEFAULT_PILOT_MANIFEST = Path("/workspace/manifest/manifest_common_labels_pilot5h.csv")
DEFAULT_BINARY_MIMIC_MANIFEST = Path("/tmp/cxr_mimic_run/manifest/manifest_mimic_target_1000_binary.csv")
DEFAULT_NIH_EMBEDDING_ROOT = Path(
    "/workspace/experiments/by_id/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128"
)
DEFAULT_CHEXPERT_EMBEDDING_ROOT = Path(
    "/workspace/experiments/by_id/exp0054__cxr_foundation_embedding_export__chexpert_target_1000_cxr_foundation_avg_batch128"
)
DEFAULT_MIMIC_EMBEDDING_ROOT = Path(
    "/tmp/cxr_mimic_run/experiments/by_id/exp0001__cxr_foundation_embedding_export__mimic_target_1000_cxr_foundation_avg_batch128"
)
DEFAULT_MERGE_SCRIPT = Path("/workspace/scripts/30_merge_domain_transfer_manifests.py")
DEFAULT_VIEW_SCRIPT = Path("/workspace/scripts/31_build_domain_split_embedding_view.py")
DEFAULT_TRAINER_SCRIPT = Path("/workspace/scripts/15_train_domain_transfer_linear_probe.py")
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5
DEFAULT_SEED = 1337
DEFAULT_DEVICE = "auto"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(argv: list[str]) -> None:
    print(format_command(argv), flush=True)
    subprocess.run(argv, check=True)


def reserve_experiment_names(*, experiments_root: Path) -> dict[str, str]:
    next_number = experiment_layout.next_experiment_number(experiments_root)
    reserved: dict[str, str] = {}
    for key, slug in (
        (
            "nih_source",
            "domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear",
        ),
        (
            "chexpert_adapt",
            "domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_warmstart_chain__cxr_foundation_linear",
        ),
        (
            "mimic_adapt",
            "domain_transfer_head_training__mimic_adapt_from_chexpert__pilot5h_warmstart_chain__cxr_foundation_linear",
        ),
    ):
        reserved[key] = f"exp{next_number:04d}__{slug}"
        next_number += 1
    return reserved


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def macro_auroc(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = load_json(path)
    return payload.get("macro", {}).get("auroc")


def load_stage_summary(stage_dir: Path, *, stage_name: str) -> dict[str, Any]:
    config = load_json(stage_dir / "config.json")
    meta = load_json(stage_dir / "experiment_meta.json")

    if stage_name == "nih_source":
        domain_paths = {
            "nih": stage_dir / "d0_test_metrics.json",
            "chexpert": stage_dir / "d1_test_metrics.json",
            "mimic": stage_dir / "d2_test_metrics.json",
        }
    elif stage_name == "chexpert_adapt":
        domain_paths = {
            "nih": stage_dir / "d0_test_metrics.json",
            "chexpert": stage_dir / "target_test_metrics.json",
            "mimic": None,
        }
    elif stage_name == "mimic_adapt":
        domain_paths = {
            "nih": stage_dir / "d0_test_metrics.json",
            "chexpert": stage_dir / "d1_test_metrics.json",
            "mimic": stage_dir / "target_test_metrics.json",
        }
    else:
        raise ValueError(f"Unsupported stage_name={stage_name}")

    metrics = {
        domain: (macro_auroc(path) if path is not None else None) for domain, path in domain_paths.items()
    }
    return {
        "stage_name": stage_name,
        "experiment_dir": str(stage_dir),
        "checkpoint_path": str(stage_dir / "best.ckpt"),
        "best_epoch": meta.get("best_epoch"),
        "split_profile": config.get("split_profile"),
        "metrics": metrics,
    }


def delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return float(after) - float(before)


def build_summary(*, manifest_csv: Path, embedding_view_root: Path, stage_runs: dict[str, Path]) -> dict[str, Any]:
    stage_summaries = {
        name: load_stage_summary(path, stage_name=name) for name, path in stage_runs.items()
    }

    nih_source = stage_summaries["nih_source"]["metrics"]
    chexpert_adapt = stage_summaries["chexpert_adapt"]["metrics"]
    mimic_adapt = stage_summaries["mimic_adapt"]["metrics"]

    forgetting = {
        "nih_after_chexpert_delta": delta(chexpert_adapt["nih"], nih_source["nih"]),
        "nih_after_mimic_delta_vs_nih_source": delta(mimic_adapt["nih"], nih_source["nih"]),
        "nih_after_mimic_delta_vs_chexpert_stage": delta(mimic_adapt["nih"], chexpert_adapt["nih"]),
        "chexpert_after_mimic_delta_vs_chexpert_stage": delta(mimic_adapt["chexpert"], chexpert_adapt["chexpert"]),
    }
    transfer = {
        "nih_to_chexpert_zero_shot": nih_source["chexpert"],
        "nih_to_mimic_zero_shot": nih_source["mimic"],
        "chexpert_finetune_gain_vs_nih_zero_shot": delta(chexpert_adapt["chexpert"], nih_source["chexpert"]),
        "mimic_finetune_gain_vs_nih_zero_shot": delta(mimic_adapt["mimic"], nih_source["mimic"]),
    }

    return {
        "run_date_utc": utc_now_iso(),
        "study_type": "sequential_warmstart_forgetting",
        "manifest_csv": str(manifest_csv),
        "embedding_view_root": str(embedding_view_root),
        "stages": stage_summaries,
        "forgetting": forgetting,
        "transfer": transfer,
    }


def write_summary_csv(path: Path, *, summary: dict[str, Any]) -> None:
    rows = [
        ("nih_source", "nih", summary["stages"]["nih_source"]["metrics"]["nih"]),
        ("nih_source", "chexpert", summary["stages"]["nih_source"]["metrics"]["chexpert"]),
        ("nih_source", "mimic", summary["stages"]["nih_source"]["metrics"]["mimic"]),
        ("chexpert_adapt", "nih", summary["stages"]["chexpert_adapt"]["metrics"]["nih"]),
        ("chexpert_adapt", "chexpert", summary["stages"]["chexpert_adapt"]["metrics"]["chexpert"]),
        ("mimic_adapt", "nih", summary["stages"]["mimic_adapt"]["metrics"]["nih"]),
        ("mimic_adapt", "chexpert", summary["stages"]["mimic_adapt"]["metrics"]["chexpert"]),
        ("mimic_adapt", "mimic", summary["stages"]["mimic_adapt"]["metrics"]["mimic"]),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stage", "domain", "macro_auroc"))
        writer.writerows(rows)


def write_summary_report(path: Path, *, summary: dict[str, Any]) -> None:
    stages = summary["stages"]
    forgetting = summary["forgetting"]
    transfer = summary["transfer"]
    lines = [
        "# Sequential Warm-Start Forgetting Study",
        "",
        f"- Manifest: `{summary['manifest_csv']}`",
        f"- Embedding view: `{summary['embedding_view_root']}`",
        "",
        "## Stage AUROC",
        "",
        f"- NIH source: NIH `{stages['nih_source']['metrics']['nih']}`, CheXpert `{stages['nih_source']['metrics']['chexpert']}`, MIMIC `{stages['nih_source']['metrics']['mimic']}`",
        f"- After CheXpert train: NIH `{stages['chexpert_adapt']['metrics']['nih']}`, CheXpert `{stages['chexpert_adapt']['metrics']['chexpert']}`",
        f"- After MIMIC train: NIH `{stages['mimic_adapt']['metrics']['nih']}`, CheXpert `{stages['mimic_adapt']['metrics']['chexpert']}`, MIMIC `{stages['mimic_adapt']['metrics']['mimic']}`",
        "",
        "## Forgetting",
        "",
        f"- NIH after CheXpert delta: `{forgetting['nih_after_chexpert_delta']}`",
        f"- NIH after MIMIC delta vs NIH source: `{forgetting['nih_after_mimic_delta_vs_nih_source']}`",
        f"- NIH after MIMIC delta vs CheXpert stage: `{forgetting['nih_after_mimic_delta_vs_chexpert_stage']}`",
        f"- CheXpert after MIMIC delta vs CheXpert stage: `{forgetting['chexpert_after_mimic_delta_vs_chexpert_stage']}`",
        "",
        "## Transfer",
        "",
        f"- NIH to CheXpert zero-shot: `{transfer['nih_to_chexpert_zero_shot']}`",
        f"- NIH to MIMIC zero-shot: `{transfer['nih_to_mimic_zero_shot']}`",
        f"- CheXpert finetune gain vs NIH zero-shot: `{transfer['chexpert_finetune_gain_vs_nih_zero_shot']}`",
        f"- MIMIC finetune gain vs NIH zero-shot: `{transfer['mimic_finetune_gain_vs_nih_zero_shot']}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compose the existing NIH, CheXpert, and MIMIC CXR Foundation exports into a single "
            "pilot5h warm-start study and run NIH -> CheXpert -> MIMIC linear-head training."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pilot-manifest-csv", type=Path, default=DEFAULT_PILOT_MANIFEST)
    parser.add_argument("--binary-mimic-manifest-csv", type=Path, default=DEFAULT_BINARY_MIMIC_MANIFEST)
    parser.add_argument("--nih-embedding-root", type=Path, default=DEFAULT_NIH_EMBEDDING_ROOT)
    parser.add_argument("--chexpert-embedding-root", type=Path, default=DEFAULT_CHEXPERT_EMBEDDING_ROOT)
    parser.add_argument("--mimic-embedding-root", type=Path, default=DEFAULT_MIMIC_EMBEDDING_ROOT)
    parser.add_argument("--merge-script", type=Path, default=DEFAULT_MERGE_SCRIPT)
    parser.add_argument("--view-script", type=Path, default=DEFAULT_VIEW_SCRIPT)
    parser.add_argument("--trainer-script", type=Path, default=DEFAULT_TRAINER_SCRIPT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_root = args.output_root.resolve()
    manifest_root = output_root / "manifest"
    view_root = output_root / "embedding_views" / "pilot5h_nih_chexpert_mimic_cxr_foundation"
    experiments_root = output_root / "experiments" / "by_id"
    summary_json = output_root / "study_summary.json"
    summary_csv = output_root / "study_summary.csv"
    summary_report = output_root / "study_report.md"
    run_config_path = output_root / "run_config.json"
    combined_manifest = manifest_root / "manifest_pilot5h_binary_mimic.csv"

    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output root already exists: {output_root}")
        shutil.rmtree(output_root)

    experiments_root.mkdir(parents=True, exist_ok=True)
    reserved_names = reserve_experiment_names(experiments_root=experiments_root)

    write_json(
        run_config_path,
        {
            "run_date_utc": utc_now_iso(),
            "output_root": str(output_root),
            "pilot_manifest_csv": str(args.pilot_manifest_csv.resolve()),
            "binary_mimic_manifest_csv": str(args.binary_mimic_manifest_csv.resolve()),
            "nih_embedding_root": str(args.nih_embedding_root.resolve()),
            "chexpert_embedding_root": str(args.chexpert_embedding_root.resolve()),
            "mimic_embedding_root": str(args.mimic_embedding_root.resolve()),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "seed": int(args.seed),
            "device": str(args.device),
            "reserved_names": reserved_names,
        },
    )

    run_command(
        [
            sys.executable,
            str(args.merge_script.resolve()),
            "--input-csv",
            str(args.pilot_manifest_csv.resolve()),
            str(args.binary_mimic_manifest_csv.resolve()),
            "--output-csv",
            str(combined_manifest),
            "--prefer-last-domain-split",
            "d2_mimic",
            "train",
            "--prefer-last-domain-split",
            "d2_mimic",
            "val",
            "--prefer-last-domain-split",
            "d2_mimic",
            "test",
            "--overwrite",
        ]
    )

    run_command(
        [
            sys.executable,
            str(args.view_script.resolve()),
            "--output-root",
            str(view_root),
            "--label-source-root",
            str(args.nih_embedding_root.resolve()),
            "--mapping",
            "d0_nih",
            "train",
            str(args.nih_embedding_root.resolve()),
            "d0_nih",
            "train",
            "--mapping",
            "d0_nih",
            "val",
            str(args.nih_embedding_root.resolve()),
            "d0_nih",
            "val",
            "--mapping",
            "d0_nih",
            "test",
            str(args.nih_embedding_root.resolve()),
            "d0_nih",
            "test",
            "--mapping",
            "d1_chexpert",
            "train",
            str(args.chexpert_embedding_root.resolve()),
            "d1_chexpert",
            "train",
            "--mapping",
            "d1_chexpert",
            "val",
            str(args.chexpert_embedding_root.resolve()),
            "d1_chexpert",
            "val",
            "--mapping",
            "d1_chexpert",
            "test",
            str(args.chexpert_embedding_root.resolve()),
            "d1_chexpert",
            "test",
            "--mapping",
            "d2_mimic",
            "train",
            str(args.mimic_embedding_root.resolve()),
            "d2_mimic",
            "train",
            "--mapping",
            "d2_mimic",
            "val",
            str(args.mimic_embedding_root.resolve()),
            "d2_mimic",
            "val",
            "--mapping",
            "d2_mimic",
            "test",
            str(args.mimic_embedding_root.resolve()),
            "d2_mimic",
            "test",
            "--overwrite",
        ]
    )

    common_train_args = [
        "--embedding-root",
        str(view_root),
        "--manifest-csv",
        str(combined_manifest),
        "--experiments-root",
        str(experiments_root),
        "--embedding-layout",
        "domain_split",
        "--token-pooling",
        "avg",
        "--head-type",
        "linear",
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--overwrite",
    ]

    nih_source_dir = experiments_root / reserved_names["nih_source"]
    chexpert_adapt_dir = experiments_root / reserved_names["chexpert_adapt"]
    mimic_adapt_dir = experiments_root / reserved_names["mimic_adapt"]

    run_command(
        [
            sys.executable,
            str(args.trainer_script.resolve()),
            "--experiment-name",
            reserved_names["nih_source"],
            "--split-profile",
            "nih_source_all_test",
            *common_train_args,
        ]
    )

    run_command(
        [
            sys.executable,
            str(args.trainer_script.resolve()),
            "--experiment-name",
            reserved_names["chexpert_adapt"],
            "--split-profile",
            "chexpert_adapt_from_nih",
            "--init-checkpoint",
            str(nih_source_dir / "best.ckpt"),
            *common_train_args,
        ]
    )

    run_command(
        [
            sys.executable,
            str(args.trainer_script.resolve()),
            "--experiment-name",
            reserved_names["mimic_adapt"],
            "--split-profile",
            "mimic_adapt_from_chexpert",
            "--init-checkpoint",
            str(chexpert_adapt_dir / "best.ckpt"),
            *common_train_args,
        ]
    )

    summary = build_summary(
        manifest_csv=combined_manifest,
        embedding_view_root=view_root,
        stage_runs={
            "nih_source": nih_source_dir,
            "chexpert_adapt": chexpert_adapt_dir,
            "mimic_adapt": mimic_adapt_dir,
        },
    )
    write_json(summary_json, summary)
    write_summary_csv(summary_csv, summary=summary)
    write_summary_report(summary_report, summary=summary)

    print(f"[done] summary_json={summary_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
