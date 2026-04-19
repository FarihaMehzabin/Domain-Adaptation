#!/usr/bin/env python3
"""Run rehearsal-free sequential LwF and MAS sweeps on the pilot5h chain."""

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


DEFAULT_OUTPUT_ROOT = Path("/tmp/cxr_sequential_regularizer_sweeps")
DEFAULT_BASELINE_CAMPAIGN_ROOT = Path("/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h")
DEFAULT_EMBEDDING_VIEW_ROOT = Path(
    "/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation"
)
DEFAULT_MANIFEST_CSV = Path(
    "/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/manifest/manifest_pilot5h_binary_mimic.csv"
)
DEFAULT_TRAINER_SCRIPT = Path("/workspace/scripts/15_train_domain_transfer_linear_probe.py")
DEFAULT_MAS_SCRIPT = Path("/workspace/scripts/38_compute_online_mas_state.py")
DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_WORKERS = 0
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5
DEFAULT_SEED = 1337
DEFAULT_DEVICE = "auto"
DEFAULT_LWF_ALPHAS = (0.25, 0.5, 1.0)
DEFAULT_LWF_TEMPERATURES = (2.0, 4.0, 8.0)
DEFAULT_MAS_LAMBDAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
REPORT_PARAMS_TEXT = "5.4k"
REPORT_FLOPS_TEXT = "10.8k"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def run_command(argv: list[str]) -> None:
    print(format_command(argv), flush=True)
    subprocess.run(argv, check=True)


def format_float_slug(value: float) -> str:
    return str(value).replace(".", "p")


def format_metric(value: float | None) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def find_single_directory(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one match for pattern '{pattern}' in {root}, found {len(matches)}.")
    return matches[0]


def macro_auroc(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = load_json(path)
    value = payload.get("macro", {}).get("auroc")
    return float(value) if value is not None else None


def domain_metrics_for_profile(stage_dir: Path, split_profile: str) -> dict[str, dict[str, float | None]]:
    if split_profile == "nih_source_all_test":
        val_paths = {
            "nih": stage_dir / "d0_val_metrics.json",
            "chexpert": stage_dir / "d1_val_metrics.json",
            "mimic": stage_dir / "d2_val_metrics.json",
        }
        test_paths = {
            "nih": stage_dir / "d0_test_metrics.json",
            "chexpert": stage_dir / "d1_test_metrics.json",
            "mimic": stage_dir / "d2_test_metrics.json",
        }
    elif split_profile == "chexpert_adapt_from_nih":
        val_paths = {
            "nih": stage_dir / "d0_val_metrics.json",
            "chexpert": stage_dir / "target_val_metrics.json",
            "mimic": stage_dir / "d2_val_metrics.json",
        }
        test_paths = {
            "nih": stage_dir / "d0_test_metrics.json",
            "chexpert": stage_dir / "target_test_metrics.json",
            "mimic": stage_dir / "d2_test_metrics.json",
        }
    elif split_profile == "mimic_adapt_from_chexpert":
        val_paths = {
            "nih": stage_dir / "d0_val_metrics.json",
            "chexpert": stage_dir / "d1_val_metrics.json",
            "mimic": stage_dir / "target_val_metrics.json",
        }
        test_paths = {
            "nih": stage_dir / "d0_test_metrics.json",
            "chexpert": stage_dir / "d1_test_metrics.json",
            "mimic": stage_dir / "target_test_metrics.json",
        }
    elif split_profile == "chexpert_target":
        val_paths = {
            "nih": None,
            "chexpert": stage_dir / "target_val_metrics.json",
            "mimic": None,
        }
        test_paths = {
            "nih": None,
            "chexpert": stage_dir / "target_test_metrics.json",
            "mimic": None,
        }
    elif split_profile == "mimic_target":
        val_paths = {
            "nih": None,
            "chexpert": None,
            "mimic": stage_dir / "target_val_metrics.json",
        }
        test_paths = {
            "nih": None,
            "chexpert": None,
            "mimic": stage_dir / "target_test_metrics.json",
        }
    else:
        raise SystemExit(f"Unsupported split_profile for metric extraction: {split_profile}")

    return {
        "val": {domain: (macro_auroc(path) if path is not None else None) for domain, path in val_paths.items()},
        "test": {domain: (macro_auroc(path) if path is not None else None) for domain, path in test_paths.items()},
    }


def load_stage_summary(stage_dir: Path) -> dict[str, Any]:
    config = load_json(stage_dir / "config.json")
    meta = load_json(stage_dir / "experiment_meta.json")
    split_profile = str(config["split_profile"])
    metrics = domain_metrics_for_profile(stage_dir, split_profile)
    return {
        "experiment_dir": str(stage_dir),
        "checkpoint_path": str(stage_dir / "best.ckpt"),
        "split_profile": split_profile,
        "preservation_method": config.get("preservation_method", "none"),
        "best_epoch": meta.get("best_epoch"),
        "val": metrics["val"],
        "test": metrics["test"],
    }


def flatten_stage_metrics(prefix: str, stage_summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        f"{prefix}_experiment_dir": stage_summary["experiment_dir"],
        f"{prefix}_best_epoch": stage_summary["best_epoch"],
    }
    for split_kind in ("val", "test"):
        for domain in ("nih", "chexpert", "mimic"):
            row[f"{prefix}_{split_kind}_{domain}"] = stage_summary[split_kind].get(domain)
    return row


def dominance_value(value: float | None) -> float:
    return float(value) if value is not None else float("-inf")


def dominates(lhs: dict[str, Any], rhs: dict[str, Any], *, split_kind: str) -> bool:
    lhs_values = [dominance_value(lhs["stage_c"][split_kind].get(domain)) for domain in ("nih", "chexpert", "mimic")]
    rhs_values = [dominance_value(rhs["stage_c"][split_kind].get(domain)) for domain in ("nih", "chexpert", "mimic")]
    return all(left >= right for left, right in zip(lhs_values, rhs_values)) and any(
        left > right for left, right in zip(lhs_values, rhs_values)
    )


def annotate_pareto(records: list[dict[str, Any]]) -> None:
    for split_kind in ("val", "test"):
        flag_name = f"stage_c_{split_kind}_pareto"
        for index, record in enumerate(records):
            dominated = any(
                other_index != index and dominates(other_record, record, split_kind=split_kind)
                for other_index, other_record in enumerate(records)
            )
            record.setdefault("pareto", {})[flag_name] = not dominated


def reserve_name_factory(experiments_root: Path):
    next_number = experiment_layout.next_experiment_number(experiments_root)

    def reserve(slug: str) -> str:
        nonlocal next_number
        name = f"exp{next_number:04d}__{slug}"
        next_number += 1
        return name

    return reserve


def baseline_experiment_dirs(campaign_root: Path) -> dict[str, Path]:
    experiments_root = campaign_root / "experiments" / "by_id"
    return {
        "nih_source": find_single_directory(
            experiments_root,
            "*__domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear",
        ),
        "chexpert_adapt": find_single_directory(
            experiments_root,
            "*__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_warmstart_chain__cxr_foundation_linear",
        ),
        "mimic_adapt": find_single_directory(
            experiments_root,
            "*__domain_transfer_head_training__mimic_adapt_from_chexpert__pilot5h_warmstart_chain__cxr_foundation_linear",
        ),
        "chexpert_oracle": find_single_directory(
            experiments_root,
            "*__domain_transfer_head_training__chexpert_oracle__pilot5h_warmstart_chain__cxr_foundation_linear",
        ),
        "mimic_oracle": find_single_directory(
            experiments_root,
            "*__domain_transfer_head_training__mimic_oracle__pilot5h_warmstart_chain__cxr_foundation_linear",
        ),
    }


def build_baseline_summary(campaign_root: Path) -> dict[str, Any]:
    dirs = baseline_experiment_dirs(campaign_root)
    nih_source = load_stage_summary(dirs["nih_source"])
    chexpert_adapt = load_stage_summary(dirs["chexpert_adapt"])
    mimic_adapt = load_stage_summary(dirs["mimic_adapt"])
    chexpert_oracle = load_stage_summary(dirs["chexpert_oracle"])
    mimic_oracle = load_stage_summary(dirs["mimic_oracle"])
    return {
        "oracle": {
            "nih": nih_source["test"]["nih"],
            "chexpert": chexpert_oracle["test"]["chexpert"],
            "mimic": mimic_oracle["test"]["mimic"],
            "dirs": {
                "nih": nih_source["experiment_dir"],
                "chexpert": chexpert_oracle["experiment_dir"],
                "mimic": mimic_oracle["experiment_dir"],
            },
        },
        "transfer": {
            "nih_source_checkpoint": nih_source["checkpoint_path"],
            "nih_to_chexpert": nih_source["test"]["chexpert"],
            "nih_to_mimic": nih_source["test"]["mimic"],
            "stage_summary": nih_source,
        },
        "naive_continual": {
            "stage_b": chexpert_adapt,
            "stage_c": mimic_adapt,
            "note": (
                "Legacy campaign-09 naive runs are reused when present. Missing Stage-B all-domain val/test files "
                "remain null unless the naive chain is rerun with the updated trainer."
            ),
        },
    }


def base_train_args(args: argparse.Namespace, *, experiments_root: Path) -> list[str]:
    argv = [
        "--embedding-root",
        str(args.embedding_view_root.resolve()),
        "--manifest-csv",
        str(args.manifest_csv.resolve()),
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
    if args.fp16_on_cuda:
        argv.append("--fp16-on-cuda")
    if args.max_rows_per_split is not None:
        argv.extend(["--max-rows-per-split", str(args.max_rows_per_split)])
    return argv


def stage_record(
    *,
    method: str,
    setting_id: str,
    hyperparameters: dict[str, float],
    stage_b_dir: Path,
    stage_c_dir: Path,
) -> dict[str, Any]:
    return {
        "method": method,
        "setting_id": setting_id,
        "hyperparameters": hyperparameters,
        "stage_b": load_stage_summary(stage_b_dir),
        "stage_c": load_stage_summary(stage_c_dir),
        "pareto": {},
    }


def run_lwf_setting(
    *,
    args: argparse.Namespace,
    experiments_root: Path,
    trainer_script: Path,
    reserve_name,
    common_args: list[str],
    source_checkpoint: Path,
    alpha: float,
    temperature: float,
) -> dict[str, Any]:
    setting_id = f"lwf_alpha-{format_float_slug(alpha)}_temp-{format_float_slug(temperature)}"
    stage_b_name = reserve_name(
        f"domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_lwf__alpha-{format_float_slug(alpha)}__temp-{format_float_slug(temperature)}__cxr_foundation_linear"
    )
    stage_c_name = reserve_name(
        f"domain_transfer_head_training__mimic_adapt_from_chexpert__pilot5h_lwf__alpha-{format_float_slug(alpha)}__temp-{format_float_slug(temperature)}__cxr_foundation_linear"
    )
    stage_b_dir = experiments_root / stage_b_name
    stage_c_dir = experiments_root / stage_c_name

    run_command(
        [
            sys.executable,
            str(trainer_script),
            "--experiment-name",
            stage_b_name,
            "--split-profile",
            "chexpert_adapt_from_nih",
            "--init-checkpoint",
            str(source_checkpoint),
            "--enable-lwf",
            "--lwf-teacher-checkpoint",
            str(source_checkpoint),
            "--lwf-alpha",
            str(alpha),
            "--lwf-temperature",
            str(temperature),
            *common_args,
        ]
    )
    run_command(
        [
            sys.executable,
            str(trainer_script),
            "--experiment-name",
            stage_c_name,
            "--split-profile",
            "mimic_adapt_from_chexpert",
            "--init-checkpoint",
            str(stage_b_dir / "best.ckpt"),
            "--enable-lwf",
            "--lwf-teacher-checkpoint",
            str(stage_b_dir / "best.ckpt"),
            "--lwf-alpha",
            str(alpha),
            "--lwf-temperature",
            str(temperature),
            *common_args,
        ]
    )
    return stage_record(
        method="lwf",
        setting_id=setting_id,
        hyperparameters={"alpha": float(alpha), "temperature": float(temperature)},
        stage_b_dir=stage_b_dir,
        stage_c_dir=stage_c_dir,
    )


def run_mas_setting(
    *,
    args: argparse.Namespace,
    experiments_root: Path,
    trainer_script: Path,
    mas_script: Path,
    reserve_name,
    common_args: list[str],
    source_checkpoint: Path,
    stage_a_mas_state_path: Path,
    output_root: Path,
    mas_lambda: float,
) -> dict[str, Any]:
    setting_id = f"mas_lambda-{format_float_slug(mas_lambda)}"
    stage_b_name = reserve_name(
        f"domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_mas__lambda-{format_float_slug(mas_lambda)}__cxr_foundation_linear"
    )
    stage_c_name = reserve_name(
        f"domain_transfer_head_training__mimic_adapt_from_chexpert__pilot5h_mas__lambda-{format_float_slug(mas_lambda)}__cxr_foundation_linear"
    )
    stage_b_dir = experiments_root / stage_b_name
    stage_c_dir = experiments_root / stage_c_name
    stage_b_mas_state = output_root / "mas_states" / setting_id / "stage_b_chexpert_mas_state.pt"

    run_command(
        [
            sys.executable,
            str(trainer_script),
            "--experiment-name",
            stage_b_name,
            "--split-profile",
            "chexpert_adapt_from_nih",
            "--init-checkpoint",
            str(source_checkpoint),
            "--enable-mas",
            "--mas-state-path",
            str(stage_a_mas_state_path),
            "--mas-lambda",
            str(mas_lambda),
            *common_args,
        ]
    )
    run_command(
        [
            sys.executable,
            str(mas_script),
            "--embedding-root",
            str(args.embedding_view_root.resolve()),
            "--manifest-csv",
            str(args.manifest_csv.resolve()),
            "--checkpoint-path",
            str(stage_b_dir / "best.ckpt"),
            "--output-path",
            str(stage_b_mas_state),
            "--domain",
            "d1_chexpert",
            "--split",
            "train",
            "--alias",
            "d1_chexpert_train",
            "--embedding-layout",
            "domain_split",
            "--previous-state-path",
            str(stage_a_mas_state_path),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            str(args.device),
            "--overwrite",
            *([] if not args.fp16_on_cuda else ["--fp16-on-cuda"]),
            *(
                ["--max-rows-per-split", str(args.max_rows_per_split)]
                if args.max_rows_per_split is not None
                else []
            ),
        ]
    )
    run_command(
        [
            sys.executable,
            str(trainer_script),
            "--experiment-name",
            stage_c_name,
            "--split-profile",
            "mimic_adapt_from_chexpert",
            "--init-checkpoint",
            str(stage_b_dir / "best.ckpt"),
            "--enable-mas",
            "--mas-state-path",
            str(stage_b_mas_state),
            "--mas-lambda",
            str(mas_lambda),
            *common_args,
        ]
    )
    return stage_record(
        method="mas",
        setting_id=setting_id,
        hyperparameters={"lambda": float(mas_lambda)},
        stage_b_dir=stage_b_dir,
        stage_c_dir=stage_c_dir,
    )


def sweep_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "method": record["method"],
            "setting_id": record["setting_id"],
            "stage_c_val_pareto": record["pareto"].get("stage_c_val_pareto"),
            "stage_c_test_pareto": record["pareto"].get("stage_c_test_pareto"),
        }
        row.update(record["hyperparameters"])
        row.update(flatten_stage_metrics("stage_b", record["stage_b"]))
        row.update(flatten_stage_metrics("stage_c", record["stage_c"]))
        rows.append(row)
    return rows


def method_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    annotate_pareto(records)
    return {
        "records": records,
        "validation_pareto_setting_ids": [
            record["setting_id"] for record in records if record["pareto"].get("stage_c_val_pareto")
        ],
        "test_pareto_setting_ids": [
            record["setting_id"] for record in records if record["pareto"].get("stage_c_test_pareto")
        ],
    }


def build_report(summary: dict[str, Any]) -> str:
    baseline = summary["baselines"]
    lwf = summary["lwf"]
    mas = summary["mas"]

    def oracle_line() -> str:
        return (
            f"- NIH `{format_metric(baseline['oracle']['nih'])}`, "
            f"CheXpert `{format_metric(baseline['oracle']['chexpert'])}`, "
            f"MIMIC `{format_metric(baseline['oracle']['mimic'])}`"
        )

    def sequential_block(label: str, block: dict[str, Any]) -> list[str]:
        lines = [
            f"## {label}",
            "",
            "- Final-stage Pareto highlighting uses NIH/CheXpert/MIMIC validation AUROC only.",
            "- Test Pareto flags are report-only.",
            "",
            "| Setting | Stage B NIH test | Stage B CheXpert test | Stage B MIMIC test | Stage C NIH test | Stage C CheXpert test | Stage C MIMIC test | Val Pareto | Test Pareto |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for record in block["records"]:
            lines.append(
                "| "
                f"{record['setting_id']} | "
                f"{format_metric(record['stage_b']['test']['nih'])} | "
                f"{format_metric(record['stage_b']['test']['chexpert'])} | "
                f"{format_metric(record['stage_b']['test']['mimic'])} | "
                f"{format_metric(record['stage_c']['test']['nih'])} | "
                f"{format_metric(record['stage_c']['test']['chexpert'])} | "
                f"{format_metric(record['stage_c']['test']['mimic'])} | "
                f"{'yes' if record['pareto'].get('stage_c_val_pareto') else 'no'} | "
                f"{'yes' if record['pareto'].get('stage_c_test_pareto') else 'no'} |"
            )
        lines.extend(
            [
                "",
                f"- Validation Pareto settings: `{', '.join(block['validation_pareto_setting_ids']) or 'none'}`",
                f"- Test Pareto settings: `{', '.join(block['test_pareto_setting_ids']) or 'none'}`",
                "",
            ]
        )
        return lines

    lines = [
        "# Sequential Regularizer Sweeps",
        "",
        "- Framing: head-only, frozen-embedding, rehearsal-free continual-learning baselines on CXR Foundation embeddings.",
        f"- Trainable params: `{summary['head']['params_text']}`",
        f"- Trainable-head FLOPs per image: `{summary['head']['flops_text']}`",
        "- Stage-B-on-MIMIC is report-only and is not used for model selection or hyperparameter choice.",
        "",
        "## Independent Single-Domain Oracle Baselines",
        "",
        oracle_line(),
        "",
        "## Cross-Domain Transfer Baseline",
        "",
        f"- NIH -> CheXpert test AUROC: `{format_metric(baseline['transfer']['nih_to_chexpert'])}`",
        f"- NIH -> MIMIC test AUROC: `{format_metric(baseline['transfer']['nih_to_mimic'])}`",
        "",
        "## Naive Continual Baseline",
        "",
        f"- After CheXpert: NIH `{format_metric(baseline['naive_continual']['stage_b']['test']['nih'])}`, CheXpert `{format_metric(baseline['naive_continual']['stage_b']['test']['chexpert'])}`, MIMIC `{format_metric(baseline['naive_continual']['stage_b']['test']['mimic'])}`",
        f"- After MIMIC: NIH `{format_metric(baseline['naive_continual']['stage_c']['test']['nih'])}`, CheXpert `{format_metric(baseline['naive_continual']['stage_c']['test']['chexpert'])}`, MIMIC `{format_metric(baseline['naive_continual']['stage_c']['test']['mimic'])}`",
        f"- Note: {baseline['naive_continual']['note']}",
        "",
    ]
    lines.extend(sequential_block("Sequential LwF", lwf))
    lines.extend(sequential_block("Sequential MAS", mas))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run strict rehearsal-free LwF and MAS sweeps for the NIH -> CheXpert -> MIMIC pilot5h chain."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-campaign-root", type=Path, default=DEFAULT_BASELINE_CAMPAIGN_ROOT)
    parser.add_argument("--embedding-view-root", type=Path, default=DEFAULT_EMBEDDING_VIEW_ROOT)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--trainer-script", type=Path, default=DEFAULT_TRAINER_SCRIPT)
    parser.add_argument("--mas-script", type=Path, default=DEFAULT_MAS_SCRIPT)
    parser.add_argument("--methods", nargs="*", choices=("lwf", "mas"), default=["lwf", "mas"])
    parser.add_argument("--lwf-alpha-grid", type=float, nargs="*", default=list(DEFAULT_LWF_ALPHAS))
    parser.add_argument("--lwf-temperature-grid", type=float, nargs="*", default=list(DEFAULT_LWF_TEMPERATURES))
    parser.add_argument("--mas-lambda-grid", type=float, nargs="*", default=list(DEFAULT_MAS_LAMBDAS))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--fp16-on-cuda", action="store_true")
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0.")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive.")
    if args.weight_decay < 0:
        raise SystemExit("--weight-decay must be >= 0.")
    if args.patience < 0:
        raise SystemExit("--patience must be >= 0.")
    if args.max_rows_per_split is not None and args.max_rows_per_split <= 0:
        raise SystemExit("--max-rows-per-split must be positive when provided.")
    if not args.methods:
        raise SystemExit("--methods must include at least one of: lwf, mas.")
    if "lwf" in args.methods and (not args.lwf_alpha_grid or not args.lwf_temperature_grid):
        raise SystemExit("LwF sweeps require non-empty --lwf-alpha-grid and --lwf-temperature-grid.")
    if "mas" in args.methods and not args.mas_lambda_grid:
        raise SystemExit("MAS sweeps require a non-empty --mas-lambda-grid.")

    output_root = args.output_root.resolve()
    experiments_root = output_root / "experiments" / "by_id"
    run_config_path = output_root / "run_config.json"
    summary_json_path = output_root / "study_summary.json"
    summary_report_path = output_root / "study_report.md"
    lwf_csv_path = output_root / "lwf_sweep_summary.csv"
    mas_csv_path = output_root / "mas_sweep_summary.csv"

    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output root already exists: {output_root}")
        shutil.rmtree(output_root)
    experiments_root.mkdir(parents=True, exist_ok=True)

    baseline = build_baseline_summary(args.baseline_campaign_root.resolve())
    source_checkpoint = Path(baseline["transfer"]["nih_source_checkpoint"])
    reserve_name = reserve_name_factory(experiments_root)
    common_args = base_train_args(args, experiments_root=experiments_root)

    write_json(
        run_config_path,
        {
            "run_date_utc": utc_now_iso(),
            "output_root": str(output_root),
            "baseline_campaign_root": str(args.baseline_campaign_root.resolve()),
            "embedding_view_root": str(args.embedding_view_root.resolve()),
            "manifest_csv": str(args.manifest_csv.resolve()),
            "trainer_script": str(args.trainer_script.resolve()),
            "mas_script": str(args.mas_script.resolve()),
            "methods": list(args.methods),
            "lwf_alpha_grid": list(args.lwf_alpha_grid),
            "lwf_temperature_grid": list(args.lwf_temperature_grid),
            "mas_lambda_grid": list(args.mas_lambda_grid),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "seed": int(args.seed),
            "device": str(args.device),
            "fp16_on_cuda": bool(args.fp16_on_cuda),
            "max_rows_per_split": int(args.max_rows_per_split) if args.max_rows_per_split is not None else None,
            "source_checkpoint": str(source_checkpoint),
        },
    )

    lwf_records: list[dict[str, Any]] = []
    if "lwf" in args.methods:
        for alpha in args.lwf_alpha_grid:
            for temperature in args.lwf_temperature_grid:
                lwf_records.append(
                    run_lwf_setting(
                        args=args,
                        experiments_root=experiments_root,
                        trainer_script=args.trainer_script.resolve(),
                        reserve_name=reserve_name,
                        common_args=common_args,
                        source_checkpoint=source_checkpoint,
                        alpha=float(alpha),
                        temperature=float(temperature),
                    )
                )
    lwf_summary = method_summary(lwf_records)
    if lwf_records:
        write_csv(
            lwf_csv_path,
            sweep_rows(lwf_summary["records"]),
            fieldnames=list(sweep_rows(lwf_summary["records"])[0].keys()),
        )

    mas_records: list[dict[str, Any]] = []
    if "mas" in args.methods:
        stage_a_mas_state_path = output_root / "mas_states" / "stage_a_nih_source_mas_state.pt"
        run_command(
            [
                sys.executable,
                str(args.mas_script.resolve()),
                "--embedding-root",
                str(args.embedding_view_root.resolve()),
                "--manifest-csv",
                str(args.manifest_csv.resolve()),
                "--checkpoint-path",
                str(source_checkpoint),
                "--output-path",
                str(stage_a_mas_state_path),
                "--domain",
                "d0_nih",
                "--split",
                "train",
                "--alias",
                "d0_nih_train",
                "--embedding-layout",
                "domain_split",
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                str(args.device),
                "--overwrite",
                *([] if not args.fp16_on_cuda else ["--fp16-on-cuda"]),
                *(
                    ["--max-rows-per-split", str(args.max_rows_per_split)]
                    if args.max_rows_per_split is not None
                    else []
                ),
            ]
        )
        for mas_lambda in args.mas_lambda_grid:
            mas_records.append(
                run_mas_setting(
                    args=args,
                    experiments_root=experiments_root,
                    trainer_script=args.trainer_script.resolve(),
                    mas_script=args.mas_script.resolve(),
                    reserve_name=reserve_name,
                    common_args=common_args,
                    source_checkpoint=source_checkpoint,
                    stage_a_mas_state_path=stage_a_mas_state_path,
                    output_root=output_root,
                    mas_lambda=float(mas_lambda),
                )
            )
    mas_summary = method_summary(mas_records)
    if mas_records:
        write_csv(
            mas_csv_path,
            sweep_rows(mas_summary["records"]),
            fieldnames=list(sweep_rows(mas_summary["records"])[0].keys()),
        )

    summary = {
        "run_date_utc": utc_now_iso(),
        "study_type": "sequential_regularizer_sweeps",
        "framing": {
            "head_only": True,
            "frozen_embedding": True,
            "rehearsal_free": True,
            "selection_policy": "validation_pareto_only",
            "test_pareto_policy": "report_only",
            "stage_b_future_domain_policy": "report_only",
        },
        "head": {
            "params_text": REPORT_PARAMS_TEXT,
            "flops_text": REPORT_FLOPS_TEXT,
        },
        "manifest_csv": str(args.manifest_csv.resolve()),
        "embedding_view_root": str(args.embedding_view_root.resolve()),
        "baseline_campaign_root": str(args.baseline_campaign_root.resolve()),
        "baselines": baseline,
        "lwf": lwf_summary,
        "mas": mas_summary,
    }
    write_json(summary_json_path, summary)
    summary_report_path.write_text(build_report(summary), encoding="utf-8")
    print(f"[done] summary_json={summary_json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
