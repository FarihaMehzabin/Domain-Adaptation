#!/usr/bin/env python3
"""Run a small NIH-source MLP sweep on existing direct-transfer embedding roots."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import experiment_layout


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest/manifest_common_labels_pilot5h.csv")
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_LOGS_ROOT = Path("/workspace/logs")
DEFAULT_OPERATION_LABEL = "domain_transfer_mlp_sweep"
DEFAULT_EXPERIMENT_ID_WIDTH = 4
DEFAULT_HIDDEN_DIMS = (256, 512, 1024)
DEFAULT_MLP_DROPOUT = 0.2
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5
DEFAULT_SEED = 1337
DEFAULT_DEVICE = "auto"
DEFAULT_TRAINER_SCRIPT = Path("/workspace/scripts/15_train_domain_transfer_linear_probe.py")
DEFAULT_EMBEDDING_ROOTS = (
    experiment_layout.find_experiment_dir("exp0012", experiments_root=DEFAULT_EXPERIMENTS_ROOT),
    experiment_layout.find_experiment_dir("exp0014", experiments_root=DEFAULT_EXPERIMENTS_ROOT),
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str, *, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or fallback


def ensure_operation_prefix(name: str, operation_label: str = DEFAULT_OPERATION_LABEL) -> str:
    normalized_label = slugify(operation_label, fallback="operation")
    if name.startswith("exp") and "__" in name:
        prefix, remainder = name.split("__", 1)
        if remainder.startswith(normalized_label):
            return name
        return f"{prefix}__{normalized_label}__{remainder}"
    if name.startswith(normalized_label):
        return name
    return f"{normalized_label}__{name}"


def strip_experiment_prefix(name: str) -> str:
    if name.startswith("exp") and "__" in name:
        prefix, remainder = name.split("__", 1)
        if extract_experiment_number(prefix) is not None:
            return remainder
    return name


def extract_experiment_number(name: str) -> int | None:
    if not name.startswith("exp"):
        return None
    prefix = name.split("__", 1)[0]
    digits = prefix.removeprefix("exp")
    if not digits.isdigit():
        return None
    return int(digits)


def next_experiment_number(experiments_root: Path) -> int:
    return experiment_layout.next_experiment_number(experiments_root)


def resolve_experiment_identity(
    *,
    experiments_root: Path,
    requested_name: str | None,
    generated_slug: str,
    overwrite: bool,
    id_width: int = DEFAULT_EXPERIMENT_ID_WIDTH,
) -> tuple[int, str, str, Path]:
    requested = (requested_name or "").strip() or None
    base_name = ensure_operation_prefix(requested or generated_slug)
    return experiment_layout.resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=base_name if requested else None,
        generated_slug=base_name,
        overwrite=overwrite,
        id_width=id_width,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def log(message: str, *, handle: Any | None = None) -> None:
    line = f"[{utc_now_iso()}] {message}"
    print(line, flush=True)
    if handle is not None:
        handle.write(line + "\n")
        handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep small source-trained MLP heads on existing NIH-source transfer embeddings and "
            "rank candidates by D0 val macro AUROC."
        )
    )
    parser.add_argument(
        "--embedding-root",
        dest="embedding_roots",
        type=Path,
        nargs="+",
        default=list(DEFAULT_EMBEDDING_ROOTS),
        help="One or more embedding roots to sweep. Defaults to the current pilot ResNet50 and CXR Foundation roots.",
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--trainer-script", type=Path, default=DEFAULT_TRAINER_SCRIPT)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=list(DEFAULT_HIDDEN_DIMS))
    parser.add_argument("--mlp-dropout", type=float, default=DEFAULT_MLP_DROPOUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--fp16-on-cuda", action="store_true")
    parser.add_argument("--token-pooling", choices=("avg", "cls", "flatten"), default="avg")
    parser.add_argument("--l2-normalize-features", action="store_true")
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-child-runs", action="store_true")
    return parser.parse_args()


def metric_sort_key(summary: dict[str, Any]) -> tuple[float, float, float]:
    d0_val = summary["metrics"]["d0_val"]
    macro = d0_val["macro"]
    auroc = float(macro["auroc"]) if macro["auroc"] is not None else float("-inf")
    ap = float(macro["average_precision"]) if macro["average_precision"] is not None else float("-inf")
    loss = float(d0_val["loss"])
    return (auroc, ap, -loss)


def extract_macro(summary: dict[str, Any], alias: str) -> dict[str, Any]:
    payload = summary["metrics"].get(alias)
    if payload is None:
        return {"auroc": None, "average_precision": None, "loss": None}
    return {
        "auroc": payload["macro"]["auroc"],
        "average_precision": payload["macro"]["average_precision"],
        "loss": payload["loss"],
    }


def load_run_artifacts(run_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for alias, filename in {
        "d0_val": "d0_val_metrics.json",
        "d0_test": "d0_test_metrics.json",
        "d1_transfer": "d1_transfer_metrics.json",
        "d2_transfer": "d2_transfer_metrics.json",
    }.items():
        path = run_dir / filename
        if path.exists():
            metrics[alias] = json.loads(path.read_text(encoding="utf-8"))
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return {
        "run_dir": str(run_dir),
        "experiment_name": run_dir.name,
        "config": config,
        "metrics": metrics,
    }


def find_linear_baseline(
    *,
    experiments_root: Path,
    embedding_root: Path,
    manifest_csv: Path,
    max_rows_per_split: int | None,
) -> dict[str, Any] | None:
    candidates: list[tuple[int, Path]] = []
    for entry in experiment_layout.iter_experiment_entries(experiments_root):
        child = entry.path
        config_path = child / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if Path(str(config.get("embedding_root") or "")).resolve() != embedding_root.resolve():
            continue
        if Path(str(config.get("manifest_csv") or "")).resolve() != manifest_csv.resolve():
            continue
        config_max_rows = config.get("max_rows_per_split")
        normalized_config_max_rows = int(config_max_rows) if config_max_rows is not None else None
        if normalized_config_max_rows != max_rows_per_split:
            continue
        head_type = str(config.get("head_type") or "linear")
        if head_type != "linear":
            continue
        number = extract_experiment_number(child.name)
        candidates.append((number or -1, child))
    if not candidates:
        return None
    _, selected_dir = max(candidates, key=lambda item: item[0])
    return load_run_artifacts(selected_dir)


def run_child_experiment(
    *,
    trainer_script: Path,
    embedding_root: Path,
    manifest_csv: Path,
    hidden_dim: int,
    mlp_dropout: float,
    batch_size: int,
    num_workers: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    device: str,
    fp16_on_cuda: bool,
    token_pooling: str,
    l2_normalize_features: bool,
    max_rows_per_split: int | None,
    overwrite_child_runs: bool,
    log_path: Path,
) -> Path:
    requested_name = "__".join(
        [
            slugify(strip_experiment_prefix(embedding_root.name), fallback="embedding-root"),
            "pilot5h",
            "head-mlp",
            f"hidden-{hidden_dim}",
            f"dropout-{str(mlp_dropout).replace('.', 'p')}",
        ]
    )
    command = [
        sys.executable,
        "-u",
        str(trainer_script),
        "--embedding-root",
        str(embedding_root),
        "--manifest-csv",
        str(manifest_csv),
        "--head-type",
        "mlp",
        "--mlp-hidden-dims",
        str(hidden_dim),
        "--mlp-dropout",
        str(mlp_dropout),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--patience",
        str(patience),
        "--seed",
        str(seed),
        "--device",
        device,
        "--token-pooling",
        token_pooling,
        "--experiment-name",
        requested_name,
    ]
    if fp16_on_cuda:
        command.append("--fp16-on-cuda")
    if l2_normalize_features:
        command.append("--l2-normalize-features")
    if max_rows_per_split is not None:
        command.extend(["--max-rows-per-split", str(max_rows_per_split)])
    if overwrite_child_runs:
        command.append("--overwrite")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_dir: Path | None = None
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("# Command\n")
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            print(line, flush=True)
            handle.write(line + "\n")
            handle.flush()
            if line.startswith("[info] experiment_dir="):
                experiment_dir = Path(line.split("=", 1)[1].strip())
        return_code = process.wait()
        if return_code != 0:
            raise SystemExit(f"Child run failed with exit code {return_code}. See log: {log_path}")
    if experiment_dir is None:
        raise SystemExit(f"Could not determine child experiment directory from log: {log_path}")
    return experiment_dir


def build_leaderboard_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(results, key=metric_sort_key, reverse=True)
    rows: list[dict[str, Any]] = []
    for rank, result in enumerate(ordered, start=1):
        d0_val = extract_macro(result, "d0_val")
        d0_test = extract_macro(result, "d0_test")
        d1 = extract_macro(result, "d1_transfer")
        d2 = extract_macro(result, "d2_transfer")
        baseline = result.get("linear_baseline")
        baseline_deltas = {}
        if baseline is not None:
            for alias in ("d0_test", "d1_transfer", "d2_transfer"):
                current = extract_macro(result, alias)
                reference = extract_macro(baseline, alias)
                if current["auroc"] is None or reference["auroc"] is None:
                    baseline_deltas[alias] = None
                else:
                    baseline_deltas[alias] = float(current["auroc"]) - float(reference["auroc"])
        rows.append(
            {
                "rank": rank,
                "experiment_name": result["experiment_name"],
                "run_dir": result["run_dir"],
                "embedding_root": result["config"].get("embedding_root"),
                "head_type": result["config"].get("head_type"),
                "mlp_hidden_dims": result["config"].get("mlp_hidden_dims"),
                "mlp_dropout": result["config"].get("mlp_dropout"),
                "model_num_parameters": result["config"].get("model_num_parameters"),
                "d0_val_macro_auroc": d0_val["auroc"],
                "d0_val_macro_average_precision": d0_val["average_precision"],
                "d0_val_loss": d0_val["loss"],
                "d0_test_macro_auroc": d0_test["auroc"],
                "d1_transfer_macro_auroc": d1["auroc"],
                "d2_transfer_macro_auroc": d2["auroc"],
                "delta_vs_linear_d0_test_auroc": baseline_deltas.get("d0_test"),
                "delta_vs_linear_d1_transfer_auroc": baseline_deltas.get("d1_transfer"),
                "delta_vs_linear_d2_transfer_auroc": baseline_deltas.get("d2_transfer"),
                "linear_baseline_run_dir": baseline["run_dir"] if baseline is not None else None,
            }
        )
    return rows


def write_leaderboard_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "experiment_name",
        "run_dir",
        "embedding_root",
        "head_type",
        "mlp_hidden_dims",
        "mlp_dropout",
        "model_num_parameters",
        "d0_val_macro_auroc",
        "d0_val_macro_average_precision",
        "d0_val_loss",
        "d0_test_macro_auroc",
        "d1_transfer_macro_auroc",
        "d2_transfer_macro_auroc",
        "delta_vs_linear_d0_test_auroc",
        "delta_vs_linear_d1_transfer_auroc",
        "delta_vs_linear_d2_transfer_auroc",
        "linear_baseline_run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_summary_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Domain Transfer MLP Sweep Summary",
        "",
        "| Rank | Backbone | Hidden | D0 Val AUROC | D0 Test AUROC | D1 AUROC | D2 AUROC | Delta vs Linear D0 | Delta vs Linear D1 | Delta vs Linear D2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        backbone = Path(str(row["embedding_root"])).name
        hidden_dims = row["mlp_hidden_dims"] or []
        hidden = "x".join(str(dim) for dim in hidden_dims) if hidden_dims else "n/a"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rank"]),
                    backbone,
                    hidden,
                    format_metric(row["d0_val_macro_auroc"]),
                    format_metric(row["d0_test_macro_auroc"]),
                    format_metric(row["d1_transfer_macro_auroc"]),
                    format_metric(row["d2_transfer_macro_auroc"]),
                    format_metric(row["delta_vs_linear_d0_test_auroc"]),
                    format_metric(row["delta_vs_linear_d1_transfer_auroc"]),
                    format_metric(row["delta_vs_linear_d2_transfer_auroc"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def format_metric(value: float | None) -> str:
    return "null" if value is None else f"{float(value):.6f}"


def main() -> int:
    args = parse_args()
    if not args.hidden_dims:
        raise SystemExit("--hidden-dims must contain at least one candidate.")
    if not (0.0 <= float(args.mlp_dropout) < 1.0):
        raise SystemExit("--mlp-dropout must be in [0.0, 1.0).")
    for hidden_dim in args.hidden_dims:
        if hidden_dim <= 0:
            raise SystemExit("--hidden-dims values must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
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

    embedding_roots = [path.resolve() for path in args.embedding_roots]
    manifest_csv = args.manifest_csv.resolve()
    trainer_script = args.trainer_script.resolve()
    experiments_root = args.experiments_root.resolve()
    logs_root = args.logs_root.resolve()

    for embedding_root in embedding_roots:
        if not embedding_root.exists():
            raise SystemExit(f"Embedding root not found: {embedding_root}")
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest not found: {manifest_csv}")
    if not trainer_script.exists():
        raise SystemExit(f"Trainer script not found: {trainer_script}")

    generated_slug = "__".join(
        [
            DEFAULT_OPERATION_LABEL,
            slugify(manifest_csv.stem, fallback="manifest"),
            f"hidden-{'x'.join(str(dim) for dim in args.hidden_dims)}",
        ]
    )
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=bool(args.overwrite),
    )
    log_path = logs_root / f"{experiment_name}.log"
    logs_root.mkdir(parents=True, exist_ok=True)

    config = {
        "argv": list(sys.argv),
        "run_date_utc": utc_now_iso(),
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "manifest_csv": str(manifest_csv),
        "trainer_script": str(trainer_script),
        "embedding_roots": [str(path) for path in embedding_roots],
        "hidden_dims": list(args.hidden_dims),
        "mlp_dropout": float(args.mlp_dropout),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "device": args.device,
        "fp16_on_cuda": bool(args.fp16_on_cuda),
        "token_pooling": args.token_pooling,
        "l2_normalize_features": bool(args.l2_normalize_features),
        "max_rows_per_split": int(args.max_rows_per_split) if args.max_rows_per_split is not None else None,
    }
    write_json(experiment_dir / "config.json", config)

    results: list[dict[str, Any]] = []
    with log_path.open("w", encoding="utf-8") as handle:
        log(f"experiment_dir={experiment_dir}", handle=handle)
        log(f"manifest_csv={manifest_csv}", handle=handle)
        for embedding_root in embedding_roots:
            baseline = find_linear_baseline(
                experiments_root=experiments_root,
                embedding_root=embedding_root,
                manifest_csv=manifest_csv,
                max_rows_per_split=int(args.max_rows_per_split) if args.max_rows_per_split is not None else None,
            )
            if baseline is None:
                log(f"no_linear_baseline_found embedding_root={embedding_root}", handle=handle)
            else:
                log(
                    f"linear_baseline embedding_root={embedding_root} run_dir={baseline['run_dir']}",
                    handle=handle,
                )
            for hidden_dim in args.hidden_dims:
                child_log_path = logs_root / (
                    f"{experiment_name}__{slugify(embedding_root.name, fallback='embedding-root')}__hidden-{hidden_dim}.log"
                )
                log(
                    f"starting embedding_root={embedding_root} hidden_dim={hidden_dim} child_log={child_log_path}",
                    handle=handle,
                )
                child_dir = run_child_experiment(
                    trainer_script=trainer_script,
                    embedding_root=embedding_root,
                    manifest_csv=manifest_csv,
                    hidden_dim=hidden_dim,
                    mlp_dropout=float(args.mlp_dropout),
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    epochs=int(args.epochs),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    patience=int(args.patience),
                    seed=int(args.seed),
                    device=str(args.device),
                    fp16_on_cuda=bool(args.fp16_on_cuda),
                    token_pooling=args.token_pooling,
                    l2_normalize_features=bool(args.l2_normalize_features),
                    max_rows_per_split=int(args.max_rows_per_split) if args.max_rows_per_split is not None else None,
                    overwrite_child_runs=bool(args.overwrite_child_runs),
                    log_path=child_log_path,
                )
                result = load_run_artifacts(child_dir)
                result["linear_baseline"] = baseline
                results.append(result)
                log(
                    "completed "
                    f"embedding_root={embedding_root} hidden_dim={hidden_dim} "
                    f"d0_val_macro_auroc={format_metric(extract_macro(result, 'd0_val')['auroc'])}",
                    handle=handle,
                )

    leaderboard_rows = build_leaderboard_rows(results)
    write_json(
        experiment_dir / "leaderboard.json",
        {
            "run_date_utc": utc_now_iso(),
            "ranking_metric": {
                "primary": "d0_val.macro.auroc",
                "tie_break_1": "d0_val.macro.average_precision",
                "tie_break_2": "lower_d0_val.loss",
            },
            "rows": leaderboard_rows,
        },
    )
    write_leaderboard_csv(experiment_dir / "leaderboard.csv", leaderboard_rows)
    (experiment_dir / "summary.md").write_text(render_summary_markdown(leaderboard_rows), encoding="utf-8")
    write_json(
        experiment_dir / "experiment_meta.json",
        {
            "run_date_utc": utc_now_iso(),
            "experiment_number": experiment_number,
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "experiment_dir": str(experiment_dir),
            "operation_label": DEFAULT_OPERATION_LABEL,
            "manifest_csv": str(manifest_csv),
            "trainer_script": str(trainer_script),
            "log_path": str(log_path),
            "num_child_runs": len(results),
            "leaderboard_path": str(experiment_dir / "leaderboard.json"),
        },
    )
    print(f"[done] experiment_dir={experiment_dir} leaderboard={experiment_dir / 'leaderboard.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
