#!/usr/bin/env python3
"""Evaluate held-out test memory probabilities using a frozen validation-selected configuration."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_module(script_name: str, module_name: str) -> tuple[Path, Any]:
    script_path = Path(__file__).resolve().with_name(script_name)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return script_path, module


SOURCE_SELECTION_SCRIPT_PATH, source_memory_mod = load_module(
    "06_evaluate_source_memory_only.py",
    "source_memory_only_validation",
)

DEFAULT_MANIFEST_CSV = source_memory_mod.DEFAULT_MANIFEST_CSV
DEFAULT_MEMORY_ROOT = source_memory_mod.DEFAULT_MEMORY_ROOT
DEFAULT_MEMORY_SELECTION_ROOT = Path(
    "/workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val"
)
DEFAULT_QUERY_EMBEDDING_ROOT = source_memory_mod.DEFAULT_QUERY_EMBEDDING_ROOT
DEFAULT_BASELINE_EXPERIMENT_DIR = source_memory_mod.DEFAULT_BASELINE_EXPERIMENT_DIR
DEFAULT_EXPERIMENTS_ROOT = source_memory_mod.DEFAULT_EXPERIMENTS_ROOT
DEFAULT_OPERATION_LABEL = "source_memory_only_test_evaluation"
DEFAULT_EXPERIMENT_ID_WIDTH = source_memory_mod.DEFAULT_EXPERIMENT_ID_WIDTH
DEFAULT_SPLIT = "test"
DEFAULT_QUALITATIVE_QUERIES = source_memory_mod.DEFAULT_QUALITATIVE_QUERIES
DEFAULT_SEED = source_memory_mod.DEFAULT_SEED
FROZEN_SELECTION_MODE = "frozen_validation_config"


def ensure_operation_prefix(name: str, operation_label: str = DEFAULT_OPERATION_LABEL) -> str:
    normalized_label = source_memory_mod.slugify(operation_label, fallback="operation")
    if name.startswith("exp") and "__" in name:
        prefix, remainder = name.split("__", 1)
        if remainder.startswith(normalized_label):
            return name
        return f"{prefix}__{normalized_label}__{remainder}"
    if name.startswith(normalized_label):
        return name
    return f"{normalized_label}__{name}"


def resolve_experiment_identity(
    *,
    experiments_root: Path,
    requested_name: str | None,
    generated_slug: str,
    overwrite: bool,
    id_width: int = DEFAULT_EXPERIMENT_ID_WIDTH,
) -> tuple[int, str, str, Path]:
    experiments_root.mkdir(parents=True, exist_ok=True)
    if requested_name:
        requested_name = requested_name.strip()
        if not requested_name:
            raise SystemExit("--experiment-name cannot be empty.")
    base_name = ensure_operation_prefix(requested_name or generated_slug)
    explicit_number = source_memory_mod.extract_experiment_number(base_name)
    if explicit_number is not None:
        experiment_number = explicit_number
        experiment_name = base_name
    else:
        experiment_number = source_memory_mod.next_experiment_number(experiments_root)
        experiment_name = f"exp{experiment_number:0{id_width}d}__{base_name}"

    experiment_id = f"exp{experiment_number:0{id_width}d}"
    experiment_dir = experiments_root / experiment_name
    if experiment_dir.exists() and not overwrite:
        raise SystemExit(
            f"Experiment directory already exists: {experiment_dir}\n"
            "Pass --overwrite to reuse it or choose a different --experiment-name."
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_number, experiment_id, experiment_name, experiment_dir


def format_shell_command(argv: list[str]) -> str:
    return shlex.join(argv)


def format_metric(value: float | None) -> str:
    return source_memory_mod.format_metric(value)


def extract_thresholds(
    threshold_payload: dict[str, Any],
    label_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    values: list[float] = []
    normalized_payload: dict[str, Any] = {}
    for label_name in label_names:
        current = threshold_payload.get(label_name)
        if not isinstance(current, dict) or "threshold" not in current:
            raise SystemExit(f"Frozen threshold for label '{label_name}' is missing from the validation artifact.")
        threshold = float(current["threshold"])
        values.append(threshold)
        normalized_payload[label_name] = {
            "threshold": threshold,
            "best_f1_on_val": current.get("best_f1"),
            "prevalence_on_val": current.get("prevalence"),
        }
    return np.asarray(values, dtype=np.float32), normalized_payload


def evaluate_probabilities_with_frozen_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    label_names: list[str],
    thresholds: np.ndarray,
    threshold_payload: dict[str, Any],
) -> dict[str, Any]:
    per_label: dict[str, dict[str, Any]] = {}
    macro_auroc_values: list[float] = []
    macro_ap_values: list[float] = []
    macro_f1_values: list[float] = []
    macro_f1_frozen_values: list[float] = []
    macro_ece_values: list[float] = []

    for idx, label_name in enumerate(label_names):
        targets = y_true[:, idx]
        scores = probs[:, idx]
        threshold = float(thresholds[idx])
        auroc = source_memory_mod.binary_auroc(targets, scores)
        average_precision = source_memory_mod.binary_average_precision(targets, scores)
        f1_at_0p5 = source_memory_mod.binary_f1(targets, scores, 0.5)
        f1_at_frozen = source_memory_mod.binary_f1(targets, scores, threshold)
        ece = source_memory_mod.binary_ece(targets, scores)

        if auroc is not None:
            macro_auroc_values.append(auroc)
        if average_precision is not None:
            macro_ap_values.append(average_precision)
        macro_f1_values.append(f1_at_0p5)
        macro_f1_frozen_values.append(f1_at_frozen)
        macro_ece_values.append(ece)

        current_threshold_payload = dict(threshold_payload[label_name])
        per_label[label_name] = {
            "auroc": auroc,
            "average_precision": average_precision,
            "ece": ece,
            "f1_at_0.5": f1_at_0p5,
            "f1_at_frozen_threshold": f1_at_frozen,
            "positive_count": int(targets.sum()),
            "threshold": threshold,
            "frozen_threshold_source": current_threshold_payload,
        }

    return {
        "macro_auroc": float(np.mean(macro_auroc_values)) if macro_auroc_values else None,
        "macro_average_precision": float(np.mean(macro_ap_values)) if macro_ap_values else None,
        "macro_ece": float(np.mean(macro_ece_values)) if macro_ece_values else None,
        "macro_f1_at_0.5": float(np.mean(macro_f1_values)) if macro_f1_values else None,
        "macro_f1_at_frozen_thresholds": (
            float(np.mean(macro_f1_frozen_values)) if macro_f1_frozen_values else None
        ),
        "thresholds": threshold_payload,
        "per_label": per_label,
    }


def build_recreation_report(
    *,
    experiment_dir: Path,
    experiment_id: str,
    operation_label: str,
    script_path: Path,
    argv_exact: list[str],
    argv_fresh: list[str],
    memory_root: Path,
    memory_selection_root: Path,
    baseline_experiment_dir: Path,
    query_embedding_root: Path,
    manifest_csv: Path,
    split: str,
    row_count: int,
    memory_rows: int,
    memory_dim: int,
    query_dim: int,
    label_names: list[str],
    applied_k: int,
    applied_tau: float,
    norm_summary_before: dict[str, float],
    norm_summary_after: dict[str, float],
    index_details: dict[str, Any],
    test_metrics: dict[str, Any],
    output_paths: list[Path],
) -> str:
    size_lines = [f"- {path.name}: `{source_memory_mod.human_size(path.stat().st_size)}`" for path in output_paths if path.exists()]
    total_size = sum(path.stat().st_size for path in output_paths if path.exists())
    hash_lines = [f"{source_memory_mod.sha256_file(path)}  {path}" for path in output_paths if path.exists()]
    faiss_version = getattr(source_memory_mod.faiss, "__version__", "unknown")
    lines = [
        "# Source Memory-Only Test Evaluation Recreation Report",
        "",
        "## Scope",
        "",
        "This report documents how to recreate the held-out test memory evaluation experiment stored at:",
        "",
        f"`{experiment_dir}`",
        "",
        "The producing script is:",
        "",
        f"`{script_path}`",
        "",
        "Script SHA-256:",
        "",
        f"`{source_memory_mod.script_sha256(script_path)}`",
        "",
        "## Final Experiment Identity",
        "",
        f"- Experiment directory: `{experiment_dir}`",
        f"- Experiment id: `{experiment_id}`",
        f"- Operation label: `{operation_label}`",
        f"- Memory root: `{memory_root}`",
        f"- Validation selection root: `{memory_selection_root}`",
        f"- Baseline reference experiment: `{baseline_experiment_dir}`",
        f"- Query embedding root: `{query_embedding_root}`",
        f"- Manifest: `{manifest_csv}`",
        f"- Evaluation split: `{split}`",
        f"- Test query rows: `{row_count:,}`",
        f"- Train memory rows: `{memory_rows:,}`",
        f"- Query embedding dimension: `{query_dim}`",
        f"- Memory embedding dimension: `{memory_dim}`",
        f"- Label count: `{len(label_names)}`",
        f"- Label names: `{' '.join(label_names)}`",
        f"- Applied selection mode: `{FROZEN_SELECTION_MODE}`",
        "",
        "## Environment",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- NumPy: `{np.__version__}`",
        f"- faiss: `{faiss_version}`",
        f"- Platform: `{platform.platform()}`",
        "",
        "## Exact Recreation Command",
        "",
        "If you want to recreate the same directory name in place, use this command:",
        "",
        "```bash",
        format_shell_command(argv_exact),
        "```",
        "",
        "If you want a fresh numbered run instead of overwriting the existing directory, use:",
        "",
        "```bash",
        format_shell_command(argv_fresh),
        "```",
        "",
        "## Preconditions",
        "",
        f"- The train memory must already exist at `{memory_root}`.",
        f"- The validation memory-selection experiment must already exist at `{memory_selection_root}`.",
        f"- The query embeddings must already exist at `{query_embedding_root / split}`.",
        f"- The manifest must be present at `{manifest_csv}`.",
        "- The required Python packages must be importable: `numpy`, `faiss`.",
        "- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected configuration.",
        "",
        "## Input Summary",
        "",
        f"- Query split directory: `{query_embedding_root / split}`",
        f"- Query rows: `{row_count:,}`",
        f"- Query embedding dim: `{query_dim}`",
        f"- Memory rows: `{memory_rows:,}`",
        f"- Memory embedding dim: `{memory_dim}`",
        f"- Index loaded from disk: `{str(index_details['loaded_from_disk']).lower()}`",
        f"- Index rebuilt from embeddings: `{str(index_details['rebuilt_from_embeddings']).lower()}`",
        "",
        "## Applied Configuration",
        "",
        f"- Frozen `k`: `{applied_k}`",
        f"- Frozen `tau`: `{applied_tau:g}`",
        f"- Threshold source: `{memory_selection_root / 'best_val_metrics.json'}`",
        "",
        "## Test Metrics",
        "",
        f"- Test macro AUROC: `{format_metric(test_metrics['macro_auroc'])}`",
        f"- Test macro average precision: `{format_metric(test_metrics['macro_average_precision'])}`",
        f"- Test macro ECE: `{format_metric(test_metrics['macro_ece'])}`",
        f"- Test macro F1 @ 0.5: `{format_metric(test_metrics['macro_f1_at_0.5'])}`",
        f"- Test macro F1 @ frozen val thresholds: `{format_metric(test_metrics['macro_f1_at_frozen_thresholds'])}`",
        "",
        "## Query Normalization",
        "",
        f"- Raw norm mean: `{norm_summary_before['mean']:.8f}`",
        f"- Post-normalization norm mean: `{norm_summary_after['mean']:.8f}`",
        "",
        "## Expected Outputs",
        "",
        "- `experiment_meta.json`",
        "- `recreation_report.md`",
        "- `applied_config.json`",
        "- `test_metrics.json`",
        "- `test_probabilities.npy`",
        "- `qualitative_neighbors.json`",
        "- `memory_only_test_summary.md`",
        "",
        "## Output Sizes",
        "",
    ]
    lines.extend(size_lines)
    lines.extend(
        [
            f"- Total output size: `{source_memory_mod.human_size(total_size)}`",
            "",
            "## Final Artifact SHA-256",
            "",
            "```text",
            "\n".join(hash_lines),
            "```",
            "",
            "## Important Reproduction Notes",
            "",
            "- This test stage does not sweep `k` or `tau`; it reuses the validation-selected configuration from `exp0009`.",
            "- Threshold-based F1 on test uses frozen thresholds from the validation artifact, not thresholds retuned on test.",
            "- `test_probabilities.npy` stores held-out test probabilities in test row order and is small enough for plain Git.",
            "",
            "## Agent Handoff Text",
            "",
            "```text",
            (
                "Use /workspace/scripts/08_evaluate_source_memory_test.py and the report "
                f"{experiment_dir / 'recreation_report.md'} to recreate the held-out test memory evaluation for "
                f"{memory_root}. Apply the frozen validation-selected k/tau from {memory_selection_root} on the test "
                "split, reuse the validation thresholds from exp0009 for threshold-based F1 reporting, and verify the "
                "saved applied_config.json, test_metrics.json, and test_probabilities.npy artifacts."
            ),
            "```",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out test memory probabilities using frozen validation-selected hyperparameters."
    )
    parser.add_argument("--memory-root", type=Path, default=DEFAULT_MEMORY_ROOT)
    parser.add_argument("--memory-selection-root", type=Path, default=DEFAULT_MEMORY_SELECTION_ROOT)
    parser.add_argument("--query-embedding-root", type=Path, default=DEFAULT_QUERY_EMBEDDING_ROOT)
    parser.add_argument("--baseline-experiment-dir", type=Path, default=DEFAULT_BASELINE_EXPERIMENT_DIR)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["test"])
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--qualitative-queries", type=int, default=DEFAULT_QUALITATIVE_QUERIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    memory_root = args.memory_root.resolve()
    memory_selection_root = args.memory_selection_root.resolve()
    query_embedding_root = args.query_embedding_root.resolve()
    baseline_experiment_dir = args.baseline_experiment_dir.resolve()
    manifest_csv = args.manifest_csv.resolve()
    experiments_root = args.experiments_root.resolve()
    split = args.split

    generated_slug = "nih_cxr14_exp0009_test"
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=args.overwrite,
        id_width=DEFAULT_EXPERIMENT_ID_WIDTH,
    )

    script_path = Path(__file__).resolve()
    script_hash = source_memory_mod.script_sha256(script_path)

    label_columns, label_names, manifest_records = source_memory_mod.load_manifest_records(manifest_csv, split=split)
    selection_meta = source_memory_mod.read_json(memory_selection_root / "experiment_meta.json")
    selection_config = source_memory_mod.read_json(memory_selection_root / "best_config.json")
    selection_metrics = source_memory_mod.read_json(memory_selection_root / "best_val_metrics.json")
    selection_threshold_payload = selection_metrics.get("thresholds")
    if not isinstance(selection_threshold_payload, dict):
        raise SystemExit(f"Validation threshold payload is missing from {memory_selection_root / 'best_val_metrics.json'}")
    frozen_thresholds, frozen_threshold_payload = extract_thresholds(selection_threshold_payload, label_names)
    applied_k = int(selection_config["k"])
    applied_tau = float(selection_config["tau"])

    train_labels = source_memory_mod.load_embedding_array(memory_root / "labels.npy")
    train_row_ids = source_memory_mod.read_json(memory_root / "row_ids.json")
    if not isinstance(train_row_ids, list) or not train_row_ids:
        raise SystemExit(f"Invalid row_ids.json in {memory_root}")
    train_row_ids = [str(item) for item in train_row_ids]
    train_image_paths = source_memory_mod.read_lines(memory_root / "image_paths.txt")
    memory_meta = source_memory_mod.read_json(memory_root / "experiment_meta.json")
    train_embeddings, train_embedding_details = source_memory_mod.load_memory_embeddings(memory_root, memory_meta)
    source_memory_mod.check_memory_consistency(train_embeddings, train_labels, train_row_ids, train_image_paths)

    index, index_details = source_memory_mod.load_faiss_index(memory_root / "index.faiss", train_embeddings)
    if int(index.ntotal) != train_embeddings.shape[0]:
        raise SystemExit(
            f"FAISS index row count {int(index.ntotal)} does not match train embeddings {train_embeddings.shape[0]}."
        )
    if int(index.ntotal) < applied_k:
        raise SystemExit(
            f"Memory contains only {int(index.ntotal)} rows, which is less than frozen k={applied_k}."
        )

    test_embeddings, test_row_ids, test_sidecar_image_paths, query_run_meta = source_memory_mod.load_query_split(
        query_embedding_root,
        split=split,
    )
    test_labels, test_manifest_image_paths = source_memory_mod.build_labels_from_records(test_row_ids, manifest_records)
    source_memory_mod.validate_query_alignment(test_row_ids, test_sidecar_image_paths, test_manifest_image_paths)
    normalized_test_embeddings, norm_summary_before, norm_summary_after = source_memory_mod.normalize_rows(test_embeddings)

    neighbor_scores, neighbor_indices = index.search(
        np.ascontiguousarray(normalized_test_embeddings.astype(np.float32)),
        applied_k,
    )
    test_probabilities = source_memory_mod.compute_memory_probabilities(
        neighbor_indices,
        neighbor_scores,
        train_labels,
        k=applied_k,
        tau=applied_tau,
    )
    test_metrics = evaluate_probabilities_with_frozen_thresholds(
        test_labels,
        test_probabilities,
        label_names,
        frozen_thresholds,
        frozen_threshold_payload,
    )

    selected_queries = source_memory_mod.choose_qualitative_query_indices(test_labels, args.qualitative_queries)
    qualitative_neighbors = source_memory_mod.build_qualitative_neighbors(
        selected_queries,
        test_row_ids,
        test_manifest_image_paths,
        test_labels,
        neighbor_indices[:, :applied_k],
        neighbor_scores[:, :applied_k],
        train_row_ids,
        train_labels,
        label_names,
    )

    experiment_meta_path = experiment_dir / "experiment_meta.json"
    recreation_report_path = experiment_dir / "recreation_report.md"
    applied_config_path = experiment_dir / "applied_config.json"
    test_metrics_path = experiment_dir / "test_metrics.json"
    test_probabilities_path = experiment_dir / "test_probabilities.npy"
    qualitative_neighbors_path = experiment_dir / "qualitative_neighbors.json"
    summary_path = experiment_dir / "memory_only_test_summary.md"

    source_memory_mod.write_json(
        applied_config_path,
        {
            "selection_mode": FROZEN_SELECTION_MODE,
            "selection_root": str(memory_selection_root),
            "selection_metric": selection_config.get("selection_metric"),
            "selection_trace": selection_config.get("selection_trace"),
            "k": applied_k,
            "tau": applied_tau,
            "threshold_source": str(memory_selection_root / "best_val_metrics.json"),
        },
    )
    source_memory_mod.write_json(test_metrics_path, test_metrics)
    np.save(test_probabilities_path, test_probabilities.astype(np.float32))
    source_memory_mod.write_json(qualitative_neighbors_path, qualitative_neighbors)

    summary_lines = [
        "# Memory-Only Test Evaluation",
        "",
        "The held-out test memory-only evaluation for the current source memory stage uses the frozen validation-selected configuration:",
        "",
        f"- selection root: `{memory_selection_root}`",
        f"- k: `{applied_k}`",
        f"- tau: `{applied_tau:g}`",
        f"- test macro AUROC: `{format_metric(test_metrics['macro_auroc'])}`",
        f"- test macro average precision: `{format_metric(test_metrics['macro_average_precision'])}`",
        f"- test macro ECE: `{format_metric(test_metrics['macro_ece'])}`",
        f"- test macro F1 @ 0.5: `{format_metric(test_metrics['macro_f1_at_0.5'])}`",
        f"- test macro F1 @ frozen val thresholds: `{format_metric(test_metrics['macro_f1_at_frozen_thresholds'])}`",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    experiment_meta = {
        "argv": sys.argv,
        "baseline_experiment_dir": str(baseline_experiment_dir),
        "experiment_dir": str(experiment_dir),
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_number": experiment_number,
        "index_details": index_details,
        "label_columns": label_columns,
        "label_names": label_names,
        "manifest_csv": str(manifest_csv),
        "memory_root": str(memory_root),
        "memory_selection_root": str(memory_selection_root),
        "memory_selection_reference": selection_meta,
        "memory_source_embeddings": train_embedding_details,
        "memory_summary": {
            "row_count": int(train_embeddings.shape[0]),
            "embedding_dim": int(train_embeddings.shape[1]),
        },
        "operation_label": DEFAULT_OPERATION_LABEL,
        "query_embedding_root": str(query_embedding_root),
        "query_split": {
            "embedding_dim": int(test_embeddings.shape[1]),
            "num_rows": int(test_embeddings.shape[0]),
            "split": split,
            "run_meta": query_run_meta,
        },
        "query_norm_summary": {
            "raw": norm_summary_before,
            "normalized": norm_summary_after,
        },
        "run_date_utc": source_memory_mod.utc_now_iso(),
        "script_path": str(script_path),
        "script_sha256": script_hash,
        "seed": args.seed,
        "selection_mode": FROZEN_SELECTION_MODE,
        "applied_config": {
            "k": applied_k,
            "tau": applied_tau,
        },
        "frozen_threshold_source": {
            "path": str(memory_selection_root / "best_val_metrics.json"),
            "thresholds": frozen_threshold_payload,
        },
        "test_metrics": test_metrics,
        "artifacts": {
            "experiment_meta": str(experiment_meta_path),
            "recreation_report": str(recreation_report_path),
            "applied_config": str(applied_config_path),
            "test_metrics": str(test_metrics_path),
            "test_probabilities": str(test_probabilities_path),
            "qualitative_neighbors": str(qualitative_neighbors_path),
            "summary": str(summary_path),
        },
    }
    source_memory_mod.write_json(experiment_meta_path, experiment_meta)

    argv_exact = [
        "python",
        str(script_path),
        "--memory-root",
        str(memory_root),
        "--memory-selection-root",
        str(memory_selection_root),
        "--query-embedding-root",
        str(query_embedding_root),
        "--baseline-experiment-dir",
        str(baseline_experiment_dir),
        "--manifest-csv",
        str(manifest_csv),
        "--split",
        split,
        "--qualitative-queries",
        str(args.qualitative_queries),
        "--seed",
        str(args.seed),
        "--experiment-name",
        experiment_name,
        "--overwrite",
    ]
    argv_fresh = [
        "python",
        str(script_path),
        "--memory-root",
        str(memory_root),
        "--memory-selection-root",
        str(memory_selection_root),
        "--query-embedding-root",
        str(query_embedding_root),
        "--baseline-experiment-dir",
        str(baseline_experiment_dir),
        "--manifest-csv",
        str(manifest_csv),
        "--split",
        split,
        "--qualitative-queries",
        str(args.qualitative_queries),
        "--seed",
        str(args.seed),
        "--experiment-name",
        source_memory_mod.strip_experiment_number_prefix(experiment_name),
    ]
    output_paths = [
        experiment_meta_path,
        recreation_report_path,
        applied_config_path,
        test_metrics_path,
        test_probabilities_path,
        qualitative_neighbors_path,
        summary_path,
    ]
    recreation_report = build_recreation_report(
        experiment_dir=experiment_dir,
        experiment_id=experiment_id,
        operation_label=DEFAULT_OPERATION_LABEL,
        script_path=script_path,
        argv_exact=argv_exact,
        argv_fresh=argv_fresh,
        memory_root=memory_root,
        memory_selection_root=memory_selection_root,
        baseline_experiment_dir=baseline_experiment_dir,
        query_embedding_root=query_embedding_root,
        manifest_csv=manifest_csv,
        split=split,
        row_count=int(test_embeddings.shape[0]),
        memory_rows=int(train_embeddings.shape[0]),
        memory_dim=int(train_embeddings.shape[1]),
        query_dim=int(test_embeddings.shape[1]),
        label_names=label_names,
        applied_k=applied_k,
        applied_tau=applied_tau,
        norm_summary_before=norm_summary_before,
        norm_summary_after=norm_summary_after,
        index_details=index_details,
        test_metrics=test_metrics,
        output_paths=output_paths,
    )
    recreation_report_path.write_text(recreation_report + "\n", encoding="utf-8")

    print(f"[saved] experiment_dir={experiment_dir}")
    print(
        "[applied_config] "
        f"k={applied_k} tau={applied_tau:g} "
        f"macro_auroc={format_metric(test_metrics['macro_auroc'])} "
        f"macro_ap={format_metric(test_metrics['macro_average_precision'])} "
        f"macro_ece={format_metric(test_metrics['macro_ece'])} "
        f"macro_f1_0p5={format_metric(test_metrics['macro_f1_at_0.5'])} "
        f"macro_f1_frozen={format_metric(test_metrics['macro_f1_at_frozen_thresholds'])}"
    )
    print(
        "[query_norms] "
        f"raw_mean={norm_summary_before['mean']:.8f} "
        f"post_mean={norm_summary_after['mean']:.8f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
