#!/usr/bin/env python3
"""Select probability mixing on D0 validation for the new transfer branch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from domain_transfer_rag_common import (
    ALPHA_GRID,
    DEFAULT_BASELINE_EXPERIMENT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_ROOT,
    DEFAULT_EXPERIMENTS_ROOT,
    DEFAULT_MANIFEST_CSV,
    DEFAULT_SELECTION_DOMAIN,
    DEFAULT_SELECTION_SPLIT,
    DEFAULT_SEED,
    baseline_metrics_path,
    build_labels_from_records,
    build_simple_recreation_report,
    compare_metrics_to_archived,
    evaluate_probabilities,
    format_alpha_value,
    format_metric,
    load_embedding_split,
    load_manifest_records,
    mix_probabilities,
    normalize_rows,
    read_json,
    reconstruct_baseline_probabilities,
    resolve_experiment_identity,
    select_best_alpha_row,
    split_alias_from_domain_split,
    utc_now_iso,
    validate_query_alignment,
    write_json,
)


OPERATION_LABEL = "domain_transfer_probability_mixing_selection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation-only probability mixing for the new transfer branch.")
    parser.add_argument("--memory-eval-root", type=Path, required=True)
    parser.add_argument("--baseline-experiment-dir", type=Path, default=DEFAULT_BASELINE_EXPERIMENT_DIR)
    parser.add_argument("--query-embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--query-domain", type=str, default=DEFAULT_SELECTION_DOMAIN)
    parser.add_argument("--query-split", type=str, default=DEFAULT_SELECTION_SPLIT)
    parser.add_argument("--split-alias", type=str, default=None)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--alpha-values", type=float, nargs="+", default=ALPHA_GRID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    memory_eval_root = args.memory_eval_root.resolve()
    baseline_experiment_dir = args.baseline_experiment_dir.resolve()
    query_embedding_root = args.query_embedding_root.resolve()
    manifest_csv = args.manifest_csv.resolve()
    experiments_root = args.experiments_root.resolve()
    query_domain = args.query_domain.strip()
    query_split = args.query_split.strip().lower()
    split_alias = args.split_alias or split_alias_from_domain_split(query_domain, query_split)

    generated_slug = f"{memory_eval_root.name}__{split_alias}"
    _, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        operation_label=OPERATION_LABEL,
        overwrite=args.overwrite,
    )

    _, label_names, records = load_manifest_records(manifest_csv, domain=query_domain, split=query_split)
    query_data = load_embedding_split(query_embedding_root, domain=query_domain, split=query_split)
    query_labels, manifest_image_paths, expected_row_ids = build_labels_from_records(query_data.row_ids, records)
    validate_query_alignment(query_data.row_ids, expected_row_ids, query_data.image_paths, manifest_image_paths)
    normalized_queries, _, _ = normalize_rows(query_data.embeddings)

    memory_probabilities = np.load(memory_eval_root / "val_probabilities.npy")
    if tuple(memory_probabilities.shape) != tuple(query_labels.shape):
        raise SystemExit(
            f"Memory probability shape {memory_probabilities.shape} does not match query label shape {query_labels.shape}."
        )

    baseline_probabilities, baseline_reconstruction = reconstruct_baseline_probabilities(
        checkpoint_path=baseline_experiment_dir / "best.ckpt",
        normalized_embeddings=normalized_queries,
        label_names=label_names,
        batch_size=int(args.batch_size),
    )
    baseline_metrics = evaluate_probabilities(query_labels, baseline_probabilities, label_names)
    archived_baseline_metrics = read_json(baseline_metrics_path(baseline_experiment_dir, split_alias))
    baseline_comparison = compare_metrics_to_archived(baseline_metrics, archived_baseline_metrics, label_names)

    sweep_rows: list[dict[str, float | None]] = []
    metrics_by_alpha: dict[float, dict[str, object]] = {}
    for alpha in [float(value) for value in args.alpha_values]:
        mixed_probabilities = mix_probabilities(baseline_probabilities, memory_probabilities, alpha)
        metrics = evaluate_probabilities(query_labels, mixed_probabilities, label_names)
        metrics_by_alpha[alpha] = {"probabilities": mixed_probabilities, "metrics": metrics}
        sweep_rows.append(
            {
                "alpha": alpha,
                "macro_auroc": metrics["macro"]["auroc"],
                "macro_average_precision": metrics["macro"]["average_precision"],
                "macro_ece": metrics["macro"]["ece"],
                "macro_f1_at_0.5": metrics["macro"]["f1_at_0.5"],
            }
        )

    best_row, selection_trace = select_best_alpha_row(sweep_rows)
    best_alpha = float(best_row["alpha"])
    best_probabilities = metrics_by_alpha[best_alpha]["probabilities"]
    best_metrics = metrics_by_alpha[best_alpha]["metrics"]

    sweep_results_path = experiment_dir / "sweep_results.json"
    best_config_path = experiment_dir / "best_config.json"
    best_val_metrics_path = experiment_dir / "best_val_metrics.json"
    val_mixed_probabilities_path = experiment_dir / "val_mixed_probabilities.npy"
    summary_path = experiment_dir / "probability_mixing_selection.md"
    experiment_meta_path = experiment_dir / "experiment_meta.json"
    recreation_report_path = experiment_dir / "recreation_report.md"

    write_json(sweep_results_path, sweep_rows)
    write_json(
        best_config_path,
        {
            "memory_eval_root": str(memory_eval_root),
            "query_domain": query_domain,
            "query_split": query_split,
            "split_alias": split_alias,
            "selection_metric": "macro_auroc",
            "selection_trace": selection_trace,
            "alpha": best_alpha,
        },
    )
    write_json(best_val_metrics_path, best_metrics)
    np.save(val_mixed_probabilities_path, np.asarray(best_probabilities, dtype=np.float32))
    write_json(
        experiment_meta_path,
        {
            "argv": sys.argv,
            "baseline_comparison": baseline_comparison,
            "baseline_experiment_dir": str(baseline_experiment_dir),
            "baseline_reconstruction": baseline_reconstruction,
            "experiment_dir": str(experiment_dir),
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "label_names": label_names,
            "manifest_csv": str(manifest_csv),
            "memory_eval_root": str(memory_eval_root),
            "operation_label": OPERATION_LABEL,
            "query_domain": query_domain,
            "query_split": query_split,
            "run_date_utc": utc_now_iso(),
            "split_alias": split_alias,
        },
    )

    summary_lines = [
        "# Probability-Mixing Selection",
        "",
        f"- Query split: `{split_alias}`",
        f"- Best `alpha`: `{format_alpha_value(best_alpha)}`",
        f"- Best macro AUROC: `{format_metric(best_metrics['macro']['auroc'])}`",
        f"- Best macro average precision: `{format_metric(best_metrics['macro']['average_precision'])}`",
        f"- Baseline macro AUROC: `{format_metric(baseline_metrics['macro']['auroc'])}`",
        f"- Baseline reconstruction matches archived metrics within `5e-4`: "
        f"`{str(baseline_comparison['matches_archived_metrics_within_5e-4']).lower()}`",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    recreation_report = build_simple_recreation_report(
        experiment_dir=experiment_dir,
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        summary_lines=[
            f"- Memory evaluation root: `{memory_eval_root}`",
            f"- Query split: `{query_domain}/{query_split}`",
            f"- Split alias: `{split_alias}`",
            f"- Best alpha: `{format_alpha_value(best_alpha)}`",
            f"- Best macro AUROC: `{format_metric(best_metrics['macro']['auroc'])}`",
        ],
    )
    recreation_report_path.write_text(recreation_report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
