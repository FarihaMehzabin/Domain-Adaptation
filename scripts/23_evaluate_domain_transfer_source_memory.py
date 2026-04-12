#!/usr/bin/env python3
"""Select retrieval hyperparameters on D0 validation for the new transfer branch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss  # type: ignore
import numpy as np

from domain_transfer_rag_common import (
    DEFAULT_BASELINE_EXPERIMENT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_ROOT,
    DEFAULT_EXPERIMENTS_ROOT,
    DEFAULT_MANIFEST_CSV,
    DEFAULT_QUALITATIVE_QUERIES,
    DEFAULT_SEED,
    DEFAULT_SELECTION_DOMAIN,
    DEFAULT_SELECTION_SPLIT,
    SWEEP_K_VALUES,
    SWEEP_TAU_VALUES,
    baseline_metrics_path,
    build_labels_from_records,
    build_qualitative_neighbors,
    build_simple_recreation_report,
    choose_qualitative_query_indices,
    compare_metrics_to_archived,
    compute_memory_probabilities,
    evaluate_probabilities,
    format_metric,
    load_embedding_array,
    load_embedding_split,
    load_manifest_records,
    normalize_rows,
    read_json,
    resolve_experiment_identity,
    reconstruct_baseline_probabilities,
    search_index,
    select_best_retrieval_row,
    split_alias_from_domain_split,
    utc_now_iso,
    validate_query_alignment,
    write_json,
)


OPERATION_LABEL = "domain_transfer_source_memory_selection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image-only source memory on a held-out validation split.")
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument("--query-embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--baseline-experiment-dir", type=Path, default=DEFAULT_BASELINE_EXPERIMENT_DIR)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--query-domain", type=str, default=DEFAULT_SELECTION_DOMAIN)
    parser.add_argument("--query-split", type=str, default=DEFAULT_SELECTION_SPLIT)
    parser.add_argument("--split-alias", type=str, default=None)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sweep-k-values", type=int, nargs="+", default=SWEEP_K_VALUES)
    parser.add_argument("--sweep-tau-values", type=float, nargs="+", default=SWEEP_TAU_VALUES)
    parser.add_argument("--qualitative-queries", type=int, default=DEFAULT_QUALITATIVE_QUERIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    memory_root = args.memory_root.resolve()
    query_embedding_root = args.query_embedding_root.resolve()
    baseline_experiment_dir = args.baseline_experiment_dir.resolve()
    manifest_csv = args.manifest_csv.resolve()
    experiments_root = args.experiments_root.resolve()
    query_domain = args.query_domain.strip()
    query_split = args.query_split.strip().lower()
    split_alias = args.split_alias or split_alias_from_domain_split(query_domain, query_split)

    generated_slug = f"{memory_root.name}__{split_alias}"
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
    normalized_queries, query_norm_summary_before, query_norm_summary_after = normalize_rows(query_data.embeddings)

    index_path = memory_root / "index.faiss"
    index = faiss.read_index(str(index_path))
    train_labels = load_embedding_array(memory_root / "labels.npy")
    train_row_ids = json.loads((memory_root / "row_ids.json").read_text(encoding="utf-8"))
    train_image_paths_path = memory_root / "image_paths.txt"
    train_image_paths = (
        train_image_paths_path.read_text(encoding="utf-8").splitlines() if train_image_paths_path.exists() else []
    )

    max_k = min(int(index.ntotal), max(int(value) for value in args.sweep_k_values))
    neighbor_indices, neighbor_scores = search_index(index, normalized_queries, max_k)

    sweep_rows: list[dict[str, float | int | None]] = []
    metrics_by_key: dict[tuple[int, float], dict[str, object]] = {}
    for k in [int(value) for value in args.sweep_k_values]:
        for tau in [float(value) for value in args.sweep_tau_values]:
            probabilities = compute_memory_probabilities(neighbor_indices, neighbor_scores, train_labels, k=k, tau=tau)
            metrics = evaluate_probabilities(query_labels, probabilities, label_names)
            metrics_by_key[(k, tau)] = {"probabilities": probabilities, "metrics": metrics}
            sweep_rows.append(
                {
                    "k": k,
                    "tau": tau,
                    "macro_auroc": metrics["macro"]["auroc"],
                    "macro_average_precision": metrics["macro"]["average_precision"],
                    "macro_ece": metrics["macro"]["ece"],
                    "macro_f1_at_0.5": metrics["macro"]["f1_at_0.5"],
                }
            )

    best_row, selection_trace = select_best_retrieval_row(sweep_rows)
    best_key = (int(best_row["k"]), float(best_row["tau"]))
    best_probabilities = metrics_by_key[best_key]["probabilities"]
    best_metrics = metrics_by_key[best_key]["metrics"]

    baseline_probabilities, baseline_reconstruction = reconstruct_baseline_probabilities(
        checkpoint_path=baseline_experiment_dir / "best.ckpt",
        normalized_embeddings=normalized_queries,
        label_names=label_names,
        batch_size=int(args.batch_size),
    )
    baseline_metrics = evaluate_probabilities(query_labels, baseline_probabilities, label_names)
    archived_baseline_metrics = read_json(baseline_metrics_path(baseline_experiment_dir, split_alias))
    baseline_comparison = compare_metrics_to_archived(baseline_metrics, archived_baseline_metrics, label_names)

    qualitative_indices = choose_qualitative_query_indices(query_labels, int(args.qualitative_queries))
    qualitative_neighbors = build_qualitative_neighbors(
        query_indices=qualitative_indices,
        query_row_ids=query_data.row_ids,
        query_image_paths=manifest_image_paths,
        query_labels=query_labels,
        neighbor_indices=neighbor_indices[:, : int(best_row["k"])],
        neighbor_scores=neighbor_scores[:, : int(best_row["k"])],
        train_row_ids=train_row_ids,
        train_labels=train_labels,
        label_names=label_names,
    )

    sweep_results_path = experiment_dir / "sweep_results.json"
    best_config_path = experiment_dir / "best_config.json"
    best_val_metrics_path = experiment_dir / "best_val_metrics.json"
    val_probabilities_path = experiment_dir / "val_probabilities.npy"
    qualitative_path = experiment_dir / "qualitative_neighbors.json"
    summary_path = experiment_dir / "memory_only_selection.md"
    experiment_meta_path = experiment_dir / "experiment_meta.json"
    recreation_report_path = experiment_dir / "recreation_report.md"

    write_json(sweep_results_path, sweep_rows)
    write_json(
        best_config_path,
        {
            "memory_root": str(memory_root),
            "query_domain": query_domain,
            "query_split": query_split,
            "split_alias": split_alias,
            "selection_metric": "macro_auroc",
            "selection_trace": selection_trace,
            "k": int(best_row["k"]),
            "tau": float(best_row["tau"]),
        },
    )
    write_json(best_val_metrics_path, best_metrics)
    np.save(val_probabilities_path, np.asarray(best_probabilities, dtype=np.float32))
    write_json(qualitative_path, qualitative_neighbors)
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
            "memory_root": str(memory_root),
            "operation_label": OPERATION_LABEL,
            "query_domain": query_domain,
            "query_norm_summary_before": query_norm_summary_before,
            "query_norm_summary_after": query_norm_summary_after,
            "query_split": query_split,
            "run_date_utc": utc_now_iso(),
            "split_alias": split_alias,
        },
    )

    summary_lines = [
        "# Memory-Only Selection",
        "",
        f"- Query split: `{split_alias}`",
        f"- Best `k`: `{int(best_row['k'])}`",
        f"- Best `tau`: `{float(best_row['tau'])}`",
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
            f"- Memory root: `{memory_root}`",
            f"- Query split: `{query_domain}/{query_split}`",
            f"- Split alias: `{split_alias}`",
            f"- Best k/tau: `{int(best_row['k'])}` / `{float(best_row['tau'])}`",
            f"- Best macro AUROC: `{format_metric(best_metrics['macro']['auroc'])}`",
        ],
    )
    recreation_report_path.write_text(recreation_report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
