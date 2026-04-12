#!/usr/bin/env python3
"""Apply frozen retrieval settings to a held-out target split."""

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
    build_labels_from_records,
    build_qualitative_neighbors,
    build_simple_recreation_report,
    choose_qualitative_query_indices,
    compare_metrics_to_archived,
    compute_memory_probabilities,
    evaluate_probabilities_with_frozen_thresholds,
    extract_thresholds,
    format_metric,
    load_baseline_thresholds,
    load_embedding_array,
    load_embedding_split,
    load_manifest_records,
    normalize_rows,
    read_json,
    reconstruct_baseline_probabilities,
    resolve_experiment_identity,
    search_index,
    split_alias_from_domain_split,
    utc_now_iso,
    validate_query_alignment,
    write_json,
    baseline_metrics_path,
)


OPERATION_LABEL = "domain_transfer_source_memory_target_evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a frozen retrieval configuration to a target split.")
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument("--memory-selection-root", type=Path, required=True)
    parser.add_argument("--baseline-experiment-dir", type=Path, default=DEFAULT_BASELINE_EXPERIMENT_DIR)
    parser.add_argument("--query-embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--query-domain", type=str, required=True)
    parser.add_argument("--query-split", type=str, required=True)
    parser.add_argument("--split-alias", type=str, default=None)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--qualitative-queries", type=int, default=DEFAULT_QUALITATIVE_QUERIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    memory_root = args.memory_root.resolve()
    memory_selection_root = args.memory_selection_root.resolve()
    baseline_experiment_dir = args.baseline_experiment_dir.resolve()
    query_embedding_root = args.query_embedding_root.resolve()
    manifest_csv = args.manifest_csv.resolve()
    experiments_root = args.experiments_root.resolve()
    query_domain = args.query_domain.strip()
    query_split = args.query_split.strip().lower()
    split_alias = args.split_alias or split_alias_from_domain_split(query_domain, query_split)

    generated_slug = f"{memory_selection_root.name}__{split_alias}"
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

    selection_config = read_json(memory_selection_root / "best_config.json")
    selection_metrics = read_json(memory_selection_root / "best_val_metrics.json")
    selection_threshold_payload = selection_metrics.get("thresholds")
    if not isinstance(selection_threshold_payload, dict):
        raise SystemExit(f"Selection thresholds are missing from {memory_selection_root / 'best_val_metrics.json'}.")
    selection_thresholds, normalized_selection_thresholds = extract_thresholds(selection_threshold_payload, label_names)

    index = faiss.read_index(str(memory_root / "index.faiss"))
    train_labels = load_embedding_array(memory_root / "labels.npy")
    train_row_ids = json.loads((memory_root / "row_ids.json").read_text(encoding="utf-8"))
    train_image_paths_path = memory_root / "image_paths.txt"
    train_image_paths = (
        train_image_paths_path.read_text(encoding="utf-8").splitlines() if train_image_paths_path.exists() else []
    )

    applied_k = int(selection_config["k"])
    applied_tau = float(selection_config["tau"])
    neighbor_indices, neighbor_scores = search_index(index, normalized_queries, applied_k)
    evaluation_probabilities = compute_memory_probabilities(
        neighbor_indices,
        neighbor_scores,
        train_labels,
        k=applied_k,
        tau=applied_tau,
    )
    evaluation_metrics = evaluate_probabilities_with_frozen_thresholds(
        query_labels,
        evaluation_probabilities,
        label_names,
        thresholds=selection_thresholds,
        threshold_payload=normalized_selection_thresholds,
    )

    baseline_thresholds, normalized_baseline_thresholds = load_baseline_thresholds(baseline_experiment_dir, label_names)
    baseline_probabilities, baseline_reconstruction = reconstruct_baseline_probabilities(
        checkpoint_path=baseline_experiment_dir / "best.ckpt",
        normalized_embeddings=normalized_queries,
        label_names=label_names,
        batch_size=int(args.batch_size),
    )
    baseline_metrics = evaluate_probabilities_with_frozen_thresholds(
        query_labels,
        baseline_probabilities,
        label_names,
        thresholds=baseline_thresholds,
        threshold_payload=normalized_baseline_thresholds,
    )
    archived_baseline_metrics = read_json(baseline_metrics_path(baseline_experiment_dir, split_alias))
    baseline_comparison = compare_metrics_to_archived(baseline_metrics, archived_baseline_metrics, label_names)

    qualitative_indices = choose_qualitative_query_indices(query_labels, int(args.qualitative_queries))
    qualitative_neighbors = build_qualitative_neighbors(
        query_indices=qualitative_indices,
        query_row_ids=query_data.row_ids,
        query_image_paths=manifest_image_paths,
        query_labels=query_labels,
        neighbor_indices=neighbor_indices,
        neighbor_scores=neighbor_scores,
        train_row_ids=train_row_ids,
        train_labels=train_labels,
        label_names=label_names,
    )

    delta_summary: dict[str, float | None] = {}
    for key in ("auroc", "average_precision", "ece", "f1_at_0.5", "f1_at_tuned_thresholds"):
        current = evaluation_metrics["macro"].get(key)
        baseline = baseline_metrics["macro"].get(key)
        delta_summary[key] = None if current is None or baseline is None else float(current - baseline)

    applied_config_path = experiment_dir / "applied_config.json"
    evaluation_metrics_path = experiment_dir / "evaluation_metrics.json"
    evaluation_probabilities_path = experiment_dir / "evaluation_probabilities.npy"
    qualitative_path = experiment_dir / "qualitative_neighbors.json"
    summary_path = experiment_dir / "evaluation_summary.md"
    experiment_meta_path = experiment_dir / "experiment_meta.json"
    recreation_report_path = experiment_dir / "recreation_report.md"

    write_json(
        applied_config_path,
        {
            "memory_root": str(memory_root),
            "memory_selection_root": str(memory_selection_root),
            "query_domain": query_domain,
            "query_split": query_split,
            "split_alias": split_alias,
            "k": applied_k,
            "tau": applied_tau,
            "threshold_source": str(memory_selection_root / "best_val_metrics.json"),
        },
    )
    write_json(evaluation_metrics_path, evaluation_metrics)
    np.save(evaluation_probabilities_path, np.asarray(evaluation_probabilities, dtype=np.float32))
    write_json(qualitative_path, qualitative_neighbors)
    write_json(
        experiment_meta_path,
        {
            "argv": sys.argv,
            "baseline_comparison": baseline_comparison,
            "baseline_delta_summary": delta_summary,
            "baseline_experiment_dir": str(baseline_experiment_dir),
            "baseline_reconstruction": baseline_reconstruction,
            "experiment_dir": str(experiment_dir),
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "label_names": label_names,
            "manifest_csv": str(manifest_csv),
            "memory_root": str(memory_root),
            "memory_selection_root": str(memory_selection_root),
            "operation_label": OPERATION_LABEL,
            "query_domain": query_domain,
            "query_split": query_split,
            "run_date_utc": utc_now_iso(),
            "split_alias": split_alias,
        },
    )

    summary_lines = [
        "# Memory-Only Target Evaluation",
        "",
        f"- Query split: `{split_alias}`",
        f"- Frozen `k`: `{applied_k}`",
        f"- Frozen `tau`: `{applied_tau}`",
        f"- Memory macro AUROC: `{format_metric(evaluation_metrics['macro']['auroc'])}`",
        f"- Memory macro average precision: `{format_metric(evaluation_metrics['macro']['average_precision'])}`",
        f"- Delta vs baseline macro AUROC: `{format_metric(delta_summary['auroc'])}`",
        f"- Delta vs baseline macro average precision: `{format_metric(delta_summary['average_precision'])}`",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    recreation_report = build_simple_recreation_report(
        experiment_dir=experiment_dir,
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        summary_lines=[
            f"- Memory selection root: `{memory_selection_root}`",
            f"- Query split: `{query_domain}/{query_split}`",
            f"- Split alias: `{split_alias}`",
            f"- Frozen k/tau: `{applied_k}` / `{applied_tau}`",
            f"- Memory macro AUROC: `{format_metric(evaluation_metrics['macro']['auroc'])}`",
        ],
    )
    recreation_report_path.write_text(recreation_report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
