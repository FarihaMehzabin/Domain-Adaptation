"""Run the NIH -> CheXpert top-k sparse correction seed sweep."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper_v1.analysis.postmortem import (
    active_label_fraction_row,
    aggregate_metric_rows,
    collect_internal_outputs,
    per_label_auroc_rows,
    per_label_value_rows,
    prediction_dict_from_artifact,
    selection_count_distribution_rows,
    summarize_result_row,
    write_per_label_auroc_delta,
)
from paper_v1.analysis.seed_sweep import add_source_only_deltas, aggregate_seed_rows, seed_result_row
from paper_v1.baselines.source_only import run_source_only
from paper_v1.data.embeddings import FrozenEmbeddingDataset
from paper_v1.data.splits import SplitSelector, select_records
from paper_v1.evaluation.reporting import write_stage_report, write_summary_table
from paper_v1.models.prototype_memory import build_label_state_prototypes
from paper_v1.runners.common import collect_alignment, default_device, load_config, load_data_config, parse_config_arg, raise_for_issues
from paper_v1.training.stage_adapt import train_main_method, train_tiny_logit_correction, train_topk_labelwise_trust_region
from paper_v1.utils.io import ensure_dir, write_json
from paper_v1.utils.registry import init_run_paths
from paper_v1.utils.seeds import set_seed


def _merged_stage1_config(stage1_config: dict, config: dict, key: str) -> dict[str, Any]:
    merged = dict(stage1_config.get(key, {}))
    merged.update(config.get(key, {}))
    return merged


def _per_label_delta_rows(
    *,
    method_name: str,
    seed: int,
    source_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    label_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alias, domain_name in (("d0_nih_test", "nih"), ("d1_chexpert_test", "chexpert")):
        source_predictions = prediction_dict_from_artifact(source_metrics[alias]["prediction_artifact"])
        candidate_predictions = prediction_dict_from_artifact(candidate_metrics[alias]["prediction_artifact"])
        source_rows = per_label_auroc_rows(source_predictions, label_names=label_names)
        candidate_rows = per_label_auroc_rows(candidate_predictions, label_names=label_names)
        for source_row, candidate_row in zip(source_rows, candidate_rows):
            source_auroc = source_row["auroc"]
            candidate_auroc = candidate_row["auroc"]
            rows.append(
                {
                    "method": method_name,
                    "seed": seed,
                    "domain": domain_name,
                    "label_index": source_row["label_index"],
                    "label_name": source_row["label_name"],
                    "source_only_auroc": source_auroc,
                    "candidate_auroc": candidate_auroc,
                    "delta_auroc": (
                        float(candidate_auroc - source_auroc)
                        if source_auroc is not None and candidate_auroc is not None
                        else None
                    ),
                }
            )
    return rows


def _collect_topk_diagnostic_rows(
    *,
    method_name: str,
    seed: int,
    model,
    datasets: dict[str, Any],
    batch_size: int,
    device,
    label_names: list[str],
    active_threshold: float,
) -> dict[str, Any]:
    selection_rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    for alias, dataset in datasets.items():
        domain_name = alias.replace("_test", "")
        internals = collect_internal_outputs(model, dataset, batch_size=batch_size, device=device)
        selection_rows.extend(
            per_label_value_rows(
                internals["gate"],
                label_names=label_names,
                method=method_name,
                domain_name=domain_name,
                metric_name="selection_frequency",
                seed=seed,
            )
        )
        active_rows.append(
            active_label_fraction_row(
                internals["gate"],
                method=method_name,
                domain_name=domain_name,
                threshold=active_threshold,
                seed=seed,
            )
        )
        count_rows.extend(
            selection_count_distribution_rows(
                internals["gate"],
                method=method_name,
                domain_name=domain_name,
                seed=seed,
            )
        )
    return {
        "selection_rows": selection_rows,
        "active_rows": active_rows,
        "count_rows": count_rows,
    }


def _candidate_evaluations(
    aggregate_rows: list[dict[str, Any]],
    *,
    candidates: list[str],
    max_mean_forgetting_delta_vs_reference: float,
    max_seen_average_std: float,
    max_chexpert_std: float,
) -> dict[str, Any]:
    by_method = {row["method"]: row for row in aggregate_rows}
    source_row = by_method["source_only"]
    harder_row = by_method["harder_gate_clipping"]
    evaluations: dict[str, Any] = {}
    for candidate in candidates:
        candidate_row = by_method[candidate]
        criteria = {
            "mean_seen_average_above_source_only": bool(
                candidate_row["seen_average_macro_auroc_mean"] > source_row["seen_average_macro_auroc_mean"]
            ),
            "mean_seen_average_above_harder_gate": bool(
                candidate_row["seen_average_macro_auroc_mean"] > harder_row["seen_average_macro_auroc_mean"]
            ),
            "mean_chexpert_above_source_only": bool(
                candidate_row["d1_chexpert_test_macro_auroc_mean"] > source_row["d1_chexpert_test_macro_auroc_mean"]
            ),
            "mean_forgetting_close_to_harder_gate": bool(
                candidate_row["nih_forgetting_macro_auroc_mean"]
                <= harder_row["nih_forgetting_macro_auroc_mean"] + max_mean_forgetting_delta_vs_reference
            ),
            "seen_average_variance_not_crazy": bool(
                candidate_row["seen_average_macro_auroc_std"] <= max_seen_average_std
            ),
            "chexpert_variance_not_crazy": bool(candidate_row["d1_chexpert_test_macro_auroc_std"] <= max_chexpert_std),
        }
        evaluations[candidate] = {
            "candidate_summary": candidate_row,
            "criteria": criteria,
            "promote_to_stage2_candidate": bool(all(criteria.values())),
        }
    best_candidate = max(candidates, key=lambda name: by_method[name]["seen_average_macro_auroc_mean"])
    return {
        "best_topk_candidate": best_candidate,
        "source_only_reference": source_row,
        "harder_gate_reference": harder_row,
        "candidate_evaluations": evaluations,
    }


def run(config: dict) -> dict[str, Any]:
    stage1_config = load_config(config["stage1_config_path"])
    set_seed(int(config["seeds"][0]))
    data_config = load_data_config(stage1_config["data_config"])
    manifest, alignment = collect_alignment(
        data_config=data_config,
        domains={"d0_nih", "d1_chexpert"},
        splits={"train", "val", "test"},
        missing_embedding_policy=str(stage1_config["validation"].get("missing_embedding_policy", "error")),
        require_patient_disjoint=bool(stage1_config["validation"].get("require_patient_disjoint", True)),
    )
    raise_for_issues(alignment.issues, allowed_codes=set(stage1_config["validation"].get("allowed_issue_codes", [])))

    output_paths = init_run_paths(config["output_root"], config["experiment_name"])
    device = default_device()
    stage0_checkpoint = Path(stage1_config["stage0_checkpoint"])
    base_training = _merged_stage1_config(stage1_config, config, "training")
    base_main_method = _merged_stage1_config(stage1_config, config, "main_method")
    harder_gate_config = dict(base_main_method)
    harder_gate_config.update(config["harder_gate_clipping"])
    topk_k1_config = dict(base_main_method)
    topk_k1_config.update(config["topk_labelwise_trust_region_k1"])
    topk_k2_config = dict(base_main_method)
    topk_k2_config.update(config["topk_labelwise_trust_region_k2"])
    tiny_config = dict(config["tiny_logit_correction"])
    diagnostic_seed = int(config.get("diagnostic_seed", config["seeds"][0]))
    active_threshold = float(config.get("diagnostic_active_threshold", 0.5))

    source_train_dataset = FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d0_nih", "train")]))
    train_dataset = FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d1_chexpert", "train")]))
    val_dataset = FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d1_chexpert", "val")]))
    eval_datasets = {
        "d0_nih_test": FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d0_nih", "test")])),
        "d1_chexpert_test": FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d1_chexpert", "test")])),
    }
    diagnostics_eval_datasets = {
        "nih_test": eval_datasets["d0_nih_test"],
        "chexpert_test": eval_datasets["d1_chexpert_test"],
    }
    source_embeddings, source_targets, _ = source_train_dataset.materialize_numpy()
    label_names = list(source_train_dataset.label_space.names)

    per_seed_rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    auroc_delta_rows: list[dict[str, Any]] = []
    representative_source_metrics: dict[str, Any] | None = None
    representative_topk_k1_result: dict[str, Any] | None = None
    representative_topk_k2_result: dict[str, Any] | None = None

    for seed in [int(value) for value in config["seeds"]]:
        set_seed(seed)
        seed_root = ensure_dir(output_paths.root / f"seed_{seed}")
        source_bank = build_label_state_prototypes(
            source_embeddings,
            source_targets,
            domain="d0_nih",
            positive_k=int(stage1_config["memory"].get("positive_k", 4)),
            negative_k=int(stage1_config["memory"].get("negative_k", 2)),
            seed=seed,
        )

        source_only_paths = init_run_paths(seed_root, "source_only")
        source_only_metrics = run_source_only(
            checkpoint_path=stage0_checkpoint,
            eval_datasets=eval_datasets,
            output_dir=source_only_paths.root,
            batch_size=int(base_training.get("batch_size", 256)),
            device=device,
        )
        source_row = summarize_result_row(
            "source_only",
            source_only_metrics,
            source_nih_auroc=float(source_only_metrics["d0_nih_test"]["macro_auroc"]),
        )
        source_nih_auroc = float(source_row["d0_nih_test_macro_auroc"])
        per_seed_rows.append(seed_result_row("source_only", seed, source_row))

        harder_paths = init_run_paths(seed_root, "harder_gate_clipping")
        harder_result = train_main_method(
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            old_bank=source_bank,
            output_dir=harder_paths.root,
            training_config=harder_gate_config,
            device=device,
            seed=seed,
        )
        per_seed_rows.append(
            seed_result_row(
                "harder_gate_clipping",
                seed,
                summarize_result_row("harder_gate_clipping", harder_result["metrics"], source_nih_auroc=source_nih_auroc),
            )
        )
        auroc_delta_rows.extend(
            _per_label_delta_rows(
                method_name="harder_gate_clipping",
                seed=seed,
                source_metrics=source_only_metrics,
                candidate_metrics=harder_result["metrics"],
                label_names=label_names,
            )
        )

        topk_k1_paths = init_run_paths(seed_root, "topk_labelwise_trust_region_k1")
        topk_k1_result = train_topk_labelwise_trust_region(
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            old_bank=source_bank,
            output_dir=topk_k1_paths.root,
            training_config=topk_k1_config,
            device=device,
            seed=seed,
        )
        per_seed_rows.append(
            seed_result_row(
                "topk_labelwise_trust_region_k1",
                seed,
                summarize_result_row(
                    "topk_labelwise_trust_region_k1",
                    topk_k1_result["metrics"],
                    source_nih_auroc=source_nih_auroc,
                ),
            )
        )
        auroc_delta_rows.extend(
            _per_label_delta_rows(
                method_name="topk_labelwise_trust_region_k1",
                seed=seed,
                source_metrics=source_only_metrics,
                candidate_metrics=topk_k1_result["metrics"],
                label_names=label_names,
            )
        )
        topk_k1_diagnostics = _collect_topk_diagnostic_rows(
            method_name="topk_labelwise_trust_region_k1",
            seed=seed,
            model=topk_k1_result["model"],
            datasets=diagnostics_eval_datasets,
            batch_size=int(topk_k1_config.get("batch_size", 256)),
            device=device,
            label_names=label_names,
            active_threshold=active_threshold,
        )
        selection_rows.extend(topk_k1_diagnostics["selection_rows"])
        active_rows.extend(topk_k1_diagnostics["active_rows"])
        count_rows.extend(topk_k1_diagnostics["count_rows"])

        topk_k2_paths = init_run_paths(seed_root, "topk_labelwise_trust_region_k2")
        topk_k2_result = train_topk_labelwise_trust_region(
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            old_bank=source_bank,
            output_dir=topk_k2_paths.root,
            training_config=topk_k2_config,
            device=device,
            seed=seed,
        )
        per_seed_rows.append(
            seed_result_row(
                "topk_labelwise_trust_region_k2",
                seed,
                summarize_result_row(
                    "topk_labelwise_trust_region_k2",
                    topk_k2_result["metrics"],
                    source_nih_auroc=source_nih_auroc,
                ),
            )
        )
        auroc_delta_rows.extend(
            _per_label_delta_rows(
                method_name="topk_labelwise_trust_region_k2",
                seed=seed,
                source_metrics=source_only_metrics,
                candidate_metrics=topk_k2_result["metrics"],
                label_names=label_names,
            )
        )
        topk_k2_diagnostics = _collect_topk_diagnostic_rows(
            method_name="topk_labelwise_trust_region_k2",
            seed=seed,
            model=topk_k2_result["model"],
            datasets=diagnostics_eval_datasets,
            batch_size=int(topk_k2_config.get("batch_size", 256)),
            device=device,
            label_names=label_names,
            active_threshold=active_threshold,
        )
        selection_rows.extend(topk_k2_diagnostics["selection_rows"])
        active_rows.extend(topk_k2_diagnostics["active_rows"])
        count_rows.extend(topk_k2_diagnostics["count_rows"])

        tiny_paths = init_run_paths(seed_root, "tiny_logit_correction")
        tiny_result = train_tiny_logit_correction(
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            output_dir=tiny_paths.root,
            training_config=tiny_config,
            device=device,
            seed=seed,
        )
        per_seed_rows.append(
            seed_result_row(
                "tiny_logit_correction",
                seed,
                summarize_result_row("tiny_logit_correction", tiny_result["metrics"], source_nih_auroc=source_nih_auroc),
            )
        )
        auroc_delta_rows.extend(
            _per_label_delta_rows(
                method_name="tiny_logit_correction",
                seed=seed,
                source_metrics=source_only_metrics,
                candidate_metrics=tiny_result["metrics"],
                label_names=label_names,
            )
        )

        if seed == diagnostic_seed:
            representative_source_metrics = source_only_metrics
            representative_topk_k1_result = topk_k1_result
            representative_topk_k2_result = topk_k2_result

    seed_results_csv = write_summary_table(output_paths.reports, "seed_results.csv", per_seed_rows)
    aggregate_rows = add_source_only_deltas(aggregate_seed_rows(per_seed_rows))
    aggregate_rows = sorted(aggregate_rows, key=lambda row: row["seen_average_macro_auroc_mean"], reverse=True)
    aggregate_csv = write_summary_table(output_paths.reports, "seed_summary.csv", aggregate_rows)

    active_csv = write_summary_table(output_paths.reports, "active_label_fraction.csv", active_rows)
    active_aggregate_csv = write_summary_table(
        output_paths.reports,
        "active_label_fraction_aggregate.csv",
        aggregate_metric_rows(
            active_rows,
            group_keys=["method", "domain", "threshold"],
            value_keys=["mean_active_fraction", "std_active_fraction"],
        ),
    )
    selection_csv = write_summary_table(output_paths.reports, "selection_frequency_by_label.csv", selection_rows)
    selection_aggregate_csv = write_summary_table(
        output_paths.reports,
        "selection_frequency_by_label_aggregate.csv",
        aggregate_metric_rows(
            selection_rows,
            group_keys=["method", "domain", "metric", "label_index", "label_name"],
            value_keys=["mean", "std"],
        ),
    )
    count_csv = write_summary_table(output_paths.reports, "selected_label_count_distribution.csv", count_rows)
    count_aggregate_csv = write_summary_table(
        output_paths.reports,
        "selected_label_count_distribution_aggregate.csv",
        aggregate_metric_rows(
            count_rows,
            group_keys=["method", "domain", "selected_label_count"],
            value_keys=["fraction"],
        ),
    )
    auroc_delta_csv = write_summary_table(output_paths.reports, "per_label_auroc_delta.csv", auroc_delta_rows)
    auroc_delta_aggregate_csv = write_summary_table(
        output_paths.reports,
        "per_label_auroc_delta_aggregate.csv",
        aggregate_metric_rows(
            auroc_delta_rows,
            group_keys=["method", "domain", "label_index", "label_name"],
            value_keys=["delta_auroc"],
        ),
    )

    promotion = _candidate_evaluations(
        aggregate_rows,
        candidates=["topk_labelwise_trust_region_k1", "topk_labelwise_trust_region_k2"],
        max_mean_forgetting_delta_vs_reference=float(
            config["promotion"].get("max_mean_forgetting_delta_vs_reference", 0.002)
        ),
        max_seen_average_std=float(config["promotion"]["max_seen_average_std"]),
        max_chexpert_std=float(config["promotion"]["max_chexpert_std"]),
    )
    promotion_path = write_json(output_paths.artifacts / "promotion_decision.json", promotion)

    diagnostics_dir = ensure_dir(output_paths.reports / "topk_diagnostics")
    representative_delta_paths: list[str] = []
    if representative_source_metrics is not None and representative_topk_k1_result is not None and representative_topk_k2_result is not None:
        for method_name, result in (
            ("topk_labelwise_trust_region_k1", representative_topk_k1_result),
            ("topk_labelwise_trust_region_k2", representative_topk_k2_result),
        ):
            for alias, domain_name in (("d0_nih_test", "nih"), ("d1_chexpert_test", "chexpert")):
                delta_path = write_per_label_auroc_delta(
                    diagnostics_dir / f"{method_name}__{domain_name}__auroc_delta_vs_source_only.csv",
                    source_predictions=prediction_dict_from_artifact(
                        representative_source_metrics[alias]["prediction_artifact"]
                    ),
                    candidate_predictions=prediction_dict_from_artifact(result["metrics"][alias]["prediction_artifact"]),
                    label_names=label_names,
                    domain_name=domain_name,
                    source_name="source_only",
                    candidate_name=method_name,
                )
                representative_delta_paths.append(str(delta_path))

    sections = [
        ("Manifest", [f"- `{manifest.path}`"]),
        (
            "Comparison Set",
            [
                "- `source_only`",
                "- `harder_gate_clipping`",
                "- `tiny_logit_correction`",
                "- `topk_labelwise_trust_region_k1`",
                "- `topk_labelwise_trust_region_k2`",
            ],
        ),
        (
            "Seed Summary",
            [
                f"- Per-seed results: `{seed_results_csv}`",
                f"- Aggregate summary: `{aggregate_csv}`",
                f"- Candidate decision: `{promotion_path}`",
            ],
        ),
        (
            "Sparse Diagnostics",
            [
                f"- Active-label fraction: `{active_csv}`",
                f"- Active-label fraction aggregate: `{active_aggregate_csv}`",
                f"- Selection frequency by label: `{selection_csv}`",
                f"- Selection frequency aggregate: `{selection_aggregate_csv}`",
                f"- Selected-label count distribution: `{count_csv}`",
                f"- Selected-label count distribution aggregate: `{count_aggregate_csv}`",
                f"- Per-label AUROC delta: `{auroc_delta_csv}`",
                f"- Per-label AUROC delta aggregate: `{auroc_delta_aggregate_csv}`",
                f"- Representative delta diagnostics: `{diagnostics_dir}`" if representative_delta_paths else "- Representative delta diagnostics: `not_run`",
            ],
        ),
    ]
    report_path = write_stage_report(
        output_paths.reports,
        "nih_to_chexpert_topk_seed_sweep_report.md",
        title="NIH -> CheXpert Top-K Seed Sweep",
        sections=sections,
    )
    return {
        "run_root": str(output_paths.root),
        "seed_results_csv": str(seed_results_csv),
        "aggregate_csv": str(aggregate_csv),
        "promotion_decision_path": str(promotion_path),
        "report_path": str(report_path),
    }


def main() -> None:
    args = parse_config_arg("Run the NIH -> CheXpert top-k sparse correction sweep.")
    config = load_config(args.config)
    result = run(config)
    print(f"[done] wrote top-k seed-sweep artifacts to {result['run_root']}")


if __name__ == "__main__":
    main()
