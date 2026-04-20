"""Run the limited NIH -> CheXpert seed sweep and representative gate ablations."""

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
    summarize_result_row,
    write_histogram_csv,
    write_histogram_summary,
    write_per_label_auroc_delta,
)
from paper_v1.analysis.seed_sweep import add_source_only_deltas, aggregate_seed_rows, seed_result_row
from paper_v1.baselines.lwf import run_lwf
from paper_v1.baselines.source_only import run_source_only
from paper_v1.baselines.vq_summary_replay import run_vq_summary_replay
from paper_v1.data.embeddings import FrozenEmbeddingDataset
from paper_v1.data.splits import SplitSelector, select_records
from paper_v1.evaluation.reporting import write_stage_report, write_summary_table
from paper_v1.models.prototype_memory import PrototypeBank, build_label_state_prototypes, build_vq_summary_bank
from paper_v1.runners.common import collect_alignment, default_device, load_config, load_data_config, parse_config_arg, raise_for_issues
from paper_v1.training.stage_adapt import train_labelwise_trust_region, train_main_method, train_tiny_logit_correction
from paper_v1.utils.io import ensure_dir, write_json
from paper_v1.utils.registry import init_run_paths
from paper_v1.utils.seeds import set_seed


def _merged_stage1_config(stage1_config: dict, config: dict, key: str) -> dict[str, Any]:
    merged = dict(stage1_config.get(key, {}))
    merged.update(config.get(key, {}))
    return merged


def _write_internal_diagnostics(
    *,
    diagnostics_dir: Path,
    prefix: str,
    model,
    datasets: dict[str, Any],
    batch_size: int,
    device,
) -> None:
    for alias, dataset in datasets.items():
        internals = collect_internal_outputs(model, dataset, batch_size=batch_size, device=device)
        write_histogram_csv(diagnostics_dir / f"{prefix}__{alias}__gate_histogram.csv", internals["gate"], bins=20, min_value=0.0, max_value=1.0)
        write_histogram_summary(diagnostics_dir / f"{prefix}__{alias}__gate_summary.json", internals["gate"])
        write_histogram_csv(diagnostics_dir / f"{prefix}__{alias}__residual_norm_histogram.csv", internals["residual_norm"], bins=20)
        write_histogram_summary(diagnostics_dir / f"{prefix}__{alias}__residual_norm_summary.json", internals["residual_norm"])


def _collect_diagnostic_rows(
    *,
    method_name: str,
    seed: int,
    model,
    datasets: dict[str, Any],
    batch_size: int,
    device,
    label_names: list[str],
    active_threshold: float,
    diagnostics_dir: Path | None = None,
) -> dict[str, Any]:
    gate_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    internals_by_domain: dict[str, dict[str, Any]] = {}
    for alias, dataset in datasets.items():
        internals = collect_internal_outputs(model, dataset, batch_size=batch_size, device=device)
        internals_by_domain[alias] = internals
        domain_name = alias.replace("_test", "")
        gate_rows.extend(
            per_label_value_rows(
                internals["gate"],
                label_names=label_names,
                method=method_name,
                domain_name=domain_name,
                metric_name="gate",
                seed=seed,
            )
        )
        residual_rows.extend(
            per_label_value_rows(
                internals["residual_abs"],
                label_names=label_names,
                method=method_name,
                domain_name=domain_name,
                metric_name="residual_abs",
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
        if diagnostics_dir is not None:
            write_histogram_csv(
                diagnostics_dir / f"{method_name}__{alias}__gate_histogram.csv",
                internals["gate"],
                bins=20,
                min_value=0.0,
                max_value=1.0,
            )
            write_histogram_summary(diagnostics_dir / f"{method_name}__{alias}__gate_summary.json", internals["gate"])
            write_histogram_csv(
                diagnostics_dir / f"{method_name}__{alias}__residual_norm_histogram.csv",
                internals["residual_norm"],
                bins=20,
            )
            write_histogram_summary(
                diagnostics_dir / f"{method_name}__{alias}__residual_norm_summary.json",
                internals["residual_norm"],
            )
    return {
        "gate_rows": gate_rows,
        "residual_rows": residual_rows,
        "active_rows": active_rows,
        "internals_by_domain": internals_by_domain,
    }


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


def _pilot_promotion_decision(
    aggregate_rows: list[dict[str, Any]],
    *,
    candidate_method: str,
    reference_method: str,
    max_mean_forgetting_delta_vs_reference: float,
    max_seen_average_std: float,
    max_chexpert_std: float,
) -> dict[str, Any]:
    by_method = {row["method"]: row for row in aggregate_rows}
    source_row = by_method["source_only"]
    reference_row = by_method[reference_method]
    candidate_row = by_method[candidate_method]
    criteria = {
        "mean_seen_average_above_reference": bool(
            candidate_row["seen_average_macro_auroc_mean"] > reference_row["seen_average_macro_auroc_mean"]
        ),
        "mean_chexpert_above_source_only": bool(
            candidate_row["d1_chexpert_test_macro_auroc_mean"] > source_row["d1_chexpert_test_macro_auroc_mean"]
        ),
        "mean_forgetting_close_to_reference": bool(
            candidate_row["nih_forgetting_macro_auroc_mean"]
            <= reference_row["nih_forgetting_macro_auroc_mean"] + max_mean_forgetting_delta_vs_reference
        ),
        "seen_average_variance_not_crazy": bool(candidate_row["seen_average_macro_auroc_std"] <= max_seen_average_std),
        "chexpert_variance_not_crazy": bool(candidate_row["d1_chexpert_test_macro_auroc_std"] <= max_chexpert_std),
    }
    return {
        "candidate_method": candidate_method,
        "reference_method": reference_method,
        "source_only_reference": source_row,
        "reference_summary": reference_row,
        "candidate_summary": candidate_row,
        "criteria": criteria,
        "promote_to_stage2_candidate": bool(all(criteria.values())),
    }


def _gate_sweep_rows(
    *,
    stage0_checkpoint: Path,
    train_dataset,
    val_dataset,
    eval_datasets: dict[str, Any],
    output_root: Path,
    device,
    seed: int,
    source_bank: PrototypeBank,
    base_main_method: dict[str, Any],
    caps: list[float],
    threshold: float,
    source_nih_auroc: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    original_paths = init_run_paths(output_root, "original_gate")
    original_result = train_main_method(
        previous_checkpoint_path=stage0_checkpoint,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_datasets=eval_datasets,
        old_bank=source_bank,
        output_dir=original_paths.root,
        training_config=base_main_method,
        device=device,
        seed=seed,
    )
    original_row = summarize_result_row("original_gate", original_result["metrics"], source_nih_auroc=source_nih_auroc)
    original_row["old_like_similarity_threshold"] = None
    original_row["old_like_gate_cap"] = None
    rows.append(original_row)

    for gate_cap in caps:
        sweep_paths = init_run_paths(output_root, f"gate_cap_{str(gate_cap).replace('.', 'p')}")
        sweep_config = dict(base_main_method)
        sweep_config["old_like_similarity_threshold"] = threshold
        sweep_config["old_like_gate_cap"] = gate_cap
        sweep_result = train_main_method(
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            old_bank=source_bank,
            output_dir=sweep_paths.root,
            training_config=sweep_config,
            device=device,
            seed=seed,
        )
        sweep_row = summarize_result_row(
            f"gate_cap_{str(gate_cap).replace('.', 'p')}",
            sweep_result["metrics"],
            source_nih_auroc=source_nih_auroc,
        )
        sweep_row["old_like_similarity_threshold"] = threshold
        sweep_row["old_like_gate_cap"] = float(gate_cap)
        rows.append(sweep_row)
    return rows


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
    labelwise_config = dict(base_main_method)
    labelwise_config.update(config.get("labelwise_trust_region", {}))
    tiny_config = dict(config["tiny_logit_correction"])
    diagnostic_seed = int(config.get("diagnostic_seed", config["seeds"][0]))

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
    active_threshold = float(config.get("diagnostic_active_threshold", 0.1))

    per_seed_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    auroc_delta_rows: list[dict[str, Any]] = []
    representative_harder_result: dict[str, Any] | None = None
    representative_labelwise_result: dict[str, Any] | None = None
    representative_source_only_metrics: dict[str, Any] | None = None
    representative_source_bank: PrototypeBank | None = None
    source_nih_auroc_reference: float | None = None

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
        vq_bank = build_vq_summary_bank(
            source_embeddings,
            source_targets,
            budget_bytes=source_bank.memory_size_bytes(),
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
        source_row = summarize_result_row("source_only", source_only_metrics, source_nih_auroc=float(source_only_metrics["d0_nih_test"]["macro_auroc"]))
        per_seed_rows.append(seed_result_row("source_only", seed, source_row))
        source_nih_auroc = float(source_row["d0_nih_test_macro_auroc"])
        source_nih_auroc_reference = source_nih_auroc

        lwf_paths = init_run_paths(seed_root, "lwf")
        lwf_result = run_lwf(
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            output_dir=lwf_paths.root,
            training_config=base_training,
            device=device,
            seed=seed,
        )
        per_seed_rows.append(seed_result_row("lwf", seed, summarize_result_row("lwf", lwf_result["metrics"], source_nih_auroc=source_nih_auroc)))

        vq_paths = init_run_paths(seed_root, "vq_summary_replay")
        vq_result = run_vq_summary_replay(
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            output_dir=vq_paths.root,
            training_config=base_training,
            device=device,
            seed=seed,
            replay_bank=vq_bank,
        )
        per_seed_rows.append(
            seed_result_row(
                "vq_summary_replay",
                seed,
                summarize_result_row("vq_summary_replay", vq_result["metrics"], source_nih_auroc=source_nih_auroc),
            )
        )

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
        harder_diagnostics = _collect_diagnostic_rows(
            method_name="harder_gate_clipping",
            seed=seed,
            model=harder_result["model"],
            datasets=diagnostics_eval_datasets,
            batch_size=int(harder_gate_config.get("batch_size", 256)),
            device=device,
            label_names=label_names,
            active_threshold=active_threshold,
        )
        gate_rows.extend(harder_diagnostics["gate_rows"])
        residual_rows.extend(harder_diagnostics["residual_rows"])
        active_rows.extend(harder_diagnostics["active_rows"])

        labelwise_paths = init_run_paths(seed_root, "labelwise_trust_region")
        labelwise_result = train_labelwise_trust_region(
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            old_bank=source_bank,
            output_dir=labelwise_paths.root,
            training_config=labelwise_config,
            device=device,
            seed=seed,
        )
        per_seed_rows.append(
            seed_result_row(
                "labelwise_trust_region",
                seed,
                summarize_result_row("labelwise_trust_region", labelwise_result["metrics"], source_nih_auroc=source_nih_auroc),
            )
        )
        auroc_delta_rows.extend(
            _per_label_delta_rows(
                method_name="labelwise_trust_region",
                seed=seed,
                source_metrics=source_only_metrics,
                candidate_metrics=labelwise_result["metrics"],
                label_names=label_names,
            )
        )
        labelwise_diagnostics = _collect_diagnostic_rows(
            method_name="labelwise_trust_region",
            seed=seed,
            model=labelwise_result["model"],
            datasets=diagnostics_eval_datasets,
            batch_size=int(labelwise_config.get("batch_size", 256)),
            device=device,
            label_names=label_names,
            active_threshold=active_threshold,
        )
        gate_rows.extend(labelwise_diagnostics["gate_rows"])
        residual_rows.extend(labelwise_diagnostics["residual_rows"])
        active_rows.extend(labelwise_diagnostics["active_rows"])

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

        if seed == diagnostic_seed:
            representative_harder_result = harder_result
            representative_labelwise_result = labelwise_result
            representative_source_only_metrics = source_only_metrics
            representative_source_bank = source_bank

    seed_results_csv = write_summary_table(output_paths.reports, "seed_results.csv", per_seed_rows)
    aggregate_rows = add_source_only_deltas(aggregate_seed_rows(per_seed_rows))
    aggregate_rows = sorted(aggregate_rows, key=lambda row: row["seen_average_macro_auroc_mean"], reverse=True)
    aggregate_csv = write_summary_table(output_paths.reports, "seed_summary.csv", aggregate_rows)

    gate_csv = write_summary_table(output_paths.reports, "gate_by_label.csv", gate_rows)
    gate_aggregate_csv = write_summary_table(
        output_paths.reports,
        "gate_by_label_aggregate.csv",
        aggregate_metric_rows(
            gate_rows,
            group_keys=["method", "domain", "metric", "label_index", "label_name"],
            value_keys=["mean", "std"],
        ),
    )
    residual_csv = write_summary_table(output_paths.reports, "residual_by_label.csv", residual_rows)
    residual_aggregate_csv = write_summary_table(
        output_paths.reports,
        "residual_by_label_aggregate.csv",
        aggregate_metric_rows(
            residual_rows,
            group_keys=["method", "domain", "metric", "label_index", "label_name"],
            value_keys=["mean", "std"],
        ),
    )
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

    promotion = _pilot_promotion_decision(
        aggregate_rows,
        candidate_method="labelwise_trust_region",
        reference_method="harder_gate_clipping",
        max_mean_forgetting_delta_vs_reference=float(
            config["promotion"].get("max_mean_forgetting_delta_vs_reference", 0.002)
        ),
        max_seen_average_std=float(config["promotion"]["max_seen_average_std"]),
        max_chexpert_std=float(config["promotion"]["max_chexpert_std"]),
    )
    promotion_path = write_json(output_paths.artifacts / "promotion_decision.json", promotion)

    diagnostics_dir = ensure_dir(output_paths.reports / "gate_ablation")
    pilot_diagnostics_dir = ensure_dir(output_paths.reports / "pilot_diagnostics")
    ablation_rows: list[dict[str, Any]] = []
    if representative_source_bank is not None and representative_harder_result is not None and source_nih_auroc_reference is not None:
        representative_seed_root = ensure_dir(output_paths.root / f"seed_{diagnostic_seed}" / "ablations")
        original_paths = init_run_paths(representative_seed_root, "original_gate")
        original_result = train_main_method(
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            old_bank=representative_source_bank,
            output_dir=original_paths.root,
            training_config=base_main_method,
            device=device,
            seed=diagnostic_seed,
        )
        ablation_rows.append(
            summarize_result_row("original_gate", original_result["metrics"], source_nih_auroc=source_nih_auroc_reference)
        )
        ablation_rows.append(
            summarize_result_row(
                "harder_gate_clipping",
                representative_harder_result["metrics"],
                source_nih_auroc=source_nih_auroc_reference,
            )
        )
        _write_internal_diagnostics(
            diagnostics_dir=diagnostics_dir,
            prefix="original_gate",
            model=original_result["model"],
            datasets=diagnostics_eval_datasets,
            batch_size=int(base_main_method.get("batch_size", 256)),
            device=device,
        )
        _write_internal_diagnostics(
            diagnostics_dir=diagnostics_dir,
            prefix="harder_gate_clipping",
            model=representative_harder_result["model"],
            datasets=diagnostics_eval_datasets,
            batch_size=int(harder_gate_config.get("batch_size", 256)),
            device=device,
        )
        ablation_csv = write_summary_table(output_paths.reports, "original_vs_harder_gate.csv", ablation_rows)

        gate_sweep_rows = _gate_sweep_rows(
            stage0_checkpoint=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            output_root=representative_seed_root,
            device=device,
            seed=diagnostic_seed,
            source_bank=representative_source_bank,
            base_main_method=base_main_method,
            caps=[float(value) for value in config.get("gate_cap_sweep", [])],
            threshold=float(harder_gate_config["old_like_similarity_threshold"]),
            source_nih_auroc=source_nih_auroc_reference,
        )
        gate_sweep_csv = write_summary_table(output_paths.reports, "gate_cap_sweep.csv", gate_sweep_rows)
    else:
        ablation_csv = None
        gate_sweep_csv = None

    representative_auroc_delta_csvs: list[str] = []
    if (
        representative_source_only_metrics is not None
        and representative_harder_result is not None
        and representative_labelwise_result is not None
    ):
        _collect_diagnostic_rows(
            method_name="harder_gate_clipping",
            seed=diagnostic_seed,
            model=representative_harder_result["model"],
            datasets=diagnostics_eval_datasets,
            batch_size=int(harder_gate_config.get("batch_size", 256)),
            device=device,
            label_names=label_names,
            active_threshold=active_threshold,
            diagnostics_dir=pilot_diagnostics_dir,
        )
        _collect_diagnostic_rows(
            method_name="labelwise_trust_region",
            seed=diagnostic_seed,
            model=representative_labelwise_result["model"],
            datasets=diagnostics_eval_datasets,
            batch_size=int(labelwise_config.get("batch_size", 256)),
            device=device,
            label_names=label_names,
            active_threshold=active_threshold,
            diagnostics_dir=pilot_diagnostics_dir,
        )
        for method_name, result in (
            ("harder_gate_clipping", representative_harder_result),
            ("labelwise_trust_region", representative_labelwise_result),
        ):
            for alias, domain_name in (("d0_nih_test", "nih"), ("d1_chexpert_test", "chexpert")):
                delta_path = write_per_label_auroc_delta(
                    pilot_diagnostics_dir / f"{method_name}__{domain_name}__auroc_delta_vs_source_only.csv",
                    source_predictions=prediction_dict_from_artifact(
                        representative_source_only_metrics[alias]["prediction_artifact"]
                    ),
                    candidate_predictions=prediction_dict_from_artifact(result["metrics"][alias]["prediction_artifact"]),
                    label_names=label_names,
                    domain_name=domain_name,
                    source_name="source_only",
                    candidate_name=method_name,
                )
                representative_auroc_delta_csvs.append(str(delta_path))

    sections = [
        ("Manifest", [f"- `{manifest.path}`"]),
        (
            "Seed Summary",
            [
                f"- Per-seed results: `{seed_results_csv}`",
                f"- Aggregate summary: `{aggregate_csv}`",
                f"- Promotion decision: `{promotion_path}`",
            ],
        ),
        (
            "Promotion Decision",
            [f"- `promote_to_stage2_candidate`: `{promotion['promote_to_stage2_candidate']}`"]
            + [f"- `{key}`: `{value}`" for key, value in promotion["criteria"].items()],
        ),
        (
            "Lead Diagnostics",
            [
                f"- Per-seed gate summary: `{gate_csv}`",
                f"- Aggregate gate summary: `{gate_aggregate_csv}`",
                f"- Per-seed active-label fraction: `{active_csv}`",
                f"- Aggregate active-label fraction: `{active_aggregate_csv}`",
                f"- Per-seed residual summary: `{residual_csv}`",
                f"- Aggregate residual summary: `{residual_aggregate_csv}`",
                f"- Per-seed AUROC deltas: `{auroc_delta_csv}`",
                f"- Aggregate AUROC deltas: `{auroc_delta_aggregate_csv}`",
                f"- Representative AUROC delta CSVs: `{pilot_diagnostics_dir}`" if representative_auroc_delta_csvs else "- Representative AUROC delta CSVs: `not_run`",
                f"- Representative diagnostics: `{pilot_diagnostics_dir}`",
            ],
        ),
        (
            "Minimal Ablations",
            [
                f"- Original vs harder gate summary: `{ablation_csv}`" if ablation_csv is not None else "- Original vs harder gate summary: `not_run`",
                f"- Gate/residual diagnostics: `{diagnostics_dir}`",
                f"- Gate-cap sweep: `{gate_sweep_csv}`" if gate_sweep_csv is not None else "- Gate-cap sweep: `not_run`",
            ],
        ),
    ]
    report_path = write_stage_report(
        output_paths.reports,
        "nih_to_chexpert_seed_sweep_report.md",
        title="NIH -> CheXpert Seed Sweep",
        sections=sections,
    )
    return {
        "run_root": str(output_paths.root),
        "seed_results_csv": str(seed_results_csv),
        "aggregate_csv": str(aggregate_csv),
        "promotion_decision_path": str(promotion_path),
        "report_path": str(report_path),
        "promote_to_stage2_candidate": bool(promotion["promote_to_stage2_candidate"]),
    }


def main() -> None:
    args = parse_config_arg("Run the limited NIH -> CheXpert seed sweep.")
    config = load_config(args.config)
    result = run(config)
    print(f"[done] wrote seed-sweep artifacts to {result['run_root']}")


if __name__ == "__main__":
    main()
