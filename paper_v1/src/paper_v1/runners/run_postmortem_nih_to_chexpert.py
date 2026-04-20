"""Run the NIH -> CheXpert postmortem and limited rescue variants."""

from __future__ import annotations

from pathlib import Path

import torch

from paper_v1.analysis.postmortem import (
    collect_internal_outputs,
    comparison_rows,
    default_label_names,
    prediction_dict_from_artifact,
    summarize_result_row,
    verify_main_method_invariants,
    write_histogram_csv,
    write_histogram_summary,
    write_per_label_auroc_delta,
)
from paper_v1.baselines.lwf import run_lwf
from paper_v1.baselines.source_only import run_source_only
from paper_v1.baselines.vq_summary_replay import run_vq_summary_replay
from paper_v1.data.embeddings import FrozenEmbeddingDataset
from paper_v1.data.splits import SplitSelector, select_records
from paper_v1.evaluation.reporting import write_stage_report, write_summary_table
from paper_v1.models.prototype_memory import build_label_state_prototypes, build_vq_summary_bank
from paper_v1.runners.common import collect_alignment, default_device, load_config, load_data_config, parse_config_arg, raise_for_issues
from paper_v1.training.stage_adapt import train_linear_adaptation, train_main_method, train_tiny_logit_correction
from paper_v1.utils.io import write_json
from paper_v1.utils.registry import init_run_paths
from paper_v1.utils.seeds import set_seed


def _merged_config(config: dict, key: str) -> dict:
    merged = dict(config["stage1_config"][key])
    merged.update(config.get(key, {}))
    return merged


def run(config: dict) -> dict:
    set_seed(int(config.get("seed", 1337)))
    stage1_config = load_config(config["stage1_config_path"])
    config["stage1_config"] = stage1_config
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
    train_dataset = FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d1_chexpert", "train")]))
    val_dataset = FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d1_chexpert", "val")]))
    eval_datasets = {
        "d0_nih_test": FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d0_nih", "test")])),
        "d1_chexpert_test": FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d1_chexpert", "test")])),
    }

    stage0_checkpoint = Path(stage1_config["stage0_checkpoint"])
    device = default_device()
    base_training = _merged_config(config, "training")
    base_main_method = _merged_config(config, "main_method")
    source_embeddings, source_targets, _ = FrozenEmbeddingDataset(
        select_records(alignment.records, [SplitSelector("d0_nih", "train")])
    ).materialize_numpy()
    source_bank = build_label_state_prototypes(
        source_embeddings,
        source_targets,
        domain="d0_nih",
        positive_k=int(stage1_config["memory"].get("positive_k", 4)),
        negative_k=int(stage1_config["memory"].get("negative_k", 2)),
        seed=int(config.get("seed", 1337)),
    )
    source_bank.save(output_paths.artifacts / "nih_source_bank")
    vq_bank = build_vq_summary_bank(
        source_embeddings,
        source_targets,
        budget_bytes=source_bank.memory_size_bytes(),
        seed=int(config.get("seed", 1337)),
    )

    results: dict[str, dict] = {}
    rescue_methods: set[str] = set()

    source_only_paths = init_run_paths(output_paths.root, "source_only")
    source_only_metrics = run_source_only(
        checkpoint_path=stage0_checkpoint,
        eval_datasets=eval_datasets,
        output_dir=source_only_paths.root,
        batch_size=int(base_training.get("batch_size", 256)),
        device=device,
    )
    results["source_only"] = {"metrics": source_only_metrics, "output_dir": str(source_only_paths.root)}
    source_nih_auroc = float(source_only_metrics["d0_nih_test"]["macro_auroc"])

    lwf_paths = init_run_paths(output_paths.root, "lwf")
    lwf_result = run_lwf(
        previous_checkpoint_path=stage0_checkpoint,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_datasets=eval_datasets,
        output_dir=lwf_paths.root,
        training_config=base_training,
        device=device,
        seed=int(config.get("seed", 1337)),
    )
    results["lwf"] = {**lwf_result, "output_dir": str(lwf_paths.root)}

    vq_paths = init_run_paths(output_paths.root, "vq_summary_replay")
    vq_result = run_vq_summary_replay(
        previous_checkpoint_path=stage0_checkpoint,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_datasets=eval_datasets,
        output_dir=vq_paths.root,
        training_config=base_training,
        device=device,
        seed=int(config.get("seed", 1337)),
        replay_bank=vq_bank,
    )
    results["vq_summary_replay"] = {**vq_result, "output_dir": str(vq_paths.root)}

    main_paths = init_run_paths(output_paths.root, "main_method")
    main_result = train_main_method(
        previous_checkpoint_path=stage0_checkpoint,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_datasets=eval_datasets,
        old_bank=source_bank,
        output_dir=main_paths.root,
        training_config=base_main_method,
        device=device,
        seed=int(config.get("seed", 1337)),
    )
    results["main_method"] = {**main_result, "output_dir": str(main_paths.root)}

    rescue_configs = config["rescue_variants"]
    smaller_paths = init_run_paths(output_paths.root, "smaller_bottleneck")
    smaller_main_config = dict(base_main_method)
    smaller_main_config.update(rescue_configs["smaller_bottleneck"])
    results["smaller_bottleneck"] = {
        **train_main_method(
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            old_bank=source_bank,
            output_dir=smaller_paths.root,
            training_config=smaller_main_config,
            device=device,
            seed=int(config.get("seed", 1337)),
        ),
        "output_dir": str(smaller_paths.root),
    }
    rescue_methods.add("smaller_bottleneck")

    strong_paths = init_run_paths(output_paths.root, "stronger_replay_zero")
    strong_main_config = dict(base_main_method)
    strong_main_config.update(rescue_configs["stronger_replay_zero"])
    results["stronger_replay_zero"] = {
        **train_main_method(
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            old_bank=source_bank,
            output_dir=strong_paths.root,
            training_config=strong_main_config,
            device=device,
            seed=int(config.get("seed", 1337)),
        ),
        "output_dir": str(strong_paths.root),
    }
    rescue_methods.add("stronger_replay_zero")

    clipped_paths = init_run_paths(output_paths.root, "harder_gate_clipping")
    clipped_main_config = dict(base_main_method)
    clipped_main_config.update(rescue_configs["harder_gate_clipping"])
    results["harder_gate_clipping"] = {
        **train_main_method(
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            old_bank=source_bank,
            output_dir=clipped_paths.root,
            training_config=clipped_main_config,
            device=device,
            seed=int(config.get("seed", 1337)),
        ),
        "output_dir": str(clipped_paths.root),
    }
    rescue_methods.add("harder_gate_clipping")

    lwf_proto_paths = init_run_paths(output_paths.root, "lwf_prototype_replay")
    lwf_proto_training = dict(base_training)
    lwf_proto_training.update(rescue_configs["lwf_prototype_replay"])
    results["lwf_prototype_replay"] = {
        **train_linear_adaptation(
            method_name="lwf_prototype_replay",
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            output_dir=lwf_proto_paths.root,
            training_config=lwf_proto_training,
            device=device,
            seed=int(config.get("seed", 1337)),
            replay_bank=source_bank,
        ),
        "output_dir": str(lwf_proto_paths.root),
    }
    rescue_methods.add("lwf_prototype_replay")

    tiny_paths = init_run_paths(output_paths.root, "tiny_logit_correction")
    tiny_training = dict(config["tiny_logit_correction"])
    results["tiny_logit_correction"] = {
        **train_tiny_logit_correction(
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            output_dir=tiny_paths.root,
            training_config=tiny_training,
            device=device,
            seed=int(config.get("seed", 1337)),
        ),
        "output_dir": str(tiny_paths.root),
    }
    rescue_methods.add("tiny_logit_correction")

    summary_rows = [
        summarize_result_row(
            method,
            result["metrics"] if "metrics" in result else result,
            source_nih_auroc=source_nih_auroc,
        )
        for method, result in results.items()
    ]
    summary_rows = sorted(summary_rows, key=lambda row: row["seen_average_macro_auroc"], reverse=True)
    summary_csv = write_summary_table(output_paths.reports, "postmortem_summary.csv", summary_rows)

    invariants = verify_main_method_invariants(
        model=results["main_method"]["model"],
        old_bank=source_bank,
        train_dataset=train_dataset,
        device=device,
        history=results["main_method"]["history"],
    )
    invariants_path = write_json(output_paths.artifacts / "implementation_checks.json", invariants)

    diagnostics_dir = output_paths.reports / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    internal_aliases = {
        "nih_test": eval_datasets["d0_nih_test"],
        "chexpert_test": eval_datasets["d1_chexpert_test"],
    }
    for alias, dataset in internal_aliases.items():
        internals = collect_internal_outputs(
            results["main_method"]["model"],
            dataset,
            batch_size=int(base_main_method.get("batch_size", 256)),
            device=device,
        )
        write_histogram_csv(diagnostics_dir / f"main_method__{alias}__gate_histogram.csv", internals["gate"], bins=20, min_value=0.0, max_value=1.0)
        write_histogram_summary(diagnostics_dir / f"main_method__{alias}__gate_summary.json", internals["gate"])
        write_histogram_csv(diagnostics_dir / f"main_method__{alias}__residual_norm_histogram.csv", internals["residual_norm"], bins=20)
        write_histogram_summary(diagnostics_dir / f"main_method__{alias}__residual_norm_summary.json", internals["residual_norm"])
        write_histogram_summary(diagnostics_dir / f"main_method__{alias}__retrieval_max_similarity_summary.json", internals["retrieval_max_similarity"])

    label_names = default_label_names()
    main_root = Path(results["main_method"]["output_dir"])
    source_root = Path(results["source_only"]["output_dir"])
    write_per_label_auroc_delta(
        diagnostics_dir / "main_method_vs_source_only__nih_test__per_label_auroc_delta.csv",
        source_predictions=prediction_dict_from_artifact(source_root / "metrics" / "d0_nih_test_predictions.npz"),
        candidate_predictions=prediction_dict_from_artifact(main_root / "metrics" / "d0_nih_test_predictions.npz"),
        label_names=label_names,
        domain_name="d0_nih_test",
        source_name="source_only",
        candidate_name="main_method",
    )
    write_per_label_auroc_delta(
        diagnostics_dir / "main_method_vs_source_only__chexpert_test__per_label_auroc_delta.csv",
        source_predictions=prediction_dict_from_artifact(source_root / "metrics" / "d1_chexpert_test_predictions.npz"),
        candidate_predictions=prediction_dict_from_artifact(main_root / "metrics" / "d1_chexpert_test_predictions.npz"),
        label_names=label_names,
        domain_name="d1_chexpert_test",
        source_name="source_only",
        candidate_name="main_method",
    )

    comparison = comparison_rows(summary_rows, rescue_methods=rescue_methods)
    comparison_csv = write_summary_table(output_paths.reports, "rescue_comparison.csv", comparison)
    viable_rescues = [
        row["method"]
        for row in comparison
        if row["goal_preserve_source_only_seen_avg"] and row["goal_gain_chexpert_or_calibration"]
    ]
    recommendation = {
        "continue_with_rescue_method": bool(viable_rescues),
        "viable_rescues": viable_rescues,
        "pivot_to_protocol_benchmark_story": not bool(viable_rescues),
        "best_seen_average_method": summary_rows[0]["method"],
    }
    recommendation_path = write_json(output_paths.artifacts / "pivot_recommendation.json", recommendation)

    write_stage_report(
        output_paths.reports,
        "postmortem_report.md",
        title="NIH -> CheXpert Postmortem",
        sections=[
            ("Verification", [f"- `{key}`: `{value}`" for key, value in invariants.items() if key != "toy_forgetting"]),
            (
                "Main Method Diagnostics",
                [
                    f"- Gate histogram CSVs: `{diagnostics_dir / 'main_method__nih_test__gate_histogram.csv'}`, `{diagnostics_dir / 'main_method__chexpert_test__gate_histogram.csv'}`",
                    f"- Residual histogram CSVs: `{diagnostics_dir / 'main_method__nih_test__residual_norm_histogram.csv'}`, `{diagnostics_dir / 'main_method__chexpert_test__residual_norm_histogram.csv'}`",
                    f"- Per-label AUROC deltas: `{diagnostics_dir / 'main_method_vs_source_only__nih_test__per_label_auroc_delta.csv'}`, `{diagnostics_dir / 'main_method_vs_source_only__chexpert_test__per_label_auroc_delta.csv'}`",
                    f"- Main-method training curves: `{main_root / 'artifacts' / 'main_method_history.csv'}`",
                ],
            ),
            (
                "Rescue Comparison",
                [f"- `{row['method']}` seen_avg=`{row['seen_average_macro_auroc']:.4f}` chex_delta_vs_source=`{row['chexpert_delta_vs_source_only']:.4f}` ece_delta_vs_source=`{row['ece_delta_vs_source_only']:.4f}`" for row in comparison],
            ),
            ("Recommendation", [f"- `{key}`: `{value}`" for key, value in recommendation.items()]),
        ],
    )
    return {
        "run_root": str(output_paths.root),
        "summary_csv": str(summary_csv),
        "comparison_csv": str(comparison_csv),
        "invariants_path": str(invariants_path),
        "recommendation_path": str(recommendation_path),
    }


def main() -> None:
    args = parse_config_arg("Run the NIH -> CheXpert postmortem.")
    config = load_config(args.config)
    result = run(config)
    print(f"[done] wrote postmortem artifacts to {result['run_root']}")


if __name__ == "__main__":
    main()
