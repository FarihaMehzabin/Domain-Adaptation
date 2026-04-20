"""Run the limited NIH -> CheXpert -> MIMIC chain once Stage 1 survives."""

from __future__ import annotations

from pathlib import Path

from paper_v1.analysis.postmortem import summarize_result_row
from paper_v1.baselines.lwf import run_lwf
from paper_v1.baselines.source_only import run_source_only
from paper_v1.baselines.vq_summary_replay import run_vq_summary_replay
from paper_v1.data.embeddings import FrozenEmbeddingDataset
from paper_v1.data.splits import SplitSelector, select_records
from paper_v1.evaluation.reporting import write_stage_report, write_summary_table
from paper_v1.models.prototype_memory import build_label_state_prototypes, build_vq_summary_bank, merge_prototype_banks
from paper_v1.runners.common import collect_alignment, default_device, find_latest_checkpoint, load_config, load_data_config, parse_config_arg, raise_for_issues
from paper_v1.training.stage_adapt import load_continual_model_checkpoint, train_main_method, train_tiny_logit_correction
from paper_v1.utils.io import read_json, write_json
from paper_v1.utils.registry import init_run_paths
from paper_v1.utils.seeds import set_seed


CHECKPOINT_FILENAMES = {
    "lwf": "lwf_best.pt",
    "vq_summary_replay": "vq_summary_replay_best.pt",
    "harder_gate_clipping": "main_method_best.pt",
    "tiny_logit_correction": "tiny_logit_correction_best.pt",
}


def _stage1_method_dir(stage1_seed_root: Path, method_name: str) -> Path:
    matches = sorted(stage1_seed_root.glob(f"*__{method_name}"))
    if not matches:
        raise SystemExit(f"missing Stage 1 directory for method {method_name} under {stage1_seed_root}")
    return matches[-1]


def _stage1_checkpoint(stage1_seed_root: Path, method_name: str) -> Path:
    filename = CHECKPOINT_FILENAMES[method_name]
    checkpoint = find_latest_checkpoint(str(stage1_seed_root / f"*__{method_name}" / "checkpoints" / filename))
    if checkpoint is None:
        raise SystemExit(f"missing Stage 1 checkpoint for method {method_name} under {stage1_seed_root}")
    return checkpoint


def _stage1_metrics(stage1_seed_root: Path, method_name: str) -> dict[str, float]:
    method_root = _stage1_method_dir(stage1_seed_root, method_name)
    nih_metrics = read_json(method_root / "metrics" / "d0_nih_test_metrics.json")
    chex_metrics = read_json(method_root / "metrics" / "d1_chexpert_test_metrics.json")
    return {
        "d0_nih_test_macro_auroc": float(nih_metrics["macro_auroc"]),
        "d1_chexpert_test_macro_auroc": float(chex_metrics["macro_auroc"]),
    }


def _stage2_summary_row(method: str, metrics: dict, *, stage1_reference: dict[str, float]) -> dict[str, float]:
    base_row = summarize_result_row(method, metrics, source_nih_auroc=stage1_reference["d0_nih_test_macro_auroc"])
    mimic_metrics = metrics["d2_mimic_test"]
    chex_forgetting = max(stage1_reference["d1_chexpert_test_macro_auroc"] - float(metrics["d1_chexpert_test"]["macro_auroc"]), 0.0)
    base_row["d2_mimic_test_macro_auroc"] = float(mimic_metrics["macro_auroc"])
    base_row["seen_average_macro_auroc"] = float(
        (
            float(metrics["d0_nih_test"]["macro_auroc"])
            + float(metrics["d1_chexpert_test"]["macro_auroc"])
            + float(metrics["d2_mimic_test"]["macro_auroc"])
        )
        / 3.0
    )
    base_row["nih_forgetting_macro_auroc"] = float(
        max(stage1_reference["d0_nih_test_macro_auroc"] - float(metrics["d0_nih_test"]["macro_auroc"]), 0.0)
    )
    base_row["chexpert_forgetting_macro_auroc"] = float(chex_forgetting)
    base_row["old_domain_average_forgetting_macro_auroc"] = float((base_row["nih_forgetting_macro_auroc"] + chex_forgetting) / 2.0)
    base_row["average_macro_ece"] = float(
        (
            float(metrics["d0_nih_test"].get("macro_ece") or 0.0)
            + float(metrics["d1_chexpert_test"].get("macro_ece") or 0.0)
            + float(metrics["d2_mimic_test"].get("macro_ece") or 0.0)
        )
        / 3.0
    )
    base_row["average_brier_score"] = float(
        (
            float(metrics["d0_nih_test"].get("brier_score") or 0.0)
            + float(metrics["d1_chexpert_test"].get("brier_score") or 0.0)
            + float(metrics["d2_mimic_test"].get("brier_score") or 0.0)
        )
        / 3.0
    )
    return base_row


def run(config: dict) -> dict:
    set_seed(int(config.get("seed", 1337)))
    data_config = load_data_config(config["data_config"])
    manifest, alignment = collect_alignment(
        data_config=data_config,
        domains={"d0_nih", "d1_chexpert", "d2_mimic"},
        splits={"train", "val", "test"},
        missing_embedding_policy=str(config["validation"].get("missing_embedding_policy", "error")),
        require_patient_disjoint=bool(config["validation"].get("require_patient_disjoint", True)),
    )
    raise_for_issues(alignment.issues, allowed_codes=set(config["validation"].get("allowed_issue_codes", [])))
    expected_dropped_row_ids = sorted(str(value) for value in config.get("expected_dropped_row_ids", []))
    observed_dropped_row_ids = sorted(str(value) for value in alignment.dropped_row_ids)
    if expected_dropped_row_ids and observed_dropped_row_ids != expected_dropped_row_ids:
        raise SystemExit(
            "unexpected dropped row ids during Stage 2 alignment:\n"
            f"expected={expected_dropped_row_ids}\n"
            f"observed={observed_dropped_row_ids}"
        )

    stage1_seed = int(config["stage1_seed"])
    stage1_seed_root = Path(config["stage1_seed_sweep_root"]) / f"seed_{stage1_seed}"
    if not stage1_seed_root.exists():
        raise SystemExit(f"missing Stage 1 seed directory: {stage1_seed_root}")

    output_paths = init_run_paths(config["output_root"], config["experiment_name"])
    device = default_device()
    source_checkpoint = Path(config["source_only_checkpoint"])

    source_train_dataset = FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d0_nih", "train")]))
    chex_train_dataset = FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d1_chexpert", "train")]))
    mimic_train_dataset = FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d2_mimic", "train")]))
    mimic_val_dataset = FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d2_mimic", "val")]))
    eval_datasets = {
        "d0_nih_test": FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d0_nih", "test")])),
        "d1_chexpert_test": FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d1_chexpert", "test")])),
        "d2_mimic_test": FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d2_mimic", "test")])),
    }
    if len(mimic_train_dataset) == 0:
        raise SystemExit("MIMIC train split has zero aligned embeddings; restore/export the train split before Stage 2.")

    source_embeddings, source_targets, _ = source_train_dataset.materialize_numpy()
    chex_embeddings, chex_targets, _ = chex_train_dataset.materialize_numpy()
    nih_bank = build_label_state_prototypes(
        source_embeddings,
        source_targets,
        domain="d0_nih",
        positive_k=int(config["memory"].get("positive_k", 4)),
        negative_k=int(config["memory"].get("negative_k", 2)),
        seed=int(config.get("seed", 1337)),
    )
    chex_bank = build_label_state_prototypes(
        chex_embeddings,
        chex_targets,
        domain="d1_chexpert",
        positive_k=int(config["memory"].get("positive_k", 4)),
        negative_k=int(config["memory"].get("negative_k", 2)),
        seed=int(config.get("seed", 1337)) + 17,
    )
    merged_bank = merge_prototype_banks([nih_bank, chex_bank])
    merged_bank.save(output_paths.artifacts / "nih_chexpert_stage1_bank")
    seen_embeddings = source_embeddings
    seen_targets = source_targets
    if len(chex_embeddings):
        import numpy as np

        seen_embeddings = np.concatenate([source_embeddings, chex_embeddings], axis=0)
        seen_targets = np.concatenate([source_targets, chex_targets], axis=0)
    vq_bank = build_vq_summary_bank(
        seen_embeddings,
        seen_targets,
        budget_bytes=merged_bank.memory_size_bytes(),
        seed=int(config.get("seed", 1337)),
    )

    summary_rows = []

    source_only_paths = init_run_paths(output_paths.root, "source_only")
    source_metrics = run_source_only(
        checkpoint_path=source_checkpoint,
        eval_datasets=eval_datasets,
        output_dir=source_only_paths.root,
        batch_size=int(config["training"].get("batch_size", 256)),
        device=device,
    )
    source_reference = {
        "d0_nih_test_macro_auroc": float(source_metrics["d0_nih_test"]["macro_auroc"]),
        "d1_chexpert_test_macro_auroc": float(source_metrics["d1_chexpert_test"]["macro_auroc"]),
    }
    summary_rows.append(_stage2_summary_row("source_only", source_metrics, stage1_reference=source_reference))

    lwf_paths = init_run_paths(output_paths.root, "lwf")
    lwf_metrics = run_lwf(
        previous_checkpoint_path=_stage1_checkpoint(stage1_seed_root, "lwf"),
        train_dataset=mimic_train_dataset,
        val_dataset=mimic_val_dataset,
        eval_datasets=eval_datasets,
        output_dir=lwf_paths.root,
        training_config=config["training"],
        device=device,
        seed=int(config.get("seed", 1337)),
    )["metrics"]
    summary_rows.append(_stage2_summary_row("lwf", lwf_metrics, stage1_reference=_stage1_metrics(stage1_seed_root, "lwf")))

    vq_paths = init_run_paths(output_paths.root, "vq_summary_replay")
    vq_metrics = run_vq_summary_replay(
        previous_checkpoint_path=_stage1_checkpoint(stage1_seed_root, "vq_summary_replay"),
        train_dataset=mimic_train_dataset,
        val_dataset=mimic_val_dataset,
        eval_datasets=eval_datasets,
        output_dir=vq_paths.root,
        training_config=config["training"],
        device=device,
        seed=int(config.get("seed", 1337)),
        replay_bank=vq_bank,
    )["metrics"]
    summary_rows.append(
        _stage2_summary_row(
            "vq_summary_replay",
            vq_metrics,
            stage1_reference=_stage1_metrics(stage1_seed_root, "vq_summary_replay"),
        )
    )

    harder_paths = init_run_paths(output_paths.root, "harder_gate_clipping")
    stage1_harder_model = load_continual_model_checkpoint(
        checkpoint_path=_stage1_checkpoint(stage1_seed_root, "harder_gate_clipping"),
        previous_checkpoint_path=source_checkpoint,
        old_bank=nih_bank,
        training_config=config["harder_gate_clipping"],
        device=device,
    )
    harder_metrics = train_main_method(
        previous_checkpoint_path=source_checkpoint,
        train_dataset=mimic_train_dataset,
        val_dataset=mimic_val_dataset,
        eval_datasets=eval_datasets,
        old_bank=merged_bank,
        output_dir=harder_paths.root,
        training_config=config["harder_gate_clipping"],
        device=device,
        seed=int(config.get("seed", 1337)),
        previous_model_override=stage1_harder_model,
    )["metrics"]
    summary_rows.append(
        _stage2_summary_row(
            "harder_gate_clipping",
            harder_metrics,
            stage1_reference=_stage1_metrics(stage1_seed_root, "harder_gate_clipping"),
        )
    )

    if bool(config.get("run_tiny_logit_correction", True)):
        tiny_paths = init_run_paths(output_paths.root, "tiny_logit_correction")
        tiny_metrics = train_tiny_logit_correction(
            previous_checkpoint_path=_stage1_checkpoint(stage1_seed_root, "tiny_logit_correction"),
            train_dataset=mimic_train_dataset,
            val_dataset=mimic_val_dataset,
            eval_datasets=eval_datasets,
            output_dir=tiny_paths.root,
            training_config=config["tiny_logit_correction"],
            device=device,
            seed=int(config.get("seed", 1337)),
        )["metrics"]
        summary_rows.append(
            _stage2_summary_row(
                "tiny_logit_correction",
                tiny_metrics,
                stage1_reference=_stage1_metrics(stage1_seed_root, "tiny_logit_correction"),
            )
        )

    summary_rows = sorted(summary_rows, key=lambda row: row["seen_average_macro_auroc"], reverse=True)
    summary_csv = write_summary_table(output_paths.reports, "nih_to_chexpert_to_mimic_summary.csv", summary_rows)
    report_path = write_stage_report(
        output_paths.reports,
        "nih_to_chexpert_to_mimic_report.md",
        title="NIH -> CheXpert -> MIMIC Limited Stage 2",
        sections=[
            ("Manifest", [f"- `{manifest.path}`"]),
            (
                "Leaderboard",
                [
                    (
                        f"- `{row['method']}`: seen_avg=`{row['seen_average_macro_auroc']:.4f}` "
                        f"nih=`{row['d0_nih_test_macro_auroc']:.4f}` "
                        f"chexpert=`{row['d1_chexpert_test_macro_auroc']:.4f}` "
                        f"mimic=`{row['d2_mimic_test_macro_auroc']:.4f}` "
                        f"old_forgetting=`{row['old_domain_average_forgetting_macro_auroc']:.4f}`"
                    )
                    for row in summary_rows
                ],
            ),
        ],
    )
    write_json(
        output_paths.artifacts / "stage2_config_echo.json",
        {
            "stage1_seed_sweep_root": str(stage1_seed_root),
            "data_config": str(config["data_config"]),
            "summary_csv": str(summary_csv),
            "dropped_row_ids": observed_dropped_row_ids,
        },
    )
    return {
        "run_root": str(output_paths.root),
        "summary_csv": str(summary_csv),
        "report_path": str(report_path),
    }


def main() -> None:
    args = parse_config_arg("Run the limited NIH -> CheXpert -> MIMIC chain.")
    config = load_config(args.config)
    result = run(config)
    print(f"[done] wrote Stage 2 artifacts to {result['run_root']}")


if __name__ == "__main__":
    main()
