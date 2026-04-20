"""Synthetic smoke runner."""

from __future__ import annotations

from pathlib import Path

from paper_v1.baselines.finetune_seq import run_finetune_seq
from paper_v1.data.synthetic import generate_synthetic_stage_datasets
from paper_v1.evaluation.reporting import write_stage_report, write_summary_table
from paper_v1.models.prototype_memory import build_label_state_prototypes
from paper_v1.runners.common import default_device, load_config, parse_config_arg
from paper_v1.training.stage0_source import train_stage0
from paper_v1.training.stage_adapt import train_main_method
from paper_v1.utils.registry import init_run_paths
from paper_v1.utils.seeds import set_seed


def run(config: dict) -> dict:
    set_seed(int(config.get("seed", 1337)))
    datasets = generate_synthetic_stage_datasets(
        feature_dim=int(config.get("feature_dim", 16)),
        num_labels=int(config.get("num_labels", 7)),
        seed=int(config.get("seed", 1337)),
    )
    device = default_device()
    stage0_paths = init_run_paths(config["output_root"], f"{config['experiment_name']}__stage0")
    stage0_result = train_stage0(
        train_dataset=datasets["d0_nih_train"],
        val_dataset=datasets["d0_nih_val"],
        eval_datasets={
            "d0_nih_test": datasets["d0_nih_test"],
            "d1_chexpert_test": datasets["d1_chexpert_test"],
            "d2_mimic_test": datasets["d2_mimic_test"],
        },
        feature_dim=int(config.get("feature_dim", 16)),
        num_labels=int(config.get("num_labels", 7)),
        output_dir=stage0_paths.root,
        training_config=config["training"],
        device=device,
        seed=int(config.get("seed", 1337)),
    )
    source_embeddings, source_targets, _ = datasets["d0_nih_train"].materialize_numpy()
    source_bank = build_label_state_prototypes(
        source_embeddings,
        source_targets,
        domain="d0_nih",
        positive_k=int(config["memory"].get("positive_k", 2)),
        negative_k=int(config["memory"].get("negative_k", 1)),
        seed=int(config.get("seed", 1337)),
    )
    source_bank.save(stage0_paths.artifacts / "nih_source_bank")

    finetune_paths = init_run_paths(config["output_root"], f"{config['experiment_name']}__finetune")
    finetune_result = run_finetune_seq(
        previous_checkpoint_path=stage0_result["checkpoint_path"],
        train_dataset=datasets["d1_chexpert_train"],
        val_dataset=datasets["d1_chexpert_val"],
        eval_datasets={
            "d0_nih_test": datasets["d0_nih_test"],
            "d1_chexpert_test": datasets["d1_chexpert_test"],
        },
        output_dir=finetune_paths.root,
        training_config=config["training"],
        device=device,
        seed=int(config.get("seed", 1337)),
    )

    main_paths = init_run_paths(config["output_root"], f"{config['experiment_name']}__main_method")
    main_result = train_main_method(
        previous_checkpoint_path=stage0_result["checkpoint_path"],
        train_dataset=datasets["d1_chexpert_train"],
        val_dataset=datasets["d1_chexpert_val"],
        eval_datasets={
            "d0_nih_test": datasets["d0_nih_test"],
            "d1_chexpert_test": datasets["d1_chexpert_test"],
        },
        old_bank=source_bank,
        output_dir=main_paths.root,
        training_config=config["main_method"],
        device=device,
        seed=int(config.get("seed", 1337)),
    )

    rows = [
        {
            "method": "stage0_source",
            "d0_nih_test_macro_auroc": stage0_result["best_val_macro_auroc"],
            "d1_chexpert_test_macro_auroc": stage0_result["metrics"]["d1_chexpert_test"]["macro_auroc"],
        },
        {
            "method": "finetune_seq",
            "d0_nih_test_macro_auroc": finetune_result["metrics"]["d0_nih_test"]["macro_auroc"],
            "d1_chexpert_test_macro_auroc": finetune_result["metrics"]["d1_chexpert_test"]["macro_auroc"],
        },
        {
            "method": "main_method",
            "d0_nih_test_macro_auroc": main_result["metrics"]["d0_nih_test"]["macro_auroc"],
            "d1_chexpert_test_macro_auroc": main_result["metrics"]["d1_chexpert_test"]["macro_auroc"],
        },
    ]
    summary_dir = Path(config["output_root"])
    write_summary_table(summary_dir, "smoke_summary.csv", rows)
    write_stage_report(
        summary_dir,
        "smoke_report.md",
        title="Smoke Report",
        sections=[
            ("Methods", [f"- {row['method']}" for row in rows]),
            ("Outputs", [f"- Stage0: `{stage0_paths.root}`", f"- Finetune: `{finetune_paths.root}`", f"- Main method: `{main_paths.root}`"]),
        ],
    )
    return {
        "stage0": stage0_result,
        "finetune": finetune_result,
        "main_method": main_result,
        "summary_csv": str(summary_dir / "smoke_summary.csv"),
    }


def main() -> None:
    args = parse_config_arg("Run the paper_v1 smoke pipeline.")
    config = load_config(args.config)
    result = run(config)
    print(f"[done] wrote smoke summary to {result['summary_csv']}")


if __name__ == "__main__":
    main()
