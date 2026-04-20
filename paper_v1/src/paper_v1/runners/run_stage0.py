"""Run the real NIH source-only baseline on current assets."""

from __future__ import annotations

from pathlib import Path

from paper_v1.data.embeddings import FrozenEmbeddingDataset
from paper_v1.data.splits import SplitSelector, select_records
from paper_v1.evaluation.reporting import write_stage_report
from paper_v1.models.prototype_memory import build_label_state_prototypes
from paper_v1.runners.common import collect_alignment, default_device, load_config, load_data_config, parse_config_arg, raise_for_issues
from paper_v1.training.stage0_source import train_stage0
from paper_v1.utils.registry import init_run_paths
from paper_v1.utils.seeds import set_seed


def run(config: dict) -> dict:
    set_seed(int(config.get("seed", 1337)))
    data_config = load_data_config(config["data_config"])
    requested_selectors = [
        SplitSelector("d0_nih", "train"),
        SplitSelector("d0_nih", "val"),
        SplitSelector("d0_nih", "test"),
        SplitSelector("d1_chexpert", "test"),
        SplitSelector("d2_mimic", "test"),
    ]
    manifest, alignment = collect_alignment(
        data_config=data_config,
        domains={"d0_nih", "d1_chexpert", "d2_mimic"},
        splits={"train", "val", "test"},
        selectors=requested_selectors,
        missing_embedding_policy=str(config["validation"].get("missing_embedding_policy", "error")),
        require_patient_disjoint=False,
    )
    allowed_codes = set(config["validation"].get("allowed_issue_codes", []))
    raise_for_issues(alignment.issues, allowed_codes=allowed_codes)
    train_records = select_records(alignment.records, [SplitSelector("d0_nih", "train")])
    val_records = select_records(alignment.records, [SplitSelector("d0_nih", "val")])
    eval_records = {
        "d0_nih_test": FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d0_nih", "test")])),
        "d1_chexpert_test": FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d1_chexpert", "test")])),
        "d2_mimic_test": FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d2_mimic", "test")])),
    }
    train_dataset = FrozenEmbeddingDataset(train_records)
    val_dataset = FrozenEmbeddingDataset(val_records)
    output_paths = init_run_paths(config["output_root"], config["experiment_name"])
    result = train_stage0(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_datasets=eval_records,
        feature_dim=int(config["model"]["feature_dim"]),
        num_labels=int(config["model"]["num_labels"]),
        output_dir=output_paths.root,
        training_config=config["training"],
        device=default_device(),
        seed=int(config.get("seed", 1337)),
    )
    embeddings, targets, _ = train_dataset.materialize_numpy()
    bank = build_label_state_prototypes(
        embeddings,
        targets,
        domain="d0_nih",
        positive_k=int(config["memory"].get("positive_k", 4)),
        negative_k=int(config["memory"].get("negative_k", 2)),
        seed=int(config.get("seed", 1337)),
    )
    bank.save(output_paths.artifacts / "nih_source_bank")
    write_stage_report(
        output_paths.reports,
        "stage0_report.md",
        title="Stage 0 NIH Source Report",
        sections=[
            ("Manifest", [f"- `{manifest.path}`"]),
            ("Checkpoint", [f"- `{result['checkpoint_path']}`"]),
            ("Prototype bank", [f"- `{output_paths.artifacts / 'nih_source_bank.npz'}`"]),
        ],
    )
    return {"run_root": str(output_paths.root), "checkpoint_path": str(result["checkpoint_path"])}


def main() -> None:
    args = parse_config_arg("Run the paper_v1 NIH source baseline.")
    config = load_config(args.config)
    result = run(config)
    print(f"[done] wrote Stage 0 artifacts to {result['run_root']}")


if __name__ == "__main__":
    main()
