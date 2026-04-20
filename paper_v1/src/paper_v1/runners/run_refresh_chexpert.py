"""Refresh CheXpert Stage 1 assets with patient-disjoint splits."""

from __future__ import annotations

import subprocess
from pathlib import Path

from paper_v1.data.chexpert_refresh import (
    authenticate_kaggle,
    build_combined_manifest,
    build_patient_disjoint_chexpert_manifest,
    build_refreshed_data_config,
    download_chexpert_subset,
    ensure_kaggle_environment,
    init_refresh_paths,
    load_csv_rows,
    render_refresh_report,
    write_manifest,
)
from paper_v1.runners.common import find_latest_checkpoint, load_config, load_data_config, parse_config_arg
from paper_v1.utils.io import write_json
from paper_v1.utils.registry import init_run_paths


def _download_metadata_if_needed(paths, config: dict) -> tuple[Path, Path]:
    train_csv = paths.raw_root / "train.csv"
    valid_csv = paths.raw_root / "valid.csv"
    if train_csv.exists() and valid_csv.exists() and not bool(config.get("force_metadata_download", False)):
        return train_csv, valid_csv
    ensure_kaggle_environment(config["kaggle_config"])
    api = authenticate_kaggle()
    download_chexpert_subset(
        api=api,
        dataset_ref=config["dataset_ref"],
        requested_relative_paths=["train.csv", "valid.csv"],
        destination_root=paths.raw_root,
        force=bool(config.get("force_metadata_download", False)),
        download_delay_sec=float(config.get("download_delay_sec", 0.25)),
        max_download_retries=int(config.get("max_download_retries", 5)),
        initial_backoff_sec=float(config.get("initial_backoff_sec", 8.0)),
    )
    return train_csv, valid_csv


def _download_images(paths, config: dict, chexpert_rows: list[dict[str, str]]) -> Path:
    ensure_kaggle_environment(config["kaggle_config"])
    api = authenticate_kaggle()
    relative_paths = sorted(
        {
            row["image_path"].removeprefix("chexpert_small/raw/")
            for row in chexpert_rows
            if row["image_path"].startswith("chexpert_small/raw/")
        }
    )
    summary = download_chexpert_subset(
        api=api,
        dataset_ref=config["dataset_ref"],
        requested_relative_paths=relative_paths,
        destination_root=paths.raw_root,
        force=bool(config.get("force_image_download", False)),
        download_delay_sec=float(config.get("download_delay_sec", 0.25)),
        max_download_retries=int(config.get("max_download_retries", 5)),
        initial_backoff_sec=float(config.get("initial_backoff_sec", 8.0)),
        bulk_download=bool(config.get("bulk_image_download", False)),
        bulk_work_root=paths.root / "tmp",
    )
    return write_json(paths.downloads_dir / "image_download_summary.json", summary)


def _run_embedding_export(paths, config: dict, chexpert_manifest_path: Path) -> Path:
    experiments_root = paths.embeddings_root / "by_id"
    experiments_root.mkdir(parents=True, exist_ok=True)
    experiment_name = str(config.get("embedding_experiment_name", "paper_v1_chexpert_refresh"))
    command = [
        str(config.get("export_python", "python")),
        "/workspace/scripts/14_generate_cxr_foundation_embeddings.py",
        "--manifest-csv",
        str(chexpert_manifest_path),
        "--data-root",
        str(paths.root / "data"),
        "--experiments-root",
        str(experiments_root),
        "--experiment-name",
        experiment_name,
        "--batch-size",
        str(int(config.get("embedding_batch_size", 64))),
        "--embedding-kind",
        str(config.get("embedding_kind", "general")),
        "--token-pooling",
        str(config.get("token_pooling", "avg")),
        "--overwrite",
    ]
    if config.get("embedding_model_dir"):
        command.extend(["--model-dir", str(config["embedding_model_dir"])])
    if config.get("hf_token_env_var"):
        command.extend(["--hf-token-env-var", str(config["hf_token_env_var"])])
    subprocess.run(command, check=True)
    candidates = sorted(experiments_root.glob(f"*{experiment_name}*"))
    if not candidates:
        candidates = sorted(experiments_root.iterdir())
    if not candidates:
        raise SystemExit(f"no embedding export directory found under {experiments_root}")
    return candidates[-1] / "embedding_index.csv"


def _write_ready_stage1_config(paths, config: dict, refreshed_data_config_path: Path) -> Path:
    stage1_config = load_config("/workspace/paper_v1/configs/experiments/nih_to_chexpert_refresh_required.json")
    stage1_config["data_config"] = str(refreshed_data_config_path)
    latest_stage0 = find_latest_checkpoint("/workspace/paper_v1/outputs/stage0_nih_current/*/checkpoints/stage0_best.pt")
    latest_fisher = find_latest_checkpoint("/workspace/paper_v1/outputs/stage0_nih_current/*/checkpoints/stage0_fisher.pt")
    if latest_stage0 is not None:
        stage1_config["stage0_checkpoint"] = str(latest_stage0)
    if latest_fisher is not None:
        stage1_config["stage0_fisher_path"] = str(latest_fisher)
    return write_json(paths.configs_dir / "nih_to_chexpert_refresh_ready.json", stage1_config)


def run(config: dict) -> dict:
    current_data_config = load_data_config(config["current_data_config"])
    run_paths = init_run_paths(config["output_root"], config["experiment_name"])
    paths = init_refresh_paths(run_paths.root)

    train_csv, valid_csv = _download_metadata_if_needed(paths, config)
    refreshed_chexpert_rows, refresh_summary = build_patient_disjoint_chexpert_manifest(
        train_csv=train_csv,
        valid_csv=valid_csv,
        target_train_count=int(config["target_counts"]["train"]),
        target_val_count=int(config["target_counts"]["val"]),
        target_test_count=int(config["target_counts"]["test"]),
        seed=int(config.get("seed", 1337)),
    )

    current_manifest_rows = load_csv_rows(current_data_config["manifest_path"])
    combined_rows = build_combined_manifest(current_manifest_rows, refreshed_chexpert_rows)

    chexpert_manifest_path = write_manifest(paths.manifest_dir / "manifest_chexpert_refreshed.csv", refreshed_chexpert_rows)
    combined_manifest_path = write_manifest(paths.manifest_dir / "manifest_current_plus_refreshed_chexpert.csv", combined_rows)
    refresh_summary["chexpert_manifest_path"] = str(chexpert_manifest_path)
    refresh_summary["combined_manifest_path"] = str(combined_manifest_path)
    refresh_summary["downloaded_images"] = False
    refresh_summary["exported_embeddings"] = False

    if bool(config.get("download_images", False)):
        download_summary_path = _download_images(paths, config, refreshed_chexpert_rows)
        refresh_summary["downloaded_images"] = True
        refresh_summary["image_download_summary_path"] = str(download_summary_path)

    refreshed_index_csv = None
    if bool(config.get("export_embeddings", False)):
        refreshed_index_csv = _run_embedding_export(paths, config, chexpert_manifest_path)
        refresh_summary["exported_embeddings"] = True
        refresh_summary["refreshed_embedding_index_csv"] = str(refreshed_index_csv)

    if refreshed_index_csv is not None:
        refreshed_data_config_path = build_refreshed_data_config(
            current_data_config=current_data_config,
            combined_manifest_path=combined_manifest_path,
            refreshed_chexpert_index_csv=refreshed_index_csv,
            output_path=paths.configs_dir / "data_config_refreshed_chexpert.json",
        )
        refresh_summary["refreshed_data_config_path"] = str(refreshed_data_config_path)
        ready_stage1_config_path = _write_ready_stage1_config(paths, config, refreshed_data_config_path)
        refresh_summary["ready_stage1_config_path"] = str(ready_stage1_config_path)

    summary_json = write_json(paths.reports_dir / "chexpert_refresh_summary.json", refresh_summary)
    report_path = render_refresh_report(refresh_summary, paths)
    return {
        "run_root": str(paths.root),
        "summary_json": str(summary_json),
        "report_path": str(report_path),
        "combined_manifest_path": str(combined_manifest_path),
    }


def main() -> None:
    args = parse_config_arg("Refresh CheXpert Stage 1 assets.")
    config = load_config(args.config)
    result = run(config)
    print(f"[done] wrote CheXpert refresh outputs to {result['run_root']}")


if __name__ == "__main__":
    main()
