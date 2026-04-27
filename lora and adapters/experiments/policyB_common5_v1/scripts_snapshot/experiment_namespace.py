#!/usr/bin/env python3
"""Shared helpers for isolated experiment namespaces and Policy B manifest guards."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICYB_BASE_DIR = Path("experiments/policyB_common5_v1")
LEGACY_TARGET_MANIFEST_BASENAMES = {
    "mimic_common5_train_pool.csv",
    "mimic_common5_val.csv",
    "mimic_common5_test.csv",
}
POLICYB_TARGET_MANIFEST_BASENAMES = {
    "train": "mimic_common5_policyB_train_pool.csv",
    "val": "mimic_common5_policyB_val.csv",
    "test": "mimic_common5_policyB_test.csv",
}


@dataclass(frozen=True)
class NamespaceConfig:
    base_dir: Path | None
    output_root: Path
    manifests_dir: Path | None
    checkpoints_dir: Path
    outputs_dir: Path
    reports_dir: Path
    logs_dir: Path


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def build_namespace_config(base_dir_arg: str | None, out_dir_arg: str | None) -> NamespaceConfig:
    if base_dir_arg is None and out_dir_arg is None:
        raise ValueError("Either --out_dir or --base_dir must be provided.")
    base_dir = resolve_project_path(base_dir_arg) if base_dir_arg else None
    output_root = base_dir if base_dir is not None else resolve_project_path(out_dir_arg)
    manifests_dir = (base_dir / "manifests").resolve() if base_dir is not None else None
    return NamespaceConfig(
        base_dir=base_dir,
        output_root=output_root,
        manifests_dir=manifests_dir,
        checkpoints_dir=(output_root / "checkpoints").resolve(),
        outputs_dir=(output_root / "outputs").resolve(),
        reports_dir=(output_root / "reports").resolve(),
        logs_dir=(output_root / "logs").resolve(),
    )


def resolve_manifest_path(
    raw_value: str | None,
    *,
    namespace: NamespaceConfig,
    default_filename: str | None = None,
) -> Path | None:
    if raw_value is None:
        if default_filename is None:
            return None
        if namespace.manifests_dir is not None:
            return (namespace.manifests_dir / default_filename).resolve()
        return (PROJECT_ROOT / "manifests" / default_filename).resolve()

    path = Path(raw_value)
    if path.is_absolute():
        return path.resolve()

    if namespace.manifests_dir is not None:
        if path.parts and path.parts[0] == "manifests":
            path = Path(*path.parts[1:]) if len(path.parts) > 1 else Path(default_filename or "")
        return (namespace.manifests_dir / path).resolve()

    return (PROJECT_ROOT / path).resolve()


def resolve_input_path(
    raw_value: str | None,
    *,
    default_relative: str | None = None,
) -> Path | None:
    if raw_value is None:
        if default_relative is None:
            return None
        return resolve_project_path(default_relative)

    path = Path(raw_value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def resolve_report_input(
    raw_value: str | None,
    *,
    namespace: NamespaceConfig,
    default_filename: str | None = None,
) -> Path | None:
    if raw_value is None:
        if default_filename is None:
            return None
        if namespace.base_dir is not None:
            return (namespace.reports_dir / default_filename).resolve()
        return (PROJECT_ROOT / "reports" / default_filename).resolve()

    path = Path(raw_value)
    if path.is_absolute():
        return path.resolve()

    if namespace.base_dir is not None:
        if len(path.parts) == 1:
            return (namespace.reports_dir / path).resolve()
        if path.parts and path.parts[0] == "reports":
            tail = Path(*path.parts[1:]) if len(path.parts) > 1 else Path(default_filename or "")
            return (namespace.reports_dir / tail).resolve()

    return (PROJECT_ROOT / path).resolve()


def infer_policyb_support_manifest(run_name: str | None, seed: int) -> str | None:
    if not run_name:
        return None
    match = re.search(r"(?:^|_)k(\d+)(?:_|$)", run_name)
    if match is None:
        return None
    k_shot = match.group(1)
    return f"mimic_common5_policyB_support_k{k_shot}_seed{seed}.csv"


def default_policyb_source_report(seed: int) -> str:
    return f"policyB_no_adaptation_eval_seed{seed}.json"


def enforce_policy_b_manifest_guard(
    label_policy: str,
    *,
    train_manifest: Path | None = None,
    val_manifest: Path | None = None,
    test_manifest: Path | None = None,
) -> None:
    if label_policy != "uignore_blankzero":
        return

    split_paths = {
        "train": train_manifest,
        "val": val_manifest,
        "test": test_manifest,
    }
    for split_name, manifest_path in split_paths.items():
        if manifest_path is None:
            continue
        basename = manifest_path.name
        required_basename = POLICYB_TARGET_MANIFEST_BASENAMES[split_name]
        if basename in LEGACY_TARGET_MANIFEST_BASENAMES:
            raise ValueError(
                f"--label_policy uignore_blankzero refuses legacy target manifest {manifest_path}. "
                f"Use {required_basename} instead."
            )
        if basename.startswith("mimic_common5") and basename != required_basename:
            raise ValueError(
                f"--label_policy uignore_blankzero requires {required_basename} for the {split_name} split, "
                f"but resolved {manifest_path}."
            )


def collect_missing_paths(required_paths: dict[str, Path | None]) -> list[str]:
    failures: list[str] = []
    for label, path in required_paths.items():
        if path is None:
            failures.append(f"Missing required argument for {label}.")
            continue
        if not path.exists():
            failures.append(f"Missing required file for {label}: {path}")
        elif not path.is_file():
            failures.append(f"Expected a file for {label}, but found something else: {path}")
    return failures


def print_resolved_configuration(
    *,
    script_name: str,
    base_dir: Path | None,
    run_name: str | None,
    label_policy: str | None,
    train_manifest: Path | None = None,
    val_manifest: Path | None = None,
    test_manifest: Path | None = None,
    support_manifest: Path | None = None,
    source_checkpoint: Path | None = None,
    source_only_report: Path | None = None,
    checkpoint_output_path: Path | None = None,
    prediction_val_output_path: Path | None = None,
    prediction_test_output_path: Path | None = None,
    report_output_path: Path | None = None,
    report_markdown_path: Path | None = None,
) -> None:
    print(f"script: {script_name}")
    print(f"base_dir: {base_dir if base_dir is not None else 'n/a'}")
    print(f"run_name: {run_name if run_name else 'n/a'}")
    print(f"label_policy: {label_policy if label_policy else 'n/a'}")
    print(f"train manifest: {train_manifest if train_manifest is not None else 'n/a'}")
    print(f"val manifest: {val_manifest if val_manifest is not None else 'n/a'}")
    print(f"test manifest: {test_manifest if test_manifest is not None else 'n/a'}")
    print(f"support manifest: {support_manifest if support_manifest is not None else 'n/a'}")
    print(f"source checkpoint: {source_checkpoint if source_checkpoint is not None else 'n/a'}")
    print(f"source-only report: {source_only_report if source_only_report is not None else 'n/a'}")
    print(
        "checkpoint output path: "
        f"{checkpoint_output_path if checkpoint_output_path is not None else 'n/a'}"
    )
    print(
        "prediction output path (val): "
        f"{prediction_val_output_path if prediction_val_output_path is not None else 'n/a'}"
    )
    print(
        "prediction output path (test): "
        f"{prediction_test_output_path if prediction_test_output_path is not None else 'n/a'}"
    )
    print(f"report output path: {report_output_path if report_output_path is not None else 'n/a'}")
    print(
        "report markdown path: "
        f"{report_markdown_path if report_markdown_path is not None else 'n/a'}"
    )


def build_named_run_paths(
    namespace: NamespaceConfig,
    run_name: str,
    *,
    include_checkpoints: bool = False,
    include_loss_curve: bool = False,
) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "report_json": (namespace.reports_dir / f"{run_name}.json").resolve(),
        "report_md": (namespace.reports_dir / f"{run_name}.md").resolve(),
        "val_predictions": (namespace.outputs_dir / f"{run_name}_val_predictions.csv").resolve(),
        "test_predictions": (namespace.outputs_dir / f"{run_name}_test_predictions.csv").resolve(),
    }
    if include_checkpoints:
        paths["best_checkpoint"] = (namespace.checkpoints_dir / f"{run_name}_best.pt").resolve()
        paths["last_checkpoint"] = (namespace.checkpoints_dir / f"{run_name}_last.pt").resolve()
    if include_loss_curve:
        paths["loss_curve"] = (namespace.reports_dir / f"{run_name}_loss_curve.png").resolve()
    return paths
