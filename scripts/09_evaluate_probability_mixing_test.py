#!/usr/bin/env python3
"""Evaluate held-out test probability mixing using a frozen validation-selected alpha."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_module(script_name: str, module_name: str) -> tuple[Path, Any]:
    script_path = Path(__file__).resolve().with_name(script_name)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return script_path, module


MIXING_SELECTION_SCRIPT_PATH, probability_mixing_mod = load_module(
    "07_evaluate_probability_mixing.py",
    "source_probability_mixing_validation",
)

DEFAULT_MANIFEST_CSV = probability_mixing_mod.DEFAULT_MANIFEST_CSV
DEFAULT_MEMORY_TEST_ROOT = Path(
    "/workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test"
)
DEFAULT_MIXING_SELECTION_ROOT = Path(
    "/workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val"
)
DEFAULT_BASELINE_EXPERIMENT_DIR = probability_mixing_mod.DEFAULT_BASELINE_EXPERIMENT_DIR
DEFAULT_QUERY_EMBEDDING_ROOT = probability_mixing_mod.DEFAULT_QUERY_EMBEDDING_ROOT
DEFAULT_EXPERIMENTS_ROOT = probability_mixing_mod.DEFAULT_EXPERIMENTS_ROOT
DEFAULT_OPERATION_LABEL = "source_probability_mixing_test_evaluation"
DEFAULT_EXPERIMENT_ID_WIDTH = probability_mixing_mod.DEFAULT_EXPERIMENT_ID_WIDTH
DEFAULT_SPLIT = "test"
DEFAULT_BATCH_SIZE = probability_mixing_mod.DEFAULT_BATCH_SIZE
DEFAULT_SEED = probability_mixing_mod.DEFAULT_SEED
FROZEN_SELECTION_MODE = "frozen_validation_config"


def ensure_operation_prefix(name: str, operation_label: str = DEFAULT_OPERATION_LABEL) -> str:
    normalized_label = probability_mixing_mod.slugify(operation_label, fallback="operation")
    if name.startswith("exp") and "__" in name:
        prefix, remainder = name.split("__", 1)
        if remainder.startswith(normalized_label):
            return name
        return f"{prefix}__{normalized_label}__{remainder}"
    if name.startswith(normalized_label):
        return name
    return f"{normalized_label}__{name}"


def resolve_experiment_identity(
    *,
    experiments_root: Path,
    requested_name: str | None,
    generated_slug: str,
    overwrite: bool,
    id_width: int = DEFAULT_EXPERIMENT_ID_WIDTH,
) -> tuple[int, str, str, Path]:
    experiments_root.mkdir(parents=True, exist_ok=True)
    if requested_name:
        requested_name = requested_name.strip()
        if not requested_name:
            raise SystemExit("--experiment-name cannot be empty.")
    base_name = ensure_operation_prefix(requested_name or generated_slug)
    explicit_number = probability_mixing_mod.extract_experiment_number(base_name)
    if explicit_number is not None:
        experiment_number = explicit_number
        experiment_name = base_name
    else:
        experiment_number = probability_mixing_mod.next_experiment_number(experiments_root)
        experiment_name = f"exp{experiment_number:0{id_width}d}__{base_name}"

    experiment_id = f"exp{experiment_number:0{id_width}d}"
    experiment_dir = experiments_root / experiment_name
    if experiment_dir.exists() and not overwrite:
        raise SystemExit(
            f"Experiment directory already exists: {experiment_dir}\n"
            "Pass --overwrite to reuse it or choose a different --experiment-name."
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_number, experiment_id, experiment_name, experiment_dir


def format_shell_command(argv: list[str]) -> str:
    return shlex.join(argv)


def format_metric(value: float | None) -> str:
    return probability_mixing_mod.format_metric(value)


def extract_thresholds(
    threshold_payload: dict[str, Any],
    label_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    values: list[float] = []
    normalized_payload: dict[str, Any] = {}
    for label_name in label_names:
        current = threshold_payload.get(label_name)
        if not isinstance(current, dict) or "threshold" not in current:
            raise SystemExit(f"Frozen threshold for label '{label_name}' is missing from the validation artifact.")
        threshold = float(current["threshold"])
        values.append(threshold)
        normalized_payload[label_name] = {
            "threshold": threshold,
            "best_f1_on_val": current.get("best_f1"),
            "prevalence_on_val": current.get("prevalence"),
        }
    return np.asarray(values, dtype=np.float32), normalized_payload


def evaluate_probabilities_with_frozen_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    label_names: list[str],
    thresholds: np.ndarray,
    threshold_payload: dict[str, Any],
) -> dict[str, Any]:
    per_label: dict[str, dict[str, Any]] = {}
    macro_auroc_values: list[float] = []
    macro_ap_values: list[float] = []
    macro_f1_values: list[float] = []
    macro_f1_frozen_values: list[float] = []
    macro_ece_values: list[float] = []

    for idx, label_name in enumerate(label_names):
        targets = y_true[:, idx]
        scores = probs[:, idx]
        threshold = float(thresholds[idx])
        auroc = probability_mixing_mod.binary_auroc(targets, scores)
        average_precision = probability_mixing_mod.binary_average_precision(targets, scores)
        f1_at_0p5 = probability_mixing_mod.binary_f1(targets, scores, 0.5)
        f1_at_frozen = probability_mixing_mod.binary_f1(targets, scores, threshold)
        ece = probability_mixing_mod.binary_ece(targets, scores)

        if auroc is not None:
            macro_auroc_values.append(auroc)
        if average_precision is not None:
            macro_ap_values.append(average_precision)
        macro_f1_values.append(f1_at_0p5)
        macro_f1_frozen_values.append(f1_at_frozen)
        macro_ece_values.append(ece)

        current_threshold_payload = dict(threshold_payload[label_name])
        per_label[label_name] = {
            "auroc": auroc,
            "average_precision": average_precision,
            "ece": ece,
            "f1_at_0.5": f1_at_0p5,
            "f1_at_frozen_threshold": f1_at_frozen,
            "positive_count": int(targets.sum()),
            "threshold": threshold,
            "frozen_threshold_source": current_threshold_payload,
        }

    return {
        "macro_auroc": float(np.mean(macro_auroc_values)) if macro_auroc_values else None,
        "macro_average_precision": float(np.mean(macro_ap_values)) if macro_ap_values else None,
        "macro_ece": float(np.mean(macro_ece_values)) if macro_ece_values else None,
        "macro_f1_at_0.5": float(np.mean(macro_f1_values)) if macro_f1_values else None,
        "macro_f1_at_frozen_thresholds": (
            float(np.mean(macro_f1_frozen_values)) if macro_f1_frozen_values else None
        ),
        "thresholds": threshold_payload,
        "per_label": per_label,
    }


def compare_baseline_to_archived_test(
    reconstructed_metrics: dict[str, Any],
    archived_test_metrics: dict[str, Any],
    label_names: list[str],
) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    max_abs_delta = 0.0
    macro_keys = {
        "macro_auroc": ("macro", "auroc"),
        "macro_average_precision": ("macro", "average_precision"),
        "macro_ece": ("macro", "ece"),
        "macro_f1_at_0.5": ("macro", "f1_at_0.5"),
        "macro_f1_at_frozen_thresholds": ("macro", "f1_at_tuned_thresholds"),
    }
    macro_deltas: dict[str, Any] = {}
    for target_key, archived_path in macro_keys.items():
        archived_value = archived_test_metrics[archived_path[0]][archived_path[1]]
        reconstructed_value = reconstructed_metrics[target_key]
        delta = None
        if archived_value is not None and reconstructed_value is not None:
            delta = float(reconstructed_value - archived_value)
            max_abs_delta = max(max_abs_delta, abs(delta))
        macro_deltas[target_key] = {
            "archived": archived_value,
            "reconstructed": reconstructed_value,
            "delta": delta,
        }

    for label_name in label_names:
        archived_label = archived_test_metrics["label_metrics"][label_name]
        reconstructed_label = reconstructed_metrics["per_label"][label_name]
        current: dict[str, Any] = {}
        for metric_name, archived_key, reconstructed_key in (
            ("auroc", "auroc", "auroc"),
            ("average_precision", "average_precision", "average_precision"),
            ("ece", "ece", "ece"),
            ("f1_at_0.5", "f1_at_0.5", "f1_at_0.5"),
            ("f1_at_frozen_threshold", "f1_at_tuned_threshold", "f1_at_frozen_threshold"),
        ):
            archived_value = archived_label.get(archived_key)
            reconstructed_value = reconstructed_label.get(reconstructed_key)
            delta = None
            if archived_value is not None and reconstructed_value is not None:
                delta = float(reconstructed_value - archived_value)
                max_abs_delta = max(max_abs_delta, abs(delta))
            current[metric_name] = {
                "archived": archived_value,
                "reconstructed": reconstructed_value,
                "delta": delta,
            }
        per_label[label_name] = current

    return {
        "macro": macro_deltas,
        "per_label": per_label,
        "max_abs_delta": float(max_abs_delta),
        "comparison_scope": "forward_metrics_plus_frozen_val_threshold_f1_on_test",
        "matches_archived_metrics_within_5e-4": bool(max_abs_delta <= 5e-4),
    }


def build_recreation_report(
    *,
    experiment_dir: Path,
    experiment_id: str,
    operation_label: str,
    script_path: Path,
    argv_exact: list[str],
    argv_fresh: list[str],
    memory_test_root: Path,
    mixing_selection_root: Path,
    baseline_experiment_dir: Path,
    query_embedding_root: Path,
    manifest_csv: Path,
    split: str,
    row_count: int,
    label_names: list[str],
    applied_alpha: float,
    mixed_test_metrics: dict[str, Any],
    baseline_test_metrics: dict[str, Any],
    delta_summary: dict[str, float | None],
    baseline_comparison: dict[str, Any],
    output_paths: list[Path],
) -> str:
    size_lines = [f"- {path.name}: `{probability_mixing_mod.human_size(path.stat().st_size)}`" for path in output_paths if path.exists()]
    total_size = sum(path.stat().st_size for path in output_paths if path.exists())
    hash_lines = [f"{probability_mixing_mod.sha256_file(path)}  {path}" for path in output_paths if path.exists()]
    lines = [
        "# Source Probability-Mixing Test Evaluation Recreation Report",
        "",
        "## Scope",
        "",
        "This report documents how to recreate the held-out test probability-mixing experiment stored at:",
        "",
        f"`{experiment_dir}`",
        "",
        "The producing script is:",
        "",
        f"`{script_path}`",
        "",
        "Script SHA-256:",
        "",
        f"`{probability_mixing_mod.script_sha256(script_path)}`",
        "",
        "## Final Experiment Identity",
        "",
        f"- Experiment directory: `{experiment_dir}`",
        f"- Experiment id: `{experiment_id}`",
        f"- Operation label: `{operation_label}`",
        f"- Memory-test root: `{memory_test_root}`",
        f"- Validation mixing-selection root: `{mixing_selection_root}`",
        f"- Baseline experiment: `{baseline_experiment_dir}`",
        f"- Query embedding root: `{query_embedding_root}`",
        f"- Manifest: `{manifest_csv}`",
        f"- Evaluation split: `{split}`",
        f"- Test rows: `{row_count:,}`",
        f"- Label count: `{len(label_names)}`",
        f"- Label names: `{' '.join(label_names)}`",
        f"- Applied selection mode: `{FROZEN_SELECTION_MODE}`",
        "",
        "## Environment",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- NumPy: `{np.__version__}`",
        f"- PyTorch: `{probability_mixing_mod.torch.__version__}`",
        f"- Platform: `{platform.platform()}`",
        "",
        "## Exact Recreation Command",
        "",
        "If you want to recreate the same directory name in place, use this command:",
        "",
        "```bash",
        format_shell_command(argv_exact),
        "```",
        "",
        "If you want a fresh numbered run instead of overwriting the existing directory, use:",
        "",
        "```bash",
        format_shell_command(argv_fresh),
        "```",
        "",
        "## Preconditions",
        "",
        f"- The memory-only test evaluation must already exist at `{memory_test_root}`.",
        f"- The validation probability-mixing selection must already exist at `{mixing_selection_root}`.",
        f"- The baseline experiment must already exist at `{baseline_experiment_dir}`.",
        f"- The query embeddings must already exist at `{query_embedding_root / split}`.",
        f"- The manifest must be present at `{manifest_csv}`.",
        "- The required Python packages must be importable: `numpy`, `torch`.",
        "- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected alpha.",
        "",
        "## Input Summary",
        "",
        f"- Baseline checkpoint: `{baseline_experiment_dir / 'best.ckpt'}`",
        f"- Memory test probabilities: `{memory_test_root / 'test_probabilities.npy'}`",
        f"- Validation selection root: `{mixing_selection_root}`",
        f"- Query embedding split: `{query_embedding_root / split}`",
        f"- Test rows: `{row_count:,}`",
        "",
        "## Applied Configuration",
        "",
        f"- Frozen `alpha`: `{applied_alpha:.1f}`",
        f"- Threshold source: `{mixing_selection_root / 'best_val_metrics.json'}`",
        "",
        "## Test Metrics",
        "",
        f"- Mixed test macro AUROC: `{format_metric(mixed_test_metrics['macro_auroc'])}`",
        f"- Mixed test macro average precision: `{format_metric(mixed_test_metrics['macro_average_precision'])}`",
        f"- Mixed test macro ECE: `{format_metric(mixed_test_metrics['macro_ece'])}`",
        f"- Mixed test macro F1 @ 0.5: `{format_metric(mixed_test_metrics['macro_f1_at_0.5'])}`",
        f"- Mixed test macro F1 @ frozen val thresholds: `{format_metric(mixed_test_metrics['macro_f1_at_frozen_thresholds'])}`",
        "",
        "## Baseline Comparison",
        "",
        f"- Frozen baseline test macro AUROC: `{format_metric(baseline_test_metrics['macro_auroc'])}`",
        f"- Frozen baseline test macro average precision: `{format_metric(baseline_test_metrics['macro_average_precision'])}`",
        f"- Mixed minus baseline macro AUROC: `{format_metric(delta_summary['macro_auroc'])}`",
        f"- Mixed minus baseline macro average precision: `{format_metric(delta_summary['macro_average_precision'])}`",
        f"- Mixed minus baseline macro ECE: `{format_metric(delta_summary['macro_ece'])}`",
        f"- Mixed minus baseline macro F1 @ 0.5: `{format_metric(delta_summary['macro_f1_at_0.5'])}`",
        f"- Mixed minus baseline macro F1 @ frozen val thresholds: `{format_metric(delta_summary['macro_f1_at_frozen_thresholds'])}`",
        f"- Baseline reconstruction matches archived exp0006 test metrics within 5e-4: `{str(baseline_comparison['matches_archived_metrics_within_5e-4']).lower()}`",
        f"- Baseline reconstruction max absolute metric delta: `{baseline_comparison['max_abs_delta']:.12f}`",
        "",
        "## Expected Outputs",
        "",
        "- `experiment_meta.json`",
        "- `recreation_report.md`",
        "- `applied_config.json`",
        "- `test_metrics.json`",
        "- `test_mixed_probabilities.npy`",
        "- `probability_mixing_test_summary.md`",
        "",
        "## Output Sizes",
        "",
    ]
    lines.extend(size_lines)
    lines.extend(
        [
            f"- Total output size: `{probability_mixing_mod.human_size(total_size)}`",
            "",
            "## Final Artifact SHA-256",
            "",
            "```text",
            "\n".join(hash_lines),
            "```",
            "",
            "## Important Reproduction Notes",
            "",
            "- This test stage does not sweep alpha; it reuses the validation-selected alpha from `exp0010`.",
            "- Threshold-based F1 on test uses frozen thresholds from the validation mixing artifact, not thresholds retuned on test.",
            "- `test_mixed_probabilities.npy` stores held-out mixed probabilities in test row order and is small enough for plain Git.",
            "",
            "## Agent Handoff Text",
            "",
            "```text",
            (
                "Use /workspace/scripts/09_evaluate_probability_mixing_test.py and the report "
                f"{experiment_dir / 'recreation_report.md'} to recreate the held-out test probability-mixing stage "
                f"that combines {baseline_experiment_dir} with {memory_test_root}. Apply the frozen validation-"
                f"selected alpha from {mixing_selection_root}, reuse the validation thresholds from exp0010 for "
                "threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and "
                "test_mixed_probabilities.npy artifacts."
            ),
            "```",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out test probability mixing using the frozen validation-selected alpha."
    )
    parser.add_argument("--memory-test-root", type=Path, default=DEFAULT_MEMORY_TEST_ROOT)
    parser.add_argument("--mixing-selection-root", type=Path, default=DEFAULT_MIXING_SELECTION_ROOT)
    parser.add_argument("--baseline-experiment-dir", type=Path, default=DEFAULT_BASELINE_EXPERIMENT_DIR)
    parser.add_argument("--query-embedding-root", type=Path, default=DEFAULT_QUERY_EMBEDDING_ROOT)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["test"])
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    memory_test_root = args.memory_test_root.resolve()
    mixing_selection_root = args.mixing_selection_root.resolve()
    baseline_experiment_dir = args.baseline_experiment_dir.resolve()
    query_embedding_root = args.query_embedding_root.resolve()
    manifest_csv = args.manifest_csv.resolve()
    experiments_root = args.experiments_root.resolve()
    split = args.split

    generated_slug = "nih_cxr14_exp0010_test"
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=args.overwrite,
        id_width=DEFAULT_EXPERIMENT_ID_WIDTH,
    )

    script_path = Path(__file__).resolve()
    script_hash = probability_mixing_mod.script_sha256(script_path)

    label_columns, label_names, manifest_records = probability_mixing_mod.load_manifest_records(manifest_csv, split=split)
    test_embeddings, test_row_ids, test_sidecar_image_paths = probability_mixing_mod.load_query_split(
        query_embedding_root,
        split=split,
    )
    test_labels, test_manifest_image_paths = probability_mixing_mod.build_labels_from_records(test_row_ids, manifest_records)
    probability_mixing_mod.validate_query_alignment(test_row_ids, test_sidecar_image_paths, test_manifest_image_paths)
    normalized_test_embeddings, query_norm_summary_before, query_norm_summary_after = probability_mixing_mod.normalize_rows(
        test_embeddings
    )

    memory_test_meta = probability_mixing_mod.read_json(memory_test_root / "experiment_meta.json")
    memory_test_config = probability_mixing_mod.read_json(memory_test_root / "applied_config.json")
    memory_test_metrics = probability_mixing_mod.read_json(memory_test_root / "test_metrics.json")
    memory_test_probabilities = probability_mixing_mod.load_embedding_array(memory_test_root / "test_probabilities.npy")
    if memory_test_probabilities.shape != test_labels.shape:
        raise SystemExit(
            f"Memory test probability shape {memory_test_probabilities.shape} does not match test labels {test_labels.shape}."
        )

    mixing_selection_meta = probability_mixing_mod.read_json(mixing_selection_root / "experiment_meta.json")
    mixing_selection_config = probability_mixing_mod.read_json(mixing_selection_root / "best_config.json")
    mixing_selection_metrics = probability_mixing_mod.read_json(mixing_selection_root / "best_val_metrics.json")
    mixing_threshold_payload = mixing_selection_metrics.get("thresholds")
    if not isinstance(mixing_threshold_payload, dict):
        raise SystemExit(f"Validation threshold payload is missing from {mixing_selection_root / 'best_val_metrics.json'}")
    mixing_thresholds, normalized_mixing_threshold_payload = extract_thresholds(mixing_threshold_payload, label_names)
    applied_alpha = float(mixing_selection_config["alpha"])

    baseline_val_thresholds_payload = probability_mixing_mod.read_json(
        baseline_experiment_dir / "val_f1_thresholds.json"
    ).get("labels")
    if not isinstance(baseline_val_thresholds_payload, dict):
        raise SystemExit(
            f"Baseline validation threshold payload is missing from {baseline_experiment_dir / 'val_f1_thresholds.json'}"
        )
    baseline_thresholds, normalized_baseline_threshold_payload = extract_thresholds(
        baseline_val_thresholds_payload,
        label_names,
    )

    baseline_test_metrics_archived = probability_mixing_mod.read_json(baseline_experiment_dir / "test_metrics.json")
    baseline_checkpoint_path = baseline_experiment_dir / "best.ckpt"
    baseline_probabilities, baseline_reconstruction = probability_mixing_mod.reconstruct_baseline_probabilities(
        checkpoint_path=baseline_checkpoint_path,
        val_embeddings=normalized_test_embeddings,
        label_names=label_names,
        batch_size=int(args.batch_size),
    )
    if baseline_probabilities.shape != test_labels.shape:
        raise SystemExit(
            f"Baseline probability shape {baseline_probabilities.shape} does not match test labels {test_labels.shape}."
        )
    baseline_test_metrics = evaluate_probabilities_with_frozen_thresholds(
        test_labels,
        baseline_probabilities,
        label_names,
        baseline_thresholds,
        normalized_baseline_threshold_payload,
    )
    baseline_comparison = compare_baseline_to_archived_test(
        baseline_test_metrics,
        baseline_test_metrics_archived,
        label_names,
    )

    test_mixed_probabilities = probability_mixing_mod.mix_probabilities(
        baseline_probabilities,
        memory_test_probabilities,
        applied_alpha,
    )
    mixed_test_metrics = evaluate_probabilities_with_frozen_thresholds(
        test_labels,
        test_mixed_probabilities,
        label_names,
        mixing_thresholds,
        normalized_mixing_threshold_payload,
    )

    delta_summary: dict[str, float | None] = {}
    for key in (
        "macro_auroc",
        "macro_average_precision",
        "macro_ece",
        "macro_f1_at_0.5",
        "macro_f1_at_frozen_thresholds",
    ):
        mixed_value = mixed_test_metrics.get(key)
        baseline_value = baseline_test_metrics.get(key)
        if mixed_value is None or baseline_value is None:
            delta_summary[key] = None
        else:
            delta_summary[key] = float(mixed_value - baseline_value)

    experiment_meta_path = experiment_dir / "experiment_meta.json"
    recreation_report_path = experiment_dir / "recreation_report.md"
    applied_config_path = experiment_dir / "applied_config.json"
    test_metrics_path = experiment_dir / "test_metrics.json"
    test_mixed_probabilities_path = experiment_dir / "test_mixed_probabilities.npy"
    summary_path = experiment_dir / "probability_mixing_test_summary.md"

    probability_mixing_mod.write_json(
        applied_config_path,
        {
            "selection_mode": FROZEN_SELECTION_MODE,
            "mixing_selection_root": str(mixing_selection_root),
            "selection_metric": mixing_selection_config.get("selection_metric"),
            "selection_trace": mixing_selection_config.get("selection_trace"),
            "alpha": applied_alpha,
            "memory_test_root": str(memory_test_root),
            "memory_test_config": memory_test_config,
            "threshold_source": str(mixing_selection_root / "best_val_metrics.json"),
        },
    )
    probability_mixing_mod.write_json(test_metrics_path, mixed_test_metrics)
    np.save(test_mixed_probabilities_path, test_mixed_probabilities.astype(np.float32))

    summary_lines = [
        "# Probability-Mixing Test Evaluation",
        "",
        "The held-out test probability-mixing evaluation for the current source stage uses the frozen validation-selected alpha:",
        "",
        f"- selection root: `{mixing_selection_root}`",
        f"- memory test root: `{memory_test_root}`",
        f"- alpha: `{applied_alpha:.1f}`",
        f"- mixed test macro AUROC: `{format_metric(mixed_test_metrics['macro_auroc'])}`",
        f"- mixed test macro average precision: `{format_metric(mixed_test_metrics['macro_average_precision'])}`",
        f"- mixed test macro ECE: `{format_metric(mixed_test_metrics['macro_ece'])}`",
        f"- mixed test macro F1 @ 0.5: `{format_metric(mixed_test_metrics['macro_f1_at_0.5'])}`",
        f"- mixed test macro F1 @ frozen val thresholds: `{format_metric(mixed_test_metrics['macro_f1_at_frozen_thresholds'])}`",
        f"- delta vs frozen baseline macro AUROC: `{format_metric(delta_summary['macro_auroc'])}`",
        f"- delta vs frozen baseline macro average precision: `{format_metric(delta_summary['macro_average_precision'])}`",
        f"- delta vs frozen baseline macro ECE: `{format_metric(delta_summary['macro_ece'])}`",
        f"- delta vs frozen baseline macro F1 @ 0.5: `{format_metric(delta_summary['macro_f1_at_0.5'])}`",
        f"- delta vs frozen baseline macro F1 @ frozen val thresholds: `{format_metric(delta_summary['macro_f1_at_frozen_thresholds'])}`",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    experiment_meta = {
        "argv": sys.argv,
        "baseline_experiment_dir": str(baseline_experiment_dir),
        "baseline_reconstruction": baseline_reconstruction,
        "baseline_reconstruction_metric_check": baseline_comparison,
        "experiment_dir": str(experiment_dir),
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_number": experiment_number,
        "label_columns": label_columns,
        "label_names": label_names,
        "manifest_csv": str(manifest_csv),
        "memory_test_root": str(memory_test_root),
        "memory_test_reference": memory_test_meta,
        "memory_test_config": memory_test_config,
        "memory_test_metrics": memory_test_metrics,
        "mixing_selection_root": str(mixing_selection_root),
        "mixing_selection_reference": mixing_selection_meta,
        "operation_label": DEFAULT_OPERATION_LABEL,
        "query_embedding_root": str(query_embedding_root),
        "query_norm_summary": {
            "raw": query_norm_summary_before,
            "normalized": query_norm_summary_after,
        },
        "run_date_utc": probability_mixing_mod.utc_now_iso(),
        "script_path": str(script_path),
        "script_sha256": script_hash,
        "seed": args.seed,
        "selection_mode": FROZEN_SELECTION_MODE,
        "applied_config": {
            "alpha": applied_alpha,
        },
        "frozen_threshold_source": {
            "path": str(mixing_selection_root / "best_val_metrics.json"),
            "thresholds": normalized_mixing_threshold_payload,
        },
        "baseline_test_metrics": baseline_test_metrics,
        "mixed_test_metrics": mixed_test_metrics,
        "delta_vs_baseline": delta_summary,
        "artifacts": {
            "experiment_meta": str(experiment_meta_path),
            "recreation_report": str(recreation_report_path),
            "applied_config": str(applied_config_path),
            "test_metrics": str(test_metrics_path),
            "test_mixed_probabilities": str(test_mixed_probabilities_path),
            "summary": str(summary_path),
        },
    }
    probability_mixing_mod.write_json(experiment_meta_path, experiment_meta)

    argv_exact = [
        "python",
        str(script_path),
        "--memory-test-root",
        str(memory_test_root),
        "--mixing-selection-root",
        str(mixing_selection_root),
        "--baseline-experiment-dir",
        str(baseline_experiment_dir),
        "--query-embedding-root",
        str(query_embedding_root),
        "--manifest-csv",
        str(manifest_csv),
        "--split",
        split,
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--experiment-name",
        experiment_name,
        "--overwrite",
    ]
    argv_fresh = [
        "python",
        str(script_path),
        "--memory-test-root",
        str(memory_test_root),
        "--mixing-selection-root",
        str(mixing_selection_root),
        "--baseline-experiment-dir",
        str(baseline_experiment_dir),
        "--query-embedding-root",
        str(query_embedding_root),
        "--manifest-csv",
        str(manifest_csv),
        "--split",
        split,
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--experiment-name",
        probability_mixing_mod.strip_experiment_number_prefix(experiment_name),
    ]
    output_paths = [
        experiment_meta_path,
        recreation_report_path,
        applied_config_path,
        test_metrics_path,
        test_mixed_probabilities_path,
        summary_path,
    ]
    recreation_report = build_recreation_report(
        experiment_dir=experiment_dir,
        experiment_id=experiment_id,
        operation_label=DEFAULT_OPERATION_LABEL,
        script_path=script_path,
        argv_exact=argv_exact,
        argv_fresh=argv_fresh,
        memory_test_root=memory_test_root,
        mixing_selection_root=mixing_selection_root,
        baseline_experiment_dir=baseline_experiment_dir,
        query_embedding_root=query_embedding_root,
        manifest_csv=manifest_csv,
        split=split,
        row_count=int(test_embeddings.shape[0]),
        label_names=label_names,
        applied_alpha=applied_alpha,
        mixed_test_metrics=mixed_test_metrics,
        baseline_test_metrics=baseline_test_metrics,
        delta_summary=delta_summary,
        baseline_comparison=baseline_comparison,
        output_paths=output_paths,
    )
    recreation_report_path.write_text(recreation_report + "\n", encoding="utf-8")

    print(f"[saved] experiment_dir={experiment_dir}")
    print(
        "[applied_config] "
        f"alpha={applied_alpha:.1f} "
        f"macro_auroc={format_metric(mixed_test_metrics['macro_auroc'])} "
        f"macro_ap={format_metric(mixed_test_metrics['macro_average_precision'])} "
        f"macro_ece={format_metric(mixed_test_metrics['macro_ece'])} "
        f"macro_f1_0p5={format_metric(mixed_test_metrics['macro_f1_at_0.5'])} "
        f"macro_f1_frozen={format_metric(mixed_test_metrics['macro_f1_at_frozen_thresholds'])}"
    )
    print(
        "[baseline_compare] "
        f"baseline_macro_auroc={format_metric(baseline_test_metrics['macro_auroc'])} "
        f"delta_vs_baseline={format_metric(delta_summary['macro_auroc'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
