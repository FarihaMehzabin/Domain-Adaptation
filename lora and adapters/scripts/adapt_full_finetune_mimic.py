#!/usr/bin/env python3
"""Full fine-tuning baseline from NIH to MIMIC."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_namespace import (  # noqa: E402
    POLICYB_TARGET_MANIFEST_BASENAMES,
    build_named_run_paths,
    build_namespace_config,
    collect_missing_paths,
    default_policyb_source_report,
    enforce_policy_b_manifest_guard,
    infer_policyb_support_manifest,
    print_resolved_configuration,
    resolve_input_path,
    resolve_manifest_path,
    resolve_report_input,
)
from scripts.adapt_head_only_mimic import (  # noqa: E402
    LABELS,
    StageFailure,
    build_dataloader,
    build_model,
    check_leakage,
    check_required_files,
    empty_split_summary,
    ensure_dir,
    evaluate_split,
    format_metric,
    json_ready,
    load_checkpoint,
    load_source_only_report,
    metric_delta,
    metrics_for_report,
    print_split_summary,
    save_json,
    save_predictions_csv,
    select_is_better,
    set_seed,
    validate_manifest,
)
from scripts.masked_multilabel_utils import masked_bce_with_logits_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full DenseNet-121 fine-tuning from NIH to MIMIC.")
    parser.add_argument("--checkpoint", "--source_checkpoint", dest="checkpoint", type=str, default=None)
    parser.add_argument("--support_csv", "--support_manifest", dest="support_csv", type=str, default=None)
    parser.add_argument("--val_csv", "--manifest_val", dest="val_csv", type=str, default=None)
    parser.add_argument("--test_csv", "--manifest_test", dest="test_csv", type=str, default=None)
    parser.add_argument("--source_only_report", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--label_policy", type=str, default="uignore_blankzero")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    namespace = build_namespace_config(args.base_dir, args.out_dir)
    if args.run_name is None:
        args.run_name = f"full_ft_seed{args.seed}"

    args.base_dir = str(namespace.base_dir) if namespace.base_dir is not None else None
    args.out_dir = str(namespace.output_root)
    args.checkpoint = str(
        resolve_input_path(args.checkpoint, default_relative="checkpoints/nih_2k_densenet121_best.pt")
    )

    support_default = None
    if args.label_policy == "uignore_blankzero":
        support_default = infer_policyb_support_manifest(args.run_name, args.seed)

    support_manifest = resolve_manifest_path(
        args.support_csv,
        namespace=namespace,
        default_filename=support_default,
    )
    val_manifest = resolve_manifest_path(
        args.val_csv,
        namespace=namespace,
        default_filename=(
            POLICYB_TARGET_MANIFEST_BASENAMES["val"]
            if args.label_policy == "uignore_blankzero"
            else None
        ),
    )
    test_manifest = resolve_manifest_path(
        args.test_csv,
        namespace=namespace,
        default_filename=(
            POLICYB_TARGET_MANIFEST_BASENAMES["test"]
            if args.label_policy == "uignore_blankzero"
            else None
        ),
    )
    source_only_report = resolve_report_input(
        args.source_only_report,
        namespace=namespace,
        default_filename=(
            default_policyb_source_report(args.seed)
            if namespace.base_dir is not None and args.label_policy == "uignore_blankzero"
            else "stage5_source_baseline.json"
        ),
    )

    args.support_csv = str(support_manifest) if support_manifest is not None else None
    args.val_csv = str(val_manifest) if val_manifest is not None else None
    args.test_csv = str(test_manifest) if test_manifest is not None else None
    args.source_only_report = str(source_only_report) if source_only_report is not None else None
    args._namespace_config = namespace
    args._artifact_paths = build_named_run_paths(
        namespace,
        args.run_name,
        include_checkpoints=True,
        include_loss_curve=True,
    )
    return args


def required_input_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    return {
        "checkpoint": Path(args.checkpoint) if args.checkpoint else None,
        "support_manifest": Path(args.support_csv) if args.support_csv else None,
        "val_manifest": Path(args.val_csv) if args.val_csv else None,
        "test_manifest": Path(args.test_csv) if args.test_csv else None,
        "source_only_report": Path(args.source_only_report) if args.source_only_report else None,
    }


def print_runtime_configuration(args: argparse.Namespace) -> None:
    artifact_paths = args._artifact_paths
    print_resolved_configuration(
        script_name=Path(__file__).name,
        base_dir=Path(args.base_dir) if args.base_dir else None,
        run_name=args.run_name,
        label_policy=args.label_policy,
        val_manifest=Path(args.val_csv) if args.val_csv else None,
        test_manifest=Path(args.test_csv) if args.test_csv else None,
        support_manifest=Path(args.support_csv) if args.support_csv else None,
        source_checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        source_only_report=Path(args.source_only_report) if args.source_only_report else None,
        checkpoint_output_path=artifact_paths["best_checkpoint"],
        prediction_val_output_path=artifact_paths["val_predictions"],
        prediction_test_output_path=artifact_paths["test_predictions"],
        report_output_path=artifact_paths["report_json"],
        report_markdown_path=artifact_paths["report_md"],
    )


def run_dry_run(args: argparse.Namespace) -> None:
    try:
        enforce_policy_b_manifest_guard(
            args.label_policy,
            val_manifest=Path(args.val_csv) if args.val_csv else None,
            test_manifest=Path(args.test_csv) if args.test_csv else None,
            support_manifest=Path(args.support_csv) if args.support_csv else None,
        )
    except ValueError as exc:
        raise StageFailure(str(exc)) from exc
    print_runtime_configuration(args)
    missing_files = collect_missing_paths(required_input_paths(args))
    if missing_files:
        raise StageFailure("Dry run failed:\n- " + "\n- ".join(missing_files))
    support_df, support_summary = validate_manifest(Path(args.support_csv), "support", args.label_policy)
    val_df, val_summary = validate_manifest(Path(args.val_csv), "val", args.label_policy)
    test_df, test_summary = validate_manifest(Path(args.test_csv), "test", args.label_policy)
    _ = support_df, val_df, test_df
    print_split_summary(support_summary)
    print_split_summary(val_summary)
    print_split_summary(test_summary)
    print("dry_run: configuration resolved and all required files exist")


def summarize_parameter_trainability(model: nn.Module) -> dict[str, Any]:
    total_parameters = 0
    trainable_parameters = 0
    parameter_names: list[str] = []
    trainable_parameter_names: list[str] = []
    frozen_parameter_names: list[str] = []

    for name, parameter in model.named_parameters():
        parameter_names.append(name)
        total_parameters += parameter.numel()
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()
            trainable_parameter_names.append(name)
        else:
            frozen_parameter_names.append(name)

    return {
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "total_parameter_tensors": int(len(parameter_names)),
        "trainable_parameter_tensors": int(len(trainable_parameter_names)),
        "frozen_parameter_tensors": int(len(frozen_parameter_names)),
        "trainable_parameter_names": trainable_parameter_names,
        "frozen_parameter_names": frozen_parameter_names,
        "trainable_parameter_names_sample": trainable_parameter_names[:20],
        "frozen_parameter_names_sample": frozen_parameter_names[:20],
    }


def enable_full_finetune(model: nn.Module) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad = True
    return summarize_parameter_trainability(model)


def verify_full_finetune_trainability(summary: dict[str, Any]) -> None:
    required_classifier_names = {"classifier.weight", "classifier.bias"}
    trainable_names = set(summary["trainable_parameter_names"])
    frozen_names = set(summary["frozen_parameter_names"])

    if summary["trainable_parameter_tensors"] == 0:
        raise StageFailure("No trainable parameters remain, but full fine-tuning requires all parameters.")
    if summary["total_parameter_tensors"] == 0:
        raise StageFailure("Model has no parameters to fine-tune.")
    if frozen_names:
        raise StageFailure(
            "Some parameters are unexpectedly non-trainable during full fine-tuning: "
            f"{summary['frozen_parameter_names_sample']}"
        )
    if summary["trainable_parameters"] != summary["total_parameters"]:
        raise StageFailure(
            "Full fine-tuning requires all parameters to be trainable, but the parameter counts do not match."
        )
    if summary["trainable_parameter_tensors"] != summary["total_parameter_tensors"]:
        raise StageFailure(
            "Full fine-tuning requires every parameter tensor to be trainable."
        )
    if not required_classifier_names.issubset(trainable_names):
        missing = sorted(required_classifier_names - trainable_names)
        raise StageFailure(
            "Classifier parameters are not trainable during full fine-tuning: "
            f"{missing}"
        )


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_valid = 0.0

    for images, labels, masks in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = masked_bce_with_logits_loss(logits, labels, masks)
        if not torch.isfinite(loss):
            raise StageFailure("Training loss became NaN or infinite.")
        loss.backward()
        optimizer.step()

        valid_count = float(masks.sum().item())
        total_loss += float(loss.item()) * valid_count
        total_valid += valid_count

    return float(total_loss / total_valid) if total_valid > 0 else 0.0


def save_loss_curve(train_losses: list[float], val_losses: list[float], output_path: Path) -> str:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, marker="o", label="support train loss")
    plt.plot(epochs, val_losses, marker="o", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Full fine-tuning adaptation loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path.resolve())


def checkpoint_payload(
    model: nn.Module,
    epoch: int,
    train_loss: float,
    val_metrics: dict[str, Any],
    args: argparse.Namespace,
    source_checkpoint: str,
    parameter_summary: dict[str, Any],
    best_metric_name: str,
    best_metric_value: float,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_metrics["loss"]),
        "label_names": LABELS,
        "num_classes": len(LABELS),
        "architecture": "torchvision_densenet121",
        "image_size": int(args.image_size),
        "adaptation_method": "full_finetune",
        "source_checkpoint": source_checkpoint,
        "best_metric_name": best_metric_name,
        "best_metric_value": float(best_metric_value),
        "parameter_summary": parameter_summary,
        "model_state_dict": model.state_dict(),
    }


def initial_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "FAILED",
        "safe_to_continue": False,
        "goal": "Adapt the NIH-trained DenseNet-121 model to MIMIC using full fine-tuning on the support set.",
        "run_name": args.run_name,
        "adaptation_method": "full_finetune",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "support_manifest": str(Path(args.support_csv).resolve()) if args.support_csv else None,
        "val_manifest": str(Path(args.val_csv).resolve()) if args.val_csv else None,
        "test_manifest": str(Path(args.test_csv).resolve()) if args.test_csv else None,
        "source_only_report": str(Path(args.source_only_report).resolve()) if args.source_only_report else None,
        "label_policy": args.label_policy,
        "base_dir": args.base_dir,
        "label_order": LABELS,
        "training_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "image_size": int(args.image_size),
            "patience": int(args.patience),
            "seed": int(args.seed),
        },
        "checkpoint_metadata": {},
        "support_counts": empty_split_summary("support"),
        "val_counts": empty_split_summary("val"),
        "test_counts": empty_split_summary("test"),
        "leakage_checks": {},
        "parameter_summary": {},
        "training_history": [],
        "best_epoch": None,
        "stopped_early": False,
        "best_checkpoint": None,
        "last_checkpoint": None,
        "prediction_files": {},
        "plot_files": {},
        "source_only_metrics": {},
        "metric_deltas": {},
        "val_metrics": {},
        "test_metrics": {},
        "device": None,
        "warnings": [],
        "failure_reasons": [],
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Mini-Stage G Full Fine-Tune NIH to MIMIC Adaptation",
        "",
        "## Goal",
        "Adapt the NIH-trained DenseNet-121 model to MIMIC by fine-tuning all DenseNet parameters on the small support set.",
        "",
        "## Inputs",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- support manifest: `{report['support_manifest']}`",
        f"- val manifest: `{report['val_manifest']}`",
        f"- test manifest: `{report['test_manifest']}`",
        f"- source-only report: `{report['source_only_report']}`",
        "",
        "## Training Setup",
        f"- run name: `{report['run_name']}`",
        f"- epochs: {report['training_config']['epochs']}",
        f"- batch size: {report['training_config']['batch_size']}",
        f"- learning rate: {report['training_config']['lr']}",
        f"- image size: {report['training_config']['image_size']}",
        f"- patience: {report['training_config']['patience']}",
        f"- seed: {report['training_config']['seed']}",
        f"- device: `{report['device']}`",
        "",
        "## Split Sizes",
    ]

    for split_name in ["support_counts", "val_counts", "test_counts"]:
        split = report[split_name]
        lines.append(f"### {split['split_name']}")
        lines.append(f"- images: {split['num_images']}")
        lines.append(f"- subjects: {split['num_subjects']}")
        lines.append(f"- studies: {split['num_studies']}")
        lines.append(f"- label policy: {split.get('label_policy', 'n/a')}")
        if split["dicom_id_available"]:
            lines.append(f"- dicoms: {split['num_dicoms']}")
        else:
            lines.append("- dicom_id: not available")
        for label in LABELS:
            counts = split["label_counts"][label]
            lines.append(
                f"- {label}: positives={counts['positives']}, negatives={counts['negatives']}, "
                f"masked={counts['masked']}, n_valid={counts['n_valid']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Parameter Trainability Check",
            f"- total parameters: {report['parameter_summary'].get('total_parameters')}",
            f"- trainable parameters: {report['parameter_summary'].get('trainable_parameters')}",
            f"- total parameter tensors: {report['parameter_summary'].get('total_parameter_tensors')}",
            f"- trainable parameter tensors: {report['parameter_summary'].get('trainable_parameter_tensors')}",
            f"- frozen parameter tensors: {report['parameter_summary'].get('frozen_parameter_tensors')}",
            f"- trainable parameter names sample: {report['parameter_summary'].get('trainable_parameter_names_sample')}",
            "",
            "## Training Outcome",
            f"- best epoch: {report['best_epoch']}",
            f"- stopped early: {'yes' if report['stopped_early'] else 'no'}",
            f"- best checkpoint: `{report['best_checkpoint']}`",
            f"- last checkpoint: `{report['last_checkpoint']}`",
        ]
    )

    if report["training_history"]:
        last_epoch = report["training_history"][-1]
        lines.append(f"- final train loss: {last_epoch['train_loss']:.4f}")
        lines.append(f"- final val loss: {last_epoch['val_loss']:.4f}")
        lines.append(f"- final val macro AUROC: {format_metric(last_epoch['val_macro_auroc'])}")
        lines.append(f"- final val macro AUPRC: {format_metric(last_epoch['val_macro_auprc'])}")

    for section_name in ["val_metrics", "test_metrics"]:
        metrics = report.get(section_name, {})
        lines.extend(["", f"## {section_name.replace('_', ' ').title()}"])
        if not metrics:
            lines.append("- not available")
            continue
        lines.append(f"- loss: {metrics['loss']:.4f}")
        lines.append(f"- macro AUROC: {format_metric(metrics['macro_auroc'])}")
        lines.append(f"- macro AUPRC: {format_metric(metrics['macro_auprc'])}")
        lines.append("- micro AUROC: n/a")
        lines.append("- micro AUPRC: n/a")
        for label in LABELS:
            item = metrics["per_label"][label]
            prob_mean = "n/a" if item["probability_mean"] is None else f"{item['probability_mean']:.4f}"
            prob_std = "n/a" if item["probability_std"] is None else f"{item['probability_std']:.4f}"
            lines.append(
                f"- {label}: "
                f"positives={item['positives']}, "
                f"negatives={item['negatives']}, "
                f"masked={item['masked']}, "
                f"n_valid={item['n_valid']}, "
                f"AUROC={format_metric(item['auroc'])}, "
                f"AUPRC={format_metric(item['auprc'])}, "
                f"prob_mean={prob_mean}, "
                f"prob_std={prob_std}"
            )
            if "reason" in item:
                lines.append(f"- {label} note: {item['reason']}")

    if report["source_only_metrics"]:
        lines.extend(
            [
                "",
                "## Source-only Comparison",
                f"- source-only val macro AUROC: {format_metric(report['source_only_metrics']['val']['macro_auroc'])}",
                f"- source-only val macro AUPRC: {format_metric(report['source_only_metrics']['val']['macro_auprc'])}",
                f"- source-only test macro AUROC: {format_metric(report['source_only_metrics']['test']['macro_auroc'])}",
                f"- source-only test macro AUPRC: {format_metric(report['source_only_metrics']['test']['macro_auprc'])}",
                f"- val macro AUROC delta: {format_metric(report['metric_deltas'].get('val_macro_auroc_delta'))}",
                f"- val macro AUPRC delta: {format_metric(report['metric_deltas'].get('val_macro_auprc_delta'))}",
                f"- test macro AUROC delta: {format_metric(report['metric_deltas'].get('test_macro_auroc_delta'))}",
                f"- test macro AUPRC delta: {format_metric(report['metric_deltas'].get('test_macro_auprc_delta'))}",
            ]
        )

    lines.extend(["", "## Warnings"])
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")

    lines.extend(["", "## Final Decision"])
    lines.append(f"- status: {report['status']}")
    lines.append(f"- safe to continue: {'yes' if report['safe_to_continue'] else 'no'}")

    if report["failure_reasons"]:
        lines.extend(["", "## Failure Reasons"])
        for reason in report["failure_reasons"]:
            lines.append(f"- {reason}")

    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], report_json_path: Path, report_md_path: Path) -> None:
    save_json(json_ready(report), report_json_path)
    ensure_dir(report_md_path.parent)
    report_md_path.write_text(build_markdown_report(report), encoding="utf-8")


def run_adaptation(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    set_seed(args.seed)

    namespace = args._namespace_config
    artifact_paths = args._artifact_paths
    checkpoints_dir = namespace.checkpoints_dir
    outputs_dir = namespace.outputs_dir
    reports_dir = namespace.reports_dir
    ensure_dir(checkpoints_dir)
    ensure_dir(outputs_dir)
    ensure_dir(reports_dir)

    best_checkpoint_path = artifact_paths["best_checkpoint"]
    last_checkpoint_path = artifact_paths["last_checkpoint"]
    val_predictions_path = artifact_paths["val_predictions"]
    test_predictions_path = artifact_paths["test_predictions"]
    loss_curve_path = artifact_paths["loss_curve"]
    report_json_path = artifact_paths["report_json"]
    report_md_path = artifact_paths["report_md"]

    try:
        enforce_policy_b_manifest_guard(
            args.label_policy,
            val_manifest=Path(args.val_csv) if args.val_csv else None,
            test_manifest=Path(args.test_csv) if args.test_csv else None,
            support_manifest=Path(args.support_csv) if args.support_csv else None,
        )
    except ValueError as exc:
        raise StageFailure(str(exc)) from exc
    print_runtime_configuration(args)
    missing_files = collect_missing_paths(required_input_paths(args))
    if missing_files:
        report["failure_reasons"].extend(missing_files)
        raise StageFailure("Required files are missing.")

    source_only_report = load_source_only_report(Path(args.source_only_report))
    report["source_only_metrics"] = {
        "val": {
            "macro_auroc": source_only_report["val_metrics"].get("macro_auroc"),
            "macro_auprc": source_only_report["val_metrics"].get("macro_auprc"),
        },
        "test": {
            "macro_auroc": source_only_report["test_metrics"].get("macro_auroc"),
            "macro_auprc": source_only_report["test_metrics"].get("macro_auprc"),
        },
    }

    support_df, support_summary = validate_manifest(Path(args.support_csv), "support", args.label_policy)
    val_df, val_summary = validate_manifest(Path(args.val_csv), "val", args.label_policy)
    test_df, test_summary = validate_manifest(Path(args.test_csv), "test", args.label_policy)
    report["support_counts"] = support_summary
    report["val_counts"] = val_summary
    report["test_counts"] = test_summary

    print_split_summary(support_summary)
    print_split_summary(val_summary)
    print_split_summary(test_summary)

    leakage_checks, leakage_failures = check_leakage(
        {"support": support_df, "val": val_df, "test": test_df}
    )
    report["leakage_checks"] = leakage_checks
    if leakage_failures:
        report["failure_reasons"].extend(leakage_failures)
        raise StageFailure("Leakage detected between support, val, and test.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report["device"] = str(device)
    if args.debug:
        print(f"device: {device}")

    model, checkpoint_metadata = load_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        warnings_list=report["warnings"],
    )
    report["checkpoint_metadata"] = checkpoint_metadata

    parameter_summary = enable_full_finetune(model)
    report["parameter_summary"] = parameter_summary
    print(f"total parameters: {parameter_summary['total_parameters']}")
    print(f"trainable parameters: {parameter_summary['trainable_parameters']}")
    print(f"trainable parameter tensors: {parameter_summary['trainable_parameter_tensors']}")
    print(f"frozen parameter tensors: {parameter_summary['frozen_parameter_tensors']}")
    print(f"trainable parameter names sample: {parameter_summary['trainable_parameter_names_sample']}")
    verify_full_finetune_trainability(parameter_summary)

    support_loader = build_dataloader(
        support_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = build_dataloader(
        val_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_record: dict[str, Any] | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, support_loader, optimizer, device)
        val_metrics = evaluate_split(model, val_loader, device)
        history_item = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
        }
        report["training_history"].append(history_item)

        is_better, metric_name, metric_value = select_is_better(val_metrics, best_record)
        if is_better:
            best_record = {
                "epoch": int(epoch),
                "loss": float(val_metrics["loss"]),
                "macro_auroc": val_metrics["macro_auroc"],
                "macro_auprc": val_metrics["macro_auprc"],
                "metric_name": metric_name,
                "metric_value": float(metric_value),
            }
            torch.save(
                checkpoint_payload(
                    model=model,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_metrics=val_metrics,
                    args=args,
                    source_checkpoint=str(Path(args.checkpoint).resolve()),
                    parameter_summary=parameter_summary,
                    best_metric_name=metric_name,
                    best_metric_value=float(metric_value),
                ),
                best_checkpoint_path,
            )
            report["best_epoch"] = int(epoch)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_macro_auroc={format_metric(val_metrics['macro_auroc'])} | "
            f"val_macro_auprc={format_metric(val_metrics['macro_auprc'])}"
        )

        if epochs_without_improvement >= args.patience:
            stopped_early = True
            if args.debug:
                print(f"early stopping at epoch {epoch} after {epochs_without_improvement} non-improving epochs")
            break

    report["stopped_early"] = stopped_early
    if best_record is None or not best_checkpoint_path.exists():
        raise StageFailure("Training did not produce a best checkpoint.")

    final_epoch = report["training_history"][-1]["epoch"]
    final_train_loss = report["training_history"][-1]["train_loss"]
    final_val_metrics = evaluate_split(model, val_loader, device)
    torch.save(
        checkpoint_payload(
            model=model,
            epoch=int(final_epoch),
            train_loss=float(final_train_loss),
            val_metrics=final_val_metrics,
            args=args,
            source_checkpoint=str(Path(args.checkpoint).resolve()),
            parameter_summary=parameter_summary,
            best_metric_name=best_record["metric_name"],
            best_metric_value=float(best_record["metric_value"]),
        ),
        last_checkpoint_path,
    )
    report["best_checkpoint"] = str(best_checkpoint_path.resolve())
    report["last_checkpoint"] = str(last_checkpoint_path.resolve())

    train_losses = [item["train_loss"] for item in report["training_history"]]
    val_losses = [item["val_loss"] for item in report["training_history"]]
    report["plot_files"]["loss_curve"] = save_loss_curve(train_losses, val_losses, loss_curve_path)

    best_model, _ = load_checkpoint(best_checkpoint_path, device, report["warnings"])
    reloaded_summary = enable_full_finetune(best_model)
    verify_full_finetune_trainability(reloaded_summary)

    final_val_raw = evaluate_split(best_model, val_loader, device)
    test_loader = build_dataloader(
        test_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    final_test_raw = evaluate_split(best_model, test_loader, device)

    report["val_metrics"] = metrics_for_report(final_val_raw)
    report["test_metrics"] = metrics_for_report(final_test_raw)
    report["prediction_files"] = {
        "val": save_predictions_csv(
            dataframe=val_df,
            probabilities=final_val_raw["probabilities"],
            output_path=val_predictions_path,
            path_column=val_summary["path_column"],
        ),
        "test": save_predictions_csv(
            dataframe=test_df,
            probabilities=final_test_raw["probabilities"],
            output_path=test_predictions_path,
            path_column=test_summary["path_column"],
        ),
    }

    report["metric_deltas"] = {
        "val_macro_auroc_delta": metric_delta(
            report["val_metrics"]["macro_auroc"],
            report["source_only_metrics"]["val"]["macro_auroc"],
        ),
        "val_macro_auprc_delta": metric_delta(
            report["val_metrics"]["macro_auprc"],
            report["source_only_metrics"]["val"]["macro_auprc"],
        ),
        "test_macro_auroc_delta": metric_delta(
            report["test_metrics"]["macro_auroc"],
            report["source_only_metrics"]["test"]["macro_auroc"],
        ),
        "test_macro_auprc_delta": metric_delta(
            report["test_metrics"]["macro_auprc"],
            report["source_only_metrics"]["test"]["macro_auprc"],
        ),
    }

    write_reports(report, report_json_path, report_md_path)

    for output_path in [
        best_checkpoint_path,
        last_checkpoint_path,
        val_predictions_path,
        test_predictions_path,
        report_json_path,
        report_md_path,
        loss_curve_path,
    ]:
        if not output_path.exists():
            raise StageFailure(f"Expected output file was not created: {output_path}")

    report["status"] = "DONE"
    report["safe_to_continue"] = True
    write_reports(report, report_json_path, report_md_path)
    return report


def main() -> None:
    try:
        args = resolve_cli_args(parse_args())
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    if args.dry_run:
        try:
            run_dry_run(args)
        except StageFailure as exc:
            print(str(exc))
            sys.exit(1)
        print("DRY_RUN_OK")
        return

    report = initial_report(args)
    reports_dir = args._namespace_config.reports_dir
    ensure_dir(reports_dir)
    report_json_path = args._artifact_paths["report_json"]
    report_md_path = args._artifact_paths["report_md"]

    try:
        report = run_adaptation(args, report)
    except StageFailure as exc:
        if not report["failure_reasons"]:
            report["failure_reasons"].append(str(exc))
        if report["training_history"] or report["prediction_files"]:
            report["status"] = "PARTIAL"
        else:
            report["status"] = "FAILED"
        report["safe_to_continue"] = False
    except Exception as exc:  # pragma: no cover - depends on unexpected runtime failures
        report["failure_reasons"].append(f"Unexpected error: {exc}")
        report["warnings"].append("Unexpected traceback was captured in the report.")
        report["warnings"].append(traceback.format_exc())
        if report["training_history"] or report["prediction_files"]:
            report["status"] = "PARTIAL"
        else:
            report["status"] = "FAILED"
        report["safe_to_continue"] = False

    write_reports(report, report_json_path, report_md_path)
    print(report["status"])
    if report["status"] != "DONE":
        sys.exit(1)


if __name__ == "__main__":
    main()
