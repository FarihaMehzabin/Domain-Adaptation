#!/usr/bin/env python3
"""LoRA + BN-affine + classifier adaptation from NIH to MIMIC for Policy B."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adapt_head_only_mimic import (  # noqa: E402
    LABELS,
    StageFailure,
    build_dataloader,
    build_model,
    check_leakage,
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
    set_seed,
    validate_manifest,
)
from scripts.adapt_lora_mimic import (  # noqa: E402
    LoRAConv2d,
    get_module_by_name,
    set_module_by_name,
)
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
from scripts.masked_multilabel_utils import masked_bce_with_logits_loss  # noqa: E402


ADAPTATION_METHOD = "lora_bnhead"
GOAL = (
    "Adapt the NIH-trained DenseNet-121 model to MIMIC using LoRA on denseblock4, "
    "full classifier-head adaptation, and BN affine adaptation on denseblock4 + norm5."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA + BN-affine + classifier DenseNet-121 adaptation from NIH to MIMIC."
    )
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
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_lr", type=float, default=3e-4)
    parser.add_argument("--classifier_lr", type=float, default=1e-4)
    parser.add_argument("--bn_lr", type=float, default=1e-4)
    parser.add_argument("--lora_weight_decay", type=float, default=1e-4)
    parser.add_argument("--classifier_weight_decay", type=float, default=1e-4)
    parser.add_argument("--bn_weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--bn_mode",
        type=str,
        choices=["frozen_stats", "update_stats"],
        default="frozen_stats",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    namespace = build_namespace_config(args.base_dir, args.out_dir)
    if args.run_name is None:
        args.run_name = f"{ADAPTATION_METHOD}_seed{args.seed}"

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
    source_report_default = (
        default_policyb_source_report(args.seed)
        if namespace.base_dir is not None and args.label_policy == "uignore_blankzero"
        else "stage5_source_baseline.json"
    )
    source_only_report = resolve_report_input(
        args.source_only_report,
        namespace=namespace,
        default_filename=source_report_default,
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
    print(f"last checkpoint output path: {artifact_paths['last_checkpoint']}")
    print(f"loss curve output path: {artifact_paths['loss_curve']}")
    print(f"bn_mode: {args.bn_mode}")


def find_denseblock4_lora_target_module_names(model: nn.Module) -> list[str]:
    target_names: list[str] = []
    for name, module in model.named_modules():
        if (
            name.startswith("features.denseblock4.")
            and name.endswith(("conv1", "conv2"))
            and isinstance(module, nn.Conv2d)
        ):
            target_names.append(name)
    return target_names


def find_bn_affine_module_names(model: nn.Module) -> list[str]:
    module_names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        if name == "features.norm5" or name.startswith("features.denseblock4."):
            module_names.append(name)
    return module_names


def inject_denseblock4_lora_modules(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    if rank <= 0:
        raise StageFailure(f"LoRA rank must be positive, got {rank}")

    target_module_names = find_denseblock4_lora_target_module_names(model)
    if not target_module_names:
        raise StageFailure("No denseblock4 convolution targets were found for LoRA injection.")

    for module_name in target_module_names:
        module = get_module_by_name(model, module_name)
        if not isinstance(module, nn.Conv2d):
            raise StageFailure(
                f"Unsupported LoRA target module type for {module_name}: {module.__class__.__name__}"
            )
        replacement = LoRAConv2d(module, rank=rank, alpha=alpha, dropout=dropout)
        set_module_by_name(model, module_name, replacement)

    return target_module_names


def configure_lora_bnhead_trainability(
    model: nn.Module,
    *,
    bn_module_names: list[str],
) -> dict[str, list[str]]:
    lora_parameter_names: list[str] = []
    classifier_parameter_names: list[str] = []
    bn_affine_parameter_names: list[str] = []

    bn_affine_name_set = {
        f"{module_name}.{suffix}"
        for module_name in bn_module_names
        for suffix in ("weight", "bias")
    }

    for _, parameter in model.named_parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        if ".lora_down.weight" in name or ".lora_up.weight" in name:
            parameter.requires_grad = True
            lora_parameter_names.append(name)
        elif name.startswith("classifier."):
            parameter.requires_grad = True
            classifier_parameter_names.append(name)
        elif name in bn_affine_name_set:
            parameter.requires_grad = True
            bn_affine_parameter_names.append(name)

    return {
        "lora": lora_parameter_names,
        "classifier": classifier_parameter_names,
        "bn_affine": bn_affine_parameter_names,
    }


def summarize_parameter_trainability(
    model: nn.Module,
    *,
    lora_target_module_names: list[str],
    bn_module_names: list[str],
    trainable_groups: dict[str, list[str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    total_parameters = 0
    trainable_parameters = 0
    parameter_names: list[str] = []
    trainable_parameter_names: list[str] = []
    frozen_parameter_names: list[str] = []

    trainable_name_to_group: dict[str, str] = {}
    for group_name, names in trainable_groups.items():
        for name in names:
            trainable_name_to_group[name] = group_name

    expected_trainable_names = set(trainable_name_to_group.keys())
    unexpected_trainable_names: list[str] = []

    for name, parameter in model.named_parameters():
        parameter_names.append(name)
        total_parameters += parameter.numel()
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()
            trainable_parameter_names.append(name)
            if name not in expected_trainable_names:
                unexpected_trainable_names.append(name)
        else:
            frozen_parameter_names.append(name)

    missing_trainable_names = sorted(expected_trainable_names - set(trainable_parameter_names))

    original_conv_parameter_names: list[str] = []
    original_conv_trainable_names: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module_name.endswith("lora_down") or module_name.endswith("lora_up"):
            continue
        for param_name, parameter in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}"
            original_conv_parameter_names.append(full_name)
            if parameter.requires_grad:
                original_conv_trainable_names.append(full_name)

    trainable_group_summary: dict[str, Any] = {}
    named_parameters = dict(model.named_parameters())
    for group_name, group_parameter_names in trainable_groups.items():
        parameter_count = sum(named_parameters[name].numel() for name in group_parameter_names)
        trainable_group_summary[group_name] = {
            "parameter_count": int(parameter_count),
            "tensor_count": int(len(group_parameter_names)),
            "parameter_names": list(group_parameter_names),
            "parameter_names_sample": list(group_parameter_names[:20]),
        }

    trainable_percentage = (
        100.0 * float(trainable_parameters) / float(total_parameters)
        if total_parameters > 0
        else 0.0
    )

    return {
        "adaptation_method": ADAPTATION_METHOD,
        "bn_mode": args.bn_mode,
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "classifier_base_changed_allowed": True,
        "bn_affine_changed_allowed": True,
        "lora_target_module_names": lora_target_module_names,
        "lora_target_module_count": int(len(lora_target_module_names)),
        "bn_module_names": bn_module_names,
        "bn_module_count": int(len(bn_module_names)),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "trainable_parameter_percentage": float(trainable_percentage),
        "total_parameter_tensors": int(len(parameter_names)),
        "trainable_parameter_tensors": int(len(trainable_parameter_names)),
        "frozen_parameter_tensors": int(len(frozen_parameter_names)),
        "trainable_parameter_names": trainable_parameter_names,
        "frozen_parameter_names": frozen_parameter_names,
        "trainable_parameter_names_sample": trainable_parameter_names[:20],
        "frozen_parameter_names_sample": frozen_parameter_names[:20],
        "trainable_groups": trainable_group_summary,
        "unexpected_trainable_names": unexpected_trainable_names,
        "missing_trainable_names": missing_trainable_names,
        "original_conv_parameter_count": int(len(original_conv_parameter_names)),
        "original_conv_parameter_names_sample": original_conv_parameter_names[:20],
        "original_non_lora_conv_weights_frozen": len(original_conv_trainable_names) == 0,
        "original_non_lora_conv_trainable_names": original_conv_trainable_names,
    }


def verify_trainability_plan(summary: dict[str, Any]) -> None:
    if summary["lora_target_module_count"] == 0:
        raise StageFailure("LoRA did not target any denseblock4 convolution modules.")
    if summary["bn_module_count"] == 0:
        raise StageFailure("No BN modules were selected for affine adaptation.")
    if summary["trainable_parameter_tensors"] == 0:
        raise StageFailure("No parameters are trainable under the LoRA + BN-head plan.")
    if not summary["trainable_groups"]["lora"]["parameter_names"]:
        raise StageFailure("LoRA parameters are unexpectedly frozen.")
    if summary["trainable_groups"]["classifier"]["parameter_names"] != [
        "classifier.weight",
        "classifier.bias",
    ]:
        raise StageFailure(
            "Classifier head trainability is incorrect. Expected classifier.weight and classifier.bias only."
        )
    if not summary["trainable_groups"]["bn_affine"]["parameter_names"]:
        raise StageFailure("BN affine parameters are unexpectedly frozen.")
    if summary["unexpected_trainable_names"]:
        raise StageFailure(
            "Unexpected trainable parameters were found: "
            f"{summary['unexpected_trainable_names'][:20]}"
        )
    if summary["missing_trainable_names"]:
        raise StageFailure(
            "Expected trainable parameters are missing: "
            f"{summary['missing_trainable_names'][:20]}"
        )
    if not summary["original_non_lora_conv_weights_frozen"]:
        raise StageFailure(
            "Original non-LoRA convolution weights are trainable: "
            f"{summary['original_non_lora_conv_trainable_names'][:20]}"
        )


def print_trainability_summary(summary: dict[str, Any]) -> None:
    print(f"trainable plan adaptation method: {summary['adaptation_method']}")
    print(f"selected bn_mode: {summary['bn_mode']}")
    print(f"LoRA target module count: {summary['lora_target_module_count']}")
    print(f"BN affine module count: {summary['bn_module_count']}")
    print(f"total parameters: {summary['total_parameters']}")
    print(f"trainable parameters: {summary['trainable_parameters']}")
    print(f"trainable percentage: {summary['trainable_parameter_percentage']:.4f}%")
    print(
        "trainable groups: "
        f"LoRA={summary['trainable_groups']['lora']['parameter_count']}, "
        f"classifier={summary['trainable_groups']['classifier']['parameter_count']}, "
        f"bn_affine={summary['trainable_groups']['bn_affine']['parameter_count']}"
    )
    print(f"trainable parameter names sample: {summary['trainable_parameter_names_sample']}")
    print(
        "original non-LoRA conv weights frozen: "
        f"{'yes' if summary['original_non_lora_conv_weights_frozen'] else 'no'}"
    )


def set_lora_bnhead_train_mode(
    model: nn.Module,
    *,
    bn_module_names: list[str],
    bn_mode: str,
) -> None:
    model.eval()
    model.classifier.train()

    for module in model.modules():
        if isinstance(module, LoRAConv2d):
            module.train()

    for module_name in bn_module_names:
        module = get_module_by_name(model, module_name)
        if bn_mode == "update_stats":
            module.train()
        else:
            module.eval()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    bn_module_names: list[str],
    bn_mode: str,
) -> float:
    set_lora_bnhead_train_mode(model, bn_module_names=bn_module_names, bn_mode=bn_mode)
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
    plt.title("LoRA + BN-head adaptation loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path.resolve())


def select_is_better(
    current_metrics: dict[str, Any],
    best_record: dict[str, Any] | None,
) -> tuple[bool, str, float]:
    current_auprc = current_metrics["macro_auprc"]
    current_auroc = current_metrics["macro_auroc"]
    current_loss = float(current_metrics["loss"])

    if current_auprc is not None:
        if best_record is None or best_record["macro_auprc"] is None:
            return True, "val_macro_auprc", float(current_auprc)
        if current_auprc > best_record["macro_auprc"] + 1e-12:
            return True, "val_macro_auprc", float(current_auprc)
        if math.isclose(current_auprc, best_record["macro_auprc"], abs_tol=1e-12):
            if current_loss < best_record["loss"] - 1e-12:
                return True, "val_macro_auprc", float(current_auprc)
        return False, "val_macro_auprc", float(current_auprc)

    if current_auroc is not None:
        if best_record is None or best_record["macro_auroc"] is None:
            return True, "val_macro_auroc", float(current_auroc)
        if current_auroc > best_record["macro_auroc"] + 1e-12:
            return True, "val_macro_auroc", float(current_auroc)
        if math.isclose(current_auroc, best_record["macro_auroc"], abs_tol=1e-12):
            if current_loss < best_record["loss"] - 1e-12:
                return True, "val_macro_auroc", float(current_auroc)
        return False, "val_macro_auroc", float(current_auroc)

    if best_record is None or best_record["metric_name"] != "val_loss":
        return True, "val_loss", float(current_loss)
    if current_loss < best_record["loss"] - 1e-12:
        return True, "val_loss", float(current_loss)
    return False, "val_loss", float(current_loss)


def checkpoint_payload(
    model: nn.Module,
    *,
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
        "adaptation_method": ADAPTATION_METHOD,
        "bn_mode": args.bn_mode,
        "source_checkpoint": source_checkpoint,
        "best_metric_name": best_metric_name,
        "best_metric_value": float(best_metric_value),
        "parameter_summary": parameter_summary,
        "model_state_dict": model.state_dict(),
    }


def clone_named_parameters(model: nn.Module, parameter_names: list[str]) -> dict[str, torch.Tensor]:
    named_parameters = dict(model.named_parameters())
    return {name: named_parameters[name].detach().cpu().clone() for name in parameter_names}


def compare_parameter_group_changes(
    before: dict[str, torch.Tensor],
    after_model: nn.Module,
) -> dict[str, Any]:
    after_parameters = dict(after_model.named_parameters())
    changed_names: list[str] = []
    unchanged_names: list[str] = []
    max_abs_diff = 0.0

    for name, before_tensor in before.items():
        after_tensor = after_parameters[name].detach().cpu()
        diff = float(torch.max(torch.abs(after_tensor - before_tensor)).item())
        max_abs_diff = max(max_abs_diff, diff)
        if diff > 0.0:
            changed_names.append(name)
        else:
            unchanged_names.append(name)

    return {
        "changed": bool(changed_names),
        "changed_names": changed_names,
        "unchanged_names": unchanged_names,
        "changed_name_sample": changed_names[:20],
        "unchanged_name_sample": unchanged_names[:20],
        "max_abs_diff": float(max_abs_diff),
    }


def compute_per_label_metric_deltas(
    current_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, dict[str, float | None]]:
    deltas: dict[str, dict[str, float | None]] = {}
    current_per_label = current_metrics.get("per_label", {})
    baseline_per_label = baseline_metrics.get("per_label", {})
    for label in LABELS:
        current_label = current_per_label.get(label, {})
        baseline_label = baseline_per_label.get(label, {})
        deltas[label] = {
            "auroc_delta": metric_delta(current_label.get("auroc"), baseline_label.get("auroc")),
            "auprc_delta": metric_delta(current_label.get("auprc"), baseline_label.get("auprc")),
        }
    return deltas


def build_lora_bnhead_model_from_saved_state(
    state_dict: dict[str, Any],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    model = build_model(num_labels=len(LABELS))
    inject_denseblock4_lora_modules(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    return model


def initial_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "FAILED",
        "safe_to_continue": False,
        "goal": GOAL,
        "run_name": args.run_name,
        "adaptation_method": ADAPTATION_METHOD,
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
            "image_size": int(args.image_size),
            "patience": int(args.patience),
            "seed": int(args.seed),
            "lora_rank": int(args.lora_rank),
            "lora_alpha": float(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "lora_lr": float(args.lora_lr),
            "classifier_lr": float(args.classifier_lr),
            "bn_lr": float(args.bn_lr),
            "lora_weight_decay": float(args.lora_weight_decay),
            "classifier_weight_decay": float(args.classifier_weight_decay),
            "bn_weight_decay": float(args.bn_weight_decay),
            "bn_mode": args.bn_mode,
            "primary_model_selection_metric": "val_macro_auprc",
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
        "per_label_metric_deltas": {},
        "parameter_change_checks": {},
        "device": None,
        "warnings": [],
        "failure_reasons": [],
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Policy B LoRA + BN-head NIH to MIMIC Adaptation",
        "",
        "## Goal",
        report["goal"],
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
        f"- image size: {report['training_config']['image_size']}",
        f"- patience: {report['training_config']['patience']}",
        f"- seed: {report['training_config']['seed']}",
        f"- LoRA rank: {report['training_config']['lora_rank']}",
        f"- LoRA alpha: {report['training_config']['lora_alpha']}",
        f"- LoRA dropout: {report['training_config']['lora_dropout']}",
        f"- LoRA lr / wd: {report['training_config']['lora_lr']} / {report['training_config']['lora_weight_decay']}",
        (
            "- classifier lr / wd: "
            f"{report['training_config']['classifier_lr']} / {report['training_config']['classifier_weight_decay']}"
        ),
        f"- BN affine lr / wd: {report['training_config']['bn_lr']} / {report['training_config']['bn_weight_decay']}",
        f"- BN mode: `{report['training_config']['bn_mode']}`",
        f"- primary model selection metric: `{report['training_config']['primary_model_selection_metric']}`",
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

    parameter_summary = report.get("parameter_summary", {})
    trainable_groups = parameter_summary.get("trainable_groups", {})
    lines.extend(
        [
            "## Trainability",
            f"- LoRA target module count: {parameter_summary.get('lora_target_module_count')}",
            f"- BN affine module count: {parameter_summary.get('bn_module_count')}",
            f"- total parameters: {parameter_summary.get('total_parameters')}",
            f"- trainable parameters: {parameter_summary.get('trainable_parameters')}",
            f"- trainable percentage: {parameter_summary.get('trainable_parameter_percentage')}",
            (
                "- trainable groups: "
                f"LoRA={trainable_groups.get('lora', {}).get('parameter_count')}, "
                f"classifier={trainable_groups.get('classifier', {}).get('parameter_count')}, "
                f"bn_affine={trainable_groups.get('bn_affine', {}).get('parameter_count')}"
            ),
            f"- trainable parameter names sample: {parameter_summary.get('trainable_parameter_names_sample')}",
            (
                "- original non-LoRA conv weights frozen: "
                f"{'yes' if parameter_summary.get('original_non_lora_conv_weights_frozen') else 'no'}"
            ),
            f"- classifier base changed allowed: {parameter_summary.get('classifier_base_changed_allowed')}",
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

    parameter_change_checks = report.get("parameter_change_checks", {})
    if parameter_change_checks:
        lines.extend(
            [
                "",
                "## Parameter Change Checks",
                (
                    "- classifier changed: "
                    f"{'yes' if parameter_change_checks.get('classifier', {}).get('changed') else 'no'}"
                ),
                (
                    "- BN affine changed: "
                    f"{'yes' if parameter_change_checks.get('bn_affine', {}).get('changed') else 'no'}"
                ),
                (
                    "- classifier max abs diff: "
                    f"{parameter_change_checks.get('classifier', {}).get('max_abs_diff')}"
                ),
                (
                    "- BN affine max abs diff: "
                    f"{parameter_change_checks.get('bn_affine', {}).get('max_abs_diff')}"
                ),
            ]
        )

    for section_name in ["val_metrics", "test_metrics"]:
        metrics = report.get(section_name, {})
        per_label_deltas = report.get("per_label_metric_deltas", {}).get(section_name.replace("_metrics", ""), {})
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
            delta_item = per_label_deltas.get(label, {})
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
                f"delta_AUROC={format_metric(delta_item.get('auroc_delta'))}, "
                f"delta_AUPRC={format_metric(delta_item.get('auprc_delta'))}, "
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
                f"- source-only val macro AUROC: {format_metric(report['source_only_metrics']['val'].get('macro_auroc'))}",
                f"- source-only val macro AUPRC: {format_metric(report['source_only_metrics']['val'].get('macro_auprc'))}",
                f"- source-only test macro AUROC: {format_metric(report['source_only_metrics']['test'].get('macro_auroc'))}",
                f"- source-only test macro AUPRC: {format_metric(report['source_only_metrics']['test'].get('macro_auprc'))}",
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


def print_final_summary(report: dict[str, Any], *, dry_run_status: str) -> None:
    parameter_summary = report.get("parameter_summary", {})
    parameter_change_checks = report.get("parameter_change_checks", {})
    warnings_list = report.get("warnings", [])

    print(f"dry_run_status: {dry_run_status}")
    print(
        "final_val: "
        f"macro_auroc={format_metric(report.get('val_metrics', {}).get('macro_auroc'))}, "
        f"macro_auprc={format_metric(report.get('val_metrics', {}).get('macro_auprc'))}"
    )
    print(
        "final_test: "
        f"macro_auroc={format_metric(report.get('test_metrics', {}).get('macro_auroc'))}, "
        f"macro_auprc={format_metric(report.get('test_metrics', {}).get('macro_auprc'))}"
    )
    print(
        "delta_vs_source_only: "
        f"val_macro_auroc={format_metric(report.get('metric_deltas', {}).get('val_macro_auroc_delta'))}, "
        f"val_macro_auprc={format_metric(report.get('metric_deltas', {}).get('val_macro_auprc_delta'))}, "
        f"test_macro_auroc={format_metric(report.get('metric_deltas', {}).get('test_macro_auroc_delta'))}, "
        f"test_macro_auprc={format_metric(report.get('metric_deltas', {}).get('test_macro_auprc_delta'))}"
    )
    print(
        "trainable_parameters: "
        f"{parameter_summary.get('trainable_parameters')} / {parameter_summary.get('total_parameters')} "
        f"({parameter_summary.get('trainable_parameter_percentage')})"
    )
    print(f"best_epoch: {report.get('best_epoch')}")
    print(f"early_stopping: {'yes' if report.get('stopped_early') else 'no'}")
    print(
        "classifier_changed: "
        f"{'yes' if parameter_change_checks.get('classifier', {}).get('changed') else 'no'}"
    )
    print(
        "bn_affine_changed: "
        f"{'yes' if parameter_change_checks.get('bn_affine', {}).get('changed') else 'no'}"
    )
    if warnings_list:
        print(f"warnings: {warnings_list}")
    else:
        print("warnings: none")
    print(f"safe_to_continue: {'yes' if report.get('safe_to_continue') else 'no'}")


def build_optimizer(
    model: nn.Module,
    *,
    trainable_groups: dict[str, list[str]],
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    named_parameters = dict(model.named_parameters())
    optimizer_groups = [
        {
            "params": [named_parameters[name] for name in trainable_groups["lora"]],
            "lr": args.lora_lr,
            "weight_decay": args.lora_weight_decay,
            "group_name": "lora",
        },
        {
            "params": [named_parameters[name] for name in trainable_groups["classifier"]],
            "lr": args.classifier_lr,
            "weight_decay": args.classifier_weight_decay,
            "group_name": "classifier",
        },
        {
            "params": [named_parameters[name] for name in trainable_groups["bn_affine"]],
            "lr": args.bn_lr,
            "weight_decay": args.bn_weight_decay,
            "group_name": "bn_affine",
        },
    ]
    return torch.optim.AdamW(optimizer_groups)


def prepare_model_for_adaptation(
    *,
    checkpoint_path: Path,
    device: torch.device,
    args: argparse.Namespace,
    warnings_list: list[str],
) -> tuple[nn.Module, dict[str, Any], dict[str, Any], dict[str, list[str]]]:
    model, checkpoint_metadata = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        warnings_list=warnings_list,
    )
    lora_target_module_names = inject_denseblock4_lora_modules(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model = model.to(device)
    bn_module_names = find_bn_affine_module_names(model)
    trainable_groups = configure_lora_bnhead_trainability(
        model,
        bn_module_names=bn_module_names,
    )
    parameter_summary = summarize_parameter_trainability(
        model,
        lora_target_module_names=lora_target_module_names,
        bn_module_names=bn_module_names,
        trainable_groups=trainable_groups,
        args=args,
    )
    verify_trainability_plan(parameter_summary)
    return model, checkpoint_metadata, parameter_summary, trainable_groups


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

    _ = load_source_only_report(Path(args.source_only_report))

    dry_run_warnings: list[str] = []
    model, _, parameter_summary, _ = prepare_model_for_adaptation(
        checkpoint_path=Path(args.checkpoint),
        device=torch.device("cpu"),
        args=args,
        warnings_list=dry_run_warnings,
    )
    _ = model
    print_trainability_summary(parameter_summary)
    if dry_run_warnings:
        print(f"dry_run_warnings: {dry_run_warnings}")
    print("dry_run: configuration resolved and trainable parameter plan verified")


def run_adaptation(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    set_seed(args.seed)

    namespace = args._namespace_config
    artifact_paths = args._artifact_paths
    ensure_dir(namespace.checkpoints_dir)
    ensure_dir(namespace.outputs_dir)
    ensure_dir(namespace.reports_dir)

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
        "val": source_only_report["val_metrics"],
        "test": source_only_report["test_metrics"],
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

    model, checkpoint_metadata, parameter_summary, trainable_groups = prepare_model_for_adaptation(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        args=args,
        warnings_list=report["warnings"],
    )
    report["checkpoint_metadata"] = checkpoint_metadata
    report["parameter_summary"] = parameter_summary
    print_trainability_summary(parameter_summary)

    initial_classifier_state = clone_named_parameters(
        model,
        trainable_groups["classifier"],
    )
    initial_bn_affine_state = clone_named_parameters(
        model,
        trainable_groups["bn_affine"],
    )

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
    optimizer = build_optimizer(model, trainable_groups=trainable_groups, args=args)

    best_record: dict[str, Any] | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            support_loader,
            optimizer,
            device,
            bn_module_names=parameter_summary["bn_module_names"],
            bn_mode=args.bn_mode,
        )
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
                print(
                    f"early stopping at epoch {epoch} after "
                    f"{epochs_without_improvement} non-improving epochs"
                )
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

    saved_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(saved_checkpoint, dict) or "model_state_dict" not in saved_checkpoint:
        raise StageFailure(f"Best checkpoint is missing model_state_dict: {best_checkpoint_path}")

    best_model = build_lora_bnhead_model_from_saved_state(
        saved_checkpoint["model_state_dict"],
        args=args,
        device=device,
    )

    report["parameter_change_checks"] = {
        "classifier": compare_parameter_group_changes(initial_classifier_state, best_model),
        "bn_affine": compare_parameter_group_changes(initial_bn_affine_state, best_model),
    }

    if not report["parameter_change_checks"]["classifier"]["changed"]:
        report["warnings"].append("Classifier parameters did not change from initialization.")
    if not report["parameter_change_checks"]["bn_affine"]["changed"]:
        report["warnings"].append("BN affine parameters did not change from initialization.")

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
            report["source_only_metrics"]["val"].get("macro_auroc"),
        ),
        "val_macro_auprc_delta": metric_delta(
            report["val_metrics"]["macro_auprc"],
            report["source_only_metrics"]["val"].get("macro_auprc"),
        ),
        "test_macro_auroc_delta": metric_delta(
            report["test_metrics"]["macro_auroc"],
            report["source_only_metrics"]["test"].get("macro_auroc"),
        ),
        "test_macro_auprc_delta": metric_delta(
            report["test_metrics"]["macro_auprc"],
            report["source_only_metrics"]["test"].get("macro_auprc"),
        ),
    }
    report["per_label_metric_deltas"] = {
        "val": compute_per_label_metric_deltas(
            report["val_metrics"],
            report["source_only_metrics"]["val"],
        ),
        "test": compute_per_label_metric_deltas(
            report["test_metrics"],
            report["source_only_metrics"]["test"],
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
            print("dry_run_status: failed")
            sys.exit(1)
        print("dry_run_status: passed")
        print("DRY_RUN_OK")
        return

    report = initial_report(args)
    ensure_dir(args._namespace_config.reports_dir)
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
    print_final_summary(report, dry_run_status="not_run_in_this_invocation")
    print(report["status"])
    if report["status"] != "DONE":
        sys.exit(1)


if __name__ == "__main__":
    main()
