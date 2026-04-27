#!/usr/bin/env python3
"""Train a hospital-specific residual feature adapter on top of a frozen DenseNet-121."""

from __future__ import annotations

import argparse
import copy
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hospital_adapters_v1.common import (  # noqa: E402
    POLICY_B_LABEL_POLICY,
    StageFailure,
    build_dataloader,
    build_split_report,
    check_leakage,
    compute_pos_weight_from_dataframe,
    ensure_dir,
    evaluate_model,
    format_metric,
    infer_label_names,
    infer_split_name,
    json_ready,
    load_base_checkpoint,
    masked_bce_with_logits_loss,
    report_ready_metrics,
    resolve_project_or_absolute_path,
    resolve_run_dir,
    save_json,
    save_predictions_csv,
    set_seed,
    validate_manifest,
    write_train_log,
)
from experiments.hospital_adapters_v1.models.hospital_adapter import (  # noqa: E402
    HospitalAdapterClassifier,
    ResidualFeatureAdapter,
    apply_adapter_checkpoint,
    build_adapter_checkpoint_payload,
    configure_trainable_parameters,
    save_adapter_checkpoint,
    set_adapter_train_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a hospital-specific residual feature adapter on a frozen DenseNet-121."
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--source-hospital", type=str, default=None)
    parser.add_argument("--target-hospital", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/nih_2k_densenet121_best.pt")
    parser.add_argument("--support-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, default=None)
    parser.add_argument("--source-val-csv", type=str, default=None)
    parser.add_argument("--source-test-csv", type=str, default=None)
    parser.add_argument("--label-policy", type=str, default=POLICY_B_LABEL_POLICY)
    parser.add_argument("--out-dir", type=str, default="experiments/hospital_adapters_v1/runs")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--adapter-bottleneck", type=int, default=128)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--adapter-scale-init", type=float, default=1e-3)
    parser.add_argument("--train-classifier-head", action="store_true")
    parser.add_argument("--disable-hospital-bias", action="store_true")
    parser.add_argument("--use-pos-weight", dest="use_pos_weight", action="store_true")
    parser.add_argument("--no-pos-weight", dest="use_pos_weight", action="store_false")
    parser.set_defaults(use_pos_weight=True)
    parser.add_argument("--pos-weight-max", type=float, default=10.0)
    parser.add_argument("--debug-overfit-one-batch", action="store_true")
    parser.add_argument("--debug-overfit-steps", type=int, default=25)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.run_name is None:
        source_name = args.source_hospital or "source"
        args.run_name = f"{source_name}_to_{args.target_hospital}_adapter_seed{args.seed}"

    args.checkpoint = resolve_project_or_absolute_path(args.checkpoint)
    args.support_csv = resolve_project_or_absolute_path(args.support_csv)
    args.val_csv = resolve_project_or_absolute_path(args.val_csv)
    args.test_csv = resolve_project_or_absolute_path(args.test_csv)
    args.source_val_csv = resolve_project_or_absolute_path(args.source_val_csv)
    args.source_test_csv = resolve_project_or_absolute_path(args.source_test_csv)
    args.run_dir = resolve_run_dir(args.out_dir, args.run_name)
    args.paths = {
        "run_dir": args.run_dir,
        "adapter_best": (args.run_dir / "adapter_best.pt").resolve(),
        "adapter_last": (args.run_dir / "adapter_last.pt").resolve(),
        "config": (args.run_dir / "config.json").resolve(),
        "train_log": (args.run_dir / "train_log.csv").resolve(),
        "target_val_report": (args.run_dir / "target_val_report.json").resolve(),
        "target_test_report": (args.run_dir / "target_test_report.json").resolve(),
        "source_eval_report": (args.run_dir / "source_eval_report.json").resolve(),
        "predictions_target_val": (args.run_dir / "predictions_target_val.csv").resolve(),
        "predictions_target_test": (args.run_dir / "predictions_target_test.csv").resolve(),
    }
    return args


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_adapter_model(
    *,
    checkpoint_path: Path,
    device: torch.device,
    adapter_bottleneck: int,
    adapter_dropout: float,
    adapter_scale_init: float,
    disable_hospital_bias: bool,
    train_classifier_head: bool,
) -> tuple[HospitalAdapterClassifier, dict[str, Any], list[str]]:
    base_model, checkpoint_metadata, label_names = load_base_checkpoint(checkpoint_path, device)
    classifier = getattr(base_model, "classifier")
    pooled_feature_dim = int(classifier.in_features)
    adapter = ResidualFeatureAdapter(
        input_dim=pooled_feature_dim,
        bottleneck_dim=adapter_bottleneck,
        dropout=adapter_dropout,
        scale_init=adapter_scale_init,
    )
    hospital_model = HospitalAdapterClassifier(base_model=base_model, adapter=adapter)
    hospital_model = hospital_model.to(device)

    parameter_summary = configure_trainable_parameters(
        hospital_model,
        train_classifier_head=train_classifier_head,
        train_hospital_bias=not disable_hospital_bias,
    )
    checkpoint_metadata = dict(checkpoint_metadata)
    checkpoint_metadata["pooled_feature_dim"] = pooled_feature_dim
    checkpoint_metadata["label_names"] = label_names
    return hospital_model, checkpoint_metadata, label_names


def verify_parameter_summary(parameter_summary: dict[str, Any], train_classifier_head: bool) -> None:
    if parameter_summary["unexpectedly_trainable_base_names"]:
        raise StageFailure(
            "Frozen backbone verification failed. Unexpected trainable base parameters: "
            f"{parameter_summary['unexpectedly_trainable_base_names']}"
        )

    trainable_names = list(parameter_summary["trainable_parameter_names"])
    if not trainable_names:
        raise StageFailure("No trainable parameters remain after adapter setup.")

    allowed_prefixes = ["adapter.", "hospital_bias"]
    if train_classifier_head:
        allowed_prefixes.append("base_model.classifier.")
    for name in trainable_names:
        if not any(name == prefix or name.startswith(prefix) for prefix in allowed_prefixes):
            raise StageFailure(f"Unexpected trainable parameter for hospital adapter baseline: {name}")

    if not train_classifier_head:
        classifier_names = [name for name in trainable_names if name.startswith("base_model.classifier.")]
        if classifier_names:
            raise StageFailure(f"Classifier head is trainable despite --train-classifier-head being disabled: {classifier_names}")


def run_identity_check(
    base_model: torch.nn.Module,
    hospital_model: HospitalAdapterClassifier,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> float:
    images, _, _ = batch
    images = images.to(device, non_blocking=True)
    base_model.eval()
    hospital_model.eval()
    with torch.no_grad():
        base_logits = base_model(images)
        adapted_logits = hospital_model(images)
    return float((base_logits - adapted_logits).abs().max().item())


def build_optimizer(
    model: HospitalAdapterClassifier,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise StageFailure("No trainable parameters found for optimizer construction.")
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def train_one_epoch(
    model: HospitalAdapterClassifier,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    train_classifier_head: bool,
    pos_weight: torch.Tensor | None,
) -> float:
    set_adapter_train_mode(model, train_classifier_head=train_classifier_head)
    total_loss = 0.0
    total_valid = 0.0

    for images, labels, masks in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = masked_bce_with_logits_loss(logits, labels, masks, pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise StageFailure("Training loss became NaN or infinite.")
        loss.backward()
        optimizer.step()

        valid_count = float(masks.sum().item())
        total_loss += float(loss.item()) * valid_count
        total_valid += valid_count

    return float(total_loss / total_valid) if total_valid > 0 else 0.0


def run_debug_overfit_one_batch(
    model: HospitalAdapterClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    train_classifier_head: bool,
    lr: float,
    weight_decay: float,
    steps: int,
    pos_weight: torch.Tensor | None,
) -> dict[str, Any]:
    batch = next(iter(dataloader))
    images, labels, masks = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)

    adapter_state = copy.deepcopy(model.adapter.state_dict())
    bias_state = model.hospital_bias.detach().cpu().clone()
    classifier_state = copy.deepcopy(model.get_classifier().state_dict())

    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    for step in range(steps):
        set_adapter_train_mode(model, train_classifier_head=train_classifier_head)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = masked_bce_with_logits_loss(logits, labels, masks, pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise StageFailure("Overfit-one-batch debug loss became NaN or infinite.")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        print(f"debug_overfit step {step + 1:02d}/{steps}: loss={losses[-1]:.6f}")

    model.adapter.load_state_dict(adapter_state, strict=True)
    with torch.no_grad():
        model.hospital_bias.copy_(bias_state.to(device=model.hospital_bias.device, dtype=model.hospital_bias.dtype))
    model.get_classifier().load_state_dict(classifier_state, strict=True)

    return {
        "steps": int(steps),
        "losses": losses,
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
    }


def select_is_better(
    current_metrics: dict[str, Any],
    best_record: dict[str, Any] | None,
) -> tuple[bool, str, float]:
    current_auroc = current_metrics["macro_auroc"]
    current_loss = float(current_metrics["loss"])

    if current_auroc is not None:
        if best_record is None or best_record["macro_auroc"] is None:
            return True, "target_val_macro_auroc", float(current_auroc)
        if current_auroc > best_record["macro_auroc"] + 1e-12:
            return True, "target_val_macro_auroc", float(current_auroc)
        if abs(current_auroc - best_record["macro_auroc"]) <= 1e-12 and current_loss < best_record["loss"] - 1e-12:
            return True, "target_val_macro_auroc", float(current_auroc)
        return False, "target_val_macro_auroc", float(current_auroc)

    if best_record is None or best_record["macro_auroc"] is not None:
        return True, "target_val_bce_loss", float(current_loss)
    if current_loss < best_record["loss"] - 1e-12:
        return True, "target_val_bce_loss", float(current_loss)
    return False, "target_val_bce_loss", float(current_loss)


def save_checkpoint(
    *,
    path: Path,
    model: HospitalAdapterClassifier,
    epoch: int,
    args: argparse.Namespace,
    label_names: list[str],
    pooled_feature_dim: int,
    best_metric_name: str,
    best_metric_value: float | None,
    parameter_summary: dict[str, Any],
    identity_max_abs_logit_diff: float,
    debug_overfit: dict[str, Any] | None,
) -> None:
    payload = build_adapter_checkpoint_payload(
        model,
        epoch=epoch,
        target_hospital=args.target_hospital,
        source_hospital=args.source_hospital,
        label_names=label_names,
        pooled_feature_dim=pooled_feature_dim,
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_dropout=args.adapter_dropout,
        adapter_scale_init=args.adapter_scale_init,
        base_checkpoint_path=str(args.checkpoint.resolve()),
        classifier_head_trained=args.train_classifier_head,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
        extra={
            "label_policy": args.label_policy,
            "hospital_bias_trained": not args.disable_hospital_bias,
            "parameter_summary": parameter_summary,
            "identity_max_abs_logit_diff": float(identity_max_abs_logit_diff),
            "debug_overfit_one_batch": debug_overfit,
        },
    )
    save_adapter_checkpoint(payload, path)


def load_best_adapter_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[HospitalAdapterClassifier, dict[str, Any], list[str], dict[str, Any]]:
    model, checkpoint_metadata, label_names = build_adapter_model(
        checkpoint_path=args.checkpoint,
        device=device,
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_dropout=args.adapter_dropout,
        adapter_scale_init=args.adapter_scale_init,
        disable_hospital_bias=args.disable_hospital_bias,
        train_classifier_head=args.train_classifier_head,
    )
    adapter_checkpoint = torch.load(args.paths["adapter_best"], map_location=device, weights_only=False)
    apply_adapter_checkpoint(model, adapter_checkpoint)
    return model, checkpoint_metadata, label_names, adapter_checkpoint


def main() -> int:
    args = resolve_args(parse_args())
    ensure_dir(args.run_dir)

    device = resolve_device(args.device)
    set_seed(args.seed)

    try:
        base_model_for_labels, checkpoint_metadata, _ = load_base_checkpoint(args.checkpoint, device)
        label_names = infer_label_names(checkpoint_metadata, manifest_paths=[args.support_csv, args.val_csv])
        del base_model_for_labels

        image_size = args.image_size or int(checkpoint_metadata.get("image_size") or 224)
        support_df, support_summary = validate_manifest(
            args.support_csv,
            infer_split_name(args.support_csv, "support"),
            label_names,
            args.label_policy,
        )
        val_df, val_summary = validate_manifest(
            args.val_csv,
            infer_split_name(args.val_csv, "val"),
            label_names,
            args.label_policy,
        )
        test_df: pd.DataFrame | None = None
        test_summary: dict[str, Any] | None = None
        if args.test_csv is not None:
            test_df, test_summary = validate_manifest(
                args.test_csv,
                infer_split_name(args.test_csv, "test"),
                label_names,
                args.label_policy,
            )

        source_val_df: pd.DataFrame | None = None
        source_val_summary: dict[str, Any] | None = None
        if args.source_val_csv is not None:
            source_val_df, source_val_summary = validate_manifest(
                args.source_val_csv,
                infer_split_name(args.source_val_csv, "source_val"),
                label_names,
                args.label_policy,
            )

        source_test_df: pd.DataFrame | None = None
        source_test_summary: dict[str, Any] | None = None
        if args.source_test_csv is not None:
            source_test_df, source_test_summary = validate_manifest(
                args.source_test_csv,
                infer_split_name(args.source_test_csv, "source_test"),
                label_names,
                args.label_policy,
            )

        split_frames = {"support": support_df, "val": val_df}
        if test_df is not None:
            split_frames["test"] = test_df
        leakage_checks, leakage_failures = check_leakage(split_frames)
        if leakage_failures:
            raise StageFailure("Leakage detected between target splits:\n- " + "\n- ".join(leakage_failures))

        hospital_model, checkpoint_metadata, label_names = build_adapter_model(
            checkpoint_path=args.checkpoint,
            device=device,
            adapter_bottleneck=args.adapter_bottleneck,
            adapter_dropout=args.adapter_dropout,
            adapter_scale_init=args.adapter_scale_init,
            disable_hospital_bias=args.disable_hospital_bias,
            train_classifier_head=args.train_classifier_head,
        )
        parameter_summary = configure_trainable_parameters(
            hospital_model,
            train_classifier_head=args.train_classifier_head,
            train_hospital_bias=not args.disable_hospital_bias,
        )
        verify_parameter_summary(parameter_summary, args.train_classifier_head)

        print(f"device: {device}")
        print(f"run_dir: {args.run_dir}")
        print(f"labels: {label_names}")
        print(f"image_size: {image_size}")
        print(f"total parameters: {parameter_summary['total_parameters']}")
        print(f"trainable parameters: {parameter_summary['trainable_parameters']}")
        print(f"trainable parameter names: {parameter_summary['trainable_parameter_names']}")

        support_loader = build_dataloader(
            support_df,
            label_names=label_names,
            image_size=image_size,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        val_loader = build_dataloader(
            val_df,
            label_names=label_names,
            image_size=image_size,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        test_loader = (
            build_dataloader(
                test_df,
                label_names=label_names,
                image_size=image_size,
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
                num_workers=args.num_workers,
            )
            if test_df is not None
            else None
        )
        source_val_loader = (
            build_dataloader(
                source_val_df,
                label_names=label_names,
                image_size=image_size,
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
                num_workers=args.num_workers,
            )
            if source_val_df is not None
            else None
        )
        source_test_loader = (
            build_dataloader(
                source_test_df,
                label_names=label_names,
                image_size=image_size,
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
                num_workers=args.num_workers,
            )
            if source_test_df is not None
            else None
        )

        first_support_batch = next(iter(support_loader))
        identity_max_abs_logit_diff = run_identity_check(
            hospital_model.base_model,
            hospital_model,
            first_support_batch,
            device,
        )
        print(f"identity max abs logit diff: {identity_max_abs_logit_diff:.8f}")

        pos_weight = None
        pos_weight_values: list[float] | None = None
        if args.use_pos_weight:
            pos_weight = compute_pos_weight_from_dataframe(
                support_df,
                label_names=label_names,
                label_policy=args.label_policy,
                clamp_max=args.pos_weight_max,
            ).to(device)
            pos_weight_values = [float(value) for value in pos_weight.detach().cpu().tolist()]
            print(f"support pos_weight: {pos_weight_values}")

        debug_overfit_report = None
        if args.debug_overfit_one_batch:
            debug_overfit_report = run_debug_overfit_one_batch(
                hospital_model,
                support_loader,
                device,
                train_classifier_head=args.train_classifier_head,
                lr=args.lr,
                weight_decay=args.weight_decay,
                steps=args.debug_overfit_steps,
                pos_weight=pos_weight,
            )

        pooled_feature_dim = int(hospital_model.pooled_feature_dim)
        config = {
            "run_name": args.run_name,
            "source_hospital": args.source_hospital,
            "target_hospital": args.target_hospital,
            "checkpoint": args.checkpoint,
            "support_csv": args.support_csv,
            "val_csv": args.val_csv,
            "test_csv": args.test_csv,
            "source_val_csv": args.source_val_csv,
            "source_test_csv": args.source_test_csv,
            "label_policy": args.label_policy,
            "image_size": image_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "adapter_bottleneck": args.adapter_bottleneck,
            "adapter_dropout": args.adapter_dropout,
            "adapter_scale_init": args.adapter_scale_init,
            "train_classifier_head": args.train_classifier_head,
            "use_pos_weight": args.use_pos_weight,
            "pos_weight_max": args.pos_weight_max,
            "pos_weight": pos_weight_values,
            "disable_hospital_bias": args.disable_hospital_bias,
            "device": str(device),
            "paths": args.paths,
            "label_names": label_names,
            "checkpoint_metadata": checkpoint_metadata,
            "parameter_summary": parameter_summary,
            "identity_max_abs_logit_diff": identity_max_abs_logit_diff,
            "debug_overfit_one_batch": debug_overfit_report,
            "target_split_summaries": {
                "support": support_summary,
                "val": val_summary,
                "test": test_summary,
            },
            "source_split_summaries": {
                "source_val": source_val_summary,
                "source_test": source_test_summary,
            },
            "leakage_checks": leakage_checks,
        }
        save_json(config, args.paths["config"])

        optimizer = build_optimizer(hospital_model, lr=args.lr, weight_decay=args.weight_decay)

        history: list[dict[str, Any]] = []
        best_record: dict[str, Any] | None = None
        best_epoch = 0
        epochs_without_improvement = 0
        stopped_early = False

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                hospital_model,
                support_loader,
                optimizer,
                device,
                train_classifier_head=args.train_classifier_head,
                pos_weight=pos_weight,
            )
            val_metrics = evaluate_model(hospital_model, val_loader, device, label_names)
            history_item = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "val_macro_auroc": val_metrics["macro_auroc"],
                "val_macro_auprc": val_metrics["macro_auprc"],
                "val_mean_ap": val_metrics["mean_ap"],
                "val_defined_auroc_labels": int(val_metrics["defined_auroc_labels"]),
                "val_invalid_auroc_labels": "|".join(val_metrics["invalid_auroc_labels"]),
            }
            history.append(history_item)
            write_train_log(history, args.paths["train_log"])

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
                best_epoch = int(epoch)
                save_checkpoint(
                    path=args.paths["adapter_best"],
                    model=hospital_model,
                    epoch=epoch,
                    args=args,
                    label_names=label_names,
                    pooled_feature_dim=pooled_feature_dim,
                    best_metric_name=metric_name,
                    best_metric_value=metric_value,
                    parameter_summary=parameter_summary,
                    identity_max_abs_logit_diff=identity_max_abs_logit_diff,
                    debug_overfit=debug_overfit_report,
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_bce={val_metrics['loss']:.4f} | "
                f"val_macro_auroc={format_metric(val_metrics['macro_auroc'])} | "
                f"val_mean_ap={format_metric(val_metrics['mean_ap'])}"
            )

            if epochs_without_improvement >= args.patience:
                stopped_early = True
                print(f"early stopping after epoch {epoch} due to {epochs_without_improvement} non-improving epochs")
                break

        if best_record is None:
            raise StageFailure("Training did not produce a best adapter checkpoint.")

        save_checkpoint(
            path=args.paths["adapter_last"],
            model=hospital_model,
            epoch=history[-1]["epoch"],
            args=args,
            label_names=label_names,
            pooled_feature_dim=pooled_feature_dim,
            best_metric_name=best_record["metric_name"],
            best_metric_value=best_record["metric_value"],
            parameter_summary=parameter_summary,
            identity_max_abs_logit_diff=identity_max_abs_logit_diff,
            debug_overfit=debug_overfit_report,
        )

        best_model, _, _, adapter_metadata = load_best_adapter_model(args, device)
        source_only_model, _, source_only_label_names = load_base_checkpoint(args.checkpoint, device)
        if source_only_label_names != label_names:
            raise StageFailure("Source-only checkpoint labels do not match adapter labels.")

        target_val_source_only_metrics = evaluate_model(source_only_model, val_loader, device, label_names)
        target_val_adapter_metrics = evaluate_model(best_model, val_loader, device, label_names)
        save_predictions_csv(
            val_df,
            target_val_adapter_metrics["probabilities"],
            args.paths["predictions_target_val"],
            label_names,
        )

        target_val_report = build_split_report(
            run_name=args.run_name,
            model_name="target_adapter",
            split_summary=val_summary,
            label_names=label_names,
            metrics=target_val_adapter_metrics,
            checkpoint_path=args.checkpoint,
            adapter_checkpoint_path=args.paths["adapter_best"],
            extra={
                "source_hospital": args.source_hospital,
                "target_hospital": args.target_hospital,
                "source_only_metrics": report_ready_metrics(target_val_source_only_metrics),
                "metric_deltas": {
                    "macro_auroc_delta": (
                        None
                        if target_val_adapter_metrics["macro_auroc"] is None or target_val_source_only_metrics["macro_auroc"] is None
                        else float(target_val_adapter_metrics["macro_auroc"] - target_val_source_only_metrics["macro_auroc"])
                    ),
                    "mean_ap_delta": (
                        None
                        if target_val_adapter_metrics["mean_ap"] is None or target_val_source_only_metrics["mean_ap"] is None
                        else float(target_val_adapter_metrics["mean_ap"] - target_val_source_only_metrics["mean_ap"])
                    ),
                },
            },
        )
        save_json(target_val_report, args.paths["target_val_report"])

        target_test_source_only_metrics = None
        target_test_adapter_metrics = None
        if test_loader is not None and test_df is not None and test_summary is not None:
            target_test_source_only_metrics = evaluate_model(source_only_model, test_loader, device, label_names)
            target_test_adapter_metrics = evaluate_model(best_model, test_loader, device, label_names)
            save_predictions_csv(
                test_df,
                target_test_adapter_metrics["probabilities"],
                args.paths["predictions_target_test"],
                label_names,
            )
            target_test_report = build_split_report(
                run_name=args.run_name,
                model_name="target_adapter",
                split_summary=test_summary,
                label_names=label_names,
                metrics=target_test_adapter_metrics,
                checkpoint_path=args.checkpoint,
                adapter_checkpoint_path=args.paths["adapter_best"],
                extra={
                    "source_hospital": args.source_hospital,
                    "target_hospital": args.target_hospital,
                    "source_only_metrics": report_ready_metrics(target_test_source_only_metrics),
                    "metric_deltas": {
                        "macro_auroc_delta": (
                            None
                            if target_test_adapter_metrics["macro_auroc"] is None or target_test_source_only_metrics["macro_auroc"] is None
                            else float(target_test_adapter_metrics["macro_auroc"] - target_test_source_only_metrics["macro_auroc"])
                        ),
                        "mean_ap_delta": (
                            None
                            if target_test_adapter_metrics["mean_ap"] is None or target_test_source_only_metrics["mean_ap"] is None
                            else float(target_test_adapter_metrics["mean_ap"] - target_test_source_only_metrics["mean_ap"])
                        ),
                    },
                },
            )
            save_json(target_test_report, args.paths["target_test_report"])
        else:
            target_test_report = None

        if source_val_loader is not None or source_test_loader is not None:
            source_eval_report: dict[str, Any] = {
                "run_name": args.run_name,
                "checkpoint": str(args.checkpoint.resolve()),
                "adapter_checkpoint": str(args.paths["adapter_best"].resolve()),
                "source_hospital": args.source_hospital,
                "target_hospital": args.target_hospital,
                "label_names": label_names,
                "splits": {},
            }
            if source_val_loader is not None and source_val_summary is not None:
                source_val_source_only_metrics = evaluate_model(source_only_model, source_val_loader, device, label_names)
                source_val_adapter_metrics = evaluate_model(best_model, source_val_loader, device, label_names)
                source_eval_report["splits"]["source_val"] = {
                    "summary": source_val_summary,
                    "source_only_metrics": report_ready_metrics(source_val_source_only_metrics),
                    "adapter_metrics": report_ready_metrics(source_val_adapter_metrics),
                }
            if source_test_loader is not None and source_test_summary is not None:
                source_test_source_only_metrics = evaluate_model(source_only_model, source_test_loader, device, label_names)
                source_test_adapter_metrics = evaluate_model(best_model, source_test_loader, device, label_names)
                source_eval_report["splits"]["source_test"] = {
                    "summary": source_test_summary,
                    "source_only_metrics": report_ready_metrics(source_test_source_only_metrics),
                    "adapter_metrics": report_ready_metrics(source_test_adapter_metrics),
                }
            save_json(source_eval_report, args.paths["source_eval_report"])

        final_config = {
            **config,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_metric_name": best_record["metric_name"],
            "best_metric_value": best_record["metric_value"],
            "best_checkpoint": args.paths["adapter_best"],
            "last_checkpoint": args.paths["adapter_last"],
            "adapter_checkpoint_metadata": adapter_metadata,
            "source_only_target_val_metrics": report_ready_metrics(target_val_source_only_metrics),
            "target_val_metrics": report_ready_metrics(target_val_adapter_metrics),
            "source_only_target_test_metrics": (
                report_ready_metrics(target_test_source_only_metrics)
                if target_test_source_only_metrics is not None
                else None
            ),
            "target_test_metrics": (
                report_ready_metrics(target_test_adapter_metrics)
                if target_test_adapter_metrics is not None
                else None
            ),
        }
        save_json(final_config, args.paths["config"])

        print(f"best epoch: {best_epoch}")
        print(f"target val macro AUROC: {format_metric(target_val_adapter_metrics['macro_auroc'])}")
        if target_test_report is not None:
            print(f"target test macro AUROC: {format_metric(target_test_report['metrics']['macro_auroc'])}")
        print(f"run folder: {args.run_dir}")
        return 0
    except StageFailure as exc:
        print(f"StageFailure: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - top-level safety
        print(f"Unhandled error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
