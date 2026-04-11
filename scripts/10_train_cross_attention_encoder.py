#!/usr/bin/env python3
"""Train a frozen-encoder image-report cross-attention model for NIH CXR14."""

from __future__ import annotations

import argparse
import platform
import shlex
import sys
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'numpy'.") from exc

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'torch'.") from exc

from source_cross_attention_common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_ROOT,
    DEFAULT_DEVICE,
    DEFAULT_DROPOUT,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EXPERIMENTS_ROOT,
    DEFAULT_FUSION_DIM,
    DEFAULT_FUSION_HEADS,
    DEFAULT_FUSION_LAYERS,
    DEFAULT_MANIFEST_CSV,
    DEFAULT_MAX_LENGTH,
    DEFAULT_NUM_WORKERS,
    DEFAULT_REPORTS_ROOT,
    DEFAULT_SEED,
    DEFAULT_TEXT_MODEL_ID,
    DEFAULT_ECE_BINS,
    CrossAttentionMultimodalModel,
    NIHMultimodalDataset,
    MultimodalCollator,
    append_jsonl,
    build_image_encoder_bundle,
    build_text_encoder_bundle,
    compute_pos_weight,
    format_metric,
    get_autocast_context,
    load_manifest_records,
    move_batch_to_device,
    resolve_device,
    resolve_experiment_identity,
    seed_everything,
    selection_tuple,
    sha256_file,
    slugify,
    summarize_split_metrics,
    to_serializable,
    tune_thresholds,
    utc_now_iso,
    write_json,
)

DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5
DEFAULT_OPERATION_LABEL = "source_cross_attention_training"
DEFAULT_SELECTION_METRIC = "macro_auroc"
DEFAULT_LOG_EVERY_STEPS = 200


def labels_from_samples(samples: list[Any]) -> np.ndarray:
    return np.asarray([sample.labels for sample in samples], dtype=np.float32)


def build_dataloader(
    samples: list[Any],
    *,
    tokenizer: Any,
    image_transform: Any,
    batch_size: int,
    num_workers: int,
    max_length: int,
    text_prefix: str,
    text_suffix: str,
    normalize_whitespace: bool,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    dataset = NIHMultimodalDataset(
        samples,
        image_transform=image_transform,
        text_prefix=text_prefix,
        text_suffix=text_suffix,
        normalize_whitespace=normalize_whitespace,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=MultimodalCollator(tokenizer=tokenizer, max_length=max_length),
    )


def train_one_epoch(
    *,
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    fp16_on_cuda: bool,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_batches = len(loader)
    for step_idx, batch in enumerate(loader, start=1):
        encoded = batch["encoded"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        if encoded is None or pixel_values is None or labels is None:
            continue

        text_inputs = move_batch_to_device(encoded, device)
        pixel_values = pixel_values.to(device, non_blocking=(device.type == "cuda"))
        labels = labels.to(device, non_blocking=(device.type == "cuda"))
        optimizer.zero_grad(set_to_none=True)
        with get_autocast_context(device, fp16_on_cuda):
            logits, _ = model(
                pixel_values=pixel_values,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        if step_idx == 1 or step_idx % DEFAULT_LOG_EVERY_STEPS == 0 or step_idx == total_batches:
            running_loss = total_loss / max(total_examples, 1)
            print(
                f"[train] step={step_idx}/{total_batches} examples={total_examples} "
                f"running_loss={running_loss:.6f}",
                flush=True,
            )

    if total_examples == 0:
        raise SystemExit("Training loader produced zero usable examples.")
    return total_loss / total_examples, total_examples


@torch.no_grad()
def evaluate_model(
    *,
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    fp16_on_cuda: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_chunks: list[torch.Tensor] = []
    targets_chunks: list[torch.Tensor] = []

    total_batches = len(loader)
    for step_idx, batch in enumerate(loader, start=1):
        encoded = batch["encoded"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        if encoded is None or pixel_values is None or labels is None:
            continue
        text_inputs = move_batch_to_device(encoded, device)
        pixel_values = pixel_values.to(device, non_blocking=(device.type == "cuda"))
        labels = labels.to(device, non_blocking=(device.type == "cuda"))
        with get_autocast_context(device, fp16_on_cuda):
            logits, _ = model(
                pixel_values=pixel_values,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            loss = criterion(logits, labels)
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        logits_chunks.append(logits.detach().cpu())
        targets_chunks.append(labels.detach().cpu())
        if step_idx == 1 or step_idx % DEFAULT_LOG_EVERY_STEPS == 0 or step_idx == total_batches:
            running_loss = total_loss / max(total_examples, 1)
            print(
                f"[eval] step={step_idx}/{total_batches} examples={total_examples} "
                f"running_loss={running_loss:.6f}",
                flush=True,
            )

    if total_examples == 0:
        raise SystemExit("Evaluation loader produced zero usable examples.")
    logits = torch.cat(logits_chunks, dim=0).numpy().astype(np.float32)
    targets = torch.cat(targets_chunks, dim=0).numpy().astype(np.float32)
    return total_loss / total_examples, logits, targets


def build_checkpoint_payload(
    *,
    model: nn.Module,
    epoch: int,
    best_summary: dict[str, Any],
    label_names: list[str],
    tuned_thresholds: np.ndarray | None,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    trainable_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("image_encoder.") and not key.startswith("text_encoder.")
    }
    return {
        "epoch": int(epoch),
        "state_dict": trainable_state,
        "state_dict_kind": "trainable_only_without_frozen_backbones",
        "best_summary": to_serializable(best_summary),
        "label_names": list(label_names),
        "tuned_thresholds": tuned_thresholds.tolist() if tuned_thresholds is not None else None,
        "model_config": to_serializable(model_config),
    }


def format_bash_command(argv: list[str]) -> str:
    return " \\\n  ".join(shlex.quote(part) for part in argv)


def build_recreation_report(
    *,
    experiment_dir: Path,
    config: dict[str, Any],
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> str:
    command = ["python", str(Path(__file__).resolve()), *config["argv"][1:]]
    return "\n".join(
        [
            "# Source Cross-Attention Training Recreation Report",
            "",
            "## Experiment",
            "",
            f"- Experiment directory: `{experiment_dir}`",
            f"- Manifest: `{config['manifest_csv']}`",
            f"- Data root: `{config['data_root']}`",
            f"- Reports root: `{config['reports_root']}`",
            f"- Frozen image encoder: `{config['image_encoder']['encoder_id']}`",
            f"- Frozen text encoder: `{config['text_encoder']['model_id']}`",
            f"- Fusion dim: `{config['fusion_dim']}`",
            f"- Fusion layers: `{config['fusion_layers']}`",
            f"- Embedding dim: `{config['embedding_dim']}`",
            "",
            "## Exact Command",
            "",
            "```bash",
            format_bash_command(command),
            "```",
            "",
            "## Final Metrics",
            "",
            f"- Validation macro AUROC: `{format_metric(val_metrics['macro']['auroc'])}`",
            f"- Validation macro average precision: `{format_metric(val_metrics['macro']['average_precision'])}`",
            f"- Test macro AUROC: `{format_metric(test_metrics['macro']['auroc'])}`",
            f"- Test macro average precision: `{format_metric(test_metrics['macro']['average_precision'])}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen-backbone multimodal cross-attention model on NIH CXR14 image/report pairs. "
            "The learned fusion stack produces better fused embeddings for downstream export."
        )
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--selection-metric", choices=("macro_auroc", "macro_average_precision"), default=DEFAULT_SELECTION_METRIC)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--fp16-on-cuda", action="store_true")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--text-model-id", type=str, default=DEFAULT_TEXT_MODEL_ID)
    parser.add_argument("--tokenizer-id", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--trust-manifest-paths",
        action="store_true",
        help=(
            "Skip upfront per-row image/report existence checks during manifest load. "
            "Use this when the manifest paths are already known to be valid."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--fusion-dim", type=int, default=DEFAULT_FUSION_DIM)
    parser.add_argument("--fusion-heads", type=int, default=DEFAULT_FUSION_HEADS)
    parser.add_argument("--fusion-layers", type=int, default=DEFAULT_FUSION_LAYERS)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--text-prefix", type=str, default="")
    parser.add_argument("--text-suffix", type=str, default="")
    parser.add_argument("--disable-normalize-whitespace", action="store_true")
    parser.add_argument(
        "--disable-gated-hybrid",
        action="store_true",
        help=(
            "Disable the gated hybrid export path and train the original pure cross-attention embedding head only."
        ),
    )
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0.")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.lr <= 0.0:
        raise SystemExit("--lr must be positive.")
    if args.weight_decay < 0.0:
        raise SystemExit("--weight-decay must be >= 0.")
    if args.patience < 0:
        raise SystemExit("--patience must be >= 0.")
    if args.max_length <= 0:
        raise SystemExit("--max-length must be positive.")
    if args.fusion_dim <= 0 or args.fusion_heads <= 0 or args.fusion_layers <= 0 or args.embedding_dim <= 0:
        raise SystemExit("Fusion and embedding dimensions must be positive.")
    if args.fusion_dim % args.fusion_heads != 0:
        raise SystemExit("--fusion-dim must be divisible by --fusion-heads.")
    if not 0.0 <= args.dropout < 1.0:
        raise SystemExit("--dropout must be in [0, 1).")

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    manifest_csv = args.manifest_csv.resolve()
    data_root = args.data_root.resolve()
    reports_root = args.reports_root.resolve()
    experiments_root = args.experiments_root.resolve()
    normalize_whitespace = not bool(args.disable_normalize_whitespace)
    gated_hybrid = not bool(args.disable_gated_hybrid)

    print(f"[startup] device={device} manifest={manifest_csv}", flush=True)
    print(
        f"[startup] loading manifest records verify_files={not args.trust_manifest_paths}",
        flush=True,
    )
    label_columns, records_by_split = load_manifest_records(
        manifest_csv,
        data_root=data_root,
        reports_root=reports_root,
        splits=["train", "val", "test"],
        max_samples_per_split=args.max_samples_per_split,
        verify_files=not args.trust_manifest_paths,
    )
    label_names = [column.removeprefix("label_") for column in label_columns]
    print(
        "[startup] manifest loaded "
        f"train={len(records_by_split['train'])} val={len(records_by_split['val'])} test={len(records_by_split['test'])}",
        flush=True,
    )

    print("[startup] building frozen image encoder", flush=True)
    image_bundle = build_image_encoder_bundle()
    print("[startup] image encoder ready", flush=True)
    print("[startup] building frozen text encoder", flush=True)
    text_bundle = build_text_encoder_bundle(
        model_id=args.text_model_id,
        tokenizer_id=args.tokenizer_id,
        revision=args.revision,
        trust_remote_code=bool(args.trust_remote_code),
        cache_dir=args.cache_dir,
    )
    print("[startup] text encoder ready", flush=True)

    model_config = {
        "image_encoder_id": "resnet50",
        "image_feature_dim": image_bundle.feature_dim,
        "image_token_count": image_bundle.spatial_token_count,
        "text_model_id": text_bundle.model_id,
        "tokenizer_id": text_bundle.tokenizer_id,
        "text_hidden_size": text_bundle.hidden_size,
        "legacy_text_feature_dim": text_bundle.projected_embedding_size,
        "fusion_dim": int(args.fusion_dim),
        "fusion_heads": int(args.fusion_heads),
        "fusion_layers": int(args.fusion_layers),
        "embedding_dim": int(args.embedding_dim),
        "dropout": float(args.dropout),
        "gated_hybrid": gated_hybrid,
        "num_labels": len(label_names),
    }

    generated_slug = "__".join(
        [
            DEFAULT_OPERATION_LABEL,
            "nih_cxr14",
            slugify("resnet50", fallback="image"),
            slugify(args.text_model_id.split("/")[-1], fallback="text"),
            f"fd{args.fusion_dim}",
            f"fl{args.fusion_layers}",
            f"ed{args.embedding_dim}",
        ]
    )
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        operation_label=DEFAULT_OPERATION_LABEL,
        overwrite=bool(args.overwrite),
    )

    print("[startup] building dataloaders", flush=True)
    train_loader = build_dataloader(
        records_by_split["train"],
        tokenizer=text_bundle.tokenizer,
        image_transform=image_bundle.preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        text_prefix=args.text_prefix,
        text_suffix=args.text_suffix,
        normalize_whitespace=normalize_whitespace,
        device=device,
        shuffle=True,
    )
    val_loader = build_dataloader(
        records_by_split["val"],
        tokenizer=text_bundle.tokenizer,
        image_transform=image_bundle.preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        text_prefix=args.text_prefix,
        text_suffix=args.text_suffix,
        normalize_whitespace=normalize_whitespace,
        device=device,
        shuffle=False,
    )
    test_loader = build_dataloader(
        records_by_split["test"],
        tokenizer=text_bundle.tokenizer,
        image_transform=image_bundle.preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        text_prefix=args.text_prefix,
        text_suffix=args.text_suffix,
        normalize_whitespace=normalize_whitespace,
        device=device,
        shuffle=False,
    )
    print("[startup] dataloaders ready", flush=True)

    print("[startup] initializing cross-attention model", flush=True)
    model = CrossAttentionMultimodalModel(
        image_encoder=image_bundle.model,
        text_encoder=text_bundle.model,
        text_hidden_size=text_bundle.hidden_size,
        legacy_text_feature_dim=text_bundle.projected_embedding_size,
        num_labels=len(label_names),
        image_feature_dim=image_bundle.feature_dim,
        image_token_count=image_bundle.spatial_token_count,
        fusion_dim=args.fusion_dim,
        num_heads=args.fusion_heads,
        num_layers=args.fusion_layers,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        gated_hybrid=gated_hybrid,
    ).to(device)
    print("[startup] model ready", flush=True)

    pos_weight = compute_pos_weight(labels_from_samples(records_by_split["train"])).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16_on_cuda and device.type == "cuda"))

    script_path = Path(__file__).resolve()
    script_sha256 = sha256_file(script_path)
    config = {
        "argv": list(sys.argv),
        "run_date_utc": utc_now_iso(),
        "operation_label": DEFAULT_OPERATION_LABEL,
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "manifest_csv": str(manifest_csv),
        "data_root": str(data_root),
        "reports_root": str(reports_root),
        "label_columns": label_columns,
        "label_names": label_names,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "selection_metric": str(args.selection_metric),
        "device_requested": str(args.device),
        "device_resolved": str(device),
        "fp16_on_cuda": bool(args.fp16_on_cuda),
        "max_length": int(args.max_length),
        "normalize_whitespace": bool(normalize_whitespace),
        "trust_manifest_paths": bool(args.trust_manifest_paths),
        "text_prefix": str(args.text_prefix),
        "text_suffix": str(args.text_suffix),
        "max_samples_per_split": args.max_samples_per_split,
        "script_path": str(script_path),
        "script_sha256": script_sha256,
        "image_encoder": {
            "encoder_id": "resnet50",
            "weights": image_bundle.weights_label,
            "input_size": list(image_bundle.resolved_input_size),
            "feature_dim": int(image_bundle.feature_dim),
            "spatial_token_count": int(image_bundle.spatial_token_count),
        },
        "text_encoder": {
            "model_id": text_bundle.model_id,
            "tokenizer_id": text_bundle.tokenizer_id,
            "revision": args.revision,
            "trust_remote_code": bool(args.trust_remote_code),
            "hidden_size": int(text_bundle.hidden_size),
            "projected_embedding_size": int(text_bundle.projected_embedding_size),
            "max_position_embeddings": text_bundle.max_position_embeddings,
        },
        "fusion_dim": int(args.fusion_dim),
        "fusion_heads": int(args.fusion_heads),
        "fusion_layers": int(args.fusion_layers),
        "embedding_dim": int(args.embedding_dim),
        "dropout": float(args.dropout),
        "gated_hybrid": gated_hybrid,
        "split_inputs": {
            split: {
                "num_rows": len(records_by_split[split]),
                "first_row_id": records_by_split[split][0].row_id if records_by_split[split] else None,
            }
            for split in ("train", "val", "test")
        },
    }
    write_json(experiment_dir / "config.json", config)
    (experiment_dir / "train_log.jsonl").write_text("", encoding="utf-8")

    print(f"[info] experiment_dir={experiment_dir}")
    print(f"[info] train_rows={len(records_by_split['train'])} val_rows={len(records_by_split['val'])} test_rows={len(records_by_split['test'])}")
    print(f"[info] device={device} fusion_dim={args.fusion_dim} fusion_layers={args.fusion_layers} embedding_dim={args.embedding_dim}")

    best_epoch = 0
    best_summary: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_loss, train_examples = train_one_epoch(
            loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            fp16_on_cuda=bool(args.fp16_on_cuda),
        )
        val_loss, val_logits, val_targets = evaluate_model(
            loader=val_loader,
            model=model,
            criterion=criterion,
            device=device,
            fp16_on_cuda=bool(args.fp16_on_cuda),
        )
        current_summary = summarize_split_metrics(
            split="val",
            loss=val_loss,
            targets=val_targets,
            logits=val_logits,
            label_names=label_names,
            tuned_thresholds=np.full((len(label_names),), 0.5, dtype=np.float32),
        )
        improved = best_summary is None or selection_tuple(current_summary) > selection_tuple(best_summary)
        if improved:
            best_epoch = epoch
            best_summary = current_summary
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                build_checkpoint_payload(
                    model=model,
                    epoch=epoch,
                    best_summary=current_summary,
                    label_names=label_names,
                    tuned_thresholds=None,
                    model_config=model_config,
                ),
                experiment_dir / "best.ckpt",
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_log = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_examples": int(train_examples),
            "val_loss": float(val_loss),
            "val_macro_auroc": current_summary["macro"]["auroc"],
            "val_macro_average_precision": current_summary["macro"]["average_precision"],
            "val_macro_ece": current_summary["macro"]["ece"],
            "val_macro_f1_at_0.5": current_summary["macro"]["f1_at_0.5"],
            "improved": bool(improved),
            "elapsed_sec": float(time.time() - epoch_started),
        }
        append_jsonl(experiment_dir / "train_log.jsonl", epoch_log)
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_macro_auroc={format_metric(current_summary['macro']['auroc'])} improved={str(improved).lower()}"
        )

        if args.patience >= 0 and epochs_without_improvement > args.patience:
            print(f"[early-stop] epoch={epoch} patience={args.patience}")
            break

    if best_state is None or best_summary is None:
        raise SystemExit("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    val_loss, val_logits, val_targets = evaluate_model(
        loader=val_loader,
        model=model,
        criterion=criterion,
        device=device,
        fp16_on_cuda=bool(args.fp16_on_cuda),
    )
    test_loss, test_logits, test_targets = evaluate_model(
        loader=test_loader,
        model=model,
        criterion=criterion,
        device=device,
        fp16_on_cuda=bool(args.fp16_on_cuda),
    )

    val_probs = 1.0 / (1.0 + np.exp(-val_logits.astype(np.float64)))
    tuned_thresholds, threshold_payload = tune_thresholds(val_targets, val_probs, label_names)
    val_metrics = summarize_split_metrics(
        split="val",
        loss=val_loss,
        targets=val_targets,
        logits=val_logits,
        label_names=label_names,
        tuned_thresholds=tuned_thresholds,
    )
    test_metrics = summarize_split_metrics(
        split="test",
        loss=test_loss,
        targets=test_targets,
        logits=test_logits,
        label_names=label_names,
        tuned_thresholds=tuned_thresholds,
    )

    torch.save(
        build_checkpoint_payload(
            model=model,
            epoch=best_epoch,
            best_summary=val_metrics,
            label_names=label_names,
            tuned_thresholds=tuned_thresholds,
            model_config=model_config,
        ),
        experiment_dir / "best.ckpt",
    )
    write_json(experiment_dir / "val_metrics.json", val_metrics)
    write_json(experiment_dir / "test_metrics.json", test_metrics)
    write_json(experiment_dir / "val_f1_thresholds.json", threshold_payload)

    experiment_meta = {
        "argv": list(sys.argv),
        "run_date_utc": utc_now_iso(),
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "operation_label": DEFAULT_OPERATION_LABEL,
        "manifest_csv": str(manifest_csv),
        "data_root": str(data_root),
        "reports_root": str(reports_root),
        "device_resolved": str(device),
        "selection_metric": str(args.selection_metric),
        "best_epoch": int(best_epoch),
        "stopped_early": bool(best_epoch < args.epochs),
        "checkpoint_path": str(experiment_dir / "best.ckpt"),
        "thresholds_path": str(experiment_dir / "val_f1_thresholds.json"),
        "macro_metrics": {
            "val": val_metrics["macro"],
            "test": test_metrics["macro"],
        },
        "model_config": model_config,
    }
    write_json(experiment_dir / "experiment_meta.json", experiment_meta)
    (experiment_dir / "recreation_report.md").write_text(
        build_recreation_report(
            experiment_dir=experiment_dir,
            config=config,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        ),
        encoding="utf-8",
    )

    print(f"[done] best_epoch={best_epoch} val_macro_auroc={format_metric(val_metrics['macro']['auroc'])} test_macro_auroc={format_metric(test_metrics['macro']['auroc'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
