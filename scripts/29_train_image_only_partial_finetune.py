#!/usr/bin/env python3
"""Train a target-only image-level partial-backbone finetune baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import experiment_layout
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ViT_B_16_Weights, vit_b_16


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest/manifest_mimic_target_1900_train_100_val.csv")
DEFAULT_DATA_ROOT = Path("/workspace")
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 4
DEFAULT_EPOCHS = 10
DEFAULT_LR = 5e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 3
DEFAULT_SEED = 1337
DEFAULT_DEVICE = "auto"
DEFAULT_OPERATION_LABEL = "domain_transfer_partial_finetune_training"
DEFAULT_EXPERIMENT_ID_WIDTH = 4
DEFAULT_TRAINABLE_BLOCKS = 1
DEFAULT_BACKBONE_NAME = "torchvision/vit_b_16.imagenet1k_v1"
HELPER_SCRIPT = Path("/workspace/scripts/15_train_domain_transfer_linear_probe.py")


@dataclass(frozen=True)
class ImageSample:
    row_id: str
    image_path: str
    resolved_path: Path
    labels: tuple[float, ...]


@dataclass
class SplitImageData:
    alias: str
    domain: str
    split: str
    samples: list[ImageSample]
    labels: np.ndarray


@dataclass(frozen=True)
class EvaluationPlan:
    name: str
    train_alias: str
    selection_alias: str
    primary_test_alias: str
    split_specs: tuple[tuple[str, str, str], ...]
    output_name_map: dict[str, str]
    thresholds_filename: str


class ManifestImageDataset(Dataset):
    def __init__(self, samples: list[ImageSample], *, transform: Any) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample.resolved_path) as image:
            rgb = image.convert("RGB")
            pixel_values = self.transform(rgb)
        return {
            "pixel_values": pixel_values,
            "targets": torch.tensor(sample.labels, dtype=torch.float32),
        }


def load_transfer_helpers() -> Any:
    if not HELPER_SCRIPT.exists():
        raise SystemExit(f"Helper script not found: {HELPER_SCRIPT}")
    module_name = "domain_transfer_partial_finetune_helpers"
    spec = importlib.util.spec_from_file_location(module_name, HELPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load helper module from {HELPER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def slugify(value: str, *, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or fallback


def ensure_operation_prefix(name: str, operation_label: str = DEFAULT_OPERATION_LABEL) -> str:
    normalized_label = slugify(operation_label, fallback="operation")
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
    requested = (requested_name or "").strip() or None
    base_name = ensure_operation_prefix(requested or generated_slug)
    return experiment_layout.resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=base_name if requested else None,
        generated_slug=base_name,
        overwrite=overwrite,
        id_width=id_width,
    )


def format_bash_command(argv: list[str]) -> str:
    return " \\\n+  ".join(shlex.quote(part) for part in argv)


def candidate_image_paths(data_root: Path, image_path: str) -> list[Path]:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return [candidate]
    roots: list[Path] = []
    for root in [data_root.resolve(), Path("/workspace").resolve(), (Path("/workspace") / "data").resolve()]:
        if root not in roots:
            roots.append(root)
    return [root / image_path for root in roots]


def resolve_manifest_image_path(data_root: Path, image_path: str) -> Path:
    for candidate in candidate_image_paths(data_root, image_path):
        if candidate.exists():
            return candidate
    attempted = ", ".join(str(path) for path in candidate_image_paths(data_root, image_path))
    raise FileNotFoundError(f"{image_path} not found. Tried: {attempted}")


def build_split_image_data(
    *,
    alias: str,
    domain: str,
    split: str,
    manifest_records: dict[str, Any],
    data_root: Path,
    max_rows: int | None,
    num_labels: int,
) -> SplitImageData:
    selected_records = list(manifest_records.values())
    if max_rows is not None:
        selected_records = selected_records[:max_rows]
    samples: list[ImageSample] = []
    failures: list[str] = []
    labels = np.zeros((len(selected_records), num_labels), dtype=np.float32)
    for index, record in enumerate(selected_records):
        try:
            resolved_path = resolve_manifest_image_path(data_root, record.image_path)
        except FileNotFoundError as exc:
            if len(failures) < 10:
                failures.append(str(exc))
            continue
        samples.append(
            ImageSample(
                row_id=record.row_id,
                image_path=record.image_path,
                resolved_path=resolved_path,
                labels=tuple(record.labels),
            )
        )
        labels[len(samples) - 1] = np.asarray(record.labels, dtype=np.float32)
    if failures:
        sample = "\n".join(failures)
        raise SystemExit(
            f"Could not resolve some images for alias={alias} domain={domain} split={split}.\nExamples:\n{sample}"
        )
    if not samples:
        raise SystemExit(f"No usable image rows for alias={alias} domain={domain} split={split}.")
    return SplitImageData(
        alias=alias,
        domain=domain,
        split=split,
        samples=samples,
        labels=labels[: len(samples)],
    )


def build_dataloader(
    split_data: SplitImageData,
    *,
    transform: Any,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ManifestImageDataset(split_data.samples, transform=transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def build_evaluation_plan(*, split_profile: str) -> EvaluationPlan:
    if split_profile == "mimic_target":
        return EvaluationPlan(
            name=split_profile,
            train_alias="target_train",
            selection_alias="target_val",
            primary_test_alias="target_test",
            split_specs=(
                ("target_train", "d2_mimic", "train"),
                ("target_val", "d2_mimic", "val"),
                ("target_test", "d2_mimic", "test"),
            ),
            output_name_map={
                "target_train": "target_train_metrics.json",
                "target_val": "target_val_metrics.json",
                "target_test": "target_test_metrics.json",
            },
            thresholds_filename="target_val_f1_thresholds.json",
        )
    if split_profile == "chexpert_target":
        return EvaluationPlan(
            name=split_profile,
            train_alias="target_train",
            selection_alias="target_val",
            primary_test_alias="target_test",
            split_specs=(
                ("target_train", "d1_chexpert", "train"),
                ("target_val", "d1_chexpert", "val"),
                ("target_test", "d1_chexpert", "test"),
            ),
            output_name_map={
                "target_train": "target_train_metrics.json",
                "target_val": "target_val_metrics.json",
                "target_test": "target_test_metrics.json",
            },
            thresholds_filename="target_val_f1_thresholds.json",
        )
    raise SystemExit(f"Unsupported --split-profile: {split_profile}")


def train_one_epoch(
    *,
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    fp16_on_cuda: bool,
    get_autocast_context: Any,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=(device.type == "cuda"))
        targets = batch["targets"].to(device, non_blocking=(device.type == "cuda"))
        optimizer.zero_grad(set_to_none=True)
        with get_autocast_context(device, fp16_on_cuda):
            logits = model(pixel_values)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise SystemExit("Training loader produced zero usable examples.")
    return total_loss / total_examples


@torch.no_grad()
def evaluate_model(
    *,
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    fp16_on_cuda: bool,
    get_autocast_context: Any,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_chunks: list[torch.Tensor] = []
    targets_chunks: list[torch.Tensor] = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=(device.type == "cuda"))
        targets = batch["targets"].to(device, non_blocking=(device.type == "cuda"))
        with get_autocast_context(device, fp16_on_cuda):
            logits = model(pixel_values)
            loss = criterion(logits, targets)
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        logits_chunks.append(logits.detach().cpu())
        targets_chunks.append(targets.detach().cpu())
    if total_examples == 0:
        raise SystemExit("Evaluation loader produced zero usable examples.")
    logits = torch.cat(logits_chunks, dim=0).numpy().astype(np.float32)
    targets = torch.cat(targets_chunks, dim=0).numpy().astype(np.float32)
    return total_loss / total_examples, logits, targets


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    return total, trainable


def configure_partial_finetune(
    model: nn.Module,
    *,
    trainable_blocks: int,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    if trainable_blocks <= 0:
        raise SystemExit("--trainable-blocks must be positive.")
    for parameter in model.parameters():
        parameter.requires_grad = False
    total_blocks = len(model.encoder.layers)
    if trainable_blocks > total_blocks:
        raise SystemExit(
            f"Requested {trainable_blocks} trainable blocks, but backbone only has {total_blocks} encoder blocks."
        )
    trainable_prefixes = [
        f"encoder.layers.encoder_layer_{block_idx}"
        for block_idx in range(total_blocks - trainable_blocks, total_blocks)
    ]
    trainable_prefixes.extend(["encoder.ln", "heads"])
    trainable_names: list[str] = []
    initial_backbone_state: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            parameter.requires_grad = True
            trainable_names.append(name)
            if not name.startswith("heads."):
                initial_backbone_state[name] = parameter.detach().cpu().clone()
    if not trainable_names:
        raise SystemExit("No trainable parameters were selected for partial finetuning.")
    return trainable_names, initial_backbone_state


def summarize_backbone_updates(
    *,
    model: nn.Module,
    initial_backbone_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    per_parameter: dict[str, Any] = {}
    updated_count = 0
    for name, parameter in model.named_parameters():
        initial = initial_backbone_state.get(name)
        if initial is None:
            continue
        current = parameter.detach().cpu()
        delta = current - initial
        max_abs_delta = float(delta.abs().max().item())
        changed = bool(max_abs_delta > 0.0)
        if changed:
            updated_count += 1
        per_parameter[name] = {
            "shape": list(current.shape),
            "initial_l2": float(initial.norm().item()),
            "final_l2": float(current.norm().item()),
            "delta_l2": float(delta.norm().item()),
            "max_abs_delta": max_abs_delta,
            "changed": changed,
        }
    return {
        "tracked_backbone_parameters": len(per_parameter),
        "updated_backbone_parameters": updated_count,
        "all_tracked_parameters_changed": bool(per_parameter) and updated_count == len(per_parameter),
        "per_parameter": per_parameter,
    }


def render_recreation_report(
    *,
    experiment_dir: Path,
    config: dict[str, Any],
    split_data: dict[str, SplitImageData],
    metrics_by_alias: dict[str, dict[str, Any]],
) -> str:
    split_lines: list[str] = []
    for alias, payload in split_data.items():
        split_lines.append(
            f"- `{alias}` -> domain=`{payload.domain}` split=`{payload.split}` rows=`{len(payload.samples)}`"
        )
    metric_lines: list[str] = []
    for alias in sorted(metrics_by_alias):
        summary = metrics_by_alias[alias]
        metric_lines.append(
            f"- `{alias}` macro AUROC `{config['format_metric'](summary['macro']['auroc'])}`, "
            f"macro AP `{config['format_metric'](summary['macro']['average_precision'])}`"
        )
    return "\n".join(
        [
            "# Image-Only Partial Finetune Recreation Report",
            "",
            "## Scope",
            "",
            f"- Experiment directory: `{experiment_dir}`",
            f"- Manifest: `{config['manifest_csv']}`",
            f"- Backbone model: `{config['backbone_name']}`",
            f"- Adaptation method: `{config['adaptation_method']}`",
            f"- Split profile: `{config['split_profile']}`",
            "",
            "## Recreation Command",
            "",
            "```bash",
            format_bash_command(["python", *config["argv"]]),
            "```",
            "",
            "## Split Inputs",
            "",
            *split_lines,
            "",
            "## Final Metrics",
            "",
            *metric_lines,
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a target-only image-level partial-backbone finetune baseline."
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--split-profile",
        choices=("mimic_target", "chexpert_target"),
        default="mimic_target",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--fp16-on-cuda", action="store_true")
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trainable-blocks", type=int, default=DEFAULT_TRAINABLE_BLOCKS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0.")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive.")
    if args.weight_decay < 0:
        raise SystemExit("--weight-decay must be >= 0.")
    if args.patience < 0:
        raise SystemExit("--patience must be >= 0.")
    if args.max_rows_per_split is not None and args.max_rows_per_split <= 0:
        raise SystemExit("--max-rows-per-split must be positive when provided.")

    helpers = load_transfer_helpers()
    helpers.seed_everything(int(args.seed))
    device = helpers.resolve_device(args.device)
    manifest_csv = args.manifest_csv.resolve()
    data_root = args.data_root.resolve()
    print(
        f"[startup] manifest_csv={manifest_csv} data_root={data_root} "
        f"split_profile={args.split_profile} batch_size={args.batch_size} device={device}",
        flush=True,
    )

    label_names, manifest_by_key = helpers.load_manifest_records(manifest_csv)
    evaluation_plan = build_evaluation_plan(split_profile=args.split_profile)
    split_data: dict[str, SplitImageData] = {}
    for alias, domain, split in evaluation_plan.split_specs:
        manifest_records = manifest_by_key.get((domain, split))
        if manifest_records is None:
            raise SystemExit(f"Manifest does not contain records for domain={domain} split={split}.")
        split_data[alias] = build_split_image_data(
            alias=alias,
            domain=domain,
            split=split,
            manifest_records=manifest_records,
            data_root=data_root,
            max_rows=args.max_rows_per_split,
            num_labels=len(label_names),
        )
        print(
            f"[split-load] alias={alias} domain={domain} split={split} rows={len(split_data[alias].samples)}",
            flush=True,
        )

    print("[model-load] loading torchvision ViT-B/16 pretrained weights", flush=True)
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    in_features = int(model.heads.head.in_features)
    model.heads.head = nn.Linear(in_features, len(label_names))
    nn.init.xavier_uniform_(model.heads.head.weight)
    nn.init.zeros_(model.heads.head.bias)
    transform = weights.transforms()

    trainable_names, initial_backbone_state = configure_partial_finetune(
        model,
        trainable_blocks=int(args.trainable_blocks),
    )
    model = model.to(device)

    generated_slug = "__".join(
        [
            DEFAULT_OPERATION_LABEL,
            slugify(args.split_profile, fallback="target"),
            "torchvision-vit-b-16",
            f"last-{args.trainable_blocks}-block",
        ]
    )
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=args.experiments_root.resolve(),
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=bool(args.overwrite),
    )

    train_loader = build_dataloader(
        split_data[evaluation_plan.train_alias],
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=True,
    )
    eval_loaders = {
        alias: build_dataloader(
            payload,
            transform=transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            shuffle=False,
        )
        for alias, payload in split_data.items()
    }

    pos_weight = helpers.compute_pos_weight(split_data[evaluation_plan.train_alias].labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(scaler_device, enabled=bool(args.fp16_on_cuda and device.type == "cuda"))
    total_parameters, trainable_parameters = count_parameters(model)

    config = {
        "argv": list(sys.argv),
        "run_date_utc": helpers.utc_now_iso(),
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "manifest_csv": str(manifest_csv),
        "data_root": str(data_root),
        "split_profile": args.split_profile,
        "backbone_name": DEFAULT_BACKBONE_NAME,
        "adaptation_method": "partial_last_block_finetune",
        "trainable_blocks": int(args.trainable_blocks),
        "label_names": label_names,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "device_requested": str(args.device),
        "device_resolved": str(device),
        "fp16_on_cuda": bool(args.fp16_on_cuda),
        "max_rows_per_split": int(args.max_rows_per_split) if args.max_rows_per_split is not None else None,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_parameter_names": trainable_names,
        "split_inputs": {
            alias: {
                "domain": payload.domain,
                "split": payload.split,
                "num_rows": len(payload.samples),
            }
            for alias, payload in split_data.items()
        },
    }
    helpers.write_json(experiment_dir / "config.json", config)
    (experiment_dir / "train_log.jsonl").write_text("", encoding="utf-8")

    print(f"[info] experiment_dir={experiment_dir}", flush=True)
    print(
        f"[info] total_parameters={total_parameters} trainable_parameters={trainable_parameters} "
        f"trainable_blocks={args.trainable_blocks}",
        flush=True,
    )

    best_epoch = 0
    best_summary: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_loss = train_one_epoch(
            loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            fp16_on_cuda=bool(args.fp16_on_cuda),
            get_autocast_context=helpers.get_autocast_context,
        )
        selection_loss, selection_logits, selection_targets = evaluate_model(
            loader=eval_loaders[evaluation_plan.selection_alias],
            model=model,
            criterion=criterion,
            device=device,
            fp16_on_cuda=bool(args.fp16_on_cuda),
            get_autocast_context=helpers.get_autocast_context,
        )
        current_summary = helpers.summarize_split_metrics(
            split_alias=evaluation_plan.selection_alias,
            loss=selection_loss,
            targets=selection_targets,
            logits=selection_logits,
            label_names=label_names,
            tuned_thresholds=np.full((len(label_names),), 0.5, dtype=np.float32),
        )
        improved = best_summary is None or helpers.selection_tuple(current_summary) > helpers.selection_tuple(best_summary)
        if improved:
            best_epoch = epoch
            best_summary = current_summary
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        helpers.append_jsonl(
            experiment_dir / "train_log.jsonl",
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                f"{evaluation_plan.selection_alias}_loss": float(selection_loss),
                f"{evaluation_plan.selection_alias}_macro_auroc": current_summary["macro"]["auroc"],
                f"{evaluation_plan.selection_alias}_macro_average_precision": current_summary["macro"]["average_precision"],
                "improved": bool(improved),
                "elapsed_sec": float(time.time() - epoch_started),
            },
        )
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"{evaluation_plan.selection_alias}_loss={selection_loss:.6f} "
            f"{evaluation_plan.selection_alias}_macro_auroc={helpers.format_metric(current_summary['macro']['auroc'])} "
            f"improved={str(improved).lower()}",
            flush=True,
        )
        if epochs_without_improvement > args.patience:
            print(f"[early-stop] epoch={epoch} patience={args.patience}", flush=True)
            break

    if best_state is None or best_summary is None:
        raise SystemExit("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    raw_eval_results: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for alias, loader in eval_loaders.items():
        raw_eval_results[alias] = evaluate_model(
            loader=loader,
            model=model,
            criterion=criterion,
            device=device,
            fp16_on_cuda=bool(args.fp16_on_cuda),
            get_autocast_context=helpers.get_autocast_context,
        )

    selection_loss, selection_logits, selection_targets = raw_eval_results[evaluation_plan.selection_alias]
    selection_probs = helpers.sigmoid_np(selection_logits.astype(np.float64))
    tuned_thresholds, threshold_payload = helpers.tune_thresholds(
        selection_targets,
        selection_probs,
        label_names,
        selection_split=evaluation_plan.selection_alias,
    )

    metrics_by_alias: dict[str, dict[str, Any]] = {}
    for alias, (loss, logits, targets) in raw_eval_results.items():
        metrics = helpers.summarize_split_metrics(
            split_alias=alias,
            loss=loss,
            targets=targets,
            logits=logits,
            label_names=label_names,
            tuned_thresholds=tuned_thresholds,
        )
        metrics_by_alias[alias] = metrics
        helpers.write_json(experiment_dir / evaluation_plan.output_name_map[alias], metrics)

    helpers.write_json(experiment_dir / evaluation_plan.thresholds_filename, threshold_payload)
    update_summary = summarize_backbone_updates(model=model, initial_backbone_state=initial_backbone_state)
    helpers.write_json(experiment_dir / "backbone_param_update_summary.json", update_summary)

    torch.save(
        {
            "epoch": best_epoch,
            "state_dict": best_state,
            "label_names": label_names,
            "tuned_thresholds": tuned_thresholds.tolist(),
            "best_summary": metrics_by_alias[evaluation_plan.selection_alias],
            "backbone_name": DEFAULT_BACKBONE_NAME,
            "adaptation_method": "partial_last_block_finetune",
            "trainable_blocks": int(args.trainable_blocks),
        },
        experiment_dir / "best.ckpt",
    )

    experiment_meta = {
        "run_date_utc": helpers.utc_now_iso(),
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "operation_label": DEFAULT_OPERATION_LABEL,
        "manifest_csv": str(manifest_csv),
        "data_root": str(data_root),
        "split_profile": args.split_profile,
        "backbone_name": DEFAULT_BACKBONE_NAME,
        "adaptation_method": "partial_last_block_finetune",
        "device_resolved": str(device),
        "best_epoch": int(best_epoch),
        "num_labels": len(label_names),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_parameter_names": trainable_names,
        "macro_metrics": {alias: summary["macro"] for alias, summary in metrics_by_alias.items()},
        "thresholds_path": str(experiment_dir / evaluation_plan.thresholds_filename),
        "checkpoint_path": str(experiment_dir / "best.ckpt"),
        "backbone_param_update_summary_path": str(experiment_dir / "backbone_param_update_summary.json"),
    }
    helpers.write_json(experiment_dir / "experiment_meta.json", experiment_meta)

    recreation_report = render_recreation_report(
        experiment_dir=experiment_dir,
        config={**config, "format_metric": helpers.format_metric},
        split_data=split_data,
        metrics_by_alias=metrics_by_alias,
    )
    (experiment_dir / "recreation_report.md").write_text(recreation_report, encoding="utf-8")

    print(
        "[done] "
        f"best_epoch={best_epoch} "
        f"{evaluation_plan.selection_alias}_macro_auroc="
        f"{helpers.format_metric(metrics_by_alias[evaluation_plan.selection_alias]['macro']['auroc'])} "
        f"{evaluation_plan.primary_test_alias}_macro_auroc="
        f"{helpers.format_metric(metrics_by_alias[evaluation_plan.primary_test_alias]['macro']['auroc'])} "
        f"output_dir={experiment_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
