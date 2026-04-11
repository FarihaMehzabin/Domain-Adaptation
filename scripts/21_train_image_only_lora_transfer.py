#!/usr/bin/env python3
"""Train an image-only LoRA vision transformer for NIH-source domain transfer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_common_labels_pilot5h.csv")
DEFAULT_DATA_ROOT = Path("/workspace")
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_EPOCHS = 12
DEFAULT_LR = 2e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 3
DEFAULT_SEED = 1337
DEFAULT_DEVICE = "auto"
DEFAULT_OPERATION_LABEL = "domain_transfer_lora_training"
DEFAULT_EXPERIMENT_ID_WIDTH = 4
DEFAULT_MODEL_ID = "google/vit-base-patch16-224-in21k"
DEFAULT_POOLING = "cls"
DEFAULT_CLASSIFIER_DROPOUT = 0.1
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_LORA_TARGET_MODULES = ("query", "value")
DEFAULT_LOG_EVERY_STEPS = 50
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


class ManifestImageDataset(Dataset):
    def __init__(self, samples: list[ImageSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        try:
            with Image.open(sample.resolved_path) as image:
                rgb = image.convert("RGB").copy()
            return {
                "sample": sample,
                "image": rgb,
                "targets": np.asarray(sample.labels, dtype=np.float32),
                "error": None,
            }
        except Exception as exc:
            return {
                "sample": sample,
                "image": None,
                "targets": np.asarray(sample.labels, dtype=np.float32),
                "error": str(exc),
            }


class ImageBatchCollator:
    def __init__(self, image_processor: Any) -> None:
        self.image_processor = image_processor

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        valid_samples: list[ImageSample] = []
        images: list[Image.Image] = []
        labels: list[np.ndarray] = []
        failures: list[dict[str, str]] = []

        for item in items:
            sample = item["sample"]
            image = item.get("image")
            if image is None:
                failures.append(
                    {
                        "row_id": sample.row_id,
                        "image_path": sample.image_path,
                        "error": str(item.get("error") or "unknown_error"),
                    }
                )
                continue
            valid_samples.append(sample)
            images.append(image)
            labels.append(np.asarray(item["targets"], dtype=np.float32))

        pixel_values = None
        labels_tensor = None
        if images:
            batch = self.image_processor(images=images, return_tensors="pt")
            pixel_values = batch["pixel_values"]
            labels_tensor = torch.from_numpy(np.stack(labels, axis=0).astype(np.float32))

        return {
            "samples": valid_samples,
            "pixel_values": pixel_values,
            "targets": labels_tensor,
            "failures": failures,
        }


def load_transfer_helpers() -> Any:
    if not HELPER_SCRIPT.exists():
        raise SystemExit(f"Helper script not found: {HELPER_SCRIPT}")
    module_name = "domain_transfer_head_training_helpers"
    spec = importlib.util.spec_from_file_location(module_name, HELPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load helper module from {HELPER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_lora_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'peft'. Install LoRA requirements with:\n"
            "  python -m pip install -r /workspace/scripts/requirements_vision_lora.txt"
        ) from exc
    try:
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'transformers'. Install LoRA requirements with:\n"
            "  python -m pip install -r /workspace/scripts/requirements_vision_lora.txt"
        ) from exc
    return AutoImageProcessor, AutoModel, LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict


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


def extract_experiment_number(name: str) -> int | None:
    if not name.startswith("exp"):
        return None
    prefix = name.split("__", 1)[0]
    digits = prefix.removeprefix("exp")
    if not digits.isdigit():
        return None
    return int(digits)


def next_experiment_number(experiments_root: Path) -> int:
    if not experiments_root.exists():
        return 1
    max_number = 0
    for child in experiments_root.iterdir():
        if not child.is_dir():
            continue
        number = extract_experiment_number(child.name)
        if number is None:
            continue
        max_number = max(max_number, number)
    return max_number + 1


def resolve_experiment_identity(
    *,
    experiments_root: Path,
    requested_name: str | None,
    generated_slug: str,
    overwrite: bool,
    id_width: int = DEFAULT_EXPERIMENT_ID_WIDTH,
) -> tuple[int, str, str, Path]:
    experiments_root.mkdir(parents=True, exist_ok=True)
    requested = (requested_name or "").strip() or None
    base_name = ensure_operation_prefix(requested or generated_slug)
    explicit_number = extract_experiment_number(base_name)
    if explicit_number is not None:
        experiment_number = explicit_number
        experiment_name = base_name
    else:
        experiment_number = next_experiment_number(experiments_root)
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


def format_bash_command(argv: list[str]) -> str:
    return " \\\n  ".join(shlex.quote(part) for part in argv)


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
    image_processor: Any,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ManifestImageDataset(split_data.samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=ImageBatchCollator(image_processor=image_processor),
    )


class ImageOnlyLoraClassifier(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        hidden_size: int,
        num_labels: int,
        pooling: str,
        classifier_dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def pool_outputs(self, outputs: Any) -> torch.Tensor:
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        pooler_output = getattr(outputs, "pooler_output", None)
        if self.pooling == "cls":
            if isinstance(last_hidden_state, torch.Tensor) and last_hidden_state.ndim == 3:
                return last_hidden_state[:, 0, :]
            if isinstance(pooler_output, torch.Tensor):
                return pooler_output
            raise RuntimeError("Could not extract CLS-pooled features from the backbone outputs.")
        if self.pooling == "avg":
            if isinstance(last_hidden_state, torch.Tensor) and last_hidden_state.ndim == 3:
                return last_hidden_state.mean(dim=1)
            if isinstance(pooler_output, torch.Tensor):
                return pooler_output
            raise RuntimeError("Could not extract average-pooled features from the backbone outputs.")
        raise RuntimeError(f"Unsupported pooling mode: {self.pooling}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = self.pool_outputs(outputs)
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    return total, trainable


def train_one_epoch(
    *,
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    fp16_on_cuda: bool,
    get_autocast_context: Any,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_batches = len(loader)
    for step_idx, batch in enumerate(loader, start=1):
        pixel_values = batch["pixel_values"]
        targets = batch["targets"]
        if pixel_values is None or targets is None:
            continue
        pixel_values = pixel_values.to(device, non_blocking=(device.type == "cuda"))
        targets = targets.to(device, non_blocking=(device.type == "cuda"))
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
        if step_idx == 1 or step_idx % DEFAULT_LOG_EVERY_STEPS == 0 or step_idx == total_batches:
            running_loss = total_loss / max(total_examples, 1)
            print(
                f"[train] step={step_idx}/{total_batches} examples={total_examples} "
                f"running_loss={running_loss:.6f}",
                flush=True,
            )
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
    total_batches = len(loader)
    for step_idx, batch in enumerate(loader, start=1):
        pixel_values = batch["pixel_values"]
        targets = batch["targets"]
        if pixel_values is None or targets is None:
            continue
        pixel_values = pixel_values.to(device, non_blocking=(device.type == "cuda"))
        targets = targets.to(device, non_blocking=(device.type == "cuda"))
        with get_autocast_context(device, fp16_on_cuda):
            logits = model(pixel_values)
            loss = criterion(logits, targets)
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        logits_chunks.append(logits.detach().cpu())
        targets_chunks.append(targets.detach().cpu())
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
            "# Image-Only LoRA Transfer Recreation Report",
            "",
            "## Scope",
            "",
            f"- Experiment directory: `{experiment_dir}`",
            f"- Manifest: `{config['manifest_csv']}`",
            f"- Backbone model: `{config['model_id']}`",
            f"- Pooling: `{config['pooling']}`",
            f"- LoRA target modules: `{config['lora_target_modules']}`",
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
            "## Notes",
            "",
            "- Training uses only `d0_train` images.",
            "- Early stopping is driven by `d0_val` macro AUROC.",
            "- Validation-tuned thresholds are reused unchanged for D0 test and transfer evaluations.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an image-only LoRA vision transformer on NIH and evaluate direct transfer "
            "to CheXpert and MIMIC using the common 7-label pilot manifest."
        )
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--pooling", choices=("cls", "avg"), default=DEFAULT_POOLING)
    parser.add_argument("--classifier-dropout", type=float, default=DEFAULT_CLASSIFIER_DROPOUT)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=list(DEFAULT_LORA_TARGET_MODULES),
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
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
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
    if args.lora_r <= 0:
        raise SystemExit("--lora-r must be positive.")
    if args.lora_alpha <= 0:
        raise SystemExit("--lora-alpha must be positive.")
    if not (0.0 <= float(args.lora_dropout) < 1.0):
        raise SystemExit("--lora-dropout must be in [0.0, 1.0).")
    if not (0.0 <= float(args.classifier_dropout) < 1.0):
        raise SystemExit("--classifier-dropout must be in [0.0, 1.0).")
    if not args.lora_target_modules:
        raise SystemExit("--lora-target-modules must contain at least one module name.")

    helpers = load_transfer_helpers()
    helpers.seed_everything(int(args.seed))
    device = helpers.resolve_device(args.device)
    manifest_csv = args.manifest_csv.resolve()
    data_root = args.data_root.resolve()
    print(
        f"[startup] manifest_csv={manifest_csv} data_root={data_root} model_id={args.model_id} "
        f"batch_size={args.batch_size} device={device}",
        flush=True,
    )

    print("[manifest-load] loading common-label transfer manifest", flush=True)
    label_names, manifest_by_key = helpers.load_manifest_records(manifest_csv)
    eval_specs = [
        ("d0_train", "d0_nih", "train"),
        ("d0_val", "d0_nih", "val"),
        ("d0_test", "d0_nih", "test"),
        ("d1_transfer", "d1_chexpert", "val"),
        ("d2_transfer", "d2_mimic", "test"),
    ]

    split_data: dict[str, SplitImageData] = {}
    for alias, domain, split in eval_specs:
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

    AutoImageProcessor, AutoModel, LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict = (
        import_lora_dependencies()
    )
    processor_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    if args.revision:
        processor_kwargs["revision"] = args.revision
        model_kwargs["revision"] = args.revision
    if args.cache_dir is not None:
        cache_dir = args.cache_dir.resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        processor_kwargs["cache_dir"] = str(cache_dir)
        model_kwargs["cache_dir"] = str(cache_dir)
    else:
        cache_dir = None

    print("[model-load] loading image processor", flush=True)
    image_processor = AutoImageProcessor.from_pretrained(args.model_id, use_fast=True, **processor_kwargs)
    print(f"[model-load] image processor ready class={type(image_processor).__name__}", flush=True)
    print("[model-load] loading backbone from Hugging Face", flush=True)
    backbone = AutoModel.from_pretrained(args.model_id, **model_kwargs)
    print("[model-load] backbone ready", flush=True)
    if bool(args.gradient_checkpointing) and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
        print("[model-load] gradient checkpointing enabled", flush=True)

    lora_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        target_modules=list(args.lora_target_modules),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    print(
        f"[lora] attaching adapters target_modules={list(args.lora_target_modules)} "
        f"r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}",
        flush=True,
    )
    backbone = get_peft_model(backbone, lora_config)

    hidden_size = getattr(backbone.config, "hidden_size", None)
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise SystemExit(f"Unable to determine hidden_size from backbone config for {args.model_id}.")
    model = ImageOnlyLoraClassifier(
        backbone=backbone,
        hidden_size=hidden_size,
        num_labels=len(label_names),
        pooling=args.pooling,
        classifier_dropout=float(args.classifier_dropout),
    ).to(device)
    print("[model-load] LoRA classifier model moved to target device", flush=True)

    generated_slug = "__".join(
        [
            DEFAULT_OPERATION_LABEL,
            slugify(args.model_id.split("/")[-1], fallback="vit"),
            slugify(manifest_csv.stem, fallback="manifest"),
            f"pool-{args.pooling}",
            f"lora-r{args.lora_r}",
            f"a{args.lora_alpha}",
        ]
    )
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=args.experiments_root.resolve(),
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=bool(args.overwrite),
    )

    train_loader = build_dataloader(
        split_data["d0_train"],
        image_processor=image_processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=True,
    )
    print(f"[dataloader] d0_train batches={len(train_loader)}", flush=True)
    eval_loaders = {
        alias: build_dataloader(
            payload,
            image_processor=image_processor,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            shuffle=False,
        )
        for alias, payload in split_data.items()
        if alias != "d0_train"
    }
    for alias, loader in eval_loaders.items():
        print(f"[dataloader] {alias} batches={len(loader)}", flush=True)

    pos_weight = helpers.compute_pos_weight(split_data["d0_train"].labels).to(device)
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
        "model_id": args.model_id,
        "revision": args.revision,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "pooling": args.pooling,
        "classifier_dropout": float(args.classifier_dropout),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_target_modules": list(args.lora_target_modules),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "label_names": label_names,
        "hidden_size": int(hidden_size),
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

    print(f"[info] experiment_dir={experiment_dir}")
    print(f"[info] manifest_csv={manifest_csv}")
    print(f"[info] model_id={args.model_id} device={device} pooling={args.pooling}")
    print(
        f"[info] total_parameters={total_parameters} trainable_parameters={trainable_parameters} "
        f"lora_target_modules={list(args.lora_target_modules)}"
    )

    best_epoch = 0
    best_summary: dict[str, Any] | None = None
    best_head_state: dict[str, torch.Tensor] | None = None
    best_lora_state: dict[str, torch.Tensor] | None = None
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
        d0_val_loss, d0_val_logits, d0_val_targets = evaluate_model(
            loader=eval_loaders["d0_val"],
            model=model,
            criterion=criterion,
            device=device,
            fp16_on_cuda=bool(args.fp16_on_cuda),
            get_autocast_context=helpers.get_autocast_context,
        )
        current_summary = helpers.summarize_split_metrics(
            split_alias="d0_val",
            loss=d0_val_loss,
            targets=d0_val_targets,
            logits=d0_val_logits,
            label_names=label_names,
            tuned_thresholds=np.full((len(label_names),), 0.5, dtype=np.float32),
        )
        improved = best_summary is None or helpers.selection_tuple(current_summary) > helpers.selection_tuple(best_summary)
        if improved:
            best_epoch = epoch
            best_summary = current_summary
            best_head_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if not key.startswith("backbone.")
            }
            best_lora_state = {
                key: value.detach().cpu().clone()
                for key, value in model.backbone.state_dict().items()
                if "lora_" in key
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        helpers.append_jsonl(
            experiment_dir / "train_log.jsonl",
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "d0_val_loss": float(d0_val_loss),
                "d0_val_macro_auroc": current_summary["macro"]["auroc"],
                "d0_val_macro_average_precision": current_summary["macro"]["average_precision"],
                "improved": bool(improved),
                "elapsed_sec": float(time.time() - epoch_started),
            },
        )
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"d0_val_loss={d0_val_loss:.6f} d0_val_macro_auroc={helpers.format_metric(current_summary['macro']['auroc'])} "
            f"improved={str(improved).lower()}",
            flush=True,
        )
        if epochs_without_improvement > args.patience:
            print(f"[early-stop] epoch={epoch} patience={args.patience}", flush=True)
            break

    if best_head_state is None or best_lora_state is None or best_summary is None:
        raise SystemExit("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_head_state, strict=False)
    set_peft_model_state_dict(model.backbone, best_lora_state)

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

    d0_val_loss, d0_val_logits, d0_val_targets = raw_eval_results["d0_val"]
    d0_val_probs = helpers.sigmoid_np(d0_val_logits.astype(np.float64))
    tuned_thresholds, threshold_payload = helpers.tune_thresholds(d0_val_targets, d0_val_probs, label_names)

    metrics_by_alias: dict[str, dict[str, Any]] = {}
    output_name_map = {
        "d0_val": "d0_val_metrics.json",
        "d0_test": "d0_test_metrics.json",
        "d1_transfer": "d1_transfer_metrics.json",
        "d2_transfer": "d2_transfer_metrics.json",
    }
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
        helpers.write_json(experiment_dir / output_name_map[alias], metrics)

    helpers.write_json(experiment_dir / "d0_val_f1_thresholds.json", threshold_payload)
    adapter_dir = experiment_dir / "best_adapter"
    processor_dir = experiment_dir / "image_processor"
    model.backbone.save_pretrained(adapter_dir)
    image_processor.save_pretrained(processor_dir)
    torch.save(
        {
            "epoch": best_epoch,
            "head_state_dict": best_head_state,
            "label_names": label_names,
            "tuned_thresholds": tuned_thresholds.tolist(),
            "pooling": args.pooling,
            "classifier_dropout": float(args.classifier_dropout),
            "model_id": args.model_id,
            "adapter_dir": str(adapter_dir),
            "image_processor_dir": str(processor_dir),
            "lora_config": {
                "r": int(args.lora_r),
                "alpha": int(args.lora_alpha),
                "dropout": float(args.lora_dropout),
                "target_modules": list(args.lora_target_modules),
            },
            "best_summary": metrics_by_alias["d0_val"],
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
        "model_id": args.model_id,
        "device_resolved": str(device),
        "best_epoch": int(best_epoch),
        "num_labels": len(label_names),
        "pooling": args.pooling,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "lora_target_modules": list(args.lora_target_modules),
        "macro_metrics": {alias: summary["macro"] for alias, summary in metrics_by_alias.items()},
        "thresholds_path": str(experiment_dir / "d0_val_f1_thresholds.json"),
        "checkpoint_path": str(experiment_dir / "best.ckpt"),
        "adapter_dir": str(adapter_dir),
        "image_processor_dir": str(processor_dir),
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
        f"d0_val_macro_auroc={helpers.format_metric(metrics_by_alias['d0_val']['macro']['auroc'])} "
        f"d0_test_macro_auroc={helpers.format_metric(metrics_by_alias['d0_test']['macro']['auroc'])} "
        f"output_dir={experiment_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
