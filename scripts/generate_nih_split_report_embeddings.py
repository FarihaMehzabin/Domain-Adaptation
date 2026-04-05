#!/usr/bin/env python3
"""Generate split-aware NIH CXR14 report embeddings for configurable text encoders."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'numpy'.") from exc

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'torch'.") from exc


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_nih_cxr14_all14.csv")
DEFAULT_REPORTS_ROOT = Path("/workspace/reports")
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_SPLITS = ["train", "val", "test"]
DEFAULT_MODEL_ID = "microsoft/BiomedVLP-CXR-BERT-specialized"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICE = "auto"
DEFAULT_MAX_LENGTH = 512
DEFAULT_OPERATION_LABEL = "report_embedding_generation"
DEFAULT_EXPERIMENT_PREFIX = f"{DEFAULT_OPERATION_LABEL}__nih_cxr14_report_embeddings"
DEFAULT_EXPERIMENT_ID_WIDTH = 4

SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "validation": "val",
    "test": "test",
    "tst": "test",
}


@dataclass
class TextEncoderBundle:
    model: Any
    tokenizer: Any
    backend: str
    model_id: str
    tokenizer_id: str
    revision: str | None
    build_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportSample:
    split: str
    report_id: str
    report_path: Path
    image_path: str


class NIHReportDataset(Dataset):
    def __init__(
        self,
        samples: list[ReportSample],
        *,
        text_prefix: str,
        text_suffix: str,
        normalize_whitespace: bool,
    ) -> None:
        self.samples = samples
        self.text_prefix = text_prefix
        self.text_suffix = text_suffix
        self.normalize_whitespace = normalize_whitespace

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        try:
            text = sample.report_path.read_text(encoding="utf-8")
            text = text.strip()
            if self.normalize_whitespace:
                text = " ".join(text.split())
            text = f"{self.text_prefix}{text}{self.text_suffix}"
            if not text.strip():
                raise ValueError("report text is empty after preprocessing")
            return {
                "split": sample.split,
                "report_id": sample.report_id,
                "report_path": str(sample.report_path),
                "image_path": sample.image_path,
                "text": text,
                "error": None,
            }
        except Exception as exc:
            return {
                "split": sample.split,
                "report_id": sample.report_id,
                "report_path": str(sample.report_path),
                "image_path": sample.image_path,
                "text": None,
                "error": str(exc),
            }


@dataclass
class ReportCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_batch(items, tokenizer=self.tokenizer, max_length=self.max_length)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, torch.Tensor):
        return to_serializable(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2, sort_keys=True), encoding="utf-8")


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def slugify(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or fallback


def ensure_operation_prefix(name: str, operation_label: str = DEFAULT_OPERATION_LABEL) -> str:
    normalized_label = slugify(operation_label, fallback="operation")
    explicit_match = re.match(r"^(exp\d+__)(.+)$", name)
    if explicit_match is not None:
        numbered_prefix, remainder = explicit_match.groups()
        if re.match(rf"^{re.escape(normalized_label)}(?:__|$)", remainder):
            return name
        return f"{numbered_prefix}{normalized_label}__{remainder}"

    if re.match(rf"^{re.escape(normalized_label)}(?:__|$)", name):
        return name
    return f"{normalized_label}__{name}"


def extract_experiment_number(name: str) -> int | None:
    match = re.match(r"^exp(\d+)(?:__|$)", name)
    if match is None:
        return None
    return int(match.group(1))


def next_experiment_number(experiments_root: Path) -> int:
    if not experiments_root.exists():
        return 1

    max_number = 0
    for child in experiments_root.iterdir():
        if not child.is_dir():
            continue
        experiment_number = extract_experiment_number(child.name)
        if experiment_number is None:
            continue
        max_number = max(max_number, experiment_number)
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

    if requested_name:
        requested_name = requested_name.strip()
        if not requested_name:
            raise SystemExit("--experiment-name cannot be empty.")
    base_name = ensure_operation_prefix(requested_name or generated_slug)

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

    return experiment_number, experiment_id, experiment_name, experiment_dir


def canonicalize_split(raw_split: str) -> str:
    normalized = SPLIT_ALIASES.get(raw_split.strip().lower())
    if normalized is None:
        valid = ", ".join(sorted(SPLIT_ALIASES))
        raise SystemExit(f"Unsupported split '{raw_split}'. Valid values: {valid}")
    return normalized


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but CUDA is not available.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit("Requested --device mps, but MPS is not available.")
    return device


def resolve_max_length(requested: int | None, tokenizer: Any) -> int:
    if requested is not None:
        if requested <= 0:
            raise SystemExit("--max-length must be a positive integer.")
        return int(requested)

    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int) and 0 < tokenizer_max < 100_000:
        return int(tokenizer_max)
    return DEFAULT_MAX_LENGTH


def load_manifest_split_reports(
    manifest_csv: Path,
    reports_root: Path,
    splits: list[str],
    *,
    max_reports: int | None,
) -> dict[str, list[ReportSample]]:
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV not found: {manifest_csv}")

    split_set = set(splits)
    resolved_samples: dict[str, list[ReportSample]] = {split: [] for split in splits}
    missing_examples: dict[str, list[str]] = {split: [] for split in splits}

    manifest_text = manifest_csv.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(manifest_text))
    required_columns = {"image_path", "split"}
    if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
        raise SystemExit(f"Manifest CSV must contain columns: {sorted(required_columns)}")

    for row in reader:
        if row.get("dataset") and row["dataset"] != "nih_cxr14":
            continue
        split = canonicalize_split(row.get("split") or "")
        if split not in split_set:
            continue
        if max_reports is not None and len(resolved_samples[split]) >= max_reports:
            continue

        image_path = (row.get("image_path") or "").strip()
        if not image_path:
            continue

        report_id = Path(image_path).stem
        report_path = reports_root / split / f"{report_id}.txt"
        if not report_path.exists():
            if len(missing_examples[split]) < 5:
                missing_examples[split].append(str(report_path))
            continue

        resolved_samples[split].append(
            ReportSample(
                split=split,
                report_id=report_id,
                report_path=report_path,
                image_path=image_path,
            )
        )

    for split in splits:
        if not resolved_samples[split]:
            sample = "\n".join(missing_examples[split]) if missing_examples[split] else "No matching rows found in manifest."
            raise SystemExit(
                f"No usable reports found for split '{split}'. Check --reports-root.\n"
                f"Examples:\n{sample}"
            )
        if missing_examples[split]:
            sample = "\n".join(missing_examples[split])
            raise SystemExit(
                f"Found split '{split}' rows in {manifest_csv}, but some reports could not be resolved.\n"
                f"Pass the correct --reports-root.\nExamples:\n{sample}"
            )

    return resolved_samples


def collate_batch(items: list[dict[str, Any]], *, tokenizer: Any, max_length: int) -> dict[str, Any]:
    report_ids: list[str] = []
    report_paths: list[str] = []
    image_paths: list[str] = []
    texts: list[str] = []
    failures: list[dict[str, str]] = []

    for item in items:
        if item.get("text") is None:
            failures.append(
                {
                    "report_id": str(item["report_id"]),
                    "report_path": str(item["report_path"]),
                    "image_path": str(item["image_path"]),
                    "error": str(item.get("error") or "unknown_error"),
                }
            )
            continue

        report_ids.append(str(item["report_id"]))
        report_paths.append(str(item["report_path"]))
        image_paths.append(str(item["image_path"]))
        texts.append(str(item["text"]))

    encoded = None
    if texts:
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    return {
        "report_ids": report_ids,
        "report_paths": report_paths,
        "image_paths": image_paths,
        "texts": texts,
        "encoded": encoded,
        "failures": failures,
    }


def build_dataloader(
    samples: list[ReportSample],
    *,
    tokenizer: Any,
    batch_size: int,
    num_workers: int,
    max_length: int,
    text_prefix: str,
    text_suffix: str,
    normalize_whitespace: bool,
    device: torch.device,
) -> DataLoader:
    dataset = NIHReportDataset(
        samples,
        text_prefix=text_prefix,
        text_suffix=text_suffix,
        normalize_whitespace=normalize_whitespace,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=ReportCollator(tokenizer=tokenizer, max_length=max_length),
    )


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)

    mask = attention_mask.to(last_hidden_state.device)
    mask = mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return last_hidden_state[:, -1, :]

    lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
    batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_indices, lengths, :]


def get_autocast_context(device: torch.device, fp16_on_cuda: bool) -> Any:
    if bool(fp16_on_cuda and device.type == "cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def extract_embeddings(
    *,
    model: Any,
    model_inputs: dict[str, torch.Tensor],
    pooling: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if pooling == "auto":
        if hasattr(model, "get_projected_text_embeddings") and callable(model.get_projected_text_embeddings):
            embeddings = model.get_projected_text_embeddings(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                normalize_embeddings=False,
            )
            return embeddings, {
                "feature_source": "model_method",
                "pooling_applied": "get_projected_text_embeddings",
            }

        if hasattr(model, "get_text_features") and callable(model.get_text_features):
            embeddings = model.get_text_features(**model_inputs)
            return embeddings, {
                "feature_source": "model_method",
                "pooling_applied": "get_text_features",
            }

    outputs = model(**model_inputs, return_dict=True)

    if pooling == "auto":
        if getattr(outputs, "text_embeds", None) is not None:
            return outputs.text_embeds, {
                "feature_source": "model_output",
                "pooling_applied": "text_embeds",
            }
        if getattr(outputs, "sentence_embedding", None) is not None:
            return outputs.sentence_embedding, {
                "feature_source": "model_output",
                "pooling_applied": "sentence_embedding",
            }
        if getattr(outputs, "cls_projected_embedding", None) is not None:
            return outputs.cls_projected_embedding, {
                "feature_source": "model_output",
                "pooling_applied": "cls_projected_embedding",
            }
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output, {
                "feature_source": "model_output",
                "pooling_applied": "pooler_output",
            }
        if getattr(outputs, "last_hidden_state", None) is not None:
            attention_mask = model_inputs.get("attention_mask")
            return masked_mean_pool(outputs.last_hidden_state, attention_mask), {
                "feature_source": "model_output",
                "pooling_applied": "masked_mean_pool",
                "raw_feature_shape": list(outputs.last_hidden_state.shape),
            }
        raise RuntimeError("Unable to infer text embeddings for --pooling auto.")

    if pooling == "projected_cls":
        if hasattr(model, "get_projected_text_embeddings") and callable(model.get_projected_text_embeddings):
            embeddings = model.get_projected_text_embeddings(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                normalize_embeddings=False,
            )
            return embeddings, {
                "feature_source": "model_method",
                "pooling_applied": "get_projected_text_embeddings",
            }
        projected = getattr(outputs, "cls_projected_embedding", None)
        if projected is not None:
            return projected, {
                "feature_source": "model_output",
                "pooling_applied": "cls_projected_embedding",
            }
        raise RuntimeError(
            "Requested --pooling projected_cls, but the model does not expose projected CLS embeddings."
        )

    if pooling == "text_features":
        if hasattr(model, "get_text_features") and callable(model.get_text_features):
            return model.get_text_features(**model_inputs), {
                "feature_source": "model_method",
                "pooling_applied": "get_text_features",
            }
        text_embeds = getattr(outputs, "text_embeds", None)
        if text_embeds is not None:
            return text_embeds, {
                "feature_source": "model_output",
                "pooling_applied": "text_embeds",
            }
        raise RuntimeError("Requested --pooling text_features, but the model has no projected text feature API.")

    if getattr(outputs, "last_hidden_state", None) is None:
        raise RuntimeError(
            f"Requested --pooling {pooling}, but the model output has no last_hidden_state tensor."
        )

    last_hidden_state = outputs.last_hidden_state
    attention_mask = model_inputs.get("attention_mask")

    if pooling == "cls":
        return last_hidden_state[:, 0, :], {
            "feature_source": "model_output",
            "pooling_applied": "cls_token",
            "raw_feature_shape": list(last_hidden_state.shape),
        }
    if pooling == "mean":
        return masked_mean_pool(last_hidden_state, attention_mask), {
            "feature_source": "model_output",
            "pooling_applied": "masked_mean_pool",
            "raw_feature_shape": list(last_hidden_state.shape),
        }
    if pooling == "last_token":
        return last_token_pool(last_hidden_state, attention_mask), {
            "feature_source": "model_output",
            "pooling_applied": "last_token",
            "raw_feature_shape": list(last_hidden_state.shape),
        }
    if pooling == "pooler":
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            raise RuntimeError("Requested --pooling pooler, but the model output has no pooler_output tensor.")
        return pooled, {
            "feature_source": "model_output",
            "pooling_applied": "pooler_output",
        }

    raise RuntimeError(f"Unsupported pooling mode: {pooling}")


def save_manifest_csv(path: Path, samples: list[ReportSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "report_id", "report_path", "image_path"])
        for idx, sample in enumerate(samples):
            writer.writerow([idx, sample.report_id, str(sample.report_path), sample.image_path])


def save_failed_jsonl(path: Path, failures: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row) + "\n")


def save_split_outputs(
    output_dir: Path,
    *,
    embeddings: np.ndarray,
    samples: list[ReportSample],
    failed_reports: list[dict[str, str]],
    run_meta: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings.astype(np.float32))
    save_manifest_csv(output_dir / "report_manifest.csv", samples)
    (output_dir / "report_ids.json").write_text(
        json.dumps([sample.report_id for sample in samples], indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report_paths.txt").write_text(
        "\n".join(str(sample.report_path) for sample in samples) + "\n",
        encoding="utf-8",
    )
    if failed_reports:
        save_failed_jsonl(output_dir / "failed_reports.jsonl", failed_reports)
    write_json(output_dir / "run_meta.json", run_meta)


def build_text_encoder(
    *,
    model_id: str,
    tokenizer_id: str | None,
    revision: str | None,
    trust_remote_code: bool,
    cache_dir: Path | None,
) -> TextEncoderBundle:
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'transformers'. Install with: python -m pip install transformers safetensors"
        ) from exc

    cache_dir_str = str(cache_dir.resolve()) if cache_dir else None
    resolved_tokenizer_id = tokenizer_id or model_id

    config = AutoConfig.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir_str,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_tokenizer_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir_str,
    )
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir_str,
    )

    build_meta = {
        "config_class": type(config).__name__,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "model_type": getattr(config, "model_type", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "projection_size": getattr(config, "projection_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "trust_remote_code": trust_remote_code,
    }
    return TextEncoderBundle(
        model=model,
        tokenizer=tokenizer,
        backend="huggingface",
        model_id=model_id,
        tokenizer_id=resolved_tokenizer_id,
        revision=revision,
        build_meta=build_meta,
    )


def build_experiment_slug(
    *,
    experiment_prefix: str,
    manifest_csv: Path,
    splits: list[str],
    model_id: str,
    pooling: str,
    normalization: str,
    max_length: int,
    fp16_on_cuda: bool,
    revision: str | None,
    max_reports_per_split: int | None,
) -> str:
    parts = [
        slugify(experiment_prefix, fallback="experiment"),
        f"manifest-{slugify(manifest_csv.stem, fallback='manifest')}",
        f"splits-{slugify('-'.join(splits), fallback='splits')}",
        f"model-{slugify(model_id, fallback='model')}",
        f"pool-{slugify(pooling, fallback='pool')}",
        f"norm-{slugify(normalization, fallback='norm')}",
        f"maxlen-{max_length}",
        f"precision-{'amp-fp16' if fp16_on_cuda else 'fp32'}",
    ]
    if revision:
        parts.append(f"rev-{slugify(revision, fallback='revision')}")
    if max_reports_per_split is not None:
        parts.append(f"maxper-{max_reports_per_split}")
    parts.append(f"ts-{utc_now_compact()}")
    return "__".join(parts)


def embed_split(
    split: str,
    samples: list[ReportSample],
    encoder_bundle: TextEncoderBundle,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_length: int,
    pooling: str,
    normalization: str,
    fp16_on_cuda: bool,
    text_prefix: str,
    text_suffix: str,
    normalize_whitespace: bool,
) -> tuple[np.ndarray, list[ReportSample], list[dict[str, str]], dict[str, Any]]:
    dataloader = build_dataloader(
        samples,
        tokenizer=encoder_bundle.tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        text_prefix=text_prefix,
        text_suffix=text_suffix,
        normalize_whitespace=normalize_whitespace,
        device=device,
    )

    embeddings_chunks: list[np.ndarray] = []
    embedded_samples: list[ReportSample] = []
    failed_reports: list[dict[str, str]] = []
    feature_summary: dict[str, Any] | None = None

    with torch.inference_mode():
        for batch in dataloader:
            failed_reports.extend(batch["failures"])
            encoded = batch["encoded"]
            if encoded is None:
                continue

            model_inputs = move_batch_to_device(encoded, device)
            with get_autocast_context(device, fp16_on_cuda):
                batch_embeddings, batch_feature_summary = extract_embeddings(
                    model=encoder_bundle.model,
                    model_inputs=model_inputs,
                    pooling=pooling,
                )

            batch_embeddings = batch_embeddings.detach().to(torch.float32)
            if batch_embeddings.ndim != 2:
                raise RuntimeError(
                    f"Expected 2D embeddings for split '{split}', found shape {tuple(batch_embeddings.shape)}."
                )

            if normalization == "l2":
                batch_embeddings = F.normalize(batch_embeddings, dim=1)

            embeddings_chunks.append(batch_embeddings.cpu().numpy())
            embedded_samples.extend(
                [
                    ReportSample(
                        split=split,
                        report_id=report_id,
                        report_path=Path(report_path),
                        image_path=image_path,
                    )
                    for report_id, report_path, image_path in zip(
                        batch["report_ids"],
                        batch["report_paths"],
                        batch["image_paths"],
                        strict=True,
                    )
                ]
            )

            if feature_summary is None:
                feature_summary = {
                    "feature_source": batch_feature_summary.get("feature_source"),
                    "pooling_requested": pooling,
                    "pooling_applied": batch_feature_summary.get("pooling_applied"),
                    "raw_feature_shape": batch_feature_summary.get("raw_feature_shape"),
                    "normalization": normalization,
                    "tokenizer_model_max_length": getattr(encoder_bundle.tokenizer, "model_max_length", None),
                    "effective_max_length": max_length,
                }

    if not embeddings_chunks:
        raise RuntimeError(f"No report embeddings were produced for split '{split}'.")

    embeddings = np.concatenate(embeddings_chunks, axis=0)
    if embeddings.shape[0] != len(embedded_samples):
        raise RuntimeError(
            f"Split '{split}' produced {embeddings.shape[0]} embedding rows but {len(embedded_samples)} report ids."
        )
    return embeddings, embedded_samples, failed_reports, feature_summary or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate configurable report embeddings for NIH CXR14 train/val/test splits."
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--experiment-prefix", type=str, default=DEFAULT_EXPERIMENT_PREFIX)
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help=(
            "Optional experiment directory suffix or full numbered name. If omitted, a detailed "
            "numbered name is generated automatically."
        ),
    )
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--model-id", "--encoder-id", dest="model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-id", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument(
        "--pooling",
        choices=["auto", "projected_cls", "text_features", "cls", "mean", "last_token", "pooler"],
        default="auto",
    )
    parser.add_argument("--normalization", choices=["l2", "none"], default="l2")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-reports-per-split", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default=DEFAULT_DEVICE)
    parser.add_argument("--fp16-on-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--text-prefix", type=str, default="")
    parser.add_argument("--text-suffix", type=str, default="")
    parser.add_argument("--normalize-whitespace", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_csv = args.manifest_csv.resolve()
    reports_root = args.reports_root.resolve()
    experiments_root = args.experiments_root.resolve()
    splits = dedupe_preserve_order([canonicalize_split(split) for split in args.splits])
    device = resolve_device(args.device)

    if not manifest_csv.exists():
        raise SystemExit(f"manifest-csv not found: {manifest_csv}")
    if not reports_root.exists():
        raise SystemExit(f"reports-root not found: {reports_root}")

    encoder_bundle = build_text_encoder(
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
        revision=args.revision,
        trust_remote_code=bool(args.trust_remote_code),
        cache_dir=args.cache_dir,
    )
    encoder_bundle.model.to(device)
    encoder_bundle.model.eval()

    max_length = resolve_max_length(args.max_length, encoder_bundle.tokenizer)

    experiment_slug = build_experiment_slug(
        experiment_prefix=args.experiment_prefix,
        manifest_csv=manifest_csv,
        splits=splits,
        model_id=encoder_bundle.model_id,
        pooling=args.pooling,
        normalization=args.normalization,
        max_length=max_length,
        fp16_on_cuda=bool(args.fp16_on_cuda),
        revision=args.revision,
        max_reports_per_split=args.max_reports_per_split,
    )
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=args.experiment_name,
        generated_slug=experiment_slug,
        overwrite=bool(args.overwrite),
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    experiment_meta = {
        "run_date_utc": utc_now_iso(),
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "manifest_csv": str(manifest_csv),
        "reports_root": str(reports_root),
        "splits": splits,
        "encoder_backend": encoder_bundle.backend,
        "model_id": encoder_bundle.model_id,
        "tokenizer_id": encoder_bundle.tokenizer_id,
        "revision": args.revision,
        "pooling": args.pooling,
        "normalization": args.normalization,
        "max_length": max_length,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device_requested": args.device,
        "device_resolved": str(device),
        "fp16_on_cuda": bool(args.fp16_on_cuda),
        "trust_remote_code": bool(args.trust_remote_code),
        "max_reports_per_split": args.max_reports_per_split,
        "text_prefix": args.text_prefix,
        "text_suffix": args.text_suffix,
        "normalize_whitespace": bool(args.normalize_whitespace),
        "argv": sys.argv,
        "encoder_build_meta": encoder_bundle.build_meta,
        "split_output_dirs": {split: str(experiment_dir / split) for split in splits},
    }
    write_json(experiment_dir / "experiment_meta.json", experiment_meta)

    print(
        f"[info] experiment_id={experiment_id} backend={encoder_bundle.backend} model={encoder_bundle.model_id} "
        f"pooling={args.pooling} normalization={args.normalization} max_length={max_length} device={device}"
    )
    print(f"[info] experiment_dir={experiment_dir}")
    print(f"[info] manifest_csv={manifest_csv} reports_root={reports_root}")

    split_samples = load_manifest_split_reports(
        manifest_csv,
        reports_root,
        splits,
        max_reports=args.max_reports_per_split,
    )

    for split in splits:
        output_dir = experiment_dir / split
        samples = split_samples[split]
        print(f"[info] split={split} reports={len(samples)}")

        embeddings, embedded_samples, failed_reports, feature_summary = embed_split(
            split,
            samples,
            encoder_bundle,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_length=max_length,
            pooling=args.pooling,
            normalization=args.normalization,
            fp16_on_cuda=bool(args.fp16_on_cuda),
            text_prefix=args.text_prefix,
            text_suffix=args.text_suffix,
            normalize_whitespace=bool(args.normalize_whitespace),
        )

        run_meta = {
            "run_date_utc": utc_now_iso(),
            "experiment_number": experiment_number,
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "experiment_dir": str(experiment_dir),
            "split": split,
            "manifest_csv": str(manifest_csv),
            "reports_root": str(reports_root),
            "encoder_backend": encoder_bundle.backend,
            "model_id": encoder_bundle.model_id,
            "tokenizer_id": encoder_bundle.tokenizer_id,
            "revision": args.revision,
            "pooling": args.pooling,
            "normalization": args.normalization,
            "max_length": max_length,
            "output_dir": str(output_dir),
            "num_input_reports": len(samples),
            "num_embedded_reports": int(embeddings.shape[0]),
            "num_failed_reports": len(failed_reports),
            "embedding_dim": int(embeddings.shape[1]),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device_requested": args.device,
            "device_resolved": str(device),
            "fp16_on_cuda": bool(args.fp16_on_cuda),
            "trust_remote_code": bool(args.trust_remote_code),
            "max_reports_per_split": args.max_reports_per_split,
            "text_prefix": args.text_prefix,
            "text_suffix": args.text_suffix,
            "normalize_whitespace": bool(args.normalize_whitespace),
            "feature_summary": feature_summary,
            "encoder_build_meta": encoder_bundle.build_meta,
        }
        save_split_outputs(
            output_dir,
            embeddings=embeddings,
            samples=embedded_samples,
            failed_reports=failed_reports,
            run_meta=run_meta,
        )
        print(f"[saved] split={split} output_dir={output_dir}")


if __name__ == "__main__":
    main()
