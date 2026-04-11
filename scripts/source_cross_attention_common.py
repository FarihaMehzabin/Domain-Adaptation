#!/usr/bin/env python3
"""Shared utilities for the NIH source-stage cross-attention branch."""

from __future__ import annotations

import csv
import io
import json
import math
import random
import re
from contextlib import nullcontext
from dataclasses import dataclass
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
    from PIL import Image
    from torch import nn
    from torch.utils.data import Dataset
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'torch' and related vision libraries.") from exc


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_nih_cxr14_all14.csv")
DEFAULT_DATA_ROOT = Path("/workspace/data")
DEFAULT_REPORTS_ROOT = Path("/workspace/reports")
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_IMAGE_MODEL_ID = "resnet50"
DEFAULT_TEXT_MODEL_ID = "microsoft/BiomedVLP-CXR-BERT-specialized"
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICE = "auto"
DEFAULT_MAX_LENGTH = 512
DEFAULT_FUSION_DIM = 256
DEFAULT_FUSION_HEADS = 8
DEFAULT_FUSION_LAYERS = 2
DEFAULT_EMBEDDING_DIM = 512
DEFAULT_DROPOUT = 0.1
DEFAULT_ECE_BINS = 15
DEFAULT_SEED = 1337
DEFAULT_EXPERIMENT_ID_WIDTH = 4

SPLIT_ALIASES = {
    "train": "train",
    "trn": "train",
    "val": "val",
    "validation": "val",
    "valid": "val",
    "test": "test",
    "tst": "test",
}


@dataclass(frozen=True)
class ManifestRecord:
    split: str
    row_id: str
    image_path: str
    image_abspath: Path
    report_path: Path
    labels: tuple[float, ...]


@dataclass(frozen=True)
class ImageEncoderBundle:
    model: nn.Module
    preprocess: Any
    weights_label: str
    resolved_input_size: tuple[int, int]
    feature_dim: int
    spatial_token_count: int


@dataclass(frozen=True)
class TextEncoderBundle:
    model: Any
    tokenizer: Any
    model_id: str
    tokenizer_id: str
    hidden_size: int
    projected_embedding_size: int
    max_position_embeddings: int | None


class NIHMultimodalDataset(Dataset):
    def __init__(
        self,
        samples: list[ManifestRecord],
        *,
        image_transform: Any,
        text_prefix: str,
        text_suffix: str,
        normalize_whitespace: bool,
    ) -> None:
        self.samples = samples
        self.image_transform = image_transform
        self.text_prefix = text_prefix
        self.text_suffix = text_suffix
        self.normalize_whitespace = normalize_whitespace

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        try:
            with Image.open(sample.image_abspath) as image:
                rgb_image = image.convert("RGB")
            pixel_values = self.image_transform(rgb_image)
            report_text = sample.report_path.read_text(encoding="utf-8").strip()
            if self.normalize_whitespace:
                report_text = " ".join(report_text.split())
            report_text = f"{self.text_prefix}{report_text}{self.text_suffix}".strip()
            if not report_text:
                raise ValueError("report text is empty after preprocessing")
            return {
                "row_id": sample.row_id,
                "image_path": sample.image_path,
                "pixel_values": pixel_values,
                "text": report_text,
                "labels": np.asarray(sample.labels, dtype=np.float32),
                "error": None,
            }
        except Exception as exc:
            return {
                "row_id": sample.row_id,
                "image_path": sample.image_path,
                "pixel_values": None,
                "text": None,
                "labels": np.asarray(sample.labels, dtype=np.float32),
                "error": str(exc),
            }


@dataclass
class MultimodalCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        row_ids: list[str] = []
        image_paths: list[str] = []
        images: list[torch.Tensor] = []
        texts: list[str] = []
        labels: list[np.ndarray] = []
        failures: list[dict[str, str]] = []

        for item in items:
            if item["pixel_values"] is None or item["text"] is None:
                failures.append(
                    {
                        "row_id": str(item["row_id"]),
                        "image_path": str(item["image_path"]),
                        "error": str(item.get("error") or "unknown_error"),
                    }
                )
                continue
            row_ids.append(str(item["row_id"]))
            image_paths.append(str(item["image_path"]))
            images.append(item["pixel_values"])
            texts.append(str(item["text"]))
            labels.append(np.asarray(item["labels"], dtype=np.float32))

        encoded = None
        if texts:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        labels_tensor = None
        pixel_values = None
        if labels:
            labels_tensor = torch.from_numpy(np.stack(labels, axis=0).astype(np.float32))
            pixel_values = torch.stack(images, dim=0)

        return {
            "row_ids": row_ids,
            "image_paths": image_paths,
            "pixel_values": pixel_values,
            "encoded": encoded,
            "labels": labels_tensor,
            "failures": failures,
        }


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
    if isinstance(value, torch.Tensor):
        return to_serializable(value.detach().cpu().numpy())
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_serializable(payload), sort_keys=True) + "\n")


def slugify(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or fallback


def ensure_operation_prefix(name: str, operation_label: str) -> str:
    normalized_label = slugify(operation_label, fallback="operation")
    explicit_match = re.match(r"^(exp\d+__)(.+)$", name)
    if explicit_match is not None:
        numbered_prefix, remainder = explicit_match.groups()
        if remainder.startswith(normalized_label):
            return name
        return f"{numbered_prefix}{normalized_label}__{remainder}"
    if name.startswith(normalized_label):
        return name
    return f"{normalized_label}__{name}"


def strip_experiment_number_prefix(name: str) -> str:
    if name.startswith("exp") and "__" in name:
        _, remainder = name.split("__", 1)
        return remainder
    return name


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
    operation_label: str,
    overwrite: bool,
    id_width: int = DEFAULT_EXPERIMENT_ID_WIDTH,
) -> tuple[int, str, str, Path]:
    experiments_root.mkdir(parents=True, exist_ok=True)
    if requested_name:
        requested_name = requested_name.strip()
        if not requested_name:
            raise SystemExit("--experiment-name cannot be empty.")
    base_name = ensure_operation_prefix(requested_name or generated_slug, operation_label)

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


def get_autocast_context(device: torch.device, fp16_on_cuda: bool) -> Any:
    if bool(fp16_on_cuda and device.type == "cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def format_metric(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{float(value):.6f}"


def canonicalize_split(raw_split: str) -> str:
    normalized = SPLIT_ALIASES.get(raw_split.strip().lower())
    if normalized is None:
        valid = ", ".join(sorted(SPLIT_ALIASES))
        raise SystemExit(f"Unsupported split '{raw_split}'. Valid values: {valid}")
    return normalized


def candidate_image_paths(data_root: Path, image_path: str) -> list[Path]:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return [candidate]
    roots: list[Path] = []
    for root in (data_root, Path("/workspace"), Path("/workspace/data")):
        normalized = root.expanduser()
        if normalized not in roots:
            roots.append(normalized)
    return [root / image_path for root in roots]


def resolve_manifest_image_path(
    data_root: Path,
    image_path: str,
    *,
    verify_exists: bool,
) -> Path:
    candidate = Path(image_path)
    if not verify_exists:
        if candidate.is_absolute():
            return candidate
        return data_root / image_path

    candidates = candidate_image_paths(data_root, image_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{image_path} not found. Tried: {attempted}")


def load_manifest_records(
    manifest_csv: Path,
    *,
    data_root: Path,
    reports_root: Path,
    splits: list[str],
    max_samples_per_split: int | None,
    verify_files: bool = True,
) -> tuple[list[str], dict[str, list[ManifestRecord]]]:
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV not found: {manifest_csv}")

    manifest_text = manifest_csv.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(manifest_text))
    if reader.fieldnames is None:
        raise SystemExit(f"Manifest CSV has no header: {manifest_csv}")
    required = {"dataset", "split", "image_path"}
    if not required.issubset(reader.fieldnames):
        raise SystemExit(f"Manifest CSV must contain columns: {sorted(required)}")

    label_columns = [field for field in reader.fieldnames if field.startswith("label_")]
    if not label_columns:
        raise SystemExit("Manifest CSV does not contain any label_... columns.")

    split_set = set(splits)
    records_by_split: dict[str, list[ManifestRecord]] = {split: [] for split in splits}
    missing_images: dict[str, list[str]] = {split: [] for split in splits}
    missing_reports: dict[str, list[str]] = {split: [] for split in splits}
    duplicate_ids: dict[str, set[str]] = {split: set() for split in splits}
    seen_ids: dict[str, set[str]] = {split: set() for split in splits}

    for row in reader:
        dataset = (row.get("dataset") or "").strip()
        if dataset and dataset != "nih_cxr14":
            continue
        split = canonicalize_split(row.get("split") or "")
        if split not in split_set:
            continue
        if max_samples_per_split is not None and len(records_by_split[split]) >= max_samples_per_split:
            continue

        image_path = (row.get("image_path") or "").strip()
        if not image_path:
            continue
        row_id = Path(image_path).stem
        if row_id in seen_ids[split]:
            duplicate_ids[split].add(row_id)
            continue

        try:
            image_abspath = resolve_manifest_image_path(
                data_root,
                image_path,
                verify_exists=verify_files,
            )
        except FileNotFoundError as exc:
            if len(missing_images[split]) < 5:
                missing_images[split].append(str(exc))
            continue

        report_path = reports_root / split / f"{row_id}.txt"
        if verify_files and not report_path.exists():
            if len(missing_reports[split]) < 5:
                missing_reports[split].append(str(report_path))
            continue

        labels: list[float] = []
        for label_column in label_columns:
            raw_value = str(row.get(label_column) or "0").strip()
            try:
                labels.append(float(raw_value))
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid value '{raw_value}' in column '{label_column}' for row '{row_id}'."
                ) from exc

        records_by_split[split].append(
            ManifestRecord(
                split=split,
                row_id=row_id,
                image_path=str(image_abspath),
                image_abspath=image_abspath,
                report_path=report_path,
                labels=tuple(labels),
            )
        )
        seen_ids[split].add(row_id)

    for split in splits:
        if duplicate_ids[split]:
            examples = sorted(duplicate_ids[split])[:5]
            raise SystemExit(f"Duplicate row IDs found in split '{split}'. Examples: {examples}")
        if not records_by_split[split]:
            if missing_images[split]:
                sample = "\n".join(missing_images[split])
                raise SystemExit(f"No usable images found for split '{split}'. Examples:\n{sample}")
            if missing_reports[split]:
                sample = "\n".join(missing_reports[split])
                raise SystemExit(f"No usable reports found for split '{split}'. Examples:\n{sample}")
            raise SystemExit(f"No usable records found for split '{split}'.")
        if verify_files and missing_images[split]:
            sample = "\n".join(missing_images[split])
            raise SystemExit(
                f"Found rows for split '{split}', but some images could not be resolved.\nExamples:\n{sample}"
            )
        if verify_files and missing_reports[split]:
            sample = "\n".join(missing_reports[split])
            raise SystemExit(
                f"Found rows for split '{split}', but some reports were missing.\nExamples:\n{sample}"
            )

    return label_columns, records_by_split


def build_image_encoder_bundle() -> ImageEncoderBundle:
    try:
        from torchvision.models import ResNet50_Weights, resnet50
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Missing dependency 'torchvision'.") from exc

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    children = list(model.children())
    if len(children) < 2:
        raise SystemExit("Unexpected ResNet50 structure; unable to derive the feature trunk.")
    trunk = nn.Sequential(*children[:-2])
    preprocess = weights.transforms()
    return ImageEncoderBundle(
        model=trunk,
        preprocess=preprocess,
        weights_label="DEFAULT",
        resolved_input_size=(224, 224),
        feature_dim=2048,
        spatial_token_count=49,
    )


def build_text_encoder_bundle(
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
        raise SystemExit("Missing dependency 'transformers'.") from exc

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
    hidden_size = getattr(config, "hidden_size", None)
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise SystemExit(f"Unable to determine text hidden size from {model_id}.")
    projected_embedding_size = infer_projected_text_embedding_size(
        model=model,
        tokenizer=tokenizer,
    )
    return TextEncoderBundle(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        tokenizer_id=resolved_tokenizer_id,
        hidden_size=hidden_size,
        projected_embedding_size=projected_embedding_size,
        max_position_embeddings=getattr(config, "max_position_embeddings", None),
    )


def forward_text_encoder(
    *,
    text_encoder: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    request_projected_embedding: bool,
) -> Any:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_dict": True,
    }
    if request_projected_embedding:
        kwargs["output_cls_projected_embedding"] = True
    try:
        return text_encoder(**kwargs)
    except TypeError:
        if request_projected_embedding:
            kwargs.pop("output_cls_projected_embedding", None)
            return text_encoder(**kwargs)
        raise


def extract_projected_text_embedding(
    *,
    text_encoder: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    outputs: Any | None,
) -> torch.Tensor:
    projected = getattr(outputs, "cls_projected_embedding", None) if outputs is not None else None
    if projected is None:
        projected = getattr(outputs, "text_embeds", None) if outputs is not None else None
    if projected is None:
        projected = getattr(outputs, "sentence_embedding", None) if outputs is not None else None
    if projected is None:
        projected = getattr(outputs, "pooler_output", None) if outputs is not None else None
    if projected is None and hasattr(text_encoder, "get_projected_text_embeddings"):
        projected = text_encoder.get_projected_text_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            normalize_embeddings=False,
        )
    if projected is None and hasattr(text_encoder, "get_text_features"):
        projected = text_encoder.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    if projected is None and outputs is not None:
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is not None:
            projected = masked_mean_pool(last_hidden_state, attention_mask=attention_mask)
    if projected is None:
        raise RuntimeError("Unable to extract a pooled/projected text embedding from the text encoder.")
    if projected.ndim != 2:
        raise RuntimeError(f"Expected a 2D text embedding tensor, found {tuple(projected.shape)}")
    return projected


def infer_projected_text_embedding_size(*, model: Any, tokenizer: Any) -> int:
    probe_inputs = tokenizer(
        ["projection size probe"],
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = forward_text_encoder(
            text_encoder=model,
            input_ids=probe_inputs["input_ids"],
            attention_mask=probe_inputs["attention_mask"],
            request_projected_embedding=True,
        )
        projected = extract_projected_text_embedding(
            text_encoder=model,
            input_ids=probe_inputs["input_ids"],
            attention_mask=probe_inputs["attention_mask"],
            outputs=outputs,
        )
    projected_size = int(projected.shape[1])
    if projected_size <= 0:
        raise SystemExit("Unable to determine projected text embedding size.")
    return projected_size


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.text_attn_norm = nn.LayerNorm(d_model)
        self.image_kv_norm = nn.LayerNorm(d_model)
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_ffn_norm = nn.LayerNorm(d_model)
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.image_attn_norm = nn.LayerNorm(d_model)
        self.text_kv_norm = nn.LayerNorm(d_model)
        self.image_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_ffn_norm = nn.LayerNorm(d_model)
        self.image_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        *,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_text = self.text_attn_norm(text_tokens)
        normalized_image = self.image_kv_norm(image_tokens)
        text_update, _ = self.text_cross_attn(
            query=normalized_text,
            key=normalized_image,
            value=normalized_image,
            need_weights=False,
        )
        text_tokens = text_tokens + self.dropout(text_update)
        text_tokens = text_tokens + self.dropout(self.text_ffn(self.text_ffn_norm(text_tokens)))

        normalized_image = self.image_attn_norm(image_tokens)
        normalized_text = self.text_kv_norm(text_tokens)
        image_update, _ = self.image_cross_attn(
            query=normalized_image,
            key=normalized_text,
            value=normalized_text,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )
        image_tokens = image_tokens + self.dropout(image_update)
        image_tokens = image_tokens + self.dropout(self.image_ffn(self.image_ffn_norm(image_tokens)))
        return text_tokens, image_tokens


class CrossAttentionMultimodalModel(nn.Module):
    def __init__(
        self,
        *,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        text_hidden_size: int,
        legacy_text_feature_dim: int,
        num_labels: int,
        image_feature_dim: int,
        image_token_count: int,
        fusion_dim: int,
        num_heads: int,
        num_layers: int,
        embedding_dim: int,
        dropout: float,
        gated_hybrid: bool = True,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        freeze_module(self.image_encoder)
        freeze_module(self.text_encoder)
        self.gated_hybrid = bool(gated_hybrid)

        self.image_proj = nn.Linear(image_feature_dim, fusion_dim)
        self.text_proj = nn.Linear(text_hidden_size, fusion_dim)
        self.image_pos_embed = nn.Parameter(torch.zeros(1, image_token_count, fusion_dim))
        self.image_type_embed = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model=fusion_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        pooled_dim = fusion_dim * 2
        self.cross_attention_embedding_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        legacy_concat_dim = image_feature_dim + legacy_text_feature_dim
        self.legacy_embedding_head = None
        self.hybrid_gate = None
        self.hybrid_refine = None
        if self.gated_hybrid:
            self.legacy_embedding_head = nn.Sequential(
                nn.LayerNorm(legacy_concat_dim),
                nn.Linear(legacy_concat_dim, embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
            )
            self.hybrid_gate = nn.Sequential(
                nn.LayerNorm(embedding_dim * 2),
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Sigmoid(),
            )
            self.hybrid_refine = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
            )
        self.classifier = nn.Linear(embedding_dim, num_labels)
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.image_token_count = image_token_count
        self.image_feature_dim = image_feature_dim
        self.legacy_text_feature_dim = legacy_text_feature_dim
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.image_pos_embed, std=0.02)
        nn.init.normal_(self.image_type_embed, std=0.02)
        nn.init.normal_(self.text_type_embed, std=0.02)
        nn.init.xavier_uniform_(self.image_proj.weight)
        nn.init.zeros_(self.image_proj.bias)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)
        if self.hybrid_gate is not None:
            gate_linear = self.hybrid_gate[1]
            nn.init.zeros_(gate_linear.weight)
            nn.init.zeros_(gate_linear.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def train(self, mode: bool = True) -> "CrossAttentionMultimodalModel":
        super().train(mode)
        self.image_encoder.eval()
        self.text_encoder.eval()
        return self

    def encode(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            image_features = self.image_encoder(pixel_values)
            text_outputs = forward_text_encoder(
                text_encoder=self.text_encoder,
                input_ids=input_ids,
                attention_mask=attention_mask,
                request_projected_embedding=self.gated_hybrid,
            )
            last_hidden_state = getattr(text_outputs, "last_hidden_state", None)
            if last_hidden_state is None:
                raise RuntimeError("Text encoder output does not contain last_hidden_state.")
            legacy_text_embedding = None
            if self.gated_hybrid:
                legacy_text_embedding = extract_projected_text_embedding(
                    text_encoder=self.text_encoder,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    outputs=text_outputs,
                )

        if image_features.ndim != 4:
            raise RuntimeError(f"Expected 4D image features, found {tuple(image_features.shape)}")
        image_tokens = image_features.flatten(2).transpose(1, 2)
        if image_tokens.shape[1] != self.image_token_count:
            raise RuntimeError(
                f"Unexpected image token count {image_tokens.shape[1]}; expected {self.image_token_count}."
            )
        image_tokens = self.image_proj(image_tokens)
        image_tokens = image_tokens + self.image_pos_embed[:, : image_tokens.shape[1], :] + self.image_type_embed

        text_tokens = self.text_proj(last_hidden_state) + self.text_type_embed
        text_key_padding_mask = attention_mask == 0

        image_tokens = self.input_dropout(image_tokens)
        text_tokens = self.input_dropout(text_tokens)
        for block in self.blocks:
            text_tokens, image_tokens = block(
                text_tokens=text_tokens,
                text_key_padding_mask=text_key_padding_mask,
                image_tokens=image_tokens,
            )

        text_pooled = masked_mean_pool(text_tokens, attention_mask=attention_mask)
        image_pooled = image_tokens.mean(dim=1)
        pooled = torch.cat([image_pooled, text_pooled], dim=1)
        cross_attention_embedding = self.cross_attention_embedding_head(pooled)
        if not self.gated_hybrid:
            return F.normalize(cross_attention_embedding, p=2, dim=1)

        if legacy_text_embedding is None or self.legacy_embedding_head is None or self.hybrid_gate is None or self.hybrid_refine is None:
            raise RuntimeError("Gated hybrid mode requires legacy text embeddings and hybrid heads.")

        legacy_image_embedding = F.normalize(image_features.mean(dim=(-2, -1)), p=2, dim=1)
        legacy_text_embedding = F.normalize(legacy_text_embedding, p=2, dim=1)
        legacy_concat = F.normalize(
            torch.cat([legacy_image_embedding, legacy_text_embedding], dim=1),
            p=2,
            dim=1,
        )
        legacy_embedding = self.legacy_embedding_head(legacy_concat)
        gate = self.hybrid_gate(torch.cat([cross_attention_embedding, legacy_embedding], dim=1))
        hybrid_embedding = gate * cross_attention_embedding + (1.0 - gate) * legacy_embedding
        hybrid_embedding = hybrid_embedding + self.hybrid_refine(hybrid_embedding)
        return F.normalize(hybrid_embedding, p=2, dim=1)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = self.classifier(embedding)
        return logits, embedding


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=(device.type == "cuda")) for key, value in batch.items()}


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    positive = labels.sum(axis=0)
    negative = float(labels.shape[0]) - positive
    safe_positive = np.where(positive > 0.0, positive, 1.0)
    ratio = negative / safe_positive
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    ratio = np.where(positive > 0.0, ratio, 1.0)
    return torch.tensor(ratio.astype(np.float32))


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(y.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)

    start = 0
    total = sorted_scores.shape[0]
    while start < total:
        end = start + 1
        while end < total and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[start:end] = average_rank
        start = end

    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks
    sum_ranks_pos = float(full_ranks[y == 1].sum())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def binary_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.int64)
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, y_sorted.shape[0] + 1, dtype=np.float64)
    return float(precision[y_sorted == 1].sum() / positives)


def binary_f1_stats(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    preds = (np.asarray(probs, dtype=np.float64) >= float(threshold)).astype(np.int64)
    tp = float(np.sum((preds == 1) & (y == 1)))
    fp = float(np.sum((preds == 1) & (y == 0)))
    fn = float(np.sum((preds == 0) & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def binary_ece(y_true: np.ndarray, probs: np.ndarray, num_bins: int = DEFAULT_ECE_BINS) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total = float(p.shape[0])
    for idx in range(num_bins):
        left = bins[idx]
        right = bins[idx + 1]
        if idx == num_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(p[mask].mean())
        empirical = float(y[mask].mean())
        ece += (count / total) * abs(confidence - empirical)
    return float(ece)


def tune_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    label_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    thresholds = np.full((len(label_names),), 0.5, dtype=np.float32)
    label_payload: dict[str, Any] = {}
    for idx, label_name in enumerate(label_names):
        y = np.asarray(y_true[:, idx], dtype=np.int64)
        p = np.asarray(probs[:, idx], dtype=np.float64)
        positives = int(y.sum())
        if positives == 0:
            label_payload[label_name] = {
                "threshold": 0.5,
                "best_f1": 0.0,
                "prevalence": float(y.mean()),
                "reason": "no_positive_examples_in_val",
            }
            continue

        order = np.argsort(-p, kind="mergesort")
        y_sorted = y[order]
        p_sorted = p[order]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        fn = positives - tp
        denom = (2 * tp) + fp + fn
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=np.float64), where=denom > 0)
        best_index = int(np.argmax(f1))
        threshold = float(np.clip(p_sorted[best_index], 1e-6, 1.0 - 1e-6))
        thresholds[idx] = threshold
        label_payload[label_name] = {
            "threshold": threshold,
            "best_f1": float(f1[best_index]),
            "prevalence": float(y.mean()),
            "reason": "argmax_exact_f1_on_val",
        }

    return thresholds, {
        "selection_split": "val",
        "selection_metric": "per_label_f1",
        "macro_threshold": float(np.mean(thresholds)) if thresholds.size else None,
        "labels": label_payload,
    }


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def summarize_split_metrics(
    *,
    split: str,
    loss: float,
    targets: np.ndarray,
    logits: np.ndarray,
    label_names: list[str],
    tuned_thresholds: np.ndarray,
) -> dict[str, Any]:
    probs = sigmoid_np(logits.astype(np.float64))
    label_metrics: dict[str, Any] = {}
    aurocs: list[float | None] = []
    aps: list[float | None] = []
    eces: list[float] = []
    f1_half: list[float] = []
    f1_tuned: list[float] = []

    for idx, label_name in enumerate(label_names):
        y = np.asarray(targets[:, idx], dtype=np.int64)
        p = np.asarray(probs[:, idx], dtype=np.float64)
        auroc = binary_auroc(y, p)
        ap = binary_average_precision(y, p)
        ece = binary_ece(y, p)
        half_stats = binary_f1_stats(y, p, 0.5)
        tuned_stats = binary_f1_stats(y, p, float(tuned_thresholds[idx]))
        aurocs.append(auroc)
        aps.append(ap)
        eces.append(ece)
        f1_half.append(half_stats["f1"])
        f1_tuned.append(tuned_stats["f1"])
        label_metrics[label_name] = {
            "prevalence": float(y.mean()),
            "positive_count": int(y.sum()),
            "negative_count": int(y.shape[0] - y.sum()),
            "auroc": auroc,
            "average_precision": ap,
            "ece": ece,
            "f1_at_0.5": half_stats["f1"],
            "f1_at_tuned_threshold": tuned_stats["f1"],
            "precision_at_0.5": half_stats["precision"],
            "recall_at_0.5": half_stats["recall"],
            "precision_at_tuned_threshold": tuned_stats["precision"],
            "recall_at_tuned_threshold": tuned_stats["recall"],
            "threshold_used": float(tuned_thresholds[idx]),
        }

    return {
        "split": split,
        "num_examples": int(targets.shape[0]),
        "loss": float(loss),
        "macro": {
            "auroc": mean_or_none(aurocs),
            "average_precision": mean_or_none(aps),
            "ece": mean_or_none(eces),
            "f1_at_0.5": mean_or_none(f1_half),
            "f1_at_tuned_thresholds": mean_or_none(f1_tuned),
        },
        "valid_label_counts": {
            "macro_auroc": int(sum(value is not None for value in aurocs)),
            "macro_average_precision": int(sum(value is not None for value in aps)),
        },
        "label_metrics": label_metrics,
    }


def selection_tuple(summary: dict[str, Any]) -> tuple[float, float, float]:
    macro = summary["macro"]
    auroc = float(macro["auroc"]) if macro["auroc"] is not None else float("-inf")
    ap = float(macro["average_precision"]) if macro["average_precision"] is not None else float("-inf")
    loss = float(summary["loss"])
    return (auroc, ap, -loss)
