#!/usr/bin/env python3
"""Generate split-aware NIH CXR14 image embeddings for configurable vision encoders."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import experiment_layout

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'numpy'.") from exc

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'torch'.") from exc

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'pillow'.") from exc


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest/manifest_nih_cxr14 .csv")
DEFAULT_DATA_ROOT = Path("/workspace")
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_SPLITS = ["train", "val", "test"]
DEFAULT_EXTENSIONS = [".png", ".jpg", ".jpeg"]
DEFAULT_ENCODER_BACKEND = "torchvision"
DEFAULT_ENCODER_ID = "resnet50"
DEFAULT_WEIGHTS = "DEFAULT"
DEFAULT_POOLING = "avg"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICE = "auto"
DEFAULT_OPERATION_LABEL = "embedding_generation"
DEFAULT_EXPERIMENT_PREFIX = f"{DEFAULT_OPERATION_LABEL}__nih_cxr14_image_embeddings"
DEFAULT_EXPERIMENT_ID_WIDTH = 4


@dataclass
class EncoderBundle:
    model: nn.Module
    preprocess: Callable[[Image.Image], torch.Tensor]
    forward_fn: Callable[[torch.Tensor], Any]
    backend: str
    encoder_id: str
    weights_label: str
    resolved_input_size: tuple[int, int]
    default_prefix_tokens: int = 0
    build_meta: dict[str, Any] = field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def normalize_extensions(raw_extensions: list[str]) -> set[str]:
    normalized: set[str] = set()
    for ext in raw_extensions:
        cleaned = ext.strip().lower()
        if not cleaned:
            continue
        if not cleaned.startswith("."):
            cleaned = "." + cleaned
        normalized.add(cleaned)
    if not normalized:
        raise SystemExit("No valid image extensions were provided.")
    return normalized


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


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


def parse_input_size(raw_input_size: list[int] | None) -> tuple[int, int] | None:
    if raw_input_size is None:
        return None
    if len(raw_input_size) == 1:
        side = int(raw_input_size[0])
        if side <= 0:
            raise SystemExit("--input-size values must be positive integers.")
        return side, side
    if len(raw_input_size) == 2:
        height, width = int(raw_input_size[0]), int(raw_input_size[1])
        if height <= 0 or width <= 0:
            raise SystemExit("--input-size values must be positive integers.")
        return height, width
    raise SystemExit("--input-size accepts either one integer or two integers: H W")


def coerce_hw(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return int(value[0]), int(value[0])
        if len(value) >= 2:
            return int(value[0]), int(value[1])
    if isinstance(value, dict):
        height = value.get("height")
        width = value.get("width")
        shortest_edge = value.get("shortest_edge")
        if height is not None and width is not None:
            return int(height), int(width)
        if shortest_edge is not None:
            return int(shortest_edge), int(shortest_edge)
    return None


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


def candidate_image_paths(data_root: Path, image_path: str) -> list[Path]:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return [candidate]

    roots = dedupe_preserve_order(
        [
            str(data_root.resolve()),
            str(Path("/workspace").resolve()),
            str((Path("/workspace") / "data").resolve()),
        ]
    )
    return [Path(root) / image_path for root in roots]


def resolve_manifest_image_path(
    data_root: Path,
    image_path: str,
    *,
    verify_exists: bool,
) -> Path:
    if not verify_exists:
        candidate = Path(image_path)
        if candidate.is_absolute():
            return candidate
        return data_root / image_path
    candidates = candidate_image_paths(data_root, image_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{image_path} not found. Tried: {attempted}")


def load_manifest_split_image_paths(
    manifest_csv: Path,
    data_root: Path,
    splits: list[str],
    image_extensions: set[str],
    *,
    verify_exists: bool,
    max_images: int | None,
) -> dict[str, list[Path]]:
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV not found: {manifest_csv}")

    split_set = set(splits)
    resolved_paths: dict[str, list[Path]] = {split: [] for split in splits}
    missing_examples: dict[str, list[str]] = {split: [] for split in splits}

    manifest_text = manifest_csv.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(manifest_text))
    required_columns = {"image_path", "split"}
    if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
        raise SystemExit(f"Manifest CSV must contain columns: {sorted(required_columns)}")

    for row in reader:
        if row.get("dataset") and row["dataset"] != "nih_cxr14":
            continue
        split = row.get("split") or ""
        if split not in split_set:
            continue
        if max_images is not None and len(resolved_paths[split]) >= max_images:
            continue

        image_path = (row.get("image_path") or "").strip()
        if not image_path:
            continue
        if Path(image_path).suffix.lower() not in image_extensions:
            continue

        try:
            resolved = resolve_manifest_image_path(
                data_root,
                image_path,
                verify_exists=verify_exists,
            )
        except FileNotFoundError as exc:
            if len(missing_examples[split]) < 5:
                missing_examples[split].append(str(exc))
            continue

        resolved_paths[split].append(resolved)

    for split in splits:
        if not resolved_paths[split]:
            sample = "\n".join(missing_examples[split]) if missing_examples[split] else "No matching rows found in manifest."
            raise SystemExit(
                f"No usable images found for split '{split}'. Check --data-root.\n"
                f"Examples:\n{sample}"
            )

        if missing_examples[split]:
            sample = "\n".join(missing_examples[split])
            raise SystemExit(
                f"Found split '{split}' rows in {manifest_csv}, but some images could not be resolved.\n"
                f"Pass the correct --data-root.\nExamples:\n{sample}"
            )

    return resolved_paths


class CXRImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
            pixel_values = self.transform(rgb_image)
            return {
                "path": str(image_path),
                "pixel_values": pixel_values,
                "error": None,
            }
        except Exception as exc:
            return {
                "path": str(image_path),
                "pixel_values": None,
                "error": str(exc),
            }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    paths: list[str] = []
    tensors: list[torch.Tensor] = []
    errors: list[dict[str, str]] = []
    for item in items:
        if item.get("pixel_values") is None:
            errors.append(
                {
                    "path": str(item["path"]),
                    "error": str(item.get("error") or "unknown_error"),
                }
            )
            continue
        paths.append(str(item["path"]))
        tensors.append(item["pixel_values"])

    stacked = torch.stack(tensors, dim=0) if tensors else None
    return {
        "paths": paths,
        "pixel_values": stacked,
        "errors": errors,
    }


def build_dataloader(
    image_paths: list[Path],
    transform: Callable[[Image.Image], torch.Tensor],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = CXRImageDataset(image_paths, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_batch,
    )


def save_manifest_csv(path: Path, image_paths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "image_path"])
        for idx, image_path in enumerate(image_paths):
            writer.writerow([idx, image_path])


def save_failed_jsonl(path: Path, failures: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row) + "\n")


def save_split_outputs(
    output_dir: Path,
    *,
    embeddings: np.ndarray,
    embedded_paths: list[str],
    failed_images: list[dict[str, str]],
    run_meta: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings.astype(np.float32))
    save_manifest_csv(output_dir / "image_manifest.csv", embedded_paths)
    (output_dir / "image_paths.txt").write_text("\n".join(embedded_paths) + "\n", encoding="utf-8")
    if failed_images:
        save_failed_jsonl(output_dir / "failed_images.jsonl", failed_images)
    write_json(output_dir / "run_meta.json", run_meta)


def resolve_torchvision_weights(encoder_id: str, weights_name: str) -> tuple[Any, str]:
    try:
        from torchvision.models import get_model_weights
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Missing dependency 'torchvision'.") from exc

    normalized = weights_name.strip()
    if normalized.upper() in {"NONE", "NULL", "RANDOM"}:
        return None, "none"

    try:
        weights_enum = get_model_weights(encoder_id)
    except Exception as exc:
        raise SystemExit(
            f"Unable to resolve torchvision weights for encoder '{encoder_id}'. "
            f"If this encoder is not from torchvision, use --encoder-backend timm or huggingface."
        ) from exc

    if normalized.upper() == "DEFAULT":
        default_weights = getattr(weights_enum, "DEFAULT", None)
        if default_weights is None:
            raise SystemExit(
                f"torchvision encoder '{encoder_id}' has no DEFAULT weights. "
                f"Pass --weights NONE or a valid explicit weight enum."
            )
        return default_weights, "DEFAULT"

    try:
        return weights_enum[normalized], normalized
    except KeyError as exc:
        valid = ", ".join(weights_enum.__members__.keys())
        raise SystemExit(
            f"Unsupported torchvision --weights value '{weights_name}' for encoder '{encoder_id}'. "
            f"Valid values: {valid}, NONE"
        ) from exc


def build_torchvision_transform(weights: Any, input_size: tuple[int, int] | None) -> tuple[Callable[[Image.Image], torch.Tensor], tuple[int, int]]:
    try:
        from torchvision import transforms
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Missing dependency 'torchvision'.") from exc

    if weights is not None:
        if input_size is None:
            transform = weights.transforms()
            resolved_size = coerce_hw(getattr(transform, "crop_size", None)) or (224, 224)
            return transform, resolved_size
        height, width = input_size
        try:
            transform = weights.transforms(crop_size=[height, width], resize_size=[height, width])
            return transform, input_size
        except TypeError:
            pass

    resolved_size = input_size or (224, 224)
    transform = transforms.Compose(
        [
            transforms.Resize(resolved_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform, resolved_size


def build_torchvision_feature_forward(model: nn.Module) -> tuple[Callable[[torch.Tensor], Any], str]:
    features = getattr(model, "features", None)
    if isinstance(features, nn.Module):
        return features, "features"

    if hasattr(model, "forward_features") and callable(getattr(model, "forward_features")):
        return getattr(model, "forward_features"), "forward_features"

    children = list(model.children())
    if not children:
        raise SystemExit("The selected torchvision encoder has no child modules to build a feature extractor from.")

    head_like_names = ("fc", "classifier", "head", "heads")
    if hasattr(model, "avgpool") and any(hasattr(model, name) for name in head_like_names) and len(children) >= 2:
        feature_model = nn.Sequential(*children[:-2])
        return feature_model, "children[:-2]"

    if any(hasattr(model, name) for name in head_like_names) and len(children) >= 1:
        feature_model = nn.Sequential(*children[:-1])
        return feature_model, "children[:-1]"

    raise SystemExit(
        f"Could not derive a feature extractor for torchvision encoder '{type(model).__name__}'. "
        f"Use a CNN-style torchvision model or switch to --encoder-backend timm/huggingface."
    )


def build_torchvision_encoder(
    *,
    encoder_id: str,
    weights_name: str,
    input_size: tuple[int, int] | None,
) -> EncoderBundle:
    try:
        from torchvision.models import get_model
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Missing dependency 'torchvision'.") from exc

    weights, resolved_weights_label = resolve_torchvision_weights(encoder_id, weights_name)
    model = get_model(encoder_id, weights=weights)
    forward_fn, feature_source = build_torchvision_feature_forward(model)
    preprocess, resolved_input_size = build_torchvision_transform(weights, input_size)
    return EncoderBundle(
        model=model,
        preprocess=preprocess,
        forward_fn=forward_fn,
        backend="torchvision",
        encoder_id=encoder_id,
        weights_label=resolved_weights_label,
        resolved_input_size=resolved_input_size,
        default_prefix_tokens=0,
        build_meta={
            "feature_source_builder": feature_source,
        },
    )


def build_timm_encoder(
    *,
    encoder_id: str,
    weights_name: str,
    input_size: tuple[int, int] | None,
    checkpoint_path: Path | None,
) -> EncoderBundle:
    try:
        import timm
        from timm.data import create_transform

        try:
            from timm.data import resolve_model_data_config

            resolve_config = resolve_model_data_config
        except Exception:
            from timm.data import resolve_data_config

            resolve_config = resolve_data_config
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'timm'. Install it first to use --encoder-backend timm."
        ) from exc

    normalized_weights = weights_name.strip().upper()
    if normalized_weights not in {"DEFAULT", "PRETRAINED", "NONE"}:
        raise SystemExit(
            "For --encoder-backend timm, --weights must be one of: DEFAULT, PRETRAINED, NONE"
        )

    pretrained = normalized_weights != "NONE"
    model_kwargs: dict[str, Any] = {"pretrained": pretrained}
    if checkpoint_path is not None:
        model_kwargs["checkpoint_path"] = str(checkpoint_path.resolve())

    model = timm.create_model(encoder_id, **model_kwargs)
    try:
        resolved_data_config = resolve_config(model=model)
    except TypeError:
        resolved_data_config = resolve_config({}, model=model)
    data_config = dict(resolved_data_config)
    if input_size is not None:
        data_config["input_size"] = (3, input_size[0], input_size[1])
        data_config["crop_pct"] = 1.0
    preprocess = create_transform(**data_config, is_training=False)
    resolved_input_size = coerce_hw(data_config.get("input_size")) or (224, 224)

    if hasattr(model, "forward_features") and callable(getattr(model, "forward_features")):
        forward_fn = getattr(model, "forward_features")
        forward_source = "forward_features"
    else:
        forward_fn = model
        forward_source = "forward"

    weights_label = "checkpoint" if checkpoint_path is not None else ("pretrained" if pretrained else "none")
    return EncoderBundle(
        model=model,
        preprocess=preprocess,
        forward_fn=forward_fn,
        backend="timm",
        encoder_id=encoder_id,
        weights_label=weights_label,
        resolved_input_size=resolved_input_size,
        default_prefix_tokens=int(getattr(model, "num_prefix_tokens", 0) or 0),
        build_meta={
            "feature_source_builder": forward_source,
            "data_config": {
                key: value
                for key, value in data_config.items()
                if isinstance(value, (str, int, float, bool, list, tuple))
            },
        },
    )


def build_huggingface_encoder(
    *,
    encoder_id: str,
    input_size: tuple[int, int] | None,
    revision: str | None,
) -> EncoderBundle:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'transformers'. Install it first to use --encoder-backend huggingface."
        ) from exc

    processor_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    if revision:
        processor_kwargs["revision"] = revision
        model_kwargs["revision"] = revision

    processor = AutoImageProcessor.from_pretrained(encoder_id, **processor_kwargs)
    model = AutoModel.from_pretrained(encoder_id, **model_kwargs)
    default_size = coerce_hw(getattr(processor, "size", None)) or (224, 224)
    resolved_input_size = input_size or default_size

    def preprocess(image: Image.Image) -> torch.Tensor:
        if input_size is not None:
            resized = image.resize((resolved_input_size[1], resolved_input_size[0]), resample=Image.BICUBIC)
            try:
                batch = processor(
                    images=resized,
                    return_tensors="pt",
                    do_resize=False,
                    do_center_crop=False,
                )
            except TypeError:
                batch = processor(images=resized, return_tensors="pt")
            return batch["pixel_values"].squeeze(0)
        batch = processor(images=image, return_tensors="pt")
        return batch["pixel_values"].squeeze(0)

    return EncoderBundle(
        model=model,
        preprocess=preprocess,
        forward_fn=lambda pixel_values: model(pixel_values=pixel_values),
        backend="huggingface",
        encoder_id=encoder_id,
        weights_label="from_pretrained",
        resolved_input_size=resolved_input_size,
        default_prefix_tokens=0,
        build_meta={
            "processor_class": type(processor).__name__,
            "processor_size": getattr(processor, "size", None),
            "revision": revision,
        },
    )


def build_encoder_bundle(
    *,
    backend: str,
    encoder_id: str,
    weights_name: str,
    input_size: tuple[int, int] | None,
    checkpoint_path: Path | None,
    revision: str | None,
) -> EncoderBundle:
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise SystemExit(f"--checkpoint-path not found: {checkpoint_path}")

    if backend == "torchvision":
        return build_torchvision_encoder(
            encoder_id=encoder_id,
            weights_name=weights_name,
            input_size=input_size,
        )
    if backend == "timm":
        return build_timm_encoder(
            encoder_id=encoder_id,
            weights_name=weights_name,
            input_size=input_size,
            checkpoint_path=checkpoint_path,
        )
    if backend == "huggingface":
        if checkpoint_path is not None:
            raise SystemExit("--checkpoint-path is not supported for --encoder-backend huggingface.")
        return build_huggingface_encoder(
            encoder_id=encoder_id,
            input_size=input_size,
            revision=revision,
        )
    raise SystemExit(f"Unsupported --encoder-backend value: {backend}")


def extract_feature_tensor(
    raw_outputs: Any,
    *,
    default_prefix_tokens: int,
) -> tuple[torch.Tensor, int, str]:
    if isinstance(raw_outputs, torch.Tensor):
        prefix_tokens = default_prefix_tokens if raw_outputs.ndim == 3 else 0
        return raw_outputs, prefix_tokens, "tensor"

    if hasattr(raw_outputs, "last_hidden_state") and getattr(raw_outputs, "last_hidden_state") is not None:
        last_hidden_state = getattr(raw_outputs, "last_hidden_state")
        prefix_tokens = default_prefix_tokens if last_hidden_state.ndim == 3 else 0
        return last_hidden_state, prefix_tokens, "last_hidden_state"

    if hasattr(raw_outputs, "image_embeds") and getattr(raw_outputs, "image_embeds") is not None:
        return getattr(raw_outputs, "image_embeds"), 0, "image_embeds"

    if hasattr(raw_outputs, "pooler_output") and getattr(raw_outputs, "pooler_output") is not None:
        return getattr(raw_outputs, "pooler_output"), 0, "pooler_output"

    if isinstance(raw_outputs, dict):
        cls_token = raw_outputs.get("x_norm_clstoken")
        patch_tokens = raw_outputs.get("x_norm_patchtokens")
        if isinstance(cls_token, torch.Tensor) and isinstance(patch_tokens, torch.Tensor):
            if cls_token.ndim == 2 and patch_tokens.ndim == 3:
                combined = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
                return combined, 1, "x_norm_clstoken+x_norm_patchtokens"

        preferred_keys: list[tuple[str, int]] = [
            ("last_hidden_state", default_prefix_tokens),
            ("x_prenorm", default_prefix_tokens),
            ("x", default_prefix_tokens),
            ("x_norm_patchtokens", 0),
            ("features", default_prefix_tokens),
            ("feature_map", 0),
            ("image_embeds", 0),
            ("pooler_output", 0),
            ("x_norm_clstoken", 0),
        ]
        for key, prefix_tokens in preferred_keys:
            value = raw_outputs.get(key)
            if isinstance(value, torch.Tensor):
                if value.ndim != 3:
                    prefix_tokens = 0
                return value, prefix_tokens, key

        for key, value in raw_outputs.items():
            if isinstance(value, torch.Tensor):
                prefix_tokens = default_prefix_tokens if value.ndim == 3 else 0
                return value, prefix_tokens, key

    if isinstance(raw_outputs, (list, tuple)):
        for index, value in enumerate(raw_outputs):
            if isinstance(value, torch.Tensor):
                prefix_tokens = default_prefix_tokens if value.ndim == 3 else 0
                return value, prefix_tokens, f"sequence[{index}]"
            if isinstance(value, dict):
                try:
                    tensor, prefix_tokens, source = extract_feature_tensor(
                        value,
                        default_prefix_tokens=default_prefix_tokens,
                    )
                except RuntimeError:
                    continue
                return tensor, prefix_tokens, f"sequence[{index}].{source}"

    raise RuntimeError(f"Unable to extract a feature tensor from encoder output type {type(raw_outputs).__name__}.")


def pool_features(
    features: torch.Tensor,
    *,
    pooling: str,
    prefix_tokens: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if features.ndim == 4:
        if pooling == "cls":
            raise RuntimeError("Requested --pooling cls but encoder produced a 2D feature map without a CLS token.")
        if pooling == "mean_tokens":
            flattened = features.flatten(2).transpose(1, 2)
            return flattened.mean(dim=1), {
                "pooling_applied": "mean_over_spatial_tokens",
                "raw_feature_shape": list(features.shape),
            }
        return features.mean(dim=(-2, -1)), {
            "pooling_applied": "global_average_pool",
            "raw_feature_shape": list(features.shape),
        }

    if features.ndim == 3:
        token_count = int(features.shape[1])
        usable_prefix_tokens = min(max(prefix_tokens, 0), token_count)
        if pooling == "cls":
            return features[:, 0, :], {
                "pooling_applied": "token_0",
                "raw_feature_shape": list(features.shape),
                "prefix_tokens_used": usable_prefix_tokens,
            }
        if pooling == "mean_tokens":
            tokens = features[:, usable_prefix_tokens:, :]
            if tokens.shape[1] == 0:
                raise RuntimeError(
                    "Requested --pooling mean_tokens but no non-prefix tokens were available after removing prefix tokens."
                )
            return tokens.mean(dim=1), {
                "pooling_applied": "mean_over_non_prefix_tokens",
                "raw_feature_shape": list(features.shape),
                "prefix_tokens_used": usable_prefix_tokens,
            }
        return features.mean(dim=1), {
            "pooling_applied": "mean_over_all_tokens",
            "raw_feature_shape": list(features.shape),
            "prefix_tokens_used": usable_prefix_tokens,
        }

    if features.ndim == 2:
        if pooling != "avg":
            raise RuntimeError(
                f"Requested --pooling {pooling}, but encoder returned already-pooled 2D embeddings. "
                f"Use --pooling avg or choose an encoder/backend that exposes tokens or feature maps."
            )
        return features, {
            "pooling_applied": "identity_already_pooled",
            "raw_feature_shape": list(features.shape),
        }

    raise RuntimeError(f"Unexpected feature tensor shape: {tuple(features.shape)}")


def embed_split(
    split: str,
    image_paths: list[Path],
    encoder_bundle: EncoderBundle,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pooling: str,
    fp16_on_cuda: bool,
) -> tuple[np.ndarray, list[str], list[dict[str, str]], dict[str, Any]]:
    dataloader = build_dataloader(
        image_paths,
        encoder_bundle.preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    all_embeddings: list[np.ndarray] = []
    embedded_paths: list[str] = []
    failed_images: list[dict[str, str]] = []
    total_batches = len(dataloader)
    feature_summary: dict[str, Any] = {}

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader, start=1):
            failed_images.extend(batch["errors"])
            pixel_values = batch["pixel_values"]
            if pixel_values is None:
                print(f"[warn] split={split} batch={batch_idx}/{total_batches} had 0 valid images")
                continue

            pixel_values = pixel_values.to(device, non_blocking=(device.type == "cuda"))
            if device.type == "cuda" and fp16_on_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    raw_outputs = encoder_bundle.forward_fn(pixel_values)
            else:
                raw_outputs = encoder_bundle.forward_fn(pixel_values)

            features, prefix_tokens, feature_source = extract_feature_tensor(
                raw_outputs,
                default_prefix_tokens=encoder_bundle.default_prefix_tokens,
            )
            pooled, pooling_meta = pool_features(
                features,
                pooling=pooling,
                prefix_tokens=prefix_tokens,
            )
            pooled = F.normalize(pooled, p=2, dim=1)

            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))
            embedded_paths.extend(batch["paths"])

            if not feature_summary:
                feature_summary = {
                    "feature_source": feature_source,
                    "pooling": pooling,
                    **pooling_meta,
                }

            if batch_idx == total_batches or batch_idx % 20 == 0:
                print(
                    f"[progress] split={split} batch={batch_idx}/{total_batches} "
                    f"embedded={len(embedded_paths)} failed={len(failed_images)}"
                )

    if not all_embeddings:
        raise RuntimeError(f"No embeddings were generated for split '{split}'.")

    embeddings_array = np.concatenate(all_embeddings, axis=0)
    if embeddings_array.ndim != 2:
        raise RuntimeError(f"Unexpected embedding shape for split '{split}': {embeddings_array.shape}")
    if embeddings_array.shape[0] != len(embedded_paths):
        raise RuntimeError(
            f"Embedding count mismatch for split '{split}': "
            f"embeddings={embeddings_array.shape[0]} paths={len(embedded_paths)}"
        )

    return embeddings_array, embedded_paths, failed_images, feature_summary


def build_experiment_slug(
    *,
    experiment_prefix: str,
    manifest_csv: Path,
    splits: list[str],
    backend: str,
    encoder_id: str,
    weights_label: str,
    resolved_input_size: tuple[int, int],
    pooling: str,
    fp16_on_cuda: bool,
    max_images_per_split: int | None,
    checkpoint_path: Path | None,
    revision: str | None,
) -> str:
    parts = [
        slugify(experiment_prefix, fallback="experiment"),
        f"manifest-{slugify(manifest_csv.stem, fallback='manifest')}",
        f"splits-{slugify('-'.join(splits), fallback='splits')}",
        f"backend-{slugify(backend, fallback='backend')}",
        f"encoder-{slugify(encoder_id, fallback='encoder')}",
        f"weights-{slugify(weights_label, fallback='weights')}",
        f"input-{resolved_input_size[0]}x{resolved_input_size[1]}",
        f"pool-{slugify(pooling, fallback='pool')}",
        f"norm-l2",
        f"precision-{'amp-fp16' if fp16_on_cuda else 'fp32'}",
    ]
    if max_images_per_split is not None:
        parts.append(f"maxper-{max_images_per_split}")
    if checkpoint_path is not None:
        parts.append(f"ckpt-{slugify(checkpoint_path.stem, fallback='checkpoint')}")
    if revision:
        parts.append(f"rev-{slugify(revision, fallback='revision')}")
    parts.append(f"ts-{utc_now_compact()}")
    return "__".join(parts)


def extract_experiment_number(name: str) -> int | None:
    match = re.match(r"^exp(\d+)(?:__|$)", name)
    if match is None:
        return None
    return int(match.group(1))


def next_experiment_number(experiments_root: Path) -> int:
    return experiment_layout.next_experiment_number(experiments_root)


def resolve_experiment_identity(
    *,
    experiments_root: Path,
    requested_name: str | None,
    generated_slug: str,
    overwrite: bool,
    id_width: int = DEFAULT_EXPERIMENT_ID_WIDTH,
) -> tuple[int, str, str, Path]:
    if requested_name:
        requested_name = requested_name.strip()
        if not requested_name:
            raise SystemExit("--experiment-name cannot be empty.")
    base_name = ensure_operation_prefix(requested_name or generated_slug)
    return experiment_layout.resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=base_name if requested_name else None,
        generated_slug=base_name,
        overwrite=overwrite,
        id_width=id_width,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate configurable image embeddings for NIH CXR14 train/val/test splits."
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
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
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=DEFAULT_SPLITS)
    parser.add_argument(
        "--encoder-backend",
        choices=["torchvision", "timm", "huggingface"],
        default=DEFAULT_ENCODER_BACKEND,
    )
    parser.add_argument("--encoder-id", "--model-id", dest="encoder_id", type=str, default=DEFAULT_ENCODER_ID)
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help=(
            "Backend-specific weights selector. torchvision: DEFAULT/NONE/or enum name. "
            "timm: DEFAULT/PRETRAINED/NONE. huggingface: ignored."
        ),
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--revision", type=str, default=None, help="Optional Hugging Face model revision.")
    parser.add_argument("--pooling", choices=["avg", "mean_tokens", "cls"], default=DEFAULT_POOLING)
    parser.add_argument(
        "--input-size",
        nargs="+",
        type=int,
        default=None,
        help="Pass one integer for square input or two integers for height width.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--extensions", nargs="+", default=DEFAULT_EXTENSIONS)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default=DEFAULT_DEVICE)
    parser.add_argument("--fp16-on-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--trust-manifest-paths",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip per-row filesystem existence checks and trust that manifest image_path entries resolve "
            "correctly under --data-root."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_csv = args.manifest_csv.resolve()
    data_root = args.data_root.resolve()
    experiments_root = args.experiments_root.resolve()
    splits = dedupe_preserve_order(list(args.splits))
    image_extensions = normalize_extensions(args.extensions)
    device = resolve_device(args.device)
    input_size = parse_input_size(args.input_size)

    if not manifest_csv.exists():
        raise SystemExit(f"manifest-csv not found: {manifest_csv}")

    encoder_bundle = build_encoder_bundle(
        backend=args.encoder_backend,
        encoder_id=args.encoder_id,
        weights_name=args.weights,
        input_size=input_size,
        checkpoint_path=args.checkpoint_path,
        revision=args.revision,
    )
    encoder_bundle.model.to(device)
    encoder_bundle.model.eval()

    experiment_slug = build_experiment_slug(
        experiment_prefix=args.experiment_prefix,
        manifest_csv=manifest_csv,
        splits=splits,
        backend=encoder_bundle.backend,
        encoder_id=encoder_bundle.encoder_id,
        weights_label=encoder_bundle.weights_label,
        resolved_input_size=encoder_bundle.resolved_input_size,
        pooling=args.pooling,
        fp16_on_cuda=bool(args.fp16_on_cuda),
        max_images_per_split=args.max_images_per_split,
        checkpoint_path=args.checkpoint_path,
        revision=args.revision,
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
        "data_root": str(data_root),
        "splits": splits,
        "encoder_backend": encoder_bundle.backend,
        "encoder_id": encoder_bundle.encoder_id,
        "weights": encoder_bundle.weights_label,
        "checkpoint_path": str(args.checkpoint_path.resolve()) if args.checkpoint_path else None,
        "revision": args.revision,
        "requested_input_size": list(input_size) if input_size is not None else None,
        "resolved_input_size": list(encoder_bundle.resolved_input_size),
        "pooling": args.pooling,
        "normalization": "l2",
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device_requested": args.device,
        "device_resolved": str(device),
        "fp16_on_cuda": bool(args.fp16_on_cuda),
        "trust_manifest_paths": bool(args.trust_manifest_paths),
        "max_images_per_split": args.max_images_per_split,
        "extensions": sorted(image_extensions),
        "argv": sys.argv,
        "encoder_build_meta": encoder_bundle.build_meta,
        "split_output_dirs": {split: str(experiment_dir / split) for split in splits},
    }
    write_json(experiment_dir / "experiment_meta.json", experiment_meta)

    print(
        f"[info] experiment_id={experiment_id} backend={encoder_bundle.backend} encoder={encoder_bundle.encoder_id} "
        f"weights={encoder_bundle.weights_label} pooling={args.pooling} "
        f"input_size={encoder_bundle.resolved_input_size} device={device}"
    )
    print(f"[info] experiment_dir={experiment_dir}")
    print(f"[info] manifest_csv={manifest_csv} data_root={data_root}")

    split_image_paths = load_manifest_split_image_paths(
        manifest_csv,
        data_root,
        splits,
        image_extensions,
        verify_exists=not bool(args.trust_manifest_paths),
        max_images=args.max_images_per_split,
    )

    for split in splits:
        output_dir = experiment_dir / split
        image_paths = split_image_paths[split]
        print(f"[info] split={split} images={len(image_paths)}")

        embeddings, embedded_paths, failed_images, feature_summary = embed_split(
            split,
            image_paths,
            encoder_bundle,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pooling=args.pooling,
            fp16_on_cuda=bool(args.fp16_on_cuda),
        )

        run_meta = {
            "run_date_utc": utc_now_iso(),
            "experiment_number": experiment_number,
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "experiment_dir": str(experiment_dir),
            "split": split,
            "manifest_csv": str(manifest_csv),
            "data_root": str(data_root),
            "encoder_backend": encoder_bundle.backend,
            "encoder_id": encoder_bundle.encoder_id,
            "weights": encoder_bundle.weights_label,
            "checkpoint_path": str(args.checkpoint_path.resolve()) if args.checkpoint_path else None,
            "revision": args.revision,
            "pooling": args.pooling,
            "normalization": "l2",
            "output_dir": str(output_dir),
            "num_input_images": len(image_paths),
            "num_embedded_images": int(embeddings.shape[0]),
            "num_failed_images": len(failed_images),
            "embedding_dim": int(embeddings.shape[1]),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device_requested": args.device,
            "device_resolved": str(device),
            "fp16_on_cuda": bool(args.fp16_on_cuda),
            "trust_manifest_paths": bool(args.trust_manifest_paths),
            "requested_input_size": list(input_size) if input_size is not None else None,
            "resolved_input_size": list(encoder_bundle.resolved_input_size),
            "max_images_per_split": args.max_images_per_split,
            "extensions": sorted(image_extensions),
            "feature_summary": feature_summary,
            "encoder_build_meta": encoder_bundle.build_meta,
        }
        save_split_outputs(
            output_dir,
            embeddings=embeddings,
            embedded_paths=embedded_paths,
            failed_images=failed_images,
            run_meta=run_meta,
        )
        print(f"[saved] split={split} output_dir={output_dir}")


if __name__ == "__main__":
    main()
