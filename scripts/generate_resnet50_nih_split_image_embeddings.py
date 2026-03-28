#!/usr/bin/env python3
"""Generate split-aware NIH CXR14 ResNet50 embeddings for the fused pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'torch'.") from exc

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'pillow'.") from exc

try:
    from torchvision.models import ResNet50_Weights, resnet50
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'torchvision'.") from exc


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_nih_cxr14 .csv")
DEFAULT_DATA_ROOT = Path("/workspace")
DEFAULT_OUTPUT_ROOT = Path("/workspace/image_embeddings/resnet50")
DEFAULT_SPLITS = ["train", "val", "test"]
DEFAULT_EXTENSIONS = [".png", ".jpg", ".jpeg"]
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICE = "auto"
DEFAULT_WEIGHTS = "DEFAULT"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def resolve_weights(weights_name: str) -> ResNet50_Weights:
    try:
        return ResNet50_Weights[weights_name]
    except KeyError as exc:
        valid = ", ".join(ResNet50_Weights.__members__.keys())
        raise SystemExit(f"Unsupported --weights value '{weights_name}'. Valid values: {valid}") from exc


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


def resolve_manifest_image_path(data_root: Path, image_path: str) -> Path:
    candidates = candidate_image_paths(data_root, image_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{image_path} not found. Tried: {attempted}")


def load_split_image_paths(
    manifest_csv: Path,
    data_root: Path,
    split: str,
    image_extensions: set[str],
    *,
    max_images: int | None,
) -> list[Path]:
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV not found: {manifest_csv}")

    resolved_paths: list[Path] = []
    missing_examples: list[str] = []

    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"image_path", "split"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise SystemExit(f"Manifest CSV must contain columns: {sorted(required_columns)}")

        for row in reader:
            if row.get("dataset") and row["dataset"] != "nih_cxr14":
                continue
            if row["split"] != split:
                continue

            image_path = (row.get("image_path") or "").strip()
            if not image_path:
                continue
            if Path(image_path).suffix.lower() not in image_extensions:
                continue

            try:
                resolved = resolve_manifest_image_path(data_root, image_path)
            except FileNotFoundError as exc:
                if len(missing_examples) < 5:
                    missing_examples.append(str(exc))
                continue

            resolved_paths.append(resolved)
            if max_images is not None and len(resolved_paths) >= max_images:
                break

    if not resolved_paths:
        sample = "\n".join(missing_examples) if missing_examples else "No matching rows found in manifest."
        raise SystemExit(
            f"No usable images found for split '{split}'. Check --data-root.\n"
            f"Examples:\n{sample}"
        )

    if missing_examples:
        sample = "\n".join(missing_examples)
        raise SystemExit(
            f"Found split '{split}' rows in {manifest_csv}, but some images could not be resolved.\n"
            f"Pass the correct --data-root.\nExamples:\n{sample}"
        )

    return resolved_paths


class CXRImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform: Any) -> None:
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
    transform: Any,
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


def build_embedding_model(weights_name: str) -> tuple[nn.Module, Any]:
    weights = resolve_weights(weights_name)
    backbone = resnet50(weights=weights)
    transform = weights.transforms()
    model = nn.Sequential(*list(backbone.children())[:-1])
    return model, transform


def embed_split(
    split: str,
    image_paths: list[Path],
    transform: Any,
    model: nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    fp16_on_cuda: bool,
) -> tuple[np.ndarray, list[str], list[dict[str, str]]]:
    dataloader = build_dataloader(
        image_paths,
        transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    all_embeddings: list[np.ndarray] = []
    embedded_paths: list[str] = []
    failed_images: list[dict[str, str]] = []
    total_batches = len(dataloader)

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
                    outputs = model(pixel_values)
            else:
                outputs = model(pixel_values)

            embeddings = outputs.flatten(1)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
            embedded_paths.extend(batch["paths"])

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

    return embeddings_array, embedded_paths, failed_images


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


def split_output_exists(output_dir: Path) -> bool:
    return (output_dir / "embeddings.npy").exists() and (output_dir / "image_paths.txt").exists()


def bootstrap_legacy_train_root(output_root: Path, manifest_csv: Path, overwrite: bool) -> bool:
    legacy_files = [
        output_root / "embeddings.npy",
        output_root / "image_manifest.csv",
        output_root / "image_paths.txt",
        output_root / "run_meta.json",
    ]
    if not all(path.exists() for path in legacy_files):
        return False

    train_output_dir = output_root / "train"
    if split_output_exists(train_output_dir) and not overwrite:
        return True

    train_output_dir.mkdir(parents=True, exist_ok=True)
    for file_path in legacy_files:
        destination = train_output_dir / file_path.name
        shutil.copy2(file_path, destination)

    failed_images_path = output_root / "failed_images.jsonl"
    if failed_images_path.exists():
        shutil.copy2(failed_images_path, train_output_dir / failed_images_path.name)

    legacy_meta = json.loads((output_root / "run_meta.json").read_text(encoding="utf-8"))
    legacy_meta["run_date_utc"] = utc_now_iso()
    legacy_meta["split"] = "train"
    legacy_meta["manifest_csv"] = str(manifest_csv)
    legacy_meta["output_dir"] = str(train_output_dir)
    legacy_meta["legacy_source_dir"] = str(output_root)
    write_json(train_output_dir / "run_meta.json", legacy_meta)
    print(f"[info] bootstrapped legacy train embeddings into {train_output_dir}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ResNet50 image embeddings for NIH CXR14 train/val/test splits."
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=DEFAULT_SPLITS)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--extensions", nargs="+", default=DEFAULT_EXTENSIONS)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default=DEFAULT_DEVICE)
    parser.add_argument("--fp16-on-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-legacy-train-root", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_csv = args.manifest_csv.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    splits = dedupe_preserve_order(list(args.splits))
    image_extensions = normalize_extensions(args.extensions)
    device = resolve_device(args.device)

    if not manifest_csv.exists():
        raise SystemExit(f"manifest-csv not found: {manifest_csv}")
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] model=torchvision/resnet50 weights={args.weights} device={device}")
    print(f"[info] manifest_csv={manifest_csv} data_root={data_root} output_root={output_root}")

    model: nn.Module | None = None
    transform: Any = None

    for split in splits:
        output_dir = output_root / split
        if split == "train" and args.bootstrap_legacy_train_root:
            if bootstrap_legacy_train_root(output_root, manifest_csv, overwrite=bool(args.overwrite)):
                if not args.overwrite:
                    continue

        if split_output_exists(output_dir) and not args.overwrite:
            print(f"[info] split={split} already exists at {output_dir}; skipping")
            continue

        image_paths = load_split_image_paths(
            manifest_csv,
            data_root,
            split,
            image_extensions,
            max_images=args.max_images_per_split,
        )
        print(f"[info] split={split} images={len(image_paths)}")

        if model is None:
            model, transform = build_embedding_model(args.weights)
            model.to(device)
            model.eval()

        embeddings, embedded_paths, failed_images = embed_split(
            split,
            image_paths,
            transform,
            model,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fp16_on_cuda=bool(args.fp16_on_cuda),
        )

        run_meta = {
            "run_date_utc": utc_now_iso(),
            "split": split,
            "manifest_csv": str(manifest_csv),
            "data_root": str(data_root),
            "model_id": "torchvision/resnet50",
            "resnet_weights": args.weights,
            "pooling": "avgpool",
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
            "max_images_per_split": args.max_images_per_split,
            "extensions": sorted(image_extensions),
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
