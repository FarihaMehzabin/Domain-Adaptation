#!/usr/bin/env python3
"""Generate Hugging Face vision-model embeddings for NIH CXR14 split CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'numpy'. Install: pip install -r requirements/hf_image_embeddings.txt"
    ) from exc

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'torch'. Install: pip install -r requirements/hf_image_embeddings.txt"
    ) from exc

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'pillow'. Install: pip install -r requirements/hf_image_embeddings.txt"
    ) from exc

try:
    from transformers import AutoImageProcessor, AutoModel
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'transformers'. Install: pip install -r requirements/hf_image_embeddings.txt"
    ) from exc


DEFAULT_DATA_ROOT = Path("/workspace/data")
DEFAULT_SPLIT_CSV_DIR = DEFAULT_DATA_ROOT / "nih_cxr14" / "splits"
DEFAULT_OUTPUT_ROOT = Path("/workspace/outputs/nih_cxr14_hf_image_embeddings")
DEFAULT_MODEL_ID = "facebook/dinov2-large"
DEFAULT_SPLITS = ["val", "test"]
DEFAULT_EXTENSIONS = [".png", ".jpg", ".jpeg"]
DEFAULT_POOLING = "auto"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICE = "auto"


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


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but CUDA is not available.")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit("Requested --device mps, but MPS is not available.")
    return torch.device(requested)


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate
    return data_root / candidate


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def load_split_image_paths(
    data_root: Path,
    split_csv_dir: Path,
    split: str,
    image_extensions: set[str],
    *,
    max_images: int | None,
) -> tuple[Path, list[Path]]:
    split_csv_path = split_csv_dir / f"{split}.csv"
    if not split_csv_path.exists():
        raise SystemExit(f"Split CSV not found: {split_csv_path}")

    image_paths: list[Path] = []
    with split_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("dataset") != "nih_cxr14":
                continue
            if row.get("split") and row["split"] != split:
                continue

            image_path = (row.get("image_path") or "").strip()
            if not image_path:
                continue

            resolved = resolve_image_path(data_root, image_path)
            if resolved.suffix.lower() not in image_extensions:
                continue

            image_paths.append(resolved)
            if max_images is not None and len(image_paths) >= max_images:
                break

    if not image_paths:
        ext_msg = ", ".join(sorted(image_extensions))
        raise SystemExit(
            f"No images found for split '{split}' in {split_csv_path} with extensions: {ext_msg}"
        )

    return split_csv_path, image_paths


class CXRImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], processor: Any) -> None:
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
            pixel_values = self.processor(images=rgb_image, return_tensors="pt")["pixel_values"].squeeze(0)
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


def pool_embeddings(model_outputs: Any, pooling: str) -> torch.Tensor:
    pooled = getattr(model_outputs, "pooler_output", None)
    if pooling == "pooler":
        if pooled is None:
            raise RuntimeError("Requested --pooling pooler but model output has no pooler_output.")
        return pooled

    last_hidden_state = model_outputs.last_hidden_state
    if pooling == "auto":
        if last_hidden_state.ndim == 3:
            return last_hidden_state[:, 0, :]
        if last_hidden_state.ndim == 4:
            if pooled is not None:
                return pooled
            return last_hidden_state.mean(dim=(-2, -1))
        raise RuntimeError(f"Unexpected last_hidden_state shape: {tuple(last_hidden_state.shape)}")

    if last_hidden_state.ndim == 4:
        if pooling == "cls":
            raise RuntimeError("Requested --pooling cls but model produced 2D feature maps (no CLS token).")
        return last_hidden_state.mean(dim=(-2, -1))
    if last_hidden_state.ndim == 3:
        if pooling == "mean":
            return last_hidden_state.mean(dim=1)
        return last_hidden_state[:, 0, :]
    raise RuntimeError(f"Unexpected last_hidden_state shape: {tuple(last_hidden_state.shape)}")


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


def build_dataloader(
    image_paths: list[Path],
    processor: Any,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = CXRImageDataset(image_paths, processor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_batch,
    )


def embed_split(
    split: str,
    image_paths: list[Path],
    processor: Any,
    model: Any,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pooling: str,
    fp16_on_cuda: bool,
) -> tuple[np.ndarray, list[str], list[dict[str, str]]]:
    dataloader = build_dataloader(
        image_paths,
        processor,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    all_embeddings: list[np.ndarray] = []
    embedded_paths: list[str] = []
    failed_images: list[dict[str, str]] = []
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            failed_images.extend(batch["errors"])

            pixel_values = batch["pixel_values"]
            if pixel_values is None:
                print(f"[warn] split={split} batch={batch_idx}/{total_batches} had 0 valid images")
                continue

            pixel_values = pixel_values.to(device, non_blocking=(device.type == "cuda"))

            if device.type == "cuda" and fp16_on_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(pixel_values=pixel_values)
            else:
                outputs = model(pixel_values=pixel_values)

            pooled = pool_embeddings(outputs, pooling)
            pooled = F.normalize(pooled, p=2, dim=1)

            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))
            embedded_paths.extend(batch["paths"])

            if batch_idx == total_batches or batch_idx % 20 == 0:
                print(
                    f"[progress] split={split} batch={batch_idx}/{total_batches} "
                    f"embedded={len(embedded_paths)} failed={len(failed_images)}"
                )

    if not all_embeddings:
        raise RuntimeError(
            f"No embeddings were generated for split '{split}'. Check failed_images.jsonl for load errors."
        )

    embeddings = np.concatenate(all_embeddings, axis=0)
    if embeddings.ndim != 2:
        raise RuntimeError(f"Unexpected embeddings shape for split '{split}': {embeddings.shape}")
    if embeddings.shape[0] != len(embedded_paths):
        raise RuntimeError(
            f"Embedding count mismatch for split '{split}': "
            f"embeddings={embeddings.shape[0]} paths={len(embedded_paths)}"
        )

    return embeddings, embedded_paths, failed_images


def save_split_outputs(
    output_dir: Path,
    *,
    embeddings: np.ndarray,
    embedded_paths: list[str],
    failed_images: list[dict[str, str]],
    run_meta: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "embeddings.npy"
    manifest_csv_path = output_dir / "image_manifest.csv"
    image_paths_txt_path = output_dir / "image_paths.txt"
    failures_path = output_dir / "failed_images.jsonl"
    run_meta_path = output_dir / "run_meta.json"

    np.save(embeddings_path, embeddings.astype(np.float32))
    save_manifest_csv(manifest_csv_path, embedded_paths)
    image_paths_txt_path.write_text("\n".join(embedded_paths) + "\n", encoding="utf-8")
    if failed_images:
        save_failed_jsonl(failures_path, failed_images)
    write_json(run_meta_path, run_meta)

    print(f"[saved] {embeddings_path}")
    print(f"[saved] {manifest_csv_path}")
    print(f"[saved] {image_paths_txt_path}")
    if failed_images:
        print(f"[saved] {failures_path}")
    print(f"[saved] {run_meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate image embeddings for NIH CXR14 split CSVs with a Hugging Face vision model."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-csv-dir", type=Path, default=DEFAULT_SPLIT_CSV_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=DEFAULT_SPLITS)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--pooling", type=str, choices=["auto", "cls", "mean", "pooler"], default=DEFAULT_POOLING)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-images", type=int, default=None, help="Maximum images to process per split.")
    parser.add_argument("--extensions", nargs="+", default=DEFAULT_EXTENSIONS)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default=DEFAULT_DEVICE)
    parser.add_argument("--fp16-on-cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = args.data_root.resolve()
    split_csv_dir = args.split_csv_dir.resolve()
    output_root = args.output_root.resolve()

    if not data_root.exists():
        raise SystemExit(f"data-root not found: {data_root}")
    if not split_csv_dir.exists():
        raise SystemExit(f"split-csv-dir not found: {split_csv_dir}")
    if not split_csv_dir.is_dir():
        raise SystemExit(f"split-csv-dir is not a directory: {split_csv_dir}")

    image_extensions = normalize_extensions(args.extensions)
    splits = dedupe_preserve_order(list(args.splits))
    device = resolve_device(args.device)

    print(f"[info] model={args.model_id} pooling={args.pooling} device={device}")
    print(f"[info] splits={','.join(splits)} output_root={output_root}")

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id)
    model.to(device)
    model.eval()

    for split in splits:
        split_csv_path, image_paths = load_split_image_paths(
            data_root,
            split_csv_dir,
            split,
            image_extensions,
            max_images=args.max_images,
        )

        print(f"[info] split={split} images={len(image_paths)} split_csv={split_csv_path}")

        embeddings, embedded_paths, failed_images = embed_split(
            split,
            image_paths,
            processor,
            model,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pooling=args.pooling,
            fp16_on_cuda=bool(args.fp16_on_cuda),
        )

        output_dir = output_root if split == "train" else output_root / split
        run_meta = {
            "run_date_utc": utc_now_iso(),
            "split": split,
            "split_csv_path": str(split_csv_path),
            "data_root": str(data_root),
            "model_id": args.model_id,
            "pooling": args.pooling,
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
            "max_images_per_split": args.max_images,
            "extensions": sorted(image_extensions),
        }
        save_split_outputs(
            output_dir,
            embeddings=embeddings,
            embedded_paths=embedded_paths,
            failed_images=failed_images,
            run_meta=run_meta,
        )


if __name__ == "__main__":
    main()
