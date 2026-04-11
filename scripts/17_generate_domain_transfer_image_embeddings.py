#!/usr/bin/env python3
"""Generate D0/D1/D2 image embeddings from a torch image encoder on the common transfer manifest."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_common_labels_pilot5h.csv")
DEFAULT_DATA_ROOT = Path("/workspace")
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_OPERATION_LABEL = "embedding_generation"
DEFAULT_ENCODER_BACKEND = "torchvision"
DEFAULT_ENCODER_ID = "resnet50"
DEFAULT_WEIGHTS = "DEFAULT"
DEFAULT_POOLING = "avg"
DEFAULT_BATCH_SIZE = 256
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICE = "auto"
DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_EXPERIMENT_ID_WIDTH = 4
LEGACY_HELPER_SCRIPT = Path("/workspace/scripts/01_generate_nih_split_image_embeddings.py")


@dataclass(frozen=True)
class ManifestRow:
    domain: str
    dataset: str
    split: str
    row_id: str
    image_path: str
    study_id: str
    patient_id: str
    label_vector: tuple[float, ...]
    resolved_path: Path


class ManifestImageDataset(Dataset):
    def __init__(self, rows: list[ManifestRow], transform: Any) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        try:
            with Image.open(row.resolved_path) as image:
                rgb = image.convert("RGB")
            pixel_values = self.transform(rgb)
            return {
                "row": row,
                "pixel_values": pixel_values,
                "error": None,
            }
        except Exception as exc:
            return {
                "row": row,
                "pixel_values": None,
                "error": str(exc),
            }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[ManifestRow] = []
    tensors: list[torch.Tensor] = []
    errors: list[dict[str, str]] = []
    for item in items:
        row = item["row"]
        pixel_values = item.get("pixel_values")
        if pixel_values is None:
            errors.append(
                {
                    "row_id": row.row_id,
                    "domain": row.domain,
                    "split": row.split,
                    "image_path": row.image_path,
                    "error": str(item.get("error") or "unknown_error"),
                }
            )
            continue
        rows.append(row)
        tensors.append(pixel_values)
    stacked = torch.stack(tensors, dim=0) if tensors else None
    return {
        "rows": rows,
        "pixel_values": stacked,
        "errors": errors,
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now_iso()}] {message}", flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def save_json_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def save_split_manifest(path: Path, rows: list[ManifestRow], label_names: list[str]) -> None:
    fieldnames = [
        "row_index",
        "domain",
        "dataset",
        "split",
        "row_id",
        "study_id",
        "patient_id",
        "image_path",
    ] + [f"label_{label}" for label in label_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            payload = {
                "row_index": index,
                "domain": row.domain,
                "dataset": row.dataset,
                "split": row.split,
                "row_id": row.row_id,
                "study_id": row.study_id,
                "patient_id": row.patient_id,
                "image_path": row.image_path,
            }
            for label_name, value in zip(label_names, row.label_vector):
                payload[f"label_{label_name}"] = int(value)
            writer.writerow(payload)


def load_legacy_helpers() -> Any:
    if not LEGACY_HELPER_SCRIPT.exists():
        raise SystemExit(f"Legacy helper script not found: {LEGACY_HELPER_SCRIPT}")
    module_name = "legacy_image_embeddings"
    spec = importlib.util.spec_from_file_location(module_name, LEGACY_HELPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load helper module from {LEGACY_HELPER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate domain-aware image embeddings for NIH/CheXpert/MIMIC transfer runs "
            "using the same torch encoder path as the existing NIH ResNet exporter."
        )
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--encoder-backend",
        choices=("torchvision", "timm", "huggingface"),
        default=DEFAULT_ENCODER_BACKEND,
    )
    parser.add_argument("--encoder-id", type=str, default=DEFAULT_ENCODER_ID)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--pooling", choices=("avg", "cls", "mean_tokens"), default=DEFAULT_POOLING)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--fp16-on-cuda", action="store_true")
    parser.add_argument("--input-size", type=int, nargs="*", default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_label_names(fieldnames: list[str]) -> list[str]:
    return [field.removeprefix("label_") for field in fieldnames if field.startswith("label_")]


def candidate_image_paths(data_root: Path, image_path: str) -> list[Path]:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return [candidate]
    roots = []
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


def load_manifest_rows(
    manifest_csv: Path,
    data_root: Path,
    *,
    max_rows_per_split: int | None,
) -> tuple[list[str], dict[tuple[str, str], list[ManifestRow]]]:
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV not found: {manifest_csv}")

    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest CSV has no header: {manifest_csv}")
        fieldnames = list(reader.fieldnames)
        label_names = normalize_label_names(fieldnames)
        if not label_names:
            raise SystemExit("Manifest must contain at least one label_* column.")

        grouped_rows: dict[tuple[str, str], list[ManifestRow]] = {}
        failures: list[str] = []
        for raw_row in reader:
            domain = str(raw_row.get("domain") or "").strip()
            split = str(raw_row.get("split") or "").strip()
            if not domain or not split:
                continue
            key = (domain, split)
            if max_rows_per_split is not None and len(grouped_rows.get(key, [])) >= max_rows_per_split:
                continue

            image_path = str(raw_row.get("image_path") or "").strip()
            if not image_path or Path(image_path).suffix.lower() not in DEFAULT_EXTENSIONS:
                continue
            try:
                resolved_path = resolve_manifest_image_path(data_root, image_path)
            except FileNotFoundError as exc:
                if len(failures) < 10:
                    failures.append(str(exc))
                continue

            row_id = str(raw_row.get("row_id") or Path(image_path).stem).strip() or Path(image_path).stem
            study_id = str(raw_row.get("study_id") or "NA").strip() or "NA"
            patient_id = str(raw_row.get("patient_id") or "NA").strip() or "NA"
            labels = tuple(float(str(raw_row.get(f"label_{label_name}") or "0").strip()) for label_name in label_names)
            row = ManifestRow(
                domain=domain,
                dataset=str(raw_row.get("dataset") or "unknown").strip() or "unknown",
                split=split,
                row_id=row_id,
                image_path=image_path,
                study_id=study_id,
                patient_id=patient_id,
                label_vector=labels,
                resolved_path=resolved_path,
            )
            grouped_rows.setdefault(key, []).append(row)

    if not grouped_rows:
        raise SystemExit(f"No usable rows were loaded from {manifest_csv}")
    if failures:
        sample = "\n".join(failures)
        raise SystemExit(
            "Some manifest images could not be resolved. Fix the data root before continuing.\n"
            f"Examples:\n{sample}"
        )
    return label_names, grouped_rows


def build_dataloader(rows: list[ManifestRow], transform: Any, *, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    dataset = ManifestImageDataset(rows, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_batch,
    )


def build_experiment_slug(
    *,
    encoder_backend: str,
    encoder_id: str,
    weights_label: str,
    pooling: str,
    manifest_csv: Path,
) -> str:
    manifest_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", manifest_csv.stem).strip("-._") or "manifest"
    return "__".join(
        [
            DEFAULT_OPERATION_LABEL,
            "domain_transfer",
            encoder_backend,
            encoder_id,
            weights_label.lower(),
            pooling,
            manifest_slug,
        ]
    )


def append_index_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "study_id",
        "domain",
        "split",
        "dataset",
        "row_id",
        "image_path",
        "embedding_path",
        "embedding_row",
        "label_vector",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0.")
    if args.max_rows_per_split is not None and args.max_rows_per_split <= 0:
        raise SystemExit("--max-rows-per-split must be positive.")

    legacy = load_legacy_helpers()
    device = legacy.resolve_device(args.device)
    input_size = legacy.parse_input_size(args.input_size)

    manifest_csv = args.manifest_csv.resolve()
    data_root = args.data_root.resolve()
    experiments_root = args.experiments_root.resolve()
    label_names, grouped_rows = load_manifest_rows(
        manifest_csv,
        data_root,
        max_rows_per_split=args.max_rows_per_split,
    )

    encoder_bundle = legacy.build_encoder_bundle(
        backend=args.encoder_backend,
        encoder_id=args.encoder_id,
        weights_name=args.weights,
        input_size=input_size,
        checkpoint_path=args.checkpoint_path,
        revision=args.revision,
    )
    encoder_bundle.model.to(device)
    encoder_bundle.model.eval()

    generated_slug = build_experiment_slug(
        encoder_backend=args.encoder_backend,
        encoder_id=args.encoder_id,
        weights_label=encoder_bundle.weights_label,
        pooling=args.pooling,
        manifest_csv=manifest_csv,
    )
    experiment_number, experiment_id, experiment_name, experiment_dir = legacy.resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=bool(args.overwrite),
        id_width=DEFAULT_EXPERIMENT_ID_WIDTH,
    )

    config = {
        "argv": list(sys.argv),
        "run_date_utc": utc_now_iso(),
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "manifest_csv": str(manifest_csv),
        "data_root": str(data_root),
        "encoder_backend": encoder_bundle.backend,
        "encoder_id": encoder_bundle.encoder_id,
        "weights_label": encoder_bundle.weights_label,
        "pooling": args.pooling,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "device": str(device),
        "fp16_on_cuda": bool(args.fp16_on_cuda),
        "resolved_input_size": list(encoder_bundle.resolved_input_size),
        "label_names": label_names,
        "max_rows_per_split": int(args.max_rows_per_split) if args.max_rows_per_split is not None else None,
    }
    write_json(experiment_dir / "config.json", config)
    (experiment_dir / "label_names.json").write_text(json.dumps(label_names, indent=2), encoding="utf-8")

    log(
        f"[startup] manifest_csv={manifest_csv} experiment_dir={experiment_dir} "
        f"encoder={encoder_bundle.backend}:{encoder_bundle.encoder_id} weights={encoder_bundle.weights_label} "
        f"pooling={args.pooling} batch_size={args.batch_size} device={device}"
    )

    index_rows: list[dict[str, Any]] = []
    split_meta: dict[str, Any] = {}
    split_summary_lines: list[str] = []

    for key in sorted(grouped_rows):
        domain, split = key
        rows = grouped_rows[key]
        split_dir = experiment_dir / domain / split
        split_dir.mkdir(parents=True, exist_ok=True)
        dataloader = build_dataloader(
            rows,
            encoder_bundle.preprocess,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        log(f"[split-start] domain={domain} split={split} total_rows={len(rows)}")
        all_embeddings: list[np.ndarray] = []
        successful_rows: list[ManifestRow] = []
        failed_rows: list[dict[str, str]] = []
        feature_summary: dict[str, Any] = {}

        with torch.inference_mode():
            for batch_index, batch in enumerate(dataloader, start=1):
                batch_start = (batch_index - 1) * args.batch_size + 1
                batch_end = min(len(rows), batch_start + args.batch_size - 1)
                log(
                    f"[batch-start] domain={domain} split={split} rows={batch_start}-{batch_end} "
                    f"batch_size={batch_end - batch_start + 1}"
                )
                failed_rows.extend(batch["errors"])
                pixel_values = batch["pixel_values"]
                valid_rows = batch["rows"]
                if pixel_values is None or not valid_rows:
                    log(
                        f"[batch-done] domain={domain} split={split} rows={batch_start}-{batch_end} "
                        f"embedded=0 successful=0 failed={len(batch['errors'])}"
                    )
                    continue

                pixel_values = pixel_values.to(device, non_blocking=(device.type == "cuda"))
                if device.type == "cuda" and args.fp16_on_cuda:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        raw_outputs = encoder_bundle.forward_fn(pixel_values)
                else:
                    raw_outputs = encoder_bundle.forward_fn(pixel_values)

                features, prefix_tokens, feature_source = legacy.extract_feature_tensor(
                    raw_outputs,
                    default_prefix_tokens=encoder_bundle.default_prefix_tokens,
                )
                pooled, pooling_meta = legacy.pool_features(
                    features,
                    pooling=args.pooling,
                    prefix_tokens=prefix_tokens,
                )
                pooled = F.normalize(pooled, p=2, dim=1)
                array = pooled.detach().cpu().numpy().astype(np.float32, copy=False)

                all_embeddings.append(array)
                successful_rows.extend(valid_rows)
                if not feature_summary:
                    feature_summary = {
                        "feature_source": feature_source,
                        **pooling_meta,
                    }

                log(
                    f"[batch-done] domain={domain} split={split} rows={batch_start}-{batch_end} "
                    f"embedded={int(array.shape[0])} successful={len(valid_rows)} failed={len(batch['errors'])}"
                )
                log(
                    f"[progress] domain={domain} split={split} processed={len(successful_rows)}/{len(rows)} "
                    f"embedded={len(successful_rows)} failed={len(failed_rows)}"
                )

        if not all_embeddings:
            raise SystemExit(f"No embeddings were produced for domain={domain} split={split}.")

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32, copy=False)
        if embeddings.shape[0] != len(successful_rows):
            raise SystemExit(
                f"Embedding row count mismatch for domain={domain} split={split}: "
                f"embeddings={embeddings.shape[0]} rows={len(successful_rows)}"
            )

        np.save(split_dir / "embeddings.npy", embeddings)
        save_json_list(split_dir / "row_ids.json", [row.row_id for row in successful_rows])
        save_json_list(split_dir / "study_ids.json", [row.study_id for row in successful_rows])
        (split_dir / "image_paths.txt").write_text(
            "\n".join(row.image_path for row in successful_rows) + "\n",
            encoding="utf-8",
        )
        save_split_manifest(split_dir / "image_manifest.csv", successful_rows, label_names)
        if failed_rows:
            append_jsonl(split_dir / "failed_rows.jsonl", failed_rows)
        relative_embedding_path = (split_dir / "embeddings.npy").relative_to(experiment_dir)
        log(
            f"[split-done] domain={domain} split={split} embedded_rows={len(successful_rows)} "
            f"failed_rows={len(failed_rows)} embedding_path={relative_embedding_path}"
        )

        for row_index, row in enumerate(successful_rows):
            index_rows.append(
                {
                    "study_id": row.study_id,
                    "domain": row.domain,
                    "split": row.split,
                    "dataset": row.dataset,
                    "row_id": row.row_id,
                    "image_path": row.image_path,
                    "embedding_path": str(relative_embedding_path),
                    "embedding_row": row_index,
                    "label_vector": json.dumps(list(row.label_vector)),
                }
            )

        run_meta = {
            "run_date_utc": utc_now_iso(),
            "domain": domain,
            "split": split,
            "num_requested_rows": len(rows),
            "num_embedded_rows": len(successful_rows),
            "num_failed_rows": len(failed_rows),
            "embedding_shape": list(embeddings.shape),
            "encoder_backend": encoder_bundle.backend,
            "encoder_id": encoder_bundle.encoder_id,
            "weights_label": encoder_bundle.weights_label,
            "pooling": args.pooling,
            "resolved_input_size": list(encoder_bundle.resolved_input_size),
            "feature_summary": feature_summary,
            "embeddings_path": str(split_dir / "embeddings.npy"),
            "row_ids_path": str(split_dir / "row_ids.json"),
            "image_paths_path": str(split_dir / "image_paths.txt"),
            "label_names": label_names,
        }
        write_json(split_dir / "run_meta.json", run_meta)
        split_meta[f"{domain}/{split}"] = run_meta
        split_summary_lines.append(
            f"- `{domain}/{split}`: {len(successful_rows)} rows, shape={list(embeddings.shape)}"
        )

    append_index_csv(experiment_dir / "embedding_index.csv", index_rows)
    write_json(
        experiment_dir / "experiment_meta.json",
        {
            "run_date_utc": utc_now_iso(),
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "operation_label": DEFAULT_OPERATION_LABEL,
            "manifest_csv": str(manifest_csv),
            "embedding_root": str(experiment_dir),
            "label_names": label_names,
            "split_meta": split_meta,
        },
    )
    recreation_report_lines = [
        "# Recreation Report",
        "",
        f"- Experiment: `{experiment_name}`",
        f"- Manifest: `{manifest_csv}`",
        f"- Encoder: `{encoder_bundle.backend}:{encoder_bundle.encoder_id}`",
        f"- Weights: `{encoder_bundle.weights_label}`",
        f"- Pooling: `{args.pooling}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Device: `{device}`",
        "",
        "## Split Summary",
        *split_summary_lines,
    ]
    (experiment_dir / "recreation_report.md").write_text("\n".join(recreation_report_lines) + "\n", encoding="utf-8")
    log(
        f"[done] experiment_dir={experiment_dir} index_rows={len(index_rows)} "
        f"encoder={encoder_bundle.backend}:{encoder_bundle.encoder_id} pooling={args.pooling}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
