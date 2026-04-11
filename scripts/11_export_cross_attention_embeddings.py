#!/usr/bin/env python3
"""Export split-aware fused embeddings from a trained cross-attention experiment."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'numpy'.") from exc

try:
    import torch
    from torch.utils.data import DataLoader
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'torch'.") from exc

from source_cross_attention_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DEVICE,
    DEFAULT_EXPERIMENTS_ROOT,
    DEFAULT_MANIFEST_CSV,
    DEFAULT_MAX_LENGTH,
    DEFAULT_NUM_WORKERS,
    DEFAULT_REPORTS_ROOT,
    CrossAttentionMultimodalModel,
    NIHMultimodalDataset,
    MultimodalCollator,
    build_image_encoder_bundle,
    build_text_encoder_bundle,
    get_autocast_context,
    load_manifest_records,
    move_batch_to_device,
    resolve_device,
    resolve_experiment_identity,
    sha256_file,
    slugify,
    utc_now_iso,
    write_json,
)

DEFAULT_OPERATION_LABEL = "source_cross_attention_embedding_export"
DEFAULT_LOG_EVERY_STEPS = 100


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
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=MultimodalCollator(tokenizer=tokenizer, max_length=max_length),
    )


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_trained_weights(model: CrossAttentionMultimodalModel, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Unexpected checkpoint payload type: {type(checkpoint)!r}")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise SystemExit(f"Checkpoint does not contain a state_dict: {checkpoint_path}")

    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise SystemExit(f"Unexpected checkpoint keys: {sorted(result.unexpected_keys)}")
    allowed_missing_prefixes = ("image_encoder.", "text_encoder.")
    invalid_missing = [
        key for key in result.missing_keys if not key.startswith(allowed_missing_prefixes)
    ]
    if invalid_missing:
        raise SystemExit(f"Missing required model keys after checkpoint load: {sorted(invalid_missing)}")
    return checkpoint


@torch.no_grad()
def export_split(
    *,
    split: str,
    loader: DataLoader,
    model: CrossAttentionMultimodalModel,
    device: torch.device,
    fp16_on_cuda: bool,
) -> tuple[np.ndarray, list[str], list[str]]:
    model.eval()
    embedding_chunks: list[np.ndarray] = []
    row_ids: list[str] = []
    image_paths: list[str] = []

    total_batches = len(loader)
    for step_idx, batch in enumerate(loader, start=1):
        encoded = batch["encoded"]
        pixel_values = batch["pixel_values"]
        if encoded is None or pixel_values is None:
            continue
        text_inputs = move_batch_to_device(encoded, device)
        pixel_values = pixel_values.to(device, non_blocking=(device.type == "cuda"))
        with get_autocast_context(device, fp16_on_cuda):
            embeddings = model.encode(
                pixel_values=pixel_values,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
        embedding_chunks.append(embeddings.detach().cpu().numpy().astype(np.float32))
        row_ids.extend(batch["row_ids"])
        image_paths.extend(batch["image_paths"])
        if step_idx == 1 or step_idx % DEFAULT_LOG_EVERY_STEPS == 0 or step_idx == total_batches:
            print(
                f"[export:{split}] step={step_idx}/{total_batches} rows={len(row_ids)}",
                flush=True,
            )

    if not embedding_chunks:
        raise SystemExit("No embeddings were produced for the requested split.")
    embeddings = np.ascontiguousarray(np.concatenate(embedding_chunks, axis=0).astype(np.float32))
    if embeddings.ndim != 2:
        raise SystemExit(f"Expected 2D exported embeddings, found shape {embeddings.shape}")
    if embeddings.shape[0] != len(row_ids) or embeddings.shape[0] != len(image_paths):
        raise SystemExit("Exported embedding rows do not align with sidecar rows.")
    return embeddings, row_ids, image_paths


def save_manifest_csv(path: Path, row_ids: list[str], image_paths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "row_id", "image_path"])
        for idx, (row_id, image_path) in enumerate(zip(row_ids, image_paths, strict=True)):
            writer.writerow([idx, row_id, image_path])


def save_split_outputs(
    output_dir: Path,
    *,
    embeddings: np.ndarray,
    row_ids: list[str],
    image_paths: list[str],
    run_meta: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings.astype(np.float32))
    save_manifest_csv(output_dir / "image_manifest.csv", row_ids, image_paths)
    (output_dir / "image_paths.txt").write_text("\n".join(image_paths) + "\n", encoding="utf-8")
    (output_dir / "row_ids.json").write_text(json.dumps(row_ids, indent=2) + "\n", encoding="utf-8")
    write_json(output_dir / "run_meta.json", run_meta)


def format_bash_command(argv: list[str]) -> str:
    return " \\\n  ".join(shlex.quote(part) for part in argv)


def build_recreation_report(*, experiment_dir: Path, config: dict[str, Any]) -> str:
    command = ["python", str(Path(__file__).resolve()), *config["argv"][1:]]
    return "\n".join(
        [
            "# Source Cross-Attention Embedding Export Recreation Report",
            "",
            "## Experiment",
            "",
            f"- Export directory: `{experiment_dir}`",
            f"- Training experiment: `{config['training_experiment_dir']}`",
            f"- Embedding dim: `{config['embedding_dim']}`",
            "",
            "## Exact Command",
            "",
            "```bash",
            format_bash_command(command),
            "```",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained source cross-attention model and export split-aware fused embeddings that match the "
            "existing retrieval pipeline format."
        )
    )
    parser.add_argument("--training-experiment-dir", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--trust-manifest-paths",
        action="store_true",
        help=(
            "Skip upfront per-row image/report existence checks during manifest load. "
            "Use this when the manifest paths are already known to be valid."
        ),
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--fp16-on-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0.")
    training_experiment_dir = args.training_experiment_dir.resolve()
    config_path = training_experiment_dir / "config.json"
    checkpoint_path = training_experiment_dir / "best.ckpt"
    training_config = read_json(config_path)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    manifest_csv = Path(args.manifest_csv or training_config["manifest_csv"]).resolve()
    data_root = Path(args.data_root or training_config["data_root"]).resolve()
    reports_root = Path(args.reports_root or training_config["reports_root"]).resolve()
    batch_size = int(args.batch_size or training_config["batch_size"])
    device = resolve_device(args.device)
    fp16_on_cuda = bool(args.fp16_on_cuda or training_config.get("fp16_on_cuda", False))

    trust_manifest_paths = bool(
        args.trust_manifest_paths or training_config.get("trust_manifest_paths", False)
    )
    print(f"[startup] device={device} training_experiment_dir={training_experiment_dir}", flush=True)
    print(
        f"[startup] loading manifest records verify_files={not trust_manifest_paths}",
        flush=True,
    )
    label_columns, records_by_split = load_manifest_records(
        manifest_csv,
        data_root=data_root,
        reports_root=reports_root,
        splits=["train", "val", "test"],
        max_samples_per_split=training_config.get("max_samples_per_split"),
        verify_files=not trust_manifest_paths,
    )
    print(
        "[startup] manifest loaded "
        f"train={len(records_by_split['train'])} val={len(records_by_split['val'])} test={len(records_by_split['test'])}",
        flush=True,
    )

    print("[startup] rebuilding frozen image encoder", flush=True)
    image_bundle = build_image_encoder_bundle()
    print("[startup] image encoder ready", flush=True)
    print("[startup] rebuilding frozen text encoder", flush=True)
    text_bundle = build_text_encoder_bundle(
        model_id=training_config["text_encoder"]["model_id"],
        tokenizer_id=training_config["text_encoder"].get("tokenizer_id"),
        revision=training_config["text_encoder"].get("revision"),
        trust_remote_code=bool(training_config["text_encoder"].get("trust_remote_code", False)),
        cache_dir=None,
    )
    print("[startup] text encoder ready", flush=True)

    print("[startup] initializing export model", flush=True)
    model = CrossAttentionMultimodalModel(
        image_encoder=image_bundle.model,
        text_encoder=text_bundle.model,
        text_hidden_size=int(training_config["text_encoder"]["hidden_size"]),
        legacy_text_feature_dim=int(
            training_config["text_encoder"].get("projected_embedding_size", text_bundle.projected_embedding_size)
        ),
        num_labels=len(label_columns),
        image_feature_dim=int(training_config["image_encoder"]["feature_dim"]),
        image_token_count=int(training_config["image_encoder"]["spatial_token_count"]),
        fusion_dim=int(training_config["fusion_dim"]),
        num_heads=int(training_config["fusion_heads"]),
        num_layers=int(training_config["fusion_layers"]),
        embedding_dim=int(training_config["embedding_dim"]),
        dropout=float(training_config["dropout"]),
        gated_hybrid=bool(training_config.get("gated_hybrid", False)),
    ).to(device)
    checkpoint = load_trained_weights(model, checkpoint_path)
    print("[startup] export model ready", flush=True)

    generated_slug = "__".join(
        [
            DEFAULT_OPERATION_LABEL,
            slugify(training_experiment_dir.name, fallback="training_root"),
            f"ed{int(training_config['embedding_dim'])}",
        ]
    )
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=args.experiments_root.resolve(),
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        operation_label=DEFAULT_OPERATION_LABEL,
        overwrite=bool(args.overwrite),
    )

    script_path = Path(__file__).resolve()
    export_config = {
        "argv": list(sys.argv),
        "run_date_utc": utc_now_iso(),
        "operation_label": DEFAULT_OPERATION_LABEL,
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "training_experiment_dir": str(training_experiment_dir),
        "training_config_path": str(config_path),
        "training_checkpoint_path": str(checkpoint_path),
        "manifest_csv": str(manifest_csv),
        "data_root": str(data_root),
        "reports_root": str(reports_root),
        "batch_size": batch_size,
        "num_workers": int(args.num_workers),
        "trust_manifest_paths": trust_manifest_paths,
        "device_requested": str(args.device),
        "device_resolved": str(device),
        "fp16_on_cuda": fp16_on_cuda,
        "max_length": int(training_config["max_length"]),
        "embedding_dim": int(training_config["embedding_dim"]),
        "gated_hybrid": bool(training_config.get("gated_hybrid", False)),
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split_inputs": {
            split: {
                "num_rows": len(records_by_split[split]),
                "first_row_id": records_by_split[split][0].row_id if records_by_split[split] else None,
            }
            for split in ("train", "val", "test")
        },
    }
    write_json(experiment_dir / "config.json", export_config)

    for split in ("train", "val", "test"):
        loader = build_dataloader(
            records_by_split[split],
            tokenizer=text_bundle.tokenizer,
            image_transform=image_bundle.preprocess,
            batch_size=batch_size,
            num_workers=args.num_workers,
            max_length=int(training_config["max_length"]),
            text_prefix=str(training_config.get("text_prefix", "")),
            text_suffix=str(training_config.get("text_suffix", "")),
            normalize_whitespace=bool(training_config.get("normalize_whitespace", True)),
            device=device,
        )
        embeddings, row_ids, image_paths = export_split(
            split=split,
            loader=loader,
            model=model,
            device=device,
            fp16_on_cuda=fp16_on_cuda,
        )
        run_meta = {
            "split": split,
            "run_date_utc": utc_now_iso(),
            "num_rows": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "training_experiment_dir": str(training_experiment_dir),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "model_config": checkpoint.get("model_config", training_config),
        }
        save_split_outputs(
            experiment_dir / split,
            embeddings=embeddings,
            row_ids=row_ids,
            image_paths=image_paths,
            run_meta=run_meta,
        )
        print(f"[export] split={split} rows={embeddings.shape[0]} dim={embeddings.shape[1]}")

    experiment_meta = {
        "argv": list(sys.argv),
        "run_date_utc": utc_now_iso(),
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "operation_label": DEFAULT_OPERATION_LABEL,
        "training_experiment_dir": str(training_experiment_dir),
        "training_checkpoint_path": str(checkpoint_path),
        "manifest_csv": str(manifest_csv),
        "data_root": str(data_root),
        "reports_root": str(reports_root),
        "device_resolved": str(device),
        "embedding_dim": int(training_config["embedding_dim"]),
        "split_output_dirs": {split: str(experiment_dir / split) for split in ("train", "val", "test")},
    }
    write_json(experiment_dir / "experiment_meta.json", experiment_meta)
    (experiment_dir / "recreation_report.md").write_text(
        build_recreation_report(experiment_dir=experiment_dir, config=export_config),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
