#!/usr/bin/env python3
"""Benchmark torch image-embedding batch sizes on a single domain/split batch."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import experiment_layout
import torch
import torch.nn.functional as F


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest/manifest_common_labels_pilot5h.csv")
DEFAULT_DATA_ROOT = Path("/workspace")
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_OPERATION_LABEL = "torch_image_batch_sweep"
DEFAULT_EXPERIMENT_ID_WIDTH = 4
DEFAULT_DOMAIN = "d0_nih"
DEFAULT_SPLIT = "train"
DEFAULT_ENCODER_BACKEND = "torchvision"
DEFAULT_ENCODER_ID = "resnet50"
DEFAULT_WEIGHTS = "DEFAULT"
DEFAULT_POOLING = "avg"
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICE = "auto"
DEFAULT_CANDIDATES = (256, 512, 768, 1024, 1280, 1536, 1792, 2048)
EXPORTER_HELPER_SCRIPT = Path("/workspace/scripts/17_generate_domain_transfer_image_embeddings.py")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now_iso()}] {message}", flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def slugify(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or fallback


def ensure_operation_prefix(name: str, operation_label: str = DEFAULT_OPERATION_LABEL) -> str:
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


def extract_experiment_number(name: str) -> int | None:
    if not name.startswith("exp"):
        return None
    prefix = name.split("__", 1)[0]
    digits = prefix.removeprefix("exp")
    if not digits.isdigit():
        return None
    return int(digits)


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
    base_name = ensure_operation_prefix((requested_name or "").strip() or generated_slug)
    return experiment_layout.resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=base_name if requested_name else None,
        generated_slug=base_name,
        overwrite=overwrite,
        id_width=id_width,
    )


def load_exporter_helpers() -> Any:
    if not EXPORTER_HELPER_SCRIPT.exists():
        raise SystemExit(f"Exporter helper script not found: {EXPORTER_HELPER_SCRIPT}")
    module_name = "domain_transfer_image_export_helpers"
    spec = importlib.util.spec_from_file_location(module_name, EXPORTER_HELPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load helper module from {EXPORTER_HELPER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_experiment_slug(domain: str, split: str, encoder_backend: str, encoder_id: str, pooling: str) -> str:
    return "__".join(
        [
            DEFAULT_OPERATION_LABEL,
            domain,
            split,
            encoder_backend,
            encoder_id,
            pooling,
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the largest successful torch image-embedding batch size on a deterministic domain/split batch."
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--domain", type=str, default=DEFAULT_DOMAIN)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--encoder-backend",
        choices=("torchvision", "timm", "huggingface"),
        default=DEFAULT_ENCODER_BACKEND,
    )
    parser.add_argument("--encoder-id", type=str, default=DEFAULT_ENCODER_ID)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--pooling", choices=("avg", "cls", "mean_tokens"), default=DEFAULT_POOLING)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--fp16-on-cuda", action="store_true")
    parser.add_argument("--input-size", type=int, nargs="*", default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--candidate-batch-sizes", type=int, nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = sorted(set(int(value) for value in args.candidate_batch_sizes if int(value) > 0))
    if not candidates:
        raise SystemExit("Provide at least one positive --candidate-batch-sizes value.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0.")

    helper_module = load_exporter_helpers()
    legacy = helper_module.load_legacy_helpers()
    device = legacy.resolve_device(args.device)
    input_size = legacy.parse_input_size(args.input_size)

    manifest_csv = args.manifest_csv.resolve()
    data_root = args.data_root.resolve()
    experiments_root = args.experiments_root.resolve()
    label_names, grouped_rows = helper_module.load_manifest_rows(
        manifest_csv,
        data_root,
        max_rows_per_split=max(candidates),
    )
    del label_names

    key = (args.domain, args.split)
    rows = grouped_rows.get(key)
    if rows is None or len(rows) < max(candidates):
        found = 0 if rows is None else len(rows)
        raise SystemExit(
            f"Need at least {max(candidates)} rows for domain={args.domain} split={args.split}, found {found}."
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
        args.domain,
        args.split,
        encoder_bundle.backend,
        encoder_bundle.encoder_id,
        args.pooling,
    )
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
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
        "domain": args.domain,
        "split": args.split,
        "encoder_backend": encoder_bundle.backend,
        "encoder_id": encoder_bundle.encoder_id,
        "weights_label": encoder_bundle.weights_label,
        "pooling": args.pooling,
        "num_workers": int(args.num_workers),
        "device": str(device),
        "fp16_on_cuda": bool(args.fp16_on_cuda),
        "resolved_input_size": list(encoder_bundle.resolved_input_size),
        "candidate_batch_sizes": candidates,
    }
    write_json(experiment_dir / "config.json", config)
    (experiment_dir / "selected_rows.txt").write_text(
        "\n".join(f"{index}\t{row.row_id}\t{row.image_path}" for index, row in enumerate(rows)),
        encoding="utf-8",
    )

    log(
        f"[startup] manifest_csv={manifest_csv} experiment_dir={experiment_dir} "
        f"domain={args.domain} split={args.split} candidates={candidates} "
        f"encoder={encoder_bundle.backend}:{encoder_bundle.encoder_id} weights={encoder_bundle.weights_label}"
    )

    highest_successful: int | None = None
    results: list[dict[str, Any]] = []
    for batch_size in candidates:
        log(f"[batch-test-start] batch_size={batch_size}")
        dataloader = helper_module.build_dataloader(
            rows[:batch_size],
            encoder_bundle.preprocess,
            batch_size=batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        status = "success"
        error_type = None
        error_message = None
        embedding_shape: list[int] | None = None
        effective_rows = 0
        peak_memory_mb: float | None = None
        elapsed_seconds = None
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            start_time = time.perf_counter()
            batch = next(iter(dataloader))
            pixel_values = batch["pixel_values"]
            valid_rows = batch["rows"]
            if pixel_values is None or not valid_rows:
                raise RuntimeError("No valid batch rows were produced by the dataloader.")
            pixel_values = pixel_values.to(device, non_blocking=(device.type == "cuda"))
            with torch.inference_mode():
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
                array = pooled.detach().cpu().numpy().astype("float32", copy=False)
            elapsed_seconds = time.perf_counter() - start_time
            effective_rows = int(array.shape[0])
            embedding_shape = list(array.shape)
            if effective_rows != batch_size:
                raise RuntimeError(
                    f"Expected {batch_size} embeddings, got {effective_rows}. "
                    f"Valid rows in batch: {len(valid_rows)}."
                )
            if device.type == "cuda":
                peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            del batch, pixel_values, raw_outputs, features, pooled, array
            highest_successful = batch_size
            log(
                f"[batch-test-done] batch_size={batch_size} status=success "
                f"elapsed_seconds={elapsed_seconds:.3f} seconds_per_image={elapsed_seconds / batch_size:.4f} "
                f"embedding_shape={embedding_shape} peak_memory_mb={peak_memory_mb}"
            )
        except Exception as exc:  # pragma: no cover
            elapsed_seconds = float(time.perf_counter() - start_time) if "start_time" in locals() else None
            status = "failure"
            error_type = type(exc).__name__
            error_message = str(exc)
            if device.type == "cuda":
                peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            log(
                f"[batch-test-done] batch_size={batch_size} status=failure "
                f"elapsed_seconds={0.0 if elapsed_seconds is None else elapsed_seconds:.3f} "
                f"error_type={error_type} peak_memory_mb={peak_memory_mb}"
            )
        finally:
            del dataloader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        results.append(
            {
                "batch_size": batch_size,
                "status": status,
                "elapsed_seconds": elapsed_seconds,
                "seconds_per_image": None if elapsed_seconds is None else elapsed_seconds / batch_size,
                "embedding_shape": embedding_shape,
                "effective_rows": effective_rows,
                "peak_memory_mb": peak_memory_mb,
                "error_type": error_type,
                "error_message": error_message,
            }
        )
        if status != "success":
            break

    summary = {
        "run_date_utc": utc_now_iso(),
        "highest_successful_batch_size": highest_successful,
        "results": results,
    }
    write_json(experiment_dir / "results.json", summary)
    report_lines = [
        "# Torch Image Batch Sweep",
        "",
        f"- Experiment: `{experiment_name}`",
        f"- Domain/split: `{args.domain}/{args.split}`",
        f"- Encoder: `{encoder_bundle.backend}:{encoder_bundle.encoder_id}`",
        f"- Weights: `{encoder_bundle.weights_label}`",
        f"- Candidates: `{candidates}`",
        f"- Highest successful batch size: `{highest_successful}`",
        "",
        "## Results",
    ]
    for result in results:
        line = (
            f"- batch_size={result['batch_size']}: {result['status']}, "
            f"elapsed_seconds={result['elapsed_seconds']}, "
            f"seconds_per_image={result['seconds_per_image']}, "
            f"peak_memory_mb={result['peak_memory_mb']}"
        )
        if result["error_type"]:
            line += f", error_type={result['error_type']}"
        report_lines.append(line)
    (experiment_dir / "recreation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    log(
        f"[done] experiment_dir={experiment_dir} highest_successful_batch_size={highest_successful} "
        f"results={len(results)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
