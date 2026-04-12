#!/usr/bin/env python3
"""Benchmark CXR Foundation batch sizes on a single domain/split batch."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import experiment_layout
from cxr_foundation_common import (
    CxrFoundationImageEmbedder,
    resolve_hf_token,
)


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest/manifest_common_labels_pilot5h.csv")
DEFAULT_DATA_ROOT = Path("/workspace")
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_MODEL_DIR = Path("/workspace/.cache/cxr_foundation")
DEFAULT_OPERATION_LABEL = "cxr_foundation_batch_sweep"
DEFAULT_EXPERIMENT_ID_WIDTH = 4
DEFAULT_DOMAIN = "d0_nih"
DEFAULT_SPLIT = "test"
DEFAULT_CANDIDATES = (64, 128, 256, 384, 512, 640, 768, 896, 1024)
EXPORTER_HELPER_SCRIPT = Path("/workspace/scripts/14_generate_cxr_foundation_embeddings.py")
@dataclass(frozen=True)
class ManifestRow:
    domain: str
    dataset: str
    split: str
    row_id: str
    image_path: str
    study_id: str
    label_vector: tuple[float, ...]
    resolved_path: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now_iso()}] {message}", flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_exporter_helpers() -> Any:
    if not EXPORTER_HELPER_SCRIPT.exists():
        raise SystemExit(f"Exporter helper script not found: {EXPORTER_HELPER_SCRIPT}")
    module_name = "cxr_foundation_export_helpers"
    spec = importlib.util.spec_from_file_location(module_name, EXPORTER_HELPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load helper module from {EXPORTER_HELPER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the largest successful CXR Foundation batch size on a deterministic NIH test batch."
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf-token-env-var", type=str, default="HF_TOKEN")
    parser.add_argument("--domain", type=str, default=DEFAULT_DOMAIN)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--embedding-kind", choices=("general", "contrastive"), default="general")
    parser.add_argument("--token-pooling", choices=("avg", "cls", "flatten", "none"), default="avg")
    parser.add_argument("--candidate-batch-sizes", type=int, nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def load_rows(manifest_csv: Path, data_root: Path, *, domain: str, split: str, limit: int) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        label_columns = [field for field in fieldnames if field.startswith("label_")]
        for raw_row in reader:
            if (raw_row.get("domain") or "").strip() != domain:
                continue
            if (raw_row.get("split") or "").strip() != split:
                continue
            image_path = str(raw_row.get("image_path") or "").strip()
            if not image_path:
                continue
            rows.append(
                ManifestRow(
                    domain=domain,
                    dataset=str(raw_row.get("dataset") or "unknown").strip() or "unknown",
                    split=split,
                    row_id=str(raw_row.get("row_id") or Path(image_path).stem).strip() or Path(image_path).stem,
                    image_path=image_path,
                    study_id=str(raw_row.get("study_id") or "NA").strip() or "NA",
                    label_vector=tuple(float(str(raw_row.get(column) or "0").strip()) for column in label_columns),
                    resolved_path=resolve_manifest_image_path(data_root, image_path),
                )
            )
            if len(rows) >= limit:
                break
    if len(rows) < limit:
        raise SystemExit(
            f"Needed {limit} rows for domain={domain} split={split}, but only found {len(rows)} in {manifest_csv}"
        )
    return rows


def build_experiment_slug(domain: str, split: str, embedding_kind: str, token_pooling: str) -> str:
    return "__".join(
        [
            DEFAULT_OPERATION_LABEL,
            domain,
            split,
            embedding_kind,
            token_pooling,
        ]
    )


def main() -> int:
    args = parse_args()
    candidates = sorted(set(int(value) for value in args.candidate_batch_sizes if int(value) > 0))
    if not candidates:
        raise SystemExit("Provide at least one positive --candidate-batch-sizes value.")

    manifest_csv = args.manifest_csv.resolve()
    data_root = args.data_root.resolve()
    experiments_root = args.experiments_root.resolve()
    generated_slug = build_experiment_slug(args.domain, args.split, args.embedding_kind, args.token_pooling)
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=bool(args.overwrite),
        id_width=DEFAULT_EXPERIMENT_ID_WIDTH,
    )

    max_batch_size = max(candidates)
    exporter_helpers = load_exporter_helpers()
    rows = load_rows(
        manifest_csv,
        data_root,
        domain=args.domain,
        split=args.split,
        limit=max_batch_size,
    )

    token = resolve_hf_token(args.hf_token_env_var)
    config = {
        "argv": list(sys.argv),
        "run_date_utc": utc_now_iso(),
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "manifest_csv": str(manifest_csv),
        "data_root": str(data_root),
        "model_dir": str(args.model_dir),
        "domain": args.domain,
        "split": args.split,
        "embedding_kind": args.embedding_kind,
        "token_pooling": args.token_pooling,
        "candidate_batch_sizes": candidates,
        "hf_token_env_var": args.hf_token_env_var,
    }
    write_json(experiment_dir / "config.json", config)

    selected_manifest = experiment_dir / "selected_rows.csv"
    with selected_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "row_id", "image_path", "resolved_path"])
        for index, row in enumerate(rows):
            writer.writerow([index, row.row_id, row.image_path, str(row.resolved_path)])

    log(
        f"[startup] manifest_csv={manifest_csv} experiment_dir={experiment_dir} "
        f"domain={args.domain} split={args.split} candidates={candidates} mode=exporter_fallback"
    )
    embedder = CxrFoundationImageEmbedder(model_dir=args.model_dir.resolve(), hf_token=token)
    log("[startup] CXR Foundation backbone initialized")

    results: list[dict[str, Any]] = []
    highest_successful: int | None = None
    for batch_size in candidates:
        log(f"[batch-test-start] batch_size={batch_size}")
        start_time = time.perf_counter()
        status = "success"
        error_type = None
        error_message = None
        embedding_shape: list[int] | None = None
        successful_count = 0
        failed_count = 0
        try:
            exporter_rows = [
                exporter_helpers.ManifestRow(
                    domain=row.domain,
                    dataset=row.dataset,
                    split=row.split,
                    row_id=row.row_id,
                    image_path=row.image_path,
                    study_id=row.study_id,
                    label_vector=row.label_vector,
                )
                for row in rows[:batch_size]
            ]
            embeddings, successful_rows, failed_rows = exporter_helpers.embed_rows_with_fallback(
                embedder,
                exporter_rows,
                data_root=data_root,
                embedding_kind=args.embedding_kind,
                token_pooling=args.token_pooling,
            )
            if embeddings is None:
                raise RuntimeError("Exporter fallback returned no embeddings.")
            successful_count = len(successful_rows)
            failed_count = len(failed_rows)
            embedding_shape = list(getattr(embeddings, "shape", ()))
            if successful_count != batch_size:
                raise RuntimeError(
                    f"Expected {batch_size} successful embeddings, got {successful_count} "
                    f"with {failed_count} failed rows."
                )
            highest_successful = batch_size
            del embeddings
            gc.collect()
        except Exception as exc:  # pragma: no cover
            status = "failure"
            error_type = type(exc).__name__
            error_message = str(exc)
        elapsed_seconds = time.perf_counter() - start_time
        result = {
            "batch_size": batch_size,
            "status": status,
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_image": elapsed_seconds / batch_size,
            "embedding_shape": embedding_shape,
            "successful_rows": successful_count,
            "failed_rows": failed_count,
            "error_type": error_type,
            "error_message": error_message,
        }
        results.append(result)
        if status == "success":
            log(
                f"[batch-test-done] batch_size={batch_size} status=success "
                f"elapsed_seconds={elapsed_seconds:.3f} seconds_per_image={elapsed_seconds / batch_size:.4f} "
                f"embedding_shape={embedding_shape} successful_rows={successful_count} failed_rows={failed_count}"
            )
        else:
            log(
                f"[batch-test-done] batch_size={batch_size} status=failure "
                f"elapsed_seconds={elapsed_seconds:.3f} error_type={error_type}"
            )
            break

    summary = {
        "run_date_utc": utc_now_iso(),
        "benchmark_mode": "exporter_fallback",
        "highest_successful_batch_size": highest_successful,
        "results": results,
    }
    write_json(experiment_dir / "results.json", summary)
    report_lines = [
        "# CXR Foundation Batch Sweep",
        "",
        f"- Experiment: `{experiment_name}`",
        f"- Domain/split: `{args.domain}/{args.split}`",
        f"- Candidates: `{candidates}`",
        f"- Highest successful batch size: `{highest_successful}`",
        "",
        "## Results",
    ]
    for result in results:
        line = (
            f"- batch_size={result['batch_size']}: {result['status']}, "
            f"elapsed_seconds={result['elapsed_seconds']:.3f}, "
            f"seconds_per_image={result['seconds_per_image']:.4f}"
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
