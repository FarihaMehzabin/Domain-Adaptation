#!/usr/bin/env python3
"""Export split-aware CXR Foundation image embeddings for domain-transfer experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import experiment_layout
import numpy as np
from PIL import Image

from cxr_foundation_common import (
    CxrFoundationImageEmbedder,
    resolve_hf_token,
    serialize_pil_image_to_tf_example,
)


DEFAULT_MANIFEST_CSV = Path("/workspace/manifest/manifest_common_labels_nih_train_val_test_chexpert_mimic.csv")
DEFAULT_DATA_ROOT = Path("/workspace")
DEFAULT_EXPERIMENTS_ROOT = Path("/workspace/experiments")
DEFAULT_MODEL_DIR = Path("/workspace/.cache/cxr_foundation")
DEFAULT_BATCH_SIZE = 8
DEFAULT_OPERATION_LABEL = "cxr_foundation_embedding_export"
DEFAULT_EXPERIMENT_ID_WIDTH = 4
DEFAULT_HF_TOKEN_ENV_VAR = "HF_TOKEN"


@dataclass(frozen=True)
class ManifestRow:
    domain: str
    dataset: str
    split: str
    row_id: str
    image_path: str
    study_id: str
    label_vector: tuple[float, ...]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log(message: str) -> None:
    print(f"[{utc_now_iso()}] {message}", flush=True)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_bytes(payload)
    tmp_path.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))


def write_json_list_atomic(path: Path, values: list[str]) -> None:
    write_text_atomic(path, json.dumps(values, indent=2))


def write_jsonl_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    write_text_atomic(path, payload)


def np_save_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array)
    tmp_path.replace(path)


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
    requested = (requested_name or "").strip() or None
    base_name = ensure_operation_prefix(requested or generated_slug)
    if experiments_root.name == "by_id":
        experiments_root.mkdir(parents=True, exist_ok=True)
        explicit_number = extract_experiment_number(base_name)
        if explicit_number is None:
            experiment_number = next_experiment_number(experiments_root)
            experiment_name = f"exp{experiment_number:0{id_width}d}__{base_name}"
        else:
            experiment_number = explicit_number
            experiment_name = base_name
        experiment_id = f"exp{experiment_number:0{id_width}d}"
        experiment_dir = experiments_root / experiment_name
        if experiment_dir.exists() and not overwrite:
            raise SystemExit(
                f"Experiment directory already exists: {experiment_dir}\n"
                "Pass --overwrite to reuse it or choose a different --experiment-name."
            )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_number, experiment_id, experiment_name, experiment_dir
    return experiment_layout.resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=base_name if requested else None,
        generated_slug=base_name,
        overwrite=overwrite,
        id_width=id_width,
    )


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


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


def load_manifest_rows(
    manifest_csv: Path,
    *,
    max_rows_per_split: int | None,
) -> tuple[list[str], dict[tuple[str, str], list[ManifestRow]]]:
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV not found: {manifest_csv}")
    text = manifest_csv.read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise SystemExit(f"Manifest CSV has no header: {manifest_csv}")
    required = {"domain", "dataset", "split", "row_id", "image_path", "study_id"}
    if not required.issubset(reader.fieldnames):
        raise SystemExit(f"Manifest CSV must contain columns: {sorted(required)}")
    label_columns = [field for field in reader.fieldnames if field.startswith("label_")]
    if not label_columns:
        raise SystemExit("Manifest CSV must contain at least one label_... column.")

    grouped: dict[tuple[str, str], list[ManifestRow]] = {}
    counts: dict[tuple[str, str], int] = {}
    for row in reader:
        domain = (row.get("domain") or "").strip()
        split = (row.get("split") or "").strip().lower()
        if not domain or not split:
            continue
        key = (domain, split)
        counts.setdefault(key, 0)
        if max_rows_per_split is not None and counts[key] >= max_rows_per_split:
            continue
        row_id = (row.get("row_id") or "").strip()
        image_path = (row.get("image_path") or "").strip()
        if not row_id or not image_path:
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
        grouped.setdefault(key, []).append(
            ManifestRow(
                domain=domain,
                dataset=(row.get("dataset") or "").strip(),
                split=split,
                row_id=row_id,
                image_path=image_path,
                study_id=(row.get("study_id") or "NA").strip() or "NA",
                label_vector=tuple(labels),
            )
        )
        counts[key] += 1
    return [column.removeprefix("label_") for column in label_columns], grouped


def save_split_manifest(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "row_id", "study_id", "image_path", "dataset"])
        for idx, row in enumerate(rows):
            writer.writerow([idx, row.row_id, row.study_id, row.image_path, row.dataset])


def save_json_list(path: Path, values: list[str]) -> None:
    write_json_list_atomic(path, values)


def save_failed_rows(path: Path, rows: list[dict[str, str]]) -> None:
    write_jsonl_atomic(path, rows)


def load_batch_serialized_examples(
    rows: list[ManifestRow],
    *,
    data_root: Path,
) -> tuple[list[bytes], list[ManifestRow], list[dict[str, str]]]:
    serialized_examples: list[bytes] = []
    successful_rows: list[ManifestRow] = []
    failed_rows: list[dict[str, str]] = []
    for row in rows:
        try:
            resolved_image_path = resolve_manifest_image_path(data_root, row.image_path)
            with Image.open(resolved_image_path) as image:
                serialized = serialize_pil_image_to_tf_example(image)
            serialized_examples.append(serialized)
            successful_rows.append(row)
        except Exception as exc:
            failed_rows.append(
                {
                    "row_id": row.row_id,
                    "domain": row.domain,
                    "split": row.split,
                    "image_path": row.image_path,
                    "error": str(exc),
                }
            )
    return serialized_examples, successful_rows, failed_rows


def embed_rows_with_fallback(
    embedder: CxrFoundationImageEmbedder,
    rows: list[ManifestRow],
    *,
    data_root: Path,
    embedding_kind: str,
    token_pooling: str,
) -> tuple[np.ndarray | None, list[ManifestRow], list[dict[str, str]]]:
    serialized_examples, successful_rows, failed_rows = load_batch_serialized_examples(
        rows,
        data_root=data_root,
    )
    if not serialized_examples:
        return None, [], failed_rows
    try:
        embeddings = embedder.embed_serialized_examples(
            serialized_examples,
            embedding_kind=embedding_kind,
            token_pooling=token_pooling,
        )
        return embeddings, successful_rows, failed_rows
    except Exception as batch_exc:
        if len(rows) == 1:
            failed_rows.append(
                {
                    "row_id": rows[0].row_id,
                    "domain": rows[0].domain,
                    "split": rows[0].split,
                    "image_path": rows[0].image_path,
                    "error": str(batch_exc),
                }
            )
            return None, [], failed_rows

    all_embeddings: list[np.ndarray] = []
    final_rows: list[ManifestRow] = []
    for row in successful_rows:
        single_embeddings, single_rows, single_failed = embed_rows_with_fallback(
            embedder,
            [row],
            data_root=data_root,
            embedding_kind=embedding_kind,
            token_pooling=token_pooling,
        )
        failed_rows.extend(single_failed)
        if single_embeddings is None or not single_rows:
            continue
        all_embeddings.append(single_embeddings)
        final_rows.extend(single_rows)
    if not all_embeddings:
        return None, [], failed_rows
    return np.concatenate(all_embeddings, axis=0), final_rows, failed_rows


def format_bash_command(argv: list[str]) -> str:
    return " \\\n  ".join(shlex.quote(part) for part in argv)


def manifest_row_to_payload(row: ManifestRow) -> dict[str, Any]:
    return {
        "domain": row.domain,
        "dataset": row.dataset,
        "split": row.split,
        "row_id": row.row_id,
        "image_path": row.image_path,
        "study_id": row.study_id,
        "label_vector": list(row.label_vector),
    }


def manifest_row_from_payload(payload: dict[str, Any]) -> ManifestRow:
    return ManifestRow(
        domain=str(payload["domain"]),
        dataset=str(payload["dataset"]),
        split=str(payload["split"]),
        row_id=str(payload["row_id"]),
        image_path=str(payload["image_path"]),
        study_id=str(payload["study_id"]),
        label_vector=tuple(float(value) for value in payload["label_vector"]),
    )


def build_partial_batch_dir(
    split_dir: Path,
    *,
    batch_index: int,
    start_row: int,
    end_row: int,
) -> Path:
    return split_dir / "_partial_batches" / f"batch_{batch_index:05d}__rows_{start_row:07d}_{end_row:07d}"


def load_partial_batch_meta(batch_dir: Path) -> dict[str, Any] | None:
    meta_path = batch_dir / "meta.json"
    if not meta_path.exists():
        return None
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        return None
    if payload.get("embedding_shape") is not None and not (batch_dir / "embeddings.npy").exists():
        return None
    return payload


def save_partial_batch(
    *,
    batch_dir: Path,
    batch_index: int,
    start_row: int,
    end_row: int,
    batch_embeddings: np.ndarray | None,
    successful_rows: list[ManifestRow],
    failed_rows: list[dict[str, str]],
) -> dict[str, Any]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = batch_dir / "embeddings.npy"
    if batch_embeddings is not None and len(successful_rows) > 0:
        np_save_atomic(embeddings_path, batch_embeddings.astype(np.float32, copy=False))
        embedding_shape: list[int] | None = list(batch_embeddings.shape)
    else:
        embedding_shape = None
        if embeddings_path.exists():
            embeddings_path.unlink()

    meta = {
        "status": "completed",
        "saved_at_utc": utc_now_iso(),
        "batch_index": int(batch_index),
        "start_row": int(start_row),
        "end_row": int(end_row),
        "requested_rows": int(end_row - start_row + 1),
        "num_embedded_rows": int(len(successful_rows)),
        "num_failed_rows": int(len(failed_rows)),
        "embedding_shape": embedding_shape,
        "successful_rows": [manifest_row_to_payload(row) for row in successful_rows],
        "failed_rows": failed_rows,
    }
    write_json(batch_dir / "meta.json", meta)
    return meta


def load_saved_partial_batch(
    batch_dir: Path,
) -> tuple[np.ndarray | None, list[ManifestRow], list[dict[str, str]], dict[str, Any]]:
    meta = load_partial_batch_meta(batch_dir)
    if meta is None:
        raise SystemExit(f"Missing or incomplete partial batch checkpoint: {batch_dir}")
    successful_rows = [manifest_row_from_payload(payload) for payload in meta.get("successful_rows", [])]
    failed_rows = [dict(item) for item in meta.get("failed_rows", [])]
    embeddings_path = batch_dir / "embeddings.npy"
    embeddings: np.ndarray | None = None
    if meta.get("embedding_shape") is not None:
        embeddings = np.load(embeddings_path)
        if int(embeddings.shape[0]) != len(successful_rows):
            raise SystemExit(
                f"Partial batch row mismatch in {batch_dir}: "
                f"embeddings={embeddings.shape[0]} successful_rows={len(successful_rows)}"
            )
    return embeddings, successful_rows, failed_rows, meta


def write_split_progress(
    *,
    split_dir: Path,
    domain: str,
    split: str,
    batch_size: int,
    total_rows: int,
    total_batches: int,
    completed_batches: int,
    embedded_rows: int,
    failed_rows: int,
    status: str,
    last_completed_batch: str | None,
    final_embedding_path: str | None = None,
) -> None:
    payload = {
        "updated_at_utc": utc_now_iso(),
        "domain": domain,
        "split": split,
        "batch_size": int(batch_size),
        "total_rows": int(total_rows),
        "total_batches": int(total_batches),
        "completed_batches": int(completed_batches),
        "embedded_rows": int(embedded_rows),
        "failed_rows": int(failed_rows),
        "status": status,
        "last_completed_batch": last_completed_batch,
        "final_embedding_path": final_embedding_path,
    }
    write_json(split_dir / "split_progress.json", payload)


def render_recreation_report(
    *,
    experiment_dir: Path,
    manifest_csv: Path,
    model_dir: Path,
    embedding_kind: str,
    token_pooling: str,
    split_summary: list[str],
    config: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# CXR Foundation Embedding Export Recreation Report",
            "",
            "## Scope",
            "",
            f"- Experiment directory: `{experiment_dir}`",
            f"- Manifest: `{manifest_csv}`",
            f"- Model cache dir: `{model_dir}`",
            f"- Embedding kind: `{embedding_kind}`",
            f"- Token pooling: `{token_pooling}`",
            "",
            "## Recreation Command",
            "",
            "```bash",
            format_bash_command(["python", *config["argv"]]),
            "```",
            "",
            "## Split Outputs",
            "",
            *split_summary,
            "",
            "## Notes",
            "",
            "- This export follows the official Google CXR Foundation local Hugging Face path.",
            "- Image preprocessing uses grayscale PNG-backed `tf.train.Example` inputs before ELIXR-C inference.",
            "- `embedding_index.csv` maps each row to its sharded `embeddings.npy` file and row offset.",
            "- If Hugging Face terms have not been accepted for `google/cxr-foundation`, set `HF_TOKEN` after acceptance and rerun.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export image-only CXR Foundation embeddings for D0/D1/D2 chest X-ray transfer experiments."
        )
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--embedding-kind",
        choices=("general", "contrastive"),
        default="general",
        help="`general` uses ELIXR v2.0 image embeddings; `contrastive` uses the text-aligned image space.",
    )
    parser.add_argument(
        "--token-pooling",
        choices=("avg", "cls", "flatten", "none"),
        default="avg",
        help="How to store token-grid outputs. Default `avg` keeps storage manageable for classifier training.",
    )
    parser.add_argument("--hf-token-env-var", type=str, default=DEFAULT_HF_TOKEN_ENV_VAR)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.max_rows_per_split is not None and args.max_rows_per_split <= 0:
        raise SystemExit("--max-rows-per-split must be positive when provided.")

    manifest_csv = args.manifest_csv.resolve()
    data_root = args.data_root.resolve()
    experiments_root = args.experiments_root.resolve()
    model_dir = args.model_dir.resolve()
    label_names, grouped_rows = load_manifest_rows(
        manifest_csv,
        max_rows_per_split=args.max_rows_per_split,
    )
    if not grouped_rows:
        raise SystemExit(f"No rows were loaded from manifest: {manifest_csv}")

    generated_slug = "__".join(
        [
            DEFAULT_OPERATION_LABEL,
            slugify(manifest_csv.stem, fallback="manifest"),
            f"kind-{args.embedding_kind}",
            f"pool-{args.token_pooling}",
            f"batch-{args.batch_size}",
        ]
    )
    experiment_number, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=bool(args.overwrite),
    )

    hf_token = resolve_hf_token(args.hf_token_env_var)
    log(
        f"[startup] manifest_csv={manifest_csv} experiment_dir={experiment_dir} "
        f"embedding_kind={args.embedding_kind} token_pooling={args.token_pooling} batch_size={args.batch_size}"
    )
    embedder = CxrFoundationImageEmbedder(model_dir=model_dir, hf_token=hf_token)
    log("[startup] CXR Foundation backbone initialized")

    config = {
        "argv": list(sys.argv),
        "run_date_utc": utc_now_iso(),
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "manifest_csv": str(manifest_csv),
        "data_root": str(data_root),
        "model_dir": str(model_dir),
        "embedding_kind": args.embedding_kind,
        "token_pooling": args.token_pooling,
        "batch_size": int(args.batch_size),
        "hf_token_env_var": args.hf_token_env_var,
        "max_rows_per_split": int(args.max_rows_per_split) if args.max_rows_per_split is not None else None,
        "label_names": label_names,
    }
    write_json(experiment_dir / "config.json", config)
    (experiment_dir / "label_names.json").write_text(json.dumps(label_names, indent=2), encoding="utf-8")

    index_rows: list[dict[str, Any]] = []
    split_meta: dict[str, Any] = {}
    split_summary_lines: list[str] = []

    for key in sorted(grouped_rows):
        domain, split = key
        rows = grouped_rows[key]
        split_dir = experiment_dir / domain / split
        split_dir.mkdir(parents=True, exist_ok=True)
        total_rows = len(rows)
        total_batches = int(math.ceil(total_rows / args.batch_size))
        completed_batches = 0
        checkpoint_embedded_rows = 0
        checkpoint_failed_rows = 0
        last_completed_batch: str | None = None
        log(f"[split-start] domain={domain} split={split} total_rows={total_rows}")
        write_split_progress(
            split_dir=split_dir,
            domain=domain,
            split=split,
            batch_size=args.batch_size,
            total_rows=total_rows,
            total_batches=total_batches,
            completed_batches=0,
            embedded_rows=0,
            failed_rows=0,
            status="running",
            last_completed_batch=None,
        )
        for batch_index, start in enumerate(range(0, total_rows, args.batch_size), start=1):
            batch_rows = rows[start : start + args.batch_size]
            batch_end = start + len(batch_rows)
            batch_dir = build_partial_batch_dir(
                split_dir,
                batch_index=batch_index,
                start_row=start + 1,
                end_row=batch_end,
            )
            existing_meta = load_partial_batch_meta(batch_dir)
            if existing_meta is not None:
                completed_batches += 1
                checkpoint_embedded_rows += int(existing_meta.get("num_embedded_rows", 0))
                checkpoint_failed_rows += int(existing_meta.get("num_failed_rows", 0))
                last_completed_batch = batch_dir.name
                log(
                    f"[batch-resume-hit] domain={domain} split={split} rows={start + 1}-{batch_end} "
                    f"embedded={existing_meta.get('num_embedded_rows', 0)} "
                    f"failed={existing_meta.get('num_failed_rows', 0)}"
                )
                write_split_progress(
                    split_dir=split_dir,
                    domain=domain,
                    split=split,
                    batch_size=args.batch_size,
                    total_rows=total_rows,
                    total_batches=total_batches,
                    completed_batches=completed_batches,
                    embedded_rows=checkpoint_embedded_rows,
                    failed_rows=checkpoint_failed_rows,
                    status="running",
                    last_completed_batch=last_completed_batch,
                )
                continue
            log(
                f"[batch-start] domain={domain} split={split} rows={start + 1}-{batch_end} "
                f"batch_size={len(batch_rows)}"
            )
            batch_embeddings, batch_successful_rows, batch_failed_rows = embed_rows_with_fallback(
                embedder,
                batch_rows,
                data_root=data_root,
                embedding_kind=args.embedding_kind,
                token_pooling=args.token_pooling,
            )
            log(
                f"[batch-done] domain={domain} split={split} rows={start + 1}-{batch_end} "
                f"embedded={0 if batch_embeddings is None else int(batch_embeddings.shape[0])} "
                f"successful={len(batch_successful_rows)} failed={len(batch_failed_rows)}"
            )
            saved_meta = save_partial_batch(
                batch_dir=batch_dir,
                batch_index=batch_index,
                start_row=start + 1,
                end_row=batch_end,
                batch_embeddings=batch_embeddings,
                successful_rows=batch_successful_rows,
                failed_rows=batch_failed_rows,
            )
            completed_batches += 1
            checkpoint_embedded_rows += int(saved_meta["num_embedded_rows"])
            checkpoint_failed_rows += int(saved_meta["num_failed_rows"])
            last_completed_batch = batch_dir.name
            processed = min(total_rows, start + len(batch_rows))
            log(
                f"[progress] domain={domain} split={split} processed={processed}/{total_rows} "
                f"embedded={checkpoint_embedded_rows} failed={checkpoint_failed_rows}"
            )
            log(
                f"[batch-checkpoint-saved] domain={domain} split={split} rows={start + 1}-{batch_end} "
                f"checkpoint_dir={batch_dir.relative_to(experiment_dir)}"
            )
            write_split_progress(
                split_dir=split_dir,
                domain=domain,
                split=split,
                batch_size=args.batch_size,
                total_rows=total_rows,
                total_batches=total_batches,
                completed_batches=completed_batches,
                embedded_rows=checkpoint_embedded_rows,
                failed_rows=checkpoint_failed_rows,
                status="running",
                last_completed_batch=last_completed_batch,
            )

        embedded_chunks: list[np.ndarray] = []
        successful_rows: list[ManifestRow] = []
        failed_rows: list[dict[str, str]] = []
        for batch_index, start in enumerate(range(0, total_rows, args.batch_size), start=1):
            batch_rows = rows[start : start + args.batch_size]
            batch_end = start + len(batch_rows)
            batch_dir = build_partial_batch_dir(
                split_dir,
                batch_index=batch_index,
                start_row=start + 1,
                end_row=batch_end,
            )
            batch_embeddings, batch_successful_rows, batch_failed_rows, _ = load_saved_partial_batch(batch_dir)
            failed_rows.extend(batch_failed_rows)
            if batch_embeddings is None:
                if batch_successful_rows:
                    raise SystemExit(
                        f"Partial batch is missing embeddings despite successful rows: {batch_dir}"
                    )
                continue
            embedded_chunks.append(np.asarray(batch_embeddings, dtype=np.float32))
            successful_rows.extend(batch_successful_rows)

        if not embedded_chunks:
            raise SystemExit(f"No embeddings were produced for domain={domain} split={split}.")

        embeddings = np.concatenate(embedded_chunks, axis=0).astype(np.float32, copy=False)
        if embeddings.shape[0] != len(successful_rows):
            raise SystemExit(
                f"Embedding row count mismatch for domain={domain} split={split}: "
                f"embeddings={embeddings.shape[0]} rows={len(successful_rows)}"
            )

        np_save_atomic(split_dir / "embeddings.npy", embeddings)
        save_json_list(split_dir / "row_ids.json", [row.row_id for row in successful_rows])
        save_json_list(split_dir / "study_ids.json", [row.study_id for row in successful_rows])
        write_text_atomic(
            split_dir / "image_paths.txt",
            "\n".join(row.image_path for row in successful_rows) + "\n",
        )
        save_split_manifest(split_dir / "image_manifest.csv", successful_rows)
        relative_embedding_path = (split_dir / "embeddings.npy").relative_to(experiment_dir)
        if failed_rows:
            save_failed_rows(split_dir / "failed_rows.jsonl", failed_rows)
        log(
            f"[split-done] domain={domain} split={split} embedded_rows={len(successful_rows)} "
            f"failed_rows={len(failed_rows)} embedding_path={relative_embedding_path}"
        )
        for row_idx, row in enumerate(successful_rows):
            index_rows.append(
                {
                    "study_id": row.study_id,
                    "domain": row.domain,
                    "split": row.split,
                    "dataset": row.dataset,
                    "row_id": row.row_id,
                    "image_path": row.image_path,
                    "embedding_path": str(relative_embedding_path),
                    "embedding_row": row_idx,
                    "label_vector": json.dumps(list(row.label_vector)),
                }
            )

        run_meta = {
            "run_date_utc": utc_now_iso(),
            "domain": domain,
            "split": split,
            "num_requested_rows": total_rows,
            "num_embedded_rows": len(successful_rows),
            "num_failed_rows": len(failed_rows),
            "embedding_shape": list(embeddings.shape),
            "embedding_kind": args.embedding_kind,
            "token_pooling": args.token_pooling,
            "label_names": label_names,
            "embeddings_path": str(split_dir / "embeddings.npy"),
            "row_ids_path": str(split_dir / "row_ids.json"),
            "image_paths_path": str(split_dir / "image_paths.txt"),
        }
        write_json(split_dir / "run_meta.json", run_meta)
        write_split_progress(
            split_dir=split_dir,
            domain=domain,
            split=split,
            batch_size=args.batch_size,
            total_rows=total_rows,
            total_batches=total_batches,
            completed_batches=total_batches,
            embedded_rows=len(successful_rows),
            failed_rows=len(failed_rows),
            status="completed",
            last_completed_batch=last_completed_batch,
            final_embedding_path=str(relative_embedding_path),
        )
        split_meta[f"{domain}/{split}"] = run_meta
        shape_text = "x".join(str(dim) for dim in embeddings.shape[1:]) if embeddings.ndim > 1 else "1"
        split_summary_lines.append(
            f"- `{domain}/{split}`: `{len(successful_rows)}` rows, embedding tail shape `{shape_text}`, failures `{len(failed_rows)}`"
        )

    index_fields = [
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
    with (experiment_dir / "embedding_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=index_fields)
        writer.writeheader()
        writer.writerows(index_rows)

    experiment_meta = {
        "run_date_utc": utc_now_iso(),
        "experiment_number": experiment_number,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "operation_label": DEFAULT_OPERATION_LABEL,
        "manifest_csv": str(manifest_csv),
        "data_root": str(data_root),
        "model_dir": str(model_dir),
        "embedding_kind": args.embedding_kind,
        "token_pooling": args.token_pooling,
        "label_names": label_names,
        "split_outputs": split_meta,
        "embedding_index_csv": str(experiment_dir / "embedding_index.csv"),
        "label_names_json": str(experiment_dir / "label_names.json"),
    }
    write_json(experiment_dir / "experiment_meta.json", experiment_meta)

    recreation_report = render_recreation_report(
        experiment_dir=experiment_dir,
        manifest_csv=manifest_csv,
        model_dir=model_dir,
        embedding_kind=args.embedding_kind,
        token_pooling=args.token_pooling,
        split_summary=split_summary_lines,
        config=config,
    )
    (experiment_dir / "recreation_report.md").write_text(recreation_report, encoding="utf-8")

    log(
        f"[done] experiment_dir={experiment_dir} index_rows={len(index_rows)} "
        f"embedding_kind={args.embedding_kind} token_pooling={args.token_pooling}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
