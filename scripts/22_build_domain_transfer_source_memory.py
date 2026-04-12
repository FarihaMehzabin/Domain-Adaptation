#!/usr/bin/env python3
"""Build a new source-memory root for the image-only domain-transfer branch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss  # type: ignore
import numpy as np

from domain_transfer_rag_common import (
    DEFAULT_BASELINE_EXPERIMENT_DIR,
    DEFAULT_EMBEDDING_ROOT,
    DEFAULT_EXPERIMENTS_ROOT,
    DEFAULT_MANIFEST_CSV,
    DEFAULT_QUALITATIVE_QUERIES,
    DEFAULT_SEED,
    DEFAULT_SOURCE_DOMAIN,
    DEFAULT_SOURCE_SPLIT,
    build_faiss_index,
    build_labels_from_records,
    build_qualitative_neighbors,
    build_simple_recreation_report,
    choose_qualitative_query_indices,
    load_embedding_split,
    load_manifest_records,
    normalize_rows,
    read_json,
    resolve_experiment_identity,
    search_index,
    split_alias_from_domain_split,
    utc_now_iso,
    validate_query_alignment,
    write_json,
)


DEFAULT_TOP_K = 5
DEFAULT_SELF_RETRIEVAL_SAMPLE_SIZE = 512
OPERATION_LABEL = "domain_transfer_source_retrieval_memory_building"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a train-time retrieval memory for the new transfer branch.")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--baseline-experiment-dir", type=Path, default=DEFAULT_BASELINE_EXPERIMENT_DIR)
    parser.add_argument("--source-domain", type=str, default=DEFAULT_SOURCE_DOMAIN)
    parser.add_argument("--source-split", type=str, default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--self-retrieval-sample-size", type=int, default=DEFAULT_SELF_RETRIEVAL_SAMPLE_SIZE)
    parser.add_argument("--qualitative-queries", type=int, default=DEFAULT_QUALITATIVE_QUERIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_items_jsonl(
    path: Path,
    *,
    row_ids: list[str],
    image_paths: list[str],
    labels: np.ndarray,
    label_names: list[str],
    domain: str,
    split: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row_id, image_path, label_row in zip(row_ids, image_paths, labels.astype(np.float32), strict=True):
            payload = {
                "row_id": row_id,
                "domain": domain,
                "split": split,
                "image_path": image_path,
                "positive_labels": [
                    label_names[index] for index, value in enumerate(label_row.tolist()) if float(value) > 0.5
                ],
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def exclude_self_neighbors(indices: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filtered_indices = np.full((indices.shape[0], max(indices.shape[1] - 1, 1)), -1, dtype=np.int64)
    filtered_scores = np.full((scores.shape[0], max(scores.shape[1] - 1, 1)), -1.0, dtype=np.float32)
    for row_idx in range(indices.shape[0]):
        write_idx = 0
        for match_index, score in zip(indices[row_idx].tolist(), scores[row_idx].tolist(), strict=True):
            if int(match_index) == row_idx:
                continue
            if write_idx >= filtered_indices.shape[1]:
                break
            filtered_indices[row_idx, write_idx] = int(match_index)
            filtered_scores[row_idx, write_idx] = float(score)
            write_idx += 1
    return filtered_indices, filtered_scores


def main() -> int:
    args = parse_args()
    manifest_csv = args.manifest_csv.resolve()
    embedding_root = args.embedding_root.resolve()
    baseline_experiment_dir = args.baseline_experiment_dir.resolve()
    experiments_root = args.experiments_root.resolve()
    source_domain = args.source_domain.strip()
    source_split = args.source_split.strip().lower()
    source_alias = split_alias_from_domain_split(source_domain, source_split)

    generated_slug = f"{embedding_root.name}__{manifest_csv.stem}__{source_alias}"
    _, experiment_id, experiment_name, experiment_dir = resolve_experiment_identity(
        experiments_root=experiments_root,
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        operation_label=OPERATION_LABEL,
        overwrite=args.overwrite,
    )

    _, label_names, records = load_manifest_records(manifest_csv, domain=source_domain, split=source_split)
    source_data = load_embedding_split(embedding_root, domain=source_domain, split=source_split)
    labels, manifest_image_paths, expected_row_ids = build_labels_from_records(source_data.row_ids, records)
    validate_query_alignment(source_data.row_ids, expected_row_ids, source_data.image_paths, manifest_image_paths)
    normalized_embeddings, raw_norm_summary, normalized_norm_summary = normalize_rows(source_data.embeddings)

    index = build_faiss_index(normalized_embeddings)
    self_search_k = min(int(index.ntotal), max(args.top_k + 1, 2))
    self_indices, self_scores = search_index(index, normalized_embeddings, self_search_k)
    filtered_indices, filtered_scores = exclude_self_neighbors(self_indices, self_scores)

    sample_size = min(int(args.self_retrieval_sample_size), normalized_embeddings.shape[0])
    sampled = (
        np.linspace(0, normalized_embeddings.shape[0] - 1, num=sample_size, dtype=np.int64).tolist()
        if sample_size
        else []
    )
    top1_self_hits = [int(self_indices[int(index_value), 0]) == int(index_value) for index_value in sampled]
    sanity_report = {
        "embedding_sanity": {
            "row_count": int(normalized_embeddings.shape[0]),
            "embedding_dim": int(normalized_embeddings.shape[1]),
            "raw_norm_summary": raw_norm_summary,
            "normalized_norm_summary": normalized_norm_summary,
        },
        "self_retrieval": {
            "sample_size": int(sample_size),
            "top1_self_hit_rate": float(np.mean(top1_self_hits)) if top1_self_hits else None,
        },
    }

    qualitative_indices = choose_qualitative_query_indices(labels, int(args.qualitative_queries))
    qualitative_neighbors = build_qualitative_neighbors(
        query_indices=qualitative_indices,
        query_row_ids=source_data.row_ids,
        query_image_paths=manifest_image_paths,
        query_labels=labels,
        neighbor_indices=filtered_indices[:, : int(args.top_k)],
        neighbor_scores=filtered_scores[:, : int(args.top_k)],
        train_row_ids=source_data.row_ids,
        train_labels=labels,
        label_names=label_names,
    )

    gitignore_path = experiment_dir / ".gitignore"
    embeddings_path = experiment_dir / "embeddings.npy"
    labels_path = experiment_dir / "labels.npy"
    row_ids_path = experiment_dir / "row_ids.json"
    image_paths_path = experiment_dir / "image_paths.txt"
    items_path = experiment_dir / "items.jsonl"
    sanity_path = experiment_dir / "sanity_report.json"
    qualitative_path = experiment_dir / "qualitative_neighbors.json"
    index_path = experiment_dir / "index.faiss"
    experiment_meta_path = experiment_dir / "experiment_meta.json"
    recreation_report_path = experiment_dir / "recreation_report.md"

    gitignore_path.write_text("/embeddings.npy\n/index.faiss\n", encoding="utf-8")
    np.save(embeddings_path, normalized_embeddings.astype(np.float32))
    np.save(labels_path, labels.astype(np.float32))
    row_ids_path.write_text(json.dumps(source_data.row_ids, indent=2) + "\n", encoding="utf-8")
    image_paths_path.write_text("\n".join(manifest_image_paths) + "\n", encoding="utf-8")
    write_items_jsonl(
        items_path,
        row_ids=source_data.row_ids,
        image_paths=manifest_image_paths,
        labels=labels,
        label_names=label_names,
        domain=source_domain,
        split=source_split,
    )
    write_json(sanity_path, sanity_report)
    write_json(qualitative_path, qualitative_neighbors)
    faiss.write_index(index, str(index_path))

    baseline_meta_path = baseline_experiment_dir / "experiment_meta.json"
    experiment_meta = {
        "argv": sys.argv,
        "baseline_experiment_dir": str(baseline_experiment_dir),
        "baseline_reference": read_json(baseline_meta_path) if baseline_meta_path.exists() else None,
        "embedding_root": str(embedding_root),
        "experiment_dir": str(experiment_dir),
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "label_names": label_names,
        "manifest_csv": str(manifest_csv),
        "operation_label": OPERATION_LABEL,
        "run_date_utc": utc_now_iso(),
        "source_domain": source_domain,
        "source_split": source_split,
        "source_alias": source_alias,
        "top_k": int(args.top_k),
        "artifacts": {
            "embeddings": str(embeddings_path),
            "labels": str(labels_path),
            "row_ids": str(row_ids_path),
            "image_paths": str(image_paths_path),
            "items_jsonl": str(items_path),
            "sanity_report": str(sanity_path),
            "qualitative_neighbors": str(qualitative_path),
            "index": str(index_path),
        },
    }
    write_json(experiment_meta_path, experiment_meta)

    recreation_report = build_simple_recreation_report(
        experiment_dir=experiment_dir,
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        summary_lines=[
            f"- Source split: `{source_domain}/{source_split}`",
            f"- Embedding root: `{embedding_root}`",
            f"- Manifest: `{manifest_csv}`",
            f"- Memory rows: `{normalized_embeddings.shape[0]:,}`",
            f"- Embedding dim: `{normalized_embeddings.shape[1]}`",
            f"- Baseline reference: `{baseline_experiment_dir}`",
        ],
    )
    recreation_report_path.write_text(recreation_report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
