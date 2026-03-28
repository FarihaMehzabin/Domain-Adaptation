#!/usr/bin/env python3
"""Build a source-domain retrieval memory from ResNet50 fused CLS train embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss  # type: ignore
import numpy as np


LABEL_SPECS = [
    ("label_atelectasis", "atelectasis"),
    ("label_cardiomegaly", "cardiomegaly"),
    ("label_consolidation", "consolidation"),
    ("label_edema", "edema"),
    ("label_pleural_effusion", "pleural_effusion"),
]
LABEL_COLUMNS = [column for column, _ in LABEL_SPECS]
LABEL_NAMES = [name for _, name in LABEL_SPECS]

DEFAULT_FUSED_EMBEDDINGS = Path("/workspace/fused_embeddings_cls/resnet50/train/embeddings.npy")
DEFAULT_FUSED_IMAGE_PATHS = Path("/workspace/fused_embeddings_cls/resnet50/train/image_paths.txt")
DEFAULT_FUSED_RUN_META = Path("/workspace/fused_embeddings_cls/resnet50/train/run_meta.json")
DEFAULT_MANIFEST_CSV = Path("/workspace/manifest_nih_cxr14 .csv")
DEFAULT_IMAGE_EMBEDDINGS_DIR = Path("/workspace/image_embeddings/resnet50/train")
DEFAULT_TEXT_EMBEDDINGS_DIR = Path("/workspace/report_embeddings_cls/train/microsoft__BiomedVLP-CXR-BERT-specialized")
DEFAULT_BASELINE_CONFIG = Path("/workspace/outputs/models/nih_cxr14/fused/resnet50_cls_20260324T091149Z/config.json")
DEFAULT_OUTPUT_DIR = Path("/workspace/memory/nih_cxr14/resnet50_fused_cls_train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS retrieval memory from the NIH CXR14 ResNet50 fused CLS train split."
    )
    parser.add_argument("--split", type=str, default="train", choices=["train"])
    parser.add_argument("--fused-embeddings", type=Path, default=DEFAULT_FUSED_EMBEDDINGS)
    parser.add_argument("--fused-image-paths", type=Path, default=DEFAULT_FUSED_IMAGE_PATHS)
    parser.add_argument("--fused-run-meta", type=Path, default=DEFAULT_FUSED_RUN_META)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--image-embeddings-dir", type=Path, default=DEFAULT_IMAGE_EMBEDDINGS_DIR)
    parser.add_argument("--text-embeddings-dir", type=Path, default=DEFAULT_TEXT_EMBEDDINGS_DIR)
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-retrieval-sample-size", type=int, default=512)
    parser.add_argument("--label-agreement-queries", type=int, default=200)
    parser.add_argument("--qualitative-queries", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Text file is empty: {path}")
    return lines


def load_embedding_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array at {path}, found shape {array.shape}.")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Embeddings contain NaN or inf values: {path}")
    return np.ascontiguousarray(array)


def read_manifest_rows(manifest_csv: Path, split_name: str) -> list[dict[str, str]]:
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest CSV {manifest_csv} is missing a header row.")
        required = LABEL_COLUMNS + ["image_path", "split"]
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Manifest CSV {manifest_csv} is missing columns: {missing}")
        rows = [row for row in reader if row.get("split") == split_name]
    if not rows:
        raise ValueError(f"Manifest CSV {manifest_csv} has no rows for split '{split_name}'.")
    return rows


def reorder_rows_by_image_paths(rows: list[dict[str, str]], image_paths: list[str], split_name: str) -> list[dict[str, str]]:
    rows_by_basename: dict[str, dict[str, str]] = {}
    for row in rows:
        basename = Path(row["image_path"]).name
        if basename in rows_by_basename:
            raise ValueError(f"{split_name}: duplicate image in manifest: {basename}")
        rows_by_basename[basename] = row

    ordered: list[dict[str, str]] = []
    missing: list[str] = []
    for image_path in image_paths:
        basename = Path(image_path).name
        row = rows_by_basename.get(basename)
        if row is None:
            if len(missing) < 5:
                missing.append(basename)
            continue
        ordered.append(row)

    if len(ordered) != len(image_paths):
        raise ValueError(
            f"{split_name}: missing manifest rows for {len(image_paths) - len(ordered)} images. Examples: {missing}"
        )
    return ordered


def validate_label_alignment(rows: list[dict[str, str]], image_paths: list[str], embeddings: np.ndarray, split_name: str) -> None:
    if embeddings.shape[0] != len(image_paths):
        raise ValueError(
            f"{split_name}: embedding rows ({embeddings.shape[0]}) do not match image paths ({len(image_paths)})."
        )
    if len(rows) != len(image_paths):
        raise ValueError(f"{split_name}: manifest rows ({len(rows)}) do not match image paths ({len(image_paths)}).")
    for index, (row, image_path) in enumerate(zip(rows, image_paths)):
        expected = Path(row["image_path"]).name
        actual = Path(image_path).name
        if expected != actual:
            raise ValueError(
                f"{split_name}: image order mismatch at row {index}: manifest has {expected}, embeddings have {actual}."
            )


def build_labels(rows: list[dict[str, str]]) -> np.ndarray:
    labels = np.zeros((len(rows), len(LABEL_COLUMNS)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for label_index, column in enumerate(LABEL_COLUMNS):
            labels[row_index, label_index] = float(row[column])
    if not np.isfinite(labels).all():
        raise ValueError("Labels contain NaN or inf values.")
    return labels


def align_image_embeddings(image_embeddings_dir: Path, target_image_paths: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    embeddings = load_embedding_array(image_embeddings_dir / "embeddings.npy")
    source_image_paths = read_lines(image_embeddings_dir / "image_paths.txt")
    if embeddings.shape[0] != len(source_image_paths):
        raise ValueError(
            f"Image embeddings rows ({embeddings.shape[0]}) do not match image paths ({len(source_image_paths)})."
        )

    if source_image_paths == target_image_paths:
        return embeddings, {"source_rows": len(source_image_paths), "exact_order_match": True, "reordered": False}

    source_index: dict[str, int] = {}
    for index, path in enumerate(source_image_paths):
        basename = Path(path).name
        if basename in source_index:
            raise ValueError(f"Duplicate image basename in image embeddings: {basename}")
        source_index[basename] = index

    aligned_indices: list[int] = []
    missing: list[str] = []
    for path in target_image_paths:
        basename = Path(path).name
        match = source_index.get(basename)
        if match is None:
            if len(missing) < 5:
                missing.append(basename)
            continue
        aligned_indices.append(match)
    if len(aligned_indices) != len(target_image_paths):
        raise ValueError(
            f"Missing aligned image embeddings for {len(target_image_paths) - len(aligned_indices)} rows. Examples: {missing}"
        )

    aligned = np.ascontiguousarray(embeddings[np.asarray(aligned_indices, dtype=np.int64)])
    return aligned, {"source_rows": len(source_image_paths), "exact_order_match": False, "reordered": True}


def align_text_embeddings(text_embeddings_dir: Path, target_image_paths: list[str]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    embeddings = load_embedding_array(text_embeddings_dir / "embeddings.npy")
    report_ids = read_json(text_embeddings_dir / "report_ids.json")
    if not isinstance(report_ids, list) or not report_ids:
        raise ValueError(f"Invalid report_ids.json in {text_embeddings_dir}")
    if embeddings.shape[0] != len(report_ids):
        raise ValueError(f"Text embeddings rows ({embeddings.shape[0]}) do not match report ids ({len(report_ids)}).")

    report_index: dict[str, int] = {}
    for index, report_id in enumerate(report_ids):
        key = str(report_id)
        if key in report_index:
            raise ValueError(f"Duplicate report id in text embeddings: {key}")
        report_index[key] = index

    aligned_indices: list[int] = []
    aligned_ids: list[str] = []
    missing: list[str] = []
    for path in target_image_paths:
        example_id = Path(path).stem
        match = report_index.get(example_id)
        if match is None:
            if len(missing) < 5:
                missing.append(example_id)
            continue
        aligned_indices.append(match)
        aligned_ids.append(example_id)
    if len(aligned_indices) != len(target_image_paths):
        raise ValueError(
            f"Missing aligned text embeddings for {len(target_image_paths) - len(aligned_indices)} rows. Examples: {missing}"
        )

    aligned = np.ascontiguousarray(embeddings[np.asarray(aligned_indices, dtype=np.int64)])
    return aligned, aligned_ids, {"source_rows": len(report_ids), "aligned_rows": len(aligned_ids)}


def validate_fusion_consistency(fused_embeddings: np.ndarray, image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> dict[str, Any]:
    expected = np.concatenate([image_embeddings, text_embeddings], axis=1)
    max_abs_diff = float(np.max(np.abs(fused_embeddings - expected)))
    mean_abs_diff = float(np.mean(np.abs(fused_embeddings - expected)))
    matches = bool(np.allclose(fused_embeddings, expected, atol=1e-6, rtol=1e-6))
    if not matches:
        raise ValueError(
            "Fused embeddings do not match the aligned image/text source embeddings. "
            f"max_abs_diff={max_abs_diff:.8f}, mean_abs_diff={mean_abs_diff:.8f}"
        )
    return {"matches": matches, "max_abs_diff": max_abs_diff, "mean_abs_diff": mean_abs_diff}


def normalize_rows(embeddings: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    raw_norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.isfinite(raw_norms).all():
        raise ValueError("Raw embedding norms contain NaN or inf values.")
    zero_norms = int(np.count_nonzero(raw_norms <= 0.0))
    if zero_norms > 0:
        raise ValueError(f"Found {zero_norms} zero-norm embeddings; cannot normalize.")

    normalized = embeddings / raw_norms[:, None].astype(np.float32)
    normalized = np.ascontiguousarray(normalized.astype(np.float32))
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized embeddings contain NaN or inf values.")

    post_norms = np.linalg.norm(normalized.astype(np.float64), axis=1)
    summary = {
        "mean": float(post_norms.mean()),
        "std": float(post_norms.std()),
        "min": float(post_norms.min()),
        "max": float(post_norms.max()),
    }
    return normalized, summary


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(np.ascontiguousarray(embeddings.astype(np.float32)))
    return index


def search_without_self(
    index: faiss.IndexFlatIP,
    query_vectors: np.ndarray,
    query_indices: np.ndarray,
    top_k: int,
) -> tuple[list[list[int]], list[list[float]]]:
    k_search = min(index.ntotal, max(top_k + 5, top_k + 1))
    scores, indices = index.search(np.ascontiguousarray(query_vectors.astype(np.float32)), k_search)
    all_neighbors: list[list[int]] = []
    all_scores: list[list[float]] = []
    for query_row, query_index in enumerate(query_indices.tolist()):
        neigh: list[int] = []
        sims: list[float] = []
        for match_index, score in zip(indices[query_row].tolist(), scores[query_row].tolist()):
            if match_index < 0 or match_index == query_index:
                continue
            neigh.append(int(match_index))
            sims.append(float(score))
            if len(neigh) >= top_k:
                break
        all_neighbors.append(neigh)
        all_scores.append(sims)
    return all_neighbors, all_scores


def labels_to_names(label_row: np.ndarray) -> list[str]:
    indices = np.flatnonzero(label_row > 0.5)
    return [LABEL_NAMES[int(index)] for index in indices.tolist()]


def sample_indices(num_rows: int, sample_size: int, seed: int) -> np.ndarray:
    actual_size = min(num_rows, sample_size)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_rows, size=actual_size, replace=False).astype(np.int64))


def random_neighbors_excluding(query_index: int, num_rows: int, top_k: int, rng: random.Random) -> np.ndarray:
    picks: list[int] = []
    seen = {query_index}
    while len(picks) < min(top_k, num_rows - 1):
        value = rng.randrange(num_rows)
        if value in seen:
            continue
        seen.add(value)
        picks.append(value)
    return np.asarray(picks, dtype=np.int64)


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(a, b).sum()
    return float(intersection / union)


def run_self_retrieval_check(
    index: faiss.IndexFlatIP,
    normalized_embeddings: np.ndarray,
    labels: np.ndarray,
    example_ids: list[str],
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    query_indices = sample_indices(normalized_embeddings.shape[0], sample_size, seed)
    query_vectors = normalized_embeddings[query_indices]
    k_search = min(index.ntotal, 5)
    scores, indices = index.search(np.ascontiguousarray(query_vectors.astype(np.float32)), k_search)

    top1 = indices[:, 0]
    self_hit = top1 == query_indices
    top5_contains_self = np.array(
        [query_index in row.tolist() for query_index, row in zip(query_indices.tolist(), indices)], dtype=bool
    )

    miss_examples: list[dict[str, Any]] = []
    for row_index, hit in enumerate(self_hit.tolist()):
        if hit:
            continue
        query_index = int(query_indices[row_index])
        retrieved_index = int(top1[row_index])
        miss_examples.append(
            {
                "query_index": query_index,
                "query_example_id": example_ids[query_index],
                "query_labels": labels_to_names(labels[query_index]),
                "retrieved_index": retrieved_index,
                "retrieved_example_id": example_ids[retrieved_index],
                "retrieved_labels": labels_to_names(labels[retrieved_index]),
                "similarity": float(scores[row_index, 0]),
            }
        )
        if len(miss_examples) >= 10:
            break

    return {
        "sample_size": int(query_indices.shape[0]),
        "top1_self_hit_rate": float(self_hit.mean()),
        "top5_contains_self_rate": float(top5_contains_self.mean()),
        "top1_similarity_mean": float(scores[:, 0].mean()),
        "top1_similarity_min": float(scores[:, 0].min()),
        "miss_examples": miss_examples,
    }


def run_label_agreement_check(
    index: faiss.IndexFlatIP,
    normalized_embeddings: np.ndarray,
    labels: np.ndarray,
    example_ids: list[str],
    sample_size: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    query_indices = sample_indices(normalized_embeddings.shape[0], sample_size, seed)
    query_vectors = normalized_embeddings[query_indices]
    neighbors, neighbor_scores = search_without_self(index, query_vectors, query_indices, top_k=top_k)

    global_prevalence = labels.mean(axis=0)
    rng = random.Random(seed)

    positive_prevalence_values: list[float] = []
    chance_prevalence_values: list[float] = []
    positive_jaccard_values: list[float] = []
    random_jaccard_values: list[float] = []
    per_label_neighbor_prevalence: dict[str, list[float]] = defaultdict(list)
    per_label_chance_prevalence: dict[str, list[float]] = defaultdict(list)
    query_summaries: list[dict[str, Any]] = []
    positive_query_count = 0

    for row_index, query_index in enumerate(query_indices.tolist()):
        neighbor_indices = neighbors[row_index]
        if not neighbor_indices:
            continue
        query_labels = labels[query_index] > 0.5
        positive_label_indices = np.flatnonzero(query_labels)
        if positive_label_indices.size == 0:
            continue
        positive_query_count += 1

        neighbor_label_matrix = labels[np.asarray(neighbor_indices, dtype=np.int64)] > 0.5
        prevalence_vector = neighbor_label_matrix.mean(axis=0)
        random_neighbor_indices = random_neighbors_excluding(query_index, labels.shape[0], top_k, rng)
        random_label_matrix = labels[random_neighbor_indices] > 0.5

        shared_positive_prevalence = prevalence_vector[positive_label_indices]
        chance_prevalence = global_prevalence[positive_label_indices]
        positive_prevalence_values.extend(shared_positive_prevalence.astype(np.float64).tolist())
        chance_prevalence_values.extend(chance_prevalence.astype(np.float64).tolist())

        for label_index in positive_label_indices.tolist():
            label_name = LABEL_NAMES[int(label_index)]
            per_label_neighbor_prevalence[label_name].append(float(prevalence_vector[label_index]))
            per_label_chance_prevalence[label_name].append(float(global_prevalence[label_index]))

        for neighbor_row in neighbor_label_matrix:
            positive_jaccard_values.append(jaccard_similarity(query_labels, neighbor_row))
        for random_row in random_label_matrix:
            random_jaccard_values.append(jaccard_similarity(query_labels, random_row))

        query_summaries.append(
            {
                "query_index": int(query_index),
                "query_example_id": example_ids[query_index],
                "query_labels": labels_to_names(labels[query_index]),
                "neighbor_example_ids": [example_ids[index] for index in neighbor_indices],
                "neighbor_similarities": [float(score) for score in neighbor_scores[row_index]],
            }
        )

    per_label_summary: dict[str, Any] = {}
    for label_name in LABEL_NAMES:
        observed_values = per_label_neighbor_prevalence.get(label_name, [])
        chance_values = per_label_chance_prevalence.get(label_name, [])
        observed_mean = float(np.mean(observed_values)) if observed_values else None
        chance_mean = float(np.mean(chance_values)) if chance_values else float(global_prevalence[LABEL_NAMES.index(label_name)])
        lift = None
        if observed_mean is not None and chance_mean > 0.0:
            lift = float(observed_mean / chance_mean)
        per_label_summary[label_name] = {
            "positive_queries": int(len(observed_values)),
            "mean_neighbor_prevalence": observed_mean,
            "chance_prevalence": chance_mean,
            "prevalence_lift": lift,
        }

    observed_prevalence = float(np.mean(positive_prevalence_values)) if positive_prevalence_values else None
    chance_prevalence = float(np.mean(chance_prevalence_values)) if chance_prevalence_values else None
    prevalence_lift = None
    if observed_prevalence is not None and chance_prevalence and chance_prevalence > 0.0:
        prevalence_lift = float(observed_prevalence / chance_prevalence)

    observed_jaccard = float(np.mean(positive_jaccard_values)) if positive_jaccard_values else None
    random_jaccard = float(np.mean(random_jaccard_values)) if random_jaccard_values else None
    jaccard_lift = None
    if observed_jaccard is not None and random_jaccard and random_jaccard > 0.0:
        jaccard_lift = float(observed_jaccard / random_jaccard)

    return {
        "num_queries": int(query_indices.shape[0]),
        "positive_query_count": int(positive_query_count),
        "top_k": int(top_k),
        "global_label_prevalence": {name: float(global_prevalence[index]) for index, name in enumerate(LABEL_NAMES)},
        "mean_query_positive_neighbor_prevalence": observed_prevalence,
        "chance_query_positive_prevalence": chance_prevalence,
        "positive_prevalence_lift": prevalence_lift,
        "mean_positive_jaccard": observed_jaccard,
        "mean_random_positive_jaccard": random_jaccard,
        "positive_jaccard_lift": jaccard_lift,
        "per_label": per_label_summary,
        "sample_queries": query_summaries[:10],
    }


def build_qualitative_neighbors(
    index: faiss.IndexFlatIP,
    normalized_embeddings: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, str]],
    example_ids: list[str],
    image_paths: list[str],
    sample_size: int,
    top_k: int,
    seed: int,
) -> list[dict[str, Any]]:
    query_indices = sample_indices(normalized_embeddings.shape[0], sample_size, seed)
    query_vectors = normalized_embeddings[query_indices]
    neighbors, neighbor_scores = search_without_self(index, query_vectors, query_indices, top_k=top_k)

    samples: list[dict[str, Any]] = []
    for row_index, query_index in enumerate(query_indices.tolist()):
        query_row = rows[query_index]
        entry = {
            "query": {
                "index": int(query_index),
                "example_id": example_ids[query_index],
                "file_id": Path(image_paths[query_index]).name,
                "image_path": image_paths[query_index],
                "labels": labels_to_names(labels[query_index]),
                "metadata": {
                    "patient_id": query_row.get("patient_id"),
                    "study_id": query_row.get("study_id"),
                    "sex": query_row.get("sex"),
                    "age": query_row.get("age"),
                    "view_raw": query_row.get("view_raw"),
                    "view_group": query_row.get("view_group"),
                },
            },
            "neighbors": [],
        }
        for rank, (neighbor_index, similarity) in enumerate(
            zip(neighbors[row_index], neighbor_scores[row_index]),
            start=1,
        ):
            neighbor_row = rows[neighbor_index]
            entry["neighbors"].append(
                {
                    "rank": int(rank),
                    "index": int(neighbor_index),
                    "example_id": example_ids[neighbor_index],
                    "file_id": Path(image_paths[neighbor_index]).name,
                    "image_path": image_paths[neighbor_index],
                    "similarity": float(similarity),
                    "labels": labels_to_names(labels[neighbor_index]),
                    "metadata": {
                        "patient_id": neighbor_row.get("patient_id"),
                        "study_id": neighbor_row.get("study_id"),
                        "sex": neighbor_row.get("sex"),
                        "age": neighbor_row.get("age"),
                        "view_raw": neighbor_row.get("view_raw"),
                        "view_group": neighbor_row.get("view_group"),
                    },
                }
            )
        samples.append(entry)
    return samples


def write_items_jsonl(path: Path, rows: list[dict[str, str]], image_paths: list[str], example_ids: list[str], labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, (row, image_path, example_id, label_row) in enumerate(zip(rows, image_paths, example_ids, labels)):
            payload = {
                "index": int(index),
                "example_id": example_id,
                "file_id": Path(image_path).name,
                "image_path": image_path,
                "labels": labels_to_names(label_row),
                "metadata": {
                    key: row.get(key)
                    for key in ("dataset", "split", "patient_id", "study_id", "view_raw", "view_group", "sex", "age")
                },
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fused_embeddings = load_embedding_array(args.fused_embeddings.resolve())
    fused_image_paths = read_lines(args.fused_image_paths.resolve())
    manifest_rows = read_manifest_rows(args.manifest_csv.resolve(), args.split)
    ordered_rows = reorder_rows_by_image_paths(manifest_rows, fused_image_paths, args.split)
    validate_label_alignment(ordered_rows, fused_image_paths, fused_embeddings, args.split)
    labels = build_labels(ordered_rows)

    image_embeddings, image_alignment = align_image_embeddings(args.image_embeddings_dir.resolve(), fused_image_paths)
    text_embeddings, example_ids, text_alignment = align_text_embeddings(args.text_embeddings_dir.resolve(), fused_image_paths)
    fusion_consistency = validate_fusion_consistency(fused_embeddings, image_embeddings, text_embeddings)

    normalized_embeddings, norm_summary = normalize_rows(fused_embeddings)
    index = build_faiss_index(normalized_embeddings)

    self_retrieval = run_self_retrieval_check(
        index=index,
        normalized_embeddings=normalized_embeddings,
        labels=labels,
        example_ids=example_ids,
        sample_size=args.self_retrieval_sample_size,
        seed=args.seed,
    )
    label_agreement = run_label_agreement_check(
        index=index,
        normalized_embeddings=normalized_embeddings,
        labels=labels,
        example_ids=example_ids,
        sample_size=args.label_agreement_queries,
        top_k=args.top_k,
        seed=args.seed + 1,
    )
    qualitative_neighbors = build_qualitative_neighbors(
        index=index,
        normalized_embeddings=normalized_embeddings,
        labels=labels,
        rows=ordered_rows,
        example_ids=example_ids,
        image_paths=fused_image_paths,
        sample_size=args.qualitative_queries,
        top_k=args.top_k,
        seed=args.seed + 2,
    )

    embeddings_path = output_dir / "embeddings.npy"
    labels_path = output_dir / "labels.npy"
    image_embeddings_path = output_dir / "image_embeddings.npy"
    text_embeddings_path = output_dir / "text_embeddings.npy"
    example_ids_path = output_dir / "example_ids.json"
    image_paths_path = output_dir / "image_paths.txt"
    items_path = output_dir / "items.jsonl"
    metadata_path = output_dir / "metadata.json"
    sanity_path = output_dir / "sanity_report.json"
    qualitative_path = output_dir / "qualitative_neighbors.json"
    index_path = output_dir / "index.faiss"

    np.save(embeddings_path, normalized_embeddings)
    np.save(labels_path, labels.astype(np.float32))
    np.save(image_embeddings_path, image_embeddings.astype(np.float32))
    np.save(text_embeddings_path, text_embeddings.astype(np.float32))
    example_ids_path.write_text(json.dumps(example_ids, indent=2), encoding="utf-8")
    image_paths_path.write_text("\n".join(fused_image_paths) + "\n", encoding="utf-8")
    write_items_jsonl(items_path, ordered_rows, fused_image_paths, example_ids, labels)
    faiss.write_index(index, str(index_path))

    sanity_report = {
        "embedding_sanity": {
            "num_rows": int(normalized_embeddings.shape[0]),
            "embedding_dim": int(normalized_embeddings.shape[1]),
            "no_nan_or_inf": bool(np.isfinite(normalized_embeddings).all()),
            "norm_summary": norm_summary,
        },
        "self_retrieval": self_retrieval,
        "label_agreement": label_agreement,
        "qualitative_neighbors_path": str(qualitative_path),
    }
    write_json(sanity_path, sanity_report)
    write_json(qualitative_path, qualitative_neighbors)

    baseline_config = read_json(args.baseline_config.resolve()) if args.baseline_config.exists() else None
    fused_run_meta = read_json(args.fused_run_meta.resolve()) if args.fused_run_meta.exists() else None

    metadata = {
        "split": args.split,
        "row_count": int(normalized_embeddings.shape[0]),
        "embedding_dim": int(normalized_embeddings.shape[1]),
        "image_embedding_dim": int(image_embeddings.shape[1]),
        "text_embedding_dim": int(text_embeddings.shape[1]),
        "label_names": LABEL_NAMES,
        "index": {"type": "faiss.IndexFlatIP", "metric": "inner_product_on_l2_normalized_vectors"},
        "source_files": {
            "fused_embeddings": args.fused_embeddings.resolve(),
            "fused_image_paths": args.fused_image_paths.resolve(),
            "fused_run_meta": args.fused_run_meta.resolve(),
            "manifest_csv": args.manifest_csv.resolve(),
            "image_embeddings": args.image_embeddings_dir.resolve() / "embeddings.npy",
            "image_paths": args.image_embeddings_dir.resolve() / "image_paths.txt",
            "text_embeddings": args.text_embeddings_dir.resolve() / "embeddings.npy",
            "text_report_ids": args.text_embeddings_dir.resolve() / "report_ids.json",
            "baseline_config": args.baseline_config.resolve(),
        },
        "artifacts": {
            "embeddings": embeddings_path,
            "labels": labels_path,
            "image_embeddings": image_embeddings_path,
            "text_embeddings": text_embeddings_path,
            "example_ids": example_ids_path,
            "image_paths": image_paths_path,
            "items": items_path,
            "metadata": metadata_path,
            "sanity_report": sanity_path,
            "qualitative_neighbors": qualitative_path,
            "index": index_path,
        },
        "alignment_checks": {
            "labels_aligned_with_rows": True,
            "image_embeddings_alignment": image_alignment,
            "text_embeddings_alignment": text_alignment,
            "fusion_consistency": fusion_consistency,
        },
        "baseline_config_excerpt": baseline_config,
        "fused_run_meta": fused_run_meta,
    }
    write_json(metadata_path, metadata)

    print(f"[saved] memory_dir={output_dir}")
    print(
        "[embedding_sanity] "
        f"rows={normalized_embeddings.shape[0]} dim={normalized_embeddings.shape[1]} "
        f"norm_mean={norm_summary['mean']:.8f} norm_std={norm_summary['std']:.8f} "
        f"norm_min={norm_summary['min']:.8f} norm_max={norm_summary['max']:.8f}"
    )
    print(
        "[self_retrieval] "
        f"sample_size={self_retrieval['sample_size']} "
        f"top1_self_hit_rate={self_retrieval['top1_self_hit_rate']:.6f} "
        f"top5_contains_self_rate={self_retrieval['top5_contains_self_rate']:.6f}"
    )
    print(
        "[label_agreement] "
        f"queries={label_agreement['num_queries']} positive_queries={label_agreement['positive_query_count']} "
        f"top_k={label_agreement['top_k']} "
        f"positive_prevalence_lift={label_agreement['positive_prevalence_lift']:.6f} "
        f"positive_jaccard={label_agreement['mean_positive_jaccard']:.6f} "
        f"random_jaccard={label_agreement['mean_random_positive_jaccard']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
