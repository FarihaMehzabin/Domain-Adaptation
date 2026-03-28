#!/usr/bin/env python3
"""Fuse NIH CXR14 image + report embeddings into a single feature vector."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'numpy'. Install: pip install -r requirements/frozen_embeddings_classifier.txt"
    ) from exc


@dataclass
class SplitConfig:
    name: str
    image_dir: Path
    report_dir: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse NIH CXR14 DINOv2 image embeddings with report embeddings (concatenate)."
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("/workspace/image_embeddings/Dinov2-large"),
        help="Root directory with image embeddings splits.",
    )
    parser.add_argument(
        "--report-train-dir",
        type=Path,
        default=Path("/workspace/report_embeddings/train/BiomedVLP-CXR-BERT-specialized"),
    )
    parser.add_argument(
        "--report-val-dir",
        type=Path,
        default=Path("/workspace/report_embeddings/val/microsoft__BiomedVLP-CXR-BERT-specialized"),
    )
    parser.add_argument(
        "--report-test-dir",
        type=Path,
        default=Path("/workspace/report_embeddings/test/microsoft__BiomedVLP-CXR-BERT-specialized"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/fused_embeddings/Dinov2-large"),
        help="Output root for fused embeddings splits.",
    )
    return parser.parse_args()


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    head = path.read_bytes()[:64]
    return head.startswith(b"version https://git-lfs.github.com/spec/") or head.startswith(b"version https://")


def load_image_embeddings(image_dir: Path) -> tuple[np.ndarray, list[str]]:
    embeddings_path = image_dir / "embeddings.npy"
    image_paths_path = image_dir / "image_paths.txt"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Image embeddings not found: {embeddings_path}")
    if not image_paths_path.exists():
        raise FileNotFoundError(f"Image paths not found: {image_paths_path}")

    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D image embeddings at {embeddings_path}, found {embeddings.shape}.")

    image_paths = [line.strip() for line in image_paths_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(image_paths) != embeddings.shape[0]:
        raise ValueError(
            f"Image paths count ({len(image_paths)}) does not match embeddings rows ({embeddings.shape[0]})."
        )
    return np.asarray(embeddings, dtype=np.float32), image_paths


def load_report_embeddings(report_dir: Path) -> tuple[np.ndarray, list[str]]:
    embeddings_path = report_dir / "embeddings.npy"
    report_ids_path = report_dir / "report_ids.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Report embeddings not found: {embeddings_path}")
    if not report_ids_path.exists():
        raise FileNotFoundError(f"Report ids not found: {report_ids_path}")
    if is_lfs_pointer(embeddings_path):
        raise SystemExit(
            f"Report embeddings appear to be a Git LFS pointer: {embeddings_path}. "
            "Fetch the real file before fusing."
        )

    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D report embeddings at {embeddings_path}, found {embeddings.shape}.")

    report_ids = json.loads(report_ids_path.read_text(encoding="utf-8"))
    if len(report_ids) != embeddings.shape[0]:
        raise ValueError(
            f"Report ids count ({len(report_ids)}) does not match embeddings rows ({embeddings.shape[0]})."
        )
    return np.asarray(embeddings, dtype=np.float32), list(report_ids)


def build_report_index(report_ids: list[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for i, report_id in enumerate(report_ids):
        if report_id in index:
            raise ValueError(f"Duplicate report id detected: {report_id}")
        index[report_id] = i
    return index


def fuse_split(split: SplitConfig, output_root: Path) -> dict[str, Any]:
    image_embeddings, image_paths = load_image_embeddings(split.image_dir)
    report_embeddings, report_ids = load_report_embeddings(split.report_dir)

    report_index = build_report_index(report_ids)

    aligned_indices: list[int] = []
    missing_ids: list[str] = []
    for image_path in image_paths:
        report_id = Path(image_path).stem
        if report_id not in report_index:
            if len(missing_ids) < 5:
                missing_ids.append(report_id)
            continue
        aligned_indices.append(report_index[report_id])

    if len(aligned_indices) != len(image_paths):
        raise ValueError(
            f"{split.name}: missing report embeddings for {len(image_paths) - len(aligned_indices)} images. "
            f"Examples: {missing_ids}"
        )

    report_aligned = report_embeddings[aligned_indices]
    fused = np.concatenate([image_embeddings, report_aligned], axis=1)

    output_dir = output_root / split.name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", fused)
    (output_dir / "image_paths.txt").write_text("\n".join(image_paths) + "\n", encoding="utf-8")

    meta = {
        "run_date_utc": utc_now_iso(),
        "split": split.name,
        "image_embeddings_dir": str(split.image_dir),
        "report_embeddings_dir": str(split.report_dir),
        "output_dir": str(output_dir),
        "num_samples": int(fused.shape[0]),
        "image_dim": int(image_embeddings.shape[1]),
        "report_dim": int(report_aligned.shape[1]),
        "fused_dim": int(fused.shape[1]),
        "alignment": "image_basename_matches_report_id",
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta


def main() -> int:
    args = parse_args()

    splits = [
        SplitConfig("train", args.image_root / "train", args.report_train_dir),
        SplitConfig("val", args.image_root / "val", args.report_val_dir),
        SplitConfig("test", args.image_root / "test", args.report_test_dir),
    ]

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        meta = fuse_split(split, output_root)
        print(
            f"{split.name}: fused {meta['num_samples']} samples "
            f"(image_dim={meta['image_dim']}, report_dim={meta['report_dim']}, fused_dim={meta['fused_dim']})"
        )

    print(f"wrote fused embeddings to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
