#!/usr/bin/env python3
"""Train the shared frozen linear classifier on ViT-base fused CLS embeddings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_TRAIN_EMBEDDINGS = Path("/workspace/fused_embeddings_cls/vit-base-patch16-224/train/embeddings.npy")
DEFAULT_VAL_EMBEDDINGS = Path("/workspace/fused_embeddings_cls/vit-base-patch16-224/val/embeddings.npy")
DEFAULT_TEST_EMBEDDINGS = Path("/workspace/fused_embeddings_cls/vit-base-patch16-224/test/embeddings.npy")

DEFAULT_TRAIN_IMAGE_PATHS = Path("/workspace/fused_embeddings_cls/vit-base-patch16-224/train/image_paths.txt")
DEFAULT_VAL_IMAGE_PATHS = Path("/workspace/fused_embeddings_cls/vit-base-patch16-224/val/image_paths.txt")
DEFAULT_TEST_IMAGE_PATHS = Path("/workspace/fused_embeddings_cls/vit-base-patch16-224/test/image_paths.txt")

DEFAULT_OUTPUT_ROOT = Path("/workspace/outputs/models/nih_cxr14/fused/linear_cls_vit_base")
BASE_SCRIPT = Path(__file__).with_name("train_nih_frozen_dinov2_linear.py")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the shared frozen NIH linear trainer with ViT-base fused CLS embeddings "
            "so the training procedure matches the DINOv2 fused baseline."
        )
    )
    parser.add_argument("--train-embeddings", type=Path, default=DEFAULT_TRAIN_EMBEDDINGS)
    parser.add_argument("--val-embeddings", type=Path, default=DEFAULT_VAL_EMBEDDINGS)
    parser.add_argument("--test-embeddings", type=Path, default=DEFAULT_TEST_EMBEDDINGS)
    parser.add_argument("--train-image-paths", type=Path, default=DEFAULT_TRAIN_IMAGE_PATHS)
    parser.add_argument("--val-image-paths", type=Path, default=DEFAULT_VAL_IMAGE_PATHS)
    parser.add_argument("--test-image-paths", type=Path, default=DEFAULT_TEST_IMAGE_PATHS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_known_args()


def main() -> int:
    args, forwarded_args = parse_args()

    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f"Shared training script not found: {BASE_SCRIPT}")

    command = [
        sys.executable,
        str(BASE_SCRIPT),
        "--train-embeddings",
        str(args.train_embeddings),
        "--val-embeddings",
        str(args.val_embeddings),
        "--test-embeddings",
        str(args.test_embeddings),
        "--train-image-paths",
        str(args.train_image_paths),
        "--val-image-paths",
        str(args.val_image_paths),
        "--test-image-paths",
        str(args.test_image_paths),
        "--output-root",
        str(args.output_root),
        *forwarded_args,
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
