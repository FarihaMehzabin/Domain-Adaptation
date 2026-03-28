#!/usr/bin/env python3
"""Fuse ResNet50 NIH image embeddings with CLS report embeddings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_IMAGE_ROOT = Path("/workspace/image_embeddings/resnet50")
DEFAULT_REPORT_TRAIN_DIR = Path("/workspace/report_embeddings_cls/train/microsoft__BiomedVLP-CXR-BERT-specialized")
DEFAULT_REPORT_VAL_DIR = Path("/workspace/report_embeddings_cls/val/microsoft__BiomedVLP-CXR-BERT-specialized")
DEFAULT_REPORT_TEST_DIR = Path("/workspace/report_embeddings_cls/test/microsoft__BiomedVLP-CXR-BERT-specialized")
DEFAULT_OUTPUT_ROOT = Path("/workspace/fused_embeddings_cls/resnet50")
BASE_SCRIPT = Path(__file__).with_name("generate_fused_nih_embeddings.py")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the shared NIH fused-embedding generator with ResNet50 image embeddings "
            "and CLS report embeddings."
        )
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--report-train-dir", type=Path, default=DEFAULT_REPORT_TRAIN_DIR)
    parser.add_argument("--report-val-dir", type=Path, default=DEFAULT_REPORT_VAL_DIR)
    parser.add_argument("--report-test-dir", type=Path, default=DEFAULT_REPORT_TEST_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_known_args()


def main() -> int:
    args, forwarded_args = parse_args()
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f"Shared fusion script not found: {BASE_SCRIPT}")

    command = [
        sys.executable,
        str(BASE_SCRIPT),
        "--image-root",
        str(args.image_root),
        "--report-train-dir",
        str(args.report_train_dir),
        "--report-val-dir",
        str(args.report_val_dir),
        "--report-test-dir",
        str(args.report_test_dir),
        "--output-root",
        str(args.output_root),
        *forwarded_args,
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
