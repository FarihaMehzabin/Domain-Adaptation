#!/usr/bin/env python3
"""Numbered entrypoint for NIH split report embedding generation."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("generate_nih_split_report_embeddings.py")), run_name="__main__")
