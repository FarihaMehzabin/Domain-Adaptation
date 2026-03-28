#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers.cache_utils import HybridCache
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "google/medgemma-4b-it"
WORKSPACE_ROOT = Path("/workspace")
PROMPT_FILE = WORKSPACE_ROOT / "prompt.txt"
SPLIT_CSV = WORKSPACE_ROOT / "data" / "nih_cxr14" / "splits" / "train.csv"
DATASET_ROOT = WORKSPACE_ROOT / "data"
CACHE_DIR = WORKSPACE_ROOT / "hf_cache"
OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "fresh_medgemma_4b_it_one_shot_smoke"
DEFAULT_IMAGE_IDS = ["00000002_000", "00000005_001", "00000013_014"]
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"


@dataclass(frozen=True)
class TrainImage:
    image_id: str
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one-shot NIH chest X-ray reports with google/medgemma-4b-it "
            "using /workspace/prompt.txt."
        )
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=SPLIT_CSV,
        help="Split CSV to read image rows from. Defaults to the NIH train split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where <image_id>.txt outputs will be written.",
    )
    parser.add_argument(
        "--image-id",
        dest="image_ids",
        action="append",
        default=[],
        help="NIH image id stem such as 00000013_014. Repeatable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of images when selecting by split order.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset into the training split when selecting by split order.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output reports instead of skipping them.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of images to generate concurrently in each batch.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of new tokens to generate per report.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=DEFAULT_ATTN_IMPLEMENTATION,
        choices=["eager", "sdpa"],
        help="Attention implementation to use when loading the model.",
    )
    return parser.parse_args()


def read_prompt(prompt_file: Path) -> str:
    if not prompt_file.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return prompt_text


def load_train_images(split_csv: Path, dataset_root: Path) -> list[TrainImage]:
    if not split_csv.is_file():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    rows: list[TrainImage] = []
    with split_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel_path = row["image_path"]
            image_id = Path(rel_path).stem
            image_path = dataset_root / rel_path
            rows.append(TrainImage(image_id=image_id, image_path=image_path))
    if not rows:
        raise ValueError(f"No rows found in split CSV: {split_csv}")
    return rows


def select_images(
    rows: list[TrainImage],
    explicit_image_ids: list[str],
    limit: int | None,
    offset: int,
) -> list[TrainImage]:
    if offset < 0:
        raise ValueError("--offset must be >= 0")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be >= 0")

    by_id = {row.image_id: row for row in rows}
    if explicit_image_ids:
        missing = [image_id for image_id in explicit_image_ids if image_id not in by_id]
        if missing:
            missing_text = ", ".join(missing)
            raise KeyError(f"Image id(s) not found in selected split: {missing_text}")
        return [by_id[image_id] for image_id in explicit_image_ids]

    start = offset
    stop = None if limit is None else offset + limit
    selected = rows[start:stop]
    if not selected:
        raise ValueError("No images selected from the training split.")
    return selected


def resolve_hf_token() -> str:
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        return hf_token
    raise EnvironmentError(
        "HF_TOKEN is required to download google/medgemma-4b-it. "
        "Set HF_TOKEN after accepting the model access terms on Hugging Face."
    )


def load_model_and_processor(hf_token: str, attn_implementation: str):
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        token=hf_token,
        cache_dir=str(CACHE_DIR),
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        token=hf_token,
        cache_dir=str(CACHE_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    model.eval()
    return model, processor


def prepare_batch_inputs(
    processor,
    prompt_text: str,
    image_paths: list[Path],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    conversations = []
    images: list[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            image_rgb.load()
        images.append(image_rgb)
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image", "image": image_rgb},
                    ],
                }
            ]
        )

    encoded = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    inputs = {}
    for key, value in encoded.items():
        if torch.is_floating_point(value):
            inputs[key] = value.to(device=device, dtype=torch.bfloat16)
        else:
            inputs[key] = value.to(device=device)
    prompt_lengths = inputs["attention_mask"].sum(dim=-1).tolist()
    return inputs, [int(length) for length in prompt_lengths]


def decode_report(processor, generation_tokens: torch.Tensor, prompt_length: int) -> str:
    trimmed = generation_tokens[prompt_length:]
    report_text = processor.decode(trimmed, skip_special_tokens=True).strip()
    report_text = sanitize_report_text(report_text)
    if "7) DOMAIN VECTOR" not in report_text:
        report_text = f"{report_text}\n\n7) DOMAIN VECTOR".strip()
    return report_text + "\n"


def decode_batch_reports(
    processor,
    generation: torch.Tensor,
    prompt_lengths: list[int],
) -> list[str]:
    reports: list[str] = []
    for index, prompt_length in enumerate(prompt_lengths):
        reports.append(decode_report(processor, generation[index], prompt_length))
    return reports


def sanitize_report_text(report_text: str) -> str:
    sanitized_lines: list[str] = []
    for line in report_text.splitlines():
        stripped = line.strip().lower()

        if stripped.startswith("- projection:") and "label" in stripped and '"' in line:
            projection_match = re.search(r"projection:\s*([A-Za-z/]+)", line)
            projection = projection_match.group(1) if projection_match else "indeterminate"
            prefix = line.split("+", 1)[0].rstrip()
            line = f"{prefix} + visual appearance favors {projection} projection."

        if stripped.startswith("- portable-likelihood:") and '"' in line:
            line = re.sub(r'label\s*"[^"]+"', "burned-in label", line, flags=re.IGNORECASE)
            line = re.sub(r'labeled\s*"[^"]+"', "burned-in label", line, flags=re.IGNORECASE)

        if "markers/overlays:" in stripped:
            leading_whitespace = re.match(r"^\s*", line).group(0)
            body = line[len(leading_whitespace) :]
            body = re.sub(r'\(([LR])\)', "", body)
            body = re.sub(r'"[^"]+"', "", body)
            body = re.sub(r"\s{2,}", " ", body).rstrip(" ,.")
            body = re.sub(r"burned-in text\s+at", "burned-in text present at", body, flags=re.IGNORECASE)
            body = re.sub(r"burned-in label\s+at", "burned-in label present at", body, flags=re.IGNORECASE)
            if "burned-in text" in body.lower() and "present" not in body.lower():
                body = re.sub(r"burned-in text", "burned-in text present", body, flags=re.IGNORECASE)
            line = leading_whitespace + body

        sanitized_lines.append(line)

    return "\n".join(sanitized_lines).strip()


def build_generation_cache(model, batch_size: int, max_cache_len: int) -> HybridCache:
    text_config = getattr(model.config, "text_config", model.config)
    return HybridCache(
        config=text_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=model.device,
        dtype=model.dtype,
    )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def existing_outputs(output_dir: Path, rows: Iterable[TrainImage]) -> int:
    return sum(1 for row in rows if (output_dir / f"{row.image_id}.txt").exists())


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be >= 1")

    prompt_text = read_prompt(PROMPT_FILE)
    rows = load_train_images(args.split_csv, DATASET_ROOT)

    if args.image_ids:
        image_ids = args.image_ids
    elif args.split_csv == SPLIT_CSV and args.limit is None and args.offset == 0:
        image_ids = list(DEFAULT_IMAGE_IDS)
    else:
        image_ids = []
    selected = select_images(rows, image_ids, args.limit, args.offset)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected {len(selected)} image(s) from {args.split_csv}")
    print(f"Output directory: {output_dir}")
    if not args.overwrite:
        skipped = existing_outputs(output_dir, selected)
        if skipped:
            print(f"{skipped} existing output file(s) will be skipped.")
    total_batches = math.ceil(len(selected) / args.batch_size)
    print(f"Configured concurrent batch size: {args.batch_size}")
    print(f"Configured max_new_tokens: {args.max_new_tokens}")
    print(f"Configured attention implementation: {args.attn_implementation}")
    print(f"Total batches to process: {total_batches}")

    hf_token = resolve_hf_token()
    print(f"Loading model and processor: {MODEL_ID}")
    model, processor = load_model_and_processor(hf_token, args.attn_implementation)

    wrote = 0
    skipped = 0
    overall_start = time.monotonic()
    for batch_index in range(total_batches):
        batch_start = batch_index * args.batch_size
        batch_rows = selected[batch_start : batch_start + args.batch_size]
        batch_label = f"batch {batch_index + 1}/{total_batches}"
        batch_start_time = time.monotonic()
        print(
            f"Starting {batch_label}: {len(batch_rows)} image(s) "
            f"(overall {batch_start + 1}-{batch_start + len(batch_rows)} of {len(selected)})"
        )

        batch_wrote = 0
        batch_skipped = 0
        active_jobs: list[tuple[int, TrainImage]] = []
        for image_index, row in enumerate(batch_rows, start=1):
            overall_index = batch_start + image_index
            if not row.image_path.is_file():
                raise FileNotFoundError(f"Image not found for {row.image_id}: {row.image_path}")

            output_path = output_dir / f"{row.image_id}.txt"
            if output_path.exists() and not args.overwrite:
                print(
                    f"[{batch_label} | image {image_index}/{len(batch_rows)} | "
                    f"overall {overall_index}/{len(selected)}] Skipping existing report: {output_path.name}"
                )
                skipped += 1
                batch_skipped += 1
                continue
            active_jobs.append((overall_index, row))

        if active_jobs:
            print(
                f"[{batch_label}] Launching concurrent generation for "
                f"{len(active_jobs)} image(s)"
            )
            for image_index, (overall_index, row) in enumerate(active_jobs, start=1):
                print(
                    f"[{batch_label} | image {image_index}/{len(active_jobs)} | "
                    f"overall {overall_index}/{len(selected)}] Queued {row.image_id}"
                )

            try:
                inputs, prompt_lengths = prepare_batch_inputs(
                    processor=processor,
                    prompt_text=prompt_text,
                    image_paths=[row.image_path for _, row in active_jobs],
                    device=model.device,
                )
                generation_cache = build_generation_cache(
                    model=model,
                    batch_size=inputs["input_ids"].shape[0],
                    max_cache_len=max(prompt_lengths) + args.max_new_tokens,
                )

                with torch.inference_mode():
                    generation = model.generate(
                        **inputs,
                        past_key_values=generation_cache,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
            except torch.OutOfMemoryError as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OOM while generating {len(active_jobs)} images concurrently "
                    f"in {batch_label}. Try a smaller --batch-size or a different "
                    f"--attn-implementation."
                ) from exc

            reports = decode_batch_reports(processor, generation, prompt_lengths)
            for (_, row), report_text in zip(active_jobs, reports):
                output_path = output_dir / f"{row.image_id}.txt"
                ensure_parent_dir(output_path)
                output_path.write_text(report_text, encoding="utf-8")
                print(f"[{batch_label}] Wrote {output_path}")
                wrote += 1
                batch_wrote += 1

            del inputs
            del generation_cache
            del generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        batch_elapsed = time.monotonic() - batch_start_time
        print(
            f"Completed {batch_label}: wrote={batch_wrote} skipped={batch_skipped} "
            f"elapsed={batch_elapsed:.1f}s"
        )

    total_elapsed = time.monotonic() - overall_start
    print(f"Done. wrote={wrote} skipped={skipped} elapsed={total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
