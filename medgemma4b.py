#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image
import torch

HF_ROOT = Path("/workspace/hf_cache")
HF_HUB_CACHE = HF_ROOT / "hub"
MODEL_ID = "google/medgemma-4b-it"
DATA_ROOT = Path("/workspace/data")
REFERENCE_REPORTS_DIR = Path("/workspace/(partial)_Report_of_(Training-Split)_of_NIH_Dataset")
OUTPUT_ROOT = Path("/workspace/outputs/4B")
DEFAULT_LOG_PATH = Path("/workspace/medgemma4b.log")
SYSTEM = "You are an expert, Board-Certified Thoracic Radiologist."
COMPACT_PROMPT_PATH = Path("/workspace/prompt.txt")
VERBOSE_PROMPT_PATH = Path("/workspace/prompt_verbose.txt")
DEFAULT_SPLITS = ["val", "test"]
DEFAULT_BATCH_BENCHMARKS = [1, 2, 4, 6, 8]
DEFAULT_RUNTIME_TARGETS = [512, 935, 1200]
DEFAULT_PROBE_TOKENS = 256
DEFAULT_MAX_NEW_TOKENS = 1200
SMOKE_TEST_IMAGES = 1
GEN_KWARGS = {"do_sample": False}
SECTION_HEADERS = [
    "1) EXAM / TECHNIQUE",
    "2) IMAGE QUALITY / TECHNICAL FACTORS",
    "3) DEVICES / LINES / TUBES",
    "4) FINDINGS",
    "5) IMPRESSION",
    "6) LIMITATIONS / UNCERTAINTY",
]

os.environ["HF_HOME"] = str(HF_ROOT)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate NIH CXR14 reports with MedGemma 4B IT.")
    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser("generate", help="Generate reports for NIH CXR14 splits.")
    add_generation_args(generate_parser)

    validate_parser = subparsers.add_parser("validate-runtime", help="Validate a runtime against long-report targets.")
    add_common_args(validate_parser)
    add_prompt_args(validate_parser)
    add_runtime_args(validate_parser)
    validate_parser.add_argument("--validation-split", choices=["train", "val", "test"], default="val")
    validate_parser.add_argument("--validation-index", type=int, default=0)
    validate_parser.add_argument(
        "--validation-targets",
        nargs="+",
        type=int,
        default=DEFAULT_RUNTIME_TARGETS,
        help="max_new_tokens targets to test in order.",
    )
    validate_parser.add_argument("--runtime-name", default="runtime")
    validate_parser.add_argument("--output-json", type=Path)

    benchmark_parser = subparsers.add_parser("benchmark-batch", help="Benchmark batched generation throughput.")
    add_common_args(benchmark_parser)
    add_prompt_args(benchmark_parser)
    add_runtime_args(benchmark_parser)
    benchmark_parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    benchmark_parser.add_argument("--sample-size", type=int, default=20)
    benchmark_parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_BATCH_BENCHMARKS,
        help="Batch sizes to benchmark.",
    )
    benchmark_parser.add_argument("--probe-tokens", type=int, default=DEFAULT_PROBE_TOKENS)
    benchmark_parser.add_argument("--final-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    benchmark_parser.add_argument("--output-json", type=Path)

    parser.set_defaults(command="generate")
    args = parser.parse_args()
    if len(os.sys.argv) == 1:
        args.command = "generate"
    return args


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--reference-reports-dir", type=Path, default=REFERENCE_REPORTS_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)



def add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt-mode", choices=["compact", "verbose"], default="compact")
    parser.add_argument("--prompt-file", type=Path, help="Explicit prompt file path. Overrides --prompt-mode.")



def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
        help="Attention runtime to request from Transformers.",
    )
    parser.add_argument(
        "--skip-budget-guard",
        action="store_true",
        help="Skip the sliding-window preflight guard.",
    )



def add_generation_args(parser: argparse.ArgumentParser) -> None:
    add_common_args(parser)
    add_prompt_args(parser)
    add_runtime_args(parser)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=DEFAULT_SPLITS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--smoke-test-images", type=int, default=SMOKE_TEST_IMAGES)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--no-skip-existing", action="store_true")



def log(message: str, log_file) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} {message}"
    print(line, flush=True)
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()



def prompt_path_for_mode(prompt_mode: str) -> Path:
    if prompt_mode == "verbose":
        return VERBOSE_PROMPT_PATH
    return COMPACT_PROMPT_PATH



def resolve_prompt_text(prompt_mode: str, prompt_file: Path | None) -> tuple[str, Path]:
    prompt_path = prompt_file or prompt_path_for_mode(prompt_mode)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8"), prompt_path



def resolve_image_path(data_root: Path, image_path: str) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate
    return data_root / candidate



def split_csv_path(data_root: Path, split: str) -> Path:
    return data_root / "nih_cxr14" / "splits" / f"{split}.csv"



def load_split_images(data_root: Path, splits: Iterable[str]) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    for split in splits:
        csv_path = split_csv_path(data_root, split)
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("dataset") != "nih_cxr14":
                    continue
                image_path = row.get("image_path")
                if not image_path:
                    continue
                selected.append((split, resolve_image_path(data_root, image_path)))

    if not selected:
        joined = ", ".join(splits)
        raise ValueError(f"No NIH CXR14 images found for splits: {joined}")

    return selected



def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"



def generation_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16



def load_processor(model_id: str) -> AutoProcessor:
    return AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(HF_HUB_CACHE),
        local_files_only=True,
        use_fast=False,
    )



def load_model(
    model_id: str,
    attn_implementation: str = "eager",
) -> AutoModelForImageTextToText:
    torch.backends.cuda.matmul.allow_tf32 = True
    return AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=generation_dtype(),
        device_map="auto",
        cache_dir=str(HF_HUB_CACHE),
        local_files_only=True,
        attn_implementation=attn_implementation,
    )



def load_model_config(model_id: str) -> AutoConfig:
    return AutoConfig.from_pretrained(model_id, cache_dir=str(HF_HUB_CACHE), local_files_only=True)



def reference_report_paths(reference_reports_dir: Path) -> list[Path]:
    if not reference_reports_dir.exists():
        raise FileNotFoundError(f"Reference report directory not found: {reference_reports_dir}")

    paths = sorted(
        path for path in reference_reports_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"
    )
    if not paths:
        raise ValueError(f"No .txt reference reports found in {reference_reports_dir}")
    return paths



def compute_reference_token_stats(
    reference_reports_dir: Path,
    processor: AutoProcessor,
    log_file=None,
) -> dict[str, object]:
    rows: list[dict[str, int | str]] = []
    counts: list[int] = []
    for report_path in reference_report_paths(reference_reports_dir):
        text = report_path.read_text(encoding="utf-8").strip()
        token_count = len(processor.tokenizer(text, add_special_tokens=False).input_ids)
        rows.append({"name": report_path.name, "tokens": token_count})
        counts.append(token_count)
        if log_file is not None:
            log(f"[INFO] Reference report tokens: {report_path.name} -> {token_count}", log_file)

    max_new_tokens = max(counts)
    stats = {
        "count": len(counts),
        "max_new_tokens": max_new_tokens,
        "min_tokens": min(counts),
        "rows": rows,
    }
    if log_file is not None:
        log(
            f"[INFO] Reference token range: min={stats['min_tokens']} max={max_new_tokens} count={stats['count']}",
            log_file,
        )
    return stats



def ensure_cache_layout() -> None:
    HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)



def build_messages(image_paths: Sequence[Path], prompt_text: str) -> list[list[dict[str, object]]]:
    conversations: list[list[dict[str, object]]] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            conversations.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": rgb_image},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                ]
            )
    return conversations



def prepare_batch_inputs(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    image_paths: Sequence[Path],
    prompt_text: str,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    conversations = build_messages(image_paths, prompt_text)
    payload = conversations[0] if len(conversations) == 1 else conversations
    inputs = processor.apply_chat_template(
        payload,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(model.device)

    for key, value in list(inputs.items()):
        if torch.is_tensor(value) and value.is_floating_point():
            inputs[key] = value.to(dtype=generation_dtype())

    return inputs, [int(length) for length in prompt_lengths]



def generate_reports_batch(
    image_paths: Sequence[Path],
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    prompt_text: str,
    max_new_tokens: int,
) -> list[str]:
    inputs, prompt_lengths = prepare_batch_inputs(model, processor, image_paths, prompt_text)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, **GEN_KWARGS)

    reports: list[str] = []
    for idx, prompt_length in enumerate(prompt_lengths):
        reports.append(processor.decode(output[idx][prompt_length:], skip_special_tokens=True).strip())
    return reports



def batched(items: Sequence[tuple[str, Path]], batch_size: int) -> Iterable[list[tuple[str, Path]]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])



def estimate_input_tokens(
    processor: AutoProcessor,
    image_path: Path,
    prompt_text: str,
) -> int:
    messages = build_messages([image_path], prompt_text)[0]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return int(inputs["attention_mask"].sum().item())



def preflight_generation_budget(
    model_id: str,
    processor: AutoProcessor,
    image_path: Path,
    prompt_text: str,
    max_new_tokens: int,
    attn_implementation: str,
    skip_budget_guard: bool,
) -> dict[str, int | str | None]:
    config = load_model_config(model_id)
    text_config = getattr(config, "text_config", config)
    sliding_window = getattr(text_config, "sliding_window", None)
    input_tokens = estimate_input_tokens(processor, image_path, prompt_text)
    budget = {
        "input_tokens": input_tokens,
        "sliding_window": sliding_window,
        "attn_implementation": attn_implementation,
    }
    if skip_budget_guard or attn_implementation == "flash_attention_2":
        return budget
    if sliding_window is not None and input_tokens + max_new_tokens > int(sliding_window):
        raise ValueError(
            "Requested max_new_tokens is incompatible with the current runtime budget. "
            f"input_tokens={input_tokens}, max_new_tokens={max_new_tokens}, sliding_window={sliding_window}, "
            f"attn_implementation={attn_implementation}. Use a different runtime, a shorter prompt, or --skip-budget-guard."
        )
    return budget



def validate_runtime(args: argparse.Namespace) -> int:
    ensure_cache_layout()
    prompt_text, prompt_path = resolve_prompt_text(args.prompt_mode, args.prompt_file)
    processor = load_processor(args.model_id)
    items = load_split_images(args.data_root, [args.validation_split])
    if args.validation_index >= len(items):
        raise IndexError(f"validation-index {args.validation_index} is out of range for {len(items)} item(s)")

    _, image_path = items[args.validation_index]
    budget = preflight_generation_budget(
        args.model_id,
        processor,
        image_path,
        prompt_text,
        max(args.validation_targets),
        args.attn_implementation,
        args.skip_budget_guard,
    )
    model = load_model(args.model_id, attn_implementation=args.attn_implementation)

    results: list[dict[str, object]] = []
    passed = True
    for target in args.validation_targets:
        started = time.time()
        try:
            reports = generate_reports_batch([image_path], model, processor, prompt_text, target)
            elapsed = time.time() - started
            results.append(
                {
                    "max_new_tokens": target,
                    "status": "ok",
                    "elapsed_seconds": elapsed,
                    "generated_chars": len(reports[0]),
                    "generated_tokens_estimate": len(processor.tokenizer(reports[0], add_special_tokens=False).input_ids),
                }
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - started
            results.append(
                {
                    "max_new_tokens": target,
                    "status": "error",
                    "elapsed_seconds": elapsed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            passed = False
            break

    payload = {
        "runtime_name": args.runtime_name,
        "model_id": args.model_id,
        "prompt_path": str(prompt_path),
        "prompt_mode": args.prompt_mode,
        "attn_implementation": args.attn_implementation,
        "budget": budget,
        "validation_image": str(image_path),
        "targets": results,
        "passed": passed and len(results) == len(args.validation_targets),
    }
    print(json.dumps(payload, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1



def benchmark_batch_sizes(args: argparse.Namespace) -> int:
    ensure_cache_layout()
    prompt_text, prompt_path = resolve_prompt_text(args.prompt_mode, args.prompt_file)
    processor = load_processor(args.model_id)
    items = load_split_images(args.data_root, [args.split])[: args.sample_size]
    if not items:
        raise ValueError("No images available for benchmark")

    sample_image = items[0][1]
    phases = [("probe", args.probe_tokens), ("final", args.final_tokens)]
    results: dict[str, object] = {
        "prompt_path": str(prompt_path),
        "attn_implementation": args.attn_implementation,
        "sample_size": len(items),
        "phases": {},
    }

    model = load_model(args.model_id, attn_implementation=args.attn_implementation)
    model.eval()

    for phase_name, token_budget in phases:
        try:
            budget = preflight_generation_budget(
                args.model_id,
                processor,
                sample_image,
                prompt_text,
                token_budget,
                args.attn_implementation,
                args.skip_budget_guard,
            )
        except Exception as exc:  # noqa: BLE001
            results["phases"][phase_name] = {
                "max_new_tokens": token_budget,
                "status": "skipped",
                "reason": str(exc),
            }
            continue

        phase_rows: list[dict[str, object]] = []
        for batch_size in args.batch_sizes:
            total_elapsed = 0.0
            processed = 0
            peak_gib = 0.0
            error: str | None = None
            for batch in batched(items, batch_size):
                batch_paths = [image_path for _, image_path in batch]
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    _ = generate_reports_batch(batch_paths, model, processor, prompt_text, token_budget)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        peak_gib = max(peak_gib, torch.cuda.max_memory_allocated() / (1024 ** 3))
                    total_elapsed += time.perf_counter() - started
                    processed += len(batch_paths)
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    break

            row: dict[str, object] = {
                "batch_size": batch_size,
                "max_new_tokens": token_budget,
                "status": "ok" if error is None else "error",
                "images_processed": processed,
                "elapsed_seconds": total_elapsed,
                "images_per_minute": (processed / total_elapsed) * 60 if processed and total_elapsed else 0.0,
                "seconds_per_image": total_elapsed / processed if processed else None,
                "peak_memory_gib": peak_gib if peak_gib else None,
                "budget": budget,
            }
            if error is not None:
                row["error"] = error
            phase_rows.append(row)

        ok_rows = [row for row in phase_rows if row["status"] == "ok"]
        best = max(ok_rows, key=lambda row: float(row["images_per_minute"])) if ok_rows else None
        results["phases"][phase_name] = {
            "max_new_tokens": token_budget,
            "status": "ok",
            "rows": phase_rows,
            "best": best,
        }

    print(json.dumps(results, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0



def generate_command(args: argparse.Namespace) -> int:
    ensure_cache_layout()
    prompt_text, prompt_path = resolve_prompt_text(args.prompt_mode, args.prompt_file)

    with args.log_path.open("a", encoding="utf-8") as log_file:
        log("[INFO] Starting run", log_file)
        log(f"[INFO] Requested splits: {', '.join(args.splits)}", log_file)
        log(f"[INFO] Prompt path: {prompt_path}", log_file)
        log(f"[INFO] Attention runtime: {args.attn_implementation}", log_file)
        log(f"[INFO] Batch size: {args.batch_size}", log_file)
        log(f"[INFO] Requested max_new_tokens: {args.max_new_tokens}", log_file)

        processor = load_processor(args.model_id)
        reference_stats = compute_reference_token_stats(args.reference_reports_dir, processor, log_file=log_file)
        log(
            f"[INFO] Reference max token length: {reference_stats['max_new_tokens']}",
            log_file,
        )

        items = load_split_images(args.data_root, args.splits)
        if args.max_images is not None:
            items = items[: args.max_images]
        missing = [path for _, path in items if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} image(s); first missing path: {missing[0]}")
        if not items:
            raise ValueError("No images selected for generation")

        budget = preflight_generation_budget(
            args.model_id,
            processor,
            items[0][1],
            prompt_text,
            args.max_new_tokens,
            args.attn_implementation,
            args.skip_budget_guard,
        )
        log(
            f"[INFO] Prompt input tokens: {budget['input_tokens']} sliding_window={budget['sliding_window']}",
            log_file,
        )

        model = load_model(args.model_id, attn_implementation=args.attn_implementation)
        model.eval()

        if args.smoke_test_images > 0:
            smoke_batch = items[: min(args.smoke_test_images, args.batch_size)]
            log(f"[INFO] Smoke test: {len(smoke_batch)} image(s)", log_file)
            started = time.time()
            smoke_paths = [image_path for _, image_path in smoke_batch]
            _ = generate_reports_batch(smoke_paths, model, processor, prompt_text, args.max_new_tokens)
            elapsed = time.time() - started
            log(f"[INFO] Smoke test OK ({format_seconds(elapsed)})", log_file)

        args.output_root.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        processed = 0
        skipped = 0
        failed = 0
        failures: list[str] = []
        skip_existing = not args.no_skip_existing

        for batch in batched(items, args.batch_size):
            split_out_dirs: dict[str, Path] = {}
            batch_targets: list[tuple[str, Path, Path]] = []
            batch_paths: list[Path] = []
            for split, image_path in batch:
                split_out_dir = split_out_dirs.setdefault(split, args.output_root / split)
                split_out_dir.mkdir(parents=True, exist_ok=True)
                out_path = split_out_dir / f"{image_path.stem}.txt"
                if skip_existing and out_path.exists():
                    skipped += 1
                    continue
                batch_targets.append((split, image_path, out_path))
                batch_paths.append(image_path)

            if not batch_targets:
                continue

            batch_index = processed + skipped + failed + 1
            log(
                f"[INFO] Processing batch starting at item {batch_index} with {len(batch_targets)} image(s)",
                log_file,
            )
            try:
                reports = generate_reports_batch(batch_paths, model, processor, prompt_text, args.max_new_tokens)
                for (split, image_path, out_path), report in zip(batch_targets, reports, strict=True):
                    out_path.write_text(report + "\n", encoding="utf-8")
                    processed += 1
                elapsed = time.time() - start_time
                avg = elapsed / processed if processed else 0.0
                remaining = avg * max(len(items) - (processed + skipped + failed), 0)
                log(
                    f"[INFO] Done {processed}, skipped {skipped}, failed {failed}. "
                    f"Elapsed {format_seconds(elapsed)} ETA {format_seconds(remaining)}",
                    log_file,
                )
            except Exception as exc:  # noqa: BLE001
                failed += len(batch_targets)
                for split, image_path, _ in batch_targets:
                    failures.append(f"[{split}] {image_path} :: {exc}")
                log(f"[ERROR] Failed batch: {exc}", log_file)

        log(
            f"[INFO] Finished. Processed {processed}, skipped {skipped}, failed {failed}.",
            log_file,
        )
        if failures:
            log("[INFO] Failure list:", log_file)
            for item in failures:
                log(f"[INFO] {item}", log_file)

    return 0



def main() -> int:
    args = parse_args()
    if args.command == "generate":
        return generate_command(args)
    if args.command == "validate-runtime":
        return validate_runtime(args)
    if args.command == "benchmark-batch":
        return benchmark_batch_sizes(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
