#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

import medgemma4b

DEFAULT_OUTPUT_DIR = Path('/workspace/outputs/reference_eval')
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate MedGemma reports against uploaded reference reports.')
    parser.add_argument('--data-root', type=Path, default=medgemma4b.DATA_ROOT)
    parser.add_argument('--reference-reports-dir', type=Path, default=medgemma4b.REFERENCE_REPORTS_DIR)
    parser.add_argument('--model-id', default=medgemma4b.MODEL_ID)
    parser.add_argument('--prompt-mode', choices=['compact', 'verbose'], default='compact')
    parser.add_argument('--prompt-file', type=Path)
    parser.add_argument('--attn-implementation', choices=['eager', 'sdpa', 'flash_attention_2'], default='eager')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=medgemma4b.DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--embedding-model', default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument('--limit', type=int)
    return parser.parse_args()


def load_train_image_map(data_root: Path) -> dict[str, Path]:
    items = medgemma4b.load_split_images(data_root, ['train'])
    return {image_path.stem: image_path for _, image_path in items}



def load_reference_cases(reference_reports_dir: Path, data_root: Path, limit: int | None) -> list[dict[str, Path | str]]:
    image_map = load_train_image_map(data_root)
    cases: list[dict[str, Path | str]] = []
    for report_path in medgemma4b.reference_report_paths(reference_reports_dir):
        stem = report_path.stem
        image_path = image_map.get(stem)
        if image_path is None:
            continue
        cases.append({'stem': stem, 'image_path': image_path, 'reference_path': report_path})
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError('No reference cases were matched to training images.')
    return cases



def extract_section(text: str, header: str, next_header: str | None) -> str:
    start = text.find(header)
    if start == -1:
        return ''
    start += len(header)
    if next_header is None:
        return text[start:].strip()
    end = text.find(next_header, start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()



def section_compliance(text: str) -> tuple[float, bool, list[str]]:
    present = [header for header in medgemma4b.SECTION_HEADERS if header in text]
    ratio = len(present) / len(medgemma4b.SECTION_HEADERS)
    return ratio, len(present) == len(medgemma4b.SECTION_HEADERS), present



def length_score(length_ratio: float) -> float:
    return max(0.0, 1.0 - abs(1.0 - length_ratio))



def composite_score(row: dict[str, float | int | str]) -> float:
    structure = float(row['section_ratio'])
    content = (
        float(row['rougeL_full']) + float(row['rougeL_impression']) + float(row['semantic_full']) + float(row['semantic_impression'])
    ) / 4.0
    return (0.4 * structure) + (0.45 * content) + (0.15 * float(row['length_score']))



def pick_review_groups(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    sorted_rows = sorted(rows, key=lambda row: float(row['composite_score']), reverse=True)
    count = len(sorted_rows)
    middle_start = max((count // 2) - 1, 0)
    middle_end = min(middle_start + 3, count)
    return {
        'best': sorted_rows[:3],
        'middle': sorted_rows[middle_start:middle_end],
        'worst': sorted_rows[-3:],
    }



def write_review_markdown(review_path: Path, grouped_rows: dict[str, list[dict[str, object]]]) -> None:
    lines = ['# Reference Review']
    for label in ['best', 'middle', 'worst']:
        lines.append(f'\n## {label.title()} Cases')
        for row in grouped_rows[label]:
            lines.append(f"\n### {row['stem']}")
            lines.append(f"- Composite score: {float(row['composite_score']):.4f}")
            lines.append(f"- Section compliance: {row['section_ratio']} ({row['exact_section_match']})")
            lines.append(f"- Length ratio: {float(row['length_ratio']):.4f}")
            lines.append(f"- ROUGE-L full / impression: {float(row['rougeL_full']):.4f} / {float(row['rougeL_impression']):.4f}")
            lines.append(f"- Semantic full / impression: {float(row['semantic_full']):.4f} / {float(row['semantic_impression']):.4f}")
            lines.append(f"- Reference: {row['reference_path']}")
            lines.append(f"- Generated: {row['generated_path']}")
            lines.append('\nReference impression:')
            lines.append('```text')
            lines.append(str(row['reference_impression'])[:1200])
            lines.append('```')
            lines.append('Generated impression:')
            lines.append('```text')
            lines.append(str(row['generated_impression'])[:1200])
            lines.append('```')
    review_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')



def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.output_dir / 'generated_reports'
    generated_dir.mkdir(parents=True, exist_ok=True)

    prompt_text, prompt_path = medgemma4b.resolve_prompt_text(args.prompt_mode, args.prompt_file)
    cases = load_reference_cases(args.reference_reports_dir, args.data_root, args.limit)
    processor = medgemma4b.load_processor(args.model_id)
    medgemma4b.preflight_generation_budget(
        args.model_id,
        processor,
        cases[0]['image_path'],
        prompt_text,
        args.max_new_tokens,
        args.attn_implementation,
        skip_budget_guard=False,
    )
    model = medgemma4b.load_model(args.model_id, attn_implementation=args.attn_implementation)
    model.eval()

    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    embedder = SentenceTransformer(args.embedding_model)

    rows: list[dict[str, object]] = []
    generated_texts: list[str] = []
    reference_texts: list[str] = []
    generated_impressions: list[str] = []
    reference_impressions: list[str] = []

    for batch in medgemma4b.batched([(str(case['stem']), case['image_path']) for case in cases], args.batch_size):
        stems = [stem for stem, _ in batch]
        batch_paths = [image_path for _, image_path in batch]
        reports = medgemma4b.generate_reports_batch(batch_paths, model, processor, prompt_text, args.max_new_tokens)
        for stem, report, image_path in zip(stems, reports, batch_paths, strict=True):
            reference_path = args.reference_reports_dir / f'{stem}.txt'
            generated_path = generated_dir / f'{stem}.txt'
            generated_path.write_text(report + '\n', encoding='utf-8')
            reference_text = reference_path.read_text(encoding='utf-8').strip()
            generated_text = report.strip()
            generated_impression = extract_section(generated_text, '5) IMPRESSION', '6) LIMITATIONS / UNCERTAINTY')
            reference_impression = extract_section(reference_text, '5) IMPRESSION', '6) LIMITATIONS / UNCERTAINTY')
            section_ratio, exact_match, present_headers = section_compliance(generated_text)
            reference_tokens = len(processor.tokenizer(reference_text, add_special_tokens=False).input_ids)
            generated_tokens = len(processor.tokenizer(generated_text, add_special_tokens=False).input_ids)
            ratio = generated_tokens / reference_tokens if reference_tokens else 0.0
            rouge_full = rouge.score(reference_text, generated_text)['rougeL'].fmeasure
            rouge_impression = rouge.score(reference_impression or ' ', generated_impression or ' ')['rougeL'].fmeasure
            row = {
                'stem': stem,
                'image_path': str(image_path),
                'reference_path': str(reference_path),
                'generated_path': str(generated_path),
                'prompt_path': str(prompt_path),
                'section_ratio': section_ratio,
                'exact_section_match': exact_match,
                'present_headers': '|'.join(present_headers),
                'reference_tokens': reference_tokens,
                'generated_tokens': generated_tokens,
                'length_ratio': ratio,
                'length_score': length_score(ratio),
                'rougeL_full': rouge_full,
                'rougeL_impression': rouge_impression,
                'reference_impression': reference_impression,
                'generated_impression': generated_impression,
            }
            rows.append(row)
            reference_texts.append(reference_text)
            generated_texts.append(generated_text)
            reference_impressions.append(reference_impression or ' ')
            generated_impressions.append(generated_impression or ' ')

    full_embeddings = embedder.encode(reference_texts + generated_texts, normalize_embeddings=True, convert_to_tensor=True)
    full_ref = full_embeddings[: len(reference_texts)]
    full_gen = full_embeddings[len(reference_texts) :]
    impression_embeddings = embedder.encode(reference_impressions + generated_impressions, normalize_embeddings=True, convert_to_tensor=True)
    impression_ref = impression_embeddings[: len(reference_impressions)]
    impression_gen = impression_embeddings[len(reference_impressions) :]

    metrics_rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        semantic_full = float(util.cos_sim(full_ref[idx], full_gen[idx]).item())
        semantic_impression = float(util.cos_sim(impression_ref[idx], impression_gen[idx]).item())
        row['semantic_full'] = semantic_full
        row['semantic_impression'] = semantic_impression
        row['composite_score'] = composite_score(row)
        metrics_rows.append(row)

    metrics_csv = args.output_dir / 'metrics.csv'
    fieldnames = [
        'stem', 'image_path', 'reference_path', 'generated_path', 'prompt_path', 'section_ratio', 'exact_section_match',
        'present_headers', 'reference_tokens', 'generated_tokens', 'length_ratio', 'length_score', 'rougeL_full',
        'rougeL_impression', 'semantic_full', 'semantic_impression', 'composite_score', 'reference_impression',
        'generated_impression',
    ]
    with metrics_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    summary = {
        'count': len(metrics_rows),
        'prompt_path': str(prompt_path),
        'attn_implementation': args.attn_implementation,
        'max_new_tokens': args.max_new_tokens,
        'batch_size': args.batch_size,
        'averages': {
            'section_ratio': sum(float(row['section_ratio']) for row in metrics_rows) / len(metrics_rows),
            'length_ratio': sum(float(row['length_ratio']) for row in metrics_rows) / len(metrics_rows),
            'rougeL_full': sum(float(row['rougeL_full']) for row in metrics_rows) / len(metrics_rows),
            'rougeL_impression': sum(float(row['rougeL_impression']) for row in metrics_rows) / len(metrics_rows),
            'semantic_full': sum(float(row['semantic_full']) for row in metrics_rows) / len(metrics_rows),
            'semantic_impression': sum(float(row['semantic_impression']) for row in metrics_rows) / len(metrics_rows),
            'composite_score': sum(float(row['composite_score']) for row in metrics_rows) / len(metrics_rows),
        },
    }
    summary_path = args.output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    review_path = args.output_dir / 'review.md'
    write_review_markdown(review_path, pick_review_groups(metrics_rows))

    print(json.dumps({'metrics_csv': str(metrics_csv), 'summary_json': str(summary_path), 'review_markdown': str(review_path), 'summary': summary}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
