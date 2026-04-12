#!/usr/bin/env python3
"""Build deterministic subset manifests for fast NIH -> CheXpert/MIMIC pilot runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT_CSV = Path("/workspace/manifest_common_labels_nih_train_val_test_chexpert_mimic.csv")
DEFAULT_OUTPUT_CSV = Path("/workspace/manifest_common_labels_pilot5h.csv")
LABEL_COLUMNS = (
    "label_atelectasis",
    "label_cardiomegaly",
    "label_consolidation",
    "label_edema",
    "label_pleural_effusion",
    "label_pneumonia",
    "label_pneumothorax",
)

PROFILE_SETTINGS: dict[str, dict[str, object]] = {
    "smoke_2h": {
        "direct_counts": {
            ("d0_nih", "train"): 4_000,
            ("d0_nih", "val"): 500,
            ("d0_nih", "test"): 1_000,
            ("d1_chexpert", "train"): 0,
            ("d1_chexpert", "val"): -1,
            ("d2_mimic", "val"): 0,
            ("d2_mimic", "test"): -1,
        },
        "chexpert_target_counts": None,
    },
    "pilot_5h": {
        "direct_counts": {
            ("d0_nih", "train"): 10_000,
            ("d0_nih", "val"): 1_000,
            ("d0_nih", "test"): 2_000,
            ("d1_chexpert", "train"): 0,
            ("d1_chexpert", "val"): -1,
            ("d2_mimic", "val"): 0,
            ("d2_mimic", "test"): -1,
        },
        "chexpert_target_counts": None,
    },
    "pilot_4h": {
        "direct_counts": {
            ("d0_nih", "train"): 8_000,
            ("d0_nih", "val"): 1_000,
            ("d0_nih", "test"): 1_500,
            ("d1_chexpert", "train"): 0,
            ("d1_chexpert", "val"): -1,
            ("d2_mimic", "val"): 0,
            ("d2_mimic", "test"): -1,
        },
        "chexpert_target_counts": None,
    },
    "chexpert_target_250": {
        "direct_counts": {
            ("d0_nih", "train"): 0,
            ("d0_nih", "val"): 0,
            ("d0_nih", "test"): 0,
            ("d1_chexpert", "train"): 0,
            ("d1_chexpert", "val"): 0,
            ("d2_mimic", "val"): 0,
            ("d2_mimic", "test"): 0,
        },
        "chexpert_target_counts": {
            "train": 250,
            "val": 250,
            "test": -1,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic subset manifest for fast source-to-target transfer experiments. "
            "Rows are sampled independently within each domain/split, with an optional "
            "CheXpert target-train/val/test construction path."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_SETTINGS),
        default="pilot_5h",
        help="Predefined subset size profile.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--d0-train", type=int, default=None, help="Override NIH train count.")
    parser.add_argument("--d0-val", type=int, default=None, help="Override NIH val count.")
    parser.add_argument("--d0-test", type=int, default=None, help="Override NIH test count.")
    parser.add_argument(
        "--d1-val",
        type=int,
        default=None,
        help="Override direct CheXpert valid count for source-transfer manifests.",
    )
    parser.add_argument(
        "--d1-train",
        type=int,
        default=None,
        help="Sample CheXpert target-train rows from the CheXpert train pool.",
    )
    parser.add_argument(
        "--d1-target-val",
        type=int,
        default=None,
        help="Sample CheXpert target-val rows from the remaining CheXpert train pool.",
    )
    parser.add_argument(
        "--d1-test",
        type=int,
        default=None,
        help="Sample CheXpert target-test rows from the CheXpert valid split. Use -1 for all rows.",
    )
    parser.add_argument("--d2-val", type=int, default=None, help="Override MIMIC val count.")
    parser.add_argument("--d2-test", type=int, default=None, help="Override MIMIC test count.")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults next to the output manifest.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest has no header: {path}")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def label_signature(row: dict[str, str]) -> str:
    return "|".join(str(row.get(column) or "0").strip() for column in LABEL_COLUMNS)


def allocate_group_counts(
    group_sizes: dict[str, int],
    target_count: int,
) -> dict[str, int]:
    total_count = sum(group_sizes.values())
    if target_count >= total_count:
        return dict(group_sizes)
    if target_count <= 0:
        return {key: 0 for key in group_sizes}

    allocated = {key: 0 for key in group_sizes}
    remainders: list[tuple[float, str]] = []

    for key, size in group_sizes.items():
        exact = (target_count * size) / total_count
        base = min(size, int(math.floor(exact)))
        allocated[key] = base
        remainders.append((exact - base, key))

    remaining = target_count - sum(allocated.values())

    # Give one example to as many non-empty groups as possible when room exists.
    if remaining > 0:
        for _, key in sorted(remainders, reverse=True):
            if remaining <= 0:
                break
            if group_sizes[key] > 0 and allocated[key] == 0:
                allocated[key] = 1
                remaining -= 1

    if remaining > 0:
        for _, key in sorted(remainders, reverse=True):
            if remaining <= 0:
                break
            room = group_sizes[key] - allocated[key]
            if room <= 0:
                continue
            take = min(room, remaining)
            allocated[key] += take
            remaining -= take

    return allocated


def sample_grouped_rows(
    rows: list[dict[str, str]],
    target_count: int,
    *,
    seed: int,
) -> list[dict[str, str]]:
    if target_count < 0 or target_count >= len(rows):
        return list(rows)
    if target_count == 0:
        return []

    rng = random.Random(seed)
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[label_signature(row)].append((index, row))

    group_sizes = {key: len(items) for key, items in grouped.items()}
    allocations = allocate_group_counts(group_sizes, target_count)

    selected: list[tuple[int, dict[str, str]]] = []
    for key, items in grouped.items():
        take = allocations[key]
        if take <= 0:
            continue
        local_items = list(items)
        rng.shuffle(local_items)
        selected.extend(local_items[:take])

    selected.sort(key=lambda item: item[0])
    return [row for _, row in selected]


def clone_row_with_split(row: dict[str, str], *, split: str) -> dict[str, str]:
    cloned = dict(row)
    cloned["split"] = split
    return cloned


def positive_count(rows: list[dict[str, str]], column: str) -> int:
    return sum(1 for row in rows if str(row.get(column) or "0").strip() == "1")


def build_split_report(
    *,
    source_rows: list[dict[str, str]],
    sampled_rows: list[dict[str, str]],
    requested: int,
    source_key: tuple[str, str],
    output_key: tuple[str, str],
) -> dict[str, object]:
    return {
        "source_key": f"{source_key[0]}/{source_key[1]}",
        "output_key": f"{output_key[0]}/{output_key[1]}",
        "available_rows": len(source_rows),
        "selected_rows": len(sampled_rows),
        "requested_rows": requested,
        "selection_mode": "all" if requested < 0 or requested >= len(source_rows) else "sampled",
        "label_positive_counts": {
            column: {
                "available": positive_count(source_rows, column),
                "selected": positive_count(sampled_rows, column),
            }
            for column in LABEL_COLUMNS
        },
        "label_signature_counts_selected": Counter(label_signature(row) for row in sampled_rows),
    }


def resolve_sampling_plan(
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, str], int], dict[str, int] | None]:
    profile_settings = PROFILE_SETTINGS[args.profile]
    direct_counts = dict(profile_settings["direct_counts"])
    target_counts = (
        None
        if profile_settings["chexpert_target_counts"] is None
        else dict(profile_settings["chexpert_target_counts"])
    )

    direct_overrides = {
        ("d0_nih", "train"): args.d0_train,
        ("d0_nih", "val"): args.d0_val,
        ("d0_nih", "test"): args.d0_test,
        ("d1_chexpert", "val"): args.d1_val,
        ("d2_mimic", "val"): args.d2_val,
        ("d2_mimic", "test"): args.d2_test,
    }
    for key, value in direct_overrides.items():
        if value is not None:
            direct_counts[key] = value

    target_requested = target_counts is not None or any(
        value is not None for value in (args.d1_train, args.d1_target_val, args.d1_test)
    )
    if target_requested:
        counts = dict(target_counts or {"train": 0, "val": 0, "test": -1})
        if args.d1_train is not None:
            counts["train"] = args.d1_train
        if args.d1_target_val is not None:
            counts["val"] = args.d1_target_val
        if args.d1_test is not None:
            counts["test"] = args.d1_test
        target_counts = counts
        direct_counts[("d1_chexpert", "train")] = 0
        if args.d1_val is None:
            direct_counts[("d1_chexpert", "val")] = 0

    return direct_counts, target_counts


def build_disjoint_chexpert_target_splits(
    *,
    train_pool_rows: list[dict[str, str]],
    test_pool_rows: list[dict[str, str]],
    target_counts: dict[str, int],
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    sampled_train = sample_grouped_rows(train_pool_rows, int(target_counts["train"]), seed=seed)
    selected_train_ids = {row["row_id"] for row in sampled_train}
    remaining_train_pool = [row for row in train_pool_rows if row["row_id"] not in selected_train_ids]
    sampled_val_source = sample_grouped_rows(remaining_train_pool, int(target_counts["val"]), seed=seed + 1)
    sampled_test_source = sample_grouped_rows(test_pool_rows, int(target_counts["test"]), seed=seed + 2)
    return {
        "train": [clone_row_with_split(row, split="train") for row in sampled_train],
        "val": [clone_row_with_split(row, split="val") for row in sampled_val_source],
        "test": [clone_row_with_split(row, split="test") for row in sampled_test_source],
        "remaining_train_pool": remaining_train_pool,
    }


def main() -> int:
    args = parse_args()
    rows, fieldnames = load_rows(args.input_csv)
    direct_counts, chexpert_target_counts = resolve_sampling_plan(args)
    report_json = args.report_json or args.output_csv.with_suffix(".summary.json")

    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = ((row.get("domain") or "").strip(), (row.get("split") or "").strip())
        grouped_rows[key].append(row)

    sampled_rows: list[dict[str, str]] = []
    report: dict[str, object] = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "profile": args.profile,
        "seed": args.seed,
        "requested_counts": {
            f"{domain}/{split}": count for (domain, split), count in sorted(direct_counts.items())
        },
        "chexpert_target_counts": chexpert_target_counts,
        "splits": {},
    }

    for key in sorted(grouped_rows):
        domain, split = key
        source_rows = grouped_rows[key]
        requested = direct_counts.get(key, -1)
        sampled = sample_grouped_rows(source_rows, requested, seed=args.seed)
        sampled_rows.extend(sampled)
        report["splits"][f"{domain}/{split}"] = build_split_report(
            source_rows=source_rows,
            sampled_rows=sampled,
            requested=requested,
            source_key=key,
            output_key=key,
        )

    if chexpert_target_counts is not None:
        chexpert_train_key = ("d1_chexpert", "train")
        chexpert_valid_key = ("d1_chexpert", "val")
        chexpert_train_pool = grouped_rows.get(chexpert_train_key, [])
        chexpert_test_pool = grouped_rows.get(chexpert_valid_key, [])
        chexpert_samples = build_disjoint_chexpert_target_splits(
            train_pool_rows=chexpert_train_pool,
            test_pool_rows=chexpert_test_pool,
            target_counts=chexpert_target_counts,
            seed=args.seed,
        )
        sampled_rows.extend(chexpert_samples["train"])
        sampled_rows.extend(chexpert_samples["val"])
        sampled_rows.extend(chexpert_samples["test"])

        report["splits"]["d1_chexpert/train"] = build_split_report(
            source_rows=chexpert_train_pool,
            sampled_rows=chexpert_samples["train"],
            requested=int(chexpert_target_counts["train"]),
            source_key=chexpert_train_key,
            output_key=("d1_chexpert", "train"),
        )
        report["splits"]["d1_chexpert/val"] = build_split_report(
            source_rows=chexpert_samples["remaining_train_pool"],
            sampled_rows=chexpert_samples["val"],
            requested=int(chexpert_target_counts["val"]),
            source_key=chexpert_train_key,
            output_key=("d1_chexpert", "val"),
        )
        report["splits"]["d1_chexpert/test"] = build_split_report(
            source_rows=chexpert_test_pool,
            sampled_rows=chexpert_samples["test"],
            requested=int(chexpert_target_counts["test"]),
            source_key=chexpert_valid_key,
            output_key=("d1_chexpert", "test"),
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled_rows)

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[done] output_csv={args.output_csv} rows={len(sampled_rows)} profile={args.profile}")
    for key in sorted(report["splits"]):
        split_report = report["splits"][key]
        print(
            f"[split] {key} selected={split_report['selected_rows']}/{split_report['available_rows']} "
            f"requested={split_report['requested_rows']} source={split_report['source_key']}"
        )
    print(f"[done] summary_json={report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
