#!/usr/bin/env python3
"""Create, move, locate, and index numbered experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import experiment_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage numbered experiment directories.")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=experiment_layout.DEFAULT_EXPERIMENTS_ROOT,
        help="Root experiment directory. Defaults to /workspace/experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a new numbered experiment scaffold.")
    create_parser.add_argument("--slug", required=True, help="Short human-readable slug, without the expNNNN prefix.")
    create_parser.add_argument("--experiment-name", default=None, help="Optional full experiment directory name.")
    create_parser.add_argument("--status", choices=experiment_layout.STATUS_DIRS, default="active")
    create_parser.add_argument("--stage", action="append", default=[], help="Optional stage name. Can be repeated.")
    create_parser.add_argument("--parent", action="append", default=[], help="Optional parent experiment ID(s).")
    create_parser.add_argument("--kind", default="")
    create_parser.add_argument("--family", default="")
    create_parser.add_argument("--backbone", default="")
    create_parser.add_argument("--source-domain", default="")
    create_parser.add_argument("--target-domain", default="")
    create_parser.add_argument("--tag", action="append", default=[], help="Optional tag. Can be repeated.")
    create_parser.add_argument("--summary-title", default="")
    create_parser.add_argument("--notes", default="")
    create_parser.add_argument("--overwrite", action="store_true")

    move_parser = subparsers.add_parser("move", help="Move an existing experiment to a new status bucket.")
    move_parser.add_argument("reference", help="Experiment ID, name, or path.")
    move_parser.add_argument("--status", required=True, choices=experiment_layout.STATUS_DIRS)

    locate_parser = subparsers.add_parser("locate", help="Print the resolved path for an experiment reference.")
    locate_parser.add_argument("reference", help="Experiment ID, name, or path.")

    subparsers.add_parser("index", help="Regenerate experiments/INDEX.csv.")
    subparsers.add_parser("next-id", help="Print the next available experiment ID.")
    return parser.parse_args()


def command_create(args: argparse.Namespace) -> int:
    generated_slug = experiment_layout.slugify(args.slug, fallback="experiment")
    experiment_number, experiment_id, experiment_name, experiment_dir = experiment_layout.resolve_experiment_identity(
        experiments_root=args.experiments_root,
        requested_name=args.experiment_name,
        generated_slug=generated_slug,
        overwrite=args.overwrite,
        status=args.status,
    )
    experiment_layout.ensure_standard_layout(
        experiment_dir,
        stage_names=list(args.stage),
        metadata={
            "experiment_id": experiment_id,
            "experiment_number": experiment_number,
            "experiment_name": experiment_name,
            "status": args.status,
            "parents": list(args.parent),
            "kind": args.kind,
            "family": args.family,
            "backbone": args.backbone,
            "source_domain": args.source_domain,
            "target_domain": args.target_domain,
            "tags": list(args.tag),
            "notes": args.notes,
            "summary_title": args.summary_title or experiment_name,
        },
    )
    index_path = experiment_layout.write_index(args.experiments_root)
    print(experiment_dir)
    print(index_path)
    return 0


def command_move(args: argparse.Namespace) -> int:
    target_path = experiment_layout.move_experiment(
        args.reference,
        experiments_root=args.experiments_root,
        status=args.status,
    )
    index_path = experiment_layout.write_index(args.experiments_root)
    print(target_path)
    print(index_path)
    return 0


def command_locate(args: argparse.Namespace) -> int:
    print(experiment_layout.find_experiment_dir(args.reference, experiments_root=args.experiments_root))
    return 0


def command_index(args: argparse.Namespace) -> int:
    print(experiment_layout.write_index(args.experiments_root))
    return 0


def command_next_id(args: argparse.Namespace) -> int:
    number = experiment_layout.next_experiment_number(args.experiments_root)
    print(f"exp{number:04d}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "create":
        return command_create(args)
    if args.command == "move":
        return command_move(args)
    if args.command == "locate":
        return command_locate(args)
    if args.command == "index":
        return command_index(args)
    if args.command == "next-id":
        return command_next_id(args)
    raise SystemExit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
