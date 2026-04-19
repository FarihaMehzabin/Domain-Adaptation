#!/usr/bin/env python3
"""Download a TSV plan of PhysioNet files with bounded parallelism via wget."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PHYSIONET_HOST = "https://physionet.org"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a PhysioNet TSV plan with parallel HTTP basic-auth requests.")
    parser.add_argument("--plan-tsv", type=Path, required=True)
    parser.add_argument("--username", type=str, default="")
    parser.add_argument("--password", type=str, default="")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=-1, help="Optional cap on number of files to download.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_plan(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def make_worker(username: str, password: str):
    local = threading.local()

    def download_one(row: dict[str, str], force: bool) -> tuple[str, str]:
        url = row["url"]
        destination = Path(row["local_path"])
        if destination.exists() and not force:
            return "skipped_existing", str(destination)

        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_suffix(destination.suffix + ".part")
        try:
            command = [
                "wget",
                "-q",
                "-O",
                str(tmp_path),
                "--user",
                username,
                "--password",
                password,
                url,
            ]
            if not force:
                command.insert(1, "-c")
            result = subprocess.run(command, check=False, capture_output=True, text=True)
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            return f"error:{exc}", str(destination)
        if result.returncode != 0:
            if tmp_path.exists():
                tmp_path.unlink()
            stderr = (result.stderr or "").strip().splitlines()
            message = stderr[-1] if stderr else f"wget_exit_{result.returncode}"
            return f"error:{message}", str(destination)

        tmp_path.replace(destination)
        return "downloaded", str(destination)

    return download_one


def main() -> int:
    args = parse_args()
    username = args.username.strip() or os.environ.get("PHYSIONET_USERNAME", "").strip()
    password = args.password or os.environ.get("PHYSIONET_PASSWORD", "")
    if not username or not password:
        raise SystemExit("PhysioNet credentials are required via --username/--password or env vars.")

    rows = read_plan(args.plan_tsv)
    if args.limit >= 0:
        rows = rows[: args.limit]

    download_one = make_worker(username, password)
    counts: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(download_one, row, args.force) for row in rows]
        for index, future in enumerate(as_completed(futures), start=1):
            status, path = future.result()
            counts[status] = counts.get(status, 0) + 1
            if index % 25 == 0 or index == len(futures):
                print(f"[progress] completed={index}/{len(futures)} counts={counts}")

    print(f"[done] counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
