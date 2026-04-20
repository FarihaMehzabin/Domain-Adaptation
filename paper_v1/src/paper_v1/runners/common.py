"""Shared runner helpers."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import torch

from paper_v1.data.embeddings import EmbeddingAssetSpec, align_manifest_records
from paper_v1.data.manifests import ValidationIssue, load_manifest
from paper_v1.data.splits import SplitSelector, select_records
from paper_v1.utils.io import read_json, resolve_path


def parse_config_arg(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    payload = read_json(config_path)
    payload["_config_path"] = str(config_path)
    payload["_config_dir"] = str(config_path.parent)
    return payload


def load_data_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    config["manifest_path"] = str(resolve_path(config["_config_dir"], config["manifest_path"]))
    return config


def build_asset_specs(data_config: dict[str, Any]) -> dict[str, EmbeddingAssetSpec]:
    config_dir = Path(data_config["_config_dir"])
    specs = {}
    for domain, payload in data_config["domain_assets"].items():
        spec_payload = dict(payload)
        spec_payload["index_csv"] = str(resolve_path(config_dir, spec_payload["index_csv"]))
        if spec_payload.get("root_dir"):
            spec_payload["root_dir"] = str(resolve_path(config_dir, spec_payload["root_dir"]))
        specs[domain] = EmbeddingAssetSpec.from_dict(payload.get("name", domain), spec_payload)
    return specs


def selectors_from_config(rows: list[list[str]]) -> list[SplitSelector]:
    return [SplitSelector(domain=row[0], split=row[1]) for row in rows]


def collect_alignment(
    *,
    data_config: dict[str, Any],
    domains: set[str] | None = None,
    splits: set[str] | None = None,
    selectors: list[SplitSelector] | None = None,
    missing_embedding_policy: str = "error",
    require_patient_disjoint: bool = False,
) -> tuple[Any, Any]:
    manifest = load_manifest(data_config["manifest_path"], negative_one_policy=str(data_config.get("negative_one_policy", "zero")))
    manifest_records = manifest.filter(domains=domains, splits=splits)
    if selectors is not None:
        wanted = {(selector.domain, selector.split) for selector in selectors}
        manifest_records = [record for record in manifest_records if (record.domain, record.split) in wanted]
    asset_specs = build_asset_specs(data_config)
    alignment = align_manifest_records(
        manifest_records,
        asset_specs,
        missing_embedding_policy=missing_embedding_policy,
        expected_dim=data_config.get("expected_embedding_dim"),
    )
    validation_issues = manifest.validate(domains=domains, require_patient_disjoint=require_patient_disjoint)
    alignment.issues = [*validation_issues, *alignment.issues]
    return manifest, alignment


def raise_for_issues(issues: list[ValidationIssue], *, allowed_codes: set[str] | None = None) -> None:
    allowed_codes = allowed_codes or set()
    blocking = [issue for issue in issues if issue.code not in allowed_codes and issue.severity == "error"]
    if blocking:
        messages = "\n".join(f"- {issue.code}: {issue.message}" for issue in blocking[:10])
        raise SystemExit(f"blocking validation issues:\n{messages}")


def build_named_datasets(aligned_records, selector_map: dict[str, list[SplitSelector]]):
    from paper_v1.data.embeddings import FrozenEmbeddingDataset

    datasets = {}
    for alias, selectors in selector_map.items():
        datasets[alias] = FrozenEmbeddingDataset(select_records(aligned_records, selectors))
    return datasets


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_latest_checkpoint(pattern: str) -> Path | None:
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return Path(matches[-1])
