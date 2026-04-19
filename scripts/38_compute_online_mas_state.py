#!/usr/bin/env python3
"""Compute and save an online MAS state for a frozen-embedding linear probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


DEFAULT_TRAINER_SCRIPT = Path("/workspace/scripts/15_train_domain_transfer_linear_probe.py")
DEFAULT_MANIFEST_CSV = Path("/workspace/manifest/manifest_common_labels_nih_train_val_test_chexpert_mimic.csv")
DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_WORKERS = 0
DEFAULT_DEVICE = "auto"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_trainer_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("linear_probe_trainer", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load trainer module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("linear_probe_trainer", module)
    spec.loader.exec_module(module)
    return module


def mean_tensor_value(tensors: dict[str, torch.Tensor]) -> float:
    total_sum = 0.0
    total_count = 0
    for tensor in tensors.values():
        total_sum += float(tensor.sum().item())
        total_count += int(tensor.numel())
    if total_count == 0:
        raise SystemExit("No tensor entries were available to normalize MAS omega.")
    return total_sum / float(total_count)


def validate_previous_state(
    *,
    payload: dict[str, Any],
    label_names: list[str],
    head_type: str,
    input_dim: int,
    num_labels: int,
    token_pooling: str,
    l2_normalize_features: bool,
    parameter_names: set[str],
    previous_state_path: Path,
) -> dict[str, torch.Tensor]:
    if payload.get("label_names") is not None and list(payload["label_names"]) != label_names:
        raise SystemExit(
            "Previous MAS state label_names do not match the requested run.\n"
            f"state={list(payload['label_names'])}\nrequested={label_names}"
        )
    if payload.get("head_type") is not None and str(payload["head_type"]) != head_type:
        raise SystemExit(
            f"Previous MAS state head_type={payload['head_type']} does not match checkpoint head_type={head_type}."
        )
    if payload.get("input_dim") is not None and int(payload["input_dim"]) != int(input_dim):
        raise SystemExit(
            f"Previous MAS state input_dim={payload['input_dim']} does not match checkpoint input_dim={input_dim}."
        )
    if payload.get("num_labels") is not None and int(payload["num_labels"]) != int(num_labels):
        raise SystemExit(
            f"Previous MAS state num_labels={payload['num_labels']} does not match checkpoint num_labels={num_labels}."
        )
    if payload.get("token_pooling") is not None and str(payload["token_pooling"]) != token_pooling:
        raise SystemExit(
            f"Previous MAS state token_pooling={payload['token_pooling']} does not match checkpoint token_pooling={token_pooling}."
        )
    if payload.get("l2_normalize_features") is not None and bool(payload["l2_normalize_features"]) != bool(
        l2_normalize_features
    ):
        raise SystemExit(
            "Previous MAS state l2_normalize_features does not match the checkpoint.\n"
            f"state={bool(payload['l2_normalize_features'])} checkpoint={bool(l2_normalize_features)}"
        )
    omega_state = payload.get("omega_state_dict")
    if not isinstance(omega_state, dict):
        raise SystemExit(f"Previous MAS state is missing omega_state_dict: {previous_state_path}")
    if set(omega_state.keys()) != parameter_names:
        raise SystemExit(
            "Previous MAS omega keys do not match the current model.\n"
            f"expected={sorted(parameter_names)}\nfound={sorted(omega_state.keys())}"
        )
    return {name: tensor.detach().cpu().float() for name, tensor in omega_state.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a normalized online MAS omega state from one domain/split and optionally "
            "accumulate it onto a previous saved MAS state."
        )
    )
    parser.add_argument("--trainer-script", type=Path, default=DEFAULT_TRAINER_SCRIPT)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--alias", type=str, default="mas_source")
    parser.add_argument("--embedding-layout", choices=("domain_split", "source_only"), default="domain_split")
    parser.add_argument("--previous-state-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--fp16-on-cuda", action="store_true")
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0.")
    if args.max_rows_per_split is not None and args.max_rows_per_split <= 0:
        raise SystemExit("--max-rows-per-split must be positive when provided.")

    trainer = load_trainer_module(args.trainer_script.resolve())
    device = trainer.resolve_device(args.device)

    checkpoint_path = args.checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Checkpoint is not a dict: {checkpoint_path}")
    label_names = checkpoint.get("label_names")
    if not isinstance(label_names, list) or not label_names:
        raise SystemExit(f"Checkpoint is missing label_names: {checkpoint_path}")
    head_type = str(checkpoint.get("head_type") or "linear")
    input_dim = checkpoint.get("input_dim")
    num_labels = checkpoint.get("num_labels")
    token_pooling = str(checkpoint.get("token_pooling") or "avg")
    l2_normalize_features = bool(checkpoint.get("l2_normalize_features") or False)
    if input_dim is None or num_labels is None:
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise SystemExit(f"Checkpoint is missing state_dict: {checkpoint_path}")
        input_dim, num_labels = trainer.infer_checkpoint_dimensions(
            checkpoint_head_type=head_type,
            state_dict=state_dict,
        )

    manifest_csv = args.manifest_csv.resolve()
    label_names_from_manifest, manifest_by_key = trainer.load_manifest_records(manifest_csv)
    if label_names_from_manifest != label_names:
        raise SystemExit(
            "Manifest label names do not match the checkpoint.\n"
            f"manifest={label_names_from_manifest}\ncheckpoint={label_names}"
        )

    manifest_records = manifest_by_key.get((args.domain, str(args.split).lower()))
    if manifest_records is None:
        raise SystemExit(f"Manifest does not contain records for domain={args.domain} split={args.split}.")
    split_data = trainer.load_split_data(
        alias=str(args.alias),
        embedding_root=args.embedding_root.resolve(),
        embedding_layout=args.embedding_layout,
        domain=str(args.domain),
        split=str(args.split).lower(),
        manifest_records=manifest_records,
        num_labels=len(label_names),
        max_rows=args.max_rows_per_split,
    )
    loader = trainer.build_dataloader(
        split_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
        token_pooling=token_pooling,
        l2_normalize_features=l2_normalize_features,
    )

    model = trainer.build_probe_model(
        head_type=head_type,
        input_dim=int(input_dim),
        num_labels=int(num_labels),
        mlp_hidden_dims=tuple(int(dim) for dim in checkpoint.get("mlp_hidden_dims") or []),
        mlp_dropout=float(checkpoint.get("mlp_dropout") or 0.0),
    ).to(device)
    trainer.load_initial_checkpoint(
        init_checkpoint_path=checkpoint_path,
        model=model,
        label_names=label_names,
        requested_head_type=head_type,
        input_dim=int(input_dim),
        num_labels=int(num_labels),
        token_pooling=token_pooling,
        l2_normalize_features=l2_normalize_features,
    )
    model.eval()

    parameter_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    omega_accumulator = {
        name: torch.zeros_like(parameter.detach().cpu().float())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    total_examples = 0
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        batch_size = int(features.shape[0])
        model.zero_grad(set_to_none=True)
        with trainer.get_autocast_context(device, bool(args.fp16_on_cuda)):
            logits = model(features)
            objective = torch.square(torch.sigmoid(logits.float())).sum(dim=1).mean()
        objective.backward()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                raise SystemExit(f"Parameter '{name}' did not receive a gradient during MAS computation.")
            omega_accumulator[name] += parameter.grad.detach().cpu().abs().float() * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise SystemExit("MAS source loader produced zero examples.")

    new_omega = {name: tensor / float(total_examples) for name, tensor in omega_accumulator.items()}
    normalization_mean = mean_tensor_value(new_omega)
    if not torch.isfinite(torch.tensor(normalization_mean)) or normalization_mean <= 0.0:
        raise SystemExit(f"Computed MAS omega normalization mean is invalid: {normalization_mean}")
    normalized_new_omega = {name: tensor / float(normalization_mean) for name, tensor in new_omega.items()}

    previous_state_path = args.previous_state_path.resolve() if args.previous_state_path is not None else None
    previous_omega: dict[str, torch.Tensor] | None = None
    previous_state_provenance: dict[str, Any] | None = None
    if previous_state_path is not None:
        previous_payload = torch.load(previous_state_path, map_location="cpu")
        if not isinstance(previous_payload, dict):
            raise SystemExit(f"Previous MAS state is not a dict: {previous_state_path}")
        previous_omega = validate_previous_state(
            payload=previous_payload,
            label_names=label_names,
            head_type=head_type,
            input_dim=int(input_dim),
            num_labels=int(num_labels),
            token_pooling=token_pooling,
            l2_normalize_features=l2_normalize_features,
            parameter_names=parameter_names,
            previous_state_path=previous_state_path,
        )
        previous_state_provenance = {
            "path": str(previous_state_path),
            "omega_domain": previous_payload.get("omega_domain"),
            "omega_split": previous_payload.get("omega_split"),
            "checkpoint_path": previous_payload.get("checkpoint_path"),
        }

    omega_total = {
        name: (
            normalized_new_omega[name] + previous_omega[name]
            if previous_omega is not None
            else normalized_new_omega[name]
        )
        for name in normalized_new_omega
    }
    anchor_state_dict = {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    output_path = args.output_path.resolve()
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output path already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_date_utc": utc_now_iso(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "head_type": head_type,
        "input_dim": int(input_dim),
        "num_labels": int(num_labels),
        "label_names": label_names,
        "token_pooling": token_pooling,
        "l2_normalize_features": bool(l2_normalize_features),
        "embedding_root": str(args.embedding_root.resolve()),
        "manifest_csv": str(manifest_csv),
        "embedding_layout": str(args.embedding_layout),
        "omega_alias": str(args.alias),
        "omega_domain": str(args.domain),
        "omega_split": str(args.split).lower(),
        "num_examples": int(total_examples),
        "new_omega_mean_before_normalization": float(normalization_mean),
        "previous_state_path": (str(previous_state_path) if previous_state_path is not None else None),
        "previous_state_provenance": previous_state_provenance,
        "anchor_state_dict": anchor_state_dict,
        "new_omega_state_dict": normalized_new_omega,
        "omega_state_dict": omega_total,
    }
    torch.save(payload, output_path)
    write_json(
        output_path.with_suffix(".json"),
        {
            "run_date_utc": payload["run_date_utc"],
            "checkpoint_path": payload["checkpoint_path"],
            "checkpoint_epoch": payload["checkpoint_epoch"],
            "head_type": payload["head_type"],
            "input_dim": payload["input_dim"],
            "num_labels": payload["num_labels"],
            "label_names": payload["label_names"],
            "token_pooling": payload["token_pooling"],
            "l2_normalize_features": payload["l2_normalize_features"],
            "embedding_root": payload["embedding_root"],
            "manifest_csv": payload["manifest_csv"],
            "embedding_layout": payload["embedding_layout"],
            "omega_alias": payload["omega_alias"],
            "omega_domain": payload["omega_domain"],
            "omega_split": payload["omega_split"],
            "num_examples": payload["num_examples"],
            "new_omega_mean_before_normalization": payload["new_omega_mean_before_normalization"],
            "previous_state_path": payload["previous_state_path"],
            "previous_state_provenance": payload["previous_state_provenance"],
            "parameter_names": sorted(parameter_names),
        },
    )
    print(f"[done] output_path={output_path} num_examples={total_examples}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
