#!/usr/bin/env python3
"""Backfill all-domain evaluation files for an existing domain-transfer head experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch


DEFAULT_TRAINER_SCRIPT = Path("/workspace/scripts/15_train_domain_transfer_linear_probe.py")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an existing experiment directory and best checkpoint, then rewrite evaluation outputs "
            "for the current all-domain evaluation plan."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--trainer-script", type=Path, default=DEFAULT_TRAINER_SCRIPT)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--embedding-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trainer = load_trainer_module(args.trainer_script.resolve())

    experiment_dir = args.experiment_dir.resolve()
    config_path = experiment_dir / "config.json"
    checkpoint_path = experiment_dir / "best.ckpt"
    meta_path = experiment_dir / "experiment_meta.json"
    if not config_path.exists() or not checkpoint_path.exists():
        raise SystemExit(f"Experiment dir is missing config.json or best.ckpt: {experiment_dir}")

    config = load_json(config_path)
    experiment_meta = load_json(meta_path) if meta_path.exists() else {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Checkpoint is not a dict: {checkpoint_path}")

    manifest_csv = (args.manifest_csv.resolve() if args.manifest_csv is not None else Path(config["manifest_csv"]).resolve())
    embedding_root = (
        args.embedding_root.resolve() if args.embedding_root is not None else Path(config["embedding_root"]).resolve()
    )
    batch_size = int(args.batch_size) if args.batch_size is not None else int(config["batch_size"])
    num_workers = int(args.num_workers) if args.num_workers is not None else int(config["num_workers"])
    device = trainer.resolve_device(str(args.device or config["device_resolved"]))

    split_profile = str(config["split_profile"])
    embedding_layout = str(config["embedding_layout"])
    token_pooling = str(config["token_pooling"])
    l2_normalize_features = bool(config["l2_normalize_features"])
    head_type = str(config["head_type"])
    mlp_hidden_dims = tuple(int(dim) for dim in config.get("mlp_hidden_dims") or [])
    mlp_dropout = float(config.get("mlp_dropout", 0.0))

    label_names = checkpoint.get("label_names")
    if not isinstance(label_names, list) or not label_names:
        label_names = list(config["label_names"])
    tuned_thresholds = checkpoint.get("tuned_thresholds")
    if tuned_thresholds is None:
        threshold_path = experiment_dir / str(trainer.build_evaluation_plan(
            split_profile=split_profile,
            embedding_layout=embedding_layout,
        ).thresholds_filename)
        threshold_payload = load_json(threshold_path)
        label_thresholds = threshold_payload.get("labels", {})
        tuned_thresholds = [float(label_thresholds[label]["threshold"]) for label in label_names]
    tuned_thresholds_array = np.asarray(tuned_thresholds, dtype=np.float32)

    evaluation_plan = trainer.build_evaluation_plan(
        split_profile=split_profile,
        embedding_layout=embedding_layout,
    )
    label_names_from_manifest, manifest_by_key = trainer.load_manifest_records(manifest_csv)
    if list(label_names_from_manifest) != list(label_names):
        raise SystemExit(
            "Manifest label names do not match the checkpoint/config.\n"
            f"manifest={label_names_from_manifest}\ncheckpoint={label_names}"
        )

    split_data: dict[str, Any] = {}
    for alias, domain, split in evaluation_plan.split_specs:
        manifest_records = manifest_by_key.get((domain, split))
        if manifest_records is None:
            raise SystemExit(f"Manifest does not contain records for domain={domain} split={split}.")
        split_data[alias] = trainer.load_split_data(
            alias=alias,
            embedding_root=embedding_root,
            embedding_layout=embedding_layout,
            domain=domain,
            split=split,
            manifest_records=manifest_records,
            num_labels=len(label_names),
            max_rows=None,
        )

    input_dim = int(checkpoint.get("input_dim") or config["input_dim"])
    model = trainer.build_probe_model(
        head_type=head_type,
        input_dim=input_dim,
        num_labels=len(label_names),
        mlp_hidden_dims=mlp_hidden_dims,
        mlp_dropout=mlp_dropout,
    ).to(device)
    trainer.load_initial_checkpoint(
        init_checkpoint_path=checkpoint_path,
        model=model,
        label_names=label_names,
        requested_head_type=head_type,
        input_dim=input_dim,
        num_labels=len(label_names),
        token_pooling=token_pooling,
        l2_normalize_features=l2_normalize_features,
    )

    pos_weight = trainer.compute_pos_weight(split_data[evaluation_plan.train_alias].labels).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_loaders = {
        alias: trainer.build_dataloader(
            payload,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            shuffle=False,
            token_pooling=token_pooling,
            l2_normalize_features=l2_normalize_features,
        )
        for alias, payload in split_data.items()
        if alias != evaluation_plan.train_alias and alias not in evaluation_plan.auxiliary_train_aliases
    }

    metrics_by_alias: dict[str, dict[str, Any]] = {}
    for alias, loader in eval_loaders.items():
        loss, logits, targets = trainer.evaluate_model(
            loader=loader,
            model=model,
            criterion=criterion,
            device=device,
            fp16_on_cuda=False,
        )
        metrics = trainer.summarize_split_metrics(
            split_alias=alias,
            loss=loss,
            targets=targets,
            logits=logits,
            label_names=label_names,
            tuned_thresholds=tuned_thresholds_array,
        )
        metrics_by_alias[alias] = metrics
        output_name = evaluation_plan.output_name_map.get(alias, f"{alias}_metrics.json")
        output_path = experiment_dir / output_name
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing metrics file without --overwrite: {output_path}")
        write_json(output_path, metrics)

    config["manifest_csv"] = str(manifest_csv)
    config["embedding_root"] = str(embedding_root)
    config["device_resolved"] = str(device)
    config.setdefault("preservation_method", "none")
    config.setdefault("enable_mas", False)
    config.setdefault("mas_state_path", None)
    config.setdefault("mas_lambda", None)
    config.setdefault("enable_lwf", False)
    config.setdefault("lwf_teacher_checkpoint_path", None)
    config.setdefault("lwf_teacher_checkpoint_metadata", None)
    config.setdefault("lwf_source_alias", None)
    config.setdefault("lwf_alpha", None)
    config.setdefault("lwf_temperature", None)
    config["split_inputs"] = {
        alias: {
            "domain": payload.domain,
            "split": payload.split,
            "embeddings_path": str(payload.embeddings_path),
            "num_rows": len(payload.row_ids),
            "shape": list(payload.embeddings_shape),
            "sidecar_path": str(payload.split_dir / payload.sidecar.relative_path),
            "sidecar_parser": payload.sidecar.parser,
        }
        for alias, payload in split_data.items()
    }
    config["backfilled_evaluation_utc"] = utc_now_iso()
    write_json(config_path, config)

    experiment_meta["manifest_csv"] = str(manifest_csv)
    experiment_meta["embedding_root"] = str(embedding_root)
    experiment_meta["device_resolved"] = str(device)
    experiment_meta.setdefault("preservation_method", config["preservation_method"])
    experiment_meta.setdefault("enable_mas", config["enable_mas"])
    experiment_meta.setdefault("mas_state_path", config["mas_state_path"])
    experiment_meta.setdefault("mas_lambda", config["mas_lambda"])
    experiment_meta.setdefault("enable_lwf", config["enable_lwf"])
    experiment_meta.setdefault("lwf_teacher_checkpoint_path", config["lwf_teacher_checkpoint_path"])
    experiment_meta.setdefault("lwf_teacher_checkpoint_metadata", config["lwf_teacher_checkpoint_metadata"])
    experiment_meta.setdefault("lwf_source_alias", config["lwf_source_alias"])
    experiment_meta.setdefault("lwf_alpha", config["lwf_alpha"])
    experiment_meta.setdefault("lwf_temperature", config["lwf_temperature"])
    experiment_meta["train_alias"] = evaluation_plan.train_alias
    experiment_meta["auxiliary_train_aliases"] = list(evaluation_plan.auxiliary_train_aliases)
    experiment_meta["selection_alias"] = evaluation_plan.selection_alias
    experiment_meta["primary_test_alias"] = evaluation_plan.primary_test_alias
    experiment_meta["split_inputs"] = config["split_inputs"]
    experiment_meta["macro_metrics"] = {alias: summary["macro"] for alias, summary in metrics_by_alias.items()}
    experiment_meta["backfilled_evaluation_utc"] = config["backfilled_evaluation_utc"]
    write_json(meta_path, experiment_meta)

    recreation_report = trainer.render_recreation_report(
        experiment_dir=experiment_dir,
        config=config,
        split_data=split_data,
        metrics_by_alias=metrics_by_alias,
    )
    (experiment_dir / "recreation_report.md").write_text(recreation_report, encoding="utf-8")
    print(f"[done] experiment_dir={experiment_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
