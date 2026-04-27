#!/usr/bin/env python3
"""Evaluate a source-only DenseNet-121 or hospital adapter checkpoint on one manifest."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hospital_adapters_v1.common import (  # noqa: E402
    POLICY_B_LABEL_POLICY,
    StageFailure,
    build_dataloader,
    build_split_report,
    ensure_dir,
    evaluate_model,
    infer_label_names,
    infer_split_name,
    load_base_checkpoint,
    report_ready_metrics,
    resolve_project_or_absolute_path,
    resolve_run_dir,
    save_json,
    save_predictions_csv,
    set_seed,
    validate_manifest,
)
from experiments.hospital_adapters_v1.models.hospital_adapter import (  # noqa: E402
    HospitalAdapterClassifier,
    ResidualFeatureAdapter,
    apply_adapter_checkpoint,
    load_adapter_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate source-only DenseNet-121 or one or more hospital adapters on a manifest."
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/nih_2k_densenet121_best.pt")
    parser.add_argument("--adapter-checkpoint", type=str, default=None)
    parser.add_argument("--compare-adapter-checkpoint", action="append", default=[])
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--label-policy", type=str, default=POLICY_B_LABEL_POLICY)
    parser.add_argument("--out-dir", type=str, default="experiments/hospital_adapters_v1/runs")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.checkpoint = resolve_project_or_absolute_path(args.checkpoint)
    args.adapter_checkpoint = resolve_project_or_absolute_path(args.adapter_checkpoint)
    args.compare_adapter_checkpoint = [
        resolve_project_or_absolute_path(path) for path in args.compare_adapter_checkpoint
    ]
    args.manifest = resolve_project_or_absolute_path(args.manifest)
    args.run_dir = resolve_run_dir(args.out_dir, args.run_name)
    args.paths = {
        "run_dir": args.run_dir,
        "report": (args.run_dir / "eval_report.json").resolve(),
    }
    return args


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_hospital_model_from_adapter(
    checkpoint_path: Path,
    adapter_checkpoint_path: Path,
    device: torch.device,
) -> tuple[HospitalAdapterClassifier, dict[str, Any], list[str], dict[str, Any]]:
    base_model, checkpoint_metadata, label_names = load_base_checkpoint(checkpoint_path, device)
    adapter_checkpoint = load_adapter_checkpoint(adapter_checkpoint_path, device)
    checkpoint_labels = infer_label_names(checkpoint_metadata, adapter_checkpoint)
    if checkpoint_labels != label_names:
        raise StageFailure("Base checkpoint labels and adapter checkpoint labels do not match.")

    pooled_feature_dim = int(adapter_checkpoint["pooled_feature_dim"])
    classifier = getattr(base_model, "classifier")
    if int(classifier.in_features) != pooled_feature_dim:
        raise StageFailure(
            "Adapter checkpoint pooled feature dimension does not match base model classifier input: "
            f"{pooled_feature_dim} vs {int(classifier.in_features)}"
        )

    adapter = ResidualFeatureAdapter(
        input_dim=pooled_feature_dim,
        bottleneck_dim=int(adapter_checkpoint["adapter_bottleneck"]),
        dropout=float(adapter_checkpoint["adapter_dropout"]),
        scale_init=float(adapter_checkpoint.get("adapter_scale_init", 1e-3)),
    )
    hospital_model = HospitalAdapterClassifier(base_model=base_model, adapter=adapter)
    hospital_model = hospital_model.to(device)
    apply_adapter_checkpoint(hospital_model, adapter_checkpoint)
    return hospital_model, checkpoint_metadata, label_names, adapter_checkpoint


def slugify(name: str) -> str:
    slug = "".join(character if character.isalnum() else "_" for character in name.lower())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "model"


def main() -> int:
    args = resolve_args(parse_args())
    ensure_dir(args.run_dir)

    device = resolve_device(args.device)
    set_seed(args.seed)

    try:
        source_only_model, checkpoint_metadata, label_names = load_base_checkpoint(args.checkpoint, device)
        label_names = infer_label_names(checkpoint_metadata, manifest_paths=[args.manifest])
        image_size = args.image_size or int(checkpoint_metadata.get("image_size") or 224)

        manifest_df, manifest_summary = validate_manifest(
            args.manifest,
            infer_split_name(args.manifest, "eval"),
            label_names,
            args.label_policy,
        )
        dataloader = build_dataloader(
            manifest_df,
            label_names=label_names,
            image_size=image_size,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
            num_workers=args.num_workers,
        )

        evaluations: list[dict[str, Any]] = []

        source_only_metrics = evaluate_model(source_only_model, dataloader, device, label_names)
        source_only_report = build_split_report(
            run_name=args.run_name,
            model_name="source_only",
            split_summary=manifest_summary,
            label_names=label_names,
            metrics=source_only_metrics,
            checkpoint_path=args.checkpoint,
            adapter_checkpoint_path=None,
        )
        source_predictions_path = (args.run_dir / "predictions_source_only.csv").resolve()
        save_predictions_csv(
            manifest_df,
            source_only_metrics["probabilities"],
            source_predictions_path,
            label_names,
        )
        source_only_report["predictions_csv"] = str(source_predictions_path)
        evaluations.append(source_only_report)

        adapter_paths = []
        if args.adapter_checkpoint is not None:
            adapter_paths.append(args.adapter_checkpoint)
        adapter_paths.extend(path for path in args.compare_adapter_checkpoint if path is not None)

        for adapter_checkpoint_path in adapter_paths:
            hospital_model, _, adapter_label_names, adapter_metadata = build_hospital_model_from_adapter(
                checkpoint_path=args.checkpoint,
                adapter_checkpoint_path=adapter_checkpoint_path,
                device=device,
            )
            if adapter_label_names != label_names:
                raise StageFailure("Adapter checkpoint labels do not match evaluation manifest labels.")

            adapter_metrics = evaluate_model(hospital_model, dataloader, device, label_names)
            adapter_name = adapter_metadata.get("target_hospital") or adapter_checkpoint_path.stem
            report = build_split_report(
                run_name=args.run_name,
                model_name=f"adapter_{adapter_name}",
                split_summary=manifest_summary,
                label_names=label_names,
                metrics=adapter_metrics,
                checkpoint_path=args.checkpoint,
                adapter_checkpoint_path=adapter_checkpoint_path,
                extra={
                    "source_hospital": adapter_metadata.get("source_hospital"),
                    "target_hospital": adapter_metadata.get("target_hospital"),
                    "source_only_metrics": report_ready_metrics(source_only_metrics),
                    "metric_deltas": {
                        "macro_auroc_delta": (
                            None
                            if adapter_metrics["macro_auroc"] is None or source_only_metrics["macro_auroc"] is None
                            else float(adapter_metrics["macro_auroc"] - source_only_metrics["macro_auroc"])
                        ),
                        "mean_ap_delta": (
                            None
                            if adapter_metrics["mean_ap"] is None or source_only_metrics["mean_ap"] is None
                            else float(adapter_metrics["mean_ap"] - source_only_metrics["mean_ap"])
                        ),
                    },
                    "adapter_metadata": adapter_metadata,
                },
            )
            predictions_path = (args.run_dir / f"predictions_{slugify(report['model_name'])}.csv").resolve()
            save_predictions_csv(
                manifest_df,
                adapter_metrics["probabilities"],
                predictions_path,
                label_names,
            )
            report["predictions_csv"] = str(predictions_path)
            evaluations.append(report)

        final_report = {
            "run_name": args.run_name,
            "checkpoint": str(args.checkpoint.resolve()),
            "manifest": str(args.manifest.resolve()),
            "label_policy": args.label_policy,
            "device": str(device),
            "image_size": image_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "label_names": label_names,
            "manifest_summary": manifest_summary,
            "evaluations": evaluations,
        }
        save_json(final_report, args.paths["report"])

        print(f"run folder: {args.run_dir}")
        for item in evaluations:
            print(
                f"{item['model_name']}: "
                f"macro_auroc={item['metrics']['macro_auroc']} "
                f"mean_ap={item['metrics']['mean_ap']} "
                f"invalid_auroc_labels={item['metrics']['invalid_auroc_labels']}"
            )
        return 0
    except StageFailure as exc:
        print(f"StageFailure: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - top-level safety
        print(f"Unhandled error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
