"""Run NIH -> CheXpert adaptation or stop on missing refresh prerequisites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from paper_v1.baselines.ewc import run_ewc
from paper_v1.baselines.finetune_seq import run_finetune_seq
from paper_v1.baselines.fixed_alpha_mix import run_fixed_alpha_mix, select_fixed_alpha
from paper_v1.baselines.l2_anchor import run_l2_anchor
from paper_v1.baselines.lwf import run_lwf
from paper_v1.baselines.source_only import run_source_only
from paper_v1.baselines.vq_summary_replay import run_vq_summary_replay
from paper_v1.data.embeddings import FrozenEmbeddingDataset
from paper_v1.data.splits import SplitSelector, select_records
from paper_v1.evaluation.reporting import write_stage_report, write_summary_table
from paper_v1.models.prototype_memory import PrototypeBank, build_label_state_prototypes, build_vq_summary_bank
from paper_v1.runners.common import collect_alignment, default_device, find_latest_checkpoint, load_config, load_data_config, parse_config_arg, raise_for_issues
from paper_v1.training.stage_adapt import load_linear_head_checkpoint, train_main_method
from paper_v1.utils.io import write_json
from paper_v1.utils.registry import init_run_paths
from paper_v1.utils.seeds import set_seed

REPLAY_FREE_BASELINES = {"finetune_seq", "lwf", "l2_anchor", "ewc"}
SIMPLER_BASELINES = REPLAY_FREE_BASELINES | {"fixed_alpha_mix", "vq_summary_replay"}


def _require_stage0_checkpoint(config: dict) -> Path:
    explicit = config.get("stage0_checkpoint")
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
    discovered = find_latest_checkpoint("/workspace/paper_v1/outputs/stage0_nih_current/*/checkpoints/stage0_best.pt")
    if discovered is not None:
        return discovered
    raise SystemExit("missing Stage 0 checkpoint; run the NIH source baseline first")


def _method_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    if "d0_nih_test" in payload and "d1_chexpert_test" in payload:
        return payload
    return payload["metrics"]


def _maybe_mean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _build_summary_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    source_only_metrics = _method_metrics(results["source_only"])
    stage0_nih_test = float(source_only_metrics["d0_nih_test"]["macro_auroc"])
    rows: list[dict[str, Any]] = []
    for method_name, payload in results.items():
        metrics = _method_metrics(payload)
        nih_metrics = metrics["d0_nih_test"]
        chex_metrics = metrics["d1_chexpert_test"]
        nih_auroc = float(nih_metrics["macro_auroc"])
        chex_auroc = float(chex_metrics["macro_auroc"])
        rows.append(
            {
                "method": method_name,
                "d0_nih_test_macro_auroc": nih_auroc,
                "d1_chexpert_test_macro_auroc": chex_auroc,
                "seen_average_macro_auroc": float((nih_auroc + chex_auroc) / 2.0),
                "nih_forgetting_macro_auroc": float(max(stage0_nih_test - nih_auroc, 0.0)),
                "d0_nih_test_brier_score": nih_metrics.get("brier_score"),
                "d1_chexpert_test_brier_score": chex_metrics.get("brier_score"),
                "average_brier_score": _maybe_mean([nih_metrics.get("brier_score"), chex_metrics.get("brier_score")]),
                "d0_nih_test_macro_ece": nih_metrics.get("macro_ece"),
                "d1_chexpert_test_macro_ece": chex_metrics.get("macro_ece"),
                "average_macro_ece": _maybe_mean([nih_metrics.get("macro_ece"), chex_metrics.get("macro_ece")]),
            }
        )
    return rows


def _find_row(rows: list[dict[str, Any]], method_name: str) -> dict[str, Any] | None:
    for row in rows:
        if row["method"] == method_name:
            return row
    return None


def _best_row(rows: list[dict[str, Any]], methods: set[str], *, key: str, reverse: bool) -> dict[str, Any] | None:
    eligible = [row for row in rows if row["method"] in methods]
    if not eligible:
        return None
    return sorted(eligible, key=lambda row: row[key], reverse=reverse)[0]


def _evaluate_pilot_gate(rows: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    main_row = _find_row(rows, "main_method")
    if main_row is None:
        return {"status": "main_method_not_run", "pass_any": False}

    replay_free_best = _best_row(rows, REPLAY_FREE_BASELINES, key="seen_average_macro_auroc", reverse=True)
    simpler_best_seen = _best_row(rows, SIMPLER_BASELINES, key="seen_average_macro_auroc", reverse=True)
    similar_chex_candidates = [
        row
        for row in rows
        if row["method"] in SIMPLER_BASELINES
        and abs(row["d1_chexpert_test_macro_auroc"] - main_row["d1_chexpert_test_macro_auroc"]) <= 0.005
    ]
    similar_chex_reference = None
    if similar_chex_candidates:
        similar_chex_reference = sorted(
            similar_chex_candidates,
            key=lambda row: row["seen_average_macro_auroc"],
            reverse=True,
        )[0]

    seen_average_gain = None
    seen_average_gate = False
    if replay_free_best is not None:
        seen_average_gain = float(
            main_row["seen_average_macro_auroc"] - replay_free_best["seen_average_macro_auroc"]
        )
        seen_average_gate = seen_average_gain >= 0.005

    forgetting_delta = None
    forgetting_ratio = None
    forgetting_gate = False
    if similar_chex_reference is not None:
        reference_forgetting = float(similar_chex_reference["nih_forgetting_macro_auroc"])
        forgetting_delta = float(reference_forgetting - main_row["nih_forgetting_macro_auroc"])
        if reference_forgetting > 0:
            forgetting_ratio = float(main_row["nih_forgetting_macro_auroc"] / reference_forgetting)
            forgetting_gate = forgetting_ratio <= 0.8

    calibration_delta = None
    calibration_gate = False
    if simpler_best_seen is not None and simpler_best_seen.get("average_macro_ece") is not None:
        simpler_ece = float(simpler_best_seen["average_macro_ece"])
        main_ece = float(main_row["average_macro_ece"])
        calibration_delta = float(simpler_ece - main_ece)
        main_summary = results.get("main_method", {})
        main_memory_mb = float(main_summary.get("memory_size_mb", 0.0))
        seen_average_gap = float(main_row["seen_average_macro_auroc"] - simpler_best_seen["seen_average_macro_auroc"])
        # Heuristic interpretation of the plan's "clear calibration improvement without bloated memory cost".
        calibration_gate = calibration_delta >= 0.01 and seen_average_gap >= -0.002 and main_memory_mb <= 1.0

    pass_any = bool(seen_average_gate or forgetting_gate or calibration_gate)
    return {
        "status": "evaluated",
        "pass_any": pass_any,
        "main_method": main_row,
        "best_replay_free_baseline": replay_free_best,
        "similar_chex_reference": similar_chex_reference,
        "best_simpler_seen_average": simpler_best_seen,
        "criteria": {
            "seen_average_gain_vs_best_replay_free": seen_average_gain,
            "seen_average_gate_pass": seen_average_gate,
            "forgetting_delta_vs_similar_chex_reference": forgetting_delta,
            "forgetting_ratio_vs_similar_chex_reference": forgetting_ratio,
            "forgetting_gate_pass": forgetting_gate,
            "average_macro_ece_gain_vs_best_simpler_seen_average": calibration_delta,
            "calibration_gate_pass": calibration_gate,
        },
    }


def run(config: dict) -> dict:
    set_seed(int(config.get("seed", 1337)))
    data_config = load_data_config(config["data_config"])
    manifest, alignment = collect_alignment(
        data_config=data_config,
        domains={"d0_nih", "d1_chexpert"},
        splits={"train", "val", "test"},
        missing_embedding_policy=str(config["validation"].get("missing_embedding_policy", "error")),
        require_patient_disjoint=bool(config["validation"].get("require_patient_disjoint", True)),
    )
    try:
        raise_for_issues(alignment.issues, allowed_codes=set(config["validation"].get("allowed_issue_codes", [])))
    except SystemExit as exc:
        hint = config["refresh_prerequisites"]
        raise SystemExit(
            f"{exc}\nCheXpert refresh is required before Stage 1.\n"
            f"- kaggle.json: {hint['kaggle_config']}\n"
            f"- reference scripts: {', '.join(hint['reference_scripts'])}"
        ) from None

    stage0_checkpoint = _require_stage0_checkpoint(config)
    device = default_device()
    output_paths = init_run_paths(config["output_root"], config["experiment_name"])
    train_dataset = FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d1_chexpert", "train")]))
    val_dataset = FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d1_chexpert", "val")]))
    eval_datasets = {
        "d0_nih_test": FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d0_nih", "test")])),
        "d1_chexpert_test": FrozenEmbeddingDataset(select_records(alignment.records, [SplitSelector("d1_chexpert", "test")])),
    }
    source_only_paths = init_run_paths(output_paths.root, "source_only")
    source_only_metrics = run_source_only(
        checkpoint_path=stage0_checkpoint,
        eval_datasets=eval_datasets,
        output_dir=source_only_paths.root,
        batch_size=int(config["training"].get("batch_size", 256)),
        device=device,
    )
    source_embeddings, source_targets, _ = FrozenEmbeddingDataset(
        select_records(alignment.records, [SplitSelector("d0_nih", "train")])
    ).materialize_numpy()
    source_bank = build_label_state_prototypes(
        source_embeddings,
        source_targets,
        domain="d0_nih",
        positive_k=int(config["memory"].get("positive_k", 4)),
        negative_k=int(config["memory"].get("negative_k", 2)),
        seed=int(config.get("seed", 1337)),
    )
    source_bank.save(output_paths.artifacts / "nih_source_bank")
    vq_bank = build_vq_summary_bank(
        source_embeddings,
        source_targets,
        budget_bytes=source_bank.memory_size_bytes(),
        seed=int(config.get("seed", 1337)),
    )

    results = {"source_only": source_only_metrics}
    method_map = {
        "finetune_seq": run_finetune_seq,
        "lwf": run_lwf,
        "l2_anchor": run_l2_anchor,
        "ewc": run_ewc,
        "vq_summary_replay": run_vq_summary_replay,
    }
    fisher_path = config.get("stage0_fisher_path")
    fisher_state = torch.load(fisher_path, map_location="cpu", weights_only=False) if fisher_path else None
    for method_name in config.get("baselines", []):
        if method_name not in method_map:
            continue
        method_paths = init_run_paths(output_paths.root, method_name)
        kwargs = {
            "previous_checkpoint_path": stage0_checkpoint,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "eval_datasets": eval_datasets,
            "output_dir": method_paths.root,
            "training_config": config["training"],
            "device": device,
            "seed": int(config.get("seed", 1337)),
        }
        if method_name == "ewc":
            kwargs["fisher_state"] = fisher_state
        if method_name == "vq_summary_replay":
            kwargs["replay_bank"] = vq_bank
        results[method_name] = method_map[method_name](**kwargs)

    if "fixed_alpha_mix" in config.get("baselines", []):
        base_checkpoint = results["finetune_seq"]["checkpoint_path"]
        base_model = load_linear_head_checkpoint(base_checkpoint, device=device)
        memory_module = source_bank.to_module(
            top_k=int(config["memory"].get("top_k", 8)),
            temperature=float(config["memory"].get("temperature", 0.1)),
        ).to(device)
        alpha, _ = select_fixed_alpha(
            base_model=base_model,
            memory_module=memory_module,
            val_dataset=val_dataset,
            batch_size=int(config["training"].get("batch_size", 256)),
            device=device,
            alpha_grid=list(config["memory"].get("alpha_grid", [0.1, 0.25, 0.5])),
        )
        mix_paths = init_run_paths(output_paths.root, "fixed_alpha_mix")
        results["fixed_alpha_mix"] = run_fixed_alpha_mix(
            base_model=base_model,
            memory_module=memory_module,
            eval_datasets=eval_datasets,
            alpha=alpha,
            batch_size=int(config["training"].get("batch_size", 256)),
            device=device,
            output_dir=mix_paths.root,
        )

    if "main_method" in config.get("baselines", []):
        main_paths = init_run_paths(output_paths.root, "main_method")
        results["main_method"] = train_main_method(
            previous_checkpoint_path=stage0_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_datasets=eval_datasets,
            old_bank=source_bank,
            output_dir=main_paths.root,
            training_config=config["main_method"],
            device=device,
            seed=int(config.get("seed", 1337)),
        )

    rows = _build_summary_rows(results)
    pilot_gate = _evaluate_pilot_gate(rows, results)
    write_summary_table(output_paths.reports, "nih_to_chexpert_summary.csv", rows)
    write_json(output_paths.artifacts / "pilot_gate.json", pilot_gate)

    leaderboard = sorted(rows, key=lambda row: row["seen_average_macro_auroc"], reverse=True)
    summary_lines = [
        (
            f"- `{row['method']}`: seen_avg=`{row['seen_average_macro_auroc']:.4f}` "
            f"nih=`{row['d0_nih_test_macro_auroc']:.4f}` "
            f"chexpert=`{row['d1_chexpert_test_macro_auroc']:.4f}` "
            f"forgetting=`{row['nih_forgetting_macro_auroc']:.4f}` "
            f"avg_ece=`{row['average_macro_ece']:.4f}`"
        )
        for row in leaderboard
    ]
    pilot_gate_lines = [f"- Continue to Stage 2: `{pilot_gate['pass_any']}`"]
    if pilot_gate.get("best_replay_free_baseline") is not None:
        pilot_gate_lines.append(
            f"- Best replay-free baseline by seen-average: "
            f"`{pilot_gate['best_replay_free_baseline']['method']}`"
        )
    if pilot_gate.get("similar_chex_reference") is not None:
        pilot_gate_lines.append(
            f"- Similar-CheXpert forgetting reference: "
            f"`{pilot_gate['similar_chex_reference']['method']}`"
        )
    criteria = pilot_gate.get("criteria", {})
    for key, value in criteria.items():
        if isinstance(value, float):
            pilot_gate_lines.append(f"- `{key}`: `{value:.6f}`")
        else:
            pilot_gate_lines.append(f"- `{key}`: `{value}`")

    write_stage_report(
        output_paths.reports,
        "nih_to_chexpert_report.md",
        title="NIH -> CheXpert Report",
        sections=[
            ("Manifest", [f"- `{manifest.path}`"]),
            ("Leaderboard", summary_lines),
            ("Pilot Gate", pilot_gate_lines),
        ],
    )
    return {
        "run_root": str(output_paths.root),
        "pilot_gate_path": str(output_paths.artifacts / "pilot_gate.json"),
        "continue_to_stage2": pilot_gate["pass_any"],
    }


def main() -> None:
    args = parse_config_arg("Run NIH -> CheXpert sequential adaptation.")
    config = load_config(args.config)
    result = run(config)
    print(f"[done] wrote Stage 1 artifacts to {result['run_root']}")


if __name__ == "__main__":
    main()
