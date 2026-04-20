"""Run asset reconnaissance."""

from __future__ import annotations

from pathlib import Path

from paper_v1.data.recon import build_recon_summary
from paper_v1.runners.common import build_asset_specs, load_config, load_data_config, parse_config_arg
from paper_v1.utils.registry import init_run_paths


def run(config: dict) -> dict:
    data_config = load_data_config(config["data_config"])
    asset_specs = build_asset_specs(data_config)
    run_paths = init_run_paths(config["output_root"], config["experiment_name"])
    report_path = Path(config.get("report_path", "/workspace/paper_v1/reports/recon_summary.md"))
    summary = build_recon_summary(
        manifest_path=data_config["manifest_path"],
        asset_specs=asset_specs,
        require_patient_disjoint=bool(config.get("require_patient_disjoint", False)),
        missing_embedding_policy=str(config.get("missing_embedding_policy", "drop")),
        output_md=report_path,
        output_json=run_paths.artifacts / "recon_summary.json",
    )
    return {
        "report_path": str(report_path),
        "artifact_path": str(run_paths.artifacts / "recon_summary.json"),
        "summary": summary,
    }


def main() -> None:
    args = parse_config_arg("Run paper_v1 asset reconnaissance.")
    config = load_config(args.config)
    result = run(config)
    print(f"[done] wrote report to {result['report_path']}")


if __name__ == "__main__":
    main()
