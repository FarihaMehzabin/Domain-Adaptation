"""Run early ablations once the Stage 1 pipeline is available."""

from __future__ import annotations

from paper_v1.runners.common import load_config, parse_config_arg


def run(config: dict) -> dict:
    raise SystemExit(
        "Ablations are implemented at the module level, but the runner remains blocked until "
        "a refreshed CheXpert manifest and Stage 1 artifacts exist."
    )


def main() -> None:
    args = parse_config_arg("Run paper_v1 ablations.")
    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
