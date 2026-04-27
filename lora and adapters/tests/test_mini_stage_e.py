import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_mimic_kshot_support import (  # noqa: E402
    LABELS,
    build_support_summary,
    check_leakage,
    select_kshot_support,
    validate_required_columns,
    validate_support_manifest,
)


def make_row(
    row_id: str,
    subject_id: int,
    study_id: int,
    label_values: dict[str, int],
) -> dict[str, object]:
    return {
        "dicom_id": f"dicom-{row_id}",
        "subject_id": subject_id,
        "study_id": study_id,
        "abs_path": f"/tmp/image-{row_id}.jpg",
        **{label: int(label_values.get(label, 0)) for label in LABELS},
    }


def make_train_pool() -> pd.DataFrame:
    return pd.DataFrame(
        [
            make_row("a", 101, 1001, {"Atelectasis": 1, "Cardiomegaly": 1}),
            make_row("b", 102, 1002, {"Atelectasis": 1, "Consolidation": 1}),
            make_row("c", 103, 1003, {"Cardiomegaly": 1, "Edema": 1}),
            make_row("d", 104, 1004, {"Consolidation": 1, "Effusion": 1}),
            make_row("e", 105, 1005, {"Edema": 1, "Effusion": 1}),
        ]
    )


def test_kshot_sampler_returns_no_duplicate_rows() -> None:
    train_pool = make_train_pool()

    support_df = select_kshot_support(train_pool, k=2, seed=2027)

    assert not support_df.duplicated().any()
    assert not support_df["dicom_id"].duplicated().any()


def test_selected_support_samples_come_only_from_train_pool() -> None:
    train_pool = make_train_pool()

    support_df = select_kshot_support(train_pool, k=2, seed=2027)

    support_paths = set(support_df["abs_path"])
    train_pool_paths = set(train_pool["abs_path"])
    assert support_paths.issubset(train_pool_paths)


def test_achieved_positive_counts_are_computed_correctly() -> None:
    support_df = pd.DataFrame(
        [
            make_row("x", 201, 2001, {"Atelectasis": 1, "Cardiomegaly": 1, "Edema": 1}),
            make_row("y", 202, 2002, {"Consolidation": 1, "Effusion": 1}),
        ]
    )

    summary = build_support_summary(support_df, k=1)

    assert summary["positive_counts"] == {
        "Atelectasis": 1,
        "Cardiomegaly": 1,
        "Consolidation": 1,
        "Edema": 1,
        "Effusion": 1,
    }
    assert all(summary["reached_k"].values())


def test_leakage_checker_catches_subject_overlap() -> None:
    train_pool = pd.DataFrame([make_row("train", 7, 700, {})])
    val = pd.DataFrame([make_row("val", 7, 701, {})])
    test = pd.DataFrame([make_row("test", 9, 900, {})])

    checks, failures = check_leakage(
        {"train_pool": train_pool, "val": val, "test": test},
        {"train_pool": "abs_path", "val": "abs_path", "test": "abs_path"},
    )

    assert checks["subject_overlap"]["train_pool_vs_val"]["count"] == 1
    assert any("subject_id leakage" in failure for failure in failures)


def test_output_support_manifest_has_required_columns() -> None:
    train_pool = make_train_pool()
    support_df = select_kshot_support(train_pool, k=1, seed=2027)

    path_column, column_failures = validate_required_columns(support_df, "support_manifest")
    support_failures = validate_support_manifest(support_df, train_pool, "abs_path")

    assert path_column == "abs_path"
    assert column_failures == []
    assert support_failures == []
