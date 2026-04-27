import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_mimic_common5 import (  # noqa: E402
    FINAL_LABELS,
    build_common5_labels,
    check_split_leakage,
    prepare_final_manifest,
    raw_label_column_name,
    uzero_value,
)


def make_base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dicom_id": "dicom-a",
                "subject_id": 101,
                "study_id": 1001,
                "split": "train",
                "abs_path": "/tmp/dicom-a.jpg",
                "rel_path": "raw/train/p10/p101/s1001/dicom-a.jpg",
                "path_method": "recursive_search",
                "ViewPosition": "AP",
                "Rows": 2048,
                "Columns": 2048,
                "Atelectasis": -1.0,
                "Cardiomegaly": 1.0,
                "Consolidation": 0.0,
                "Edema": None,
                "Pleural Effusion": -1.0,
            }
        ]
    )


def test_label_conversion_and_effusion_mapping() -> None:
    converted = build_common5_labels(make_base_frame())
    assert converted.loc[0, raw_label_column_name("Pleural Effusion")] == -1.0
    assert converted.loc[0, "Effusion"] == 0
    assert converted.loc[0, "Cardiomegaly"] == 1
    assert converted.loc[0, "Atelectasis"] == 0


def test_uzero_converts_minus_one_and_missing_to_zero() -> None:
    assert uzero_value(-1) == 0
    assert uzero_value(0) == 0
    assert uzero_value(1) == 1
    assert uzero_value(float("nan")) == 0


def test_leakage_checker_catches_subject_overlap(tmp_path: Path) -> None:
    train_image = tmp_path / "train.jpg"
    val_image = tmp_path / "val.jpg"
    train_image.write_bytes(b"train")
    val_image.write_bytes(b"val")

    train_frame = pd.DataFrame(
        [
            {
                "dicom_id": "dicom-train",
                "subject_id": 7,
                "study_id": 70,
                "abs_path": str(train_image),
                **{label: 0 for label in FINAL_LABELS},
            }
        ]
    )
    val_frame = pd.DataFrame(
        [
            {
                "dicom_id": "dicom-val",
                "subject_id": 7,
                "study_id": 71,
                "abs_path": str(val_image),
                **{label: 0 for label in FINAL_LABELS},
            }
        ]
    )
    test_frame = pd.DataFrame(columns=train_frame.columns)

    checks, failures = check_split_leakage(
        {"train_pool": train_frame, "val": val_frame, "test": test_frame}
    )

    assert checks["subject_overlap"]["train_pool_vs_val"]["count"] == 1
    assert any("subject_id leakage" in failure for failure in failures)


def test_final_manifest_columns_are_present() -> None:
    converted = build_common5_labels(make_base_frame())
    manifest = prepare_final_manifest(converted, split_name="train_pool")

    required_columns = {
        "dicom_id",
        "subject_id",
        "study_id",
        "split",
        "abs_path",
        "rel_path",
        "path_method",
        "ViewPosition",
        "Rows",
        "Columns",
        raw_label_column_name("Atelectasis"),
        raw_label_column_name("Cardiomegaly"),
        raw_label_column_name("Consolidation"),
        raw_label_column_name("Edema"),
        raw_label_column_name("Pleural Effusion"),
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
    }
    assert required_columns.issubset(set(manifest.columns))
    assert manifest.loc[0, "split"] == "train_pool"
