from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from paper_v1.data.chexpert_refresh import (
    build_patient_disjoint_chexpert_manifest,
    download_chexpert_subset,
)


class CheXpertRefreshTest(unittest.TestCase):
    def test_patient_disjoint_refresh_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            train_csv = tmp / "train.csv"
            valid_csv = tmp / "valid.csv"
            header = "Path,Sex,Age,Frontal/Lateral,AP/PA,Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion,Pneumonia,Pneumothorax\n"
            train_csv.write_text(
                header
                + "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg,Male,50,Frontal,AP,1,0,0,0,0,0,0\n"
                + "CheXpert-v1.0-small/train/patient00001/study2/view1_frontal.jpg,Male,50,Frontal,AP,0,1,0,0,0,0,0\n"
                + "CheXpert-v1.0-small/train/patient00002/study1/view1_frontal.jpg,Female,45,Frontal,PA,0,0,1,0,0,0,0\n"
                + "CheXpert-v1.0-small/train/patient00003/study1/view1_frontal.jpg,Female,42,Frontal,PA,0,0,0,1,0,0,0\n",
                encoding="utf-8",
            )
            valid_csv.write_text(
                header
                + "CheXpert-v1.0-small/valid/patient90001/study1/view1_frontal.jpg,Male,60,Frontal,AP,1,0,0,0,0,0,0\n"
                + "CheXpert-v1.0-small/valid/patient90002/study1/view1_frontal.jpg,Female,61,Frontal,PA,0,0,0,0,1,0,0\n",
                encoding="utf-8",
            )
            rows, summary = build_patient_disjoint_chexpert_manifest(
                train_csv=train_csv,
                valid_csv=valid_csv,
                target_train_count=2,
                target_val_count=1,
                target_test_count=-1,
                seed=13,
            )
            by_split = {}
            for row in rows:
                by_split.setdefault(row["split"], []).append(row)
            train_patients = {row["patient_id"] for row in by_split["train"]}
            val_patients = {row["patient_id"] for row in by_split["val"]}
            test_patients = {row["patient_id"] for row in by_split["test"]}
            self.assertFalse(train_patients & val_patients)
            self.assertFalse(train_patients & test_patients)
            self.assertFalse(val_patients & test_patients)
            self.assertEqual(summary["patient_overlap"]["train_val"], 0)

    def test_bulk_subset_download_copies_requested_files(self) -> None:
        class FakeKaggleApi:
            def dataset_download_files(self, dataset_ref: str, path: str, force: bool, quiet: bool) -> None:
                archive_path = Path(path) / "chexpert.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("train.csv", "Path\n")
                    archive.writestr("train/patient00001/study1/view1_frontal.jpg", "fake-image")

        with tempfile.TemporaryDirectory() as tmp_dir:
            destination_root = Path(tmp_dir) / "dest"
            summary = download_chexpert_subset(
                api=FakeKaggleApi(),
                dataset_ref="ashery/chexpert",
                requested_relative_paths=[
                    "train.csv",
                    "train/patient00001/study1/view1_frontal.jpg",
                ],
                destination_root=destination_root,
                bulk_download=True,
            )

            self.assertEqual(summary["mode"], "bulk_dataset_download")
            self.assertTrue((destination_root / "train.csv").exists())
            self.assertTrue((destination_root / "train/patient00001/study1/view1_frontal.jpg").exists())


if __name__ == "__main__":
    unittest.main()
