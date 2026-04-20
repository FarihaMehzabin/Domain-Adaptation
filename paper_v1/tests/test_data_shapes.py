from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper_v1.data.embeddings import EmbeddingAssetSpec, FrozenEmbeddingDataset, align_manifest_records
from paper_v1.data.manifests import load_manifest


class DataShapeTest(unittest.TestCase):
    def test_alignment_loads_fake_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            embedding_dir = tmp / "assets" / "d0_nih" / "train"
            embedding_dir.mkdir(parents=True)
            embeddings = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            np.save(embedding_dir / "embeddings.npy", embeddings)
            manifest_path = tmp / "manifest.csv"
            manifest_path.write_text(
                "domain,dataset,split,source_split,row_id,image_path,patient_id,study_id,view_raw,view_group,sex,age,"
                "label_atelectasis,label_cardiomegaly,label_consolidation,label_edema,label_pleural_effusion,label_pneumonia,label_pneumothorax\n"
                "d0_nih,nih,train,train,row0,x,1,NA,PA,FRONTAL,F,10,1,0,0,0,0,0,0\n"
                "d0_nih,nih,train,train,row1,x,2,NA,PA,FRONTAL,F,11,0,1,0,0,0,0,0\n",
                encoding="utf-8",
            )
            index_path = tmp / "assets" / "embedding_index.csv"
            index_path.write_text(
                "study_id,domain,split,dataset,row_id,image_path,embedding_path,embedding_row,label_vector\n"
                "NA,d0_nih,train,nih,row0,x,d0_nih/train/embeddings.npy,0,\"[1,0,0,0,0,0,0]\"\n"
                "NA,d0_nih,train,nih,row1,x,d0_nih/train/embeddings.npy,1,\"[0,1,0,0,0,0,0]\"\n",
                encoding="utf-8",
            )
            manifest = load_manifest(manifest_path)
            alignment = align_manifest_records(
                manifest.records,
                {"d0_nih": EmbeddingAssetSpec(name="fake", index_csv=index_path)},
                expected_dim=2,
            )
            dataset = FrozenEmbeddingDataset(alignment.records)
            embedding, _, _ = dataset.materialize_numpy()
            self.assertEqual(embedding.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
