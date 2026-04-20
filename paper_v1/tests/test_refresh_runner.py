from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from paper_v1.runners.run_refresh_chexpert import _run_embedding_export


class RefreshRunnerTest(unittest.TestCase):
    def test_embedding_export_uses_configured_python_and_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            run_root = tmp / "run"
            embeddings_root = run_root / "embeddings" / "by_id"
            export_dir = embeddings_root / "exp0001__paper_v1_chexpert_refresh"
            export_dir.mkdir(parents=True, exist_ok=True)
            index_csv = export_dir / "embedding_index.csv"
            index_csv.write_text("row_id,embedding_file,embedding_row_index\n", encoding="utf-8")
            manifest_csv = tmp / "manifest.csv"
            manifest_csv.write_text("domain,dataset,split,row_id,image_path,study_id,label_atelectasis\n", encoding="utf-8")

            paths = type(
                "Paths",
                (),
                {
                    "root": run_root,
                    "embeddings_root": run_root / "embeddings",
                },
            )()
            config = {
                "embedding_experiment_name": "paper_v1_chexpert_refresh",
                "embedding_batch_size": 32,
                "embedding_kind": "general",
                "token_pooling": "avg",
                "embedding_model_dir": "/workspace/.cache/cxr_foundation",
                "export_python": "/workspace/.venv_cxr_foundation/bin/python",
                "hf_token_env_var": "HF_TOKEN",
            }

            with mock.patch("paper_v1.runners.run_refresh_chexpert.subprocess.run") as run_mock:
                returned_index = _run_embedding_export(paths, config, manifest_csv)

            self.assertEqual(returned_index, index_csv)
            run_mock.assert_called_once()
            command = run_mock.call_args.args[0]
            self.assertEqual(command[0], "/workspace/.venv_cxr_foundation/bin/python")
            self.assertIn("--model-dir", command)
            self.assertIn("/workspace/.cache/cxr_foundation", command)
            self.assertIn("--hf-token-env-var", command)
            self.assertIn("HF_TOKEN", command)


if __name__ == "__main__":
    unittest.main()
