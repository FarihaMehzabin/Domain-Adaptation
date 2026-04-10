# Source Baseline Recreation Report

## Scope

This report documents how to recreate the frozen-embedding multilabel baseline experiment stored at:

`/workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096`

The producing script is:

`/workspace/scripts/04_train_frozen_multilabel_baseline.py`

Script SHA-256:

`d4b2ac55787799801ab4423762a3bba01fc060ec17a54172ff70cf1cef69743a`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096`
- Experiment id: `exp0020`
- Operation label: `source_baseline_training`
- Input embedding root: `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Label count: `14`
- Selection metric: `macro_auroc`
- Batch size: `4096`
- Epoch budget: `20`
- Patience: `3`
- Learning rate: `0.001`
- Weight decay: `0.0001`
- Seed: `1337`
- Device requested: `cuda`
- Device resolved during run: `cuda`
- Mixed precision on CUDA: `true`
- Best epoch: `20`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- PyTorch: `2.4.1+cu124`
- CUDA available: `true`
- GPU used: `NVIDIA RTX A4500`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python \
  /workspace/scripts/04_train_frozen_multilabel_baseline.py \
  --embedding-root \
  /workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160 \
  --manifest-csv \
  /workspace/manifest_nih_cxr14_all14.csv \
  --device \
  cuda \
  --fp16-on-cuda \
  --batch-size \
  4096 \
  --epochs \
  20 \
  --patience \
  3 \
  --experiment-name \
  exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096 \
  --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python \
  /workspace/scripts/04_train_frozen_multilabel_baseline.py \
  --embedding-root \
  /workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160 \
  --manifest-csv \
  /workspace/manifest_nih_cxr14_all14.csv \
  --device \
  cuda \
  --fp16-on-cuda \
  --batch-size \
  4096 \
  --epochs \
  20 \
  --patience \
  3 \
  --experiment-name \
  source_baseline_training__xattn_long_e50_baseline_bs4096
```

## Preconditions

- The embedding experiment must already exist at `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The embedding experiment must contain `train/embeddings.npy`, `val/embeddings.npy`, and `test/embeddings.npy`.
- Each split must have a row-identity sidecar such as `image_paths.txt` or `report_ids.json`.
- The sidecar-derived row IDs must match `Path(image_path).stem` from the manifest for every split row.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Train rows: `78,569`
- Train embedding dim: `512`
- Train sidecar: `image_paths.txt`
- Train ID parser: `stem`
- Val rows: `11,219`
- Val embedding dim: `512`
- Val sidecar: `image_paths.txt`
- Val ID parser: `stem`
- Test rows: `22,330`
- Test embedding dim: `512`
- Test sidecar: `image_paths.txt`
- Test ID parser: `stem`

## Expected Outputs

- `config.json`
- `experiment_meta.json`
- `best.ckpt`
- `val_metrics.json`
- `test_metrics.json`
- `val_f1_thresholds.json`
- `train_log.jsonl`

## Output Sizes

- config.json: `3.85K`
- experiment_meta.json: `3.54K`
- best.ckpt: `32.2K`
- val_metrics.json: `8.34K`
- test_metrics.json: `8.37K`
- val_f1_thresholds.json: `2.67K`
- train_log.jsonl: `5.93K`

## Final Metrics

- Validation macro AUROC: `0.755293`
- Validation macro average precision: `0.139524`
- Validation macro ECE: `0.318028`
- Validation macro F1 @ 0.5: `0.168271`
- Validation macro F1 @ tuned thresholds: `0.214465`
- Test macro AUROC: `0.759056`
- Test macro average precision: `0.137536`
- Test macro ECE: `0.318471`
- Test macro F1 @ 0.5: `0.168394`
- Test macro F1 @ tuned thresholds: `0.199873`
- Macro mean tuned threshold: `0.748787`

## Final Artifact SHA-256

```text
50850e824f04132a18fd6bc05f0a24692c9353a020be8b9cd4d5ebf6769b86ee  /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096/config.json
ed6bac2b0bfb5eb3cf187a0f93edc6085babe9adf96627307d9ad12ede45f7bf  /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096/experiment_meta.json
f344b3a91ae9397e5c7608b15a20328b4ae153c62f04f349c61adfb2a2352a73  /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096/best.ckpt
b1ca49916ce9727e66bb57b9fb9d1482055970cfd388060a4591d311faafabf8  /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096/val_metrics.json
0af99ca9a3f27ce0b909a6d9f0bf63296c978c110f2d69bc0b66e27e97d3ff15  /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096/test_metrics.json
996d15f8d2896b7f80479062617c7e407ac696e85631a1a449fb35d01a75c0dc  /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096/val_f1_thresholds.json
5c154207a96f3d7762e0a914e32251ee4f51d75261ce27088af15ca3751a945d  /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096/train_log.jsonl
```

## Important Reproduction Notes

- `config.json`, `experiment_meta.json`, and `recreation_report.md` include timestamps, so their hashes change on rerun.
- Thresholds are tuned only on the validation split and then reused unchanged for test metrics.
- The model is a single linear layer on top of frozen embeddings; the input embedding experiment determines the feature space.
- The trainer auto-detects the split sidecar and aligns labels by sample ID rather than by trusting row position in the manifest.
- After a successful run, keep the generated `recreation_report.md`, commit the experiment directory, and push before starting the next experiment.

## Agent Handoff Text

```text
Use /workspace/scripts/04_train_frozen_multilabel_baseline.py and the report /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096/recreation_report.md to recreate the frozen NIH CXR14 source baseline for embeddings from /workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160. Run the exact command in the report, verify the saved metrics and checkpoint hashes, and confirm that validation-tuned thresholds are reused for test evaluation.
```
