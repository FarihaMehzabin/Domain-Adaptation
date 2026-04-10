# Source Baseline Recreation Report

## Scope

This report documents how to recreate the frozen-embedding multilabel baseline experiment stored at:

`/workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096`

The producing script is:

`/workspace/scripts/04_train_frozen_multilabel_baseline.py`

Script SHA-256:

`d4b2ac55787799801ab4423762a3bba01fc060ec17a54172ff70cf1cef69743a`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096`
- Experiment id: `exp0012`
- Operation label: `source_baseline_training`
- Input embedding root: `/workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160`
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
  /workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160 \
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
  exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096 \
  --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python \
  /workspace/scripts/04_train_frozen_multilabel_baseline.py \
  --embedding-root \
  /workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160 \
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
  source_baseline_training__xattn_full_pipeline_baseline_bs4096
```

## Preconditions

- The embedding experiment must already exist at `/workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160`.
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

- config.json: `3.90K`
- experiment_meta.json: `3.62K`
- best.ckpt: `32.2K`
- val_metrics.json: `8.33K`
- test_metrics.json: `8.33K`
- val_f1_thresholds.json: `2.67K`
- train_log.jsonl: `5.93K`

## Final Metrics

- Validation macro AUROC: `0.722824`
- Validation macro average precision: `0.117825`
- Validation macro ECE: `0.367606`
- Validation macro F1 @ 0.5: `0.153240`
- Validation macro F1 @ tuned thresholds: `0.186074`
- Test macro AUROC: `0.727785`
- Test macro average precision: `0.111756`
- Test macro ECE: `0.368609`
- Test macro F1 @ 0.5: `0.155103`
- Test macro F1 @ tuned thresholds: `0.173008`
- Macro mean tuned threshold: `0.704715`

## Final Artifact SHA-256

```text
2798325ea85f17333c70b748b3a6276ce7bab5ffd187f74a44314df34cdb3724  /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096/config.json
65f741ce986e18e43206fc6a66f32222e100a0e80b50935e7ebdd89851e140d6  /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096/experiment_meta.json
3514d1c5488663733fecbcf2e1cf4a104c1614de5e4e05bb305f1104a8a23f3c  /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096/best.ckpt
446f0abdc191ea0a9735ebe9d202fc69c755ee22d6da347bacc04ab230435e86  /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096/val_metrics.json
d5dfbc52d568f7c913a83724e2140f007616645ba979a45785e28059cbd22d1e  /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096/test_metrics.json
2366b0a31768f995bf2aebe707d95904cca95366df88c6b885c2de24eb3fa79b  /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096/val_f1_thresholds.json
f93bdc17f02ec5b91f3eceb04bf5a6e7549e0d2b73372668f22635db4aed80b9  /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096/train_log.jsonl
```

## Important Reproduction Notes

- `config.json`, `experiment_meta.json`, and `recreation_report.md` include timestamps, so their hashes change on rerun.
- Thresholds are tuned only on the validation split and then reused unchanged for test metrics.
- The model is a single linear layer on top of frozen embeddings; the input embedding experiment determines the feature space.
- The trainer auto-detects the split sidecar and aligns labels by sample ID rather than by trusting row position in the manifest.
- After a successful run, keep the generated `recreation_report.md`, commit the experiment directory, and push before starting the next experiment.

## Agent Handoff Text

```text
Use /workspace/scripts/04_train_frozen_multilabel_baseline.py and the report /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096/recreation_report.md to recreate the frozen NIH CXR14 source baseline for embeddings from /workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160. Run the exact command in the report, verify the saved metrics and checkpoint hashes, and confirm that validation-tuned thresholds are reused for test evaluation.
```
