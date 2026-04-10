# Source Baseline Recreation Report

## Scope

This report documents how to recreate the frozen-embedding multilabel baseline experiment stored at:

`/workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096`

The producing script is:

`/workspace/scripts/04_train_frozen_multilabel_baseline.py`

Script SHA-256:

`d4b2ac55787799801ab4423762a3bba01fc060ec17a54172ff70cf1cef69743a`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096`
- Experiment id: `exp0028`
- Operation label: `source_baseline_training`
- Input embedding root: `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat`
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
  /workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat \
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
  exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096 \
  --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python \
  /workspace/scripts/04_train_frozen_multilabel_baseline.py \
  --embedding-root \
  /workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat \
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
  source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096
```

## Preconditions

- The embedding experiment must already exist at `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The embedding experiment must contain `train/embeddings.npy`, `val/embeddings.npy`, and `test/embeddings.npy`.
- Each split must have a row-identity sidecar such as `image_paths.txt` or `report_ids.json`.
- The sidecar-derived row IDs must match `Path(image_path).stem` from the manifest for every split row.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Train rows: `78,569`
- Train embedding dim: `2688`
- Train sidecar: `image_paths.txt`
- Train ID parser: `stem`
- Val rows: `11,219`
- Val embedding dim: `2688`
- Val sidecar: `image_paths.txt`
- Val ID parser: `stem`
- Test rows: `22,330`
- Test embedding dim: `2688`
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

- config.json: `3.78K`
- experiment_meta.json: `3.50K`
- best.ckpt: `151K`
- val_metrics.json: `8.32K`
- test_metrics.json: `8.33K`
- val_f1_thresholds.json: `2.67K`
- train_log.jsonl: `5.92K`

## Final Metrics

- Validation macro AUROC: `0.766078`
- Validation macro average precision: `0.151849`
- Validation macro ECE: `0.324019`
- Validation macro F1 @ 0.5: `0.174500`
- Validation macro F1 @ tuned thresholds: `0.224328`
- Test macro AUROC: `0.770829`
- Test macro average precision: `0.151510`
- Test macro ECE: `0.324818`
- Test macro F1 @ 0.5: `0.173262`
- Test macro F1 @ tuned thresholds: `0.207465`
- Macro mean tuned threshold: `0.734570`

## Final Artifact SHA-256

```text
82696aa9089186b80452e0f8c231f5342408b418572255806c9c3c40aa08cbe3  /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096/config.json
b332089ff35aba94b4c612efd7f96d141c8a20811ba0ad4ead1201a0792dbfdc  /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096/experiment_meta.json
d06f74e8fa3d3b068a9e7cda5f5bbd81213ad67d5e3534b8bf931b961807ceed  /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096/best.ckpt
7f7dbca4ba0ce8db0fe05c7bf3f7e9a2ed0a2d4a347e0b8c310d12c1f6713ea3  /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096/val_metrics.json
6922eef8866063d38218a8ee4e3a09dc51ad73f6f01421f3cd5dae57a8b6eac6  /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096/test_metrics.json
51b6e571b997a7762ccf8cad54aede592c9fce84e9fe210cb07df397af60dd96  /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096/val_f1_thresholds.json
c9ed68bdf81f599f8f1487c72bcbdcc8ac4110c5adb0678ae364dbacba2cb8fc  /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096/train_log.jsonl
```

## Important Reproduction Notes

- `config.json`, `experiment_meta.json`, and `recreation_report.md` include timestamps, so their hashes change on rerun.
- Thresholds are tuned only on the validation split and then reused unchanged for test metrics.
- The model is a single linear layer on top of frozen embeddings; the input embedding experiment determines the feature space.
- The trainer auto-detects the split sidecar and aligns labels by sample ID rather than by trusting row position in the manifest.
- After a successful run, keep the generated `recreation_report.md`, commit the experiment directory, and push before starting the next experiment.

## Agent Handoff Text

```text
Use /workspace/scripts/04_train_frozen_multilabel_baseline.py and the report /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096/recreation_report.md to recreate the frozen NIH CXR14 source baseline for embeddings from /workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat. Run the exact command in the report, verify the saved metrics and checkpoint hashes, and confirm that validation-tuned thresholds are reused for test evaluation.
```
