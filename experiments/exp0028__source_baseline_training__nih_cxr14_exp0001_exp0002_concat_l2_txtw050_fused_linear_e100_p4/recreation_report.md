# Source Baseline Recreation Report

## Scope

This report documents how to recreate the frozen-embedding multilabel baseline experiment stored at:

`/workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4`

The producing script is:

`/workspace/scripts/04_train_frozen_multilabel_baseline.py`

Script SHA-256:

`d4b2ac55787799801ab4423762a3bba01fc060ec17a54172ff70cf1cef69743a`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4`
- Experiment id: `exp0028`
- Operation label: `source_baseline_training`
- Input embedding root: `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Label count: `14`
- Selection metric: `macro_auroc`
- Batch size: `512`
- Epoch budget: `100`
- Patience: `4`
- Learning rate: `0.001`
- Weight decay: `0.0001`
- Seed: `1337`
- Device requested: `auto`
- Device resolved during run: `cuda`
- Mixed precision on CUDA: `true`
- Best epoch: `100`

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
  /workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050 \
  --manifest-csv \
  /workspace/manifest_nih_cxr14_all14.csv \
  --batch-size \
  512 \
  --epochs \
  100 \
  --lr \
  1e-3 \
  --weight-decay \
  1e-4 \
  --patience \
  4 \
  --seed \
  1337 \
  --device \
  auto \
  --fp16-on-cuda \
  --experiment-name \
  exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4 \
  --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python \
  /workspace/scripts/04_train_frozen_multilabel_baseline.py \
  --embedding-root \
  /workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050 \
  --manifest-csv \
  /workspace/manifest_nih_cxr14_all14.csv \
  --batch-size \
  512 \
  --epochs \
  100 \
  --lr \
  1e-3 \
  --weight-decay \
  1e-4 \
  --patience \
  4 \
  --seed \
  1337 \
  --device \
  auto \
  --fp16-on-cuda \
  --experiment-name \
  source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4
```

## Preconditions

- The embedding experiment must already exist at `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The embedding experiment must contain `train/embeddings.npy`, `val/embeddings.npy`, and `test/embeddings.npy`.
- Each split must have a row-identity sidecar such as `image_paths.txt` or `report_ids.json`.
- The sidecar-derived row IDs must match `Path(image_path).stem` from the manifest for every split row.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Train rows: `78,571`
- Train embedding dim: `2176`
- Train sidecar: `image_paths.txt`
- Train ID parser: `stem`
- Val rows: `11,219`
- Val embedding dim: `2176`
- Val sidecar: `image_paths.txt`
- Val ID parser: `stem`
- Test rows: `22,330`
- Test embedding dim: `2176`
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

- config.json: `4.09K`
- experiment_meta.json: `3.92K`
- best.ckpt: `123K`
- val_metrics.json: `8.34K`
- test_metrics.json: `8.36K`
- val_f1_thresholds.json: `2.66K`
- train_log.jsonl: `29.6K`

## Final Metrics

- Validation macro AUROC: `0.775530`
- Validation macro average precision: `0.160135`
- Validation macro ECE: `0.322219`
- Validation macro F1 @ 0.5: `0.181880`
- Validation macro F1 @ tuned thresholds: `0.231764`
- Test macro AUROC: `0.774630`
- Test macro average precision: `0.160831`
- Test macro ECE: `0.322135`
- Test macro F1 @ 0.5: `0.180154`
- Test macro F1 @ tuned thresholds: `0.221506`
- Macro mean tuned threshold: `0.740404`

## Final Artifact SHA-256

```text
8c3d3ab52257ed8546c9973b05837ab87176a0906e35cea868cd4cecc7ade223  /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4/config.json
1d7841b2da0cc336405f7ba5853d6a783298a6a3e2d668f59e63fc8809ffabb0  /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4/experiment_meta.json
87bce987c168ab1e5d203f62687616bfc5832a04dcd7dbc5e6301619c0564dc8  /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4/best.ckpt
254360899333a05b0d5ef1c54b87a49e996a714e9a8d69e888ccfd8be708af27  /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4/val_metrics.json
3b4b9b32e6455b64c1a78366fac4e5c366699f6c32556752fcc8acd226a18793  /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4/test_metrics.json
8cdcf8c6771966a99d036d77b25b0035aefb7152e74981de32349a497ae33f27  /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4/val_f1_thresholds.json
aedeaa34a72cd38b84c351f5657d03a8786d34f22fe2a3fe77e0510a5a75e6d1  /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4/train_log.jsonl
```

## Important Reproduction Notes

- `config.json`, `experiment_meta.json`, and `recreation_report.md` include timestamps, so their hashes change on rerun.
- Thresholds are tuned only on the validation split and then reused unchanged for test metrics.
- The model is a single linear layer on top of frozen embeddings; the input embedding experiment determines the feature space.
- The trainer auto-detects the split sidecar and aligns labels by sample ID rather than by trusting row position in the manifest.
- After a successful run, keep the generated `recreation_report.md`, commit the experiment directory, and push before starting the next experiment.

## Agent Handoff Text

```text
Use /workspace/scripts/04_train_frozen_multilabel_baseline.py and the report /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4/recreation_report.md to recreate the frozen NIH CXR14 source baseline for embeddings from /workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050. Run the exact command in the report, verify the saved metrics and checkpoint hashes, and confirm that validation-tuned thresholds are reused for test evaluation.
```
