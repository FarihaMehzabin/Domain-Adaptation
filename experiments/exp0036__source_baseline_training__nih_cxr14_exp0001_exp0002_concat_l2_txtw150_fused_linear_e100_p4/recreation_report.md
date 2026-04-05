# Source Baseline Recreation Report

## Scope

This report documents how to recreate the frozen-embedding multilabel baseline experiment stored at:

`/workspace/experiments/exp0036__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw150_fused_linear_e100_p4`

The producing script is:

`/workspace/scripts/04_train_frozen_multilabel_baseline.py`

Script SHA-256:

`d4b2ac55787799801ab4423762a3bba01fc060ec17a54172ff70cf1cef69743a`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0036__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw150_fused_linear_e100_p4`
- Experiment id: `exp0036`
- Operation label: `source_baseline_training`
- Input embedding root: `/workspace/experiments/exp0035__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw150`
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
  /workspace/experiments/exp0035__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw150 \
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
  exp0036__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw150_fused_linear_e100_p4 \
  --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python \
  /workspace/scripts/04_train_frozen_multilabel_baseline.py \
  --embedding-root \
  /workspace/experiments/exp0035__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw150 \
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
  source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw150_fused_linear_e100_p4
```

## Preconditions

- The embedding experiment must already exist at `/workspace/experiments/exp0035__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw150`.
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
- val_metrics.json: `8.32K`
- test_metrics.json: `8.37K`
- val_f1_thresholds.json: `2.66K`
- train_log.jsonl: `29.7K`

## Final Metrics

- Validation macro AUROC: `0.772827`
- Validation macro average precision: `0.158282`
- Validation macro ECE: `0.329182`
- Validation macro F1 @ 0.5: `0.180575`
- Validation macro F1 @ tuned thresholds: `0.230538`
- Test macro AUROC: `0.774126`
- Test macro average precision: `0.159587`
- Test macro ECE: `0.329045`
- Test macro F1 @ 0.5: `0.178508`
- Test macro F1 @ tuned thresholds: `0.220287`
- Macro mean tuned threshold: `0.728043`

## Final Artifact SHA-256

```text
7894668e8e1dbc169869c89e5142f5f1399abba0c0c1ddbb816b57df3c13c2e6  /workspace/experiments/exp0036__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw150_fused_linear_e100_p4/config.json
193ba8f0076364ae733c012146593324113b0e2e6c88a47b87e694d71ce8f2f0  /workspace/experiments/exp0036__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw150_fused_linear_e100_p4/experiment_meta.json
8e600c77802a1bd2d960b2249b33f4add2ed7d9836fe975301af23cb43096cb2  /workspace/experiments/exp0036__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw150_fused_linear_e100_p4/best.ckpt
6a2815328f73e896f50bb2f68ee2a25d978e13bdeaca6b548eca3fd4d4aa655c  /workspace/experiments/exp0036__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw150_fused_linear_e100_p4/val_metrics.json
da640e5b41836347e5a1b19c7f3193cf0996032742a2c69347eb235c3b91175d  /workspace/experiments/exp0036__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw150_fused_linear_e100_p4/test_metrics.json
9f97ab511226cc4689f7ba69e7bef9772f96014ae26c1b1e656cb7deb0e6a296  /workspace/experiments/exp0036__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw150_fused_linear_e100_p4/val_f1_thresholds.json
3fbe58ea23f425cde1863b9c829dc96c5171f9049ea4bb524512655b74e3b8f5  /workspace/experiments/exp0036__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw150_fused_linear_e100_p4/train_log.jsonl
```

## Important Reproduction Notes

- `config.json`, `experiment_meta.json`, and `recreation_report.md` include timestamps, so their hashes change on rerun.
- Thresholds are tuned only on the validation split and then reused unchanged for test metrics.
- The model is a single linear layer on top of frozen embeddings; the input embedding experiment determines the feature space.
- The trainer auto-detects the split sidecar and aligns labels by sample ID rather than by trusting row position in the manifest.
- After a successful run, keep the generated `recreation_report.md`, commit the experiment directory, and push before starting the next experiment.

## Agent Handoff Text

```text
Use /workspace/scripts/04_train_frozen_multilabel_baseline.py and the report /workspace/experiments/exp0036__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw150_fused_linear_e100_p4/recreation_report.md to recreate the frozen NIH CXR14 source baseline for embeddings from /workspace/experiments/exp0035__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw150. Run the exact command in the report, verify the saved metrics and checkpoint hashes, and confirm that validation-tuned thresholds are reused for test evaluation.
```
