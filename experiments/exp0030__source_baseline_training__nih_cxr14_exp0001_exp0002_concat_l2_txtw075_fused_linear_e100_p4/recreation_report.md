# Source Baseline Recreation Report

## Scope

This report documents how to recreate the frozen-embedding multilabel baseline experiment stored at:

`/workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4`

The producing script is:

`/workspace/scripts/04_train_frozen_multilabel_baseline.py`

Script SHA-256:

`d4b2ac55787799801ab4423762a3bba01fc060ec17a54172ff70cf1cef69743a`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4`
- Experiment id: `exp0030`
- Operation label: `source_baseline_training`
- Input embedding root: `/workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075`
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
  /workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075 \
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
  exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4 \
  --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python \
  /workspace/scripts/04_train_frozen_multilabel_baseline.py \
  --embedding-root \
  /workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075 \
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
  source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4
```

## Preconditions

- The embedding experiment must already exist at `/workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075`.
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
- val_f1_thresholds.json: `2.67K`
- train_log.jsonl: `29.6K`

## Final Metrics

- Validation macro AUROC: `0.775114`
- Validation macro average precision: `0.159902`
- Validation macro ECE: `0.323537`
- Validation macro F1 @ 0.5: `0.181358`
- Validation macro F1 @ tuned thresholds: `0.230487`
- Test macro AUROC: `0.774719`
- Test macro average precision: `0.160710`
- Test macro ECE: `0.323413`
- Test macro F1 @ 0.5: `0.179682`
- Test macro F1 @ tuned thresholds: `0.221203`
- Macro mean tuned threshold: `0.729959`

## Final Artifact SHA-256

```text
91eab4377fdbf5ecb08eed8a7fd6a6a6be4e9225345c0d56e6ddc0c15301f28e  /workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4/config.json
5c42fbea31e592b456b1a381097280753a4ee79e594be5c105a0260f09163058  /workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4/experiment_meta.json
402a4e4462ca4debb9ae93b2604429bd301a978388f21d29c09e1fe9e4f4142e  /workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4/best.ckpt
88d2f226e53b44191c1153992eab2910fc8dc93c58c428b67e3033427b0acf27  /workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4/val_metrics.json
6168915d9c8c5b6a6b722a34be3e77c2b4a14bc55cac33378ac22c7257e3ed57  /workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4/test_metrics.json
09db9daaf9aae335400b989824c8de2acebb0e565a78ab6c80d83772eafa51a1  /workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4/val_f1_thresholds.json
d35904fb7b499d4d3d6bb0d6b3bac096f1ef51c5ced095c05c37b583ad30d747  /workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4/train_log.jsonl
```

## Important Reproduction Notes

- `config.json`, `experiment_meta.json`, and `recreation_report.md` include timestamps, so their hashes change on rerun.
- Thresholds are tuned only on the validation split and then reused unchanged for test metrics.
- The model is a single linear layer on top of frozen embeddings; the input embedding experiment determines the feature space.
- The trainer auto-detects the split sidecar and aligns labels by sample ID rather than by trusting row position in the manifest.
- After a successful run, keep the generated `recreation_report.md`, commit the experiment directory, and push before starting the next experiment.

## Agent Handoff Text

```text
Use /workspace/scripts/04_train_frozen_multilabel_baseline.py and the report /workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4/recreation_report.md to recreate the frozen NIH CXR14 source baseline for embeddings from /workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075. Run the exact command in the report, verify the saved metrics and checkpoint hashes, and confirm that validation-tuned thresholds are reused for test evaluation.
```
