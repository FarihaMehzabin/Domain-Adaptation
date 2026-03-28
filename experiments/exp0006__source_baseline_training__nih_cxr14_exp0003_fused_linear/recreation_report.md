# Source Baseline Recreation Report

## Scope

This report documents how to recreate the frozen-embedding multilabel baseline experiment stored at:

`/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear`

The producing script is:

`/workspace/scripts/04_train_frozen_multilabel_baseline.py`

Script SHA-256:

`d4b2ac55787799801ab4423762a3bba01fc060ec17a54172ff70cf1cef69743a`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear`
- Experiment id: `exp0006`
- Operation label: `source_baseline_training`
- Input embedding root: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Label count: `14`
- Selection metric: `macro_auroc`
- Batch size: `512`
- Epoch budget: `30`
- Patience: `5`
- Learning rate: `0.001`
- Weight decay: `0.0001`
- Seed: `1337`
- Device requested: `auto`
- Device resolved during run: `cuda`
- Mixed precision on CUDA: `true`
- Best epoch: `30`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- PyTorch: `2.4.1+cu124`
- CUDA available: `true`
- GPU used: `NVIDIA RTX A5000`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python \
  /workspace/scripts/04_train_frozen_multilabel_baseline.py \
  --embedding-root \
  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 \
  --batch-size \
  512 \
  --epochs \
  30 \
  --lr \
  1e-3 \
  --weight-decay \
  1e-4 \
  --patience \
  5 \
  --seed \
  1337 \
  --device \
  auto \
  --fp16-on-cuda \
  --experiment-name \
  exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear \
  --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python \
  /workspace/scripts/04_train_frozen_multilabel_baseline.py \
  --embedding-root \
  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 \
  --batch-size \
  512 \
  --epochs \
  30 \
  --lr \
  1e-3 \
  --weight-decay \
  1e-4 \
  --patience \
  5 \
  --seed \
  1337 \
  --device \
  auto \
  --fp16-on-cuda \
  --experiment-name \
  source_baseline_training__nih_cxr14_exp0003_fused_linear
```

## Preconditions

- The embedding experiment must already exist at `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`.
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

- config.json: `3.82K`
- experiment_meta.json: `3.52K`
- best.ckpt: `123K`
- val_metrics.json: `8.33K`
- test_metrics.json: `8.33K`
- val_f1_thresholds.json: `2.66K`
- train_log.jsonl: `8.87K`

## Final Metrics

- Validation macro AUROC: `0.763730`
- Validation macro average precision: `0.151467`
- Validation macro ECE: `0.347225`
- Validation macro F1 @ 0.5: `0.175942`
- Validation macro F1 @ tuned thresholds: `0.222298`
- Test macro AUROC: `0.767933`
- Test macro average precision: `0.152244`
- Test macro ECE: `0.347454`
- Test macro F1 @ 0.5: `0.175678`
- Test macro F1 @ tuned thresholds: `0.209533`
- Macro mean tuned threshold: `0.713494`

## Final Artifact SHA-256

```text
6cb34becc760a7baf39d88ec16a441bb084e744038fee3a4695605aa8868c1b8  /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear/config.json
5ec64578b235046897486e8b9a81652cf1544b9dbbacc4df378b780f92b77aca  /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear/experiment_meta.json
5fccc890e049dae6d3a833c20551d0bed7cc08f7b9c3756a665c253bc281bc1f  /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear/best.ckpt
637b9a36b0f5389af08a22f7d9d4660ac73ad3b57749e407219e40e14df9fd11  /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear/val_metrics.json
4624aae7abf0418d583d939be3a11aa77adb1c611c1ba045c397726af03bf06a  /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear/test_metrics.json
81cacefea26eced67b67bedc02b1223e5dffb02dc9c2c8e0c9d5bc1922b6b7b5  /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear/val_f1_thresholds.json
d79bed4fc48715ece7fc2753f9fbbd5053383d6cf0d8c8f4c7d35cbd691360a6  /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear/train_log.jsonl
```

## Important Reproduction Notes

- `config.json`, `experiment_meta.json`, and `recreation_report.md` include timestamps, so their hashes change on rerun.
- Thresholds are tuned only on the validation split and then reused unchanged for test metrics.
- The model is a single linear layer on top of frozen embeddings; the input embedding experiment determines the feature space.
- The trainer auto-detects the split sidecar and aligns labels by sample ID rather than by trusting row position in the manifest.
- After a successful run, keep the generated `recreation_report.md`, commit the experiment directory, and push before starting the next experiment.

## Agent Handoff Text

```text
Use /workspace/scripts/04_train_frozen_multilabel_baseline.py and the report /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear/recreation_report.md to recreate the frozen NIH CXR14 source baseline for embeddings from /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2. Run the exact command in the report, verify the saved metrics and checkpoint hashes, and confirm that validation-tuned thresholds are reused for test evaluation.
```
