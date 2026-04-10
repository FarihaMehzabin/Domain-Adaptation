# Source Baseline Recreation Report

## Scope

This report documents how to recreate the frozen-embedding multilabel baseline experiment stored at:

`/workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`

The producing script is:

`/workspace/scripts/04_train_frozen_multilabel_baseline.py`

Script SHA-256:

`d4b2ac55787799801ab4423762a3bba01fc060ec17a54172ff70cf1cef69743a`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`
- Experiment id: `exp0004`
- Operation label: `source_baseline_training`
- Input embedding root: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`
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
  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 \
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
  exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 \
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
  source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4
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

- config.json: `3.88K`
- experiment_meta.json: `3.60K`
- best.ckpt: `123K`
- val_metrics.json: `8.33K`
- test_metrics.json: `8.37K`
- val_f1_thresholds.json: `2.67K`
- train_log.jsonl: `29.6K`

## Final Metrics

- Validation macro AUROC: `0.774447`
- Validation macro average precision: `0.159506`
- Validation macro ECE: `0.325348`
- Validation macro F1 @ 0.5: `0.181341`
- Validation macro F1 @ tuned thresholds: `0.231752`
- Test macro AUROC: `0.774642`
- Test macro average precision: `0.160414`
- Test macro ECE: `0.325218`
- Test macro F1 @ 0.5: `0.179142`
- Test macro F1 @ tuned thresholds: `0.221102`
- Macro mean tuned threshold: `0.729660`

## Final Artifact SHA-256

```text
9f170dde0346f29e81866a90c78040a5c030c3079356c2f7aef5158e25abd453  /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4/config.json
a170f7789ab1df65abf3a2b0940ad3c5d1b05b42d9ffd62950528b47142dbb33  /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4/experiment_meta.json
9d7cdc6ed1a7ad40c901fa0fb2f8028d8d5bd96752336f91242908234bb44506  /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4/best.ckpt
25e63bf2c48df905ad4d7b4637cf085a80060365f2121055d1996a69d3054d77  /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4/val_metrics.json
eac986127507d59c2558079f395300c3105072603000f5b1cfeb0a6b775c3bab  /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4/test_metrics.json
5a7bead9b7de8a2539cfa451e82e142ba7698cee9941fdf1d233dc0049d148be  /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4/val_f1_thresholds.json
e6fbd5c7f31a57203f91a545d3838a0d8a98c3f6a8aebe5dac6b2325d510a4d7  /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4/train_log.jsonl
```

## Important Reproduction Notes

- `config.json`, `experiment_meta.json`, and `recreation_report.md` include timestamps, so their hashes change on rerun.
- Thresholds are tuned only on the validation split and then reused unchanged for test metrics.
- The model is a single linear layer on top of frozen embeddings; the input embedding experiment determines the feature space.
- The trainer auto-detects the split sidecar and aligns labels by sample ID rather than by trusting row position in the manifest.
- After a successful run, keep the generated `recreation_report.md`, commit the experiment directory, and push before starting the next experiment.

## Agent Handoff Text

```text
Use /workspace/scripts/04_train_frozen_multilabel_baseline.py and the report /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4/recreation_report.md to recreate the frozen NIH CXR14 source baseline for embeddings from /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2. Run the exact command in the report, verify the saved metrics and checkpoint hashes, and confirm that validation-tuned thresholds are reused for test evaluation.
```
