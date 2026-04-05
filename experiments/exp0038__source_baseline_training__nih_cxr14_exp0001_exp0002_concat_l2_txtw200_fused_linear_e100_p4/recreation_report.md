# Source Baseline Recreation Report

## Scope

This report documents how to recreate the frozen-embedding multilabel baseline experiment stored at:

`/workspace/experiments/exp0038__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw200_fused_linear_e100_p4`

The producing script is:

`/workspace/scripts/04_train_frozen_multilabel_baseline.py`

Script SHA-256:

`d4b2ac55787799801ab4423762a3bba01fc060ec17a54172ff70cf1cef69743a`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0038__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw200_fused_linear_e100_p4`
- Experiment id: `exp0038`
- Operation label: `source_baseline_training`
- Input embedding root: `/workspace/experiments/exp0037__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw200`
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
  /workspace/experiments/exp0037__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw200 \
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
  exp0038__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw200_fused_linear_e100_p4 \
  --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python \
  /workspace/scripts/04_train_frozen_multilabel_baseline.py \
  --embedding-root \
  /workspace/experiments/exp0037__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw200 \
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
  source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw200_fused_linear_e100_p4
```

## Preconditions

- The embedding experiment must already exist at `/workspace/experiments/exp0037__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw200`.
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
- val_metrics.json: `8.33K`
- test_metrics.json: `8.37K`
- val_f1_thresholds.json: `2.67K`
- train_log.jsonl: `29.7K`

## Final Metrics

- Validation macro AUROC: `0.771124`
- Validation macro average precision: `0.157103`
- Validation macro ECE: `0.332616`
- Validation macro F1 @ 0.5: `0.179993`
- Validation macro F1 @ tuned thresholds: `0.231269`
- Test macro AUROC: `0.773377`
- Test macro average precision: `0.158557`
- Test macro ECE: `0.332497`
- Test macro F1 @ 0.5: `0.178177`
- Test macro F1 @ tuned thresholds: `0.218521`
- Macro mean tuned threshold: `0.733558`

## Final Artifact SHA-256

```text
3011cefec0cd56e20b11c94964968c26ed7d09ae8108cf02a003a80103814474  /workspace/experiments/exp0038__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw200_fused_linear_e100_p4/config.json
4ff20c7accb19b9758ca50368710bb67f4aa31d237099b7b90b61595500473ad  /workspace/experiments/exp0038__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw200_fused_linear_e100_p4/experiment_meta.json
f106c0fca80bd14601cb586b3c331ab26d8974c7e3624ea96ebd6fcbb978b1a2  /workspace/experiments/exp0038__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw200_fused_linear_e100_p4/best.ckpt
37decbe876ea530ec861010a04401ed70aabf02fca70e84a5deb478b8d878e22  /workspace/experiments/exp0038__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw200_fused_linear_e100_p4/val_metrics.json
862f35212bdc4a858cd8faad1dfaaa1eaa268e4830a525023ede3e8dc1d7ec35  /workspace/experiments/exp0038__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw200_fused_linear_e100_p4/test_metrics.json
9dfb653a98c06023f2b97791efb8a057c4e4ef0dc46adaa675fe9779e1717acd  /workspace/experiments/exp0038__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw200_fused_linear_e100_p4/val_f1_thresholds.json
35532bb8e93e88e8517873012ad50033fdba17659d4573559cebffb2cf3187b0  /workspace/experiments/exp0038__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw200_fused_linear_e100_p4/train_log.jsonl
```

## Important Reproduction Notes

- `config.json`, `experiment_meta.json`, and `recreation_report.md` include timestamps, so their hashes change on rerun.
- Thresholds are tuned only on the validation split and then reused unchanged for test metrics.
- The model is a single linear layer on top of frozen embeddings; the input embedding experiment determines the feature space.
- The trainer auto-detects the split sidecar and aligns labels by sample ID rather than by trusting row position in the manifest.
- After a successful run, keep the generated `recreation_report.md`, commit the experiment directory, and push before starting the next experiment.

## Agent Handoff Text

```text
Use /workspace/scripts/04_train_frozen_multilabel_baseline.py and the report /workspace/experiments/exp0038__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw200_fused_linear_e100_p4/recreation_report.md to recreate the frozen NIH CXR14 source baseline for embeddings from /workspace/experiments/exp0037__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw200. Run the exact command in the report, verify the saved metrics and checkpoint hashes, and confirm that validation-tuned thresholds are reused for test evaluation.
```
