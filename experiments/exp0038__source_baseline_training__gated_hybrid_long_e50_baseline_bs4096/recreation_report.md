# Source Baseline Recreation Report

## Scope

This report documents how to recreate the frozen-embedding multilabel baseline experiment stored at:

`/workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096`

The producing script is:

`/workspace/scripts/04_train_frozen_multilabel_baseline.py`

Script SHA-256:

`d4b2ac55787799801ab4423762a3bba01fc060ec17a54172ff70cf1cef69743a`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096`
- Experiment id: `exp0038`
- Operation label: `source_baseline_training`
- Input embedding root: `/workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160`
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
  /workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160 \
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
  exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096 \
  --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python \
  /workspace/scripts/04_train_frozen_multilabel_baseline.py \
  --embedding-root \
  /workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160 \
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
  source_baseline_training__gated_hybrid_long_e50_baseline_bs4096
```

## Preconditions

- The embedding experiment must already exist at `/workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160`.
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

- config.json: `3.94K`
- experiment_meta.json: `3.66K`
- best.ckpt: `32.2K`
- val_metrics.json: `8.34K`
- test_metrics.json: `8.34K`
- val_f1_thresholds.json: `2.65K`
- train_log.jsonl: `5.93K`

## Final Metrics

- Validation macro AUROC: `0.754816`
- Validation macro average precision: `0.140256`
- Validation macro ECE: `0.323025`
- Validation macro F1 @ 0.5: `0.169907`
- Validation macro F1 @ tuned thresholds: `0.210925`
- Test macro AUROC: `0.759012`
- Test macro average precision: `0.141954`
- Test macro ECE: `0.323712`
- Test macro F1 @ 0.5: `0.168837`
- Test macro F1 @ tuned thresholds: `0.203334`
- Macro mean tuned threshold: `0.735479`

## Final Artifact SHA-256

```text
2ccbb0baa3ab518d254b51282a4f02359b8bcbf31c07c5b9f880d60f2a88be12  /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096/config.json
806f356a507bb9d7ad2c4dfa3d235248c8c0cda8ae26da890ee55d1774fe4c01  /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096/experiment_meta.json
61c5888e40cd4c2e087ad7c46f27be6d7ec693b30d3dedc4dc6d7003fa65f4cf  /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096/best.ckpt
23a21626563b83df8cf38b84cc7a1c35baaff2da5ad32cb9ad3ce499183a7fc9  /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096/val_metrics.json
27e552645814aeb645f90204d7328d7f5a15a78983db4e6d9b6e7d8b6796c912  /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096/test_metrics.json
493fc118a2eb19fd537600261d989dbbb2b47f40ca9a1b25300ebb09ea322f5d  /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096/val_f1_thresholds.json
8322239ba0bbf515c3465ad0eb14a1d4d0bdd8b2cc4facd4bb42d8c3ff7705a1  /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096/train_log.jsonl
```

## Important Reproduction Notes

- `config.json`, `experiment_meta.json`, and `recreation_report.md` include timestamps, so their hashes change on rerun.
- Thresholds are tuned only on the validation split and then reused unchanged for test metrics.
- The model is a single linear layer on top of frozen embeddings; the input embedding experiment determines the feature space.
- The trainer auto-detects the split sidecar and aligns labels by sample ID rather than by trusting row position in the manifest.
- After a successful run, keep the generated `recreation_report.md`, commit the experiment directory, and push before starting the next experiment.

## Agent Handoff Text

```text
Use /workspace/scripts/04_train_frozen_multilabel_baseline.py and the report /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096/recreation_report.md to recreate the frozen NIH CXR14 source baseline for embeddings from /workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160. Run the exact command in the report, verify the saved metrics and checkpoint hashes, and confirm that validation-tuned thresholds are reused for test evaluation.
```
