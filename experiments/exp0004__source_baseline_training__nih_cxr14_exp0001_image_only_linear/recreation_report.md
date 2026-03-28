# Source Baseline Recreation Report

## Scope

This report documents how to recreate the frozen-embedding multilabel baseline experiment stored at:

`/workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0001_image_only_linear`

The producing script is:

`/workspace/scripts/04_train_frozen_multilabel_baseline.py`

Script SHA-256:

`d4b2ac55787799801ab4423762a3bba01fc060ec17a54172ff70cf1cef69743a`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0001_image_only_linear`
- Experiment id: `exp0004`
- Operation label: `source_baseline_training`
- Input embedding root: `/workspace/experiments/exp0001__image_embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test`
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
  /workspace/experiments/exp0001__image_embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test \
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
  exp0004__source_baseline_training__nih_cxr14_exp0001_image_only_linear \
  --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python \
  /workspace/scripts/04_train_frozen_multilabel_baseline.py \
  --embedding-root \
  /workspace/experiments/exp0001__image_embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test \
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
  source_baseline_training__nih_cxr14_exp0001_image_only_linear
```

## Preconditions

- The embedding experiment must already exist at `/workspace/experiments/exp0001__image_embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The embedding experiment must contain `train/embeddings.npy`, `val/embeddings.npy`, and `test/embeddings.npy`.
- Each split must have a row-identity sidecar such as `image_paths.txt` or `report_ids.json`.
- The sidecar-derived row IDs must match `Path(image_path).stem` from the manifest for every split row.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Train rows: `78,571`
- Train embedding dim: `2048`
- Train sidecar: `image_paths.txt`
- Train ID parser: `stem`
- Val rows: `11,219`
- Val embedding dim: `2048`
- Val sidecar: `image_paths.txt`
- Val ID parser: `stem`
- Test rows: `22,330`
- Test embedding dim: `2048`
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

- config.json: `3.96K`
- experiment_meta.json: `3.68K`
- best.ckpt: `116K`
- val_metrics.json: `8.30K`
- test_metrics.json: `8.31K`
- val_f1_thresholds.json: `2.66K`
- train_log.jsonl: `8.87K`

## Final Metrics

- Validation macro AUROC: `0.745876`
- Validation macro average precision: `0.133964`
- Validation macro ECE: `0.358816`
- Validation macro F1 @ 0.5: `0.161989`
- Validation macro F1 @ tuned thresholds: `0.199989`
- Test macro AUROC: `0.746067`
- Test macro average precision: `0.130515`
- Test macro ECE: `0.358840`
- Test macro F1 @ 0.5: `0.160943`
- Test macro F1 @ tuned thresholds: `0.184568`
- Macro mean tuned threshold: `0.677410`

## Final Artifact SHA-256

```text
95d98d3a841ce92ad2210ec79a965ca165fc9c1df121cd547f29655a4c9c6688  /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0001_image_only_linear/config.json
c0386a372e2e8ce4b019392e4654ff4128b7de19a981c0789c06897912dfc630  /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0001_image_only_linear/experiment_meta.json
94309672d6c6efff701c0ee81b9f1df4c485550aebccdc42c78e03703eca23c2  /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0001_image_only_linear/best.ckpt
1a2a38ef395c149b4326dec795d6bc7e27f4b480856bc59ad745666679dfdf5f  /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0001_image_only_linear/val_metrics.json
fb78912781c423cce75fa50d0816e73e98b635e39ed26563bf54cdfce0206988  /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0001_image_only_linear/test_metrics.json
ec42cc6a7cd52dc5533298bd7c1649dcafc72d5db2d605b8514a4ccfde74b98d  /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0001_image_only_linear/val_f1_thresholds.json
55b201ff08ef28e83624ee5f0f036ea6443cfd33e6e26d3485b5ce04defc8f1d  /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0001_image_only_linear/train_log.jsonl
```

## Important Reproduction Notes

- `config.json`, `experiment_meta.json`, and `recreation_report.md` include timestamps, so their hashes change on rerun.
- Thresholds are tuned only on the validation split and then reused unchanged for test metrics.
- The model is a single linear layer on top of frozen embeddings; the input embedding experiment determines the feature space.
- The trainer auto-detects the split sidecar and aligns labels by sample ID rather than by trusting row position in the manifest.
- After a successful run, keep the generated `recreation_report.md`, commit the experiment directory, and push before starting the next experiment.

## Agent Handoff Text

```text
Use /workspace/scripts/04_train_frozen_multilabel_baseline.py and the report /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0001_image_only_linear/recreation_report.md to recreate the frozen NIH CXR14 source baseline for embeddings from /workspace/experiments/exp0001__image_embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test. Run the exact command in the report, verify the saved metrics and checkpoint hashes, and confirm that validation-tuned thresholds are reused for test evaluation.
```
