# Source Baseline Recreation Report

## Scope

This report documents how to recreate the frozen-embedding multilabel baseline experiment stored at:

`/workspace/experiments/exp0005__source_baseline_training__nih_cxr14_exp0002_report_only_linear`

The producing script is:

`/workspace/scripts/04_train_frozen_multilabel_baseline.py`

Script SHA-256:

`d4b2ac55787799801ab4423762a3bba01fc060ec17a54172ff70cf1cef69743a`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0005__source_baseline_training__nih_cxr14_exp0002_report_only_linear`
- Experiment id: `exp0005`
- Operation label: `source_baseline_training`
- Input embedding root: `/workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test`
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
  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test \
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
  exp0005__source_baseline_training__nih_cxr14_exp0002_report_only_linear \
  --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python \
  /workspace/scripts/04_train_frozen_multilabel_baseline.py \
  --embedding-root \
  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test \
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
  source_baseline_training__nih_cxr14_exp0002_report_only_linear
```

## Preconditions

- The embedding experiment must already exist at `/workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The embedding experiment must contain `train/embeddings.npy`, `val/embeddings.npy`, and `test/embeddings.npy`.
- Each split must have a row-identity sidecar such as `image_paths.txt` or `report_ids.json`.
- The sidecar-derived row IDs must match `Path(image_path).stem` from the manifest for every split row.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Train rows: `78,571`
- Train embedding dim: `128`
- Train sidecar: `report_ids.json`
- Train ID parser: `identity`
- Val rows: `11,219`
- Val embedding dim: `128`
- Val sidecar: `report_ids.json`
- Val ID parser: `identity`
- Test rows: `22,330`
- Test embedding dim: `128`
- Test sidecar: `report_ids.json`
- Test ID parser: `identity`

## Expected Outputs

- `config.json`
- `experiment_meta.json`
- `best.ckpt`
- `val_metrics.json`
- `test_metrics.json`
- `val_f1_thresholds.json`
- `train_log.jsonl`

## Output Sizes

- config.json: `4.11K`
- experiment_meta.json: `3.83K`
- best.ckpt: `11.2K`
- val_metrics.json: `8.34K`
- test_metrics.json: `8.35K`
- val_f1_thresholds.json: `2.66K`
- train_log.jsonl: `8.87K`

## Final Metrics

- Validation macro AUROC: `0.698718`
- Validation macro average precision: `0.123513`
- Validation macro ECE: `0.394832`
- Validation macro F1 @ 0.5: `0.159970`
- Validation macro F1 @ tuned thresholds: `0.191944`
- Test macro AUROC: `0.703594`
- Test macro average precision: `0.129127`
- Test macro ECE: `0.394382`
- Test macro F1 @ 0.5: `0.159969`
- Test macro F1 @ tuned thresholds: `0.191192`
- Macro mean tuned threshold: `0.682133`

## Final Artifact SHA-256

```text
d4555067bd7120bf344397f7d76388feba6c61f58f794389ce5fe26bbc7b2e34  /workspace/experiments/exp0005__source_baseline_training__nih_cxr14_exp0002_report_only_linear/config.json
3c696426874549d3dfd1cceb5ac6339a7d2145b6bd300668c71a35efb678ef6b  /workspace/experiments/exp0005__source_baseline_training__nih_cxr14_exp0002_report_only_linear/experiment_meta.json
721fbb2c5b94b04042cdd01f3f20587411b3ede7ba7f68a016d2c9944f9e859e  /workspace/experiments/exp0005__source_baseline_training__nih_cxr14_exp0002_report_only_linear/best.ckpt
e2a998ab7baa2b0a4c4cec06e3abd551eab938f635e55dfc4df897b67a9e451d  /workspace/experiments/exp0005__source_baseline_training__nih_cxr14_exp0002_report_only_linear/val_metrics.json
ba7c3f5f05d3bf5d749f45e409cd25fdf3cc007b27d4104a031c118571243ddd  /workspace/experiments/exp0005__source_baseline_training__nih_cxr14_exp0002_report_only_linear/test_metrics.json
cbc0eb2d62bf87d33defd0d9b4d35723ee0a2cb09d3fb88552da86955ac16b52  /workspace/experiments/exp0005__source_baseline_training__nih_cxr14_exp0002_report_only_linear/val_f1_thresholds.json
a4f9d3e0c9fbbee4db58664abc2d7a9f6ac741a0c9ec643c92703781444712d9  /workspace/experiments/exp0005__source_baseline_training__nih_cxr14_exp0002_report_only_linear/train_log.jsonl
```

## Important Reproduction Notes

- `config.json`, `experiment_meta.json`, and `recreation_report.md` include timestamps, so their hashes change on rerun.
- Thresholds are tuned only on the validation split and then reused unchanged for test metrics.
- The model is a single linear layer on top of frozen embeddings; the input embedding experiment determines the feature space.
- The trainer auto-detects the split sidecar and aligns labels by sample ID rather than by trusting row position in the manifest.
- After a successful run, keep the generated `recreation_report.md`, commit the experiment directory, and push before starting the next experiment.

## Agent Handoff Text

```text
Use /workspace/scripts/04_train_frozen_multilabel_baseline.py and the report /workspace/experiments/exp0005__source_baseline_training__nih_cxr14_exp0002_report_only_linear/recreation_report.md to recreate the frozen NIH CXR14 source baseline for embeddings from /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test. Run the exact command in the report, verify the saved metrics and checkpoint hashes, and confirm that validation-tuned thresholds are reused for test evaluation.
```
