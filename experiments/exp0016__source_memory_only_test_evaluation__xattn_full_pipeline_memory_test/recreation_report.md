# Source Memory-Only Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test memory evaluation experiment stored at:

`/workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test`

The producing script is:

`/workspace/scripts/08_evaluate_source_memory_test.py`

Script SHA-256:

`78f3458f65f8b685d96910b6e327816aeb8001a5b24e87f0cda3c5b23e953fa2`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test`
- Experiment id: `exp0016`
- Operation label: `source_memory_only_test_evaluation`
- Memory root: `/workspace/experiments/exp0013__source_retrieval_memory_building__xattn_full_pipeline_memory_train`
- Validation selection root: `/workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val`
- Baseline reference experiment: `/workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096`
- Query embedding root: `/workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Evaluation split: `test`
- Test query rows: `22,330`
- Train memory rows: `78,569`
- Query embedding dimension: `512`
- Memory embedding dimension: `512`
- Label count: `14`
- Label names: `atelectasis cardiomegaly consolidation edema pleural_effusion emphysema fibrosis hernia infiltration mass nodule pleural_thickening pneumonia pneumothorax`
- Applied selection mode: `frozen_validation_config`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- faiss: `1.13.2`
- Platform: `Linux-6.8.0-65-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0013__source_retrieval_memory_building__xattn_full_pipeline_memory_train --memory-selection-root /workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val --query-embedding-root /workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160 --baseline-experiment-dir /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0013__source_retrieval_memory_building__xattn_full_pipeline_memory_train --memory-selection-root /workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val --query-embedding-root /workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160 --baseline-experiment-dir /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_test_evaluation__xattn_full_pipeline_memory_test
```

## Preconditions

- The train memory must already exist at `/workspace/experiments/exp0013__source_retrieval_memory_building__xattn_full_pipeline_memory_train`.
- The validation memory-selection experiment must already exist at `/workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val`.
- The query embeddings must already exist at `/workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160/test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `faiss`.
- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected configuration.

## Input Summary

- Query split directory: `/workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160/test`
- Query rows: `22,330`
- Query embedding dim: `512`
- Memory rows: `78,569`
- Memory embedding dim: `512`
- Index loaded from disk: `true`
- Index rebuilt from embeddings: `false`

## Applied Configuration

- Frozen `k`: `50`
- Frozen `tau`: `20`
- Threshold source: `/workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val/best_val_metrics.json`

## Test Metrics

- Test macro AUROC: `0.682572`
- Test macro average precision: `0.107575`
- Test macro ECE: `0.010428`
- Test macro F1 @ 0.5: `0.003845`
- Test macro F1 @ frozen val thresholds: `0.162419`

## Query Normalization

- Raw norm mean: `1.00000001`
- Post-normalization norm mean: `1.00000001`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `applied_config.json`
- `test_metrics.json`
- `test_probabilities.npy`
- `qualitative_neighbors.json`
- `memory_only_test_summary.md`

## Output Sizes

- experiment_meta.json: `34.11K`
- applied_config.json: `947B`
- test_metrics.json: `8.93K`
- test_probabilities.npy: `1.19M`
- qualitative_neighbors.json: `42.35K`
- memory_only_test_summary.md: `485B`
- Total output size: `1.28M`

## Final Artifact SHA-256

```text
8d4071d648b094888d9b5e17c5cb9da853f9f7219f632b131919207527487fa2  /workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test/experiment_meta.json
55e40da8936a3b23f80f8b3c09ecaf6481e5ee8e8bbd35a45805540c24176e7e  /workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test/applied_config.json
c9b935f5687ea6dbd60d7ab9cb19e7ea730bb35ea349f314ea1d15f66bfaddac  /workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test/test_metrics.json
07296ae1c94d14fe492132d93c276146dd2b5988f7517bf28565d8bb6ee5931a  /workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test/test_probabilities.npy
822e94843e966e36ee93b5d9ef6c800e9e392031ba81cde289839e5e3c271863  /workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test/qualitative_neighbors.json
6c68023990827ee390e33ae4d14c8f5476abb790298de2b64dd1e0ca3d0a9945  /workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test/memory_only_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep `k` or `tau`; it reuses the validation-selected configuration from `exp0006`.
- Threshold-based F1 on test uses frozen thresholds from the validation artifact, not thresholds retuned on test.
- `test_probabilities.npy` stores held-out test probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/08_evaluate_source_memory_test.py and the report /workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test/recreation_report.md to recreate the held-out test memory evaluation for /workspace/experiments/exp0013__source_retrieval_memory_building__xattn_full_pipeline_memory_train. Apply the frozen validation-selected k/tau from /workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val on the test split, reuse the validation thresholds from exp0006 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_probabilities.npy artifacts.
```
