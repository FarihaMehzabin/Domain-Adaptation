# Source Memory-Only Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test memory evaluation experiment stored at:

`/workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test`

The producing script is:

`/workspace/scripts/08_evaluate_source_memory_test.py`

Script SHA-256:

`a9a2c407354255e85343f693e5d9de3ff21c1e8f281c51b2ac0743d5f425b7c4`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test`
- Experiment id: `exp0011`
- Operation label: `source_memory_only_test_evaluation`
- Memory root: `/workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory`
- Validation selection root: `/workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val`
- Baseline reference experiment: `/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear`
- Query embedding root: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Evaluation split: `test`
- Test query rows: `22,330`
- Train memory rows: `78,571`
- Query embedding dimension: `2176`
- Memory embedding dimension: `2176`
- Label count: `14`
- Label names: `atelectasis cardiomegaly consolidation edema pleural_effusion emphysema fibrosis hernia infiltration mass nodule pleural_thickening pneumonia pneumothorax`
- Applied selection mode: `frozen_validation_config`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- faiss: `1.13.2`
- Platform: `Linux-5.15.0-102-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory --memory-selection-root /workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory --memory-selection-root /workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_test_evaluation__nih_cxr14_exp0009_test
```

## Preconditions

- The train memory must already exist at `/workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory`.
- The validation memory-selection experiment must already exist at `/workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val`.
- The query embeddings must already exist at `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `faiss`.
- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected configuration.

## Input Summary

- Query split directory: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/test`
- Query rows: `22,330`
- Query embedding dim: `2176`
- Memory rows: `78,571`
- Memory embedding dim: `2176`
- Index loaded from disk: `true`
- Index rebuilt from embeddings: `false`

## Applied Configuration

- Frozen `k`: `50`
- Frozen `tau`: `1`
- Threshold source: `/workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val/best_val_metrics.json`

## Test Metrics

- Test macro AUROC: `0.691239`
- Test macro average precision: `0.130994`
- Test macro ECE: `0.008359`
- Test macro F1 @ 0.5: `0.015123`
- Test macro F1 @ frozen val thresholds: `0.188889`

## Query Normalization

- Raw norm mean: `1.00000000`
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

- experiment_meta.json: `37.30K`
- applied_config.json: `925B`
- test_metrics.json: `8.94K`
- test_probabilities.npy: `1.19M`
- qualitative_neighbors.json: `41.76K`
- memory_only_test_summary.md: `475B`
- Total output size: `1.28M`

## Final Artifact SHA-256

```text
49b48c900bcc7dbb85afcecb0fd8d8b4454df82a6da3b4becb8e77ee65dca19f  /workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test/experiment_meta.json
7376d163dc4f70a9b8322b886de5973909919d99826e2fc25f0c6433463c154c  /workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test/applied_config.json
99ed1597cd5fb72503e67c7dff3ec0e001082a42508b996161b27a1d420f43d3  /workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test/test_metrics.json
870fc80dd7cd1d033620ed74e380d19b00da88a0743ba3249c82661eb511dc08  /workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test/test_probabilities.npy
254b1fa1ba0064897e69d33a1dfa3b52e16653e681e3f2d80090fef5c00effb5  /workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test/qualitative_neighbors.json
a9deded19ae86f062a32fa78a110ff2b03a2667f206dba903a883bc075143ef3  /workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test/memory_only_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep `k` or `tau`; it reuses the validation-selected configuration from `exp0009`.
- Threshold-based F1 on test uses frozen thresholds from the validation artifact, not thresholds retuned on test.
- `test_probabilities.npy` stores held-out test probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/08_evaluate_source_memory_test.py and the report /workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test/recreation_report.md to recreate the held-out test memory evaluation for /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory. Apply the frozen validation-selected k/tau from /workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val on the test split, reuse the validation thresholds from exp0009 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_probabilities.npy artifacts.
```
