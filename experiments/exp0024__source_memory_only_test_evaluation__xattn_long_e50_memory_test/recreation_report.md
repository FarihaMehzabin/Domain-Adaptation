# Source Memory-Only Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test memory evaluation experiment stored at:

`/workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test`

The producing script is:

`/workspace/scripts/08_evaluate_source_memory_test.py`

Script SHA-256:

`78f3458f65f8b685d96910b6e327816aeb8001a5b24e87f0cda3c5b23e953fa2`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test`
- Experiment id: `exp0024`
- Operation label: `source_memory_only_test_evaluation`
- Memory root: `/workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train`
- Validation selection root: `/workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val`
- Baseline reference experiment: `/workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096`
- Query embedding root: `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160`
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
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train --memory-selection-root /workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val --query-embedding-root /workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160 --baseline-experiment-dir /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train --memory-selection-root /workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val --query-embedding-root /workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160 --baseline-experiment-dir /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_test_evaluation__xattn_long_e50_memory_test
```

## Preconditions

- The train memory must already exist at `/workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train`.
- The validation memory-selection experiment must already exist at `/workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val`.
- The query embeddings must already exist at `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160/test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `faiss`.
- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected configuration.

## Input Summary

- Query split directory: `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160/test`
- Query rows: `22,330`
- Query embedding dim: `512`
- Memory rows: `78,569`
- Memory embedding dim: `512`
- Index loaded from disk: `true`
- Index rebuilt from embeddings: `false`

## Applied Configuration

- Frozen `k`: `50`
- Frozen `tau`: `1`
- Threshold source: `/workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val/best_val_metrics.json`

## Test Metrics

- Test macro AUROC: `0.700407`
- Test macro average precision: `0.127088`
- Test macro ECE: `0.011227`
- Test macro F1 @ 0.5: `0.021696`
- Test macro F1 @ frozen val thresholds: `0.189811`

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

- experiment_meta.json: `33.86K`
- applied_config.json: `935B`
- test_metrics.json: `8.96K`
- test_probabilities.npy: `1.19M`
- qualitative_neighbors.json: `45.39K`
- memory_only_test_summary.md: `479B`
- Total output size: `1.28M`

## Final Artifact SHA-256

```text
1536b22805c1d43d0e70e97e9807e0582544678d416c1aada6157235ccd44686  /workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test/experiment_meta.json
170452ca0e2e11d1b183238f8f450a625ee1412b5a7286714e04fd34971e2dd1  /workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test/applied_config.json
7b6f3d566cbe1d73f58dd6affe9e4156e0b388ff36a02958810918be0ddd52e5  /workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test/test_metrics.json
b52f506539cd405f82d3935d16a954ad7c7dcfaf36d56c36cb1ad5c08cc24d13  /workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test/test_probabilities.npy
07a93d241cd985076335448bb369a02c3b0594d864a5acb7dcd780d7ff06bc13  /workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test/qualitative_neighbors.json
cdb34b5c0f789e89b783407c0843ceb7bdca5f3b04d26558a79ad8b41ecb8d39  /workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test/memory_only_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep `k` or `tau`; it reuses the validation-selected configuration from `exp0006`.
- Threshold-based F1 on test uses frozen thresholds from the validation artifact, not thresholds retuned on test.
- `test_probabilities.npy` stores held-out test probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/08_evaluate_source_memory_test.py and the report /workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test/recreation_report.md to recreate the held-out test memory evaluation for /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train. Apply the frozen validation-selected k/tau from /workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val on the test split, reuse the validation thresholds from exp0006 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_probabilities.npy artifacts.
```
