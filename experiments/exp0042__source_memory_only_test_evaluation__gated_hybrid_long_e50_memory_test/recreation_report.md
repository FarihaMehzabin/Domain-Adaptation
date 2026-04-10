# Source Memory-Only Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test memory evaluation experiment stored at:

`/workspace/experiments/exp0042__source_memory_only_test_evaluation__gated_hybrid_long_e50_memory_test`

The producing script is:

`/workspace/scripts/08_evaluate_source_memory_test.py`

Script SHA-256:

`78f3458f65f8b685d96910b6e327816aeb8001a5b24e87f0cda3c5b23e953fa2`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0042__source_memory_only_test_evaluation__gated_hybrid_long_e50_memory_test`
- Experiment id: `exp0042`
- Operation label: `source_memory_only_test_evaluation`
- Memory root: `/workspace/experiments/exp0039__source_retrieval_memory_building__gated_hybrid_long_e50_memory_train`
- Validation selection root: `/workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val`
- Baseline reference experiment: `/workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096`
- Query embedding root: `/workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160`
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
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0039__source_retrieval_memory_building__gated_hybrid_long_e50_memory_train --memory-selection-root /workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val --query-embedding-root /workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160 --baseline-experiment-dir /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name exp0042__source_memory_only_test_evaluation__gated_hybrid_long_e50_memory_test --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0039__source_retrieval_memory_building__gated_hybrid_long_e50_memory_train --memory-selection-root /workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val --query-embedding-root /workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160 --baseline-experiment-dir /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_test_evaluation__gated_hybrid_long_e50_memory_test
```

## Preconditions

- The train memory must already exist at `/workspace/experiments/exp0039__source_retrieval_memory_building__gated_hybrid_long_e50_memory_train`.
- The validation memory-selection experiment must already exist at `/workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val`.
- The query embeddings must already exist at `/workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160/test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `faiss`.
- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected configuration.

## Input Summary

- Query split directory: `/workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160/test`
- Query rows: `22,330`
- Query embedding dim: `512`
- Memory rows: `78,569`
- Memory embedding dim: `512`
- Index loaded from disk: `true`
- Index rebuilt from embeddings: `false`

## Applied Configuration

- Frozen `k`: `50`
- Frozen `tau`: `1`
- Threshold source: `/workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val/best_val_metrics.json`

## Test Metrics

- Test macro AUROC: `0.709685`
- Test macro average precision: `0.132874`
- Test macro ECE: `0.011057`
- Test macro F1 @ 0.5: `0.023517`
- Test macro F1 @ frozen val thresholds: `0.190231`

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

- experiment_meta.json: `34.86K`
- applied_config.json: `949B`
- test_metrics.json: `8.99K`
- test_probabilities.npy: `1.19M`
- qualitative_neighbors.json: `44.39K`
- memory_only_test_summary.md: `486B`
- Total output size: `1.28M`

## Final Artifact SHA-256

```text
f18328add9ff9ec0742c68ff103c60fdd961a23530dbd9bd28df582fc7c6268d  /workspace/experiments/exp0042__source_memory_only_test_evaluation__gated_hybrid_long_e50_memory_test/experiment_meta.json
a35194a6889b90b0cbdc1fbec11999a3df55652feff2b1685bd62628ff277e26  /workspace/experiments/exp0042__source_memory_only_test_evaluation__gated_hybrid_long_e50_memory_test/applied_config.json
8c824102c72c0ab7df5f58f44a1ceb52ede9bbbfa094937fed448bf24930f333  /workspace/experiments/exp0042__source_memory_only_test_evaluation__gated_hybrid_long_e50_memory_test/test_metrics.json
9737b9b8705b23fe5104e32eb89b520794aa0864cb24765454f94b852da0c1a0  /workspace/experiments/exp0042__source_memory_only_test_evaluation__gated_hybrid_long_e50_memory_test/test_probabilities.npy
c83ad47592d0a32c95e4d4bffbf562633febc409b7347c9fd2af8d0e27e32692  /workspace/experiments/exp0042__source_memory_only_test_evaluation__gated_hybrid_long_e50_memory_test/qualitative_neighbors.json
fc514db6a1232d617a3c4f4ccf112f4d172a9cd222cfc8374e82a6e21645c791  /workspace/experiments/exp0042__source_memory_only_test_evaluation__gated_hybrid_long_e50_memory_test/memory_only_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep `k` or `tau`; it reuses the validation-selected configuration from `exp0006`.
- Threshold-based F1 on test uses frozen thresholds from the validation artifact, not thresholds retuned on test.
- `test_probabilities.npy` stores held-out test probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/08_evaluate_source_memory_test.py and the report /workspace/experiments/exp0042__source_memory_only_test_evaluation__gated_hybrid_long_e50_memory_test/recreation_report.md to recreate the held-out test memory evaluation for /workspace/experiments/exp0039__source_retrieval_memory_building__gated_hybrid_long_e50_memory_train. Apply the frozen validation-selected k/tau from /workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val on the test split, reuse the validation thresholds from exp0006 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_probabilities.npy artifacts.
```
