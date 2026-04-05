# Source Memory-Only Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test memory evaluation experiment stored at:

`/workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined`

The producing script is:

`/workspace/scripts/08_evaluate_source_memory_test.py`

Script SHA-256:

`a9a2c407354255e85343f693e5d9de3ff21c1e8f281c51b2ac0743d5f425b7c4`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined`
- Experiment id: `exp0023`
- Operation label: `source_memory_only_test_evaluation`
- Memory root: `/workspace/experiments/exp0019__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4_expanded_retrieval`
- Validation selection root: `/workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined`
- Baseline reference experiment: `/workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`
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
- Platform: `Linux-6.8.0-85-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0019__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4_expanded_retrieval --memory-selection-root /workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0019__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4_expanded_retrieval --memory-selection-root /workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined
```

## Preconditions

- The train memory must already exist at `/workspace/experiments/exp0019__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4_expanded_retrieval`.
- The validation memory-selection experiment must already exist at `/workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined`.
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

- Frozen `k`: `3000`
- Frozen `tau`: `40`
- Threshold source: `/workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined/best_val_metrics.json`

## Test Metrics

- Test macro AUROC: `0.745989`
- Test macro average precision: `0.138152`
- Test macro ECE: `0.005927`
- Test macro F1 @ 0.5: `0.000485`
- Test macro F1 @ frozen val thresholds: `0.191466`

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

- experiment_meta.json: `40.22K`
- applied_config.json: `1003B`
- test_metrics.json: `8.92K`
- test_probabilities.npy: `1.19M`
- qualitative_neighbors.json: `2.15M`
- memory_only_test_summary.md: `513B`
- Total output size: `3.40M`

## Final Artifact SHA-256

```text
56d2a59f05983fe9b51cf45d3af3a2779d697128825e168f3666472ef02a3a3a  /workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined/experiment_meta.json
7895d9c3ffb4094b331fd0fcb213af48b4f3266d3c7af924e62ff36aefb05aa0  /workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined/applied_config.json
7799ecc458c6b17526b8cc808069071f649bd1c5b7b22cdc0aeb2319588a69ae  /workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined/test_metrics.json
b677a1d7d252c16c5fabd1ae8e289bcfb2a01ef053b3e799bf1e05a7eea55b51  /workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined/test_probabilities.npy
1a20921966f14efd2815bf471645fc56985bc9b09cd8b4cef9f60f5c7e82fc0d  /workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined/qualitative_neighbors.json
9581d1cdfc16c587b23baa33f7020793d1c1ea23a5bebd4035fc4e89d49e1882  /workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined/memory_only_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep `k` or `tau`; it reuses the validation-selected configuration from `exp0009`.
- Threshold-based F1 on test uses frozen thresholds from the validation artifact, not thresholds retuned on test.
- `test_probabilities.npy` stores held-out test probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/08_evaluate_source_memory_test.py and the report /workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined/recreation_report.md to recreate the held-out test memory evaluation for /workspace/experiments/exp0019__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4_expanded_retrieval. Apply the frozen validation-selected k/tau from /workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined on the test split, reuse the validation thresholds from exp0009 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_probabilities.npy artifacts.
```
