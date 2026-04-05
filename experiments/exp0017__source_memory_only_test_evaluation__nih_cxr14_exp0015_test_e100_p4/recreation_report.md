# Source Memory-Only Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test memory evaluation experiment stored at:

`/workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4`

The producing script is:

`/workspace/scripts/08_evaluate_source_memory_test.py`

Script SHA-256:

`a9a2c407354255e85343f693e5d9de3ff21c1e8f281c51b2ac0743d5f425b7c4`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4`
- Experiment id: `exp0017`
- Operation label: `source_memory_only_test_evaluation`
- Memory root: `/workspace/experiments/exp0014__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4`
- Validation selection root: `/workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4`
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
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0014__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4 --memory-selection-root /workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4 --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4 --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0014__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4 --memory-selection-root /workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4 --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4
```

## Preconditions

- The train memory must already exist at `/workspace/experiments/exp0014__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4`.
- The validation memory-selection experiment must already exist at `/workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4`.
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
- Threshold source: `/workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4/best_val_metrics.json`

## Test Metrics

- Test macro AUROC: `0.691239`
- Test macro average precision: `0.130995`
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

- experiment_meta.json: `38.92K`
- applied_config.json: `940B`
- test_metrics.json: `8.94K`
- test_probabilities.npy: `1.19M`
- qualitative_neighbors.json: `41.76K`
- memory_only_test_summary.md: `483B`
- Total output size: `1.28M`

## Final Artifact SHA-256

```text
4152435ada4d4f40e8403aa467d4c62e97faa48fb7a691fc3fbf84aa0c302252  /workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4/experiment_meta.json
6ea6f1e9dbe4cff81b66f708fda0e3cc59730040cedae3a9caf0986ad388e5ad  /workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4/applied_config.json
4db5bc08041fbd57d503ea3633ebc47bc898d63d2353127957ab0b501d249424  /workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4/test_metrics.json
447f9e583f893144038f27fba09a56d3e217ffdf84388c2a11da8c5c848d5d30  /workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4/test_probabilities.npy
0c7fe67166c5862bed5929da42b913fc5894b0d89e41f85f0856c06c51bde7b4  /workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4/qualitative_neighbors.json
28ccbfb6e302d70f2a907f7598920c47ede38a4fcc142d3876e3bd88199d3a60  /workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4/memory_only_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep `k` or `tau`; it reuses the validation-selected configuration from `exp0009`.
- Threshold-based F1 on test uses frozen thresholds from the validation artifact, not thresholds retuned on test.
- `test_probabilities.npy` stores held-out test probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/08_evaluate_source_memory_test.py and the report /workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4/recreation_report.md to recreate the held-out test memory evaluation for /workspace/experiments/exp0014__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4. Apply the frozen validation-selected k/tau from /workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4 on the test split, reuse the validation thresholds from exp0009 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_probabilities.npy artifacts.
```
