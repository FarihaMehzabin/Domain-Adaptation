# Source Memory-Only Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test memory evaluation experiment stored at:

`/workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4`

The producing script is:

`/workspace/scripts/08_evaluate_source_memory_test.py`

Script SHA-256:

`a9a2c407354255e85343f693e5d9de3ff21c1e8f281c51b2ac0743d5f425b7c4`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4`
- Experiment id: `exp0045`
- Operation label: `source_memory_only_test_evaluation`
- Memory root: `/workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4`
- Validation selection root: `/workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4`
- Baseline reference experiment: `/workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4`
- Query embedding root: `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050`
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
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4 --memory-selection-root /workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4 --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050 --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4 --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4 --memory-selection-root /workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4 --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050 --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4
```

## Preconditions

- The train memory must already exist at `/workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4`.
- The validation memory-selection experiment must already exist at `/workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4`.
- The query embeddings must already exist at `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050/test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `faiss`.
- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected configuration.

## Input Summary

- Query split directory: `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050/test`
- Query rows: `22,330`
- Query embedding dim: `2176`
- Memory rows: `78,571`
- Memory embedding dim: `2176`
- Index loaded from disk: `true`
- Index rebuilt from embeddings: `false`

## Applied Configuration

- Frozen `k`: `50`
- Frozen `tau`: `1`
- Threshold source: `/workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4/best_val_metrics.json`

## Test Metrics

- Test macro AUROC: `0.682750`
- Test macro average precision: `0.124807`
- Test macro ECE: `0.008228`
- Test macro F1 @ 0.5: `0.008520`
- Test macro F1 @ frozen val thresholds: `0.178964`

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

- experiment_meta.json: `39.77K`
- applied_config.json: `943B`
- test_metrics.json: `8.96K`
- test_probabilities.npy: `1.19M`
- qualitative_neighbors.json: `41.24K`
- memory_only_test_summary.md: `483B`
- Total output size: `1.28M`

## Final Artifact SHA-256

```text
24238995d4e7f9b04070ced5ef4e3b5d7a3c70a8e7c70fbd7fa490fcd5d73767  /workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4/experiment_meta.json
dc7c767ed6bdf5c8592c284f11b6ab46fdc8ce7ce61cdd716be3502753fa03eb  /workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4/applied_config.json
f8a4aa48693d492b9794ff33b6c8ebd591e6aadd46cfc681b0ce7cd564cb3075  /workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4/test_metrics.json
edecdc3d4a69b5afcedbc142619665c4428880dd2f955def650383e63bd9730e  /workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4/test_probabilities.npy
4b3a7241e27c31f16d48adafbf952d6c1ee783f13ffbeef52993526fbbea4a52  /workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4/qualitative_neighbors.json
0166af7f662980ab2cc396439be708bc340a3ff9f802a86cb03ccd04d7db0fa0  /workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4/memory_only_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep `k` or `tau`; it reuses the validation-selected configuration from `exp0009`.
- Threshold-based F1 on test uses frozen thresholds from the validation artifact, not thresholds retuned on test.
- `test_probabilities.npy` stores held-out test probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/08_evaluate_source_memory_test.py and the report /workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4/recreation_report.md to recreate the held-out test memory evaluation for /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4. Apply the frozen validation-selected k/tau from /workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4 on the test split, reuse the validation thresholds from exp0009 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_probabilities.npy artifacts.
```
