# Source Memory-Only Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test memory evaluation experiment stored at:

`/workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test`

The producing script is:

`/workspace/scripts/08_evaluate_source_memory_test.py`

Script SHA-256:

`78f3458f65f8b685d96910b6e327816aeb8001a5b24e87f0cda3c5b23e953fa2`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test`
- Experiment id: `exp0032`
- Operation label: `source_memory_only_test_evaluation`
- Memory root: `/workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train`
- Validation selection root: `/workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val`
- Baseline reference experiment: `/workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096`
- Query embedding root: `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Evaluation split: `test`
- Test query rows: `22,330`
- Train memory rows: `78,569`
- Query embedding dimension: `2688`
- Memory embedding dimension: `2688`
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
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train --memory-selection-root /workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/08_evaluate_source_memory_test.py --memory-root /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train --memory-selection-root /workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test
```

## Preconditions

- The train memory must already exist at `/workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train`.
- The validation memory-selection experiment must already exist at `/workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val`.
- The query embeddings must already exist at `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat/test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `faiss`.
- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected configuration.

## Input Summary

- Query split directory: `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat/test`
- Query rows: `22,330`
- Query embedding dim: `2688`
- Memory rows: `78,569`
- Memory embedding dim: `2688`
- Index loaded from disk: `true`
- Index rebuilt from embeddings: `false`

## Applied Configuration

- Frozen `k`: `50`
- Frozen `tau`: `1`
- Threshold source: `/workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val/best_val_metrics.json`

## Test Metrics

- Test macro AUROC: `0.717359`
- Test macro average precision: `0.142505`
- Test macro ECE: `0.010017`
- Test macro F1 @ 0.5: `0.029407`
- Test macro F1 @ frozen val thresholds: `0.202983`

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

- experiment_meta.json: `40.59K`
- applied_config.json: `948B`
- test_metrics.json: `8.98K`
- test_probabilities.npy: `1.19M`
- qualitative_neighbors.json: `44.88K`
- memory_only_test_summary.md: `486B`
- Total output size: `1.29M`

## Final Artifact SHA-256

```text
b37e25f44d8c134a64ca03fc3957354dcf92e15f60a0dfd0db28db5a4e97ec5b  /workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test/experiment_meta.json
fc5f5b773428a93a9b6c53c24f59c2ac8f0c781530c793960563e7c22a1d822f  /workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test/applied_config.json
762504ce8d97d156d026c7439c647d7f235b7361efd91aae6d80e69eafe96c7e  /workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test/test_metrics.json
24ede4b273e3a75757d0b974830bcda0c88a8d0413da87bdd950f56b804abf36  /workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test/test_probabilities.npy
03188e03c506264b6e78514fb00fdbd4c2fb28907a5fee194b1633e317c80824  /workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test/qualitative_neighbors.json
18bed7950120dd25329a38319499cfd29e4ca3ea5127fd157d3273b8b4717691  /workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test/memory_only_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep `k` or `tau`; it reuses the validation-selected configuration from `exp0006`.
- Threshold-based F1 on test uses frozen thresholds from the validation artifact, not thresholds retuned on test.
- `test_probabilities.npy` stores held-out test probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/08_evaluate_source_memory_test.py and the report /workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test/recreation_report.md to recreate the held-out test memory evaluation for /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train. Apply the frozen validation-selected k/tau from /workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val on the test split, reuse the validation thresholds from exp0006 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_probabilities.npy artifacts.
```
