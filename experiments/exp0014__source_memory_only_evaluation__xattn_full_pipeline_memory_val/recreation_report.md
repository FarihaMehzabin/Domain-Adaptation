# Source Memory-Only Evaluation Recreation Report

## Scope

This report documents how to recreate the validation-only memory evaluation experiment stored at:

`/workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val`

The producing script is:

`/workspace/scripts/06_evaluate_source_memory_only.py`

Script SHA-256:

`f84821de52732b0e7a65fbca1f163cbbca7dd08098b3b2f631bc9f4b18d6989f`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val`
- Experiment id: `exp0014`
- Operation label: `source_memory_only_evaluation`
- Memory root: `/workspace/experiments/exp0013__source_retrieval_memory_building__xattn_full_pipeline_memory_train`
- Query embedding root: `/workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160`
- Baseline reference experiment: `/workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Evaluation split: `val`
- Validation query rows: `11,219`
- Train memory rows: `78,569`
- Query embedding dimension: `512`
- Memory embedding dimension: `512`
- Label count: `14`
- Label names: `atelectasis cardiomegaly consolidation edema pleural_effusion emphysema fibrosis hernia infiltration mass nodule pleural_thickening pneumonia pneumothorax`
- Selection metric: `macro_auroc`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- faiss: `1.13.2`
- Platform: `Linux-6.8.0-65-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0013__source_retrieval_memory_building__xattn_full_pipeline_memory_train --query-embedding-root /workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160 --baseline-experiment-dir /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 1 3 5 10 20 50 --sweep-tau-values 1 5 10 20 40 --qualitative-queries 10 --seed 3407 --experiment-name exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0013__source_retrieval_memory_building__xattn_full_pipeline_memory_train --query-embedding-root /workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160 --baseline-experiment-dir /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 1 3 5 10 20 50 --sweep-tau-values 1 5 10 20 40 --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_evaluation__xattn_full_pipeline_memory_val
```

## Preconditions

- The memory experiment must already exist at `/workspace/experiments/exp0013__source_retrieval_memory_building__xattn_full_pipeline_memory_train`.
- The query embeddings must already exist at `/workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `faiss`.
- If the `exp0005` local FAISS index is missing, this script can rebuild it from the local train embeddings.

## Input Summary

- Query split directory: `/workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160/val`
- Query rows: `11,219`
- Query embedding dim: `512`
- Memory rows: `78,569`
- Memory embedding dim: `512`
- Index loaded from disk: `true`
- Index rebuilt from embeddings: `false`
- Sweep k values: `1 3 5 10 20 50`
- Sweep tau values: `1 5 10 20 40`

## Sweep Summary

| k | tau | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.520841 | 0.058770 | 0.088001 | 0.089854 |  |
| 1 | 5 | 0.520841 | 0.058770 | 0.088001 | 0.089854 |  |
| 1 | 10 | 0.520841 | 0.058770 | 0.088001 | 0.089854 |  |
| 1 | 20 | 0.520841 | 0.058770 | 0.088001 | 0.089854 |  |
| 1 | 40 | 0.520841 | 0.058770 | 0.088001 | 0.089854 |  |
| 3 | 1 | 0.546120 | 0.075256 | 0.068062 | 0.070459 |  |
| 3 | 5 | 0.546121 | 0.075257 | 0.068062 | 0.070459 |  |
| 3 | 10 | 0.546122 | 0.075259 | 0.068061 | 0.070459 |  |
| 3 | 20 | 0.546123 | 0.075261 | 0.068069 | 0.070435 |  |
| 3 | 40 | 0.546122 | 0.075258 | 0.068063 | 0.070403 |  |
| 5 | 1 | 0.565661 | 0.080759 | 0.054383 | 0.046453 |  |
| 5 | 5 | 0.565657 | 0.080756 | 0.054383 | 0.046453 |  |
| 5 | 10 | 0.565659 | 0.080758 | 0.054383 | 0.046453 |  |
| 5 | 20 | 0.565668 | 0.080778 | 0.054394 | 0.046440 |  |
| 5 | 40 | 0.565671 | 0.080727 | 0.054477 | 0.046312 |  |
| 10 | 1 | 0.600852 | 0.089798 | 0.038083 | 0.027542 |  |
| 10 | 5 | 0.600846 | 0.089798 | 0.038089 | 0.027463 |  |
| 10 | 10 | 0.600835 | 0.089800 | 0.038105 | 0.027463 |  |
| 10 | 20 | 0.600807 | 0.089816 | 0.038105 | 0.027463 |  |
| 10 | 40 | 0.600784 | 0.089465 | 0.038208 | 0.027899 |  |
| 20 | 1 | 0.637325 | 0.098756 | 0.020740 | 0.015410 |  |
| 20 | 5 | 0.637317 | 0.098750 | 0.020742 | 0.015319 |  |
| 20 | 10 | 0.637313 | 0.098766 | 0.020739 | 0.015323 |  |
| 20 | 20 | 0.637273 | 0.098756 | 0.020777 | 0.015444 |  |
| 20 | 40 | 0.637275 | 0.098315 | 0.021009 | 0.015877 |  |
| 50 | 1 | 0.679404 | 0.111163 | 0.010383 | 0.004426 |  |
| 50 | 5 | 0.679393 | 0.111156 | 0.010371 | 0.004426 |  |
| 50 | 10 | 0.679377 | 0.111047 | 0.010390 | 0.004428 |  |
| 50 | 20 | 0.679425 | 0.111083 | 0.010496 | 0.004427 | <- best |
| 50 | 40 | 0.678972 | 0.110851 | 0.010612 | 0.004322 |  |

## Best Configuration

- Best `k`: `50`
- Best `tau`: `20`
- Validation macro AUROC: `0.679425`
- Validation macro average precision: `0.111083`
- Validation macro ECE: `0.010496`
- Validation macro F1 @ 0.5: `0.004427`
- Diagnostic macro F1 @ tuned thresholds: `0.178353`

## Query Normalization

- Raw norm mean: `1.00000001`
- Post-normalization norm mean: `1.00000001`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `sweep_results.json`
- `best_config.json`
- `best_val_metrics.json`
- `val_probabilities.npy`
- `qualitative_neighbors.json`
- `memory_only_selection.md`

## Output Sizes

- experiment_meta.json: `15.92K`
- sweep_results.json: `219.82K`
- best_config.json: `641B`
- best_val_metrics.json: `6.25K`
- val_probabilities.npy: `613.66K`
- qualitative_neighbors.json: `39.64K`
- memory_only_selection.md: `300B`
- Total output size: `896.21K`

## Final Artifact SHA-256

```text
70f1e0a42c26e6f369dbf32022928ede96e3eb3243f2dfcaa521016ff1cdf096  /workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val/experiment_meta.json
c0c626a3257c74fcd6952e51622c218dece6a11f8d240a9464a3460120876a86  /workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val/sweep_results.json
45c688ac97b9370436bced7632f1529f7dd23c4ec3bc0c1926a0b253e164c7d7  /workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val/best_config.json
1affeb826a07ec6078d9280d149cd6d8ef1430bceaa9b19975c1111152f7facf  /workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val/best_val_metrics.json
fe0f2d919b6b3863906672a927a4f4a62fc55e08fe6ac59a7afba24846f92206  /workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val/val_probabilities.npy
648494239ba0a6e326d8f879e6b0873318ece27832d2020e408b44bf5d4e0937  /workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val/qualitative_neighbors.json
20d4b83bb52f21ba7e607107a852bd26878b1fe7ccb98d8c5404965f5e7f5371  /workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val/memory_only_selection.md
```

## Important Reproduction Notes

- All configuration selection in `exp0006` is validation-only.
- `val_probabilities.npy` contains the best-config validation probabilities and is small enough for plain Git.
- If the local `exp0005/index.faiss` is unavailable, rerun `exp0005` or keep the local train embeddings available so the index can be rebuilt.
- Threshold-tuned F1 is recorded as diagnostic only because thresholds are chosen on the same validation split.

## Agent Handoff Text

```text
Use /workspace/scripts/06_evaluate_source_memory_only.py and the report /workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val/recreation_report.md to recreate the validation-only memory evaluation for /workspace/experiments/exp0013__source_retrieval_memory_building__xattn_full_pipeline_memory_train. Sweep k and tau on the val split, select the best config by macro AUROC, and verify the saved best_config.json, best_val_metrics.json, and val_probabilities.npy artifacts.
```
