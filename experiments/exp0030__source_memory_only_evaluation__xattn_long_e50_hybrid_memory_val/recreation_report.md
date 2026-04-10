# Source Memory-Only Evaluation Recreation Report

## Scope

This report documents how to recreate the validation-only memory evaluation experiment stored at:

`/workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val`

The producing script is:

`/workspace/scripts/06_evaluate_source_memory_only.py`

Script SHA-256:

`f84821de52732b0e7a65fbca1f163cbbca7dd08098b3b2f631bc9f4b18d6989f`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val`
- Experiment id: `exp0030`
- Operation label: `source_memory_only_evaluation`
- Memory root: `/workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train`
- Query embedding root: `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat`
- Baseline reference experiment: `/workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Evaluation split: `val`
- Validation query rows: `11,219`
- Train memory rows: `78,569`
- Query embedding dimension: `2688`
- Memory embedding dimension: `2688`
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
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 1 3 5 10 20 50 --sweep-tau-values 1 5 10 20 40 --qualitative-queries 10 --seed 3407 --experiment-name exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 1 3 5 10 20 50 --sweep-tau-values 1 5 10 20 40 --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val
```

## Preconditions

- The memory experiment must already exist at `/workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train`.
- The query embeddings must already exist at `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `faiss`.
- If the `exp0005` local FAISS index is missing, this script can rebuild it from the local train embeddings.

## Input Summary

- Query split directory: `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat/val`
- Query rows: `11,219`
- Query embedding dim: `2688`
- Memory rows: `78,569`
- Memory embedding dim: `2688`
- Index loaded from disk: `true`
- Index rebuilt from embeddings: `false`
- Sweep k values: `1 3 5 10 20 50`
- Sweep tau values: `1 5 10 20 40`

## Sweep Summary

| k | tau | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.533371 | 0.065540 | 0.085996 | 0.113321 |  |
| 1 | 5 | 0.533371 | 0.065540 | 0.085996 | 0.113321 |  |
| 1 | 10 | 0.533371 | 0.065540 | 0.085996 | 0.113321 |  |
| 1 | 20 | 0.533371 | 0.065540 | 0.085996 | 0.113321 |  |
| 1 | 40 | 0.533371 | 0.065540 | 0.085996 | 0.113321 |  |
| 3 | 1 | 0.580356 | 0.089052 | 0.062569 | 0.090625 |  |
| 3 | 5 | 0.580349 | 0.089043 | 0.062574 | 0.090625 |  |
| 3 | 10 | 0.580341 | 0.089043 | 0.062617 | 0.090625 |  |
| 3 | 20 | 0.580331 | 0.089038 | 0.062616 | 0.090623 |  |
| 3 | 40 | 0.580317 | 0.089012 | 0.062690 | 0.092817 |  |
| 5 | 1 | 0.608237 | 0.101276 | 0.049396 | 0.072119 |  |
| 5 | 5 | 0.608230 | 0.101283 | 0.049400 | 0.072119 |  |
| 5 | 10 | 0.608218 | 0.101258 | 0.049423 | 0.072119 |  |
| 5 | 20 | 0.608193 | 0.101254 | 0.049461 | 0.072167 |  |
| 5 | 40 | 0.608101 | 0.100994 | 0.049737 | 0.073438 |  |
| 10 | 1 | 0.644650 | 0.115933 | 0.034095 | 0.056324 |  |
| 10 | 5 | 0.644625 | 0.115911 | 0.034100 | 0.056160 |  |
| 10 | 10 | 0.644589 | 0.115849 | 0.034113 | 0.056090 |  |
| 10 | 20 | 0.644513 | 0.115656 | 0.034155 | 0.056091 |  |
| 10 | 40 | 0.644122 | 0.114649 | 0.034407 | 0.057943 |  |
| 20 | 1 | 0.680458 | 0.130895 | 0.018822 | 0.040411 |  |
| 20 | 5 | 0.680425 | 0.130896 | 0.018834 | 0.040399 |  |
| 20 | 10 | 0.680376 | 0.130829 | 0.018888 | 0.040424 |  |
| 20 | 20 | 0.680263 | 0.130660 | 0.019221 | 0.040629 |  |
| 20 | 40 | 0.679560 | 0.128970 | 0.020461 | 0.041447 |  |
| 50 | 1 | 0.718101 | 0.143574 | 0.011127 | 0.027371 | <- best |
| 50 | 5 | 0.718087 | 0.143608 | 0.011141 | 0.027378 |  |
| 50 | 10 | 0.718052 | 0.143595 | 0.011096 | 0.027440 |  |
| 50 | 20 | 0.717879 | 0.143445 | 0.011229 | 0.028157 |  |
| 50 | 40 | 0.716916 | 0.141611 | 0.012047 | 0.031737 |  |

## Best Configuration

- Best `k`: `50`
- Best `tau`: `1`
- Validation macro AUROC: `0.718101`
- Validation macro average precision: `0.143574`
- Validation macro ECE: `0.011127`
- Validation macro F1 @ 0.5: `0.027371`
- Diagnostic macro F1 @ tuned thresholds: `0.215585`

## Query Normalization

- Raw norm mean: `1.00000000`
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

- experiment_meta.json: `19.00K`
- sweep_results.json: `221.61K`
- best_config.json: `638B`
- best_val_metrics.json: `6.27K`
- val_probabilities.npy: `613.66K`
- qualitative_neighbors.json: `40.74K`
- memory_only_selection.md: `299B`
- Total output size: `902.21K`

## Final Artifact SHA-256

```text
5a41dd1e54651c80e90b797c8cdaf48294e3e34c56a09124233966e714334abd  /workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val/experiment_meta.json
5cea1557728a244cb8897e2623d11630052b645db3916366be842c70c86142e1  /workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val/sweep_results.json
17ff9547b7853cb6e833cdc16cb7c921362870cb21b47259a0389c7b72c5d471  /workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val/best_config.json
cacfa83628e8529cebbb68db1136bd72f6d2b50f5cfe81dff7f36128695f500b  /workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val/best_val_metrics.json
9c52f1130978b3c3708c716746436805853f8d4cb04bd503f4804be7631a7be0  /workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val/val_probabilities.npy
a9e5d39be3d0d623c838543f55d7d92cd87f24663cbebfb159f49d58631ccb47  /workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val/qualitative_neighbors.json
efba757bcfd90a0df1e59cf8c82eda329b1747c53406579e3b0c4b73c847f555  /workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val/memory_only_selection.md
```

## Important Reproduction Notes

- All configuration selection in `exp0006` is validation-only.
- `val_probabilities.npy` contains the best-config validation probabilities and is small enough for plain Git.
- If the local `exp0005/index.faiss` is unavailable, rerun `exp0005` or keep the local train embeddings available so the index can be rebuilt.
- Threshold-tuned F1 is recorded as diagnostic only because thresholds are chosen on the same validation split.

## Agent Handoff Text

```text
Use /workspace/scripts/06_evaluate_source_memory_only.py and the report /workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val/recreation_report.md to recreate the validation-only memory evaluation for /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train. Sweep k and tau on the val split, select the best config by macro AUROC, and verify the saved best_config.json, best_val_metrics.json, and val_probabilities.npy artifacts.
```
