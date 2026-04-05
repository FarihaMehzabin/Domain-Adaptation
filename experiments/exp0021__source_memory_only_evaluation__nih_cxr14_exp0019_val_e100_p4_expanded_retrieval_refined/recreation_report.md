# Source Memory-Only Evaluation Recreation Report

## Scope

This report documents how to recreate the validation-only memory evaluation experiment stored at:

`/workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined`

The producing script is:

`/workspace/scripts/06_evaluate_source_memory_only.py`

Script SHA-256:

`86c8594ef12b8d43261951eba7643ac839f5c55f3e8bb475ee5ff1b7ae9ae4c4`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined`
- Experiment id: `exp0021`
- Operation label: `source_memory_only_evaluation`
- Memory root: `/workspace/experiments/exp0019__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4_expanded_retrieval`
- Query embedding root: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`
- Baseline reference experiment: `/workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Evaluation split: `val`
- Validation query rows: `11,219`
- Train memory rows: `78,571`
- Query embedding dimension: `2176`
- Memory embedding dimension: `2176`
- Label count: `14`
- Label names: `atelectasis cardiomegaly consolidation edema pleural_effusion emphysema fibrosis hernia infiltration mass nodule pleural_thickening pneumonia pneumothorax`
- Selection metric: `macro_auroc`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- faiss: `1.13.2`
- Platform: `Linux-6.8.0-85-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0019__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4_expanded_retrieval --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 50 100 200 500 1000 1500 2000 3000 --sweep-tau-values 5 10 20 40 80 160 --qualitative-queries 10 --seed 3407 --experiment-name exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0019__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4_expanded_retrieval --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 50 100 200 500 1000 1500 2000 3000 --sweep-tau-values 5 10 20 40 80 160 --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined
```

## Preconditions

- The memory experiment must already exist at `/workspace/experiments/exp0019__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4_expanded_retrieval`.
- The query embeddings must already exist at `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `faiss`.
- If the `exp0008` local FAISS index is missing, this script can rebuild it from the local train embeddings.

## Input Summary

- Query split directory: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val`
- Query rows: `11,219`
- Query embedding dim: `2176`
- Memory rows: `78,571`
- Memory embedding dim: `2176`
- Index loaded from disk: `true`
- Index rebuilt from embeddings: `false`
- Sweep k values: `50 100 200 500 1000 1500 2000 3000`
- Sweep tau values: `5 10 20 40 80 160`

## Sweep Summary

| k | tau | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 5 | 0.682660 | 0.128679 | 0.009416 | 0.017704 |  |
| 50 | 10 | 0.682648 | 0.128705 | 0.009397 | 0.017755 |  |
| 50 | 20 | 0.682607 | 0.128579 | 0.009503 | 0.018088 |  |
| 50 | 40 | 0.682162 | 0.127916 | 0.009928 | 0.019343 |  |
| 50 | 80 | 0.678783 | 0.122446 | 0.012321 | 0.027092 |  |
| 50 | 160 | 0.663366 | 0.106573 | 0.024101 | 0.049687 |  |
| 100 | 5 | 0.702404 | 0.134142 | 0.007112 | 0.009383 |  |
| 100 | 10 | 0.702419 | 0.134149 | 0.007081 | 0.009657 |  |
| 100 | 20 | 0.702459 | 0.134124 | 0.006991 | 0.009798 |  |
| 100 | 40 | 0.701858 | 0.133892 | 0.007149 | 0.012422 |  |
| 100 | 80 | 0.696932 | 0.127655 | 0.008575 | 0.020638 |  |
| 100 | 160 | 0.675514 | 0.109408 | 0.020660 | 0.047552 |  |
| 200 | 5 | 0.716457 | 0.138609 | 0.006271 | 0.005061 |  |
| 200 | 10 | 0.716488 | 0.138568 | 0.006209 | 0.005257 |  |
| 200 | 20 | 0.716601 | 0.138671 | 0.006071 | 0.005211 |  |
| 200 | 40 | 0.716130 | 0.138208 | 0.006101 | 0.006623 |  |
| 200 | 80 | 0.710475 | 0.131153 | 0.007342 | 0.015870 |  |
| 200 | 160 | 0.683716 | 0.111245 | 0.018339 | 0.046086 |  |
| 500 | 5 | 0.732707 | 0.139116 | 0.006415 | 0.000520 |  |
| 500 | 10 | 0.732889 | 0.139242 | 0.006432 | 0.000998 |  |
| 500 | 20 | 0.733091 | 0.139525 | 0.006458 | 0.001814 |  |
| 500 | 40 | 0.733156 | 0.139576 | 0.006347 | 0.003069 |  |
| 500 | 80 | 0.727592 | 0.132889 | 0.006648 | 0.012980 |  |
| 500 | 160 | 0.696216 | 0.112545 | 0.016922 | 0.045494 |  |
| 1000 | 5 | 0.734288 | 0.137141 | 0.006138 | 0.000000 |  |
| 1000 | 10 | 0.734499 | 0.137637 | 0.006233 | 0.000000 |  |
| 1000 | 20 | 0.734826 | 0.138278 | 0.006260 | 0.000410 |  |
| 1000 | 40 | 0.735104 | 0.138792 | 0.006275 | 0.002153 |  |
| 1000 | 80 | 0.730160 | 0.133171 | 0.006657 | 0.011328 |  |
| 1000 | 160 | 0.697104 | 0.112984 | 0.016361 | 0.045169 |  |
| 1500 | 5 | 0.735529 | 0.137159 | 0.006263 | 0.000000 |  |
| 1500 | 10 | 0.735730 | 0.137446 | 0.006221 | 0.000000 |  |
| 1500 | 20 | 0.736126 | 0.138195 | 0.006323 | 0.000103 |  |
| 1500 | 40 | 0.736349 | 0.138238 | 0.006443 | 0.001732 |  |
| 1500 | 80 | 0.731524 | 0.133070 | 0.006569 | 0.010943 |  |
| 1500 | 160 | 0.697973 | 0.113120 | 0.016099 | 0.044977 |  |
| 2000 | 5 | 0.738121 | 0.136223 | 0.006709 | 0.000000 |  |
| 2000 | 10 | 0.738300 | 0.136953 | 0.006791 | 0.000000 |  |
| 2000 | 20 | 0.738572 | 0.137214 | 0.006642 | 0.000000 |  |
| 2000 | 40 | 0.738511 | 0.137549 | 0.006565 | 0.001391 |  |
| 2000 | 80 | 0.733019 | 0.132947 | 0.006651 | 0.010703 |  |
| 2000 | 160 | 0.698315 | 0.113188 | 0.016004 | 0.044925 |  |
| 3000 | 5 | 0.737747 | 0.133304 | 0.006757 | 0.000000 |  |
| 3000 | 10 | 0.738061 | 0.134253 | 0.006795 | 0.000000 |  |
| 3000 | 20 | 0.738667 | 0.134954 | 0.006620 | 0.000000 |  |
| 3000 | 40 | 0.738898 | 0.136206 | 0.006554 | 0.001290 | <- best |
| 3000 | 80 | 0.733563 | 0.132704 | 0.006768 | 0.010545 |  |
| 3000 | 160 | 0.698346 | 0.113232 | 0.015893 | 0.044882 |  |

## Best Configuration

- Best `k`: `3000`
- Best `tau`: `40`
- Validation macro AUROC: `0.738898`
- Validation macro average precision: `0.136206`
- Validation macro ECE: `0.006554`
- Validation macro F1 @ 0.5: `0.001290`
- Diagnostic macro F1 @ tuned thresholds: `0.203282`

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

- experiment_meta.json: `18.99K`
- sweep_results.json: `354.91K`
- best_config.json: `645B`
- best_val_metrics.json: `6.25K`
- val_probabilities.npy: `613.66K`
- qualitative_neighbors.json: `2.08M`
- memory_only_selection.md: `302B`
- Total output size: `3.05M`

## Final Artifact SHA-256

```text
7f615241b65f2ea4e95bab6a8f2023db58caacc7666279a4f3d2a25aaf3905b8  /workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined/experiment_meta.json
64484ed9a06509bc726242e6e3d5ba65be36d4fedda0f50d997796a286515973  /workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined/sweep_results.json
d9f102c72d3060228bd9c6aa9d6e7ae8a1ffd7a3ce81dcac1145b6161ef5abab  /workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined/best_config.json
15b679126445bc877d182aa918188e9c01879a72ea416a860b37dc49e6f2e5db  /workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined/best_val_metrics.json
6461a3aa7ef9dce040f2dfc46c4a3f2c29049ed454c96d5be173df23f60af99d  /workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined/val_probabilities.npy
842de2c80c04dd26adbc497f273f46df8282a7369efd5fe197fbb55b884a69c5  /workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined/qualitative_neighbors.json
97f1766a47b06b65aa8fe2c0c9661dcf7785841f02387a1d4b4702b3ba333f7d  /workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined/memory_only_selection.md
```

## Important Reproduction Notes

- All configuration selection in `exp0009` is validation-only.
- `val_probabilities.npy` contains the best-config validation probabilities and is small enough for plain Git.
- If the local `exp0008/index.faiss` is unavailable, rerun `exp0008` or keep the local train embeddings available so the index can be rebuilt.
- Threshold-tuned F1 is recorded as diagnostic only because thresholds are chosen on the same validation split.

## Agent Handoff Text

```text
Use /workspace/scripts/06_evaluate_source_memory_only.py and the report /workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined/recreation_report.md to recreate the validation-only memory evaluation for /workspace/experiments/exp0019__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4_expanded_retrieval. Sweep k and tau on the val split, select the best config by macro AUROC, and verify the saved best_config.json, best_val_metrics.json, and val_probabilities.npy artifacts.
```
