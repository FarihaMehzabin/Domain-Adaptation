# Source Memory-Only Evaluation Recreation Report

## Scope

This report documents how to recreate the validation-only memory evaluation experiment stored at:

`/workspace/experiments/exp0020__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval`

The producing script is:

`/workspace/scripts/06_evaluate_source_memory_only.py`

Script SHA-256:

`86c8594ef12b8d43261951eba7643ac839f5c55f3e8bb475ee5ff1b7ae9ae4c4`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0020__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval`
- Experiment id: `exp0020`
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
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0019__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4_expanded_retrieval --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 1 3 5 10 20 50 100 200 500 1000 --sweep-tau-values 0.01 0.05 0.1 0.25 0.5 0.75 1 2 5 10 20 40 --qualitative-queries 10 --seed 3407 --experiment-name exp0020__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0019__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4_expanded_retrieval --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 1 3 5 10 20 50 100 200 500 1000 --sweep-tau-values 0.01 0.05 0.1 0.25 0.5 0.75 1 2 5 10 20 40 --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval
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
- Sweep k values: `1 3 5 10 20 50 100 200 500 1000`
- Sweep tau values: `0.01 0.05 0.1 0.25 0.5 0.75 1 2 5 10 20 40`

## Sweep Summary

| k | tau | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.01 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 0.05 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 0.1 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 0.25 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 0.5 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 0.75 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 1 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 2 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 5 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 10 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 20 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 40 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 3 | 0.01 | 0.563280 | 0.081273 | 0.064103 | 0.075549 |  |
| 3 | 0.05 | 0.563279 | 0.081274 | 0.064103 | 0.075549 |  |
| 3 | 0.1 | 0.563280 | 0.081275 | 0.064103 | 0.075549 |  |
| 3 | 0.25 | 0.563280 | 0.081274 | 0.064103 | 0.075549 |  |
| 3 | 0.5 | 0.563279 | 0.081274 | 0.064103 | 0.075549 |  |
| 3 | 0.75 | 0.563279 | 0.081274 | 0.064103 | 0.075549 |  |
| 3 | 1 | 0.563279 | 0.081273 | 0.064103 | 0.075549 |  |
| 3 | 2 | 0.563278 | 0.081267 | 0.064103 | 0.075549 |  |
| 3 | 5 | 0.563276 | 0.081268 | 0.064103 | 0.075549 |  |
| 3 | 10 | 0.563275 | 0.081261 | 0.064133 | 0.075549 |  |
| 3 | 20 | 0.563269 | 0.081263 | 0.064177 | 0.075688 |  |
| 3 | 40 | 0.563258 | 0.081223 | 0.064170 | 0.077447 |  |
| 5 | 0.01 | 0.588878 | 0.091011 | 0.050501 | 0.055797 |  |
| 5 | 0.05 | 0.588878 | 0.091010 | 0.050501 | 0.055797 |  |
| 5 | 0.1 | 0.588878 | 0.091011 | 0.050501 | 0.055797 |  |
| 5 | 0.25 | 0.588877 | 0.091010 | 0.050501 | 0.055797 |  |
| 5 | 0.5 | 0.588877 | 0.091008 | 0.050501 | 0.055797 |  |
| 5 | 0.75 | 0.588877 | 0.091007 | 0.050501 | 0.055797 |  |
| 5 | 1 | 0.588876 | 0.091008 | 0.050501 | 0.055797 |  |
| 5 | 2 | 0.588873 | 0.091006 | 0.050501 | 0.055797 |  |
| 5 | 5 | 0.588865 | 0.091013 | 0.050502 | 0.055797 |  |
| 5 | 10 | 0.588852 | 0.090986 | 0.050503 | 0.055797 |  |
| 5 | 20 | 0.588823 | 0.090937 | 0.050527 | 0.055897 |  |
| 5 | 40 | 0.588733 | 0.090964 | 0.050751 | 0.058111 |  |
| 10 | 0.01 | 0.622498 | 0.104799 | 0.034624 | 0.040756 |  |
| 10 | 0.05 | 0.622497 | 0.104807 | 0.034629 | 0.040756 |  |
| 10 | 0.1 | 0.622496 | 0.104806 | 0.034628 | 0.040756 |  |
| 10 | 0.25 | 0.622496 | 0.104807 | 0.034628 | 0.040681 |  |
| 10 | 0.5 | 0.622494 | 0.104806 | 0.034627 | 0.040691 |  |
| 10 | 0.75 | 0.622493 | 0.104805 | 0.034629 | 0.040687 |  |
| 10 | 1 | 0.622492 | 0.104806 | 0.034628 | 0.040687 |  |
| 10 | 2 | 0.622487 | 0.104795 | 0.034625 | 0.040687 |  |
| 10 | 5 | 0.622472 | 0.104783 | 0.034628 | 0.040691 |  |
| 10 | 10 | 0.622447 | 0.104761 | 0.034650 | 0.040771 |  |
| 10 | 20 | 0.622398 | 0.104743 | 0.034708 | 0.040768 |  |
| 10 | 40 | 0.622253 | 0.104579 | 0.034768 | 0.041034 |  |
| 20 | 0.01 | 0.651257 | 0.116663 | 0.017705 | 0.028620 |  |
| 20 | 0.05 | 0.651257 | 0.116681 | 0.017703 | 0.028637 |  |
| 20 | 0.1 | 0.651257 | 0.116681 | 0.017703 | 0.028637 |  |
| 20 | 0.25 | 0.651256 | 0.116681 | 0.017703 | 0.028637 |  |
| 20 | 0.5 | 0.651255 | 0.116676 | 0.017704 | 0.028637 |  |
| 20 | 0.75 | 0.651254 | 0.116676 | 0.017705 | 0.028637 |  |
| 20 | 1 | 0.651253 | 0.116673 | 0.017703 | 0.028637 |  |
| 20 | 2 | 0.651247 | 0.116671 | 0.017716 | 0.028400 |  |
| 20 | 5 | 0.651230 | 0.116679 | 0.017730 | 0.028400 |  |
| 20 | 10 | 0.651199 | 0.116670 | 0.017732 | 0.028454 |  |
| 20 | 20 | 0.651131 | 0.116601 | 0.017972 | 0.028145 |  |
| 20 | 40 | 0.650784 | 0.116094 | 0.018885 | 0.030630 |  |
| 50 | 0.01 | 0.682670 | 0.128680 | 0.009409 | 0.017704 |  |
| 50 | 0.05 | 0.682670 | 0.128678 | 0.009409 | 0.017704 |  |
| 50 | 0.1 | 0.682669 | 0.128677 | 0.009409 | 0.017704 |  |
| 50 | 0.25 | 0.682670 | 0.128679 | 0.009410 | 0.017704 |  |
| 50 | 0.5 | 0.682670 | 0.128679 | 0.009410 | 0.017704 |  |
| 50 | 0.75 | 0.682669 | 0.128678 | 0.009410 | 0.017704 |  |
| 50 | 1 | 0.682669 | 0.128678 | 0.009410 | 0.017704 |  |
| 50 | 2 | 0.682667 | 0.128679 | 0.009411 | 0.017704 |  |
| 50 | 5 | 0.682660 | 0.128679 | 0.009416 | 0.017704 |  |
| 50 | 10 | 0.682648 | 0.128705 | 0.009397 | 0.017755 |  |
| 50 | 20 | 0.682607 | 0.128579 | 0.009503 | 0.018088 |  |
| 50 | 40 | 0.682162 | 0.127916 | 0.009928 | 0.019343 |  |
| 100 | 0.01 | 0.702402 | 0.134057 | 0.007144 | 0.009450 |  |
| 100 | 0.05 | 0.702402 | 0.134056 | 0.007144 | 0.009450 |  |
| 100 | 0.1 | 0.702402 | 0.134056 | 0.007144 | 0.009450 |  |
| 100 | 0.25 | 0.702401 | 0.134056 | 0.007144 | 0.009450 |  |
| 100 | 0.5 | 0.702401 | 0.134136 | 0.007144 | 0.009450 |  |
| 100 | 0.75 | 0.702402 | 0.134137 | 0.007154 | 0.009450 |  |
| 100 | 1 | 0.702402 | 0.134138 | 0.007155 | 0.009450 |  |
| 100 | 2 | 0.702401 | 0.134141 | 0.007149 | 0.009450 |  |
| 100 | 5 | 0.702404 | 0.134142 | 0.007112 | 0.009383 |  |
| 100 | 10 | 0.702419 | 0.134149 | 0.007081 | 0.009657 |  |
| 100 | 20 | 0.702459 | 0.134124 | 0.006991 | 0.009798 |  |
| 100 | 40 | 0.701858 | 0.133892 | 0.007149 | 0.012422 |  |
| 200 | 0.01 | 0.716450 | 0.138672 | 0.006264 | 0.004993 |  |
| 200 | 0.05 | 0.716451 | 0.138672 | 0.006264 | 0.004993 |  |
| 200 | 0.1 | 0.716450 | 0.138672 | 0.006264 | 0.004993 |  |
| 200 | 0.25 | 0.716450 | 0.138672 | 0.006264 | 0.004993 |  |
| 200 | 0.5 | 0.716449 | 0.138670 | 0.006264 | 0.004993 |  |
| 200 | 0.75 | 0.716449 | 0.138670 | 0.006263 | 0.004993 |  |
| 200 | 1 | 0.716449 | 0.138670 | 0.006263 | 0.004993 |  |
| 200 | 2 | 0.716450 | 0.138671 | 0.006269 | 0.004993 |  |
| 200 | 5 | 0.716457 | 0.138609 | 0.006271 | 0.005061 |  |
| 200 | 10 | 0.716488 | 0.138568 | 0.006209 | 0.005257 |  |
| 200 | 20 | 0.716601 | 0.138671 | 0.006071 | 0.005211 |  |
| 200 | 40 | 0.716130 | 0.138208 | 0.006101 | 0.006623 |  |
| 500 | 0.01 | 0.732660 | 0.138995 | 0.006441 | 0.000417 |  |
| 500 | 0.05 | 0.732659 | 0.138998 | 0.006441 | 0.000417 |  |
| 500 | 0.1 | 0.732659 | 0.138998 | 0.006441 | 0.000417 |  |
| 500 | 0.25 | 0.732660 | 0.138998 | 0.006441 | 0.000417 |  |
| 500 | 0.5 | 0.732660 | 0.138999 | 0.006441 | 0.000417 |  |
| 500 | 0.75 | 0.732660 | 0.139001 | 0.006434 | 0.000417 |  |
| 500 | 1 | 0.732659 | 0.139002 | 0.006443 | 0.000417 |  |
| 500 | 2 | 0.732662 | 0.139155 | 0.006418 | 0.000417 |  |
| 500 | 5 | 0.732707 | 0.139116 | 0.006415 | 0.000520 |  |
| 500 | 10 | 0.732889 | 0.139242 | 0.006432 | 0.000998 |  |
| 500 | 20 | 0.733091 | 0.139525 | 0.006458 | 0.001814 |  |
| 500 | 40 | 0.733156 | 0.139576 | 0.006347 | 0.003069 |  |
| 1000 | 0.01 | 0.734156 | 0.136769 | 0.006141 | 0.000000 |  |
| 1000 | 0.05 | 0.734156 | 0.136768 | 0.006141 | 0.000000 |  |
| 1000 | 0.1 | 0.734156 | 0.136769 | 0.006141 | 0.000000 |  |
| 1000 | 0.25 | 0.734157 | 0.136769 | 0.006140 | 0.000000 |  |
| 1000 | 0.5 | 0.734156 | 0.136761 | 0.006135 | 0.000000 |  |
| 1000 | 0.75 | 0.734156 | 0.136765 | 0.006141 | 0.000000 |  |
| 1000 | 1 | 0.734157 | 0.136776 | 0.006121 | 0.000000 |  |
| 1000 | 2 | 0.734164 | 0.136846 | 0.006142 | 0.000000 |  |
| 1000 | 5 | 0.734288 | 0.137141 | 0.006138 | 0.000000 |  |
| 1000 | 10 | 0.734499 | 0.137637 | 0.006233 | 0.000000 |  |
| 1000 | 20 | 0.734826 | 0.138278 | 0.006260 | 0.000410 |  |
| 1000 | 40 | 0.735104 | 0.138792 | 0.006275 | 0.002153 | <- best |

## Best Configuration

- Best `k`: `1000`
- Best `tau`: `40`
- Validation macro AUROC: `0.735104`
- Validation macro average precision: `0.138792`
- Validation macro ECE: `0.006275`
- Validation macro F1 @ 0.5: `0.002153`
- Diagnostic macro F1 @ tuned thresholds: `0.203276`

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

- experiment_meta.json: `19.02K`
- sweep_results.json: `881.14K`
- best_config.json: `644B`
- best_val_metrics.json: `6.24K`
- val_probabilities.npy: `613.66K`
- qualitative_neighbors.json: `713.74K`
- memory_only_selection.md: `302B`
- Total output size: `2.18M`

## Final Artifact SHA-256

```text
89e56d0a8f561f98ab58926da48c6de53fc13241dd60096e3811ef26d108eec1  /workspace/experiments/exp0020__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval/experiment_meta.json
5b349db2d21dde7fe0a92de846d865b971352f51133c55675fa3b9aff06ec849  /workspace/experiments/exp0020__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval/sweep_results.json
a4aeb55fa3a0965648de551e47e510e3be7dda3653728f6210ff13db9fc9749d  /workspace/experiments/exp0020__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval/best_config.json
d39ed045c95d57a94f2c33f575ff39eddf724e25884d869d4cdf511e22344b9e  /workspace/experiments/exp0020__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval/best_val_metrics.json
6acc8ef63e88c8ce853cebe612aa180f2c5abd563fbf4eca0025303fae68c984  /workspace/experiments/exp0020__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval/val_probabilities.npy
e1dd5c72bc1892ffc69d31a3b6cc5ca3068993f5b7968b0be87f6c2e58c28b50  /workspace/experiments/exp0020__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval/qualitative_neighbors.json
f532959cbe2278d61e0fa420ca8183f9667bdf0f612618813608f6f2782c8615  /workspace/experiments/exp0020__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval/memory_only_selection.md
```

## Important Reproduction Notes

- All configuration selection in `exp0009` is validation-only.
- `val_probabilities.npy` contains the best-config validation probabilities and is small enough for plain Git.
- If the local `exp0008/index.faiss` is unavailable, rerun `exp0008` or keep the local train embeddings available so the index can be rebuilt.
- Threshold-tuned F1 is recorded as diagnostic only because thresholds are chosen on the same validation split.

## Agent Handoff Text

```text
Use /workspace/scripts/06_evaluate_source_memory_only.py and the report /workspace/experiments/exp0020__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval/recreation_report.md to recreate the validation-only memory evaluation for /workspace/experiments/exp0019__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4_expanded_retrieval. Sweep k and tau on the val split, select the best config by macro AUROC, and verify the saved best_config.json, best_val_metrics.json, and val_probabilities.npy artifacts.
```
