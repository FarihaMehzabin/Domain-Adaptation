# Source Memory-Only Evaluation Recreation Report

## Scope

This report documents how to recreate the validation-only memory evaluation experiment stored at:

`/workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val`

The producing script is:

`/workspace/scripts/06_evaluate_source_memory_only.py`

Script SHA-256:

`f84821de52732b0e7a65fbca1f163cbbca7dd08098b3b2f631bc9f4b18d6989f`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val`
- Experiment id: `exp0022`
- Operation label: `source_memory_only_evaluation`
- Memory root: `/workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train`
- Query embedding root: `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160`
- Baseline reference experiment: `/workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096`
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
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train --query-embedding-root /workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160 --baseline-experiment-dir /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 1 3 5 10 20 50 --sweep-tau-values 1 5 10 20 40 --qualitative-queries 10 --seed 3407 --experiment-name exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train --query-embedding-root /workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160 --baseline-experiment-dir /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 1 3 5 10 20 50 --sweep-tau-values 1 5 10 20 40 --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_evaluation__xattn_long_e50_memory_val
```

## Preconditions

- The memory experiment must already exist at `/workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train`.
- The query embeddings must already exist at `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `faiss`.
- If the `exp0005` local FAISS index is missing, this script can rebuild it from the local train embeddings.

## Input Summary

- Query split directory: `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160/val`
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
| 1 | 1 | 0.529116 | 0.062559 | 0.087358 | 0.106691 |  |
| 1 | 5 | 0.529116 | 0.062559 | 0.087358 | 0.106691 |  |
| 1 | 10 | 0.529116 | 0.062559 | 0.087358 | 0.106691 |  |
| 1 | 20 | 0.529116 | 0.062559 | 0.087358 | 0.106691 |  |
| 1 | 40 | 0.529116 | 0.062559 | 0.087358 | 0.106691 |  |
| 3 | 1 | 0.567103 | 0.082695 | 0.064848 | 0.079571 |  |
| 3 | 5 | 0.567102 | 0.082704 | 0.064848 | 0.079571 |  |
| 3 | 10 | 0.567102 | 0.082705 | 0.064876 | 0.079558 |  |
| 3 | 20 | 0.567100 | 0.082697 | 0.064872 | 0.080092 |  |
| 3 | 40 | 0.567100 | 0.082781 | 0.064890 | 0.080279 |  |
| 5 | 1 | 0.592344 | 0.093320 | 0.051309 | 0.065604 |  |
| 5 | 5 | 0.592332 | 0.093304 | 0.051313 | 0.065604 |  |
| 5 | 10 | 0.592332 | 0.093299 | 0.051355 | 0.065640 |  |
| 5 | 20 | 0.592320 | 0.093226 | 0.051374 | 0.065661 |  |
| 5 | 40 | 0.592337 | 0.093067 | 0.051570 | 0.066585 |  |
| 10 | 1 | 0.627936 | 0.107359 | 0.036024 | 0.046734 |  |
| 10 | 5 | 0.627935 | 0.107360 | 0.036025 | 0.046659 |  |
| 10 | 10 | 0.627937 | 0.107330 | 0.036006 | 0.046659 |  |
| 10 | 20 | 0.627913 | 0.107211 | 0.036022 | 0.046599 |  |
| 10 | 40 | 0.627891 | 0.106816 | 0.036159 | 0.046724 |  |
| 20 | 1 | 0.666667 | 0.118936 | 0.020684 | 0.029510 |  |
| 20 | 5 | 0.666648 | 0.118917 | 0.020699 | 0.029510 |  |
| 20 | 10 | 0.666622 | 0.118847 | 0.020712 | 0.029519 |  |
| 20 | 20 | 0.666523 | 0.118683 | 0.020875 | 0.029449 |  |
| 20 | 40 | 0.666327 | 0.118464 | 0.021446 | 0.030396 |  |
| 50 | 1 | 0.702234 | 0.128844 | 0.011892 | 0.018889 | <- best |
| 50 | 5 | 0.702215 | 0.128824 | 0.011896 | 0.018889 |  |
| 50 | 10 | 0.702182 | 0.128771 | 0.011946 | 0.018889 |  |
| 50 | 20 | 0.701971 | 0.128848 | 0.012037 | 0.019024 |  |
| 50 | 40 | 0.701073 | 0.128232 | 0.012593 | 0.019649 |  |

## Best Configuration

- Best `k`: `50`
- Best `tau`: `1`
- Validation macro AUROC: `0.702234`
- Validation macro average precision: `0.128844`
- Validation macro ECE: `0.011892`
- Validation macro F1 @ 0.5: `0.018889`
- Diagnostic macro F1 @ tuned thresholds: `0.202572`

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

- experiment_meta.json: `15.74K`
- sweep_results.json: `221.23K`
- best_config.json: `639B`
- best_val_metrics.json: `6.25K`
- val_probabilities.npy: `613.66K`
- qualitative_neighbors.json: `40.90K`
- memory_only_selection.md: `299B`
- Total output size: `898.70K`

## Final Artifact SHA-256

```text
25fae6e863e8c439ac086f0644058697f1dafa03e877ffab4693ad6cf003b98d  /workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val/experiment_meta.json
de2b85e56e8d4f9d7298da2e0a88111efe7ad158763ea25d46eead797047ea37  /workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val/sweep_results.json
42839e57a4635ec51955c12160f87f71aa8a074d855d9b1cdabf835d55608bf3  /workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val/best_config.json
5258cae9923a7f9a39587b69224e5d40485ba4a7b3a5d2fc01d3a62199bde2f5  /workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val/best_val_metrics.json
e1a07f17a2ebe0d0c2d4a2a041fd2766486e4d1609a7f9d54b30a4c3e089ddac  /workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val/val_probabilities.npy
9a7ff98976bc1a51377a233d0cbbaaa56c07090a9d57b31b0ae98de819c0810b  /workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val/qualitative_neighbors.json
8bff8a79ee4e706df66cd8ba47831ae2687bb0de2ea2a07b719e13ea2282915a  /workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val/memory_only_selection.md
```

## Important Reproduction Notes

- All configuration selection in `exp0006` is validation-only.
- `val_probabilities.npy` contains the best-config validation probabilities and is small enough for plain Git.
- If the local `exp0005/index.faiss` is unavailable, rerun `exp0005` or keep the local train embeddings available so the index can be rebuilt.
- Threshold-tuned F1 is recorded as diagnostic only because thresholds are chosen on the same validation split.

## Agent Handoff Text

```text
Use /workspace/scripts/06_evaluate_source_memory_only.py and the report /workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val/recreation_report.md to recreate the validation-only memory evaluation for /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train. Sweep k and tau on the val split, select the best config by macro AUROC, and verify the saved best_config.json, best_val_metrics.json, and val_probabilities.npy artifacts.
```
