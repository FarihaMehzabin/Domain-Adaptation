# Source Memory-Only Evaluation Recreation Report

## Scope

This report documents how to recreate the validation-only memory evaluation experiment stored at:

`/workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val`

The producing script is:

`/workspace/scripts/06_evaluate_source_memory_only.py`

Script SHA-256:

`f84821de52732b0e7a65fbca1f163cbbca7dd08098b3b2f631bc9f4b18d6989f`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val`
- Experiment id: `exp0040`
- Operation label: `source_memory_only_evaluation`
- Memory root: `/workspace/experiments/exp0039__source_retrieval_memory_building__gated_hybrid_long_e50_memory_train`
- Query embedding root: `/workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160`
- Baseline reference experiment: `/workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096`
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
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0039__source_retrieval_memory_building__gated_hybrid_long_e50_memory_train --query-embedding-root /workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160 --baseline-experiment-dir /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 1 3 5 10 20 50 --sweep-tau-values 1 5 10 20 40 --qualitative-queries 10 --seed 3407 --experiment-name exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0039__source_retrieval_memory_building__gated_hybrid_long_e50_memory_train --query-embedding-root /workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160 --baseline-experiment-dir /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 1 3 5 10 20 50 --sweep-tau-values 1 5 10 20 40 --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_evaluation__gated_hybrid_long_e50_memory_val
```

## Preconditions

- The memory experiment must already exist at `/workspace/experiments/exp0039__source_retrieval_memory_building__gated_hybrid_long_e50_memory_train`.
- The query embeddings must already exist at `/workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `faiss`.
- If the `exp0005` local FAISS index is missing, this script can rebuild it from the local train embeddings.

## Input Summary

- Query split directory: `/workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160/val`
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
| 1 | 1 | 0.527001 | 0.061744 | 0.086983 | 0.101847 |  |
| 1 | 5 | 0.527001 | 0.061744 | 0.086983 | 0.101847 |  |
| 1 | 10 | 0.527001 | 0.061744 | 0.086983 | 0.101847 |  |
| 1 | 20 | 0.527001 | 0.061744 | 0.086983 | 0.101847 |  |
| 1 | 40 | 0.527001 | 0.061744 | 0.086983 | 0.101847 |  |
| 3 | 1 | 0.569177 | 0.080488 | 0.064152 | 0.075983 |  |
| 3 | 5 | 0.569179 | 0.080484 | 0.064153 | 0.075983 |  |
| 3 | 10 | 0.569181 | 0.080483 | 0.064194 | 0.075975 |  |
| 3 | 20 | 0.569183 | 0.080431 | 0.064210 | 0.076620 |  |
| 3 | 40 | 0.569169 | 0.080335 | 0.064261 | 0.077807 |  |
| 5 | 1 | 0.598180 | 0.091879 | 0.050592 | 0.060804 |  |
| 5 | 5 | 0.598179 | 0.091876 | 0.050588 | 0.060804 |  |
| 5 | 10 | 0.598178 | 0.091871 | 0.050603 | 0.060796 |  |
| 5 | 20 | 0.598175 | 0.091833 | 0.050628 | 0.061704 |  |
| 5 | 40 | 0.598128 | 0.091488 | 0.050831 | 0.063723 |  |
| 10 | 1 | 0.634681 | 0.107408 | 0.035061 | 0.041106 |  |
| 10 | 5 | 0.634677 | 0.107414 | 0.035067 | 0.041110 |  |
| 10 | 10 | 0.634663 | 0.107217 | 0.035074 | 0.040928 |  |
| 10 | 20 | 0.634601 | 0.107086 | 0.035191 | 0.040818 |  |
| 10 | 40 | 0.634118 | 0.105843 | 0.035621 | 0.041979 |  |
| 20 | 1 | 0.671301 | 0.119794 | 0.019574 | 0.028038 |  |
| 20 | 5 | 0.671267 | 0.119786 | 0.019597 | 0.027976 |  |
| 20 | 10 | 0.671224 | 0.119747 | 0.019673 | 0.027984 |  |
| 20 | 20 | 0.671043 | 0.119318 | 0.020140 | 0.028182 |  |
| 20 | 40 | 0.669805 | 0.116775 | 0.021448 | 0.029556 |  |
| 50 | 1 | 0.709453 | 0.132715 | 0.011119 | 0.017974 | <- best |
| 50 | 5 | 0.709409 | 0.132465 | 0.011146 | 0.017974 |  |
| 50 | 10 | 0.709367 | 0.132451 | 0.011253 | 0.017982 |  |
| 50 | 20 | 0.708951 | 0.131710 | 0.011427 | 0.018349 |  |
| 50 | 40 | 0.706597 | 0.128387 | 0.012901 | 0.019576 |  |

## Best Configuration

- Best `k`: `50`
- Best `tau`: `1`
- Validation macro AUROC: `0.709453`
- Validation macro average precision: `0.132715`
- Validation macro ECE: `0.011119`
- Validation macro F1 @ 0.5: `0.017974`
- Diagnostic macro F1 @ tuned thresholds: `0.203497`

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

- experiment_meta.json: `16.35K`
- sweep_results.json: `220.87K`
- best_config.json: `639B`
- best_val_metrics.json: `6.26K`
- val_probabilities.npy: `613.66K`
- qualitative_neighbors.json: `41.45K`
- memory_only_selection.md: `299B`
- Total output size: `899.52K`

## Final Artifact SHA-256

```text
073d85f18ae899ef208f72911d773d7a15cdf6388173d3b9766fd7994194c319  /workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val/experiment_meta.json
265f611e4e07058eb01d8fd08bd472b0d2a958731a2760462b517d5ef3b51bdc  /workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val/sweep_results.json
44fed110b54e22219bbeddc4f3bd6a261f69dd8dc79f81d810b18bc742bd2b5c  /workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val/best_config.json
c09468c2cc8676c0033f6202354c5140514994de06a1e311b020ba24f211a6a5  /workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val/best_val_metrics.json
2bd96ce2721153b31ec78a511284ca51870af43390dcd87f31395b0397db2865  /workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val/val_probabilities.npy
95cb74a01f5674376009bda273e6e257ffc8909a20b3faa2fe046f7fe9315a04  /workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val/qualitative_neighbors.json
3f67c3358f65ffee27aa6aa4dec9f2b95da59704319e1d172720934283bf7d74  /workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val/memory_only_selection.md
```

## Important Reproduction Notes

- All configuration selection in `exp0006` is validation-only.
- `val_probabilities.npy` contains the best-config validation probabilities and is small enough for plain Git.
- If the local `exp0005/index.faiss` is unavailable, rerun `exp0005` or keep the local train embeddings available so the index can be rebuilt.
- Threshold-tuned F1 is recorded as diagnostic only because thresholds are chosen on the same validation split.

## Agent Handoff Text

```text
Use /workspace/scripts/06_evaluate_source_memory_only.py and the report /workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val/recreation_report.md to recreate the validation-only memory evaluation for /workspace/experiments/exp0039__source_retrieval_memory_building__gated_hybrid_long_e50_memory_train. Sweep k and tau on the val split, select the best config by macro AUROC, and verify the saved best_config.json, best_val_metrics.json, and val_probabilities.npy artifacts.
```
