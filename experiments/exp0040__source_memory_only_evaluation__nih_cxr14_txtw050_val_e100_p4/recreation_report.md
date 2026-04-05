# Source Memory-Only Evaluation Recreation Report

## Scope

This report documents how to recreate the validation-only memory evaluation experiment stored at:

`/workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4`

The producing script is:

`/workspace/scripts/06_evaluate_source_memory_only.py`

Script SHA-256:

`86c8594ef12b8d43261951eba7643ac839f5c55f3e8bb475ee5ff1b7ae9ae4c4`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4`
- Experiment id: `exp0040`
- Operation label: `source_memory_only_evaluation`
- Memory root: `/workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4`
- Query embedding root: `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050`
- Baseline reference experiment: `/workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4`
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
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4 --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050 --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 1 3 5 10 20 50 --sweep-tau-values 1 5 10 20 40 --qualitative-queries 10 --seed 3407 --experiment-name exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4 --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4 --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050 --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 1 3 5 10 20 50 --sweep-tau-values 1 5 10 20 40 --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4
```

## Preconditions

- The memory experiment must already exist at `/workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4`.
- The query embeddings must already exist at `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `faiss`.
- If the `exp0008` local FAISS index is missing, this script can rebuild it from the local train embeddings.

## Input Summary

- Query split directory: `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050/val`
- Query rows: `11,219`
- Query embedding dim: `2176`
- Memory rows: `78,571`
- Memory embedding dim: `2176`
- Index loaded from disk: `true`
- Index rebuilt from embeddings: `false`
- Sweep k values: `1 3 5 10 20 50`
- Sweep tau values: `1 5 10 20 40`

## Sweep Summary

| k | tau | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.524486 | 0.060966 | 0.083213 | 0.096128 |  |
| 1 | 5 | 0.524486 | 0.060966 | 0.083213 | 0.096128 |  |
| 1 | 10 | 0.524486 | 0.060966 | 0.083213 | 0.096128 |  |
| 1 | 20 | 0.524486 | 0.060966 | 0.083213 | 0.096128 |  |
| 1 | 40 | 0.524486 | 0.060966 | 0.083213 | 0.096128 |  |
| 3 | 1 | 0.558356 | 0.079107 | 0.062961 | 0.070100 |  |
| 3 | 5 | 0.558356 | 0.079108 | 0.062942 | 0.070100 |  |
| 3 | 10 | 0.558354 | 0.079114 | 0.062939 | 0.070100 |  |
| 3 | 20 | 0.558353 | 0.079117 | 0.062899 | 0.070132 |  |
| 3 | 40 | 0.558348 | 0.079093 | 0.062851 | 0.070350 |  |
| 5 | 1 | 0.580890 | 0.087597 | 0.050461 | 0.046561 |  |
| 5 | 5 | 0.580880 | 0.087592 | 0.050446 | 0.046561 |  |
| 5 | 10 | 0.580871 | 0.087593 | 0.050427 | 0.046561 |  |
| 5 | 20 | 0.580850 | 0.087570 | 0.050390 | 0.046608 |  |
| 5 | 40 | 0.580762 | 0.087361 | 0.050398 | 0.048995 |  |
| 10 | 1 | 0.615604 | 0.100681 | 0.033922 | 0.033632 |  |
| 10 | 5 | 0.615588 | 0.100672 | 0.033900 | 0.033540 |  |
| 10 | 10 | 0.615568 | 0.100662 | 0.033860 | 0.033467 |  |
| 10 | 20 | 0.615529 | 0.100638 | 0.033952 | 0.033275 |  |
| 10 | 40 | 0.615254 | 0.100096 | 0.034217 | 0.033355 |  |
| 20 | 1 | 0.649888 | 0.112272 | 0.017466 | 0.021236 |  |
| 20 | 5 | 0.649860 | 0.112287 | 0.017475 | 0.021586 |  |
| 20 | 10 | 0.649820 | 0.112262 | 0.017534 | 0.021292 |  |
| 20 | 20 | 0.649752 | 0.112198 | 0.017803 | 0.021555 |  |
| 20 | 40 | 0.649301 | 0.111370 | 0.018860 | 0.023931 |  |
| 50 | 1 | 0.683310 | 0.125202 | 0.009441 | 0.009712 | <- best |
| 50 | 5 | 0.683298 | 0.125213 | 0.009459 | 0.009773 |  |
| 50 | 10 | 0.683291 | 0.125233 | 0.009474 | 0.009773 |  |
| 50 | 20 | 0.683294 | 0.124983 | 0.009511 | 0.009508 |  |
| 50 | 40 | 0.682474 | 0.123146 | 0.010123 | 0.011908 |  |

## Best Configuration

- Best `k`: `50`
- Best `tau`: `1`
- Validation macro AUROC: `0.683310`
- Validation macro average precision: `0.125202`
- Validation macro ECE: `0.009441`
- Validation macro F1 @ 0.5: `0.009712`
- Diagnostic macro F1 @ tuned thresholds: `0.190000`

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

- experiment_meta.json: `18.90K`
- sweep_results.json: `219.83K`
- best_config.json: `639B`
- best_val_metrics.json: `6.26K`
- val_probabilities.npy: `613.66K`
- qualitative_neighbors.json: `40.45K`
- memory_only_selection.md: `299B`
- Total output size: `900.02K`

## Final Artifact SHA-256

```text
95405d42445a3ed1c317638cd64e273d7a1aa6d744b747758098a1c6a214bb54  /workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4/experiment_meta.json
fc39012e1839d3d41d22dd706e7fc701c24ec042e51d0a0da32ce6ebcb6e6168  /workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4/sweep_results.json
b5e679bbf1d696bce3d605031e704a6af2e0b5b607b64ee5a70ee8a0adc85d9e  /workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4/best_config.json
d09d532cb2de073f9eb528f30e835218debe30ced63b101f356896eb92c4c8b9  /workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4/best_val_metrics.json
e73a5363716bcc25c975286c8c914d8a238e305ee8ab646d5671c51c2858c26b  /workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4/val_probabilities.npy
fe522c07d8480207572559b78d6202be811e8a6d7a9da9555761cac76922964e  /workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4/qualitative_neighbors.json
a89a9a4c79c7b906a3c9c021c4f4ac04e966ea378c38f59088fe7732ef0b4f88  /workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4/memory_only_selection.md
```

## Important Reproduction Notes

- All configuration selection in `exp0009` is validation-only.
- `val_probabilities.npy` contains the best-config validation probabilities and is small enough for plain Git.
- If the local `exp0008/index.faiss` is unavailable, rerun `exp0008` or keep the local train embeddings available so the index can be rebuilt.
- Threshold-tuned F1 is recorded as diagnostic only because thresholds are chosen on the same validation split.

## Agent Handoff Text

```text
Use /workspace/scripts/06_evaluate_source_memory_only.py and the report /workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4/recreation_report.md to recreate the validation-only memory evaluation for /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4. Sweep k and tau on the val split, select the best config by macro AUROC, and verify the saved best_config.json, best_val_metrics.json, and val_probabilities.npy artifacts.
```
