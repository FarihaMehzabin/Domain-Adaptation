# Source Memory-Only Evaluation Recreation Report

## Scope

This report documents how to recreate the validation-only memory evaluation experiment stored at:

`/workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4`

The producing script is:

`/workspace/scripts/06_evaluate_source_memory_only.py`

Script SHA-256:

`fdee2dbe49794d7c885dddc97b23d7821f8c2eec9228b2582fe89325ade11fd1`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4`
- Experiment id: `exp0015`
- Operation label: `source_memory_only_evaluation`
- Memory root: `/workspace/experiments/exp0014__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4`
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
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0014__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4 --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --qualitative-queries 10 --seed 3407 --experiment-name exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4 --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0014__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4 --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4
```

## Preconditions

- The memory experiment must already exist at `/workspace/experiments/exp0014__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4`.
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

## Sweep Summary

| k | tau | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 5 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 10 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 20 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 1 | 40 | 0.529594 | 0.062998 | 0.085028 | 0.105222 |  |
| 3 | 1 | 0.563279 | 0.081273 | 0.064103 | 0.075549 |  |
| 3 | 5 | 0.563276 | 0.081268 | 0.064103 | 0.075549 |  |
| 3 | 10 | 0.563275 | 0.081261 | 0.064133 | 0.075549 |  |
| 3 | 20 | 0.563269 | 0.081263 | 0.064177 | 0.075688 |  |
| 3 | 40 | 0.563258 | 0.081223 | 0.064170 | 0.077447 |  |
| 5 | 1 | 0.588876 | 0.091008 | 0.050501 | 0.055797 |  |
| 5 | 5 | 0.588865 | 0.091013 | 0.050502 | 0.055797 |  |
| 5 | 10 | 0.588852 | 0.090986 | 0.050503 | 0.055797 |  |
| 5 | 20 | 0.588823 | 0.090937 | 0.050527 | 0.055897 |  |
| 5 | 40 | 0.588733 | 0.090964 | 0.050751 | 0.058111 |  |
| 10 | 1 | 0.622492 | 0.104806 | 0.034628 | 0.040687 |  |
| 10 | 5 | 0.622472 | 0.104783 | 0.034628 | 0.040691 |  |
| 10 | 10 | 0.622447 | 0.104761 | 0.034650 | 0.040771 |  |
| 10 | 20 | 0.622398 | 0.104743 | 0.034708 | 0.040768 |  |
| 10 | 40 | 0.622253 | 0.104579 | 0.034768 | 0.041034 |  |
| 20 | 1 | 0.651253 | 0.116673 | 0.017703 | 0.028637 |  |
| 20 | 5 | 0.651230 | 0.116679 | 0.017730 | 0.028400 |  |
| 20 | 10 | 0.651199 | 0.116670 | 0.017732 | 0.028454 |  |
| 20 | 20 | 0.651131 | 0.116601 | 0.017972 | 0.028145 |  |
| 20 | 40 | 0.650784 | 0.116094 | 0.018885 | 0.030630 |  |
| 50 | 1 | 0.682669 | 0.128678 | 0.009410 | 0.017704 | <- best |
| 50 | 5 | 0.682660 | 0.128679 | 0.009416 | 0.017704 |  |
| 50 | 10 | 0.682648 | 0.128705 | 0.009397 | 0.017755 |  |
| 50 | 20 | 0.682607 | 0.128579 | 0.009503 | 0.018088 |  |
| 50 | 40 | 0.682162 | 0.127916 | 0.009928 | 0.019343 |  |

## Best Configuration

- Best `k`: `50`
- Best `tau`: `1`
- Validation macro AUROC: `0.682669`
- Validation macro average precision: `0.128678`
- Validation macro ECE: `0.009410`
- Validation macro F1 @ 0.5: `0.017704`
- Diagnostic macro F1 @ tuned thresholds: `0.197692`

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

- experiment_meta.json: `18.17K`
- sweep_results.json: `220.35K`
- best_config.json: `634B`
- best_val_metrics.json: `6.25K`
- val_probabilities.npy: `613.66K`
- qualitative_neighbors.json: `40.29K`
- memory_only_selection.md: `299B`
- Total output size: `899.64K`

## Final Artifact SHA-256

```text
a4e04678a0905bc1d10cd27e815fb0f470d104b303773f2c9bddcc43e9a645d0  /workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4/experiment_meta.json
d333de9963437da58358712fbd3fe26f7e6c45cc4886b948153a91e5f8b4baf6  /workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4/sweep_results.json
53ba6d29a1365c7d9ad2158ca3c7f60eab176b4d1a5372bde3f267ac8e3ba09b  /workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4/best_config.json
8d4e06a07999734e8f1b1b5046bad13d116428e18a4a7153c50adf761019c6c8  /workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4/best_val_metrics.json
dc7118dd294409755f954b4ac19ec13cd0d01b2069d7b26846939266bb6c0d6b  /workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4/val_probabilities.npy
42155c34b96e053db9d197cdc5161f9fc7a2b1068207e577a1b3adb52cf0e1b3  /workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4/qualitative_neighbors.json
55a6ad9e5a1c23e5313580f07792c2529aaf75cfa32911b7a9ea182ef93f5182  /workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4/memory_only_selection.md
```

## Important Reproduction Notes

- All configuration selection in `exp0009` is validation-only.
- `val_probabilities.npy` contains the best-config validation probabilities and is small enough for plain Git.
- If the local `exp0008/index.faiss` is unavailable, rerun `exp0008` or keep the local train embeddings available so the index can be rebuilt.
- Threshold-tuned F1 is recorded as diagnostic only because thresholds are chosen on the same validation split.

## Agent Handoff Text

```text
Use /workspace/scripts/06_evaluate_source_memory_only.py and the report /workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4/recreation_report.md to recreate the validation-only memory evaluation for /workspace/experiments/exp0014__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory_e100_p4. Sweep k and tau on the val split, select the best config by macro AUROC, and verify the saved best_config.json, best_val_metrics.json, and val_probabilities.npy artifacts.
```
