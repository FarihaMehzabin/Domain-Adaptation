# Source Memory-Only Evaluation Recreation Report

## Scope

This report documents how to recreate the validation-only memory evaluation experiment stored at:

`/workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val`

The producing script is:

`/workspace/scripts/06_evaluate_source_memory_only.py`

Script SHA-256:

`fdee2dbe49794d7c885dddc97b23d7821f8c2eec9228b2582fe89325ade11fd1`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val`
- Experiment id: `exp0009`
- Operation label: `source_memory_only_evaluation`
- Memory root: `/workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory`
- Query embedding root: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`
- Baseline reference experiment: `/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear`
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
- Platform: `Linux-5.15.0-102-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --qualitative-queries 10 --seed 3407 --experiment-name exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_evaluation__nih_cxr14_exp0008_val
```

## Preconditions

- The memory experiment must already exist at `/workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory`.
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
| 5 | 5 | 0.588865 | 0.091012 | 0.050502 | 0.055797 |  |
| 5 | 10 | 0.588852 | 0.090986 | 0.050503 | 0.055797 |  |
| 5 | 20 | 0.588823 | 0.090936 | 0.050527 | 0.055897 |  |
| 5 | 40 | 0.588732 | 0.090963 | 0.050751 | 0.058111 |  |
| 10 | 1 | 0.622492 | 0.104806 | 0.034628 | 0.040687 |  |
| 10 | 5 | 0.622472 | 0.104783 | 0.034628 | 0.040691 |  |
| 10 | 10 | 0.622447 | 0.104761 | 0.034650 | 0.040771 |  |
| 10 | 20 | 0.622398 | 0.104743 | 0.034708 | 0.040768 |  |
| 10 | 40 | 0.622252 | 0.104579 | 0.034768 | 0.041034 |  |
| 20 | 1 | 0.651251 | 0.116673 | 0.017703 | 0.028637 |  |
| 20 | 5 | 0.651228 | 0.116679 | 0.017730 | 0.028400 |  |
| 20 | 10 | 0.651197 | 0.116670 | 0.017732 | 0.028454 |  |
| 20 | 20 | 0.651129 | 0.116601 | 0.017972 | 0.028145 |  |
| 20 | 40 | 0.650782 | 0.116094 | 0.018884 | 0.030630 |  |
| 50 | 1 | 0.682669 | 0.128676 | 0.009410 | 0.017704 | <- best |
| 50 | 5 | 0.682660 | 0.128677 | 0.009416 | 0.017704 |  |
| 50 | 10 | 0.682648 | 0.128704 | 0.009397 | 0.017755 |  |
| 50 | 20 | 0.682607 | 0.128577 | 0.009503 | 0.018088 |  |
| 50 | 40 | 0.682162 | 0.127914 | 0.009927 | 0.019343 |  |

## Best Configuration

- Best `k`: `50`
- Best `tau`: `1`
- Validation macro AUROC: `0.682669`
- Validation macro average precision: `0.128676`
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

- experiment_meta.json: `17.37K`
- sweep_results.json: `220.37K`
- best_config.json: `635B`
- best_val_metrics.json: `6.26K`
- val_probabilities.npy: `613.66K`
- qualitative_neighbors.json: `40.30K`
- memory_only_selection.md: `299B`
- Total output size: `898.88K`

## Final Artifact SHA-256

```text
39221ac1682b6fcb8cc85c577c3c830e746154cc7d3199745ebca0007241dfe6  /workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val/experiment_meta.json
0ed81fc066c7f16225bb275a6c0710621a5358555f0a614a9e8c1c0948981a44  /workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val/sweep_results.json
600a82e3e284a23f0bef04d31536126fbfea8a5065c3cc771ad288eba9cc6fbd  /workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val/best_config.json
06493a293c88a96211051c4383fa1e074463a5f82717a7900e6bf58259b22a39  /workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val/best_val_metrics.json
a4207ed645f55a41588fe373c9e608b1d2201e33cf4b6f234f3d0af1961de585  /workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val/val_probabilities.npy
2761f7430db1869acee9e07f6ac23a72060d9dda642fe04e4d304c310b473237  /workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val/qualitative_neighbors.json
bde13bf3138a1967d878a79f25f0492064c02b91e2231090094c44ebcbd1f5cf  /workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val/memory_only_selection.md
```

## Important Reproduction Notes

- All configuration selection in `exp0009` is validation-only.
- `val_probabilities.npy` contains the best-config validation probabilities and is small enough for plain Git.
- If the local `exp0008/index.faiss` is unavailable, rerun `exp0008` or keep the local train embeddings available so the index can be rebuilt.
- Threshold-tuned F1 is recorded as diagnostic only because thresholds are chosen on the same validation split.

## Agent Handoff Text

```text
Use /workspace/scripts/06_evaluate_source_memory_only.py and the report /workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val/recreation_report.md to recreate the validation-only memory evaluation for /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory. Sweep k and tau on the val split, select the best config by macro AUROC, and verify the saved best_config.json, best_val_metrics.json, and val_probabilities.npy artifacts.
```
