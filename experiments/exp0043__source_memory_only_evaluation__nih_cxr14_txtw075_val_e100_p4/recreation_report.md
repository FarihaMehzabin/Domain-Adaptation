# Source Memory-Only Evaluation Recreation Report

## Scope

This report documents how to recreate the validation-only memory evaluation experiment stored at:

`/workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4`

The producing script is:

`/workspace/scripts/06_evaluate_source_memory_only.py`

Script SHA-256:

`86c8594ef12b8d43261951eba7643ac839f5c55f3e8bb475ee5ff1b7ae9ae4c4`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4`
- Experiment id: `exp0043`
- Operation label: `source_memory_only_evaluation`
- Memory root: `/workspace/experiments/exp0042__source_retrieval_memory_building__nih_cxr14_txtw075_train_instance_memory_e100_p4`
- Query embedding root: `/workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075`
- Baseline reference experiment: `/workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4`
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
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0042__source_retrieval_memory_building__nih_cxr14_txtw075_train_instance_memory_e100_p4 --query-embedding-root /workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075 --baseline-experiment-dir /workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 1 3 5 10 20 50 --sweep-tau-values 1 5 10 20 40 --qualitative-queries 10 --seed 3407 --experiment-name exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4 --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/06_evaluate_source_memory_only.py --memory-root /workspace/experiments/exp0042__source_retrieval_memory_building__nih_cxr14_txtw075_train_instance_memory_e100_p4 --query-embedding-root /workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075 --baseline-experiment-dir /workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --sweep-k-values 1 3 5 10 20 50 --sweep-tau-values 1 5 10 20 40 --qualitative-queries 10 --seed 3407 --experiment-name source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4
```

## Preconditions

- The memory experiment must already exist at `/workspace/experiments/exp0042__source_retrieval_memory_building__nih_cxr14_txtw075_train_instance_memory_e100_p4`.
- The query embeddings must already exist at `/workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `faiss`.
- If the `exp0008` local FAISS index is missing, this script can rebuild it from the local train embeddings.

## Input Summary

- Query split directory: `/workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075/val`
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
| 1 | 1 | 0.526535 | 0.061639 | 0.083825 | 0.099783 |  |
| 1 | 5 | 0.526535 | 0.061639 | 0.083825 | 0.099783 |  |
| 1 | 10 | 0.526535 | 0.061639 | 0.083825 | 0.099783 |  |
| 1 | 20 | 0.526535 | 0.061639 | 0.083825 | 0.099783 |  |
| 1 | 40 | 0.526535 | 0.061639 | 0.083825 | 0.099783 |  |
| 3 | 1 | 0.564737 | 0.081332 | 0.063006 | 0.073084 |  |
| 3 | 5 | 0.564735 | 0.081329 | 0.062999 | 0.073084 |  |
| 3 | 10 | 0.564732 | 0.081327 | 0.063038 | 0.073084 |  |
| 3 | 20 | 0.564729 | 0.081325 | 0.063021 | 0.073208 |  |
| 3 | 40 | 0.564720 | 0.081299 | 0.063019 | 0.075207 |  |
| 5 | 1 | 0.586840 | 0.091064 | 0.050351 | 0.054678 |  |
| 5 | 5 | 0.586830 | 0.091049 | 0.050343 | 0.054678 |  |
| 5 | 10 | 0.586818 | 0.091034 | 0.050332 | 0.054678 |  |
| 5 | 20 | 0.586789 | 0.091025 | 0.050329 | 0.054883 |  |
| 5 | 40 | 0.586690 | 0.091056 | 0.050579 | 0.055314 |  |
| 10 | 1 | 0.622378 | 0.104487 | 0.034285 | 0.040589 |  |
| 10 | 5 | 0.622360 | 0.104482 | 0.034287 | 0.040802 |  |
| 10 | 10 | 0.622338 | 0.104450 | 0.034294 | 0.040807 |  |
| 10 | 20 | 0.622284 | 0.104399 | 0.034315 | 0.040965 |  |
| 10 | 40 | 0.622031 | 0.103724 | 0.034622 | 0.040616 |  |
| 20 | 1 | 0.650919 | 0.115454 | 0.017547 | 0.026407 |  |
| 20 | 5 | 0.650900 | 0.115447 | 0.017572 | 0.026326 |  |
| 20 | 10 | 0.650875 | 0.115414 | 0.017572 | 0.026241 |  |
| 20 | 20 | 0.650813 | 0.115326 | 0.017747 | 0.026335 |  |
| 20 | 40 | 0.650437 | 0.114497 | 0.018863 | 0.028672 |  |
| 50 | 1 | 0.683231 | 0.127163 | 0.009080 | 0.013449 | <- best |
| 50 | 5 | 0.683224 | 0.127152 | 0.009081 | 0.013355 |  |
| 50 | 10 | 0.683216 | 0.127151 | 0.009080 | 0.013326 |  |
| 50 | 20 | 0.683211 | 0.127016 | 0.009259 | 0.014076 |  |
| 50 | 40 | 0.682569 | 0.125938 | 0.009982 | 0.016451 |  |

## Best Configuration

- Best `k`: `50`
- Best `tau`: `1`
- Validation macro AUROC: `0.683231`
- Validation macro average precision: `0.127163`
- Validation macro ECE: `0.009080`
- Validation macro F1 @ 0.5: `0.013449`
- Diagnostic macro F1 @ tuned thresholds: `0.195595`

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
- sweep_results.json: `220.43K`
- best_config.json: `639B`
- best_val_metrics.json: `6.26K`
- val_probabilities.npy: `613.66K`
- qualitative_neighbors.json: `40.32K`
- memory_only_selection.md: `299B`
- Total output size: `900.49K`

## Final Artifact SHA-256

```text
33e0b187ae61b10e37eb5edfc6704f4b2b70155462c4a5f7a1cc88bf7ee4495d  /workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4/experiment_meta.json
79420af06679770ca7986e2b5abeeeafcb4bd62b0dc85f92d18dd5a1b2c757a2  /workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4/sweep_results.json
c8a9148fb1cd817fd217d29d3941017d1e944b6a22a56b4c7174eda9063d3738  /workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4/best_config.json
f164b3d235aca137a0b110c0e610fc18e8a00602bf20173f47e03a39478ac31e  /workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4/best_val_metrics.json
2874418f924399b6f41013c20fa5dbf2a35e1cca3adc61c26213825a6b0892fc  /workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4/val_probabilities.npy
fe223bb5ac2b25393fb8962877d928b9eb0d78cb1a5d7d56a20d5241fdb0b7fb  /workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4/qualitative_neighbors.json
299bf0c66cd0aefeea2056a8311278b4e2d0d77020f230010590443aea275119  /workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4/memory_only_selection.md
```

## Important Reproduction Notes

- All configuration selection in `exp0009` is validation-only.
- `val_probabilities.npy` contains the best-config validation probabilities and is small enough for plain Git.
- If the local `exp0008/index.faiss` is unavailable, rerun `exp0008` or keep the local train embeddings available so the index can be rebuilt.
- Threshold-tuned F1 is recorded as diagnostic only because thresholds are chosen on the same validation split.

## Agent Handoff Text

```text
Use /workspace/scripts/06_evaluate_source_memory_only.py and the report /workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4/recreation_report.md to recreate the validation-only memory evaluation for /workspace/experiments/exp0042__source_retrieval_memory_building__nih_cxr14_txtw075_train_instance_memory_e100_p4. Sweep k and tau on the val split, select the best config by macro AUROC, and verify the saved best_config.json, best_val_metrics.json, and val_probabilities.npy artifacts.
```
