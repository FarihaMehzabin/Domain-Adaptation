# Source Probability-Mixing Recreation Report

## Scope

This report documents how to recreate the validation-only probability-mixing experiment stored at:

`/workspace/experiments/exp0007__source_probability_mixing_evaluation__nih_cxr14_exp0006_val_e100_p4`

The producing script is:

`/workspace/scripts/07_evaluate_probability_mixing.py`

Script SHA-256:

`f250dd26bb4f93f275809a90d9008782146e18f5277fc6b55a726d34f3ba5280`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0007__source_probability_mixing_evaluation__nih_cxr14_exp0006_val_e100_p4`
- Experiment id: `exp0007`
- Operation label: `source_probability_mixing_evaluation`
- Memory-evaluation root: `/workspace/experiments/exp0006__source_memory_only_evaluation__nih_cxr14_exp0005_val_e100_p4`
- Baseline experiment: `/workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`
- Query embedding root: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Evaluation split: `val`
- Validation rows: `11,219`
- Label count: `14`
- Label names: `atelectasis cardiomegaly consolidation edema pleural_effusion emphysema fibrosis hernia infiltration mass nodule pleural_thickening pneumonia pneumothorax`
- Selection metric: `macro_auroc`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- PyTorch: `2.4.1+cu124`
- Platform: `Linux-6.8.0-85-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0006__source_memory_only_evaluation__nih_cxr14_exp0005_val_e100_p4 --baseline-experiment-dir /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --seed 3407 --experiment-name exp0007__source_probability_mixing_evaluation__nih_cxr14_exp0006_val_e100_p4 --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0006__source_memory_only_evaluation__nih_cxr14_exp0005_val_e100_p4 --baseline-experiment-dir /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --seed 3407 --experiment-name source_probability_mixing_evaluation__nih_cxr14_exp0006_val_e100_p4
```

## Preconditions

- The memory-only evaluation must already exist at `/workspace/experiments/exp0006__source_memory_only_evaluation__nih_cxr14_exp0005_val_e100_p4`.
- The baseline experiment must already exist at `/workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`.
- The query embeddings must already exist at `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4/best.ckpt`
- Memory probabilities: `/workspace/experiments/exp0006__source_memory_only_evaluation__nih_cxr14_exp0005_val_e100_p4/val_probabilities.npy`
- Query embedding split: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val`
- Validation rows: `11,219`
- FAISS note: `Not used directly in this stage; memory probabilities come from exp0006.`

## Sweep Summary

| alpha | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.682669 | 0.128678 | 0.009410 | 0.017704 |  |
| 0.1 | 0.752218 | 0.143036 | 0.031488 | 0.025632 |  |
| 0.2 | 0.764820 | 0.151410 | 0.063887 | 0.041084 |  |
| 0.3 | 0.770096 | 0.157421 | 0.096443 | 0.061077 |  |
| 0.4 | 0.772664 | 0.160512 | 0.129148 | 0.091241 |  |
| 0.5 | 0.773982 | 0.161734 | 0.161834 | 0.154490 |  |
| 0.6 | 0.774614 | 0.162776 | 0.194548 | 0.210068 |  |
| 0.7 | 0.774842 | 0.163146 | 0.227235 | 0.211561 | <- best |
| 0.8 | 0.774841 | 0.162559 | 0.259937 | 0.203592 |  |
| 0.9 | 0.774697 | 0.161645 | 0.292654 | 0.192027 |  |
| 1.0 | 0.774450 | 0.159498 | 0.325353 | 0.181316 |  |

## Best Configuration

- Best alpha: `0.7`
- Validation macro AUROC: `0.774842`
- Validation macro average precision: `0.163146`
- Validation macro ECE: `0.227235`
- Validation macro F1 @ 0.5: `0.211561`
- Diagnostic macro F1 @ tuned thresholds: `0.234437`

## Baseline Comparison

- Frozen baseline validation macro AUROC: `0.774450`
- Frozen baseline validation macro average precision: `0.159498`
- Best mixed minus baseline macro AUROC: `0.000392`
- Best mixed minus baseline macro average precision: `0.003648`
- Baseline reconstruction matches archived exp0004 forward metrics within 5e-4: `true`
- Baseline reconstruction max absolute metric delta: `0.000316035093`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `sweep_results.json`
- `best_config.json`
- `best_val_metrics.json`
- `val_mixed_probabilities.npy`
- `probability_mixing_selection.md`

## Output Sizes

- experiment_meta.json: `58.42K`
- sweep_results.json: `96.12K`
- best_config.json: `1.17K`
- best_val_metrics.json: `6.37K`
- val_mixed_probabilities.npy: `613.66K`
- probability_mixing_selection.md: `419B`
- Total output size: `776.15K`

## Final Artifact SHA-256

```text
52061a0de548774e2be59cf284c07e459ccfcd5ae692f612fc9a6ac825dd5fec  /workspace/experiments/exp0007__source_probability_mixing_evaluation__nih_cxr14_exp0006_val_e100_p4/experiment_meta.json
1faa212935aeead12e5f768a44d18b49ef6192128ed87342674b45d948befda1  /workspace/experiments/exp0007__source_probability_mixing_evaluation__nih_cxr14_exp0006_val_e100_p4/sweep_results.json
68c9f00e138778b9bf91afa71bf0e11a4468d1a19aca659b670d9fdf79f54d99  /workspace/experiments/exp0007__source_probability_mixing_evaluation__nih_cxr14_exp0006_val_e100_p4/best_config.json
8468be9d1faaca4f9cf2d4337851d1aa8201886bed9d3aba7c27c93d3d5832d1  /workspace/experiments/exp0007__source_probability_mixing_evaluation__nih_cxr14_exp0006_val_e100_p4/best_val_metrics.json
a190dae93647572df6359611cba33dc527909e361acf56112b013b35437b3fb6  /workspace/experiments/exp0007__source_probability_mixing_evaluation__nih_cxr14_exp0006_val_e100_p4/val_mixed_probabilities.npy
6645b7b8e39233cd1054d345ad08b49d4b4bd7e5f6397c3d72ce65e8dde16999  /workspace/experiments/exp0007__source_probability_mixing_evaluation__nih_cxr14_exp0006_val_e100_p4/probability_mixing_selection.md
```

## Important Reproduction Notes

- All selection in `exp0007` is validation-only.
- `alpha=1.0` corresponds to the frozen baseline alone, and `alpha=0.0` corresponds to the selected memory-only probabilities from `exp0006`.
- `val_mixed_probabilities.npy` stores the best-config mixed probabilities in validation row order.
- Tied settings are resolved conservatively in favor of larger alpha.

## Agent Handoff Text

```text
Use /workspace/scripts/07_evaluate_probability_mixing.py and the report /workspace/experiments/exp0007__source_probability_mixing_evaluation__nih_cxr14_exp0006_val_e100_p4/recreation_report.md to recreate the validation-only probability-mixing stage that combines /workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 with /workspace/experiments/exp0006__source_memory_only_evaluation__nih_cxr14_exp0005_val_e100_p4. Reconstruct the frozen baseline validation probabilities, mix them with the exp0006 memory probabilities across alpha in [0.0, 1.0], and verify the saved best_config.json, best_val_metrics.json, and val_mixed_probabilities.npy artifacts.
```
