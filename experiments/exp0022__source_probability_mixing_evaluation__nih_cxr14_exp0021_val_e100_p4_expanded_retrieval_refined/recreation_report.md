# Source Probability-Mixing Recreation Report

## Scope

This report documents how to recreate the validation-only probability-mixing experiment stored at:

`/workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined`

The producing script is:

`/workspace/scripts/07_evaluate_probability_mixing.py`

Script SHA-256:

`f250dd26bb4f93f275809a90d9008782146e18f5277fc6b55a726d34f3ba5280`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined`
- Experiment id: `exp0022`
- Operation label: `source_probability_mixing_evaluation`
- Memory-evaluation root: `/workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined`
- Baseline experiment: `/workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`
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
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --seed 3407 --experiment-name exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --seed 3407 --experiment-name source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined
```

## Preconditions

- The memory-only evaluation must already exist at `/workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined`.
- The baseline experiment must already exist at `/workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`.
- The query embeddings must already exist at `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4/best.ckpt`
- Memory probabilities: `/workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined/val_probabilities.npy`
- Query embedding split: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val`
- Validation rows: `11,219`
- FAISS note: `Not used directly in this stage; memory probabilities come from exp0009.`

## Sweep Summary

| alpha | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.738898 | 0.136206 | 0.006554 | 0.001290 |  |
| 0.1 | 0.767387 | 0.149847 | 0.032234 | 0.004518 |  |
| 0.2 | 0.771689 | 0.154560 | 0.063824 | 0.017470 |  |
| 0.3 | 0.773354 | 0.157286 | 0.095806 | 0.038586 |  |
| 0.4 | 0.774111 | 0.158608 | 0.128525 | 0.065437 |  |
| 0.5 | 0.774461 | 0.159397 | 0.161312 | 0.128119 |  |
| 0.6 | 0.774616 | 0.159796 | 0.194121 | 0.204762 |  |
| 0.7 | 0.774656 | 0.160150 | 0.226923 | 0.211274 | <- best |
| 0.8 | 0.774624 | 0.160183 | 0.259739 | 0.203580 |  |
| 0.9 | 0.774552 | 0.160059 | 0.292555 | 0.192033 |  |
| 1.0 | 0.774450 | 0.159498 | 0.325353 | 0.181316 |  |

## Best Configuration

- Best alpha: `0.7`
- Validation macro AUROC: `0.774656`
- Validation macro average precision: `0.160150`
- Validation macro ECE: `0.226923`
- Validation macro F1 @ 0.5: `0.211274`
- Diagnostic macro F1 @ tuned thresholds: `0.233343`

## Baseline Comparison

- Frozen baseline validation macro AUROC: `0.774450`
- Frozen baseline validation macro average precision: `0.159498`
- Best mixed minus baseline macro AUROC: `0.000206`
- Best mixed minus baseline macro average precision: `0.000653`
- Baseline reconstruction matches archived exp0006 forward metrics within 5e-4: `true`
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

- experiment_meta.json: `59.62K`
- sweep_results.json: `95.83K`
- best_config.json: `1.18K`
- best_val_metrics.json: `6.36K`
- val_mixed_probabilities.npy: `613.66K`
- probability_mixing_selection.md: `419B`
- Total output size: `777.07K`

## Final Artifact SHA-256

```text
3a0c47a97c4ed921bedacaf0544212fa288ed659cd7e7728f5d0b3a5b74dc5c4  /workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined/experiment_meta.json
9ff07a49a1484d11e9922736ca92d3d3e2e1ad5ab9c2e7493acfce38867a0fe0  /workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined/sweep_results.json
d5ab8dde8aa54a4aa8d94fff1a38566fae34cc6be05a73050baa75b8a067e1e8  /workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined/best_config.json
98353dfec55c43352fa8d4df135f2c660a2d9ca7351f0d13db14547476591a75  /workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined/best_val_metrics.json
f2bc0451301532b4fb7ea25c3bd7a49bf906edcc2c623bf357c262f73e2d851f  /workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined/val_mixed_probabilities.npy
9f07b62e19bb8540185906c8e5de980260c54e2c2f65501984387d6bb04c5f0d  /workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined/probability_mixing_selection.md
```

## Important Reproduction Notes

- All selection in `exp0010` is validation-only.
- `alpha=1.0` corresponds to the frozen baseline alone, and `alpha=0.0` corresponds to the selected memory-only probabilities from `exp0009`.
- `val_mixed_probabilities.npy` stores the best-config mixed probabilities in validation row order.
- Tied settings are resolved conservatively in favor of larger alpha.

## Agent Handoff Text

```text
Use /workspace/scripts/07_evaluate_probability_mixing.py and the report /workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined/recreation_report.md to recreate the validation-only probability-mixing stage that combines /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 with /workspace/experiments/exp0021__source_memory_only_evaluation__nih_cxr14_exp0019_val_e100_p4_expanded_retrieval_refined. Reconstruct the frozen baseline validation probabilities, mix them with the exp0009 memory probabilities across alpha in [0.0, 1.0], and verify the saved best_config.json, best_val_metrics.json, and val_mixed_probabilities.npy artifacts.
```
