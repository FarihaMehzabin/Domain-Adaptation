# Source Probability-Mixing Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test probability-mixing experiment stored at:

`/workspace/experiments/exp0024__source_probability_mixing_test_evaluation__nih_cxr14_exp0022_test_e100_p4_expanded_retrieval_refined`

The producing script is:

`/workspace/scripts/09_evaluate_probability_mixing_test.py`

Script SHA-256:

`9ea4e4811bfcfe1f242c9af7c883d04f134d0d4f657f5712bb673da3d684c183`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0024__source_probability_mixing_test_evaluation__nih_cxr14_exp0022_test_e100_p4_expanded_retrieval_refined`
- Experiment id: `exp0024`
- Operation label: `source_probability_mixing_test_evaluation`
- Memory-test root: `/workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined`
- Validation mixing-selection root: `/workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined`
- Baseline experiment: `/workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`
- Query embedding root: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Evaluation split: `test`
- Test rows: `22,330`
- Label count: `14`
- Label names: `atelectasis cardiomegaly consolidation edema pleural_effusion emphysema fibrosis hernia infiltration mass nodule pleural_thickening pneumonia pneumothorax`
- Applied selection mode: `frozen_validation_config`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- PyTorch: `2.4.1+cu124`
- Platform: `Linux-6.8.0-85-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/09_evaluate_probability_mixing_test.py --memory-test-root /workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined --mixing-selection-root /workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --batch-size 2048 --seed 3407 --experiment-name exp0024__source_probability_mixing_test_evaluation__nih_cxr14_exp0022_test_e100_p4_expanded_retrieval_refined --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/09_evaluate_probability_mixing_test.py --memory-test-root /workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined --mixing-selection-root /workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --batch-size 2048 --seed 3407 --experiment-name source_probability_mixing_test_evaluation__nih_cxr14_exp0022_test_e100_p4_expanded_retrieval_refined
```

## Preconditions

- The memory-only test evaluation must already exist at `/workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined`.
- The validation probability-mixing selection must already exist at `/workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined`.
- The baseline experiment must already exist at `/workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`.
- The query embeddings must already exist at `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.
- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected alpha.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4/best.ckpt`
- Memory test probabilities: `/workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined/test_probabilities.npy`
- Validation selection root: `/workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined`
- Query embedding split: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/test`
- Test rows: `22,330`

## Applied Configuration

- Frozen `alpha`: `0.7`
- Threshold source: `/workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined/best_val_metrics.json`

## Test Metrics

- Mixed test macro AUROC: `0.775046`
- Mixed test macro average precision: `0.159121`
- Mixed test macro ECE: `0.227055`
- Mixed test macro F1 @ 0.5: `0.211121`
- Mixed test macro F1 @ frozen val thresholds: `0.221494`

## Baseline Comparison

- Frozen baseline test macro AUROC: `0.774644`
- Frozen baseline test macro average precision: `0.160416`
- Mixed minus baseline macro AUROC: `0.000402`
- Mixed minus baseline macro average precision: `-0.001295`
- Mixed minus baseline macro ECE: `-0.098168`
- Mixed minus baseline macro F1 @ 0.5: `0.031980`
- Mixed minus baseline macro F1 @ frozen val thresholds: `0.000489`
- Baseline reconstruction matches archived exp0006 test metrics within 5e-4: `false`
- Baseline reconstruction max absolute metric delta: `0.000807065064`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `applied_config.json`
- `test_metrics.json`
- `test_mixed_probabilities.npy`
- `probability_mixing_test_summary.md`

## Output Sizes

- experiment_meta.json: `155.33K`
- applied_config.json: `2.06K`
- test_metrics.json: `9.02K`
- test_mixed_probabilities.npy: `1.19M`
- probability_mixing_test_summary.md: `979B`
- Total output size: `1.36M`

## Final Artifact SHA-256

```text
0c55dbfa5efc947fd4aecbc9749969ae88579fdc425fe8d8abe59b13b117fe39  /workspace/experiments/exp0024__source_probability_mixing_test_evaluation__nih_cxr14_exp0022_test_e100_p4_expanded_retrieval_refined/experiment_meta.json
668b632fe98eaa9c109d99055907c9336c5cc78e1624b274256d39145061d280  /workspace/experiments/exp0024__source_probability_mixing_test_evaluation__nih_cxr14_exp0022_test_e100_p4_expanded_retrieval_refined/applied_config.json
fab5ce7dbd16efcc2c1c4ce0c6699dfddce44684772507d28223638b4ba497c0  /workspace/experiments/exp0024__source_probability_mixing_test_evaluation__nih_cxr14_exp0022_test_e100_p4_expanded_retrieval_refined/test_metrics.json
1df5be28b36e586c5308d2128eb605fe1bf9ea99c1b81a53ac93021deeeffc54  /workspace/experiments/exp0024__source_probability_mixing_test_evaluation__nih_cxr14_exp0022_test_e100_p4_expanded_retrieval_refined/test_mixed_probabilities.npy
172440a96d3ffb8577c96bd8642e22bd4b155e3702cd1f73bd383212fda7e112  /workspace/experiments/exp0024__source_probability_mixing_test_evaluation__nih_cxr14_exp0022_test_e100_p4_expanded_retrieval_refined/probability_mixing_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep alpha; it reuses the validation-selected alpha from `exp0010`.
- Threshold-based F1 on test uses frozen thresholds from the validation mixing artifact, not thresholds retuned on test.
- `test_mixed_probabilities.npy` stores held-out mixed probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/09_evaluate_probability_mixing_test.py and the report /workspace/experiments/exp0024__source_probability_mixing_test_evaluation__nih_cxr14_exp0022_test_e100_p4_expanded_retrieval_refined/recreation_report.md to recreate the held-out test probability-mixing stage that combines /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 with /workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined. Apply the frozen validation-selected alpha from /workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined, reuse the validation thresholds from exp0010 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_mixed_probabilities.npy artifacts.
```
