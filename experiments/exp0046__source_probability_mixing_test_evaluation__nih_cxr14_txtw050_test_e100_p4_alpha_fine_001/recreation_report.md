# Source Probability-Mixing Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test probability-mixing experiment stored at:

`/workspace/experiments/exp0046__source_probability_mixing_test_evaluation__nih_cxr14_txtw050_test_e100_p4_alpha_fine_001`

The producing script is:

`/workspace/scripts/09_evaluate_probability_mixing_test.py`

Script SHA-256:

`088a52f983d1b9d54d00c085087cbd4448300e2d1b9d8fa7f4ffa188f836c41a`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0046__source_probability_mixing_test_evaluation__nih_cxr14_txtw050_test_e100_p4_alpha_fine_001`
- Experiment id: `exp0046`
- Operation label: `source_probability_mixing_test_evaluation`
- Memory-test root: `/workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4`
- Validation mixing-selection root: `/workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001`
- Baseline experiment: `/workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4`
- Query embedding root: `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050`
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
python /workspace/scripts/09_evaluate_probability_mixing_test.py --memory-test-root /workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4 --mixing-selection-root /workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001 --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --batch-size 2048 --seed 3407 --experiment-name exp0046__source_probability_mixing_test_evaluation__nih_cxr14_txtw050_test_e100_p4_alpha_fine_001 --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/09_evaluate_probability_mixing_test.py --memory-test-root /workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4 --mixing-selection-root /workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001 --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --batch-size 2048 --seed 3407 --experiment-name source_probability_mixing_test_evaluation__nih_cxr14_txtw050_test_e100_p4_alpha_fine_001
```

## Preconditions

- The memory-only test evaluation must already exist at `/workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4`.
- The validation probability-mixing selection must already exist at `/workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001`.
- The baseline experiment must already exist at `/workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4`.
- The query embeddings must already exist at `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050/test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.
- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected alpha.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4/best.ckpt`
- Memory test probabilities: `/workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4/test_probabilities.npy`
- Validation selection root: `/workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001`
- Query embedding split: `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050/test`
- Test rows: `22,330`

## Applied Configuration

- Frozen `alpha`: `0.7`
- Threshold source: `/workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001/best_val_metrics.json`

## Test Metrics

- Mixed test macro AUROC: `0.774877`
- Mixed test macro average precision: `0.160280`
- Mixed test macro ECE: `0.237688`
- Mixed test macro F1 @ 0.5: `0.208341`
- Mixed test macro F1 @ frozen val thresholds: `0.219805`

## Baseline Comparison

- Frozen baseline test macro AUROC: `0.774628`
- Frozen baseline test macro average precision: `0.160827`
- Mixed minus baseline macro AUROC: `0.000249`
- Mixed minus baseline macro average precision: `-0.000547`
- Mixed minus baseline macro ECE: `-0.084458`
- Mixed minus baseline macro F1 @ 0.5: `0.028187`
- Mixed minus baseline macro F1 @ frozen val thresholds: `-0.001702`
- Baseline reconstruction matches archived exp0006 test metrics within 5e-4: `true`
- Baseline reconstruction max absolute metric delta: `0.000193348570`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `applied_config.json`
- `test_metrics.json`
- `test_mixed_probabilities.npy`
- `probability_mixing_test_summary.md`

## Output Sizes

- experiment_meta.json: `157.45K`
- applied_config.json: `1.95K`
- test_metrics.json: `9.04K`
- test_mixed_probabilities.npy: `1.19M`
- probability_mixing_test_summary.md: `942B`
- Total output size: `1.36M`

## Final Artifact SHA-256

```text
45d7cc4589324de8b973d2f1a997a727bde089662dca0b4f86a26cbc8b625a15  /workspace/experiments/exp0046__source_probability_mixing_test_evaluation__nih_cxr14_txtw050_test_e100_p4_alpha_fine_001/experiment_meta.json
93712b2468c97880800341aa5852fab7245766a05c103f6f10430bbae09c1060  /workspace/experiments/exp0046__source_probability_mixing_test_evaluation__nih_cxr14_txtw050_test_e100_p4_alpha_fine_001/applied_config.json
13f95f4a28ebdb58f6835df6d2e690a6efb9dedfe1eb8b24f0e9fd951be5cba2  /workspace/experiments/exp0046__source_probability_mixing_test_evaluation__nih_cxr14_txtw050_test_e100_p4_alpha_fine_001/test_metrics.json
509ae2c9162538313faa4b30fa0adfc7f79f03ac3a724b45588d63881613242d  /workspace/experiments/exp0046__source_probability_mixing_test_evaluation__nih_cxr14_txtw050_test_e100_p4_alpha_fine_001/test_mixed_probabilities.npy
5b325864f34a84c4e6639e5bb7d28d1a2be2d450f08f0bbac2e7c76e9a13138e  /workspace/experiments/exp0046__source_probability_mixing_test_evaluation__nih_cxr14_txtw050_test_e100_p4_alpha_fine_001/probability_mixing_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep alpha; it reuses the validation-selected alpha from `exp0010`.
- Threshold-based F1 on test uses frozen thresholds from the validation mixing artifact, not thresholds retuned on test.
- `test_mixed_probabilities.npy` stores held-out mixed probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/09_evaluate_probability_mixing_test.py and the report /workspace/experiments/exp0046__source_probability_mixing_test_evaluation__nih_cxr14_txtw050_test_e100_p4_alpha_fine_001/recreation_report.md to recreate the held-out test probability-mixing stage that combines /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4 with /workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4. Apply the frozen validation-selected alpha from /workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001, reuse the validation thresholds from exp0010 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_mixed_probabilities.npy artifacts.
```
