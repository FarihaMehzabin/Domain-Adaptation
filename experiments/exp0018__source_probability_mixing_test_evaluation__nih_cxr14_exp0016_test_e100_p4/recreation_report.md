# Source Probability-Mixing Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test probability-mixing experiment stored at:

`/workspace/experiments/exp0018__source_probability_mixing_test_evaluation__nih_cxr14_exp0016_test_e100_p4`

The producing script is:

`/workspace/scripts/09_evaluate_probability_mixing_test.py`

Script SHA-256:

`9ea4e4811bfcfe1f242c9af7c883d04f134d0d4f657f5712bb673da3d684c183`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0018__source_probability_mixing_test_evaluation__nih_cxr14_exp0016_test_e100_p4`
- Experiment id: `exp0018`
- Operation label: `source_probability_mixing_test_evaluation`
- Memory-test root: `/workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4`
- Validation mixing-selection root: `/workspace/experiments/exp0016__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4`
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
python /workspace/scripts/09_evaluate_probability_mixing_test.py --memory-test-root /workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4 --mixing-selection-root /workspace/experiments/exp0016__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4 --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --batch-size 2048 --seed 3407 --experiment-name exp0018__source_probability_mixing_test_evaluation__nih_cxr14_exp0016_test_e100_p4 --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/09_evaluate_probability_mixing_test.py --memory-test-root /workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4 --mixing-selection-root /workspace/experiments/exp0016__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4 --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --batch-size 2048 --seed 3407 --experiment-name source_probability_mixing_test_evaluation__nih_cxr14_exp0016_test_e100_p4
```

## Preconditions

- The memory-only test evaluation must already exist at `/workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4`.
- The validation probability-mixing selection must already exist at `/workspace/experiments/exp0016__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4`.
- The baseline experiment must already exist at `/workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`.
- The query embeddings must already exist at `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.
- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected alpha.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4/best.ckpt`
- Memory test probabilities: `/workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4/test_probabilities.npy`
- Validation selection root: `/workspace/experiments/exp0016__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4`
- Query embedding split: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/test`
- Test rows: `22,330`

## Applied Configuration

- Frozen `alpha`: `0.7`
- Threshold source: `/workspace/experiments/exp0016__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4/best_val_metrics.json`

## Test Metrics

- Mixed test macro AUROC: `0.775239`
- Mixed test macro average precision: `0.159950`
- Mixed test macro ECE: `0.227412`
- Mixed test macro F1 @ 0.5: `0.211526`
- Mixed test macro F1 @ frozen val thresholds: `0.220846`

## Baseline Comparison

- Frozen baseline test macro AUROC: `0.774644`
- Frozen baseline test macro average precision: `0.160416`
- Mixed minus baseline macro AUROC: `0.000595`
- Mixed minus baseline macro average precision: `-0.000466`
- Mixed minus baseline macro ECE: `-0.097811`
- Mixed minus baseline macro F1 @ 0.5: `0.032386`
- Mixed minus baseline macro F1 @ frozen val thresholds: `-0.000159`
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

- experiment_meta.json: `152.31K`
- applied_config.json: `1.92K`
- test_metrics.json: `9.05K`
- test_mixed_probabilities.npy: `1.19M`
- probability_mixing_test_summary.md: `926B`
- Total output size: `1.35M`

## Final Artifact SHA-256

```text
35c8fd4eca3efdb4aada51a14944502153eaf2f6f764614b8ea780710867e924  /workspace/experiments/exp0018__source_probability_mixing_test_evaluation__nih_cxr14_exp0016_test_e100_p4/experiment_meta.json
cd41f55f62d9b3258c8c25084827e6700da8cb4a0507fe99aefe4a81364cd9f9  /workspace/experiments/exp0018__source_probability_mixing_test_evaluation__nih_cxr14_exp0016_test_e100_p4/applied_config.json
e1d7c538f7e0c8a4c5be630d64d804e67f0bbdf8a069ee1a596bc0df8257d737  /workspace/experiments/exp0018__source_probability_mixing_test_evaluation__nih_cxr14_exp0016_test_e100_p4/test_metrics.json
172b2d5da174ad3a2588bf75b3b91de08747fb23a3223ade236be566c4f29034  /workspace/experiments/exp0018__source_probability_mixing_test_evaluation__nih_cxr14_exp0016_test_e100_p4/test_mixed_probabilities.npy
168c7ff2ed66b2a80bfbf972e7b028e58b548c039c333b2b92eb859414f12833  /workspace/experiments/exp0018__source_probability_mixing_test_evaluation__nih_cxr14_exp0016_test_e100_p4/probability_mixing_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep alpha; it reuses the validation-selected alpha from `exp0010`.
- Threshold-based F1 on test uses frozen thresholds from the validation mixing artifact, not thresholds retuned on test.
- `test_mixed_probabilities.npy` stores held-out mixed probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/09_evaluate_probability_mixing_test.py and the report /workspace/experiments/exp0018__source_probability_mixing_test_evaluation__nih_cxr14_exp0016_test_e100_p4/recreation_report.md to recreate the held-out test probability-mixing stage that combines /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 with /workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4. Apply the frozen validation-selected alpha from /workspace/experiments/exp0016__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4, reuse the validation thresholds from exp0010 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_mixed_probabilities.npy artifacts.
```
