# Source Probability-Mixing Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test probability-mixing experiment stored at:

`/workspace/experiments/exp0025__source_probability_mixing_test_evaluation__xattn_long_e50_mix_test`

The producing script is:

`/workspace/scripts/09_evaluate_probability_mixing_test.py`

Script SHA-256:

`14c47f1a9784830bc127040d006f58f593435bf70ac03c9f444a6dad6f238ec4`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0025__source_probability_mixing_test_evaluation__xattn_long_e50_mix_test`
- Experiment id: `exp0025`
- Operation label: `source_probability_mixing_test_evaluation`
- Memory-test root: `/workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test`
- Validation mixing-selection root: `/workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val`
- Baseline experiment: `/workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096`
- Query embedding root: `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160`
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
- Platform: `Linux-6.8.0-65-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/09_evaluate_probability_mixing_test.py --memory-test-root /workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test --mixing-selection-root /workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val --baseline-experiment-dir /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096 --query-embedding-root /workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --batch-size 2048 --seed 3407 --experiment-name exp0025__source_probability_mixing_test_evaluation__xattn_long_e50_mix_test --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/09_evaluate_probability_mixing_test.py --memory-test-root /workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test --mixing-selection-root /workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val --baseline-experiment-dir /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096 --query-embedding-root /workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --batch-size 2048 --seed 3407 --experiment-name source_probability_mixing_test_evaluation__xattn_long_e50_mix_test
```

## Preconditions

- The memory-only test evaluation must already exist at `/workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test`.
- The validation probability-mixing selection must already exist at `/workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val`.
- The baseline experiment must already exist at `/workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096`.
- The query embeddings must already exist at `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160/test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.
- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected alpha.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096/best.ckpt`
- Memory test probabilities: `/workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test/test_probabilities.npy`
- Validation selection root: `/workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val`
- Query embedding split: `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160/test`
- Test rows: `22,330`

## Applied Configuration

- Frozen `alpha`: `0.8`
- Threshold source: `/workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val/best_val_metrics.json`

## Test Metrics

- Mixed test macro AUROC: `0.759259`
- Mixed test macro average precision: `0.139147`
- Mixed test macro ECE: `0.255039`
- Mixed test macro F1 @ 0.5: `0.186855`
- Mixed test macro F1 @ frozen val thresholds: `0.198050`

## Baseline Comparison

- Frozen baseline test macro AUROC: `0.759053`
- Frozen baseline test macro average precision: `0.137528`
- Mixed minus baseline macro AUROC: `0.000206`
- Mixed minus baseline macro average precision: `0.001619`
- Mixed minus baseline macro ECE: `-0.063432`
- Mixed minus baseline macro F1 @ 0.5: `0.018460`
- Mixed minus baseline macro F1 @ frozen val thresholds: `-0.001877`
- Baseline reconstruction matches archived exp0004 test metrics within 5e-4: `false`
- Baseline reconstruction max absolute metric delta: `0.001005711628`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `applied_config.json`
- `test_metrics.json`
- `test_mixed_probabilities.npy`
- `probability_mixing_test_summary.md`

## Output Sizes

- experiment_meta.json: `143.86K`
- applied_config.json: `1.89K`
- test_metrics.json: `9.03K`
- test_mixed_probabilities.npy: `1.19M`
- probability_mixing_test_summary.md: `914B`
- Total output size: `1.34M`

## Final Artifact SHA-256

```text
3007f92e48ce9ad738b4bf950a95957db5fc86eaa28abd7265cbd02c373f4cc1  /workspace/experiments/exp0025__source_probability_mixing_test_evaluation__xattn_long_e50_mix_test/experiment_meta.json
ff27434300a56c926fdf21d245e9b393155c0cbb3c3c046b6b44e28325903ed3  /workspace/experiments/exp0025__source_probability_mixing_test_evaluation__xattn_long_e50_mix_test/applied_config.json
e2ce1c60939b19eddadc3d45972b555f670415160e1611138fe202b6d9050e3b  /workspace/experiments/exp0025__source_probability_mixing_test_evaluation__xattn_long_e50_mix_test/test_metrics.json
9c705c8294582861a7bda108da07e679a50223e1ee4771cff00a2118149fccbb  /workspace/experiments/exp0025__source_probability_mixing_test_evaluation__xattn_long_e50_mix_test/test_mixed_probabilities.npy
9232823e63a403264eca88129fb67c80d05c470a0c72c9fea715443ab727190d  /workspace/experiments/exp0025__source_probability_mixing_test_evaluation__xattn_long_e50_mix_test/probability_mixing_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep alpha; it reuses the validation-selected alpha from `exp0007`.
- Threshold-based F1 on test uses frozen thresholds from the validation mixing artifact, not thresholds retuned on test.
- `test_mixed_probabilities.npy` stores held-out mixed probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/09_evaluate_probability_mixing_test.py and the report /workspace/experiments/exp0025__source_probability_mixing_test_evaluation__xattn_long_e50_mix_test/recreation_report.md to recreate the held-out test probability-mixing stage that combines /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096 with /workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test. Apply the frozen validation-selected alpha from /workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val, reuse the validation thresholds from exp0007 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_mixed_probabilities.npy artifacts.
```
