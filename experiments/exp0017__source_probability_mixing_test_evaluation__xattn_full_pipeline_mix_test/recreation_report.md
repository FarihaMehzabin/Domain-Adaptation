# Source Probability-Mixing Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test probability-mixing experiment stored at:

`/workspace/experiments/exp0017__source_probability_mixing_test_evaluation__xattn_full_pipeline_mix_test`

The producing script is:

`/workspace/scripts/09_evaluate_probability_mixing_test.py`

Script SHA-256:

`14c47f1a9784830bc127040d006f58f593435bf70ac03c9f444a6dad6f238ec4`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0017__source_probability_mixing_test_evaluation__xattn_full_pipeline_mix_test`
- Experiment id: `exp0017`
- Operation label: `source_probability_mixing_test_evaluation`
- Memory-test root: `/workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test`
- Validation mixing-selection root: `/workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val`
- Baseline experiment: `/workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096`
- Query embedding root: `/workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160`
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
python /workspace/scripts/09_evaluate_probability_mixing_test.py --memory-test-root /workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test --mixing-selection-root /workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val --baseline-experiment-dir /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096 --query-embedding-root /workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --batch-size 2048 --seed 3407 --experiment-name exp0017__source_probability_mixing_test_evaluation__xattn_full_pipeline_mix_test --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/09_evaluate_probability_mixing_test.py --memory-test-root /workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test --mixing-selection-root /workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val --baseline-experiment-dir /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096 --query-embedding-root /workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --batch-size 2048 --seed 3407 --experiment-name source_probability_mixing_test_evaluation__xattn_full_pipeline_mix_test
```

## Preconditions

- The memory-only test evaluation must already exist at `/workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test`.
- The validation probability-mixing selection must already exist at `/workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val`.
- The baseline experiment must already exist at `/workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096`.
- The query embeddings must already exist at `/workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160/test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.
- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected alpha.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096/best.ckpt`
- Memory test probabilities: `/workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test/test_probabilities.npy`
- Validation selection root: `/workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val`
- Query embedding split: `/workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160/test`
- Test rows: `22,330`

## Applied Configuration

- Frozen `alpha`: `0.6`
- Threshold source: `/workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val/best_val_metrics.json`

## Test Metrics

- Mixed test macro AUROC: `0.729254`
- Mixed test macro average precision: `0.114845`
- Mixed test macro ECE: `0.221457`
- Mixed test macro F1 @ 0.5: `0.140398`
- Mixed test macro F1 @ frozen val thresholds: `0.174851`

## Baseline Comparison

- Frozen baseline test macro AUROC: `0.727784`
- Frozen baseline test macro average precision: `0.111898`
- Mixed minus baseline macro AUROC: `0.001470`
- Mixed minus baseline macro average precision: `0.002947`
- Mixed minus baseline macro ECE: `-0.147151`
- Mixed minus baseline macro F1 @ 0.5: `-0.014706`
- Mixed minus baseline macro F1 @ frozen val thresholds: `0.001830`
- Baseline reconstruction matches archived exp0004 test metrics within 5e-4: `false`
- Baseline reconstruction max absolute metric delta: `0.001797422122`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `applied_config.json`
- `test_metrics.json`
- `test_mixed_probabilities.npy`
- `probability_mixing_test_summary.md`

## Output Sizes

- experiment_meta.json: `144.48K`
- applied_config.json: `1.92K`
- test_metrics.json: `9.03K`
- test_mixed_probabilities.npy: `1.19M`
- probability_mixing_test_summary.md: `924B`
- Total output size: `1.35M`

## Final Artifact SHA-256

```text
43d6f863d3a0437fd6d0ede474e84571dd24c922e95b4a55337c398ec15f4ada  /workspace/experiments/exp0017__source_probability_mixing_test_evaluation__xattn_full_pipeline_mix_test/experiment_meta.json
63e815db51cf41f438a717af46505e0c8f1acb1223c1b2550550d0ce73baa903  /workspace/experiments/exp0017__source_probability_mixing_test_evaluation__xattn_full_pipeline_mix_test/applied_config.json
af24f955cbcc3719d40d7834e53fb3d315d6f49961a1c1a6b7e320d3fd27f436  /workspace/experiments/exp0017__source_probability_mixing_test_evaluation__xattn_full_pipeline_mix_test/test_metrics.json
f19b34b334078e1a42f2607a25990b4f5f96d326860b5eb60363b43c38dc2552  /workspace/experiments/exp0017__source_probability_mixing_test_evaluation__xattn_full_pipeline_mix_test/test_mixed_probabilities.npy
5390dc5c8338baa5cff0673659de5c2772bc462b8cfdf74d7157b43ac63bb02d  /workspace/experiments/exp0017__source_probability_mixing_test_evaluation__xattn_full_pipeline_mix_test/probability_mixing_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep alpha; it reuses the validation-selected alpha from `exp0007`.
- Threshold-based F1 on test uses frozen thresholds from the validation mixing artifact, not thresholds retuned on test.
- `test_mixed_probabilities.npy` stores held-out mixed probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/09_evaluate_probability_mixing_test.py and the report /workspace/experiments/exp0017__source_probability_mixing_test_evaluation__xattn_full_pipeline_mix_test/recreation_report.md to recreate the held-out test probability-mixing stage that combines /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096 with /workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test. Apply the frozen validation-selected alpha from /workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val, reuse the validation thresholds from exp0007 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_mixed_probabilities.npy artifacts.
```
