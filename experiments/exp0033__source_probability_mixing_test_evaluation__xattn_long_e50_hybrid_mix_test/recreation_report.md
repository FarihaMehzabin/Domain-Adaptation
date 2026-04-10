# Source Probability-Mixing Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test probability-mixing experiment stored at:

`/workspace/experiments/exp0033__source_probability_mixing_test_evaluation__xattn_long_e50_hybrid_mix_test`

The producing script is:

`/workspace/scripts/09_evaluate_probability_mixing_test.py`

Script SHA-256:

`14c47f1a9784830bc127040d006f58f593435bf70ac03c9f444a6dad6f238ec4`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0033__source_probability_mixing_test_evaluation__xattn_long_e50_hybrid_mix_test`
- Experiment id: `exp0033`
- Operation label: `source_probability_mixing_test_evaluation`
- Memory-test root: `/workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test`
- Validation mixing-selection root: `/workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val`
- Baseline experiment: `/workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096`
- Query embedding root: `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat`
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
python /workspace/scripts/09_evaluate_probability_mixing_test.py --memory-test-root /workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test --mixing-selection-root /workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096 --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --batch-size 2048 --seed 3407 --experiment-name exp0033__source_probability_mixing_test_evaluation__xattn_long_e50_hybrid_mix_test --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/09_evaluate_probability_mixing_test.py --memory-test-root /workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test --mixing-selection-root /workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096 --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --batch-size 2048 --seed 3407 --experiment-name source_probability_mixing_test_evaluation__xattn_long_e50_hybrid_mix_test
```

## Preconditions

- The memory-only test evaluation must already exist at `/workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test`.
- The validation probability-mixing selection must already exist at `/workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val`.
- The baseline experiment must already exist at `/workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096`.
- The query embeddings must already exist at `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat/test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.
- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected alpha.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096/best.ckpt`
- Memory test probabilities: `/workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test/test_probabilities.npy`
- Validation selection root: `/workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val`
- Query embedding split: `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat/test`
- Test rows: `22,330`

## Applied Configuration

- Frozen `alpha`: `0.6`
- Threshold source: `/workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val/best_val_metrics.json`

## Test Metrics

- Mixed test macro AUROC: `0.772098`
- Mixed test macro average precision: `0.154132`
- Mixed test macro ECE: `0.195644`
- Mixed test macro F1 @ 0.5: `0.200968`
- Mixed test macro F1 @ frozen val thresholds: `0.214857`

## Baseline Comparison

- Frozen baseline test macro AUROC: `0.770829`
- Frozen baseline test macro average precision: `0.151515`
- Mixed minus baseline macro AUROC: `0.001269`
- Mixed minus baseline macro average precision: `0.002617`
- Mixed minus baseline macro ECE: `-0.129172`
- Mixed minus baseline macro F1 @ 0.5: `0.027706`
- Mixed minus baseline macro F1 @ frozen val thresholds: `0.007387`
- Baseline reconstruction matches archived exp0004 test metrics within 5e-4: `true`
- Baseline reconstruction max absolute metric delta: `0.000163402719`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `applied_config.json`
- `test_metrics.json`
- `test_mixed_probabilities.npy`
- `probability_mixing_test_summary.md`

## Output Sizes

- experiment_meta.json: `154.58K`
- applied_config.json: `1.93K`
- test_metrics.json: `9.03K`
- test_mixed_probabilities.npy: `1.19M`
- probability_mixing_test_summary.md: `927B`
- Total output size: `1.36M`

## Final Artifact SHA-256

```text
a30dc1fcc081dd083337e66f68536500f3d91ae32795f03ffe89a39784f981c6  /workspace/experiments/exp0033__source_probability_mixing_test_evaluation__xattn_long_e50_hybrid_mix_test/experiment_meta.json
f9e8c03c944bdaff1035d8ce2407b023f06be89a93bfa7535075e717834ce5b4  /workspace/experiments/exp0033__source_probability_mixing_test_evaluation__xattn_long_e50_hybrid_mix_test/applied_config.json
ede13773c3071ac8cd8884a0a53c893992f66cf3f92141a05a926592c0690ab9  /workspace/experiments/exp0033__source_probability_mixing_test_evaluation__xattn_long_e50_hybrid_mix_test/test_metrics.json
d2fc56a9fcda871f4cf4730311643ccc4fd4df2e4635265b77a05934b48b1382  /workspace/experiments/exp0033__source_probability_mixing_test_evaluation__xattn_long_e50_hybrid_mix_test/test_mixed_probabilities.npy
af2f7861567e758a316db6554058d6717cbaa0f3f3a2c04e83aa9c805ff4ee4a  /workspace/experiments/exp0033__source_probability_mixing_test_evaluation__xattn_long_e50_hybrid_mix_test/probability_mixing_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep alpha; it reuses the validation-selected alpha from `exp0007`.
- Threshold-based F1 on test uses frozen thresholds from the validation mixing artifact, not thresholds retuned on test.
- `test_mixed_probabilities.npy` stores held-out mixed probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/09_evaluate_probability_mixing_test.py and the report /workspace/experiments/exp0033__source_probability_mixing_test_evaluation__xattn_long_e50_hybrid_mix_test/recreation_report.md to recreate the held-out test probability-mixing stage that combines /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096 with /workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test. Apply the frozen validation-selected alpha from /workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val, reuse the validation thresholds from exp0007 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_mixed_probabilities.npy artifacts.
```
