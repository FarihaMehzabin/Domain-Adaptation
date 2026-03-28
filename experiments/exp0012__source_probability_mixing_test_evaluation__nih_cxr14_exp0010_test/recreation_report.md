# Source Probability-Mixing Test Evaluation Recreation Report

## Scope

This report documents how to recreate the held-out test probability-mixing experiment stored at:

`/workspace/experiments/exp0012__source_probability_mixing_test_evaluation__nih_cxr14_exp0010_test`

The producing script is:

`/workspace/scripts/09_evaluate_probability_mixing_test.py`

Script SHA-256:

`9ea4e4811bfcfe1f242c9af7c883d04f134d0d4f657f5712bb673da3d684c183`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0012__source_probability_mixing_test_evaluation__nih_cxr14_exp0010_test`
- Experiment id: `exp0012`
- Operation label: `source_probability_mixing_test_evaluation`
- Memory-test root: `/workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test`
- Validation mixing-selection root: `/workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val`
- Baseline experiment: `/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear`
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
- Platform: `Linux-5.15.0-102-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/09_evaluate_probability_mixing_test.py --memory-test-root /workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test --mixing-selection-root /workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val --baseline-experiment-dir /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --batch-size 2048 --seed 3407 --experiment-name exp0012__source_probability_mixing_test_evaluation__nih_cxr14_exp0010_test --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/09_evaluate_probability_mixing_test.py --memory-test-root /workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test --mixing-selection-root /workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val --baseline-experiment-dir /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split test --batch-size 2048 --seed 3407 --experiment-name source_probability_mixing_test_evaluation__nih_cxr14_exp0010_test
```

## Preconditions

- The memory-only test evaluation must already exist at `/workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test`.
- The validation probability-mixing selection must already exist at `/workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val`.
- The baseline experiment must already exist at `/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear`.
- The query embeddings must already exist at `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/test`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.
- No hyperparameters are tuned on test; this stage only applies the frozen validation-selected alpha.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear/best.ckpt`
- Memory test probabilities: `/workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test/test_probabilities.npy`
- Validation selection root: `/workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val`
- Query embedding split: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/test`
- Test rows: `22,330`

## Applied Configuration

- Frozen `alpha`: `0.7`
- Threshold source: `/workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val/best_val_metrics.json`

## Test Metrics

- Mixed test macro AUROC: `0.768586`
- Mixed test macro average precision: `0.153391`
- Mixed test macro ECE: `0.242982`
- Mixed test macro F1 @ 0.5: `0.204582`
- Mixed test macro F1 @ frozen val thresholds: `0.210987`

## Baseline Comparison

- Frozen baseline test macro AUROC: `0.767935`
- Frozen baseline test macro average precision: `0.152238`
- Mixed minus baseline macro AUROC: `0.000651`
- Mixed minus baseline macro average precision: `0.001154`
- Mixed minus baseline macro ECE: `-0.104474`
- Mixed minus baseline macro F1 @ 0.5: `0.028907`
- Mixed minus baseline macro F1 @ frozen val thresholds: `0.001397`
- Baseline reconstruction matches archived exp0006 test metrics within 5e-4: `true`
- Baseline reconstruction max absolute metric delta: `0.000494120703`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `applied_config.json`
- `test_metrics.json`
- `test_mixed_probabilities.npy`
- `probability_mixing_test_summary.md`

## Output Sizes

- experiment_meta.json: `148.58K`
- applied_config.json: `1.92K`
- test_metrics.json: `9.01K`
- test_mixed_probabilities.npy: `1.19M`
- probability_mixing_test_summary.md: `927B`
- Total output size: `1.35M`

## Final Artifact SHA-256

```text
a4a62191ed4e16624ab189c98dadb2d34125d3b49f8b5fe9528d57ae58b9d330  /workspace/experiments/exp0012__source_probability_mixing_test_evaluation__nih_cxr14_exp0010_test/experiment_meta.json
6828a51873e759b173dc08a1ec5d03161abee13e3ac5e35bbfb7f46d895bd426  /workspace/experiments/exp0012__source_probability_mixing_test_evaluation__nih_cxr14_exp0010_test/applied_config.json
e9e2489cbc4ff43072c74bd38cca8a447431a70405971a08ea875e95fcec3b3e  /workspace/experiments/exp0012__source_probability_mixing_test_evaluation__nih_cxr14_exp0010_test/test_metrics.json
58f68053500537a4ec70ebb79c0f39737420dc5b91772f329bede6550b6f3a94  /workspace/experiments/exp0012__source_probability_mixing_test_evaluation__nih_cxr14_exp0010_test/test_mixed_probabilities.npy
12eaeefed60d0ec69727f3135e3c564dc5e12e696bdd22e5a360325d6eeb838d  /workspace/experiments/exp0012__source_probability_mixing_test_evaluation__nih_cxr14_exp0010_test/probability_mixing_test_summary.md
```

## Important Reproduction Notes

- This test stage does not sweep alpha; it reuses the validation-selected alpha from `exp0010`.
- Threshold-based F1 on test uses frozen thresholds from the validation mixing artifact, not thresholds retuned on test.
- `test_mixed_probabilities.npy` stores held-out mixed probabilities in test row order and is small enough for plain Git.

## Agent Handoff Text

```text
Use /workspace/scripts/09_evaluate_probability_mixing_test.py and the report /workspace/experiments/exp0012__source_probability_mixing_test_evaluation__nih_cxr14_exp0010_test/recreation_report.md to recreate the held-out test probability-mixing stage that combines /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear with /workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test. Apply the frozen validation-selected alpha from /workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val, reuse the validation thresholds from exp0010 for threshold-based F1 reporting, and verify the saved applied_config.json, test_metrics.json, and test_mixed_probabilities.npy artifacts.
```
