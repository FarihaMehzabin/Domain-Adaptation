# Probability-Mixing Test Evaluation

The held-out test probability-mixing evaluation for the current source stage uses the frozen validation-selected alpha:

- selection root: `/workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val`
- memory test root: `/workspace/experiments/exp0011__source_memory_only_test_evaluation__nih_cxr14_exp0009_test`
- alpha: `0.7`
- mixed test macro AUROC: `0.768586`
- mixed test macro average precision: `0.153391`
- mixed test macro ECE: `0.242982`
- mixed test macro F1 @ 0.5: `0.204582`
- mixed test macro F1 @ frozen val thresholds: `0.210987`
- delta vs frozen baseline macro AUROC: `0.000651`
- delta vs frozen baseline macro average precision: `0.001154`
- delta vs frozen baseline macro ECE: `-0.104474`
- delta vs frozen baseline macro F1 @ 0.5: `0.028907`
- delta vs frozen baseline macro F1 @ frozen val thresholds: `0.001397`
