# Probability-Mixing Test Evaluation

The held-out test probability-mixing evaluation for the current source stage uses the frozen validation-selected alpha:

- selection root: `/workspace/experiments/exp0016__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4`
- memory test root: `/workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4`
- alpha: `0.7`
- mixed test macro AUROC: `0.775239`
- mixed test macro average precision: `0.159950`
- mixed test macro ECE: `0.227412`
- mixed test macro F1 @ 0.5: `0.211526`
- mixed test macro F1 @ frozen val thresholds: `0.220846`
- delta vs frozen baseline macro AUROC: `0.000595`
- delta vs frozen baseline macro average precision: `-0.000466`
- delta vs frozen baseline macro ECE: `-0.097811`
- delta vs frozen baseline macro F1 @ 0.5: `0.032386`
- delta vs frozen baseline macro F1 @ frozen val thresholds: `-0.000159`
