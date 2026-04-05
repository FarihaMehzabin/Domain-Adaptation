# Probability-Mixing Test Evaluation

The held-out test probability-mixing evaluation for the current source stage uses the frozen validation-selected alpha:

- selection root: `/workspace/experiments/exp0025__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4_alpha_fine_001`
- memory test root: `/workspace/experiments/exp0017__source_memory_only_test_evaluation__nih_cxr14_exp0015_test_e100_p4`
- alpha: `0.74`
- mixed test macro AUROC: `0.775224`
- mixed test macro average precision: `0.160043`
- mixed test macro ECE: `0.240454`
- mixed test macro F1 @ 0.5: `0.209309`
- mixed test macro F1 @ frozen val thresholds: `0.220639`
- delta vs frozen baseline macro AUROC: `0.000581`
- delta vs frozen baseline macro average precision: `-0.000373`
- delta vs frozen baseline macro ECE: `-0.084769`
- delta vs frozen baseline macro F1 @ 0.5: `0.030169`
- delta vs frozen baseline macro F1 @ frozen val thresholds: `-0.000366`
