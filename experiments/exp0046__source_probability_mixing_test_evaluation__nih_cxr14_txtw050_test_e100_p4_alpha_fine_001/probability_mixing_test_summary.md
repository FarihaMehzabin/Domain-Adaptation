# Probability-Mixing Test Evaluation

The held-out test probability-mixing evaluation for the current source stage uses the frozen validation-selected alpha:

- selection root: `/workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001`
- memory test root: `/workspace/experiments/exp0045__source_memory_only_test_evaluation__nih_cxr14_txtw050_test_e100_p4`
- alpha: `0.74`
- mixed test macro AUROC: `0.774877`
- mixed test macro average precision: `0.160280`
- mixed test macro ECE: `0.237688`
- mixed test macro F1 @ 0.5: `0.208341`
- mixed test macro F1 @ frozen val thresholds: `0.219805`
- delta vs frozen baseline macro AUROC: `0.000249`
- delta vs frozen baseline macro average precision: `-0.000547`
- delta vs frozen baseline macro ECE: `-0.084458`
- delta vs frozen baseline macro F1 @ 0.5: `0.028187`
- delta vs frozen baseline macro F1 @ frozen val thresholds: `-0.001702`
