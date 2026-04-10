# Probability-Mixing Test Evaluation

The held-out test probability-mixing evaluation for the current source stage uses the frozen validation-selected alpha:

- selection root: `/workspace/experiments/exp0041__source_probability_mixing_evaluation__gated_hybrid_long_e50_mix_val`
- memory test root: `/workspace/experiments/exp0042__source_memory_only_test_evaluation__gated_hybrid_long_e50_memory_test`
- alpha: `0.6`
- mixed test macro AUROC: `0.759226`
- mixed test macro average precision: `0.143809`
- mixed test macro ECE: `0.194621`
- mixed test macro F1 @ 0.5: `0.193147`
- mixed test macro F1 @ frozen val thresholds: `0.203632`
- delta vs frozen baseline macro AUROC: `0.000212`
- delta vs frozen baseline macro average precision: `0.001822`
- delta vs frozen baseline macro ECE: `-0.129092`
- delta vs frozen baseline macro F1 @ 0.5: `0.024308`
- delta vs frozen baseline macro F1 @ frozen val thresholds: `0.000239`
