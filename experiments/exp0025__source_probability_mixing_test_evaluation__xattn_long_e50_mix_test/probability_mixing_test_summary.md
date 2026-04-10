# Probability-Mixing Test Evaluation

The held-out test probability-mixing evaluation for the current source stage uses the frozen validation-selected alpha:

- selection root: `/workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val`
- memory test root: `/workspace/experiments/exp0024__source_memory_only_test_evaluation__xattn_long_e50_memory_test`
- alpha: `0.8`
- mixed test macro AUROC: `0.759259`
- mixed test macro average precision: `0.139147`
- mixed test macro ECE: `0.255039`
- mixed test macro F1 @ 0.5: `0.186855`
- mixed test macro F1 @ frozen val thresholds: `0.198050`
- delta vs frozen baseline macro AUROC: `0.000206`
- delta vs frozen baseline macro average precision: `0.001619`
- delta vs frozen baseline macro ECE: `-0.063432`
- delta vs frozen baseline macro F1 @ 0.5: `0.018460`
- delta vs frozen baseline macro F1 @ frozen val thresholds: `-0.001877`
