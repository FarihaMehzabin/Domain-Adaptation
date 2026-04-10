# Probability-Mixing Test Evaluation

The held-out test probability-mixing evaluation for the current source stage uses the frozen validation-selected alpha:

- selection root: `/workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val`
- memory test root: `/workspace/experiments/exp0032__source_memory_only_test_evaluation__xattn_long_e50_hybrid_memory_test`
- alpha: `0.6`
- mixed test macro AUROC: `0.772098`
- mixed test macro average precision: `0.154132`
- mixed test macro ECE: `0.195644`
- mixed test macro F1 @ 0.5: `0.200968`
- mixed test macro F1 @ frozen val thresholds: `0.214857`
- delta vs frozen baseline macro AUROC: `0.001269`
- delta vs frozen baseline macro average precision: `0.002617`
- delta vs frozen baseline macro ECE: `-0.129172`
- delta vs frozen baseline macro F1 @ 0.5: `0.027706`
- delta vs frozen baseline macro F1 @ frozen val thresholds: `0.007387`
