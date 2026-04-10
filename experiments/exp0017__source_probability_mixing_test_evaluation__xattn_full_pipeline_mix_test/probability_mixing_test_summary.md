# Probability-Mixing Test Evaluation

The held-out test probability-mixing evaluation for the current source stage uses the frozen validation-selected alpha:

- selection root: `/workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val`
- memory test root: `/workspace/experiments/exp0016__source_memory_only_test_evaluation__xattn_full_pipeline_memory_test`
- alpha: `0.6`
- mixed test macro AUROC: `0.729254`
- mixed test macro average precision: `0.114845`
- mixed test macro ECE: `0.221457`
- mixed test macro F1 @ 0.5: `0.140398`
- mixed test macro F1 @ frozen val thresholds: `0.174851`
- delta vs frozen baseline macro AUROC: `0.001470`
- delta vs frozen baseline macro average precision: `0.002947`
- delta vs frozen baseline macro ECE: `-0.147151`
- delta vs frozen baseline macro F1 @ 0.5: `-0.014706`
- delta vs frozen baseline macro F1 @ frozen val thresholds: `0.001830`
