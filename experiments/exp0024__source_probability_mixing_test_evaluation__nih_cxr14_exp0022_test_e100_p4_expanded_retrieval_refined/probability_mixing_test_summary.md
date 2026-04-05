# Probability-Mixing Test Evaluation

The held-out test probability-mixing evaluation for the current source stage uses the frozen validation-selected alpha:

- selection root: `/workspace/experiments/exp0022__source_probability_mixing_evaluation__nih_cxr14_exp0021_val_e100_p4_expanded_retrieval_refined`
- memory test root: `/workspace/experiments/exp0023__source_memory_only_test_evaluation__nih_cxr14_exp0021_test_e100_p4_expanded_retrieval_refined`
- alpha: `0.7`
- mixed test macro AUROC: `0.775046`
- mixed test macro average precision: `0.159121`
- mixed test macro ECE: `0.227055`
- mixed test macro F1 @ 0.5: `0.211121`
- mixed test macro F1 @ frozen val thresholds: `0.221494`
- delta vs frozen baseline macro AUROC: `0.000402`
- delta vs frozen baseline macro average precision: `-0.001295`
- delta vs frozen baseline macro ECE: `-0.098168`
- delta vs frozen baseline macro F1 @ 0.5: `0.031980`
- delta vs frozen baseline macro F1 @ frozen val thresholds: `0.000489`
