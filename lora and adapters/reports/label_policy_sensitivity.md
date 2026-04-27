# MIMIC Label Policy Sensitivity Audit

## 1. Why This Audit Was Needed
Current MIMIC common5 target manifests were built with the U-zero plus blank-zero policy. That means existing target-side metrics implicitly treat both uncertain `-1` labels and blank/NaN labels as negatives. Because the completed MIMIC results were evaluated on those converted manifests, reported target-side performance can change when the label policy changes even if the predictions stay fixed.

## 2. Raw Label Missingness Summary
| Split | Label | Raw +1 | Raw 0 | Raw -1 | Raw blank |
| --- | --- | ---: | ---: | ---: | ---: |
| val | Atelectasis | 194 | 4 | 52 | 708 |
| val | Cardiomegaly | 205 | 63 | 29 | 661 |
| val | Consolidation | 52 | 27 | 24 | 855 |
| val | Edema | 131 | 98 | 60 | 669 |
| val | Effusion | 249 | 94 | 27 | 588 |
| test | Atelectasis | 170 | 7 | 32 | 387 |
| test | Cardiomegaly | 159 | 54 | 13 | 370 |
| test | Consolidation | 56 | 21 | 10 | 509 |
| test | Edema | 128 | 79 | 44 | 345 |
| test | Effusion | 219 | 70 | 27 | 280 |

Cross-check note:
- val raw labels were cross-checked against `manifests/mimic_target_query.csv` because `mimic_target_val.csv` is not present here.
- test raw labels were cross-checked against `manifests/mimic_target_test.csv`.

## 3. All-Blank Row Summary
| Split | Rows | All five blank | Percent | Rows with >=1 converted positive under current manifest |
| --- | ---: | ---: | ---: | ---: |
| val | 958 | 367 | 38.31% | 472 |
| test | 596 | 131 | 21.98% | 413 |

## 4. Policy Definitions
| Policy | Name | Mapping |
| --- | --- | --- |
| A | `current_uzero_blankzero` | 1->1/m1, 0->0/m1, -1->0/m1, blank->0/m1 |
| B | `uignore_blankzero` | 1->1/m1, 0->0/m1, -1->0/m0, blank->0/m1 |
| C | `uignore_blankignore` | 1->1/m1, 0->0/m1, -1->0/m0, blank->0/m0 |
| D | `uone_blankzero` | 1->1/m1, 0->0/m1, -1->1/m1, blank->0/m1 |

## 5. Metric Sensitivity Results
Validation split:
| Run | A macro AUROC | A macro AUPRC | B macro AUROC | B macro AUPRC | C macro AUROC | C macro AUPRC | D macro AUROC | D macro AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no adaptation | 0.6790 | 0.2633 | 0.6856 | 0.2771 | 0.6864 | 0.8565 | 0.6805 | 0.3132 |
| head_only_k5 | 0.6787 | 0.2631 | 0.6852 | 0.2767 | 0.6858 | 0.8568 | 0.6801 | 0.3129 |
| head_only_k20 | 0.6780 | 0.2628 | 0.6846 | 0.2763 | 0.6860 | 0.8559 | 0.6795 | 0.3123 |
| full_ft_k5 | 0.6972 | 0.2803 | 0.7037 | 0.2941 | 0.7226 | 0.8674 | 0.6961 | 0.3286 |
| full_ft_k20 | 0.6981 | 0.2830 | 0.7050 | 0.2976 | 0.7146 | 0.8703 | 0.6979 | 0.3329 |

Test split:
| Run | A macro AUROC | A macro AUPRC | B macro AUROC | B macro AUPRC | C macro AUROC | C macro AUPRC | D macro AUROC | D macro AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no adaptation | 0.6103 | 0.3312 | 0.6183 | 0.3541 | 0.6291 | 0.8405 | 0.6183 | 0.3854 |
| head_only_k5 | 0.6102 | 0.3311 | 0.6182 | 0.3539 | 0.6294 | 0.8403 | 0.6182 | 0.3853 |
| head_only_k20 | 0.6104 | 0.3319 | 0.6183 | 0.3547 | 0.6297 | 0.8408 | 0.6182 | 0.3861 |
| full_ft_k5 | 0.6239 | 0.3436 | 0.6314 | 0.3660 | 0.6566 | 0.8633 | 0.6299 | 0.3952 |
| full_ft_k20 | 0.6250 | 0.3456 | 0.6335 | 0.3702 | 0.6558 | 0.8622 | 0.6348 | 0.4012 |

Coverage and comparability:
| Split | Policy | Valid labels | Valid label fraction | Rows with no valid labels | Macro comparable | Micro comparable |
| --- | --- | ---: | ---: | ---: | --- | --- |
| val | Policy A | 5 / 5 | 100.00% | 0.00% | yes | yes |
| val | Policy B | 5 / 5 | 95.99% | 0.00% | yes | no |
| val | Policy C | 5 / 5 | 23.32% | 41.96% | yes | no |
| val | Policy D | 5 / 5 | 100.00% | 0.00% | yes | yes |
| test | Policy A | 5 / 5 | 100.00% | 0.00% | yes | yes |
| test | Policy B | 5 / 5 | 95.77% | 0.00% | yes | no |
| test | Policy C | 5 / 5 | 32.32% | 25.17% | yes | no |
| test | Policy D | 5 / 5 | 100.00% | 0.00% | yes | yes |

## 6. Ranking Stability
| Policy | Best method on test macro AUPRC | Test macro AUPRC | Test macro AUROC |
| --- | --- | ---: | ---: |
| Policy A | full_ft_k20 | 0.3456 | 0.6250 |
| Policy B | full_ft_k20 | 0.3702 | 0.6335 |
| Policy C | full_ft_k5 | 0.8633 | 0.6566 |
| Policy D | full_ft_k20 | 0.4012 | 0.6348 |

The best method does change across policies.
Ranking was evaluated on `test macro AUPRC`.

## 7. Recommended Primary Policy
Recommend `uignore_blankzero` (Policy B).

- Policy B is the conservative default here: it stops forcing uncertain `-1` targets to negative, preserves all five labels on both splits, and keeps the evaluation set substantially intact. Blank-zero remains an assumption and must be documented explicitly.
- This recommendation still treats blank/NaN as zero, so that assumption must be stated explicitly in the paper.
- Policy C drops blank labels entirely. In this dataset that creates rows with no evaluable common5 labels, so blank-ignore is not a safe default unless that loss of supervision is intentional.

## 8. Runs That Need Repeating
- no adaptation: reevaluate existing NIH->MIMIC predictions under the new policy before citing target-side metrics
- head_only_k5: rerun training and evaluation because support labels and model selection both used the current policy
- head_only_k20: rerun training and evaluation because support labels and model selection both used the current policy
- full_ft_k5: rerun training and evaluation because support labels and model selection both used the current policy
- full_ft_k20: rerun training and evaluation because support labels and model selection both used the current policy

## 9. Final Decision Needed
Before new training, accept policy Policy B (`uignore_blankzero`) or revise it.

## Appendices
- `full_ft_k20_seed2027` verification status: `VERIFIED_COMPLETE`
- generated sensitivity JSON: `reports/label_policy_sensitivity.json`
- generated sensitivity CSV: `reports/label_policy_sensitivity.csv`
- generated full_ft_k20 verification JSON: `reports/full_ft_k20_verification.json`
