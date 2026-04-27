# Policy B Manifest Audit

Official policy: `uignore_blankzero`

## Split Rows

| Split | Rows | All-five-blank rows | Rows with no valid labels |
| --- | ---: | ---: | ---: |
| train_pool | 963 | 386 | 0 |
| val | 958 | 367 | 0 |
| test | 596 | 131 | 0 |

## Raw Label Counts

| Split | Label | Raw 1 | Raw 0 | Raw -1 | Raw blank |
| --- | --- | ---: | ---: | ---: | ---: |
| train_pool | Atelectasis | 193 | 6 | 47 | 717 |
| train_pool | Cardiomegaly | 195 | 59 | 21 | 688 |
| train_pool | Consolidation | 55 | 22 | 17 | 869 |
| train_pool | Edema | 120 | 92 | 57 | 694 |
| train_pool | Effusion | 232 | 110 | 26 | 595 |
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

## Final Label Counts

| Split | Label | Positive | Negative | Masked |
| --- | --- | ---: | ---: | ---: |
| train_pool | Atelectasis | 193 | 723 | 47 |
| train_pool | Cardiomegaly | 195 | 747 | 21 |
| train_pool | Consolidation | 55 | 891 | 17 |
| train_pool | Edema | 120 | 786 | 57 |
| train_pool | Effusion | 232 | 705 | 26 |
| val | Atelectasis | 194 | 712 | 52 |
| val | Cardiomegaly | 205 | 724 | 29 |
| val | Consolidation | 52 | 882 | 24 |
| val | Edema | 131 | 767 | 60 |
| val | Effusion | 249 | 682 | 27 |
| test | Atelectasis | 170 | 394 | 32 |
| test | Cardiomegaly | 159 | 424 | 13 |
| test | Consolidation | 56 | 530 | 10 |
| test | Edema | 128 | 424 | 44 |
| test | Effusion | 219 | 350 | 27 |

## Split Overlap Checks

| Pair | Key | Overlap count | Example values |
| --- | --- | ---: | --- |
| train_pool_vs_val | subject_id | 0 | [] |
| train_pool_vs_val | study_id | 0 | [] |
| train_pool_vs_val | dicom_id | 0 | [] |
| train_pool_vs_test | subject_id | 0 | [] |
| train_pool_vs_test | study_id | 0 | [] |
| train_pool_vs_test | dicom_id | 0 | [] |
| val_vs_test | subject_id | 0 | [] |
| val_vs_test | study_id | 0 | [] |
| val_vs_test | dicom_id | 0 | [] |

## Support Sets

| Support | Rows | Subjects | Studies | DICOMs |
| --- | ---: | ---: | ---: | ---: |
| k5 | 7 | 7 | 7 | 7 |
| k20 | 30 | 29 | 30 | 30 |

## Support Label Counts

| Support | Label | Positive | Negative | Masked |
| --- | --- | ---: | ---: | ---: |
| k5 | Atelectasis | 6 | 0 | 1 |
| k5 | Cardiomegaly | 5 | 2 | 0 |
| k5 | Consolidation | 5 | 2 | 0 |
| k5 | Edema | 6 | 1 | 0 |
| k5 | Effusion | 7 | 0 | 0 |
| k20 | Atelectasis | 20 | 8 | 2 |
| k20 | Cardiomegaly | 21 | 9 | 0 |
| k20 | Consolidation | 20 | 10 | 0 |
| k20 | Edema | 21 | 8 | 1 |
| k20 | Effusion | 26 | 4 | 0 |

## Warnings

- none
