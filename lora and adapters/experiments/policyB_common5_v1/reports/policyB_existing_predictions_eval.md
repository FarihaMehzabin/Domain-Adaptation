# Policy B Existing Predictions Evaluation

Existing prediction files were reevaluated against the new Policy B manifests only. No model was retrained.

## Headline

- no_adaptation test macro AUROC: 0.618335
- no_adaptation test macro AUPRC: 0.354095
- best existing test file: `/workspace/outputs/full_ft_k20_seed2027_test_predictions.csv`
- best existing test macro AUROC: 0.633519
- best existing test macro AUPRC: 0.370183

## Overall Metrics

| Run | Split | Match | Macro AUROC | Macro AUPRC | Micro AUROC | Micro AUPRC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| no_adaptation | val | dicom_id | 0.685583 | 0.277069 | n/a | n/a |
| no_adaptation | test | dicom_id | 0.618335 | 0.354095 | n/a | n/a |
| head_only_k5 | val | dicom_id | 0.685215 | 0.276739 | n/a | n/a |
| head_only_k5 | test | dicom_id | 0.618247 | 0.353908 | n/a | n/a |
| head_only_k20 | val | dicom_id | 0.684563 | 0.276284 | n/a | n/a |
| head_only_k20 | test | dicom_id | 0.618328 | 0.354705 | n/a | n/a |
| full_ft_k5 | val | dicom_id | 0.703721 | 0.294065 | n/a | n/a |
| full_ft_k5 | test | dicom_id | 0.631381 | 0.365972 | n/a | n/a |
| full_ft_k20 | val | dicom_id | 0.705042 | 0.297591 | n/a | n/a |
| full_ft_k20 | test | dicom_id | 0.633519 | 0.370183 | n/a | n/a |

## Per-Label Metrics

| Run | Split | Label | N valid | Positives | Negatives | Masked | AUROC | AUPRC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_adaptation | val | Atelectasis | 906 | 194 | 712 | 52 | 0.631132 | 0.280953 |
| no_adaptation | val | Cardiomegaly | 929 | 205 | 724 | 29 | 0.649272 | 0.291860 |
| no_adaptation | val | Consolidation | 934 | 52 | 882 | 24 | 0.677394 | 0.092915 |
| no_adaptation | val | Edema | 898 | 131 | 767 | 60 | 0.752311 | 0.297808 |
| no_adaptation | val | Effusion | 931 | 249 | 682 | 27 | 0.717804 | 0.421809 |
| no_adaptation | test | Atelectasis | 564 | 170 | 394 | 32 | 0.576672 | 0.349862 |
| no_adaptation | test | Cardiomegaly | 583 | 159 | 424 | 13 | 0.564999 | 0.317065 |
| no_adaptation | test | Consolidation | 586 | 56 | 530 | 10 | 0.513174 | 0.117123 |
| no_adaptation | test | Edema | 552 | 128 | 424 | 44 | 0.739203 | 0.430685 |
| no_adaptation | test | Effusion | 569 | 219 | 350 | 27 | 0.697626 | 0.555742 |
| head_only_k5 | val | Atelectasis | 906 | 194 | 712 | 52 | 0.630647 | 0.280242 |
| head_only_k5 | val | Cardiomegaly | 929 | 205 | 724 | 29 | 0.648457 | 0.291290 |
| head_only_k5 | val | Consolidation | 934 | 52 | 882 | 24 | 0.677481 | 0.092936 |
| head_only_k5 | val | Edema | 898 | 131 | 767 | 60 | 0.752232 | 0.298076 |
| head_only_k5 | val | Effusion | 931 | 249 | 682 | 27 | 0.717256 | 0.421149 |
| head_only_k5 | test | Atelectasis | 564 | 170 | 394 | 32 | 0.576680 | 0.349849 |
| head_only_k5 | test | Cardiomegaly | 583 | 159 | 424 | 13 | 0.564985 | 0.316460 |
| head_only_k5 | test | Consolidation | 586 | 56 | 530 | 10 | 0.512837 | 0.117026 |
| head_only_k5 | test | Edema | 552 | 128 | 424 | 44 | 0.738926 | 0.430225 |
| head_only_k5 | test | Effusion | 569 | 219 | 350 | 27 | 0.697808 | 0.555979 |
| head_only_k20 | val | Atelectasis | 906 | 194 | 712 | 52 | 0.629952 | 0.279501 |
| head_only_k20 | val | Cardiomegaly | 929 | 205 | 724 | 29 | 0.646847 | 0.290299 |
| head_only_k20 | val | Consolidation | 934 | 52 | 882 | 24 | 0.677089 | 0.093054 |
| head_only_k20 | val | Edema | 898 | 131 | 767 | 60 | 0.752341 | 0.298151 |
| head_only_k20 | val | Effusion | 931 | 249 | 682 | 27 | 0.716585 | 0.420412 |
| head_only_k20 | test | Atelectasis | 564 | 170 | 394 | 32 | 0.576605 | 0.350342 |
| head_only_k20 | test | Cardiomegaly | 583 | 159 | 424 | 13 | 0.565593 | 0.320232 |
| head_only_k20 | test | Consolidation | 586 | 56 | 530 | 10 | 0.512702 | 0.116502 |
| head_only_k20 | test | Edema | 552 | 128 | 424 | 44 | 0.739534 | 0.430868 |
| head_only_k20 | test | Effusion | 569 | 219 | 350 | 27 | 0.697208 | 0.555581 |
| full_ft_k5 | val | Atelectasis | 906 | 194 | 712 | 52 | 0.676467 | 0.320400 |
| full_ft_k5 | val | Cardiomegaly | 929 | 205 | 724 | 29 | 0.650296 | 0.310997 |
| full_ft_k5 | val | Consolidation | 934 | 52 | 882 | 24 | 0.685679 | 0.094668 |
| full_ft_k5 | val | Edema | 898 | 131 | 767 | 60 | 0.765727 | 0.308647 |
| full_ft_k5 | val | Effusion | 931 | 249 | 682 | 27 | 0.740434 | 0.435612 |
| full_ft_k5 | test | Atelectasis | 564 | 170 | 394 | 32 | 0.597895 | 0.364101 |
| full_ft_k5 | test | Cardiomegaly | 583 | 159 | 424 | 13 | 0.586255 | 0.333548 |
| full_ft_k5 | test | Consolidation | 586 | 56 | 530 | 10 | 0.528302 | 0.103439 |
| full_ft_k5 | test | Edema | 552 | 128 | 424 | 44 | 0.750884 | 0.467056 |
| full_ft_k5 | test | Effusion | 569 | 219 | 350 | 27 | 0.693568 | 0.561717 |
| full_ft_k20 | val | Atelectasis | 906 | 194 | 712 | 52 | 0.676814 | 0.326672 |
| full_ft_k20 | val | Cardiomegaly | 929 | 205 | 724 | 29 | 0.661973 | 0.316124 |
| full_ft_k20 | val | Consolidation | 934 | 52 | 882 | 24 | 0.686334 | 0.100542 |
| full_ft_k20 | val | Edema | 898 | 131 | 767 | 60 | 0.757377 | 0.305221 |
| full_ft_k20 | val | Effusion | 931 | 249 | 682 | 27 | 0.742713 | 0.439398 |
| full_ft_k20 | test | Atelectasis | 564 | 170 | 394 | 32 | 0.600523 | 0.372814 |
| full_ft_k20 | test | Cardiomegaly | 583 | 159 | 424 | 13 | 0.597084 | 0.345198 |
| full_ft_k20 | test | Consolidation | 586 | 56 | 530 | 10 | 0.527190 | 0.109796 |
| full_ft_k20 | test | Edema | 552 | 128 | 424 | 44 | 0.750129 | 0.463766 |
| full_ft_k20 | test | Effusion | 569 | 219 | 350 | 27 | 0.692668 | 0.559342 |

## Runs Requiring Retraining

- head_only_k5_seed2027: Support labels, adaptation training, and model selection were created under the old policy.
- head_only_k20_seed2027: Support labels, adaptation training, and model selection were created under the old policy.
- full_ft_k5_seed2027: Support labels, adaptation training, and model selection were created under the old policy.
- full_ft_k20_seed2027: Support labels, adaptation training, and model selection were created under the old policy.

## Warnings

- none
