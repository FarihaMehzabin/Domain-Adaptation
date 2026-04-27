# Mini-Stage D NIH to MIMIC Evaluation

## Goal
Evaluate the trained NIH 2k DenseNet-121 model directly on MIMIC common5 without any target adaptation.

## Checkpoint Used
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`

## MIMIC Manifests Used
- val: `/workspace/manifests/mimic_common5_val.csv`
- test: `/workspace/manifests/mimic_common5_test.csv`

## Split Sizes
- val images: 958
- val subjects: 358
- val studies: 958
- test images: 596
- test subjects: 268
- test studies: 596

## Label Counts
### Val
- Atelectasis: positives=194, negatives=764
- Cardiomegaly: positives=205, negatives=753
- Consolidation: positives=52, negatives=906
- Edema: positives=131, negatives=827
- Effusion: positives=249, negatives=709

### Test
- Atelectasis: positives=170, negatives=426
- Cardiomegaly: positives=159, negatives=437
- Consolidation: positives=56, negatives=540
- Edema: positives=128, negatives=468
- Effusion: positives=219, negatives=377

## Val Metrics
- loss: 0.6815
- macro AUROC: 0.6790
- macro AUPRC: 0.2633
- Atelectasis: AUROC=0.6243, AUPRC=0.2628, prob_mean=0.6297, prob_std=0.0944
- Cardiomegaly: AUROC=0.6467, AUPRC=0.2835, prob_mean=0.3379, prob_std=0.1407
- Consolidation: AUROC=0.6708, AUPRC=0.0885, prob_mean=0.5037, prob_std=0.1725
- Edema: AUROC=0.7399, AUPRC=0.2728, prob_mean=0.4330, prob_std=0.2139
- Effusion: AUROC=0.7133, AUPRC=0.4089, prob_mean=0.4946, prob_std=0.1431

## Test Metrics
- loss: 0.7191
- macro AUROC: 0.6103
- macro AUPRC: 0.3312
- Atelectasis: AUROC=0.5728, AUPRC=0.3305, prob_mean=0.6615, prob_std=0.0803
- Cardiomegaly: AUROC=0.5619, AUPRC=0.3083, prob_mean=0.3404, prob_std=0.1269
- Consolidation: AUROC=0.5109, AUPRC=0.1153, prob_mean=0.5289, prob_std=0.1463
- Edema: AUROC=0.7205, AUPRC=0.3765, prob_mean=0.4654, prob_std=0.1940
- Effusion: AUROC=0.6854, AUPRC=0.5253, prob_mean=0.5595, prob_std=0.1202

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
