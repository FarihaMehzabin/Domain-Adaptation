# Mini-Stage D NIH to MIMIC Evaluation

## Goal
Evaluate the trained NIH 2k DenseNet-121 model directly on MIMIC common5 without any target adaptation.

## Checkpoint Used
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`

## MIMIC Manifests Used
- val: `/workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_val.csv`
- test: `/workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_test.csv`

## Split Sizes
- val images: 958
- val subjects: 358
- val studies: 958
- test images: 596
- test subjects: 268
- test studies: 596

## Label Counts
### Val
- Atelectasis: positives=194, negatives=712, masked=52, n_valid=906
- Cardiomegaly: positives=205, negatives=724, masked=29, n_valid=929
- Consolidation: positives=52, negatives=882, masked=24, n_valid=934
- Edema: positives=131, negatives=767, masked=60, n_valid=898
- Effusion: positives=249, negatives=682, masked=27, n_valid=931

### Test
- Atelectasis: positives=170, negatives=394, masked=32, n_valid=564
- Cardiomegaly: positives=159, negatives=424, masked=13, n_valid=583
- Consolidation: positives=56, negatives=530, masked=10, n_valid=586
- Edema: positives=128, negatives=424, masked=44, n_valid=552
- Effusion: positives=219, negatives=350, masked=27, n_valid=569

## Val Metrics
- loss: 0.6739
- macro AUROC: 0.6856
- macro AUPRC: 0.2771
- micro AUROC: n/a
- micro AUPRC: n/a
- Atelectasis: positives=194, negatives=712, masked=52, n_valid=906, AUROC=0.6311, AUPRC=0.2810, prob_mean=0.6283, prob_std=0.0953
- Cardiomegaly: positives=205, negatives=724, masked=29, n_valid=929, AUROC=0.6493, AUPRC=0.2919, prob_mean=0.3373, prob_std=0.1417
- Consolidation: positives=52, negatives=882, masked=24, n_valid=934, AUROC=0.6774, AUPRC=0.0929, prob_mean=0.5007, prob_std=0.1722
- Edema: positives=131, negatives=767, masked=60, n_valid=898, AUROC=0.7523, AUPRC=0.2978, prob_mean=0.4259, prob_std=0.2151
- Effusion: positives=249, negatives=682, masked=27, n_valid=931, AUROC=0.7178, AUPRC=0.4218, prob_mean=0.4937, prob_std=0.1436

## Test Metrics
- loss: 0.7099
- macro AUROC: 0.6183
- macro AUPRC: 0.3541
- micro AUROC: n/a
- micro AUPRC: n/a
- Atelectasis: positives=170, negatives=394, masked=32, n_valid=564, AUROC=0.5767, AUPRC=0.3499, prob_mean=0.6610, prob_std=0.0807
- Cardiomegaly: positives=159, negatives=424, masked=13, n_valid=583, AUROC=0.5650, AUPRC=0.3171, prob_mean=0.3395, prob_std=0.1270
- Consolidation: positives=56, negatives=530, masked=10, n_valid=586, AUROC=0.5132, AUPRC=0.1171, prob_mean=0.5279, prob_std=0.1468
- Edema: positives=128, negatives=424, masked=44, n_valid=552, AUROC=0.7392, AUPRC=0.4307, prob_mean=0.4585, prob_std=0.1930
- Effusion: positives=219, negatives=350, masked=27, n_valid=569, AUROC=0.6976, AUPRC=0.5557, prob_mean=0.5574, prob_std=0.1215

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
