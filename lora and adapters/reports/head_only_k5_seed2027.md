# Mini-Stage F Head-only NIH to MIMIC Adaptation

## Goal
Adapt the NIH-trained DenseNet-121 model to MIMIC by freezing the DenseNet backbone and training only the classifier head on the small support set.

## Inputs
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`
- support manifest: `/workspace/manifests/mimic_support_k5_seed2027.csv`
- val manifest: `/workspace/manifests/mimic_common5_val.csv`
- test manifest: `/workspace/manifests/mimic_common5_test.csv`
- source-only report: `/workspace/reports/mini_stage_d_nih_to_mimic.json`

## Training Setup
- run name: `head_only_k5_seed2027`
- epochs: 50
- batch size: 8
- learning rate: 0.0001
- image size: 224
- patience: 10
- seed: 2027
- device: `cuda`

## Split Sizes
### support
- images: 7
- subjects: 7
- studies: 7
- dicoms: 7
- Atelectasis: positives=5, negatives=2
- Cardiomegaly: positives=6, negatives=1
- Consolidation: positives=5, negatives=2
- Edema: positives=6, negatives=1
- Effusion: positives=7, negatives=0

### val
- images: 958
- subjects: 358
- studies: 958
- dicoms: 958
- Atelectasis: positives=194, negatives=764
- Cardiomegaly: positives=205, negatives=753
- Consolidation: positives=52, negatives=906
- Edema: positives=131, negatives=827
- Effusion: positives=249, negatives=709

### test
- images: 596
- subjects: 268
- studies: 596
- dicoms: 596
- Atelectasis: positives=170, negatives=426
- Cardiomegaly: positives=159, negatives=437
- Consolidation: positives=56, negatives=540
- Edema: positives=128, negatives=468
- Effusion: positives=219, negatives=377

## Parameter Freeze Check
- total parameters: 6958981
- trainable parameters: 5125
- trainable parameter names: ['classifier.weight', 'classifier.bias']

## Training Outcome
- best epoch: 1
- stopped early: yes
- best checkpoint: `/workspace/checkpoints/head_only_k5_seed2027_best.pt`
- last checkpoint: `/workspace/checkpoints/head_only_k5_seed2027_last.pt`
- final train loss: 0.5459
- final val loss: 0.7812
- final val macro AUROC: 0.6743
- final val macro AUPRC: 0.2601

## Val Metrics
- loss: 0.6897
- macro AUROC: 0.6787
- macro AUPRC: 0.2631
- Atelectasis: AUROC=0.6239, AUPRC=0.2623, prob_mean=0.6334, prob_std=0.0937
- Cardiomegaly: AUROC=0.6459, AUPRC=0.2830, prob_mean=0.3477, prob_std=0.1420
- Consolidation: AUROC=0.6710, AUPRC=0.0886, prob_mean=0.5051, prob_std=0.1726
- Edema: AUROC=0.7398, AUPRC=0.2733, prob_mean=0.4417, prob_std=0.2145
- Effusion: AUROC=0.7127, AUPRC=0.4083, prob_mean=0.5057, prob_std=0.1427

## Test Metrics
- loss: 0.7257
- macro AUROC: 0.6102
- macro AUPRC: 0.3311
- Atelectasis: AUROC=0.5728, AUPRC=0.3307, prob_mean=0.6648, prob_std=0.0797
- Cardiomegaly: AUROC=0.5618, AUPRC=0.3077, prob_mean=0.3502, prob_std=0.1280
- Consolidation: AUROC=0.5106, AUPRC=0.1152, prob_mean=0.5306, prob_std=0.1465
- Edema: AUROC=0.7203, AUPRC=0.3763, prob_mean=0.4744, prob_std=0.1941
- Effusion: AUROC=0.6856, AUPRC=0.5256, prob_mean=0.5705, prob_std=0.1194

## Source-only Comparison
- source-only val macro AUROC: 0.6790
- source-only val macro AUPRC: 0.2633
- source-only test macro AUROC: 0.6103
- source-only test macro AUPRC: 0.3312
- val macro AUROC delta: -0.0003
- val macro AUPRC delta: -0.0002
- test macro AUROC delta: -0.0001
- test macro AUPRC delta: -0.0001

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
