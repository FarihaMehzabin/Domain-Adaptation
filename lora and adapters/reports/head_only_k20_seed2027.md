# Mini-Stage F Head-only NIH to MIMIC Adaptation

## Goal
Adapt the NIH-trained DenseNet-121 model to MIMIC by freezing the DenseNet backbone and training only the classifier head on the small support set.

## Inputs
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`
- support manifest: `/workspace/manifests/mimic_support_k20_seed2027.csv`
- val manifest: `/workspace/manifests/mimic_common5_val.csv`
- test manifest: `/workspace/manifests/mimic_common5_test.csv`
- source-only report: `/workspace/reports/mini_stage_d_nih_to_mimic.json`

## Training Setup
- run name: `head_only_k20_seed2027`
- epochs: 50
- batch size: 8
- learning rate: 0.0001
- image size: 224
- patience: 10
- seed: 2027
- device: `cuda`

## Split Sizes
### support
- images: 30
- subjects: 29
- studies: 30
- dicoms: 30
- Atelectasis: positives=20, negatives=10
- Cardiomegaly: positives=22, negatives=8
- Consolidation: positives=20, negatives=10
- Edema: positives=20, negatives=10
- Effusion: positives=26, negatives=4

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
- best checkpoint: `/workspace/checkpoints/head_only_k20_seed2027_best.pt`
- last checkpoint: `/workspace/checkpoints/head_only_k20_seed2027_last.pt`
- final train loss: 0.5669
- final val loss: 0.8632
- final val macro AUROC: 0.6700
- final val macro AUPRC: 0.2577

## Val Metrics
- loss: 0.6993
- macro AUROC: 0.6780
- macro AUPRC: 0.2628
- Atelectasis: AUROC=0.6232, AUPRC=0.2617, prob_mean=0.6326, prob_std=0.0934
- Cardiomegaly: AUROC=0.6443, AUPRC=0.2821, prob_mean=0.3724, prob_std=0.1450
- Consolidation: AUROC=0.6706, AUPRC=0.0887, prob_mean=0.5063, prob_std=0.1714
- Edema: AUROC=0.7399, AUPRC=0.2733, prob_mean=0.4394, prob_std=0.2154
- Effusion: AUROC=0.7121, AUPRC=0.4080, prob_mean=0.5345, prob_std=0.1413

## Test Metrics
- loss: 0.7327
- macro AUROC: 0.6104
- macro AUPRC: 0.3319
- Atelectasis: AUROC=0.5728, AUPRC=0.3311, prob_mean=0.6639, prob_std=0.0793
- Cardiomegaly: AUROC=0.5625, AUPRC=0.3116, prob_mean=0.3751, prob_std=0.1306
- Consolidation: AUROC=0.5105, AUPRC=0.1147, prob_mean=0.5321, prob_std=0.1455
- Edema: AUROC=0.7208, AUPRC=0.3768, prob_mean=0.4719, prob_std=0.1950
- Effusion: AUROC=0.6854, AUPRC=0.5255, prob_mean=0.5987, prob_std=0.1168

## Source-only Comparison
- source-only val macro AUROC: 0.6790
- source-only val macro AUPRC: 0.2633
- source-only test macro AUROC: 0.6103
- source-only test macro AUPRC: 0.3312
- val macro AUROC delta: -0.0010
- val macro AUPRC delta: -0.0006
- test macro AUROC delta: 0.0001
- test macro AUPRC delta: 0.0007

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
