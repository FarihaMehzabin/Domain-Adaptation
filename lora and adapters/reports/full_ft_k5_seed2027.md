# Mini-Stage G Full Fine-Tune NIH to MIMIC Adaptation

## Goal
Adapt the NIH-trained DenseNet-121 model to MIMIC by fine-tuning all DenseNet parameters on the small support set.

## Inputs
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`
- support manifest: `/workspace/manifests/mimic_support_k5_seed2027.csv`
- val manifest: `/workspace/manifests/mimic_common5_val.csv`
- test manifest: `/workspace/manifests/mimic_common5_test.csv`
- source-only report: `/workspace/reports/mini_stage_d_nih_to_mimic.json`

## Training Setup
- run name: `full_ft_k5_seed2027`
- epochs: 50
- batch size: 8
- learning rate: 1e-05
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

## Parameter Trainability Check
- total parameters: 6958981
- trainable parameters: 6958981
- total parameter tensors: 364
- trainable parameter tensors: 364
- frozen parameter tensors: 0
- trainable parameter names sample: ['features.conv0.weight', 'features.norm0.weight', 'features.norm0.bias', 'features.denseblock1.denselayer1.norm1.weight', 'features.denseblock1.denselayer1.norm1.bias', 'features.denseblock1.denselayer1.conv1.weight', 'features.denseblock1.denselayer1.norm2.weight', 'features.denseblock1.denselayer1.norm2.bias', 'features.denseblock1.denselayer1.conv2.weight', 'features.denseblock1.denselayer2.norm1.weight', 'features.denseblock1.denselayer2.norm1.bias', 'features.denseblock1.denselayer2.conv1.weight', 'features.denseblock1.denselayer2.norm2.weight', 'features.denseblock1.denselayer2.norm2.bias', 'features.denseblock1.denselayer2.conv2.weight', 'features.denseblock1.denselayer3.norm1.weight', 'features.denseblock1.denselayer3.norm1.bias', 'features.denseblock1.denselayer3.conv1.weight', 'features.denseblock1.denselayer3.norm2.weight', 'features.denseblock1.denselayer3.norm2.bias']

## Training Outcome
- best epoch: 12
- stopped early: yes
- best checkpoint: `/workspace/checkpoints/full_ft_k5_seed2027_best.pt`
- last checkpoint: `/workspace/checkpoints/full_ft_k5_seed2027_last.pt`
- final train loss: 0.4098
- final val loss: 0.5455
- final val macro AUROC: 0.6940
- final val macro AUPRC: 0.2813

## Val Metrics
- loss: 0.5458
- macro AUROC: 0.6972
- macro AUPRC: 0.2803
- Atelectasis: AUROC=0.6682, AUPRC=0.3003, prob_mean=0.5162, prob_std=0.1083
- Cardiomegaly: AUROC=0.6491, AUPRC=0.3029, prob_mean=0.3752, prob_std=0.1443
- Consolidation: AUROC=0.6789, AUPRC=0.0896, prob_mean=0.3524, prob_std=0.1640
- Edema: AUROC=0.7531, AUPRC=0.2848, prob_mean=0.2553, prob_std=0.1727
- Effusion: AUROC=0.7369, AUPRC=0.4242, prob_mean=0.4418, prob_std=0.1527

## Test Metrics
- loss: 0.6061
- macro AUROC: 0.6239
- macro AUPRC: 0.3436
- Atelectasis: AUROC=0.5910, AUPRC=0.3438, prob_mean=0.5520, prob_std=0.0948
- Cardiomegaly: AUROC=0.5830, AUPRC=0.3235, prob_mean=0.3789, prob_std=0.1385
- Consolidation: AUROC=0.5255, AUPRC=0.1012, prob_mean=0.3799, prob_std=0.1510
- Edema: AUROC=0.7350, AUPRC=0.4097, prob_mean=0.2786, prob_std=0.1671
- Effusion: AUROC=0.6851, AUPRC=0.5398, prob_mean=0.5087, prob_std=0.1367

## Source-only Comparison
- source-only val macro AUROC: 0.6790
- source-only val macro AUPRC: 0.2633
- source-only test macro AUROC: 0.6103
- source-only test macro AUPRC: 0.3312
- val macro AUROC delta: 0.0182
- val macro AUPRC delta: 0.0170
- test macro AUROC delta: 0.0136
- test macro AUPRC delta: 0.0124

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
