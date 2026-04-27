# Mini-Stage H Last-block NIH to MIMIC Adaptation

## Goal
Adapt the NIH-trained DenseNet-121 model to MIMIC by training only denseblock4, norm5, and the classifier on the support set.

## Inputs
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`
- support manifest: `/workspace/manifests/mimic_support_k20_seed2027.csv`
- val manifest: `/workspace/manifests/mimic_common5_val.csv`
- test manifest: `/workspace/manifests/mimic_common5_test.csv`
- source-only report: `/workspace/reports/mini_stage_d_nih_to_mimic.json`

## Training Setup
- run name: `lastblock_k20_seed2027`
- epochs: 50
- batch size: 8
- learning rate: 1e-05
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

## Parameter Trainability Check
- total parameters: 6958981
- trainable parameters: 2165253
- total parameter tensors: 364
- trainable parameter tensors: 100
- frozen parameter tensors: 264
- trainable prefixes: ['features.denseblock4.', 'features.norm5.', 'classifier.']
- trainable parameter names sample: ['features.denseblock4.denselayer1.norm1.weight', 'features.denseblock4.denselayer1.norm1.bias', 'features.denseblock4.denselayer1.conv1.weight', 'features.denseblock4.denselayer1.norm2.weight', 'features.denseblock4.denselayer1.norm2.bias', 'features.denseblock4.denselayer1.conv2.weight', 'features.denseblock4.denselayer2.norm1.weight', 'features.denseblock4.denselayer2.norm1.bias', 'features.denseblock4.denselayer2.conv1.weight', 'features.denseblock4.denselayer2.norm2.weight', 'features.denseblock4.denselayer2.norm2.bias', 'features.denseblock4.denselayer2.conv2.weight', 'features.denseblock4.denselayer3.norm1.weight', 'features.denseblock4.denselayer3.norm1.bias', 'features.denseblock4.denselayer3.conv1.weight', 'features.denseblock4.denselayer3.norm2.weight', 'features.denseblock4.denselayer3.norm2.bias', 'features.denseblock4.denselayer3.conv2.weight', 'features.denseblock4.denselayer4.norm1.weight', 'features.denseblock4.denselayer4.norm1.bias']

## Training Outcome
- best epoch: 4
- stopped early: yes
- best checkpoint: `/workspace/checkpoints/lastblock_k20_seed2027_best.pt`
- last checkpoint: `/workspace/checkpoints/lastblock_k20_seed2027_last.pt`
- final train loss: 0.6136
- final val loss: 0.5331
- final val macro AUROC: 0.6884
- final val macro AUPRC: 0.2731

## Val Metrics
- loss: 0.5302
- macro AUROC: 0.6892
- macro AUPRC: 0.2717
- Atelectasis: AUROC=0.6704, AUPRC=0.3006, prob_mean=0.4914, prob_std=0.1128
- Cardiomegaly: AUROC=0.6317, AUPRC=0.2803, prob_mean=0.3616, prob_std=0.1441
- Consolidation: AUROC=0.6858, AUPRC=0.1000, prob_mean=0.3364, prob_std=0.1624
- Edema: AUROC=0.7351, AUPRC=0.2647, prob_mean=0.2262, prob_std=0.1615
- Effusion: AUROC=0.7230, AUPRC=0.4129, prob_mean=0.4248, prob_std=0.1511

## Test Metrics
- loss: 0.5892
- macro AUROC: 0.6124
- macro AUPRC: 0.3341
- Atelectasis: AUROC=0.5727, AUPRC=0.3317, prob_mean=0.5246, prob_std=0.0980
- Cardiomegaly: AUROC=0.5740, AUPRC=0.3248, prob_mean=0.3578, prob_std=0.1334
- Consolidation: AUROC=0.5106, AUPRC=0.0967, prob_mean=0.3498, prob_std=0.1435
- Edema: AUROC=0.7262, AUPRC=0.3902, prob_mean=0.2395, prob_std=0.1548
- Effusion: AUROC=0.6786, AUPRC=0.5273, prob_mean=0.4908, prob_std=0.1349

## Source-only Comparison
- source-only val macro AUROC: 0.6790
- source-only val macro AUPRC: 0.2633
- source-only test macro AUROC: 0.6103
- source-only test macro AUPRC: 0.3312
- val macro AUROC delta: 0.0102
- val macro AUPRC delta: 0.0084
- test macro AUROC delta: 0.0021
- test macro AUPRC delta: 0.0029

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
