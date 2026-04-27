# Mini-Stage H Last-block NIH to MIMIC Adaptation

## Goal
Adapt the NIH-trained DenseNet-121 model to MIMIC by training only denseblock4, norm5, and the classifier on the support set.

## Inputs
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`
- support manifest: `/workspace/manifests/mimic_support_k5_seed2027.csv`
- val manifest: `/workspace/manifests/mimic_common5_val.csv`
- test manifest: `/workspace/manifests/mimic_common5_test.csv`
- source-only report: `/workspace/reports/mini_stage_d_nih_to_mimic.json`

## Training Setup
- run name: `lastblock_k5_seed2027`
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
- trainable parameters: 2165253
- total parameter tensors: 364
- trainable parameter tensors: 100
- frozen parameter tensors: 264
- trainable prefixes: ['features.denseblock4.', 'features.norm5.', 'classifier.']
- trainable parameter names sample: ['features.denseblock4.denselayer1.norm1.weight', 'features.denseblock4.denselayer1.norm1.bias', 'features.denseblock4.denselayer1.conv1.weight', 'features.denseblock4.denselayer1.norm2.weight', 'features.denseblock4.denselayer1.norm2.bias', 'features.denseblock4.denselayer1.conv2.weight', 'features.denseblock4.denselayer2.norm1.weight', 'features.denseblock4.denselayer2.norm1.bias', 'features.denseblock4.denselayer2.conv1.weight', 'features.denseblock4.denselayer2.norm2.weight', 'features.denseblock4.denselayer2.norm2.bias', 'features.denseblock4.denselayer2.conv2.weight', 'features.denseblock4.denselayer3.norm1.weight', 'features.denseblock4.denselayer3.norm1.bias', 'features.denseblock4.denselayer3.conv1.weight', 'features.denseblock4.denselayer3.norm2.weight', 'features.denseblock4.denselayer3.norm2.bias', 'features.denseblock4.denselayer3.conv2.weight', 'features.denseblock4.denselayer4.norm1.weight', 'features.denseblock4.denselayer4.norm1.bias']

## Training Outcome
- best epoch: 6
- stopped early: yes
- best checkpoint: `/workspace/checkpoints/lastblock_k5_seed2027_best.pt`
- last checkpoint: `/workspace/checkpoints/lastblock_k5_seed2027_last.pt`
- final train loss: 0.6704
- final val loss: 0.5461
- final val macro AUROC: 0.6820
- final val macro AUPRC: 0.2666

## Val Metrics
- loss: 0.5893
- macro AUROC: 0.6839
- macro AUPRC: 0.2677
- Atelectasis: AUROC=0.6506, AUPRC=0.2858, prob_mean=0.5614, prob_std=0.1040
- Cardiomegaly: AUROC=0.6317, AUPRC=0.2765, prob_mean=0.3627, prob_std=0.1431
- Consolidation: AUROC=0.6777, AUPRC=0.0924, prob_mean=0.4100, prob_std=0.1773
- Edema: AUROC=0.7400, AUPRC=0.2712, prob_mean=0.3064, prob_std=0.1927
- Effusion: AUROC=0.7196, AUPRC=0.4127, prob_mean=0.4494, prob_std=0.1489

## Test Metrics
- loss: 0.6362
- macro AUROC: 0.6131
- macro AUPRC: 0.3355
- Atelectasis: AUROC=0.5782, AUPRC=0.3351, prob_mean=0.5939, prob_std=0.0902
- Cardiomegaly: AUROC=0.5661, AUPRC=0.3172, prob_mean=0.3613, prob_std=0.1317
- Consolidation: AUROC=0.5101, AUPRC=0.1039, prob_mean=0.4303, prob_std=0.1538
- Edema: AUROC=0.7270, AUPRC=0.3897, prob_mean=0.3278, prob_std=0.1818
- Effusion: AUROC=0.6844, AUPRC=0.5317, prob_mean=0.5149, prob_std=0.1288

## Source-only Comparison
- source-only val macro AUROC: 0.6790
- source-only val macro AUPRC: 0.2633
- source-only test macro AUROC: 0.6103
- source-only test macro AUPRC: 0.3312
- val macro AUROC delta: 0.0049
- val macro AUPRC delta: 0.0044
- test macro AUROC delta: 0.0028
- test macro AUPRC delta: 0.0043

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
