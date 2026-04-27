# Mini-Stage G Full Fine-Tune NIH to MIMIC Adaptation

## Goal
Adapt the NIH-trained DenseNet-121 model to MIMIC by fine-tuning all DenseNet parameters on the small support set.

## Inputs
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`
- support manifest: `/workspace/manifests/mimic_support_k20_seed2027.csv`
- val manifest: `/workspace/manifests/mimic_common5_val.csv`
- test manifest: `/workspace/manifests/mimic_common5_test.csv`
- source-only report: `/workspace/reports/mini_stage_d_nih_to_mimic.json`

## Training Setup
- run name: `full_ft_k20_seed2027`
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
- trainable parameters: 6958981
- total parameter tensors: 364
- trainable parameter tensors: 364
- frozen parameter tensors: 0
- trainable parameter names sample: ['features.conv0.weight', 'features.norm0.weight', 'features.norm0.bias', 'features.denseblock1.denselayer1.norm1.weight', 'features.denseblock1.denselayer1.norm1.bias', 'features.denseblock1.denselayer1.conv1.weight', 'features.denseblock1.denselayer1.norm2.weight', 'features.denseblock1.denselayer1.norm2.bias', 'features.denseblock1.denselayer1.conv2.weight', 'features.denseblock1.denselayer2.norm1.weight', 'features.denseblock1.denselayer2.norm1.bias', 'features.denseblock1.denselayer2.conv1.weight', 'features.denseblock1.denselayer2.norm2.weight', 'features.denseblock1.denselayer2.norm2.bias', 'features.denseblock1.denselayer2.conv2.weight', 'features.denseblock1.denselayer3.norm1.weight', 'features.denseblock1.denselayer3.norm1.bias', 'features.denseblock1.denselayer3.conv1.weight', 'features.denseblock1.denselayer3.norm2.weight', 'features.denseblock1.denselayer3.norm2.bias']

## Training Outcome
- best epoch: 3
- stopped early: yes
- best checkpoint: `/workspace/checkpoints/full_ft_k20_seed2027_best.pt`
- last checkpoint: `/workspace/checkpoints/full_ft_k20_seed2027_last.pt`
- final train loss: 0.4745
- final val loss: 0.5166
- final val macro AUROC: 0.6973
- final val macro AUPRC: 0.2902

## Val Metrics
- loss: 0.5253
- macro AUROC: 0.6981
- macro AUPRC: 0.2830
- Atelectasis: AUROC=0.6679, AUPRC=0.3061, prob_mean=0.5048, prob_std=0.1181
- Cardiomegaly: AUROC=0.6609, AUPRC=0.3078, prob_mean=0.3401, prob_std=0.1512
- Consolidation: AUROC=0.6796, AUPRC=0.0954, prob_mean=0.3333, prob_std=0.1685
- Edema: AUROC=0.7438, AUPRC=0.2798, prob_mean=0.2391, prob_std=0.1770
- Effusion: AUROC=0.7383, AUPRC=0.4263, prob_mean=0.4070, prob_std=0.1603

## Test Metrics
- loss: 0.5911
- macro AUROC: 0.6250
- macro AUPRC: 0.3456
- Atelectasis: AUROC=0.5924, AUPRC=0.3502, prob_mean=0.5438, prob_std=0.1030
- Cardiomegaly: AUROC=0.5938, AUPRC=0.3354, prob_mean=0.3398, prob_std=0.1408
- Consolidation: AUROC=0.5239, AUPRC=0.1072, prob_mean=0.3571, prob_std=0.1541
- Edema: AUROC=0.7327, AUPRC=0.4010, prob_mean=0.2663, prob_std=0.1771
- Effusion: AUROC=0.6823, AUPRC=0.5345, prob_mean=0.4761, prob_std=0.1461

## Source-only Comparison
- source-only val macro AUROC: 0.6790
- source-only val macro AUPRC: 0.2633
- source-only test macro AUROC: 0.6103
- source-only test macro AUPRC: 0.3312
- val macro AUROC delta: 0.0191
- val macro AUPRC delta: 0.0197
- test macro AUROC delta: 0.0147
- test macro AUPRC delta: 0.0144

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
