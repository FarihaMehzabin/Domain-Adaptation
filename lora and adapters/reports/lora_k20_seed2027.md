# Mini-Stage I LoRA NIH to MIMIC Adaptation

## Goal
Adapt the NIH-trained DenseNet-121 model to MIMIC by freezing all original weights and training only LoRA adapters on denseblock4 and the classifier.

## Inputs
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`
- support manifest: `/workspace/manifests/mimic_support_k20_seed2027.csv`
- val manifest: `/workspace/manifests/mimic_common5_val.csv`
- test manifest: `/workspace/manifests/mimic_common5_test.csv`
- source-only report: `/workspace/reports/mini_stage_d_nih_to_mimic.json`

## Training Setup
- run name: `lora_k20_seed2027`
- epochs: 50
- batch size: 8
- learning rate: 0.0001
- image size: 224
- patience: 10
- seed: 2027
- LoRA rank: 4
- LoRA alpha: 4.0
- LoRA dropout: 0.0
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

## LoRA Trainability Check
- target module count: 33
- target module names sample: ['features.denseblock4.denselayer1.conv1', 'features.denseblock4.denselayer1.conv2', 'features.denseblock4.denselayer2.conv1', 'features.denseblock4.denselayer2.conv2', 'features.denseblock4.denselayer3.conv1', 'features.denseblock4.denselayer3.conv2', 'features.denseblock4.denselayer4.conv1', 'features.denseblock4.denselayer4.conv2', 'features.denseblock4.denselayer5.conv1', 'features.denseblock4.denselayer5.conv2']
- total parameters: 7046041
- trainable parameters: 87060
- total parameter tensors: 430
- trainable parameter tensors: 66
- frozen parameter tensors: 364
- trainable parameter names sample: ['features.denseblock4.denselayer1.conv1.lora_down.weight', 'features.denseblock4.denselayer1.conv1.lora_up.weight', 'features.denseblock4.denselayer1.conv2.lora_down.weight', 'features.denseblock4.denselayer1.conv2.lora_up.weight', 'features.denseblock4.denselayer2.conv1.lora_down.weight', 'features.denseblock4.denselayer2.conv1.lora_up.weight', 'features.denseblock4.denselayer2.conv2.lora_down.weight', 'features.denseblock4.denselayer2.conv2.lora_up.weight', 'features.denseblock4.denselayer3.conv1.lora_down.weight', 'features.denseblock4.denselayer3.conv1.lora_up.weight', 'features.denseblock4.denselayer3.conv2.lora_down.weight', 'features.denseblock4.denselayer3.conv2.lora_up.weight', 'features.denseblock4.denselayer4.conv1.lora_down.weight', 'features.denseblock4.denselayer4.conv1.lora_up.weight', 'features.denseblock4.denselayer4.conv2.lora_down.weight', 'features.denseblock4.denselayer4.conv2.lora_up.weight', 'features.denseblock4.denselayer5.conv1.lora_down.weight', 'features.denseblock4.denselayer5.conv1.lora_up.weight', 'features.denseblock4.denselayer5.conv2.lora_down.weight', 'features.denseblock4.denselayer5.conv2.lora_up.weight']

## Training Outcome
- best epoch: 3
- stopped early: yes
- best checkpoint: `/workspace/checkpoints/lora_k20_seed2027_best.pt`
- last checkpoint: `/workspace/checkpoints/lora_k20_seed2027_last.pt`
- final train loss: 0.6023
- final val loss: 0.7628
- final val macro AUROC: 0.6741
- final val macro AUPRC: 0.2605

## Val Metrics
- loss: 0.6861
- macro AUROC: 0.6790
- macro AUPRC: 0.2634
- Atelectasis: AUROC=0.6246, AUPRC=0.2628, prob_mean=0.6286, prob_std=0.0938
- Cardiomegaly: AUROC=0.6467, AUPRC=0.2835, prob_mean=0.3512, prob_std=0.1424
- Consolidation: AUROC=0.6706, AUPRC=0.0885, prob_mean=0.5058, prob_std=0.1719
- Edema: AUROC=0.7400, AUPRC=0.2731, prob_mean=0.4328, prob_std=0.2137
- Effusion: AUROC=0.7133, AUPRC=0.4090, prob_mean=0.5077, prob_std=0.1428

## Test Metrics
- loss: 0.7224
- macro AUROC: 0.6104
- macro AUPRC: 0.3313
- Atelectasis: AUROC=0.5730, AUPRC=0.3308, prob_mean=0.6600, prob_std=0.0797
- Cardiomegaly: AUROC=0.5626, AUPRC=0.3084, prob_mean=0.3541, prob_std=0.1281
- Consolidation: AUROC=0.5106, AUPRC=0.1151, prob_mean=0.5312, prob_std=0.1458
- Edema: AUROC=0.7204, AUPRC=0.3763, prob_mean=0.4651, prob_std=0.1937
- Effusion: AUROC=0.6855, AUPRC=0.5257, prob_mean=0.5724, prob_std=0.1192

## Source-only Comparison
- source-only val macro AUROC: 0.6790
- source-only val macro AUPRC: 0.2633
- source-only test macro AUROC: 0.6103
- source-only test macro AUPRC: 0.3312
- val macro AUROC delta: 0.0000
- val macro AUPRC delta: 0.0001
- test macro AUROC delta: 0.0001
- test macro AUPRC delta: 0.0001

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
