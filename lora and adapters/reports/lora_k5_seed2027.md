# Mini-Stage I LoRA NIH to MIMIC Adaptation

## Goal
Adapt the NIH-trained DenseNet-121 model to MIMIC by freezing all original weights and training only LoRA adapters on denseblock4 and the classifier.

## Inputs
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`
- support manifest: `/workspace/manifests/mimic_support_k5_seed2027.csv`
- val manifest: `/workspace/manifests/mimic_common5_val.csv`
- test manifest: `/workspace/manifests/mimic_common5_test.csv`
- source-only report: `/workspace/reports/mini_stage_d_nih_to_mimic.json`

## Training Setup
- run name: `lora_k5_seed2027`
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
- best epoch: 6
- stopped early: yes
- best checkpoint: `/workspace/checkpoints/lora_k5_seed2027_best.pt`
- last checkpoint: `/workspace/checkpoints/lora_k5_seed2027_last.pt`
- final train loss: 0.6179
- final val loss: 0.6958
- final val macro AUROC: 0.6787
- final val macro AUPRC: 0.2632

## Val Metrics
- loss: 0.6847
- macro AUROC: 0.6791
- macro AUPRC: 0.2634
- Atelectasis: AUROC=0.6244, AUPRC=0.2629, prob_mean=0.6297, prob_std=0.0940
- Cardiomegaly: AUROC=0.6467, AUPRC=0.2833, prob_mean=0.3453, prob_std=0.1417
- Consolidation: AUROC=0.6707, AUPRC=0.0885, prob_mean=0.5044, prob_std=0.1723
- Edema: AUROC=0.7401, AUPRC=0.2733, prob_mean=0.4362, prob_std=0.2143
- Effusion: AUROC=0.7133, AUPRC=0.4090, prob_mean=0.5005, prob_std=0.1430

## Test Metrics
- loss: 0.7216
- macro AUROC: 0.6103
- macro AUPRC: 0.3312
- Atelectasis: AUROC=0.5728, AUPRC=0.3309, prob_mean=0.6612, prob_std=0.0799
- Cardiomegaly: AUROC=0.5622, AUPRC=0.3084, prob_mean=0.3482, prob_std=0.1276
- Consolidation: AUROC=0.5106, AUPRC=0.1151, prob_mean=0.5298, prob_std=0.1462
- Edema: AUROC=0.7205, AUPRC=0.3765, prob_mean=0.4688, prob_std=0.1941
- Effusion: AUROC=0.6854, AUPRC=0.5252, prob_mean=0.5653, prob_std=0.1197

## Source-only Comparison
- source-only val macro AUROC: 0.6790
- source-only val macro AUPRC: 0.2633
- source-only test macro AUROC: 0.6103
- source-only test macro AUPRC: 0.3312
- val macro AUROC delta: 0.0000
- val macro AUPRC delta: 0.0001
- test macro AUROC delta: 0.0000
- test macro AUPRC delta: 0.0000

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
