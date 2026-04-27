# Mini-Stage I LoRA NIH to MIMIC Adaptation

## Goal
Adapt the NIH-trained DenseNet-121 model to MIMIC by freezing all original weights and training only LoRA adapters on denseblock4 and the classifier.

## Inputs
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`
- support manifest: `/workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_support_k20_seed2027.csv`
- val manifest: `/workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_val.csv`
- test manifest: `/workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_test.csv`
- source-only report: `/workspace/experiments/policyB_common5_v1/reports/policyB_no_adaptation_eval_seed2027.json`

## Training Setup
- run name: `policyB_lora_r8a16_lr3e4_k20_seed2027`
- epochs: 50
- batch size: 8
- learning rate: 0.0003
- image size: 224
- patience: 10
- seed: 2027
- LoRA rank: 8
- LoRA alpha: 16.0
- LoRA dropout: 0.0
- device: `cuda`

## Split Sizes
### support
- images: 30
- subjects: 29
- studies: 30
- label policy: uignore_blankzero
- dicoms: 30
- Atelectasis: positives=20, negatives=8, masked=2, n_valid=28
- Cardiomegaly: positives=21, negatives=9, masked=0, n_valid=30
- Consolidation: positives=20, negatives=10, masked=0, n_valid=30
- Edema: positives=21, negatives=8, masked=1, n_valid=29
- Effusion: positives=26, negatives=4, masked=0, n_valid=30

### val
- images: 958
- subjects: 358
- studies: 958
- label policy: uignore_blankzero
- dicoms: 958
- Atelectasis: positives=194, negatives=712, masked=52, n_valid=906
- Cardiomegaly: positives=205, negatives=724, masked=29, n_valid=929
- Consolidation: positives=52, negatives=882, masked=24, n_valid=934
- Edema: positives=131, negatives=767, masked=60, n_valid=898
- Effusion: positives=249, negatives=682, masked=27, n_valid=931

### test
- images: 596
- subjects: 268
- studies: 596
- label policy: uignore_blankzero
- dicoms: 596
- Atelectasis: positives=170, negatives=394, masked=32, n_valid=564
- Cardiomegaly: positives=159, negatives=424, masked=13, n_valid=583
- Consolidation: positives=56, negatives=530, masked=10, n_valid=586
- Edema: positives=128, negatives=424, masked=44, n_valid=552
- Effusion: positives=219, negatives=350, masked=27, n_valid=569

## LoRA Trainability Check
- target module count: 33
- target module names sample: ['features.denseblock4.denselayer1.conv1', 'features.denseblock4.denselayer1.conv2', 'features.denseblock4.denselayer2.conv1', 'features.denseblock4.denselayer2.conv2', 'features.denseblock4.denselayer3.conv1', 'features.denseblock4.denselayer3.conv2', 'features.denseblock4.denselayer4.conv1', 'features.denseblock4.denselayer4.conv2', 'features.denseblock4.denselayer5.conv1', 'features.denseblock4.denselayer5.conv2']
- total parameters: 7133101
- trainable parameters: 174120
- total parameter tensors: 430
- trainable parameter tensors: 66
- frozen parameter tensors: 364
- trainable parameter names sample: ['features.denseblock4.denselayer1.conv1.lora_down.weight', 'features.denseblock4.denselayer1.conv1.lora_up.weight', 'features.denseblock4.denselayer1.conv2.lora_down.weight', 'features.denseblock4.denselayer1.conv2.lora_up.weight', 'features.denseblock4.denselayer2.conv1.lora_down.weight', 'features.denseblock4.denselayer2.conv1.lora_up.weight', 'features.denseblock4.denselayer2.conv2.lora_down.weight', 'features.denseblock4.denselayer2.conv2.lora_up.weight', 'features.denseblock4.denselayer3.conv1.lora_down.weight', 'features.denseblock4.denselayer3.conv1.lora_up.weight', 'features.denseblock4.denselayer3.conv2.lora_down.weight', 'features.denseblock4.denselayer3.conv2.lora_up.weight', 'features.denseblock4.denselayer4.conv1.lora_down.weight', 'features.denseblock4.denselayer4.conv1.lora_up.weight', 'features.denseblock4.denselayer4.conv2.lora_down.weight', 'features.denseblock4.denselayer4.conv2.lora_up.weight', 'features.denseblock4.denselayer5.conv1.lora_down.weight', 'features.denseblock4.denselayer5.conv1.lora_up.weight', 'features.denseblock4.denselayer5.conv2.lora_down.weight', 'features.denseblock4.denselayer5.conv2.lora_up.weight']

## Training Outcome
- best epoch: 1
- stopped early: yes
- best checkpoint: `/workspace/experiments/policyB_common5_v1/checkpoints/policyB_lora_r8a16_lr3e4_k20_seed2027_best.pt`
- last checkpoint: `/workspace/experiments/policyB_common5_v1/checkpoints/policyB_lora_r8a16_lr3e4_k20_seed2027_last.pt`
- final train loss: 0.3996
- final val loss: 1.0081
- final val macro AUROC: 0.6728
- final val macro AUPRC: 0.2780

## Val Metrics
- loss: 0.7039
- macro AUROC: 0.6848
- macro AUPRC: 0.2770
- micro AUROC: n/a
- micro AUPRC: n/a
- Atelectasis: positives=194, negatives=712, masked=52, n_valid=906, AUROC=0.6306, AUPRC=0.2802, prob_mean=0.6316, prob_std=0.0933
- Cardiomegaly: positives=205, negatives=724, masked=29, n_valid=929, AUROC=0.6472, AUPRC=0.2900, prob_mean=0.3842, prob_std=0.1460
- Consolidation: positives=52, negatives=882, masked=24, n_valid=934, AUROC=0.6761, AUPRC=0.0925, prob_mean=0.5173, prob_std=0.1697
- Edema: positives=131, negatives=767, masked=60, n_valid=898, AUROC=0.7530, AUPRC=0.3003, prob_mean=0.4387, prob_std=0.2151
- Effusion: positives=249, negatives=682, masked=27, n_valid=931, AUROC=0.7172, AUPRC=0.4221, prob_mean=0.5495, prob_std=0.1405

## Test Metrics
- loss: 0.7334
- macro AUROC: 0.6182
- macro AUPRC: 0.3541
- micro AUROC: n/a
- micro AUPRC: n/a
- Atelectasis: positives=170, negatives=394, masked=32, n_valid=564, AUROC=0.5754, AUPRC=0.3494, prob_mean=0.6636, prob_std=0.0787
- Cardiomegaly: positives=159, negatives=424, masked=13, n_valid=583, AUROC=0.5663, AUPRC=0.3203, prob_mean=0.3876, prob_std=0.1299
- Consolidation: positives=56, negatives=530, masked=10, n_valid=586, AUROC=0.5110, AUPRC=0.1156, prob_mean=0.5452, prob_std=0.1445
- Edema: positives=128, negatives=424, masked=44, n_valid=552, AUROC=0.7393, AUPRC=0.4302, prob_mean=0.4716, prob_std=0.1922
- Effusion: positives=219, negatives=350, masked=27, n_valid=569, AUROC=0.6992, AUPRC=0.5548, prob_mean=0.6119, prob_std=0.1156

## Source-only Comparison
- source-only val macro AUROC: 0.6856
- source-only val macro AUPRC: 0.2771
- source-only test macro AUROC: 0.6183
- source-only test macro AUPRC: 0.3541
- val macro AUROC delta: -0.0008
- val macro AUPRC delta: -0.0001
- test macro AUROC delta: -0.0001
- test macro AUPRC delta: -0.0000

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
