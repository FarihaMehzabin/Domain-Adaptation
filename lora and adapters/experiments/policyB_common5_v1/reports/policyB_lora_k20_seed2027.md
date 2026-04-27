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
- run name: `policyB_lora_k20_seed2027`
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
- total parameters: 7046041
- trainable parameters: 87060
- total parameter tensors: 430
- trainable parameter tensors: 66
- frozen parameter tensors: 364
- trainable parameter names sample: ['features.denseblock4.denselayer1.conv1.lora_down.weight', 'features.denseblock4.denselayer1.conv1.lora_up.weight', 'features.denseblock4.denselayer1.conv2.lora_down.weight', 'features.denseblock4.denselayer1.conv2.lora_up.weight', 'features.denseblock4.denselayer2.conv1.lora_down.weight', 'features.denseblock4.denselayer2.conv1.lora_up.weight', 'features.denseblock4.denselayer2.conv2.lora_down.weight', 'features.denseblock4.denselayer2.conv2.lora_up.weight', 'features.denseblock4.denselayer3.conv1.lora_down.weight', 'features.denseblock4.denselayer3.conv1.lora_up.weight', 'features.denseblock4.denselayer3.conv2.lora_down.weight', 'features.denseblock4.denselayer3.conv2.lora_up.weight', 'features.denseblock4.denselayer4.conv1.lora_down.weight', 'features.denseblock4.denselayer4.conv1.lora_up.weight', 'features.denseblock4.denselayer4.conv2.lora_down.weight', 'features.denseblock4.denselayer4.conv2.lora_up.weight', 'features.denseblock4.denselayer5.conv1.lora_down.weight', 'features.denseblock4.denselayer5.conv1.lora_up.weight', 'features.denseblock4.denselayer5.conv2.lora_down.weight', 'features.denseblock4.denselayer5.conv2.lora_up.weight']

## Training Outcome
- best epoch: 1
- stopped early: yes
- best checkpoint: `/workspace/experiments/policyB_common5_v1/checkpoints/policyB_lora_k20_seed2027_best.pt`
- last checkpoint: `/workspace/experiments/policyB_common5_v1/checkpoints/policyB_lora_k20_seed2027_last.pt`
- final train loss: 0.6048
- final val loss: 0.7453
- final val macro AUROC: 0.6816
- final val macro AUPRC: 0.2748

## Val Metrics
- loss: 0.6756
- macro AUROC: 0.6856
- macro AUPRC: 0.2771
- micro AUROC: n/a
- micro AUPRC: n/a
- Atelectasis: positives=194, negatives=712, masked=52, n_valid=906, AUROC=0.6312, AUPRC=0.2813, prob_mean=0.6284, prob_std=0.0951
- Cardiomegaly: positives=205, negatives=724, masked=29, n_valid=929, AUROC=0.6493, AUPRC=0.2918, prob_mean=0.3401, prob_std=0.1421
- Consolidation: positives=52, negatives=882, masked=24, n_valid=934, AUROC=0.6773, AUPRC=0.0929, prob_mean=0.5018, prob_std=0.1721
- Edema: positives=131, negatives=767, masked=60, n_valid=898, AUROC=0.7523, AUPRC=0.2980, prob_mean=0.4265, prob_std=0.2152
- Effusion: positives=249, negatives=682, masked=27, n_valid=931, AUROC=0.7178, AUPRC=0.4218, prob_mean=0.4975, prob_std=0.1436

## Test Metrics
- loss: 0.7113
- macro AUROC: 0.6183
- macro AUPRC: 0.3541
- micro AUROC: n/a
- micro AUPRC: n/a
- Atelectasis: positives=170, negatives=394, masked=32, n_valid=564, AUROC=0.5766, AUPRC=0.3500, prob_mean=0.6611, prob_std=0.0805
- Cardiomegaly: positives=159, negatives=424, masked=13, n_valid=583, AUROC=0.5653, AUPRC=0.3172, prob_mean=0.3425, prob_std=0.1273
- Consolidation: positives=56, negatives=530, masked=10, n_valid=586, AUROC=0.5128, AUPRC=0.1169, prob_mean=0.5291, prob_std=0.1468
- Edema: positives=128, negatives=424, masked=44, n_valid=552, AUROC=0.7391, AUPRC=0.4305, prob_mean=0.4591, prob_std=0.1931
- Effusion: positives=219, negatives=350, masked=27, n_valid=569, AUROC=0.6977, AUPRC=0.5561, prob_mean=0.5613, prob_std=0.1213

## Source-only Comparison
- source-only val macro AUROC: 0.6856
- source-only val macro AUPRC: 0.2771
- source-only test macro AUROC: 0.6183
- source-only test macro AUPRC: 0.3541
- val macro AUROC delta: 0.0000
- val macro AUPRC delta: 0.0001
- test macro AUROC delta: -0.0000
- test macro AUPRC delta: 0.0000

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
