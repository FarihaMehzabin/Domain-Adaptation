# Policy B LoRA + BN-head NIH to MIMIC Adaptation

## Goal
Adapt the NIH-trained DenseNet-121 model to MIMIC using LoRA on denseblock4, full classifier-head adaptation, and BN affine adaptation on denseblock4 + norm5.

## Inputs
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`
- support manifest: `/workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_support_k20_seed2027.csv`
- val manifest: `/workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_val.csv`
- test manifest: `/workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_test.csv`
- source-only report: `/workspace/experiments/policyB_common5_v1/reports/policyB_no_adaptation_eval_seed2027.json`

## Training Setup
- run name: `policyB_lora_bnhead_k20_seed2027`
- epochs: 50
- batch size: 8
- image size: 224
- patience: 10
- seed: 2027
- LoRA rank: 8
- LoRA alpha: 16.0
- LoRA dropout: 0.05
- LoRA lr / wd: 0.0003 / 0.0001
- classifier lr / wd: 0.0001 / 0.0001
- BN affine lr / wd: 0.0001 / 0.0
- BN mode: `frozen_stats`
- primary model selection metric: `val_macro_auprc`
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

## Trainability
- LoRA target module count: 32
- BN affine module count: 33
- total parameters: 7124869
- trainable parameters: 201221
- trainable percentage: 2.8242063117230645
- trainable groups: LoRA=165888, classifier=5125, bn_affine=30208
- trainable parameter names sample: ['features.denseblock4.denselayer1.norm1.weight', 'features.denseblock4.denselayer1.norm1.bias', 'features.denseblock4.denselayer1.conv1.lora_down.weight', 'features.denseblock4.denselayer1.conv1.lora_up.weight', 'features.denseblock4.denselayer1.norm2.weight', 'features.denseblock4.denselayer1.norm2.bias', 'features.denseblock4.denselayer1.conv2.lora_down.weight', 'features.denseblock4.denselayer1.conv2.lora_up.weight', 'features.denseblock4.denselayer2.norm1.weight', 'features.denseblock4.denselayer2.norm1.bias', 'features.denseblock4.denselayer2.conv1.lora_down.weight', 'features.denseblock4.denselayer2.conv1.lora_up.weight', 'features.denseblock4.denselayer2.norm2.weight', 'features.denseblock4.denselayer2.norm2.bias', 'features.denseblock4.denselayer2.conv2.lora_down.weight', 'features.denseblock4.denselayer2.conv2.lora_up.weight', 'features.denseblock4.denselayer3.norm1.weight', 'features.denseblock4.denselayer3.norm1.bias', 'features.denseblock4.denselayer3.conv1.lora_down.weight', 'features.denseblock4.denselayer3.conv1.lora_up.weight']
- original non-LoRA conv weights frozen: yes
- classifier base changed allowed: True

## Training Outcome
- best epoch: 12
- stopped early: yes
- best checkpoint: `/workspace/experiments/policyB_common5_v1/checkpoints/policyB_lora_bnhead_k20_seed2027_best.pt`
- last checkpoint: `/workspace/experiments/policyB_common5_v1/checkpoints/policyB_lora_bnhead_k20_seed2027_last.pt`
- final train loss: 0.0174
- final val loss: 2.1763
- final val macro AUROC: 0.6481
- final val macro AUPRC: 0.2645

## Parameter Change Checks
- classifier changed: yes
- BN affine changed: yes
- classifier max abs diff: 0.0037819771096110344
- BN affine max abs diff: 0.0043861158192157745

## Val Metrics
- loss: 1.0095
- macro AUROC: 0.6763
- macro AUPRC: 0.2809
- micro AUROC: n/a
- micro AUPRC: n/a
- Atelectasis: positives=194, negatives=712, masked=52, n_valid=906, AUROC=0.5676, AUPRC=0.2559, delta_AUROC=-0.0635, delta_AUPRC=-0.0250, prob_mean=0.8091, prob_std=0.0939
- Cardiomegaly: positives=205, negatives=724, masked=29, n_valid=929, AUROC=0.6709, AUPRC=0.3129, delta_AUROC=0.0217, delta_AUPRC=0.0210, prob_mean=0.5053, prob_std=0.2146
- Consolidation: positives=52, negatives=882, masked=24, n_valid=934, AUROC=0.6643, AUPRC=0.1067, delta_AUROC=-0.0131, delta_AUPRC=0.0138, prob_mean=0.5984, prob_std=0.2143
- Edema: positives=131, negatives=767, masked=60, n_valid=898, AUROC=0.7520, AUPRC=0.2883, delta_AUROC=-0.0003, delta_AUPRC=-0.0095, prob_mean=0.3193, prob_std=0.3020
- Effusion: positives=249, negatives=682, masked=27, n_valid=931, AUROC=0.7267, AUPRC=0.4405, delta_AUROC=0.0089, delta_AUPRC=0.0187, prob_mean=0.7922, prob_std=0.1910

## Test Metrics
- loss: 1.0512
- macro AUROC: 0.6182
- macro AUPRC: 0.3493
- micro AUROC: n/a
- micro AUPRC: n/a
- Atelectasis: positives=170, negatives=394, masked=32, n_valid=564, AUROC=0.5663, AUPRC=0.3618, delta_AUROC=-0.0104, delta_AUPRC=0.0120, prob_mean=0.8254, prob_std=0.0806
- Cardiomegaly: positives=159, negatives=424, masked=13, n_valid=583, AUROC=0.6191, AUPRC=0.3605, delta_AUROC=0.0541, delta_AUPRC=0.0434, prob_mean=0.5171, prob_std=0.2009
- Consolidation: positives=56, negatives=530, masked=10, n_valid=586, AUROC=0.4903, AUPRC=0.0928, delta_AUROC=-0.0229, delta_AUPRC=-0.0243, prob_mean=0.6641, prob_std=0.1929
- Edema: positives=128, negatives=424, masked=44, n_valid=552, AUROC=0.7509, AUPRC=0.4370, delta_AUROC=0.0117, delta_AUPRC=0.0063, prob_mean=0.3480, prob_std=0.2898
- Effusion: positives=219, negatives=350, masked=27, n_valid=569, AUROC=0.6643, AUPRC=0.4944, delta_AUROC=-0.0333, delta_AUPRC=-0.0614, prob_mean=0.8735, prob_std=0.1297

## Source-only Comparison
- source-only val macro AUROC: 0.6856
- source-only val macro AUPRC: 0.2771
- source-only test macro AUROC: 0.6183
- source-only test macro AUPRC: 0.3541
- val macro AUROC delta: -0.0093
- val macro AUPRC delta: 0.0038
- test macro AUROC delta: -0.0002
- test macro AUPRC delta: -0.0048

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
