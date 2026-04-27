# Mini-Stage F Head-only NIH to MIMIC Adaptation

## Goal
Adapt the NIH-trained DenseNet-121 model to MIMIC by freezing the DenseNet backbone and training only the classifier head on the small support set.

## Inputs
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`
- support manifest: `/workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_support_k5_seed2027.csv`
- val manifest: `/workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_val.csv`
- test manifest: `/workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_test.csv`
- source-only report: `/workspace/experiments/policyB_common5_v1/reports/policyB_no_adaptation_eval_seed2027.json`

## Training Setup
- run name: `policyB_head_only_k5_seed2027`
- epochs: 50
- batch size: 8
- learning rate: 0.0001
- image size: 224
- patience: 10
- seed: 2027
- device: `cuda`

## Split Sizes
### support
- images: 7
- subjects: 7
- studies: 7
- label policy: uignore_blankzero
- dicoms: 7
- Atelectasis: positives=6, negatives=0, masked=1, n_valid=6
- Cardiomegaly: positives=5, negatives=2, masked=0, n_valid=7
- Consolidation: positives=5, negatives=2, masked=0, n_valid=7
- Edema: positives=6, negatives=1, masked=0, n_valid=7
- Effusion: positives=7, negatives=0, masked=0, n_valid=7

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

## Parameter Freeze Check
- total parameters: 6958981
- trainable parameters: 5125
- trainable parameter names: ['classifier.weight', 'classifier.bias']

## Training Outcome
- best epoch: 1
- stopped early: yes
- best checkpoint: `/workspace/experiments/policyB_common5_v1/checkpoints/policyB_head_only_k5_seed2027_best.pt`
- last checkpoint: `/workspace/experiments/policyB_common5_v1/checkpoints/policyB_head_only_k5_seed2027_last.pt`
- final train loss: 0.5001
- final val loss: 0.8118
- final val macro AUROC: 0.6775
- final val macro AUPRC: 0.2716

## Val Metrics
- loss: 0.6851
- macro AUROC: 0.6849
- macro AUPRC: 0.2765
- micro AUROC: n/a
- micro AUPRC: n/a
- Atelectasis: positives=194, negatives=712, masked=52, n_valid=906, AUROC=0.6292, AUPRC=0.2792, prob_mean=0.6392, prob_std=0.0937
- Cardiomegaly: positives=205, negatives=724, masked=29, n_valid=929, AUROC=0.6485, AUPRC=0.2913, prob_mean=0.3470, prob_std=0.1431
- Consolidation: positives=52, negatives=882, masked=24, n_valid=934, AUROC=0.6774, AUPRC=0.0930, prob_mean=0.5037, prob_std=0.1721
- Edema: positives=131, negatives=767, masked=60, n_valid=898, AUROC=0.7522, AUPRC=0.2980, prob_mean=0.4344, prob_std=0.2158
- Effusion: positives=249, negatives=682, masked=27, n_valid=931, AUROC=0.7173, AUPRC=0.4211, prob_mean=0.5048, prob_std=0.1432

## Test Metrics
- loss: 0.7189
- macro AUROC: 0.6182
- macro AUPRC: 0.3538
- micro AUROC: n/a
- micro AUPRC: n/a
- Atelectasis: positives=170, negatives=394, masked=32, n_valid=564, AUROC=0.5765, AUPRC=0.3494, prob_mean=0.6713, prob_std=0.0793
- Cardiomegaly: positives=159, negatives=424, masked=13, n_valid=583, AUROC=0.5650, AUPRC=0.3165, prob_mean=0.3493, prob_std=0.1282
- Consolidation: positives=56, negatives=530, masked=10, n_valid=586, AUROC=0.5127, AUPRC=0.1168, prob_mean=0.5313, prob_std=0.1468
- Edema: positives=128, negatives=424, masked=44, n_valid=552, AUROC=0.7390, AUPRC=0.4303, prob_mean=0.4673, prob_std=0.1933
- Effusion: positives=219, negatives=350, masked=27, n_valid=569, AUROC=0.6978, AUPRC=0.5560, prob_mean=0.5684, prob_std=0.1207

## Source-only Comparison
- source-only val macro AUROC: 0.6856
- source-only val macro AUPRC: 0.2771
- source-only test macro AUROC: 0.6183
- source-only test macro AUPRC: 0.3541
- val macro AUROC delta: -0.0007
- val macro AUPRC delta: -0.0005
- test macro AUROC delta: -0.0001
- test macro AUPRC delta: -0.0003

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
