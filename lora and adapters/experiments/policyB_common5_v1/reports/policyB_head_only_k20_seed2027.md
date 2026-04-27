# Mini-Stage F Head-only NIH to MIMIC Adaptation

## Goal
Adapt the NIH-trained DenseNet-121 model to MIMIC by freezing the DenseNet backbone and training only the classifier head on the small support set.

## Inputs
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`
- support manifest: `/workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_support_k20_seed2027.csv`
- val manifest: `/workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_val.csv`
- test manifest: `/workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_test.csv`
- source-only report: `/workspace/experiments/policyB_common5_v1/reports/policyB_no_adaptation_eval_seed2027.json`

## Training Setup
- run name: `policyB_head_only_k20_seed2027`
- epochs: 50
- batch size: 8
- learning rate: 0.0001
- image size: 224
- patience: 10
- seed: 2027
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

## Parameter Freeze Check
- total parameters: 6958981
- trainable parameters: 5125
- trainable parameter names: ['classifier.weight', 'classifier.bias']

## Training Outcome
- best epoch: 1
- stopped early: yes
- best checkpoint: `/workspace/experiments/policyB_common5_v1/checkpoints/policyB_head_only_k20_seed2027_best.pt`
- last checkpoint: `/workspace/experiments/policyB_common5_v1/checkpoints/policyB_head_only_k20_seed2027_last.pt`
- final train loss: 0.5556
- final val loss: 0.9043
- final val macro AUROC: 0.6742
- final val macro AUPRC: 0.2687

## Val Metrics
- loss: 0.7029
- macro AUROC: 0.6844
- macro AUPRC: 0.2762
- micro AUROC: n/a
- micro AUPRC: n/a
- Atelectasis: positives=194, negatives=712, masked=52, n_valid=906, AUROC=0.6296, AUPRC=0.2790, prob_mean=0.6404, prob_std=0.0934
- Cardiomegaly: positives=205, negatives=724, masked=29, n_valid=929, AUROC=0.6466, AUPRC=0.2901, prob_mean=0.3734, prob_std=0.1462
- Consolidation: positives=52, negatives=882, masked=24, n_valid=934, AUROC=0.6770, AUPRC=0.0930, prob_mean=0.5143, prob_std=0.1715
- Edema: positives=131, negatives=767, masked=60, n_valid=898, AUROC=0.7523, AUPRC=0.2985, prob_mean=0.4439, prob_std=0.2168
- Effusion: positives=249, negatives=682, masked=27, n_valid=931, AUROC=0.7164, AUPRC=0.4202, prob_mean=0.5346, prob_std=0.1417

## Test Metrics
- loss: 0.7329
- macro AUROC: 0.6180
- macro AUPRC: 0.3544
- micro AUROC: n/a
- micro AUPRC: n/a
- Atelectasis: positives=170, negatives=394, masked=32, n_valid=564, AUROC=0.5764, AUPRC=0.3507, prob_mean=0.6723, prob_std=0.0789
- Cardiomegaly: positives=159, negatives=424, masked=13, n_valid=583, AUROC=0.5653, AUPRC=0.3199, prob_mean=0.3759, prob_std=0.1309
- Consolidation: positives=56, negatives=530, masked=10, n_valid=586, AUROC=0.5117, AUPRC=0.1162, prob_mean=0.5421, prob_std=0.1462
- Edema: positives=128, negatives=424, masked=44, n_valid=552, AUROC=0.7391, AUPRC=0.4296, prob_mean=0.4771, prob_std=0.1937
- Effusion: positives=219, negatives=350, masked=27, n_valid=569, AUROC=0.6973, AUPRC=0.5557, prob_mean=0.5977, prob_std=0.1180

## Source-only Comparison
- source-only val macro AUROC: 0.6856
- source-only val macro AUPRC: 0.2771
- source-only test macro AUROC: 0.6183
- source-only test macro AUPRC: 0.3541
- val macro AUROC delta: -0.0012
- val macro AUPRC delta: -0.0009
- test macro AUROC delta: -0.0004
- test macro AUPRC delta: 0.0003

## Warnings
- none

## Final Decision
- status: DONE
- safe to continue: yes
