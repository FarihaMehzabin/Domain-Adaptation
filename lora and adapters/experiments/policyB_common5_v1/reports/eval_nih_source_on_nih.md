# Mini-Stage D NIH to MIMIC Evaluation

## Goal
Evaluate the trained NIH 2k DenseNet-121 model directly on MIMIC common5 without any target adaptation.

## Checkpoint Used
- checkpoint: `/workspace/checkpoints/nih_2k_densenet121_best.pt`

## MIMIC Manifests Used
- val: `/workspace/experiments/policyB_common5_v1/manifests/nih_dev_2k_val.csv`
- test: `/workspace/experiments/policyB_common5_v1/manifests/nih_dev_2k_test.csv`

## Split Sizes
- val images: 0
- val subjects: 0
- val studies: 0
- test images: 0
- test subjects: 0
- test studies: 0

## Label Counts
### Val
- Atelectasis: positives=0, negatives=0, masked=0, n_valid=0
- Cardiomegaly: positives=0, negatives=0, masked=0, n_valid=0
- Consolidation: positives=0, negatives=0, masked=0, n_valid=0
- Edema: positives=0, negatives=0, masked=0, n_valid=0
- Effusion: positives=0, negatives=0, masked=0, n_valid=0

### Test
- Atelectasis: positives=0, negatives=0, masked=0, n_valid=0
- Cardiomegaly: positives=0, negatives=0, masked=0, n_valid=0
- Consolidation: positives=0, negatives=0, masked=0, n_valid=0
- Edema: positives=0, negatives=0, masked=0, n_valid=0
- Effusion: positives=0, negatives=0, masked=0, n_valid=0

## Val Metrics
- not available

## Test Metrics
- not available

## Warnings
- none

## Final Decision
- status: FAILED
- safe to continue: no

## Failure Reasons
- /workspace/experiments/policyB_common5_v1/manifests/nih_dev_2k_val.csv is missing required column: subject_id
