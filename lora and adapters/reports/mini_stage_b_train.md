# Mini-Stage B Train Report

- mode: `train`
- device: `cuda`
- pretrained used: `True`

## Split Sizes
- train images: 1400
- val images: 200
- test images: 400

## Pos Weight
- Atelectasis: 9.8527
- Cardiomegaly: 55.0000
- Consolidation: 17.6667
- Edema: 45.6667
- Effusion: 7.0460

## Epoch History
- epoch 1: train_loss=1.1422, val_loss=1.7558, val_macro_auroc=0.6843
- epoch 2: train_loss=0.7994, val_loss=2.4405, val_macro_auroc=0.6863
- epoch 3: train_loss=0.5769, val_loss=2.5727, val_macro_auroc=0.6888

## Val Metrics
- loss: 1.7558
- macro AUROC: 0.6843
- macro AUPRC: 0.1705
- per-label metrics:
  - Atelectasis: AUROC=0.6417, AUPRC=0.2274, positives=29
  - Cardiomegaly: AUROC=0.6800, AUPRC=0.1439, positives=13
  - Consolidation: AUROC=0.5804, AUPRC=0.0290, positives=4
  - Edema: AUROC=0.7343, AUPRC=0.0687, positives=7
  - Effusion: AUROC=0.7850, AUPRC=0.3833, positives=29

## Test Metrics
- loss: 1.3322
- macro AUROC: 0.6574
- macro AUPRC: 0.1549
- per-label metrics:
  - Atelectasis: AUROC=0.5385, AUPRC=0.1375, positives=31
  - Cardiomegaly: AUROC=0.5762, AUPRC=0.0965, positives=7
  - Consolidation: AUROC=0.7631, AUPRC=0.1696, positives=30
  - Edema: AUROC=0.7450, AUPRC=0.0672, positives=14
  - Effusion: AUROC=0.6645, AUPRC=0.3035, positives=86
