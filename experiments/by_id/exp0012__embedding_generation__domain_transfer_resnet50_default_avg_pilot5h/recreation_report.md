# Recreation Report

- Experiment: `exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h`
- Manifest: `/workspace/manifest_common_labels_pilot5h.csv`
- Encoder: `torchvision:resnet50`
- Weights: `DEFAULT`
- Pooling: `avg`
- Batch size: `1536`
- Device: `cuda`

## Split Summary
- `d0_nih/test`: 2000 rows, shape=[2000, 2048]
- `d0_nih/train`: 10000 rows, shape=[10000, 2048]
- `d0_nih/val`: 1000 rows, shape=[1000, 2048]
- `d1_chexpert/val`: 234 rows, shape=[234, 2048]
- `d2_mimic/test`: 1455 rows, shape=[1455, 2048]
