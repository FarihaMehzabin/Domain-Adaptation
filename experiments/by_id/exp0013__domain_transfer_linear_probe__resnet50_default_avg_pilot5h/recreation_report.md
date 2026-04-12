# Domain Transfer Linear Probe Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/exp0013__domain_transfer_linear_probe__resnet50_default_avg_pilot5h`
- Embedding root: `/workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h`
- Manifest: `/workspace/manifest_common_labels_pilot5h.csv`
- Embedding layout: `domain_split`
- Token pooling: `avg`

## Recreation Command

```bash
python \
  /workspace/scripts/15_train_domain_transfer_linear_probe.py \
  --manifest-csv \
  /workspace/manifest_common_labels_pilot5h.csv \
  --embedding-root \
  /workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h \
  --batch-size \
  2048 \
  --experiment-name \
  exp0013__domain_transfer_linear_probe__resnet50_default_avg_pilot5h \
  --overwrite
```

## Split Inputs

- `d0_train` -> `/workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h/d0_nih/train` with `10000` rows and shape `[10000, 2048]`
- `d0_val` -> `/workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h/d0_nih/val` with `1000` rows and shape `[1000, 2048]`
- `d0_test` -> `/workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h/d0_nih/test` with `2000` rows and shape `[2000, 2048]`
- `d1_transfer` -> `/workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h/d1_chexpert/val` with `234` rows and shape `[234, 2048]`
- `d2_transfer` -> `/workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h/d2_mimic/test` with `1455` rows and shape `[1455, 2048]`

## Final Metrics

- `d0_test` macro AUROC `0.730563`, macro AP `0.126266`
- `d0_val` macro AUROC `0.737617`, macro AP `0.127283`
- `d1_transfer` macro AUROC `0.721777`, macro AP `0.374841`
- `d2_transfer` macro AUROC `0.499575`, macro AP `0.127804`

## Notes

- Training uses only `d0_train` embeddings.
- Early stopping is driven by `d0_val` macro AUROC.
- Validation-tuned thresholds are reused unchanged for D0 test and transfer evaluations.
