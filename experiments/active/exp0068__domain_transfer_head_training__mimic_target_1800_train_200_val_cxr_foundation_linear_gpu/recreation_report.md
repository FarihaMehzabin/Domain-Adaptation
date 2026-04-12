# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/active/exp0068__domain_transfer_head_training__mimic_target_1800_train_200_val_cxr_foundation_linear_gpu`
- Embedding root: `/workspace/experiments/active/exp0067__cxr_foundation_embedding_export__mimic_target_1800_train_200_val_from_exp0057`
- Manifest: `/workspace/manifest/manifest_mimic_target_1800_train_200_val.csv`
- Embedding layout: `domain_split`
- Token pooling: `avg`
- Head type: `linear`
- MLP hidden dims: `[]`
- MLP dropout: `0.2`

## Recreation Command

```bash
python \
  /workspace/scripts/15_train_domain_transfer_linear_probe.py \
  --embedding-root \
  /workspace/experiments/active/exp0067__cxr_foundation_embedding_export__mimic_target_1800_train_200_val_from_exp0057 \
  --manifest-csv \
  /workspace/manifest/manifest_mimic_target_1800_train_200_val.csv \
  --split-profile \
  mimic_target \
  --head-type \
  linear \
  --device \
  cuda \
  --fp16-on-cuda \
  --experiment-name \
  mimic_target_1800_train_200_val_cxr_foundation_linear_gpu
```

## Split Inputs

- `target_train` -> `/workspace/experiments/active/exp0067__cxr_foundation_embedding_export__mimic_target_1800_train_200_val_from_exp0057/d2_mimic/train` with `1800` rows and shape `[1800, 768]`
- `target_val` -> `/workspace/experiments/active/exp0067__cxr_foundation_embedding_export__mimic_target_1800_train_200_val_from_exp0057/d2_mimic/val` with `200` rows and shape `[200, 768]`
- `target_test` -> `/workspace/experiments/active/exp0067__cxr_foundation_embedding_export__mimic_target_1800_train_200_val_from_exp0057/d2_mimic/test` with `1455` rows and shape `[1455, 768]`

## Final Metrics

- `target_test` macro AUROC `0.501900`, macro AP `0.127822`
- `target_val` macro AUROC `0.507839`, macro AP `0.152033`

## Notes

- Training uses only `target_train` embeddings.
- Early stopping is driven by `target_val` macro AUROC.
- `target_val`-tuned thresholds are reused unchanged for later evaluation splits.
