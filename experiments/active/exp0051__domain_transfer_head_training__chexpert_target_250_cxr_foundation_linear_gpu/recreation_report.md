# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/active/exp0051__domain_transfer_head_training__chexpert_target_250_cxr_foundation_linear_gpu`
- Embedding root: `/workspace/experiments/active/exp0050__cxr_foundation_embedding_export__chexpert_target_250_cxr_foundation_avg_batch128`
- Manifest: `/workspace/manifest_chexpert_target_250.csv`
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
  /workspace/experiments/active/exp0050__cxr_foundation_embedding_export__chexpert_target_250_cxr_foundation_avg_batch128 \
  --manifest-csv \
  /workspace/manifest_chexpert_target_250.csv \
  --split-profile \
  chexpert_target \
  --head-type \
  linear \
  --device \
  cuda \
  --fp16-on-cuda \
  --experiment-name \
  chexpert_target_250_cxr_foundation_linear_gpu
```

## Split Inputs

- `target_train` -> `/workspace/experiments/active/exp0050__cxr_foundation_embedding_export__chexpert_target_250_cxr_foundation_avg_batch128/d1_chexpert/train` with `250` rows and shape `[250, 768]`
- `target_val` -> `/workspace/experiments/active/exp0050__cxr_foundation_embedding_export__chexpert_target_250_cxr_foundation_avg_batch128/d1_chexpert/val` with `250` rows and shape `[250, 768]`
- `target_test` -> `/workspace/experiments/active/exp0050__cxr_foundation_embedding_export__chexpert_target_250_cxr_foundation_avg_batch128/d1_chexpert/test` with `234` rows and shape `[234, 768]`

## Final Metrics

- `target_test` macro AUROC `0.753186`, macro AP `0.476353`
- `target_val` macro AUROC `0.727310`, macro AP `0.353035`

## Notes

- Training uses only `target_train` embeddings.
- Early stopping is driven by `target_val` macro AUROC.
- `target_val`-tuned thresholds are reused unchanged for later evaluation splits.
