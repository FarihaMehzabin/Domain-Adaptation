# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/by_id/exp0053__domain_transfer_head_training__chexpert_target_500_cxr_foundation_linear_gpu`
- Embedding root: `/workspace/experiments/by_id/exp0052__cxr_foundation_embedding_export__chexpert_target_500_cxr_foundation_avg_batch128`
- Manifest: `/workspace/manifest_chexpert_target_500.csv`
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
  /workspace/experiments/by_id/exp0052__cxr_foundation_embedding_export__chexpert_target_500_cxr_foundation_avg_batch128 \
  --manifest-csv \
  /workspace/manifest_chexpert_target_500.csv \
  --split-profile \
  chexpert_target \
  --head-type \
  linear \
  --device \
  cuda \
  --fp16-on-cuda \
  --experiment-name \
  chexpert_target_500_cxr_foundation_linear_gpu
```

## Split Inputs

- `target_train` -> `/workspace/experiments/by_id/exp0052__cxr_foundation_embedding_export__chexpert_target_500_cxr_foundation_avg_batch128/d1_chexpert/train` with `500` rows and shape `[500, 768]`
- `target_val` -> `/workspace/experiments/by_id/exp0052__cxr_foundation_embedding_export__chexpert_target_500_cxr_foundation_avg_batch128/d1_chexpert/val` with `500` rows and shape `[500, 768]`
- `target_test` -> `/workspace/experiments/by_id/exp0052__cxr_foundation_embedding_export__chexpert_target_500_cxr_foundation_avg_batch128/d1_chexpert/test` with `234` rows and shape `[234, 768]`

## Final Metrics

- `target_test` macro AUROC `0.761085`, macro AP `0.452167`
- `target_val` macro AUROC `0.736479`, macro AP `0.348068`

## Notes

- Training uses only `target_train` embeddings.
- Early stopping is driven by `target_val` macro AUROC.
- `target_val`-tuned thresholds are reused unchanged for later evaluation splits.
