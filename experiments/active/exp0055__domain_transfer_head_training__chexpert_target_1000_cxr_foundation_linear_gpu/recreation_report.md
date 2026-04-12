# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/active/exp0055__domain_transfer_head_training__chexpert_target_1000_cxr_foundation_linear_gpu`
- Embedding root: `/workspace/experiments/active/exp0054__cxr_foundation_embedding_export__chexpert_target_1000_cxr_foundation_avg_batch128`
- Manifest: `/workspace/manifest_chexpert_target_1000.csv`
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
  /workspace/experiments/active/exp0054__cxr_foundation_embedding_export__chexpert_target_1000_cxr_foundation_avg_batch128 \
  --manifest-csv \
  /workspace/manifest_chexpert_target_1000.csv \
  --split-profile \
  chexpert_target \
  --head-type \
  linear \
  --device \
  cuda \
  --fp16-on-cuda \
  --experiment-name \
  chexpert_target_1000_cxr_foundation_linear_gpu
```

## Split Inputs

- `target_train` -> `/workspace/experiments/active/exp0054__cxr_foundation_embedding_export__chexpert_target_1000_cxr_foundation_avg_batch128/d1_chexpert/train` with `1000` rows and shape `[1000, 768]`
- `target_val` -> `/workspace/experiments/active/exp0054__cxr_foundation_embedding_export__chexpert_target_1000_cxr_foundation_avg_batch128/d1_chexpert/val` with `1000` rows and shape `[1000, 768]`
- `target_test` -> `/workspace/experiments/active/exp0054__cxr_foundation_embedding_export__chexpert_target_1000_cxr_foundation_avg_batch128/d1_chexpert/test` with `234` rows and shape `[234, 768]`

## Final Metrics

- `target_test` macro AUROC `0.770211`, macro AP `0.471481`
- `target_val` macro AUROC `0.732933`, macro AP `0.361403`

## Notes

- Training uses only `target_train` embeddings.
- Early stopping is driven by `target_val` macro AUROC.
- `target_val`-tuned thresholds are reused unchanged for later evaluation splits.
