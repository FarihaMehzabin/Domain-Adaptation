# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/by_id/exp0075__domain_transfer_head_training__nih_warmstart_adaptation__chexpert_target_1000__cxr_foundation_linear_gpu`
- Embedding root: `/workspace/experiments/by_id/exp0074__domain_split_embedding_view__nih_pilot5h_plus_chexpert_target_1000_cxr_foundation`
- Manifest: `/workspace/manifest/manifest_nih_pilot5h_chexpert_target_1000_adapt.csv`
- Embedding layout: `domain_split`
- Token pooling: `avg`
- Head type: `linear`
- Init checkpoint: `/workspace/experiments/by_id/exp0015__domain_transfer_linear_probe__cxr_foundation_general_avg_pilot5h/best.ckpt`
- MLP hidden dims: `[]`
- MLP dropout: `0.2`

## Recreation Command

```bash
python \
  /workspace/scripts/15_train_domain_transfer_linear_probe.py \
  --embedding-root \
  /workspace/experiments/by_id/exp0074__domain_split_embedding_view__nih_pilot5h_plus_chexpert_target_1000_cxr_foundation \
  --manifest-csv \
  /workspace/manifest/manifest_nih_pilot5h_chexpert_target_1000_adapt.csv \
  --split-profile \
  chexpert_adapt_from_nih \
  --init-checkpoint \
  /workspace/experiments/by_id/exp0015__domain_transfer_linear_probe__cxr_foundation_general_avg_pilot5h/best.ckpt \
  --head-type \
  linear \
  --device \
  cuda \
  --fp16-on-cuda \
  --batch-size \
  512 \
  --experiments-root \
  /workspace/experiments/by_id \
  --experiment-name \
  exp0075__domain_transfer_head_training__nih_warmstart_adaptation__chexpert_target_1000__cxr_foundation_linear_gpu
```

## Split Inputs

- `target_train` -> `/workspace/experiments/by_id/exp0074__domain_split_embedding_view__nih_pilot5h_plus_chexpert_target_1000_cxr_foundation/d1_chexpert/train` with `1000` rows and shape `[1000, 768]`
- `target_val` -> `/workspace/experiments/by_id/exp0074__domain_split_embedding_view__nih_pilot5h_plus_chexpert_target_1000_cxr_foundation/d1_chexpert/val` with `1000` rows and shape `[1000, 768]`
- `target_test` -> `/workspace/experiments/by_id/exp0074__domain_split_embedding_view__nih_pilot5h_plus_chexpert_target_1000_cxr_foundation/d1_chexpert/test` with `234` rows and shape `[234, 768]`
- `d0_test` -> `/workspace/experiments/by_id/exp0074__domain_split_embedding_view__nih_pilot5h_plus_chexpert_target_1000_cxr_foundation/d0_nih/test` with `2000` rows and shape `[2000, 768]`

## Final Metrics

- `d0_test` macro AUROC `0.811204`, macro AP `0.218772`
- `target_test` macro AUROC `0.775738`, macro AP `0.477016`
- `target_val` macro AUROC `0.734992`, macro AP `0.365678`

## Notes

- Training uses only `target_train` embeddings.
- Initialization checkpoint: `/workspace/experiments/by_id/exp0015__domain_transfer_linear_probe__cxr_foundation_general_avg_pilot5h/best.ckpt`.
- Early stopping is driven by `target_val` macro AUROC.
- `target_val`-tuned thresholds are reused unchanged for later evaluation splits.
