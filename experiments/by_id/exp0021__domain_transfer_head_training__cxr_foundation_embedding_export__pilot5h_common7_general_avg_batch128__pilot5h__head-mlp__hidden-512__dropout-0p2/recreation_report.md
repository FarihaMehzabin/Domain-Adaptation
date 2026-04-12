# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/exp0021__domain_transfer_head_training__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128__pilot5h__head-mlp__hidden-512__dropout-0p2`
- Embedding root: `/workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128`
- Manifest: `/workspace/manifest_common_labels_pilot5h.csv`
- Embedding layout: `domain_split`
- Token pooling: `avg`
- Head type: `mlp`
- MLP hidden dims: `[512]`
- MLP dropout: `0.2`

## Recreation Command

```bash
python \
  /workspace/scripts/15_train_domain_transfer_linear_probe.py \
  --embedding-root \
  /workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128 \
  --manifest-csv \
  /workspace/manifest_common_labels_pilot5h.csv \
  --head-type \
  mlp \
  --mlp-hidden-dims \
  512 \
  --mlp-dropout \
  0.2 \
  --batch-size \
  512 \
  --num-workers \
  0 \
  --epochs \
  50 \
  --lr \
  0.001 \
  --weight-decay \
  0.0001 \
  --patience \
  5 \
  --seed \
  1337 \
  --device \
  auto \
  --token-pooling \
  avg \
  --experiment-name \
  cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128__pilot5h__head-mlp__hidden-512__dropout-0p2
```

## Split Inputs

- `d0_train` -> `/workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128/d0_nih/train` with `10000` rows and shape `[10000, 768]`
- `d0_val` -> `/workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128/d0_nih/val` with `1000` rows and shape `[1000, 768]`
- `d0_test` -> `/workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128/d0_nih/test` with `2000` rows and shape `[2000, 768]`
- `d1_transfer` -> `/workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128/d1_chexpert/val` with `234` rows and shape `[234, 768]`
- `d2_transfer` -> `/workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128/d2_mimic/test` with `1455` rows and shape `[1455, 768]`

## Final Metrics

- `d0_test` macro AUROC `0.843756`, macro AP `0.245890`
- `d0_val` macro AUROC `0.851352`, macro AP `0.249979`
- `d1_transfer` macro AUROC `0.842082`, macro AP `0.558525`
- `d2_transfer` macro AUROC `0.505566`, macro AP `0.127651`

## Notes

- Training uses only `d0_train` embeddings.
- Early stopping is driven by `d0_val` macro AUROC.
- Validation-tuned thresholds are reused unchanged for D0 test and transfer evaluations.
