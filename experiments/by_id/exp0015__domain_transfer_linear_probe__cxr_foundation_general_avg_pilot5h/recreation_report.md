# Domain Transfer Linear Probe Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/exp0015__domain_transfer_linear_probe__cxr_foundation_general_avg_pilot5h`
- Embedding root: `/workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128`
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
  /workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128 \
  --batch-size \
  2048 \
  --experiment-name \
  exp0015__domain_transfer_linear_probe__cxr_foundation_general_avg_pilot5h \
  --overwrite
```

## Split Inputs

- `d0_train` -> `/workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128/d0_nih/train` with `10000` rows and shape `[10000, 768]`
- `d0_val` -> `/workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128/d0_nih/val` with `1000` rows and shape `[1000, 768]`
- `d0_test` -> `/workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128/d0_nih/test` with `2000` rows and shape `[2000, 768]`
- `d1_transfer` -> `/workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128/d1_chexpert/val` with `234` rows and shape `[234, 768]`
- `d2_transfer` -> `/workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128/d2_mimic/test` with `1455` rows and shape `[1455, 768]`

## Final Metrics

- `d0_test` macro AUROC `0.845498`, macro AP `0.254104`
- `d0_val` macro AUROC `0.848178`, macro AP `0.266627`
- `d1_transfer` macro AUROC `0.845431`, macro AP `0.543047`
- `d2_transfer` macro AUROC `0.500706`, macro AP `0.125812`

## Notes

- Training uses only `d0_train` embeddings.
- Early stopping is driven by `d0_val` macro AUROC.
- Validation-tuned thresholds are reused unchanged for D0 test and transfer evaluations.
