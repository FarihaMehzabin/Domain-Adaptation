# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/by_id/exp0062__domain_transfer_head_training__mimic_target_1000_cxr_foundation_linear_gpu`
- Embedding root: `/workspace/experiments/by_id/exp0057__cxr_foundation_embedding_export__mimic_target_1000_cxr_foundation_avg_batch128`
- Manifest: `/workspace/manifest/manifest_mimic_target_1000.csv`
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
  /workspace/experiments/by_id/exp0057__cxr_foundation_embedding_export__mimic_target_1000_cxr_foundation_avg_batch128 \
  --manifest-csv \
  /workspace/manifest/manifest_mimic_target_1000.csv \
  --split-profile \
  mimic_target \
  --head-type \
  linear \
  --device \
  cuda \
  --fp16-on-cuda \
  --experiment-name \
  mimic_target_1000_cxr_foundation_linear_gpu \
  --overwrite
```

## Split Inputs

- `target_train` -> `/workspace/experiments/by_id/exp0057__cxr_foundation_embedding_export__mimic_target_1000_cxr_foundation_avg_batch128/d2_mimic/train` with `1000` rows and shape `[1000, 768]`
- `target_val` -> `/workspace/experiments/by_id/exp0057__cxr_foundation_embedding_export__mimic_target_1000_cxr_foundation_avg_batch128/d2_mimic/val` with `1000` rows and shape `[1000, 768]`
- `target_test` -> `/workspace/experiments/by_id/exp0057__cxr_foundation_embedding_export__mimic_target_1000_cxr_foundation_avg_batch128/d2_mimic/test` with `1455` rows and shape `[1455, 768]`

## Final Metrics

- `target_test` macro AUROC `0.501507`, macro AP `0.128013`
- `target_val` macro AUROC `0.512529`, macro AP `0.147834`

## Notes

- Training uses only `target_train` embeddings.
- Early stopping is driven by `target_val` macro AUROC.
- `target_val`-tuned thresholds are reused unchanged for later evaluation splits.
