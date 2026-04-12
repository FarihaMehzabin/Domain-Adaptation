# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/active/exp0061__domain_transfer_head_training__mimic_target_500_cxr_foundation_linear_gpu`
- Embedding root: `/workspace/experiments/active/exp0058__cxr_foundation_embedding_export__mimic_target_500_cxr_foundation_avg_batch128`
- Manifest: `/workspace/manifest/manifest_mimic_target_500_nested.csv`
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
  /workspace/experiments/active/exp0058__cxr_foundation_embedding_export__mimic_target_500_cxr_foundation_avg_batch128 \
  --manifest-csv \
  /workspace/manifest/manifest_mimic_target_500_nested.csv \
  --split-profile \
  mimic_target \
  --head-type \
  linear \
  --device \
  cuda \
  --fp16-on-cuda \
  --experiment-name \
  mimic_target_500_cxr_foundation_linear_gpu
```

## Split Inputs

- `target_train` -> `/workspace/experiments/active/exp0058__cxr_foundation_embedding_export__mimic_target_500_cxr_foundation_avg_batch128/d2_mimic/train` with `500` rows and shape `[500, 768]`
- `target_val` -> `/workspace/experiments/active/exp0058__cxr_foundation_embedding_export__mimic_target_500_cxr_foundation_avg_batch128/d2_mimic/val` with `500` rows and shape `[500, 768]`
- `target_test` -> `/workspace/experiments/active/exp0058__cxr_foundation_embedding_export__mimic_target_500_cxr_foundation_avg_batch128/d2_mimic/test` with `1455` rows and shape `[1455, 768]`

## Final Metrics

- `target_test` macro AUROC `0.502071`, macro AP `0.127760`
- `target_val` macro AUROC `0.513520`, macro AP `0.161284`

## Notes

- Training uses only `target_train` embeddings.
- Early stopping is driven by `target_val` macro AUROC.
- `target_val`-tuned thresholds are reused unchanged for later evaluation splits.
