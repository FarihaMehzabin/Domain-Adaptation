# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/tmp/cxr_sequential_regularizer_sweeps/experiments/by_id/exp0016__domain_transfer_head_training__mimic_adapt_from_chexpert__pilot5h_lwf__alpha-1p0__temp-4p0__cxr_foundation_linear`
- Embedding root: `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation`
- Manifest: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/manifest/manifest_pilot5h_binary_mimic.csv`
- Embedding layout: `domain_split`
- Token pooling: `avg`
- Head type: `linear`
- Init checkpoint: `/tmp/cxr_sequential_regularizer_sweeps/experiments/by_id/exp0015__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_lwf__alpha-1p0__temp-4p0__cxr_foundation_linear/best.ckpt`
- Preservation method: `lwf`
- LwF enabled: `True`
- LwF teacher checkpoint: `/tmp/cxr_sequential_regularizer_sweeps/experiments/by_id/exp0015__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_lwf__alpha-1p0__temp-4p0__cxr_foundation_linear/best.ckpt`
- LwF alpha: `1.0`
- LwF temperature: `4.0`
- MAS enabled: `False`
- MAS state path: `None`
- MAS lambda: `None`
- MLP hidden dims: `[]`
- MLP dropout: `0.2`

## Recreation Command

```bash
python \
  /workspace/scripts/15_train_domain_transfer_linear_probe.py \
  --experiment-name \
  exp0016__domain_transfer_head_training__mimic_adapt_from_chexpert__pilot5h_lwf__alpha-1p0__temp-4p0__cxr_foundation_linear \
  --split-profile \
  mimic_adapt_from_chexpert \
  --init-checkpoint \
  /tmp/cxr_sequential_regularizer_sweeps/experiments/by_id/exp0015__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_lwf__alpha-1p0__temp-4p0__cxr_foundation_linear/best.ckpt \
  --enable-lwf \
  --lwf-teacher-checkpoint \
  /tmp/cxr_sequential_regularizer_sweeps/experiments/by_id/exp0015__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_lwf__alpha-1p0__temp-4p0__cxr_foundation_linear/best.ckpt \
  --lwf-alpha \
  1.0 \
  --lwf-temperature \
  4.0 \
  --embedding-root \
  /workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation \
  --manifest-csv \
  /workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/manifest/manifest_pilot5h_binary_mimic.csv \
  --experiments-root \
  /tmp/cxr_sequential_regularizer_sweeps/experiments/by_id \
  --embedding-layout \
  domain_split \
  --token-pooling \
  avg \
  --head-type \
  linear \
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
  --overwrite
```

## Split Inputs

- `target_train` -> `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d2_mimic/train` with `998` rows and shape `[998, 768]`
- `target_val` -> `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d2_mimic/val` with `1000` rows and shape `[1000, 768]`
- `target_test` -> `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d2_mimic/test` with `676` rows and shape `[676, 768]`
- `d0_val` -> `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d0_nih/val` with `1000` rows and shape `[1000, 768]`
- `d0_test` -> `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d0_nih/test` with `2000` rows and shape `[2000, 768]`
- `d1_val` -> `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d1_chexpert/val` with `1000` rows and shape `[1000, 768]`
- `d1_test` -> `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d1_chexpert/test` with `234` rows and shape `[234, 768]`

## Final Metrics

- `d0_test` macro AUROC `0.831144`, macro AP `0.230739`
- `d0_val` macro AUROC `0.835412`, macro AP `0.246746`
- `d1_test` macro AUROC `0.822744`, macro AP `0.519062`
- `d1_val` macro AUROC `0.730658`, macro AP `0.349838`
- `target_test` macro AUROC `0.776732`, macro AP `0.415710`
- `target_val` macro AUROC `0.804796`, macro AP `0.389731`

## Notes

- Training uses only `target_train` embeddings.
- Auxiliary training aliases: `[]`.
- Initialization checkpoint: `/tmp/cxr_sequential_regularizer_sweeps/experiments/by_id/exp0015__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_lwf__alpha-1p0__temp-4p0__cxr_foundation_linear/best.ckpt`.
- LwF teacher checkpoint: `/tmp/cxr_sequential_regularizer_sweeps/experiments/by_id/exp0015__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_lwf__alpha-1p0__temp-4p0__cxr_foundation_linear/best.ckpt`.
- MAS state path: `None`.
- Early stopping is driven by `target_val` macro AUROC.
- `target_val`-tuned thresholds are reused unchanged for later evaluation splits.
