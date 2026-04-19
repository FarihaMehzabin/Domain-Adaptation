# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/tmp/cxr_sequential_regularizer_sweeps/experiments/by_id/exp0001__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_lwf__alpha-0p25__temp-2p0__cxr_foundation_linear`
- Embedding root: `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation`
- Manifest: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/manifest/manifest_pilot5h_binary_mimic.csv`
- Embedding layout: `domain_split`
- Token pooling: `avg`
- Head type: `linear`
- Init checkpoint: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/experiments/by_id/exp0001__domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear/best.ckpt`
- Preservation method: `lwf`
- LwF enabled: `True`
- LwF teacher checkpoint: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/experiments/by_id/exp0001__domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear/best.ckpt`
- LwF alpha: `0.25`
- LwF temperature: `2.0`
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
  exp0001__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_lwf__alpha-0p25__temp-2p0__cxr_foundation_linear \
  --split-profile \
  chexpert_adapt_from_nih \
  --init-checkpoint \
  /workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/experiments/by_id/exp0001__domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear/best.ckpt \
  --enable-lwf \
  --lwf-teacher-checkpoint \
  /workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/experiments/by_id/exp0001__domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear/best.ckpt \
  --lwf-alpha \
  0.25 \
  --lwf-temperature \
  2.0 \
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

- `target_train` -> `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d1_chexpert/train` with `1000` rows and shape `[1000, 768]`
- `target_val` -> `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d1_chexpert/val` with `1000` rows and shape `[1000, 768]`
- `target_test` -> `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d1_chexpert/test` with `234` rows and shape `[234, 768]`
- `d0_val` -> `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d0_nih/val` with `1000` rows and shape `[1000, 768]`
- `d0_test` -> `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d0_nih/test` with `2000` rows and shape `[2000, 768]`
- `d2_val` -> `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d2_mimic/val` with `1000` rows and shape `[1000, 768]`
- `d2_test` -> `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d2_mimic/test` with `676` rows and shape `[676, 768]`

## Final Metrics

- `d0_test` macro AUROC `0.821106`, macro AP `0.227142`
- `d0_val` macro AUROC `0.817350`, macro AP `0.241947`
- `d2_test` macro AUROC `0.764238`, macro AP `0.396045`
- `d2_val` macro AUROC `0.778834`, macro AP `0.361367`
- `target_test` macro AUROC `0.787915`, macro AP `0.480848`
- `target_val` macro AUROC `0.736181`, macro AP `0.365168`

## Notes

- Training uses only `target_train` embeddings.
- Auxiliary training aliases: `[]`.
- Initialization checkpoint: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/experiments/by_id/exp0001__domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear/best.ckpt`.
- LwF teacher checkpoint: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/experiments/by_id/exp0001__domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear/best.ckpt`.
- MAS state path: `None`.
- Early stopping is driven by `target_val` macro AUROC.
- `target_val`-tuned thresholds are reused unchanged for later evaluation splits.
