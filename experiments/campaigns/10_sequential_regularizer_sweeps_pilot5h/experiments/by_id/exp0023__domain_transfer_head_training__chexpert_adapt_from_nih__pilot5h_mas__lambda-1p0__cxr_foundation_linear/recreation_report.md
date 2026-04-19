# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/tmp/cxr_sequential_regularizer_sweeps/experiments/by_id/exp0023__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_mas__lambda-1p0__cxr_foundation_linear`
- Embedding root: `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation`
- Manifest: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/manifest/manifest_pilot5h_binary_mimic.csv`
- Embedding layout: `domain_split`
- Token pooling: `avg`
- Head type: `linear`
- Init checkpoint: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/experiments/by_id/exp0001__domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear/best.ckpt`
- Preservation method: `mas`
- LwF enabled: `False`
- LwF teacher checkpoint: `None`
- LwF alpha: `None`
- LwF temperature: `None`
- MAS enabled: `True`
- MAS state path: `/tmp/cxr_sequential_regularizer_sweeps/mas_states/stage_a_nih_source_mas_state.pt`
- MAS lambda: `1.0`
- MLP hidden dims: `[]`
- MLP dropout: `0.2`

## Recreation Command

```bash
python \
  /workspace/scripts/15_train_domain_transfer_linear_probe.py \
  --experiment-name \
  exp0023__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_mas__lambda-1p0__cxr_foundation_linear \
  --split-profile \
  chexpert_adapt_from_nih \
  --init-checkpoint \
  /workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/experiments/by_id/exp0001__domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear/best.ckpt \
  --enable-mas \
  --mas-state-path \
  /tmp/cxr_sequential_regularizer_sweeps/mas_states/stage_a_nih_source_mas_state.pt \
  --mas-lambda \
  1.0 \
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

- `d0_test` macro AUROC `0.842564`, macro AP `0.252503`
- `d0_val` macro AUROC `0.842362`, macro AP `0.257076`
- `d2_test` macro AUROC `0.763169`, macro AP `0.394590`
- `d2_val` macro AUROC `0.787738`, macro AP `0.366313`
- `target_test` macro AUROC `0.831937`, macro AP `0.507470`
- `target_val` macro AUROC `0.724427`, macro AP `0.349341`

## Notes

- Training uses only `target_train` embeddings.
- Auxiliary training aliases: `[]`.
- Initialization checkpoint: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/experiments/by_id/exp0001__domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear/best.ckpt`.
- LwF teacher checkpoint: `None`.
- MAS state path: `/tmp/cxr_sequential_regularizer_sweeps/mas_states/stage_a_nih_source_mas_state.pt`.
- Early stopping is driven by `target_val` macro AUROC.
- `target_val`-tuned thresholds are reused unchanged for later evaluation splits.
