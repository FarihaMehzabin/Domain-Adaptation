# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/tmp/cxr_sequential_regularizer_sweeps/experiments/by_id/exp0025__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_mas__lambda-3p0__cxr_foundation_linear`
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
- MAS lambda: `3.0`
- MLP hidden dims: `[]`
- MLP dropout: `0.2`

## Recreation Command

```bash
python \
  /workspace/scripts/15_train_domain_transfer_linear_probe.py \
  --experiment-name \
  exp0025__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_mas__lambda-3p0__cxr_foundation_linear \
  --split-profile \
  chexpert_adapt_from_nih \
  --init-checkpoint \
  /workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/experiments/by_id/exp0001__domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear/best.ckpt \
  --enable-mas \
  --mas-state-path \
  /tmp/cxr_sequential_regularizer_sweeps/mas_states/stage_a_nih_source_mas_state.pt \
  --mas-lambda \
  3.0 \
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

- `d0_test` macro AUROC `0.845440`, macro AP `0.255052`
- `d0_val` macro AUROC `0.846311`, macro AP `0.262055`
- `d2_test` macro AUROC `0.754573`, macro AP `0.387819`
- `d2_val` macro AUROC `0.785403`, macro AP `0.362733`
- `target_test` macro AUROC `0.840405`, macro AP `0.521953`
- `target_val` macro AUROC `0.718535`, macro AP `0.344187`

## Notes

- Training uses only `target_train` embeddings.
- Auxiliary training aliases: `[]`.
- Initialization checkpoint: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/experiments/by_id/exp0001__domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear/best.ckpt`.
- LwF teacher checkpoint: `None`.
- MAS state path: `/tmp/cxr_sequential_regularizer_sweeps/mas_states/stage_a_nih_source_mas_state.pt`.
- Early stopping is driven by `target_val` macro AUROC.
- `target_val`-tuned thresholds are reused unchanged for later evaluation splits.
