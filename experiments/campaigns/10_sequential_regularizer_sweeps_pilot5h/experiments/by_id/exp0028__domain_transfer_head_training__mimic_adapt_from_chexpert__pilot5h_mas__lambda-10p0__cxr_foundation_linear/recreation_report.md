# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/tmp/cxr_sequential_regularizer_sweeps/experiments/by_id/exp0028__domain_transfer_head_training__mimic_adapt_from_chexpert__pilot5h_mas__lambda-10p0__cxr_foundation_linear`
- Embedding root: `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation`
- Manifest: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/manifest/manifest_pilot5h_binary_mimic.csv`
- Embedding layout: `domain_split`
- Token pooling: `avg`
- Head type: `linear`
- Init checkpoint: `/tmp/cxr_sequential_regularizer_sweeps/experiments/by_id/exp0027__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_mas__lambda-10p0__cxr_foundation_linear/best.ckpt`
- Preservation method: `mas`
- LwF enabled: `False`
- LwF teacher checkpoint: `None`
- LwF alpha: `None`
- LwF temperature: `None`
- MAS enabled: `True`
- MAS state path: `/tmp/cxr_sequential_regularizer_sweeps/mas_states/mas_lambda-10p0/stage_b_chexpert_mas_state.pt`
- MAS lambda: `10.0`
- MLP hidden dims: `[]`
- MLP dropout: `0.2`

## Recreation Command

```bash
python \
  /workspace/scripts/15_train_domain_transfer_linear_probe.py \
  --experiment-name \
  exp0028__domain_transfer_head_training__mimic_adapt_from_chexpert__pilot5h_mas__lambda-10p0__cxr_foundation_linear \
  --split-profile \
  mimic_adapt_from_chexpert \
  --init-checkpoint \
  /tmp/cxr_sequential_regularizer_sweeps/experiments/by_id/exp0027__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_mas__lambda-10p0__cxr_foundation_linear/best.ckpt \
  --enable-mas \
  --mas-state-path \
  /tmp/cxr_sequential_regularizer_sweeps/mas_states/mas_lambda-10p0/stage_b_chexpert_mas_state.pt \
  --mas-lambda \
  10.0 \
  --embedding-root \
  /tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation \
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

- `target_train` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d2_mimic/train` with `998` rows and shape `[998, 768]`
- `target_val` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d2_mimic/val` with `1000` rows and shape `[1000, 768]`
- `target_test` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d2_mimic/test` with `676` rows and shape `[676, 768]`
- `d0_val` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d0_nih/val` with `1000` rows and shape `[1000, 768]`
- `d0_test` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d0_nih/test` with `2000` rows and shape `[2000, 768]`
- `d1_val` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d1_chexpert/val` with `1000` rows and shape `[1000, 768]`
- `d1_test` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d1_chexpert/test` with `234` rows and shape `[234, 768]`

## Final Metrics

- `d0_test` macro AUROC `0.846755`, macro AP `0.255952`
- `d0_val` macro AUROC `0.849206`, macro AP `0.266443`
- `d1_test` macro AUROC `0.844873`, macro AP `0.534900`
- `d1_val` macro AUROC `0.714490`, macro AP `0.339372`
- `target_test` macro AUROC `0.750239`, macro AP `0.385482`
- `target_val` macro AUROC `0.785119`, macro AP `0.363811`

## Notes

- Training uses only `target_train` embeddings.
- Auxiliary training aliases: `[]`.
- Initialization checkpoint: `/tmp/cxr_sequential_regularizer_sweeps/experiments/by_id/exp0027__domain_transfer_head_training__chexpert_adapt_from_nih__pilot5h_mas__lambda-10p0__cxr_foundation_linear/best.ckpt`.
- LwF teacher checkpoint: `None`.
- MAS state path: `/tmp/cxr_sequential_regularizer_sweeps/mas_states/mas_lambda-10p0/stage_b_chexpert_mas_state.pt`.
- Early stopping is driven by `target_val` macro AUROC.
- `target_val`-tuned thresholds are reused unchanged for later evaluation splits.
