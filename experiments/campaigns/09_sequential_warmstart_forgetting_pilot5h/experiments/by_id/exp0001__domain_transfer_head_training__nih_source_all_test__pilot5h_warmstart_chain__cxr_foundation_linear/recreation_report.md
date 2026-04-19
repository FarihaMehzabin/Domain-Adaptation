# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/experiments/by_id/exp0001__domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear`
- Embedding root: `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation`
- Manifest: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/manifest/manifest_pilot5h_binary_mimic.csv`
- Embedding layout: `domain_split`
- Token pooling: `avg`
- Head type: `linear`
- Init checkpoint: `None`
- Preservation method: `none`
- LwF enabled: `False`
- LwF teacher checkpoint: `None`
- LwF alpha: `None`
- LwF temperature: `None`
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
  exp0001__domain_transfer_head_training__nih_source_all_test__pilot5h_warmstart_chain__cxr_foundation_linear \
  --split-profile \
  nih_source_all_test \
  --embedding-root \
  /tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation \
  --manifest-csv \
  /tmp/cxr_sequential_forgetting_study/manifest/manifest_pilot5h_binary_mimic.csv \
  --experiments-root \
  /tmp/cxr_sequential_forgetting_study/experiments/by_id \
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

- `d0_train` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d0_nih/train` with `10000` rows and shape `[10000, 768]`
- `d0_val` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d0_nih/val` with `1000` rows and shape `[1000, 768]`
- `d0_test` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d0_nih/test` with `2000` rows and shape `[2000, 768]`
- `d1_val` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d1_chexpert/val` with `1000` rows and shape `[1000, 768]`
- `d1_test` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d1_chexpert/test` with `234` rows and shape `[234, 768]`
- `d2_val` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d2_mimic/val` with `1000` rows and shape `[1000, 768]`
- `d2_test` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d2_mimic/test` with `676` rows and shape `[676, 768]`

## Final Metrics

- `d0_test` macro AUROC `0.846135`, macro AP `0.254502`
- `d0_val` macro AUROC `0.849751`, macro AP `0.267059`
- `d1_test` macro AUROC `0.848588`, macro AP `0.547000`
- `d1_val` macro AUROC `0.709188`, macro AP `0.335451`
- `d2_test` macro AUROC `0.737037`, macro AP `0.375996`
- `d2_val` macro AUROC `0.779518`, macro AP `0.356049`

## Notes

- Training uses only `d0_train` embeddings.
- Auxiliary training aliases: `[]`.
- Initialization checkpoint: `None`.
- LwF teacher checkpoint: `None`.
- MAS state path: `None`.
- Early stopping is driven by `d0_val` macro AUROC.
- `d0_val`-tuned thresholds are reused unchanged for later evaluation splits.
