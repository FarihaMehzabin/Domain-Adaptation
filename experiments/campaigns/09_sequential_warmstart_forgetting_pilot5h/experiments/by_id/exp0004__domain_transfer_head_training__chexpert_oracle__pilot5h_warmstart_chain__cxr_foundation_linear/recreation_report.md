# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/experiments/by_id/exp0004__domain_transfer_head_training__chexpert_oracle__pilot5h_warmstart_chain__cxr_foundation_linear`
- Embedding root: `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation`
- Manifest: `/tmp/cxr_sequential_forgetting_study/manifest/manifest_pilot5h_binary_mimic.csv`
- Embedding layout: `domain_split`
- Token pooling: `avg`
- Head type: `linear`
- Init checkpoint: `None`
- LwF enabled: `False`
- LwF teacher checkpoint: `None`
- LwF source alias: `None`
- LwF alpha: `None`
- LwF temperature: `None`
- MLP hidden dims: `[]`
- MLP dropout: `0.2`

## Recreation Command

```bash
python \
  /workspace/scripts/15_train_domain_transfer_linear_probe.py \
  --experiment-name \
  exp0004__domain_transfer_head_training__chexpert_oracle__pilot5h_warmstart_chain__cxr_foundation_linear \
  --split-profile \
  chexpert_target \
  --embedding-root \
  /tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation \
  --manifest-csv \
  /tmp/cxr_sequential_forgetting_study/manifest/manifest_pilot5h_binary_mimic.csv \
  --experiments-root \
  /workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/experiments/by_id \
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

- `target_train` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d1_chexpert/train` with `1000` rows and shape `[1000, 768]`
- `target_val` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d1_chexpert/val` with `1000` rows and shape `[1000, 768]`
- `target_test` -> `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation/d1_chexpert/test` with `234` rows and shape `[234, 768]`

## Final Metrics

- `target_test` macro AUROC `0.770269`, macro AP `0.471421`
- `target_val` macro AUROC `0.732924`, macro AP `0.361405`

## Notes

- Training uses only `target_train` embeddings.
- Auxiliary training aliases: `[]`.
- Initialization checkpoint: `None`.
- LwF teacher checkpoint: `None`.
- Early stopping is driven by `target_val` macro AUROC.
- `target_val`-tuned thresholds are reused unchanged for later evaluation splits.
