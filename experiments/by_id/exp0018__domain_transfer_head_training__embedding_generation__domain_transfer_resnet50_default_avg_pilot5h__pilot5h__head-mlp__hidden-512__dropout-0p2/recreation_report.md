# Domain Transfer Head Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/exp0018__domain_transfer_head_training__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h__pilot5h__head-mlp__hidden-512__dropout-0p2`
- Embedding root: `/workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h`
- Manifest: `/workspace/manifest_common_labels_pilot5h.csv`
- Embedding layout: `domain_split`
- Token pooling: `avg`
- Head type: `mlp`
- MLP hidden dims: `[512]`
- MLP dropout: `0.2`

## Recreation Command

```bash
python \
  /workspace/scripts/15_train_domain_transfer_linear_probe.py \
  --embedding-root \
  /workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h \
  --manifest-csv \
  /workspace/manifest_common_labels_pilot5h.csv \
  --head-type \
  mlp \
  --mlp-hidden-dims \
  512 \
  --mlp-dropout \
  0.2 \
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
  --token-pooling \
  avg \
  --experiment-name \
  embedding_generation__domain_transfer_resnet50_default_avg_pilot5h__pilot5h__head-mlp__hidden-512__dropout-0p2
```

## Split Inputs

- `d0_train` -> `/workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h/d0_nih/train` with `10000` rows and shape `[10000, 2048]`
- `d0_val` -> `/workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h/d0_nih/val` with `1000` rows and shape `[1000, 2048]`
- `d0_test` -> `/workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h/d0_nih/test` with `2000` rows and shape `[2000, 2048]`
- `d1_transfer` -> `/workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h/d1_chexpert/val` with `234` rows and shape `[234, 2048]`
- `d2_transfer` -> `/workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h/d2_mimic/test` with `1455` rows and shape `[1455, 2048]`

## Final Metrics

- `d0_test` macro AUROC `0.755967`, macro AP `0.146861`
- `d0_val` macro AUROC `0.757060`, macro AP `0.158643`
- `d1_transfer` macro AUROC `0.703681`, macro AP `0.335463`
- `d2_transfer` macro AUROC `0.502472`, macro AP `0.131178`

## Notes

- Training uses only `d0_train` embeddings.
- Early stopping is driven by `d0_val` macro AUROC.
- Validation-tuned thresholds are reused unchanged for D0 test and transfer evaluations.
