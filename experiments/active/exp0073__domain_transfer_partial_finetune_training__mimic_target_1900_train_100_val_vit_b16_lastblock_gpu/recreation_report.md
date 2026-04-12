# Image-Only Partial Finetune Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/active/exp0073__domain_transfer_partial_finetune_training__mimic_target_1900_train_100_val_vit_b16_lastblock_gpu`
- Manifest: `/workspace/manifest/manifest_mimic_target_1900_train_100_val.csv`
- Backbone model: `torchvision/vit_b_16.imagenet1k_v1`
- Adaptation method: `partial_last_block_finetune`
- Split profile: `mimic_target`

## Recreation Command

```bash
python \
+  /workspace/scripts/29_train_image_only_partial_finetune.py \
+  --manifest-csv \
+  /workspace/manifest/manifest_mimic_target_1900_train_100_val.csv \
+  --split-profile \
+  mimic_target \
+  --device \
+  cuda \
+  --fp16-on-cuda \
+  --batch-size \
+  8 \
+  --epochs \
+  10 \
+  --patience \
+  3 \
+  --lr \
+  5e-5 \
+  --trainable-blocks \
+  1 \
+  --experiment-name \
+  mimic_target_1900_train_100_val_vit_b16_lastblock_gpu
```

## Split Inputs

- `target_train` -> domain=`d2_mimic` split=`train` rows=`1900`
- `target_val` -> domain=`d2_mimic` split=`val` rows=`100`
- `target_test` -> domain=`d2_mimic` split=`test` rows=`1455`

## Final Metrics

- `target_test` macro AUROC `0.484694`, macro AP `0.122003`
- `target_train` macro AUROC `0.939496`, macro AP `0.792328`
- `target_val` macro AUROC `0.510827`, macro AP `0.140284`
