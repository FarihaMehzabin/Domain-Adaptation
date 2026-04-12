# Image-Only LoRA Transfer Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/exp0027__domain_transfer_lora_training__vit_base_patch16_224_in21k_pilot5h`
- Manifest: `/workspace/manifest_common_labels_pilot5h.csv`
- Backbone model: `google/vit-base-patch16-224-in21k`
- Pooling: `cls`
- LoRA target modules: `['query', 'value']`

## Recreation Command

```bash
python \
  /workspace/scripts/21_train_image_only_lora_transfer.py \
  --manifest-csv \
  /workspace/manifest_common_labels_pilot5h.csv \
  --batch-size \
  256 \
  --fp16-on-cuda \
  --epochs \
  12 \
  --patience \
  3 \
  --overwrite \
  --experiment-name \
  exp0027__domain_transfer_lora_training__vit_base_patch16_224_in21k_pilot5h
```

## Split Inputs

- `d0_train` -> domain=`d0_nih` split=`train` rows=`10000`
- `d0_val` -> domain=`d0_nih` split=`val` rows=`1000`
- `d0_test` -> domain=`d0_nih` split=`test` rows=`2000`
- `d1_transfer` -> domain=`d1_chexpert` split=`val` rows=`234`
- `d2_transfer` -> domain=`d2_mimic` split=`test` rows=`1455`

## Final Metrics

- `d0_test` macro AUROC `0.781900`, macro AP `0.164975`
- `d0_val` macro AUROC `0.793842`, macro AP `0.187499`
- `d1_transfer` macro AUROC `0.729908`, macro AP `0.382686`
- `d2_transfer` macro AUROC `0.499711`, macro AP `0.129290`

## Notes

- Training uses only `d0_train` images.
- Early stopping is driven by `d0_val` macro AUROC.
- Validation-tuned thresholds are reused unchanged for D0 test and transfer evaluations.
