# Source Cross-Attention Training Recreation Report

## Experiment

- Experiment directory: `/workspace/experiments/exp0034__source_cross_attention_training__gated_hybrid_smoke_train`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Data root: `/workspace/data`
- Reports root: `/workspace/reports`
- Frozen image encoder: `resnet50`
- Frozen text encoder: `microsoft/BiomedVLP-CXR-BERT-specialized`
- Fusion dim: `256`
- Fusion layers: `2`
- Embedding dim: `512`

## Exact Command

```bash
python \
  /workspace/scripts/10_train_cross_attention_encoder.py \
  --experiment-name \
  gated_hybrid_smoke_train \
  --overwrite \
  --trust-remote-code \
  --trust-manifest-paths \
  --device \
  auto \
  --batch-size \
  2 \
  --num-workers \
  0 \
  --epochs \
  1 \
  --patience \
  0 \
  --max-length \
  64 \
  --max-samples-per-split \
  8
```

## Final Metrics

- Validation macro AUROC: `null`
- Validation macro average precision: `null`
- Test macro AUROC: `0.417143`
- Test macro average precision: `0.285556`
