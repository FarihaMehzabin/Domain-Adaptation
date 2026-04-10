# Source Cross-Attention Training Recreation Report

## Experiment

- Experiment directory: `/workspace/experiments/exp0010__source_cross_attention_training__xattn_full_pipeline_e1_bs32_ml256`
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
  --device \
  cuda \
  --fp16-on-cuda \
  --trust-remote-code \
  --trust-manifest-paths \
  --batch-size \
  32 \
  --num-workers \
  8 \
  --max-length \
  256 \
  --epochs \
  1 \
  --patience \
  0 \
  --experiment-name \
  xattn_full_pipeline_e1_bs32_ml256
```

## Final Metrics

- Validation macro AUROC: `0.718285`
- Validation macro average precision: `0.110125`
- Test macro AUROC: `0.722673`
- Test macro average precision: `0.108092`
