# Source Cross-Attention Embedding Export Recreation Report

## Experiment

- Export directory: `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160`
- Training experiment: `/workspace/experiments/exp0018__source_cross_attention_training__xattn_long_e50_pat5_bs32_ml256`
- Embedding dim: `512`

## Exact Command

```bash
python \
  /workspace/scripts/11_export_cross_attention_embeddings.py \
  --training-experiment-dir \
  /workspace/experiments/exp0018__source_cross_attention_training__xattn_long_e50_pat5_bs32_ml256 \
  --device \
  cuda \
  --fp16-on-cuda \
  --trust-manifest-paths \
  --batch-size \
  160 \
  --num-workers \
  8 \
  --experiment-name \
  xattn_long_e50_export_bs160
```
