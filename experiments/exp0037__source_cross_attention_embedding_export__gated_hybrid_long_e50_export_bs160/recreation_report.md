# Source Cross-Attention Embedding Export Recreation Report

## Experiment

- Export directory: `/workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160`
- Training experiment: `/workspace/experiments/exp0036__source_cross_attention_training__gated_hybrid_long_e50_pat5_bs32_ml256`
- Embedding dim: `512`

## Exact Command

```bash
python \
  /workspace/scripts/11_export_cross_attention_embeddings.py \
  --training-experiment-dir \
  /workspace/experiments/exp0036__source_cross_attention_training__gated_hybrid_long_e50_pat5_bs32_ml256 \
  --experiment-name \
  gated_hybrid_long_e50_export_bs160 \
  --overwrite \
  --trust-manifest-paths \
  --device \
  auto \
  --fp16-on-cuda \
  --batch-size \
  160 \
  --num-workers \
  8
```
