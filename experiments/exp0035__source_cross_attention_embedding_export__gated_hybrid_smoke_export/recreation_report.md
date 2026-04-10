# Source Cross-Attention Embedding Export Recreation Report

## Experiment

- Export directory: `/workspace/experiments/exp0035__source_cross_attention_embedding_export__gated_hybrid_smoke_export`
- Training experiment: `/workspace/experiments/exp0034__source_cross_attention_training__gated_hybrid_smoke_train`
- Embedding dim: `512`

## Exact Command

```bash
python \
  /workspace/scripts/11_export_cross_attention_embeddings.py \
  --training-experiment-dir \
  /workspace/experiments/exp0034__source_cross_attention_training__gated_hybrid_smoke_train \
  --experiment-name \
  gated_hybrid_smoke_export \
  --overwrite \
  --trust-manifest-paths \
  --device \
  auto \
  --batch-size \
  4 \
  --num-workers \
  0
```
