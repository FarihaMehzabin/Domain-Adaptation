# CXR Foundation Embedding Export Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/by_id/exp0050__cxr_foundation_embedding_export__chexpert_target_250_cxr_foundation_avg_batch128`
- Manifest: `/workspace/manifest_chexpert_target_250.csv`
- Model cache dir: `/workspace/.cache/cxr_foundation`
- Embedding kind: `general`
- Token pooling: `avg`

## Recreation Command

```bash
python \
  /workspace/scripts/14_generate_cxr_foundation_embeddings.py \
  --manifest-csv \
  /workspace/manifest_chexpert_target_250.csv \
  --batch-size \
  128 \
  --token-pooling \
  avg \
  --experiment-name \
  exp0050__cxr_foundation_embedding_export__chexpert_target_250_cxr_foundation_avg_batch128 \
  --overwrite
```

## Split Outputs

- `d1_chexpert/test`: `234` rows, embedding tail shape `768`, failures `0`
- `d1_chexpert/train`: `250` rows, embedding tail shape `768`, failures `0`
- `d1_chexpert/val`: `250` rows, embedding tail shape `768`, failures `0`

## Notes

- This export follows the official Google CXR Foundation local Hugging Face path.
- Image preprocessing uses grayscale PNG-backed `tf.train.Example` inputs before ELIXR-C inference.
- `embedding_index.csv` maps each row to its sharded `embeddings.npy` file and row offset.
- If Hugging Face terms have not been accepted for `google/cxr-foundation`, set `HF_TOKEN` after acceptance and rerun.
