# CXR Foundation Embedding Export Recreation Report

## Scope

- Experiment directory: `/workspace/paper_v1/outputs/refresh_chexpert/20260420T081459Z__refresh_chexpert_full/embeddings/by_id/exp0001__cxr_foundation_embedding_export__paper_v1_chexpert_refresh`
- Manifest: `/workspace/paper_v1/outputs/refresh_chexpert/20260420T081459Z__refresh_chexpert_full/manifest/manifest_chexpert_refreshed.csv`
- Model cache dir: `/workspace/.cache/cxr_foundation`
- Embedding kind: `general`
- Token pooling: `avg`

## Recreation Command

```bash
python \
  /workspace/scripts/14_generate_cxr_foundation_embeddings.py \
  --manifest-csv \
  /workspace/paper_v1/outputs/refresh_chexpert/20260420T081459Z__refresh_chexpert_full/manifest/manifest_chexpert_refreshed.csv \
  --data-root \
  /workspace/paper_v1/outputs/refresh_chexpert/20260420T081459Z__refresh_chexpert_full/data \
  --experiments-root \
  /workspace/paper_v1/outputs/refresh_chexpert/20260420T081459Z__refresh_chexpert_full/embeddings/by_id \
  --experiment-name \
  paper_v1_chexpert_refresh \
  --batch-size \
  64 \
  --embedding-kind \
  general \
  --token-pooling \
  avg \
  --overwrite \
  --model-dir \
  /workspace/.cache/cxr_foundation \
  --hf-token-env-var \
  HF_TOKEN
```

## Split Outputs

- `d1_chexpert/test`: `234` rows, embedding tail shape `768`, failures `0`
- `d1_chexpert/train`: `1000` rows, embedding tail shape `768`, failures `0`
- `d1_chexpert/val`: `1000` rows, embedding tail shape `768`, failures `0`

## Notes

- This export follows the official Google CXR Foundation local Hugging Face path.
- Image preprocessing uses grayscale PNG-backed `tf.train.Example` inputs before ELIXR-C inference.
- `embedding_index.csv` maps each row to its sharded `embeddings.npy` file and row offset.
- If Hugging Face terms have not been accepted for `google/cxr-foundation`, set `HF_TOKEN` after acceptance and rerun.
