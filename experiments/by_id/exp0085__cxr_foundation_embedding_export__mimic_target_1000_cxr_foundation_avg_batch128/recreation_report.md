# CXR Foundation Embedding Export Recreation Report

## Scope

- Experiment directory: `/tmp/cxr_mimic_run/experiments/by_id/exp0001__cxr_foundation_embedding_export__mimic_target_1000_cxr_foundation_avg_batch128`
- Manifest: `/tmp/cxr_mimic_run/manifest/manifest_mimic_target_1000_binary.csv`
- Model cache dir: `/tmp/cxr_mimic_run/cache/cxr_foundation_model`
- Embedding kind: `general`
- Token pooling: `avg`

## Recreation Command

```bash
python \
  /workspace/scripts/14_generate_cxr_foundation_embeddings.py \
  --manifest-csv \
  /tmp/cxr_mimic_run/manifest/manifest_mimic_target_1000_binary.csv \
  --data-root \
  /workspace \
  --experiments-root \
  /tmp/cxr_mimic_run/experiments/by_id \
  --model-dir \
  /tmp/cxr_mimic_run/cache/cxr_foundation_model \
  --experiment-name \
  exp0001__cxr_foundation_embedding_export__mimic_target_1000_cxr_foundation_avg_batch128 \
  --batch-size \
  128 \
  --embedding-kind \
  general \
  --token-pooling \
  avg \
  --overwrite
```

## Split Outputs

- `d2_mimic/test`: `676` rows, embedding tail shape `768`, failures `0`
- `d2_mimic/train`: `998` rows, embedding tail shape `768`, failures `2`
- `d2_mimic/val`: `1000` rows, embedding tail shape `768`, failures `0`

## Notes

- This export follows the official Google CXR Foundation local Hugging Face path.
- Image preprocessing uses grayscale PNG-backed `tf.train.Example` inputs before ELIXR-C inference.
- `embedding_index.csv` maps each row to its sharded `embeddings.npy` file and row offset.
- If Hugging Face terms have not been accepted for `google/cxr-foundation`, set `HF_TOKEN` after acceptance and rerun.
