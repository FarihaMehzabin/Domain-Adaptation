# CXR Foundation Embedding Export Recreation Report

## Scope

- Experiment directory: `/workspace/paper_v1/outputs/mimic_train_repair/20260420T104800Z__mimic_train_repair/embeddings/by_id/exp0001__cxr_foundation_embedding_export__mimic_train_repair_train_only_cxr_foundation_avg_batch128`
- Manifest: `/workspace/paper_v1/outputs/mimic_train_repair/20260420T104800Z__mimic_train_repair/manifest_mimic_train_repair.csv`
- Model cache dir: `/workspace/.cache/cxr_foundation`
- Embedding kind: `general`
- Token pooling: `avg`

## Recreation Command

```bash
python \
  /workspace/scripts/14_generate_cxr_foundation_embeddings.py \
  --manifest-csv \
  /workspace/paper_v1/outputs/mimic_train_repair/20260420T104800Z__mimic_train_repair/manifest_mimic_train_repair.csv \
  --data-root \
  /workspace \
  --experiments-root \
  /workspace/paper_v1/outputs/mimic_train_repair/20260420T104800Z__mimic_train_repair/embeddings/by_id \
  --model-dir \
  /workspace/.cache/cxr_foundation \
  --experiment-name \
  exp0001__cxr_foundation_embedding_export__mimic_train_repair_train_only_cxr_foundation_avg_batch128 \
  --batch-size \
  128 \
  --embedding-kind \
  general \
  --token-pooling \
  avg \
  --overwrite
```

## Split Outputs

- `d2_mimic/train`: `998` rows, embedding tail shape `768`, failures `0`

## Notes

- This export follows the official Google CXR Foundation local Hugging Face path.
- Image preprocessing uses grayscale PNG-backed `tf.train.Example` inputs before ELIXR-C inference.
- `embedding_index.csv` maps each row to its sharded `embeddings.npy` file and row offset.
- If Hugging Face terms have not been accepted for `google/cxr-foundation`, set `HF_TOKEN` after acceptance and rerun.
