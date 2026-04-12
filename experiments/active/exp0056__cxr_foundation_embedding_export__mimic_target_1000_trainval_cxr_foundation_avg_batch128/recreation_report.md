# CXR Foundation Embedding Export Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/active/exp0056__cxr_foundation_embedding_export__mimic_target_1000_trainval_cxr_foundation_avg_batch128`
- Manifest: `/workspace/manifest/scratch/manifest_mimic_target_1000_train_val.csv`
- Model cache dir: `/workspace/.cache/cxr_foundation`
- Embedding kind: `general`
- Token pooling: `avg`

## Recreation Command

```bash
python \
  /workspace/scripts/14_generate_cxr_foundation_embeddings.py \
  --manifest-csv \
  /workspace/scratch/manifest_mimic_target_1000_train_val.csv \
  --batch-size \
  128 \
  --token-pooling \
  avg \
  --experiment-name \
  mimic_target_1000_trainval_cxr_foundation_avg_batch128 \
  --overwrite
```

## Split Outputs

- `d2_mimic/train`: `1000` rows, embedding tail shape `768`, failures `0`
- `d2_mimic/val`: `1000` rows, embedding tail shape `768`, failures `0`

## Notes

- This export follows the official Google CXR Foundation local Hugging Face path.
- Image preprocessing uses grayscale PNG-backed `tf.train.Example` inputs before ELIXR-C inference.
- `embedding_index.csv` maps each row to its sharded `embeddings.npy` file and row offset.
- If Hugging Face terms have not been accepted for `google/cxr-foundation`, set `HF_TOKEN` after acceptance and rerun.
