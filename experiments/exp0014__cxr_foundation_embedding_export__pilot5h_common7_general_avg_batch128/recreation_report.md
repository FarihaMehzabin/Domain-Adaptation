# CXR Foundation Embedding Export Recreation Report

## Scope

- Experiment directory: `/workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128`
- Manifest: `/workspace/manifest_common_labels_pilot5h.csv`
- Model cache dir: `/workspace/.cache/cxr_foundation`
- Embedding kind: `general`
- Token pooling: `avg`

## Recreation Command

```bash
python \
  /workspace/scripts/14_generate_cxr_foundation_embeddings.py \
  --manifest-csv \
  /workspace/manifest_common_labels_pilot5h.csv \
  --batch-size \
  128 \
  --experiment-name \
  exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128 \
  --overwrite
```

## Split Outputs

- `d0_nih/test`: `2000` rows, embedding tail shape `768`, failures `0`
- `d0_nih/train`: `10000` rows, embedding tail shape `768`, failures `0`
- `d0_nih/val`: `1000` rows, embedding tail shape `768`, failures `0`
- `d1_chexpert/val`: `234` rows, embedding tail shape `768`, failures `0`
- `d2_mimic/test`: `1455` rows, embedding tail shape `768`, failures `0`

## Notes

- This export follows the official Google CXR Foundation local Hugging Face path.
- Image preprocessing uses grayscale PNG-backed `tf.train.Example` inputs before ELIXR-C inference.
- `embedding_index.csv` maps each row to its sharded `embeddings.npy` file and row offset.
- If Hugging Face terms have not been accepted for `google/cxr-foundation`, set `HF_TOKEN` after acceptance and rerun.
