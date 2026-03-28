# Fused Embedding Generation Recreation Report

## Scope

This report documents how to recreate the finalized NIH CXR14 fused embedding experiment stored at:

`/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`

The producing script is:

`/workspace/scripts/generate_split_fused_embeddings.py`

Script SHA-256:

`9df8d2f257965aed54447bd98b793fc8a7802c558ab0b994315c75a0320541d4`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`
- Experiment id: `exp0003`
- Operation label: `fused_embedding_generation`
- Source order: `image report`
- Source image experiment root: `/workspace/experiments/exp0001__image_embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test`
- Source report experiment root: `/workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test`
- Splits: `train val test`
- Fusion mode: `concat`
- Output normalization: `l2`
- Alignment mode: `reference`
- Reference source: `image`
- Reference row-id sidecar: `image_paths.txt`
- Image source embedding dimension: `2048`
- Report source embedding dimension: `128`
- Fused embedding dimension: `2176`
- Row chunk size: `4096`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- PyTorch: `2.4.1+cu124`
- CUDA available: `true`
- GPU used during validation: `NVIDIA RTX A5000`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/generate_split_fused_embeddings.py \
  --source name=image,root=/workspace/experiments/exp0001__image_embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test \
  --source name=report,root=/workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test \
  --splits train val test \
  --fusion concat \
  --normalize-output l2 \
  --alignment reference \
  --row-chunk-size 4096 \
  --experiment-name exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 \
  --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/generate_split_fused_embeddings.py \
  --source name=image,root=/workspace/experiments/exp0001__image_embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test \
  --source name=report,root=/workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test \
  --splits train val test \
  --fusion concat \
  --normalize-output l2 \
  --alignment reference \
  --row-chunk-size 4096 \
  --experiment-name fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2
```

## Preconditions

- The image source experiment must already exist at `/workspace/experiments/exp0001__image_embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test`.
- The report source experiment must already exist at `/workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test`.
- Each source split must contain `embeddings.npy` plus a row-identity sidecar. In this run the image source uses `image_paths.txt` and the report source uses `report_ids.json`.
- Source identifiers must align by image basename, so `Path(image_path).stem == report_id` for every fused row.
- The required Python package must be importable: `numpy`.
- If any source `embeddings.npy` is a Git LFS pointer instead of the real array, fetch the real file before fusion.
- You need roughly `957M` of space for the fused split outputs.

## Expected Outputs

The experiment directory should contain:

- `experiment_meta.json`
- `recreation_report.md`
- `.gitignore`
- `train/embeddings.npy`
- `train/alignment_manifest.csv`
- `train/image_paths.txt`
- `train/row_ids.json`
- `train/run_meta.json`
- `val/embeddings.npy`
- `val/alignment_manifest.csv`
- `val/image_paths.txt`
- `val/row_ids.json`
- `val/run_meta.json`
- `test/embeddings.npy`
- `test/alignment_manifest.csv`
- `test/image_paths.txt`
- `test/row_ids.json`
- `test/run_meta.json`

## Expected Counts

- Train fused rows: `78,571`
- Val fused rows: `11,219`
- Test fused rows: `22,330`
- Fused embedding dimension for all splits: `2176`
- Embedding dtype for all splits: `float32`
- Output row order matches the image source split order
- Output row IDs are the image basename and should exactly match the report IDs for aligned rows
- Saved embeddings are L2-normalized, so row norms should be approximately `1.0`

## Output Sizes

- Train directory size: `668M`
- Val directory size: `97M`
- Test directory size: `192M`
- Total split output size: `957M`

## Final Artifact SHA-256

These hashes describe the finalized output currently on disk.

```text
7d1afa8c3d5e0796b035a5c0b5db6344500b554b37d9b5f9ebfe3110290c6d9f  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/experiment_meta.json
b469a59d27a4f96d0eec1e0151017036bbb8d08940e387b91500bab6df215475  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/train/embeddings.npy
79dec8e9fca83f0cc9fac7b33e063ed2f348ae3aa7e3e9970f5ad63f8ecf2199  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/train/alignment_manifest.csv
7c90f281686c7fc23e46b5e7d0c72a4c2dfded5d0c69949ed68e398be8e036a8  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/train/image_paths.txt
d0236138712b7663f1a92bdd8d7fe9e976b53bdc53e0a49263364ee118079daa  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/train/row_ids.json
8480227e5021bdac14389badf245e297a60d6b4729f99d6a023580ebad300adb  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/train/run_meta.json
90057859be7c47cff1fa068b65e3083498b7df338af75253bec2abb5ac9e9348  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val/embeddings.npy
c63fba7ba101785e528975e1b83e6397433160b751ff1a01aee5a3353a49df27  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val/alignment_manifest.csv
fb2f49e9bf89cc3c11082ab551b59eee773ba56005eaab981437112e934b21a5  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val/image_paths.txt
a0891a0be5485b674caca3c6a14b6ce3b291a2ca6815df9aeacc35c28c66b1e4  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val/row_ids.json
96ccec26e2d50a3418e3a3d891a39d87a6efb3be4d1624bea9e9071351b0490e  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val/run_meta.json
09f63400ae29f6ad895249354d7950afb0b9ab5677f32d29f3af29d79c4670ba  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/test/embeddings.npy
f8322c452a19b2021e0f43fff577a477a13708fd1516ce18b5c7adae9a8c2f88  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/test/alignment_manifest.csv
ffb371ed06d4db2fdc1615523d33454510515d439e237f7dba88ceeed7a06f6a  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/test/image_paths.txt
095aeb101d7bbc873ea87c8ae2e6d79263da9f619df11fea907c5b948ffb54d7  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/test/row_ids.json
63a2a09880546de7c5946976b0c8854799d78f087d95a21eff26d8a444edcfc5  /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/test/run_meta.json
```

## Important Reproduction Notes

- `experiment_meta.json` and each split `run_meta.json` include timestamps, so those metadata hashes will change if you rerun the experiment.
- The output `image_paths.txt` files are copied from the reference image source after alignment. Their hashes therefore match the corresponding files from `exp0001`.
- The output `row_ids.json` files are the aligned sample IDs. For this image-plus-report run, they match the report IDs used by `exp0002`.
- Both source experiments already store L2-normalized embeddings, so concatenation produces a pre-normalization row norm close to `sqrt(2)`. The fusion script then normalizes each fused row back to unit length.
- The saved `alignment_manifest.csv` is the main debugging artifact for future modality mixes because it records, per fused row, the chosen sample ID and the exact source-sidecar item used from each input experiment.
- The `.npy` arrays are intentionally excluded from the plain Git commit for this experiment. The metadata, manifests, and ID files are small enough for normal Git without LFS.

## Script Behaviors That Matter

These script behaviors are relevant to this experiment and were present in the script version identified above:

- The CLI accepts repeated `--source` specs, one per modality or embedding source.
- Source row IDs are auto-detected from common sidecars like `image_paths.txt`, `report_ids.json`, and manifest CSVs.
- Alignment can be reference-based or intersection-based. This run used reference alignment with the image source as the row-order authority.
- Fusion is column-wise concatenation in source order.
- Source-specific scalar weights are supported before concatenation, though this run used `1.0` for both sources.
- Output embeddings are L2-normalized after fusion.
- The script writes `row_ids.json`, `alignment_manifest.csv`, and a copy of the reference sidecar for each split.

Relevant script locations:

- CLI and source spec parsing: `/workspace/scripts/generate_split_fused_embeddings.py:240`
- Sidecar auto-detection: `/workspace/scripts/generate_split_fused_embeddings.py:387`
- Source loading and row-ID parsing: `/workspace/scripts/generate_split_fused_embeddings.py:482`
- Reference/intersection row alignment: `/workspace/scripts/generate_split_fused_embeddings.py:541`
- Concatenation assembly: `/workspace/scripts/generate_split_fused_embeddings.py:612`
- L2 output normalization: `/workspace/scripts/generate_split_fused_embeddings.py:644`
- Alignment manifest and reference sidecar writers: `/workspace/scripts/generate_split_fused_embeddings.py:701`
- Main experiment writer flow: `/workspace/scripts/generate_split_fused_embeddings.py:778`

## Agent Handoff Text

If you want to hand this off to another agent, this is enough:

```text
Use /workspace/scripts/generate_split_fused_embeddings.py and the report /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/recreation_report.md to recreate the fused NIH CXR14 experiment that combines exp0001 image embeddings with exp0002 report embeddings. Run the exact command in the report, verify the split counts and artifact hashes, and confirm that the fused outputs are 2176-D float32 vectors with unit-length row norms.
```
