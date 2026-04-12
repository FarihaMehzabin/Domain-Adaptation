# Embedding Generation Recreation Report

## Scope

This report documents how to recreate the finalized NIH CXR14 image embedding experiment stored at:

`/workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test`

The producing script is:

`/workspace/scripts/generate_nih_split_image_embeddings.py`

Script SHA-256:

`82db27a6250f937939abc29f55400c0823590045d480dfd6db79649fb63cf403`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test`
- Experiment id: `exp0001`
- Operation label: `embedding_generation`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Data root: `/workspace/data`
- Splits: `train val test`
- Backend: `torchvision`
- Encoder: `resnet50`
- Weights: `DEFAULT`
- Pooling: `avg`
- Normalization: `l2`
- Resolved input size: `224 x 224`
- Batch size: `64`
- Num workers: `4`
- Device requested: `auto`
- Device resolved during run: `cuda`
- Mixed precision on CUDA: `true`
- Trust manifest paths: `true`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- PyTorch: `2.4.1+cu124`
- torchvision: `0.19.1+cu124`
- Pillow: `10.2.0`
- CUDA available: `true`
- GPU used: `NVIDIA RTX A5000`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this exact command:

```bash
python /workspace/scripts/generate_nih_split_image_embeddings.py \
  --manifest-csv /workspace/manifest_nih_cxr14_all14.csv \
  --data-root /workspace/data \
  --splits train val test \
  --encoder-backend torchvision \
  --encoder-id resnet50 \
  --weights DEFAULT \
  --pooling avg \
  --device auto \
  --fp16-on-cuda \
  --trust-manifest-paths \
  --experiment-name exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test \
  --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/generate_nih_split_image_embeddings.py \
  --manifest-csv /workspace/manifest_nih_cxr14_all14.csv \
  --data-root /workspace/data \
  --splits train val test \
  --encoder-backend torchvision \
  --encoder-id resnet50 \
  --weights DEFAULT \
  --pooling avg \
  --device auto \
  --fp16-on-cuda \
  --trust-manifest-paths \
  --experiment-name embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test
```

## Preconditions

- The NIH data must be extracted under `/workspace/data/nih_cxr14/raw`.
- The extracted image packs must exist as:
  - `/workspace/data/nih_cxr14/raw/images_001/images`
  - `/workspace/data/nih_cxr14/raw/images_002/images`
  - `/workspace/data/nih_cxr14/raw/images_003/images`
  - `/workspace/data/nih_cxr14/raw/images_004/images`
  - `/workspace/data/nih_cxr14/raw/images_005/images`
  - `/workspace/data/nih_cxr14/raw/images_006/images`
  - `/workspace/data/nih_cxr14/raw/images_007/images`
  - `/workspace/data/nih_cxr14/raw/images_008/images`
  - `/workspace/data/nih_cxr14/raw/images_009/images`
  - `/workspace/data/nih_cxr14/raw/images_010/images`
  - `/workspace/data/nih_cxr14/raw/images_011/images`
  - `/workspace/data/nih_cxr14/raw/images_012/images`
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`, `torchvision`, `PIL`.

## Data Integrity Note

During the original run, one train image was unreadable because it was a zero-byte file:

`/workspace/data/nih_cxr14/raw/images_005/images/00009676_000.png`

That file was repaired by downloading the correct PNG from Kaggle dataset `ksw2000/chestxray14`. The repaired file now exists locally and has:

- Path: `/workspace/data/nih_cxr14/raw/images_005/images/00009676_000.png`
- SHA-256: `e5aa6a1de40c219aaac2ae93b44dbb0880aed4d8cb0438638a88c492e05fd45e`

If this file becomes corrupt again, repair it with:

```bash
mkdir -p /workspace/tmp_kaggle_repair
KAGGLE_CONFIG_DIR=/workspace kaggle datasets download \
  -d ksw2000/chestxray14 \
  -f images/images_005/images/00009676_000.png \
  -p /workspace/tmp_kaggle_repair
cp /workspace/tmp_kaggle_repair/00009676_000.png \
  /workspace/data/nih_cxr14/raw/images_005/images/00009676_000.png
```

The current workspace already contains the repaired file, so a fresh rerun should finish in one pass with zero failed images.

## Expected Outputs

The experiment directory should contain:

- `experiment_meta.json`
- `train/embeddings.npy`
- `train/image_manifest.csv`
- `train/image_paths.txt`
- `train/run_meta.json`
- `val/embeddings.npy`
- `val/image_manifest.csv`
- `val/image_paths.txt`
- `val/run_meta.json`
- `test/embeddings.npy`
- `test/image_manifest.csv`
- `test/image_paths.txt`
- `test/run_meta.json`

No `failed_images.jsonl` should be present in the final successful run.

## Expected Counts

- Train input images: `78,571`
- Train embedded images: `78,571`
- Train failed images: `0`
- Val input images: `11,219`
- Val embedded images: `11,219`
- Val failed images: `0`
- Test input images: `22,330`
- Test embedded images: `22,330`
- Test failed images: `0`
- Embedding dimension for all splits: `2048`

## Output Sizes

- Train directory size: `627M`
- Val directory size: `92M`
- Test directory size: `180M`
- Total split output size: `897M`

## Final Artifact SHA-256

These hashes describe the finalized output currently on disk.

```text
b64f4040e04ec796db831f18113d40a877caf0835a9ca416882b590bc6b60d0e  /workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test/experiment_meta.json
9c1665b12c2a3e832ea8e8bf48beb575acfa60707a49987629659291c1744ddb  /workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test/train/embeddings.npy
c58f86ebe4fb2a013d1b8fd3c0bb23bbfdca989f4a09c30791add992106e1edd  /workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test/train/image_manifest.csv
7c90f281686c7fc23e46b5e7d0c72a4c2dfded5d0c69949ed68e398be8e036a8  /workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test/train/image_paths.txt
b5db4dc664b41d9f9964072fa17ee0e86ebb98aa933f0f4b5c54479e530d47ba  /workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test/train/run_meta.json
9e8f5467ea28ca4e1d01c4f25dccd06daf13a872580867fc9fed3b37ba393a3b  /workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test/val/embeddings.npy
d419f6d1f662130d3ec8154c849de53c915e8b85399996cbca62793a6562c10e  /workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test/val/image_manifest.csv
fb2f49e9bf89cc3c11082ab551b59eee773ba56005eaab981437112e934b21a5  /workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test/val/image_paths.txt
ebb1dd9396d3a9398512d25a8f19c629e5de6c86b735cedadda0bb32f8b5a45d  /workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test/val/run_meta.json
995815d3c5357c190049c11f7fbcb247113a173949d16c80420ae2a4c091a17e  /workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test/test/embeddings.npy
435b781e3d3749574f37eca96d38175c7a1e52728d786101d3310fb4a5c5835e  /workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test/test/image_manifest.csv
ffb371ed06d4db2fdc1615523d33454510515d439e237f7dba88ceeed7a06f6a  /workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test/test/image_paths.txt
245142e916a328702f2430bfa36b8b69b380aed44eaa6d70b6e1d8f8ab1d63fb  /workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test/test/run_meta.json
```

## Important Reproduction Notes

- `experiment_meta.json` and `run_meta.json` include timestamps, so those metadata hashes will change if you rerun the experiment.
- `embeddings.npy`, `image_manifest.csv`, and `image_paths.txt` are the main files to compare for recreation quality.
- Because this run used CUDA AMP (`--fp16-on-cuda`), exact bitwise reproduction of `embeddings.npy` is most likely when using the same GPU and software stack listed above.
- If you need a stable directory name, pass an explicit numbered `--experiment-name` as shown above.
- If you do not need the exact directory number, omit the explicit `exp0001__` prefix and let the script assign the next experiment id.

## Script Behaviors That Matter

These script behaviors are relevant to this experiment and were present in the script version identified above:

- Operation-prefixed naming is enforced so experiment names start with `embedding_generation__`.
- `--trust-manifest-paths` skips expensive per-row path existence checks and directly resolves `data_root / image_path`.
- Manifest rows for all requested splits are collected in a single pass instead of re-reading the CSV once per split.

Relevant script locations:

- Defaults and operation label: `/workspace/scripts/generate_nih_split_image_embeddings.py:36`
- Operation-prefix helper: `/workspace/scripts/generate_nih_split_image_embeddings.py:163`
- Trust-manifest fast path: `/workspace/scripts/generate_nih_split_image_embeddings.py:192`
- Single-pass manifest loading: `/workspace/scripts/generate_nih_split_image_embeddings.py:211`
- Experiment-name normalization: `/workspace/scripts/generate_nih_split_image_embeddings.py:961`
- `--trust-manifest-paths` CLI flag: `/workspace/scripts/generate_nih_split_image_embeddings.py:1032`

## Agent Handoff Text

If you want to hand this off to another agent, this is enough:

```text
Use /workspace/scripts/generate_nih_split_image_embeddings.py and the report /workspace/experiments/exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test/recreation_report.md to recreate the finalized NIH CXR14 image embedding experiment. Follow the exact recreation command in the report, verify the repaired image file hash, and confirm the final split counts and output artifact hashes.
```
