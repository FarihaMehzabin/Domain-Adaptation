# Embedding Generation Recreation Report

## Scope

This report documents how to recreate the finalized NIH CXR14 report embedding experiment stored at:

`/workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test`

The producing script is:

`/workspace/scripts/generate_nih_split_report_embeddings.py`

Script SHA-256:

`d91c7128d004f9a6c38c6a764a93c5570da6ead554c015c051faf744f747080f`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test`
- Experiment id: `exp0002`
- Operation label: `report_embedding_generation`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Reports root: `/workspace/reports`
- Splits: `train val test`
- Backend: `huggingface`
- Model: `microsoft/BiomedVLP-CXR-BERT-specialized`
- Tokenizer: `microsoft/BiomedVLP-CXR-BERT-specialized`
- Revision: `null`
- Pooling requested: `auto`
- Pooling applied during run: `get_projected_text_embeddings`
- Normalization: `l2`
- Effective max length: `512`
- Batch size: `64`
- Num workers: `4`
- Device requested: `auto`
- Device resolved during run: `cuda`
- Mixed precision on CUDA: `true`
- Trust remote code: `true`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- PyTorch: `2.4.1+cu124`
- transformers: `5.4.0`
- safetensors: `0.7.0`
- huggingface_hub: `1.8.0`
- CUDA available: `true`
- GPU used: `NVIDIA RTX A5000`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this exact command:

```bash
python /workspace/scripts/generate_nih_split_report_embeddings.py \
  --manifest-csv /workspace/manifest_nih_cxr14_all14.csv \
  --reports-root /workspace/reports \
  --splits train val test \
  --model-id microsoft/BiomedVLP-CXR-BERT-specialized \
  --pooling auto \
  --device auto \
  --fp16-on-cuda \
  --trust-remote-code \
  --num-workers 4 \
  --batch-size 64 \
  --experiment-name exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test \
  --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/generate_nih_split_report_embeddings.py \
  --manifest-csv /workspace/manifest_nih_cxr14_all14.csv \
  --reports-root /workspace/reports \
  --splits train val test \
  --model-id microsoft/BiomedVLP-CXR-BERT-specialized \
  --pooling auto \
  --device auto \
  --fp16-on-cuda \
  --trust-remote-code \
  --num-workers 4 \
  --batch-size 64 \
  --experiment-name report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test
```

## Preconditions

- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The report text files must exist under:
  - `/workspace/reports/train`
  - `/workspace/reports/val`
  - `/workspace/reports/test`
- The report filenames must match the image basename in the manifest, for example `image_path .../00000002_000.png` must map to `reports/train/00000002_000.txt`.
- The required Python packages must be importable: `numpy`, `torch`, `transformers`, `safetensors`.
- The selected model requires `--trust-remote-code` because Hugging Face serves custom `CXRBertConfig`, `CXRBertTokenizer`, and `CXRBertModel` code for this repository.

## Expected Outputs

The experiment directory should contain:

- `experiment_meta.json`
- `recreation_report.md`
- `train/embeddings.npy`
- `train/report_manifest.csv`
- `train/report_ids.json`
- `train/report_paths.txt`
- `train/run_meta.json`
- `val/embeddings.npy`
- `val/report_manifest.csv`
- `val/report_ids.json`
- `val/report_paths.txt`
- `val/run_meta.json`
- `test/embeddings.npy`
- `test/report_manifest.csv`
- `test/report_ids.json`
- `test/report_paths.txt`
- `test/run_meta.json`

No `failed_reports.jsonl` should be present in the final successful run.

## Expected Counts

- Train input reports: `78,571`
- Train embedded reports: `78,571`
- Train failed reports: `0`
- Val input reports: `11,219`
- Val embedded reports: `11,219`
- Val failed reports: `0`
- Test input reports: `22,330`
- Test embedded reports: `22,330`
- Test failed reports: `0`
- Embedding dimension for all splits: `128`
- Embedding dtype for all splits: `float32`
- Embeddings are L2-normalized, so row norms should be approximately `1.0`

## Output Sizes

- Train directory size: `54M`
- Val directory size: `9.2M`
- Test directory size: `17M`
- Total experiment directory size: `81M`

## Final Artifact SHA-256

These hashes describe the finalized output currently on disk.

```text
fb2eb6d25db13b2cb8f9894146f3ee0e8808bddcab162657c9d55ecc5268e959  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/experiment_meta.json
8e021ec8b584fb13eeeb93cfeda5a0758d5a4008a57b38a37cf16fa083f80a01  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/train/embeddings.npy
32ec1a8f1afe1fe29e86a5d6e26ba8c49ddbe8aed6f9e56f3e594c67839c43ce  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/train/report_manifest.csv
d0236138712b7663f1a92bdd8d7fe9e976b53bdc53e0a49263364ee118079daa  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/train/report_ids.json
e63695ab91f5d84ca14297f85195d070e927780388cb592de8d0e521673d1f44  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/train/report_paths.txt
b0e433d384d8bc8a1743fae8acb5be3e859938f6201893cd1a89b2616551cea7  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/train/run_meta.json
938dda7c134e295d17e4fd36e79eedf43655d15ce9d6ae289141893727873148  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/val/embeddings.npy
321f31f4fdbe7fe3cb29f1168de7f43a25dd900012315aeafd4dcee378db3ce6  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/val/report_manifest.csv
a0891a0be5485b674caca3c6a14b6ce3b291a2ca6815df9aeacc35c28c66b1e4  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/val/report_ids.json
9a54b12729a7cd5b6c2f444c4722835657099eeb8cc3ea5a789fdfcda0d69c48  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/val/report_paths.txt
5dac9c90fdd4ff4035c2fe3ef4d931458b4efe8397c5e439e54cec27e4e7bd82  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/val/run_meta.json
06928c91729ea4da9032ced75e0ca3013e7b75c6a1575f9dfa661fb3566d1fe5  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/test/embeddings.npy
819a12e8cad47ac473cf2aa6bc8ff2bc8a6556b7669762cc594ed0586282df56  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/test/report_manifest.csv
095aeb101d7bbc873ea87c8ae2e6d79263da9f619df11fea907c5b948ffb54d7  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/test/report_ids.json
c0a11cdfe7093cc08e552a1ff5ae69838e22384e7f68167ca39a0364bc7250cf  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/test/report_paths.txt
64be275fdd789ebe1a2ecc67ecb55ea24471c46e37e78e94194d30c1d37c10bf  /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/test/run_meta.json
```

## Important Reproduction Notes

- `experiment_meta.json` and `run_meta.json` include timestamps, so those metadata hashes will change if you rerun the experiment.
- For this model, `--pooling auto` resolved to `get_projected_text_embeddings`, which yields the 128-dimensional projected CLS embedding instead of the raw 768-dimensional hidden-state CLS token.
- The tokenizer advertises an effectively unbounded sentinel `model_max_length`, so the script intentionally falls back to `512` unless `--max-length` is passed explicitly.
- Because this run used CUDA AMP (`--fp16-on-cuda`) and Hugging Face remote code (`--trust-remote-code`) without a pinned model revision, exact bitwise reproduction is most likely when using the same software stack and cached model snapshot listed above.
- If you need a stable directory name, pass an explicit numbered `--experiment-name` as shown above.

## Script Behaviors That Matter

These script behaviors are relevant to this experiment and were present in the script version identified above:

- Operation-prefixed naming is enforced so report experiments start with `report_embedding_generation__`.
- Split aliases are normalized, so `tst` maps to `test`.
- The script loads manifest rows once and aligns each report by `Path(image_path).stem`.
- `--pooling auto` prefers model-native projected text features such as `get_projected_text_embeddings` before falling back to generic hidden-state pooling.
- `--trust-remote-code` is required for the `microsoft/BiomedVLP-CXR-BERT-specialized` custom model classes.
- Split outputs include `embeddings.npy`, `report_manifest.csv`, `report_ids.json`, `report_paths.txt`, and `run_meta.json`.

Relevant script locations:

- Defaults and operation label: `/workspace/scripts/generate_nih_split_report_embeddings.py:32`
- Operation-prefix helper: `/workspace/scripts/generate_nih_split_report_embeddings.py:181`
- Split alias normalization: `/workspace/scripts/generate_nih_split_report_embeddings.py:252`
- Max-length fallback: `/workspace/scripts/generate_nih_split_report_embeddings.py:276`
- Single-pass manifest loading: `/workspace/scripts/generate_nih_split_report_embeddings.py:288`
- Auto pooling selection: `/workspace/scripts/generate_nih_split_report_embeddings.py:457`
- Hugging Face remote-code loader: `/workspace/scripts/generate_nih_split_report_embeddings.py:627`
- Output writers: `/workspace/scripts/generate_nih_split_report_embeddings.py:603`
- CLI flags including `--trust-remote-code`: `/workspace/scripts/generate_nih_split_report_embeddings.py:813`

## Agent Handoff Text

If you want to hand this off to another agent, this is enough:

```text
Use /workspace/scripts/generate_nih_split_report_embeddings.py and the report /workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test/recreation_report.md to recreate the finalized NIH CXR14 report embedding experiment. Run the exact command in the report with --trust-remote-code, verify the split counts and SHA-256 hashes, and confirm that auto pooling resolves to get_projected_text_embeddings with 128-D L2-normalized outputs.
```
