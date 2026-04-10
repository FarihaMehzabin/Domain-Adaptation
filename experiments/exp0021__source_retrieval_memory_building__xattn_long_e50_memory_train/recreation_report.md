# Source Retrieval Memory Recreation Report

## Scope

This report documents how to recreate the source retrieval-memory experiment stored at:

`/workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train`

The producing script is:

`/workspace/scripts/05_build_source_retrieval_memory.py`

Script SHA-256:

`1c43971d984bad0f36c2a65df456ffd3c86e65e2c1fc5a6c213f687eaa9bc98e`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train`
- Experiment id: `exp0021`
- Operation label: `source_retrieval_memory_building`
- Source embedding root: `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160`
- Baseline reference experiment: `/workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Split used for memory: `train`
- Memory granularity: `instance`
- Retrieval key family: `fused image+report embeddings`
- Stored rows: `78,569`
- Embedding dimension: `512`
- Label count: `14`
- Label names: `atelectasis cardiomegaly consolidation edema pleural_effusion emphysema fibrosis hernia infiltration mass nodule pleural_thickening pneumonia pneumothorax`
- Index type: `faiss.IndexFlatIP`
- Similarity metric: `inner product on L2-normalized vectors`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- faiss: `1.13.2`
- Platform: `Linux-6.8.0-65-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/05_build_source_retrieval_memory.py --embedding-root /workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160 --baseline-experiment-dir /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split train --self-retrieval-sample-size 512 --label-agreement-queries 200 --qualitative-queries 10 --top-k 5 --seed 3407 --experiment-name exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/05_build_source_retrieval_memory.py --embedding-root /workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160 --baseline-experiment-dir /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split train --self-retrieval-sample-size 512 --label-agreement-queries 200 --qualitative-queries 10 --top-k 5 --seed 3407 --experiment-name source_retrieval_memory_building__xattn_long_e50_memory_train
```

## Preconditions

- The fused embedding experiment must already exist at `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160`.
- The selected source baseline should already exist at `/workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The source split `train` must contain `embeddings.npy`, a row-identity sidecar, and aligned image paths.
- The required Python packages must be importable: `numpy`, `faiss`.

## Input Summary

- Split directory: `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160/train`
- Source embeddings: `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160/train/embeddings.npy`
- Source sidecar: `row_ids.json`
- Source sidecar parser: `identity`
- Source rows: `78,569`
- Source embedding dim: `512`
- Run-meta fusion mode: `unknown`
- Run-meta source order: ``

## Expected Outputs

- `.gitignore`
- `experiment_meta.json`
- `recreation_report.md`
- `embeddings.npy`
- `labels.npy`
- `row_ids.json`
- `image_paths.txt`
- `items.jsonl`
- `sanity_report.json`
- `qualitative_neighbors.json`
- `index.faiss`

## Expected Counts And Sanity

- Memory rows: `78,569`
- Memory embedding dimension: `512`
- Raw norm mean before defensive normalization: `1.00000001`
- Post-normalization norm mean: `1.00000001`
- Self-retrieval top-1 hit rate on sample: `1.000000`
- Self-retrieval top-5 contains-self rate on sample: `1.000000`
- Positive prevalence lift in sampled label-agreement check: `1.973951`
- Positive Jaccard lift in sampled label-agreement check: `1.601705`

## Output Sizes

- .gitignore: `29B`
- experiment_meta.json: `19.11K`
- embeddings.npy: `153.46M`
- labels.npy: `4.20M`
- row_ids.json: `1.35M`
- image_paths.txt: `3.67M`
- items.jsonl: `24.63M`
- sanity_report.json: `9.82K`
- qualitative_neighbors.json: `32.00K`
- index.faiss: `153.46M`
- Total output size: `340.81M`

## Final Artifact SHA-256

```text
d7c5abde1aa5bf7de39fe43910ec023ad5bf34e8719d107c5774158f4373351b  /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train/.gitignore
70875f6e1ef4b0145e7b8fefcd4c1daff7237adf94ff81ad041d61d937daa7db  /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train/experiment_meta.json
da687eb8509be714b256c04c10bad7540dd91d36562b2cda89de1da536614d78  /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train/embeddings.npy
463905cd8ddeb4cdbe14b2bdfdcdc888b5fce578556b757520a8942ec8c3373c  /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train/labels.npy
89134b3866f139ebf5068fc192cc1a97ca976ac1127402371209f936ec89de97  /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train/row_ids.json
852f4e97d7c32462e20fdfa2a89d4549dd89da7a9c7456f6f7564270c97ff246  /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train/image_paths.txt
69bd7df8193105e3681737930cc0cfe63e80712fc26c588f9c8208baf6f54bcc  /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train/items.jsonl
eb46cda09970c7a0e8573e7fe4db87d7534782282a164eeab7e7b1159ba1c5a1  /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train/sanity_report.json
f65173ef435ad6a82437825648fad9625dc2d14fc499d8d52ee26a2ebd0afda6  /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train/qualitative_neighbors.json
c6e52103748541cfc71bbcf4d0501ba9800a88ad81c7f9a1bd849c9daa04baf3  /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train/index.faiss
```

## Important Reproduction Notes

- `experiment_meta.json`, `sanity_report.json`, `qualitative_neighbors.json`, and `recreation_report.md` include timestamps or sampled summaries, so their hashes can change on rerun.
- `embeddings.npy` and `index.faiss` are large duplicated retrieval artifacts and are intentionally ignored by the experiment-local `.gitignore` for plain Git commits.
- `labels.npy`, `row_ids.json`, `image_paths.txt`, and `items.jsonl` are the main pushable provenance/value artifacts for this memory stage.
- The script defensively re-normalizes the fused embeddings even though the source fused experiment already reports unit-length rows.
- The FAISS index is frozen after build. No training or hyperparameter tuning happens in `exp0005`.

## Agent Handoff Text

```text
Use /workspace/scripts/05_build_source_retrieval_memory.py and the report /workspace/experiments/exp0021__source_retrieval_memory_building__xattn_long_e50_memory_train/recreation_report.md to recreate the frozen NIH CXR14 source retrieval memory from /workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160. Build the train-only instance memory, verify the sanity_report metrics, and confirm that embeddings.npy and index.faiss are present locally even though they are excluded from plain Git.
```
