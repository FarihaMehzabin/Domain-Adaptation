# Source Retrieval Memory Recreation Report

## Scope

This report documents how to recreate the source retrieval-memory experiment stored at:

`/workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train`

The producing script is:

`/workspace/scripts/05_build_source_retrieval_memory.py`

Script SHA-256:

`1c43971d984bad0f36c2a65df456ffd3c86e65e2c1fc5a6c213f687eaa9bc98e`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train`
- Experiment id: `exp0029`
- Operation label: `source_retrieval_memory_building`
- Source embedding root: `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat`
- Baseline reference experiment: `/workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Split used for memory: `train`
- Memory granularity: `instance`
- Retrieval key family: `fused image+report embeddings`
- Stored rows: `78,569`
- Embedding dimension: `2688`
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
python /workspace/scripts/05_build_source_retrieval_memory.py --embedding-root /workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split train --self-retrieval-sample-size 512 --label-agreement-queries 200 --qualitative-queries 10 --top-k 5 --seed 3407 --experiment-name exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/05_build_source_retrieval_memory.py --embedding-root /workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split train --self-retrieval-sample-size 512 --label-agreement-queries 200 --qualitative-queries 10 --top-k 5 --seed 3407 --experiment-name source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train
```

## Preconditions

- The fused embedding experiment must already exist at `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat`.
- The selected source baseline should already exist at `/workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The source split `train` must contain `embeddings.npy`, a row-identity sidecar, and aligned image paths.
- The required Python packages must be importable: `numpy`, `faiss`.

## Input Summary

- Split directory: `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat/train`
- Source embeddings: `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat/train/embeddings.npy`
- Source sidecar: `row_ids.json`
- Source sidecar parser: `identity`
- Source rows: `78,569`
- Source embedding dim: `2688`
- Run-meta fusion mode: `concat`
- Run-meta source order: `xattn image report`

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
- Memory embedding dimension: `2688`
- Raw norm mean before defensive normalization: `1.00000000`
- Post-normalization norm mean: `1.00000001`
- Self-retrieval top-1 hit rate on sample: `1.000000`
- Self-retrieval top-5 contains-self rate on sample: `1.000000`
- Positive prevalence lift in sampled label-agreement check: `2.356007`
- Positive Jaccard lift in sampled label-agreement check: `1.983340`

## Output Sizes

- .gitignore: `29B`
- experiment_meta.json: `22.27K`
- embeddings.npy: `805.64M`
- labels.npy: `4.20M`
- row_ids.json: `1.35M`
- image_paths.txt: `3.67M`
- items.jsonl: `24.63M`
- sanity_report.json: `9.86K`
- qualitative_neighbors.json: `31.97K`
- index.faiss: `805.64M`
- Total output size: `1.61G`

## Final Artifact SHA-256

```text
d7c5abde1aa5bf7de39fe43910ec023ad5bf34e8719d107c5774158f4373351b  /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train/.gitignore
c2de5b12ca8cacea8be4284fefef42e95226a972baa0ff2fc147b3dc9c69aea4  /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train/experiment_meta.json
8437cb952689ab5bd8ea284111af12869ed732e40206e9a3a1c7175fb6beaefa  /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train/embeddings.npy
463905cd8ddeb4cdbe14b2bdfdcdc888b5fce578556b757520a8942ec8c3373c  /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train/labels.npy
89134b3866f139ebf5068fc192cc1a97ca976ac1127402371209f936ec89de97  /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train/row_ids.json
852f4e97d7c32462e20fdfa2a89d4549dd89da7a9c7456f6f7564270c97ff246  /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train/image_paths.txt
69bd7df8193105e3681737930cc0cfe63e80712fc26c588f9c8208baf6f54bcc  /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train/items.jsonl
1deae10eb2e824e5cbd1c7c580f2e640525c8092112efaa8b87ff138311c8665  /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train/sanity_report.json
192bc9333080dcc3e2fcd18ecf01fb94c1bd5f5feb382548d04aae0fae0db9ad  /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train/qualitative_neighbors.json
c42daa209582634d8f6c9a9e3e1759e27b8782ed605fb546ee25cc4e32a4ea2e  /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train/index.faiss
```

## Important Reproduction Notes

- `experiment_meta.json`, `sanity_report.json`, `qualitative_neighbors.json`, and `recreation_report.md` include timestamps or sampled summaries, so their hashes can change on rerun.
- `embeddings.npy` and `index.faiss` are large duplicated retrieval artifacts and are intentionally ignored by the experiment-local `.gitignore` for plain Git commits.
- `labels.npy`, `row_ids.json`, `image_paths.txt`, and `items.jsonl` are the main pushable provenance/value artifacts for this memory stage.
- The script defensively re-normalizes the fused embeddings even though the source fused experiment already reports unit-length rows.
- The FAISS index is frozen after build. No training or hyperparameter tuning happens in `exp0005`.

## Agent Handoff Text

```text
Use /workspace/scripts/05_build_source_retrieval_memory.py and the report /workspace/experiments/exp0029__source_retrieval_memory_building__xattn_long_e50_hybrid_memory_train/recreation_report.md to recreate the frozen NIH CXR14 source retrieval memory from /workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat. Build the train-only instance memory, verify the sanity_report metrics, and confirm that embeddings.npy and index.faiss are present locally even though they are excluded from plain Git.
```
