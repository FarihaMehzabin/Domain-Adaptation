# Source Retrieval Memory Recreation Report

## Scope

This report documents how to recreate the source retrieval-memory experiment stored at:

`/workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory`

The producing script is:

`/workspace/scripts/05_build_source_retrieval_memory.py`

Script SHA-256:

`f1719539713a2b256d1b7cedbf5eecd488fb4a157225b12bda496b33af59ebce`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory`
- Experiment id: `exp0008`
- Operation label: `source_retrieval_memory_building`
- Source embedding root: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`
- Baseline reference experiment: `/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Split used for memory: `train`
- Memory granularity: `instance`
- Retrieval key family: `fused image+report embeddings`
- Stored rows: `78,571`
- Embedding dimension: `2176`
- Label count: `14`
- Label names: `atelectasis cardiomegaly consolidation edema pleural_effusion emphysema fibrosis hernia infiltration mass nodule pleural_thickening pneumonia pneumothorax`
- Index type: `faiss.IndexFlatIP`
- Similarity metric: `inner product on L2-normalized vectors`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- faiss: `1.13.2`
- Platform: `Linux-5.15.0-102-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/05_build_source_retrieval_memory.py --embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split train --self-retrieval-sample-size 512 --label-agreement-queries 200 --qualitative-queries 10 --top-k 5 --seed 3407 --experiment-name exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/05_build_source_retrieval_memory.py --embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --baseline-experiment-dir /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split train --self-retrieval-sample-size 512 --label-agreement-queries 200 --qualitative-queries 10 --top-k 5 --seed 3407 --experiment-name source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory
```

## Preconditions

- The fused embedding experiment must already exist at `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`.
- The selected source baseline should already exist at `/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The source split `train` must contain `embeddings.npy`, a row-identity sidecar, and aligned image paths.
- The required Python packages must be importable: `numpy`, `faiss`.

## Input Summary

- Split directory: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/train`
- Source embeddings: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/train/embeddings.npy`
- Source sidecar: `row_ids.json`
- Source sidecar parser: `identity`
- Source rows: `78,571`
- Source embedding dim: `2176`
- Run-meta fusion mode: `concat`
- Run-meta source order: `image report`

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

- Memory rows: `78,571`
- Memory embedding dimension: `2176`
- Raw norm mean before defensive normalization: `1.00000000`
- Post-normalization norm mean: `1.00000001`
- Self-retrieval top-1 hit rate on sample: `0.998047`
- Self-retrieval top-5 contains-self rate on sample: `1.000000`
- Positive prevalence lift in sampled label-agreement check: `2.053173`
- Positive Jaccard lift in sampled label-agreement check: `1.885381`

## Output Sizes

- .gitignore: `29B`
- experiment_meta.json: `21.46K`
- embeddings.npy: `652.20M`
- labels.npy: `4.20M`
- row_ids.json: `1.35M`
- image_paths.txt: `3.67M`
- items.jsonl: `24.63M`
- sanity_report.json: `10.04K`
- qualitative_neighbors.json: `31.95K`
- index.faiss: `652.20M`
- Total output size: `1.31G`

## Final Artifact SHA-256

```text
d7c5abde1aa5bf7de39fe43910ec023ad5bf34e8719d107c5774158f4373351b  /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory/.gitignore
275619c3bffe405f6cd78c1fd35660d8a68289534cb6f01cc55dd98af60052f8  /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory/experiment_meta.json
3f49961992106f6b091fb105161205292712a8a7999a8f504db712badb381794  /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory/embeddings.npy
f123914b323655e88659a4cb157157f6267e27cbf9197b36a8e4acc09097a36e  /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory/labels.npy
97f392a83de62936f8ee4de44e74b9b90534fcf28ec3350c1bf3838fd8102d86  /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory/row_ids.json
3d3987160b17c97ec1e56f7e13de4f5db114f936a6b7441cc2d4990bfe239e3f  /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory/image_paths.txt
845eb182aee6a43813b45c12fce7c23bc5ed6dd53a701167a808bd94b10e5222  /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory/items.jsonl
13d19ebb54eac9f0007c673d37c848e059bcaae97b9dcd4ee458b0ec80314105  /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory/sanity_report.json
919ebf7ad0d918c3cf105cf3dbd53110e88aa8cf9fb72247cdc236d680187ff3  /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory/qualitative_neighbors.json
41f9785c45abbb5dc21d1a4f19849f0670ed99a79be5e920f4a59646031ec4cc  /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory/index.faiss
```

## Important Reproduction Notes

- `experiment_meta.json`, `sanity_report.json`, `qualitative_neighbors.json`, and `recreation_report.md` include timestamps or sampled summaries, so their hashes can change on rerun.
- `embeddings.npy` and `index.faiss` are large duplicated retrieval artifacts and are intentionally ignored by the experiment-local `.gitignore` for plain Git commits.
- `labels.npy`, `row_ids.json`, `image_paths.txt`, and `items.jsonl` are the main pushable provenance/value artifacts for this memory stage.
- The script defensively re-normalizes the fused embeddings even though the source fused experiment already reports unit-length rows.
- The FAISS index is frozen after build. No training or hyperparameter tuning happens in `exp0008`.

## Agent Handoff Text

```text
Use /workspace/scripts/05_build_source_retrieval_memory.py and the report /workspace/experiments/exp0008__source_retrieval_memory_building__nih_cxr14_exp0003_fused_train_instance_memory/recreation_report.md to recreate the frozen NIH CXR14 source retrieval memory from /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2. Build the train-only instance memory, verify the sanity_report metrics, and confirm that embeddings.npy and index.faiss are present locally even though they are excluded from plain Git.
```
