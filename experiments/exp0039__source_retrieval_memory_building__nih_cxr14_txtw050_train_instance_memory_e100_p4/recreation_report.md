# Source Retrieval Memory Recreation Report

## Scope

This report documents how to recreate the source retrieval-memory experiment stored at:

`/workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4`

The producing script is:

`/workspace/scripts/05_build_source_retrieval_memory.py`

Script SHA-256:

`f1719539713a2b256d1b7cedbf5eecd488fb4a157225b12bda496b33af59ebce`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4`
- Experiment id: `exp0039`
- Operation label: `source_retrieval_memory_building`
- Source embedding root: `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050`
- Baseline reference experiment: `/workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4`
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
- Platform: `Linux-6.8.0-85-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/05_build_source_retrieval_memory.py --embedding-root /workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050 --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split train --self-retrieval-sample-size 512 --label-agreement-queries 200 --qualitative-queries 10 --top-k 5 --seed 3407 --experiment-name exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4 --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/05_build_source_retrieval_memory.py --embedding-root /workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050 --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split train --self-retrieval-sample-size 512 --label-agreement-queries 200 --qualitative-queries 10 --top-k 5 --seed 3407 --experiment-name source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4
```

## Preconditions

- The fused embedding experiment must already exist at `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050`.
- The selected source baseline should already exist at `/workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The source split `train` must contain `embeddings.npy`, a row-identity sidecar, and aligned image paths.
- The required Python packages must be importable: `numpy`, `faiss`.

## Input Summary

- Split directory: `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050/train`
- Source embeddings: `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050/train/embeddings.npy`
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
- Positive prevalence lift in sampled label-agreement check: `2.202494`
- Positive Jaccard lift in sampled label-agreement check: `2.065645`

## Output Sizes

- .gitignore: `29B`
- experiment_meta.json: `22.62K`
- embeddings.npy: `652.20M`
- labels.npy: `4.20M`
- row_ids.json: `1.35M`
- image_paths.txt: `3.67M`
- items.jsonl: `24.63M`
- sanity_report.json: `10.08K`
- qualitative_neighbors.json: `32.09K`
- index.faiss: `652.20M`
- Total output size: `1.31G`

## Final Artifact SHA-256

```text
d7c5abde1aa5bf7de39fe43910ec023ad5bf34e8719d107c5774158f4373351b  /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4/.gitignore
86208f4b7868229ad8aa6e0c9b553e42b2a3d2ca195d697eae40861b789e8bfe  /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4/experiment_meta.json
830e3d093f1f94dfb3587e4d7c8083cabdc57a2e84888cd45de7cde77d4e18dd  /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4/embeddings.npy
f123914b323655e88659a4cb157157f6267e27cbf9197b36a8e4acc09097a36e  /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4/labels.npy
97f392a83de62936f8ee4de44e74b9b90534fcf28ec3350c1bf3838fd8102d86  /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4/row_ids.json
3d3987160b17c97ec1e56f7e13de4f5db114f936a6b7441cc2d4990bfe239e3f  /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4/image_paths.txt
845eb182aee6a43813b45c12fce7c23bc5ed6dd53a701167a808bd94b10e5222  /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4/items.jsonl
e763225492dbd91cee5cf135016c0776e77c366a6920834bf3e6f94eb5436ad5  /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4/sanity_report.json
3334221ecdadf517a45e5ad19b2f9594d332dc91117eb391b480f22718fdffb5  /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4/qualitative_neighbors.json
cc888e61d89623f211478731ccba76b9b1acc07d66eeaf2f1670fbdf7afa4c95  /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4/index.faiss
```

## Important Reproduction Notes

- `experiment_meta.json`, `sanity_report.json`, `qualitative_neighbors.json`, and `recreation_report.md` include timestamps or sampled summaries, so their hashes can change on rerun.
- `embeddings.npy` and `index.faiss` are large duplicated retrieval artifacts and are intentionally ignored by the experiment-local `.gitignore` for plain Git commits.
- `labels.npy`, `row_ids.json`, `image_paths.txt`, and `items.jsonl` are the main pushable provenance/value artifacts for this memory stage.
- The script defensively re-normalizes the fused embeddings even though the source fused experiment already reports unit-length rows.
- The FAISS index is frozen after build. No training or hyperparameter tuning happens in `exp0008`.

## Agent Handoff Text

```text
Use /workspace/scripts/05_build_source_retrieval_memory.py and the report /workspace/experiments/exp0039__source_retrieval_memory_building__nih_cxr14_txtw050_train_instance_memory_e100_p4/recreation_report.md to recreate the frozen NIH CXR14 source retrieval memory from /workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050. Build the train-only instance memory, verify the sanity_report metrics, and confirm that embeddings.npy and index.faiss are present locally even though they are excluded from plain Git.
```
