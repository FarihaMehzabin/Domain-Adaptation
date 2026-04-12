# Recreation Report

- Experiment directory: `/workspace/experiments/exp0028__domain_transfer_source_retrieval_memory_building__domain_transfer_source_retrieval_memory__cxr_foundation_general_avg_pilot5h_d0_train`
- Script: `/workspace/scripts/22_build_domain_transfer_source_memory.py`
- Script SHA-256: `3a4fe4a81775594caa25acef6f5bdae23c2817cdfbaac4a3194efffce682a326`
- Run date UTC: `2026-04-12T06:20:51.268132+00:00`

## Command

```bash
/workspace/scripts/22_build_domain_transfer_source_memory.py --embedding-root /workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128 --baseline-experiment-dir /workspace/experiments/exp0015__domain_transfer_linear_probe__cxr_foundation_general_avg_pilot5h --manifest-csv /workspace/manifest_common_labels_pilot5h.csv --experiment-name domain_transfer_source_retrieval_memory__cxr_foundation_general_avg_pilot5h_d0_train
```

## Summary

- Source split: `d0_nih/train`
- Embedding root: `/workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128`
- Manifest: `/workspace/manifest_common_labels_pilot5h.csv`
- Memory rows: `10,000`
- Embedding dim: `768`
- Baseline reference: `/workspace/experiments/exp0015__domain_transfer_linear_probe__cxr_foundation_general_avg_pilot5h`
