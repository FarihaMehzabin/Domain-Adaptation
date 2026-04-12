# Recreation Report

- Experiment directory: `/workspace/experiments/exp0029__domain_transfer_source_memory_selection__cxr_foundation_general_avg_pilot5h_d0_val`
- Script: `/workspace/scripts/23_evaluate_domain_transfer_source_memory.py`
- Script SHA-256: `e009b951c3b2ae84f031d1b52db4ca5c41c6fad1f4555f7ca5c3d28ab193abc5`
- Run date UTC: `2026-04-12T06:21:10.328534+00:00`

## Command

```bash
/workspace/scripts/23_evaluate_domain_transfer_source_memory.py --memory-root /workspace/experiments/exp0028__domain_transfer_source_retrieval_memory_building__domain_transfer_source_retrieval_memory__cxr_foundation_general_avg_pilot5h_d0_train --baseline-experiment-dir /workspace/experiments/exp0015__domain_transfer_linear_probe__cxr_foundation_general_avg_pilot5h --query-embedding-root /workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128 --manifest-csv /workspace/manifest_common_labels_pilot5h.csv --query-domain d0_nih --query-split val --split-alias d0_val --experiment-name domain_transfer_source_memory_selection__cxr_foundation_general_avg_pilot5h_d0_val
```

## Summary

- Memory root: `/workspace/experiments/exp0028__domain_transfer_source_retrieval_memory_building__domain_transfer_source_retrieval_memory__cxr_foundation_general_avg_pilot5h_d0_train`
- Query split: `d0_nih/val`
- Split alias: `d0_val`
- Best k/tau: `50` / `5.0`
- Best macro AUROC: `0.791292`
