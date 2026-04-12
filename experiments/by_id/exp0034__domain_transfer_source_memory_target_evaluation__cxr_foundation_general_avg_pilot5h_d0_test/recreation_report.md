# Recreation Report

- Experiment directory: `/workspace/experiments/exp0034__domain_transfer_source_memory_target_evaluation__cxr_foundation_general_avg_pilot5h_d0_test`
- Script: `/workspace/scripts/25_evaluate_domain_transfer_source_memory_target.py`
- Script SHA-256: `6a7835b0bac2ddc97311eaf89f6f5258e2dc93b42e0d16b93c99f9ff6a768487`
- Run date UTC: `2026-04-12T06:21:57.957285+00:00`

## Command

```bash
/workspace/scripts/25_evaluate_domain_transfer_source_memory_target.py --memory-root /workspace/experiments/exp0028__domain_transfer_source_retrieval_memory_building__domain_transfer_source_retrieval_memory__cxr_foundation_general_avg_pilot5h_d0_train --memory-selection-root /workspace/experiments/exp0029__domain_transfer_source_memory_selection__cxr_foundation_general_avg_pilot5h_d0_val --baseline-experiment-dir /workspace/experiments/exp0015__domain_transfer_linear_probe__cxr_foundation_general_avg_pilot5h --query-embedding-root /workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128 --manifest-csv /workspace/manifest_common_labels_pilot5h.csv --query-domain d0_nih --query-split test --split-alias d0_test --experiment-name domain_transfer_source_memory_target_evaluation__cxr_foundation_general_avg_pilot5h_d0_test
```

## Summary

- Memory selection root: `/workspace/experiments/exp0029__domain_transfer_source_memory_selection__cxr_foundation_general_avg_pilot5h_d0_val`
- Query split: `d0_nih/test`
- Split alias: `d0_test`
- Frozen k/tau: `50` / `5.0`
- Memory macro AUROC: `0.805224`
