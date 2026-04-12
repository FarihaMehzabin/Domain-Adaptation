# Recreation Report

- Experiment directory: `/workspace/experiments/exp0043__domain_transfer_source_memory_target_evaluation__resnet50_default_avg_pilot5h_d0_test`
- Script: `/workspace/scripts/25_evaluate_domain_transfer_source_memory_target.py`
- Script SHA-256: `51cdc657a576b8fe9f96070b18cda146a1fd683e6ae0d8860b6e925e7cf62941`
- Run date UTC: `2026-04-12T06:38:37.933151+00:00`

## Command

```bash
/workspace/scripts/25_evaluate_domain_transfer_source_memory_target.py --memory-root /workspace/experiments/exp0040__domain_transfer_source_retrieval_memory_building__domain_transfer_source_retrieval_memory__resnet50_default_avg_pilot5h_d0_train --memory-selection-root /workspace/experiments/exp0041__domain_transfer_source_memory_selection__resnet50_default_avg_pilot5h_d0_val --baseline-experiment-dir /workspace/experiments/exp0013__domain_transfer_linear_probe__resnet50_default_avg_pilot5h --query-embedding-root /workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h --manifest-csv /workspace/manifest_common_labels_pilot5h.csv --query-domain d0_nih --query-split test --split-alias d0_test --experiment-name domain_transfer_source_memory_target_evaluation__resnet50_default_avg_pilot5h_d0_test
```

## Summary

- Memory selection root: `/workspace/experiments/exp0041__domain_transfer_source_memory_selection__resnet50_default_avg_pilot5h_d0_val`
- Query split: `d0_nih/test`
- Split alias: `d0_test`
- Frozen k/tau: `50` / `5.0`
- Memory macro AUROC: `0.678102`
