# Recreation Report

- Experiment directory: `/workspace/experiments/exp0041__domain_transfer_source_memory_selection__resnet50_default_avg_pilot5h_d0_val`
- Script: `/workspace/scripts/23_evaluate_domain_transfer_source_memory.py`
- Script SHA-256: `2cf53bb76f07d51c2e4c18f3219e06aa7dda04f5ac9430dfaa500e94846ac067`
- Run date UTC: `2026-04-12T06:38:01.937576+00:00`

## Command

```bash
/workspace/scripts/23_evaluate_domain_transfer_source_memory.py --memory-root /workspace/experiments/exp0040__domain_transfer_source_retrieval_memory_building__domain_transfer_source_retrieval_memory__resnet50_default_avg_pilot5h_d0_train --baseline-experiment-dir /workspace/experiments/exp0013__domain_transfer_linear_probe__resnet50_default_avg_pilot5h --query-embedding-root /workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h --manifest-csv /workspace/manifest_common_labels_pilot5h.csv --query-domain d0_nih --query-split val --split-alias d0_val --experiment-name domain_transfer_source_memory_selection__resnet50_default_avg_pilot5h_d0_val
```

## Summary

- Memory root: `/workspace/experiments/exp0040__domain_transfer_source_retrieval_memory_building__domain_transfer_source_retrieval_memory__resnet50_default_avg_pilot5h_d0_train`
- Query split: `d0_nih/val`
- Split alias: `d0_val`
- Best k/tau: `50` / `5.0`
- Best macro AUROC: `0.675833`
