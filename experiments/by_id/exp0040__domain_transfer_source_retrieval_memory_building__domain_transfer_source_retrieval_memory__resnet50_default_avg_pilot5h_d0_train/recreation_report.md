# Recreation Report

- Experiment directory: `/workspace/experiments/exp0040__domain_transfer_source_retrieval_memory_building__domain_transfer_source_retrieval_memory__resnet50_default_avg_pilot5h_d0_train`
- Script: `/workspace/scripts/22_build_domain_transfer_source_memory.py`
- Script SHA-256: `b9b53e34767a910fade5f2bc9cbe8529dbe4f11f73fb5286029a0730abb0dddc`
- Run date UTC: `2026-04-12T06:37:48.558718+00:00`

## Command

```bash
/workspace/scripts/22_build_domain_transfer_source_memory.py --embedding-root /workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h --baseline-experiment-dir /workspace/experiments/exp0013__domain_transfer_linear_probe__resnet50_default_avg_pilot5h --manifest-csv /workspace/manifest_common_labels_pilot5h.csv --experiment-name domain_transfer_source_retrieval_memory__resnet50_default_avg_pilot5h_d0_train
```

## Summary

- Source split: `d0_nih/train`
- Embedding root: `/workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h`
- Manifest: `/workspace/manifest_common_labels_pilot5h.csv`
- Memory rows: `10,000`
- Embedding dim: `2048`
- Baseline reference: `/workspace/experiments/exp0013__domain_transfer_linear_probe__resnet50_default_avg_pilot5h`
