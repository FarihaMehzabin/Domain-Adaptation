# Recreation Report

- Experiment directory: `/workspace/experiments/exp0042__domain_transfer_probability_mixing_selection__resnet50_default_avg_pilot5h_d0_val`
- Script: `/workspace/scripts/24_evaluate_domain_transfer_probability_mixing.py`
- Script SHA-256: `e4e322ff2b9cdf48a3c0f9972ad947da7f9b354b14f3fe9699db951bc0675a6d`
- Run date UTC: `2026-04-12T06:38:15.113350+00:00`

## Command

```bash
/workspace/scripts/24_evaluate_domain_transfer_probability_mixing.py --memory-eval-root /workspace/experiments/exp0041__domain_transfer_source_memory_selection__resnet50_default_avg_pilot5h_d0_val --baseline-experiment-dir /workspace/experiments/exp0013__domain_transfer_linear_probe__resnet50_default_avg_pilot5h --query-embedding-root /workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h --manifest-csv /workspace/manifest_common_labels_pilot5h.csv --query-domain d0_nih --query-split val --split-alias d0_val --experiment-name domain_transfer_probability_mixing_selection__resnet50_default_avg_pilot5h_d0_val
```

## Summary

- Memory evaluation root: `/workspace/experiments/exp0041__domain_transfer_source_memory_selection__resnet50_default_avg_pilot5h_d0_val`
- Query split: `d0_nih/val`
- Split alias: `d0_val`
- Best alpha: `0.7`
- Best macro AUROC: `0.739558`
