# Recreation Report

- Experiment directory: `/workspace/experiments/exp0046__domain_transfer_probability_mixing_target_evaluation__resnet50_default_avg_pilot5h_d2_transfer`
- Script: `/workspace/scripts/26_evaluate_domain_transfer_probability_mixing_target.py`
- Script SHA-256: `fd7d199a586093646bea6edce65491507a746afca009df90535df401579a10c1`
- Run date UTC: `2026-04-12T06:39:05.298229+00:00`

## Command

```bash
/workspace/scripts/26_evaluate_domain_transfer_probability_mixing_target.py --memory-eval-root /workspace/experiments/exp0044__domain_transfer_source_memory_target_evaluation__resnet50_default_avg_pilot5h_d2_transfer --mixing-selection-root /workspace/experiments/exp0042__domain_transfer_probability_mixing_selection__resnet50_default_avg_pilot5h_d0_val --baseline-experiment-dir /workspace/experiments/exp0013__domain_transfer_linear_probe__resnet50_default_avg_pilot5h --query-embedding-root /workspace/experiments/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h --manifest-csv /workspace/manifest_common_labels_pilot5h.csv --query-domain d2_mimic --query-split test --split-alias d2_transfer --experiment-name domain_transfer_probability_mixing_target_evaluation__resnet50_default_avg_pilot5h_d2_transfer
```

## Summary

- Mixing selection root: `/workspace/experiments/exp0042__domain_transfer_probability_mixing_selection__resnet50_default_avg_pilot5h_d0_val`
- Memory evaluation root: `/workspace/experiments/exp0044__domain_transfer_source_memory_target_evaluation__resnet50_default_avg_pilot5h_d2_transfer`
- Query split: `d2_mimic/test`
- Split alias: `d2_transfer`
- Frozen alpha: `0.7`
- Mixed macro AUROC: `0.504165`
