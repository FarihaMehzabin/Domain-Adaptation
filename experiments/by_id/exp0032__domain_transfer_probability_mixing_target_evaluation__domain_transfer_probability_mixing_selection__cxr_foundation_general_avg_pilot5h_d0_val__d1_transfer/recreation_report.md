# Recreation Report

- Experiment directory: `/workspace/experiments/exp0031__domain_transfer_probability_mixing_target_evaluation__domain_transfer_probability_mixing_selection__cxr_foundation_general_avg_pilot5h_d0_val__d1_transfer`
- Script: `/workspace/scripts/26_evaluate_domain_transfer_probability_mixing_target.py`
- Script SHA-256: `fd7d199a586093646bea6edce65491507a746afca009df90535df401579a10c1`
- Run date UTC: `2026-04-12T06:24:09.832176+00:00`

## Command

```bash
/workspace/scripts/26_evaluate_domain_transfer_probability_mixing_target.py --memory-eval-root /workspace/experiments/exp0035__domain_transfer_source_memory_target_evaluation__cxr_foundation_general_avg_pilot5h_d1_transfer --mixing-selection-root /workspace/experiments/exp0031__domain_transfer_probability_mixing_selection__cxr_foundation_general_avg_pilot5h_d0_val --baseline-experiment-dir /workspace/experiments/exp0015__domain_transfer_linear_probe__cxr_foundation_general_avg_pilot5h --query-embedding-root /workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128 --manifest-csv /workspace/manifest_common_labels_pilot5h.csv --query-domain d1_chexpert --query-split val --split-alias d1_transfer
```

## Summary

- Mixing selection root: `/workspace/experiments/exp0031__domain_transfer_probability_mixing_selection__cxr_foundation_general_avg_pilot5h_d0_val`
- Memory evaluation root: `/workspace/experiments/exp0035__domain_transfer_source_memory_target_evaluation__cxr_foundation_general_avg_pilot5h_d1_transfer`
- Query split: `d1_chexpert/val`
- Split alias: `d1_transfer`
- Frozen alpha: `1.0`
- Mixed macro AUROC: `0.845202`
