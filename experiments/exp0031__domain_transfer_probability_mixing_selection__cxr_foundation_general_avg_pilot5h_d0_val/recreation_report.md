# Recreation Report

- Experiment directory: `/workspace/experiments/exp0031__domain_transfer_probability_mixing_selection__cxr_foundation_general_avg_pilot5h_d0_val`
- Script: `/workspace/scripts/24_evaluate_domain_transfer_probability_mixing.py`
- Script SHA-256: `288c854867f58c8cf9d559c4e94a1074b48c72e68103b0b76a64a5a2eb4a46de`
- Run date UTC: `2026-04-12T06:21:33.890317+00:00`

## Command

```bash
/workspace/scripts/24_evaluate_domain_transfer_probability_mixing.py --memory-eval-root /workspace/experiments/exp0029__domain_transfer_source_memory_selection__cxr_foundation_general_avg_pilot5h_d0_val --baseline-experiment-dir /workspace/experiments/exp0015__domain_transfer_linear_probe__cxr_foundation_general_avg_pilot5h --query-embedding-root /workspace/experiments/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128 --manifest-csv /workspace/manifest_common_labels_pilot5h.csv --query-domain d0_nih --query-split val --split-alias d0_val --experiment-name domain_transfer_probability_mixing_selection__cxr_foundation_general_avg_pilot5h_d0_val
```

## Summary

- Memory evaluation root: `/workspace/experiments/exp0029__domain_transfer_source_memory_selection__cxr_foundation_general_avg_pilot5h_d0_val`
- Query split: `d0_nih/val`
- Split alias: `d0_val`
- Best alpha: `1.0`
- Best macro AUROC: `0.847934`
