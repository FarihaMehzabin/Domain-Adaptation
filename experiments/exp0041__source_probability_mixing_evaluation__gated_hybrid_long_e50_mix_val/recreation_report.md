# Source Probability-Mixing Recreation Report

## Scope

This report documents how to recreate the validation-only probability-mixing experiment stored at:

`/workspace/experiments/exp0041__source_probability_mixing_evaluation__gated_hybrid_long_e50_mix_val`

The producing script is:

`/workspace/scripts/07_evaluate_probability_mixing.py`

Script SHA-256:

`f643aca68d35d4c2a6ae4b3255865748538f1646247af95c8d75e4c04de16588`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0041__source_probability_mixing_evaluation__gated_hybrid_long_e50_mix_val`
- Experiment id: `exp0041`
- Operation label: `source_probability_mixing_evaluation`
- Memory-evaluation root: `/workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val`
- Baseline experiment: `/workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096`
- Query embedding root: `/workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Evaluation split: `val`
- Validation rows: `11,219`
- Label count: `14`
- Label names: `atelectasis cardiomegaly consolidation edema pleural_effusion emphysema fibrosis hernia infiltration mass nodule pleural_thickening pneumonia pneumothorax`
- Selection metric: `macro_auroc`
- Alpha values: `0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- PyTorch: `2.4.1+cu124`
- Platform: `Linux-6.8.0-65-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val --baseline-experiment-dir /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096 --query-embedding-root /workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --alpha-values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --seed 3407 --experiment-name exp0041__source_probability_mixing_evaluation__gated_hybrid_long_e50_mix_val --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val --baseline-experiment-dir /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096 --query-embedding-root /workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --alpha-values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --seed 3407 --experiment-name source_probability_mixing_evaluation__gated_hybrid_long_e50_mix_val
```

## Preconditions

- The memory-only evaluation must already exist at `/workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val`.
- The baseline experiment must already exist at `/workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096`.
- The query embeddings must already exist at `/workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096/best.ckpt`
- Memory probabilities: `/workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val/val_probabilities.npy`
- Query embedding split: `/workspace/experiments/exp0037__source_cross_attention_embedding_export__gated_hybrid_long_e50_export_bs160/val`
- Validation rows: `11,219`
- FAISS note: `Not used directly in this stage; memory probabilities come from exp0006.`

## Sweep Summary

| alpha | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.709453 | 0.132715 | 0.011119 | 0.017974 |  |
| 0.1 | 0.748830 | 0.137826 | 0.032502 | 0.027761 |  |
| 0.2 | 0.752887 | 0.139633 | 0.064718 | 0.044517 |  |
| 0.3 | 0.754516 | 0.140802 | 0.097010 | 0.075873 |  |
| 0.4 | 0.755260 | 0.141600 | 0.129292 | 0.108002 |  |
| 0.5 | 0.755588 | 0.142308 | 0.161584 | 0.157233 |  |
| 0.6 | 0.755682 | 0.142884 | 0.193870 | 0.194022 | <- best |
| 0.7 | 0.755620 | 0.143196 | 0.226165 | 0.194914 |  |
| 0.8 | 0.755439 | 0.144493 | 0.258451 | 0.187062 |  |
| 0.9 | 0.755173 | 0.143621 | 0.290737 | 0.177323 |  |
| 1.0 | 0.754822 | 0.140318 | 0.323026 | 0.169874 |  |

## Best Configuration

- Best alpha: `0.6`
- Validation macro AUROC: `0.755682`
- Validation macro average precision: `0.142884`
- Validation macro ECE: `0.193870`
- Validation macro F1 @ 0.5: `0.194022`
- Diagnostic macro F1 @ tuned thresholds: `0.213489`

## Baseline Comparison

- Frozen baseline validation macro AUROC: `0.754822`
- Frozen baseline validation macro average precision: `0.140318`
- Best mixed minus baseline macro AUROC: `0.000860`
- Best mixed minus baseline macro average precision: `0.002566`
- Baseline reconstruction matches archived exp0004 forward metrics within 5e-4: `false`
- Baseline reconstruction max absolute metric delta: `0.000588444829`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `sweep_results.json`
- `best_config.json`
- `best_val_metrics.json`
- `val_mixed_probabilities.npy`
- `probability_mixing_selection.md`

## Output Sizes

- experiment_meta.json: `56.89K`
- sweep_results.json: `96.33K`
- best_config.json: `1.18K`
- best_val_metrics.json: `6.36K`
- val_mixed_probabilities.npy: `613.66K`
- probability_mixing_selection.md: `419B`
- Total output size: `774.83K`

## Final Artifact SHA-256

```text
5152b5464b07968d31cafb21bda6443995aafc4efbab76d9ad328decbbaba75e  /workspace/experiments/exp0041__source_probability_mixing_evaluation__gated_hybrid_long_e50_mix_val/experiment_meta.json
23c390c93d3321d4dd54b0d4dc7eb11130fb3e0c6d4fcb5a5e18ce39fcb5352c  /workspace/experiments/exp0041__source_probability_mixing_evaluation__gated_hybrid_long_e50_mix_val/sweep_results.json
82e244de799382dbea403f542e33867e8f59e0da39888600c7386a91a014999d  /workspace/experiments/exp0041__source_probability_mixing_evaluation__gated_hybrid_long_e50_mix_val/best_config.json
f6dcb8497a9a3e5d453b84807a7fdba9cf4d0c201f50e21f00cfab5e3bd0ed7d  /workspace/experiments/exp0041__source_probability_mixing_evaluation__gated_hybrid_long_e50_mix_val/best_val_metrics.json
9c6694ff4b7686de276056b1eed78c9eaa48fc52ff122749521099858081bcb3  /workspace/experiments/exp0041__source_probability_mixing_evaluation__gated_hybrid_long_e50_mix_val/val_mixed_probabilities.npy
b939f291e4a48852a956066de162e00bbb475af1499feae5d33aab9e06b39aa2  /workspace/experiments/exp0041__source_probability_mixing_evaluation__gated_hybrid_long_e50_mix_val/probability_mixing_selection.md
```

## Important Reproduction Notes

- All selection in `exp0007` is validation-only.
- `alpha=1.0` corresponds to the frozen baseline alone, and `alpha=0.0` corresponds to the selected memory-only probabilities from `exp0006`.
- `val_mixed_probabilities.npy` stores the best-config mixed probabilities in validation row order.
- Tied settings are resolved conservatively in favor of larger alpha.

## Agent Handoff Text

```text
Use /workspace/scripts/07_evaluate_probability_mixing.py and the report /workspace/experiments/exp0041__source_probability_mixing_evaluation__gated_hybrid_long_e50_mix_val/recreation_report.md to recreate the validation-only probability-mixing stage that combines /workspace/experiments/exp0038__source_baseline_training__gated_hybrid_long_e50_baseline_bs4096 with /workspace/experiments/exp0040__source_memory_only_evaluation__gated_hybrid_long_e50_memory_val. Reconstruct the frozen baseline validation probabilities, mix them with the exp0006 memory probabilities across alpha in [0.0, 1.0], and verify the saved best_config.json, best_val_metrics.json, and val_mixed_probabilities.npy artifacts.
```
