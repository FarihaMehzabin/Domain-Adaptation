# Source Probability-Mixing Recreation Report

## Scope

This report documents how to recreate the validation-only probability-mixing experiment stored at:

`/workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val`

The producing script is:

`/workspace/scripts/07_evaluate_probability_mixing.py`

Script SHA-256:

`f643aca68d35d4c2a6ae4b3255865748538f1646247af95c8d75e4c04de16588`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val`
- Experiment id: `exp0015`
- Operation label: `source_probability_mixing_evaluation`
- Memory-evaluation root: `/workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val`
- Baseline experiment: `/workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096`
- Query embedding root: `/workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160`
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
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val --baseline-experiment-dir /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096 --query-embedding-root /workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --alpha-values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --seed 3407 --experiment-name exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val --baseline-experiment-dir /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096 --query-embedding-root /workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --alpha-values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --seed 3407 --experiment-name source_probability_mixing_evaluation__xattn_full_pipeline_mix_val
```

## Preconditions

- The memory-only evaluation must already exist at `/workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val`.
- The baseline experiment must already exist at `/workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096`.
- The query embeddings must already exist at `/workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096/best.ckpt`
- Memory probabilities: `/workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val/val_probabilities.npy`
- Query embedding split: `/workspace/experiments/exp0011__source_cross_attention_embedding_export__xattn_full_pipeline_export_bs160/val`
- Validation rows: `11,219`
- FAISS note: `Not used directly in this stage; memory probabilities come from exp0006.`

## Sweep Summary

| alpha | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.679425 | 0.111083 | 0.010496 | 0.004427 |  |
| 0.1 | 0.716174 | 0.115322 | 0.037081 | 0.006936 |  |
| 0.2 | 0.721191 | 0.116958 | 0.073577 | 0.013807 |  |
| 0.3 | 0.723238 | 0.117791 | 0.110300 | 0.028829 |  |
| 0.4 | 0.724191 | 0.118247 | 0.147061 | 0.052638 |  |
| 0.5 | 0.724625 | 0.119064 | 0.183815 | 0.092005 |  |
| 0.6 | 0.724728 | 0.119263 | 0.220573 | 0.141794 | <- best |
| 0.7 | 0.724606 | 0.119398 | 0.257331 | 0.169083 |  |
| 0.8 | 0.724232 | 0.119476 | 0.294088 | 0.167046 |  |
| 0.9 | 0.723653 | 0.118893 | 0.330849 | 0.159015 |  |
| 1.0 | 0.722818 | 0.117866 | 0.367604 | 0.153240 |  |

## Best Configuration

- Best alpha: `0.6`
- Validation macro AUROC: `0.724728`
- Validation macro average precision: `0.119263`
- Validation macro ECE: `0.220573`
- Validation macro F1 @ 0.5: `0.141794`
- Diagnostic macro F1 @ tuned thresholds: `0.187288`

## Baseline Comparison

- Frozen baseline validation macro AUROC: `0.722818`
- Frozen baseline validation macro average precision: `0.117866`
- Best mixed minus baseline macro AUROC: `0.001910`
- Best mixed minus baseline macro average precision: `0.001398`
- Baseline reconstruction matches archived exp0004 forward metrics within 5e-4: `false`
- Baseline reconstruction max absolute metric delta: `0.000546289282`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `sweep_results.json`
- `best_config.json`
- `best_val_metrics.json`
- `val_mixed_probabilities.npy`
- `probability_mixing_selection.md`

## Output Sizes

- experiment_meta.json: `55.96K`
- sweep_results.json: `96.14K`
- best_config.json: `1.18K`
- best_val_metrics.json: `6.37K`
- val_mixed_probabilities.npy: `613.66K`
- probability_mixing_selection.md: `419B`
- Total output size: `773.73K`

## Final Artifact SHA-256

```text
c3046a03ca8c88da56f3981eed35839b80aa6de2f73328b66e4cd6b0fb6cdca0  /workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val/experiment_meta.json
1725d191d004e1ac306c84ffbc735ce54302fb549e89dd41f8caa06ac51ba00e  /workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val/sweep_results.json
0a1d8396e509742a71afc35325d4322c443f4d865947c94623bee9e113d375a3  /workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val/best_config.json
f7c4559b83fd741373a63d698009d72ebdc260af514ff464ae6a40224f51029b  /workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val/best_val_metrics.json
d418650a272b0ebabbc1231a35dc2aaddadad4ca19a4a6f2e8d7d1aadeb3ed8f  /workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val/val_mixed_probabilities.npy
94e342d30a2b2b6a2688b8c37feb84b79099d6637ace89e11103e5ae17e6a1b2  /workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val/probability_mixing_selection.md
```

## Important Reproduction Notes

- All selection in `exp0007` is validation-only.
- `alpha=1.0` corresponds to the frozen baseline alone, and `alpha=0.0` corresponds to the selected memory-only probabilities from `exp0006`.
- `val_mixed_probabilities.npy` stores the best-config mixed probabilities in validation row order.
- Tied settings are resolved conservatively in favor of larger alpha.

## Agent Handoff Text

```text
Use /workspace/scripts/07_evaluate_probability_mixing.py and the report /workspace/experiments/exp0015__source_probability_mixing_evaluation__xattn_full_pipeline_mix_val/recreation_report.md to recreate the validation-only probability-mixing stage that combines /workspace/experiments/exp0012__source_baseline_training__xattn_full_pipeline_baseline_bs4096 with /workspace/experiments/exp0014__source_memory_only_evaluation__xattn_full_pipeline_memory_val. Reconstruct the frozen baseline validation probabilities, mix them with the exp0006 memory probabilities across alpha in [0.0, 1.0], and verify the saved best_config.json, best_val_metrics.json, and val_mixed_probabilities.npy artifacts.
```
