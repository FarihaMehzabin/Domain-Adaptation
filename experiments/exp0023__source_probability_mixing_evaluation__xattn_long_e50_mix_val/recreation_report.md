# Source Probability-Mixing Recreation Report

## Scope

This report documents how to recreate the validation-only probability-mixing experiment stored at:

`/workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val`

The producing script is:

`/workspace/scripts/07_evaluate_probability_mixing.py`

Script SHA-256:

`f643aca68d35d4c2a6ae4b3255865748538f1646247af95c8d75e4c04de16588`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val`
- Experiment id: `exp0023`
- Operation label: `source_probability_mixing_evaluation`
- Memory-evaluation root: `/workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val`
- Baseline experiment: `/workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096`
- Query embedding root: `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160`
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
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val --baseline-experiment-dir /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096 --query-embedding-root /workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --alpha-values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --seed 3407 --experiment-name exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val --baseline-experiment-dir /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096 --query-embedding-root /workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --alpha-values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --seed 3407 --experiment-name source_probability_mixing_evaluation__xattn_long_e50_mix_val
```

## Preconditions

- The memory-only evaluation must already exist at `/workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val`.
- The baseline experiment must already exist at `/workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096`.
- The query embeddings must already exist at `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096/best.ckpt`
- Memory probabilities: `/workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val/val_probabilities.npy`
- Query embedding split: `/workspace/experiments/exp0019__source_cross_attention_embedding_export__xattn_long_e50_export_bs160/val`
- Validation rows: `11,219`
- FAISS note: `Not used directly in this stage; memory probabilities come from exp0006.`

## Sweep Summary

| alpha | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.702234 | 0.128844 | 0.011892 | 0.018889 |  |
| 0.1 | 0.745855 | 0.134188 | 0.032759 | 0.027401 |  |
| 0.2 | 0.750670 | 0.136133 | 0.064024 | 0.042811 |  |
| 0.3 | 0.752876 | 0.137647 | 0.095745 | 0.069018 |  |
| 0.4 | 0.754090 | 0.139007 | 0.127503 | 0.112225 |  |
| 0.5 | 0.754806 | 0.139928 | 0.159255 | 0.149178 |  |
| 0.6 | 0.755204 | 0.140374 | 0.191015 | 0.193256 |  |
| 0.7 | 0.755412 | 0.140943 | 0.222764 | 0.196422 |  |
| 0.8 | 0.755483 | 0.141482 | 0.254518 | 0.186476 | <- best |
| 0.9 | 0.755441 | 0.141102 | 0.286273 | 0.176445 |  |
| 1.0 | 0.755295 | 0.139532 | 0.318028 | 0.168275 |  |

## Best Configuration

- Best alpha: `0.8`
- Validation macro AUROC: `0.755483`
- Validation macro average precision: `0.141482`
- Validation macro ECE: `0.254518`
- Validation macro F1 @ 0.5: `0.186476`
- Diagnostic macro F1 @ tuned thresholds: `0.215076`

## Baseline Comparison

- Frozen baseline validation macro AUROC: `0.755295`
- Frozen baseline validation macro average precision: `0.139532`
- Best mixed minus baseline macro AUROC: `0.000188`
- Best mixed minus baseline macro average precision: `0.001951`
- Baseline reconstruction matches archived exp0004 forward metrics within 5e-4: `true`
- Baseline reconstruction max absolute metric delta: `0.000097361487`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `sweep_results.json`
- `best_config.json`
- `best_val_metrics.json`
- `val_mixed_probabilities.npy`
- `probability_mixing_selection.md`

## Output Sizes

- experiment_meta.json: `55.62K`
- sweep_results.json: `96.39K`
- best_config.json: `1.18K`
- best_val_metrics.json: `6.37K`
- val_mixed_probabilities.npy: `613.66K`
- probability_mixing_selection.md: `419B`
- Total output size: `773.63K`

## Final Artifact SHA-256

```text
c016aaba0e350cf014636b5eb9be0b583cfdca0b8e511b3b5050829b43dd8069  /workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val/experiment_meta.json
9c9661fa8f68c0035984822884adc94b3c6965f2abe744561c787e3d98728767  /workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val/sweep_results.json
01c4550eeac702799b706b7d07aa71634ec52f9e0ccd3777da73c88b97ca7d10  /workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val/best_config.json
cffa12e3386c48b7de09b3313d7a7fbf3d3b5d420cb9d7bf154894fa6063380e  /workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val/best_val_metrics.json
73f8b768d54cf005981a369be84855a4cc3d83691a4cf50911da52ad213d7e56  /workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val/val_mixed_probabilities.npy
740725f4917538eab507dadb00a0795b2ff20e343f4b0ace7c541f38acb153c1  /workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val/probability_mixing_selection.md
```

## Important Reproduction Notes

- All selection in `exp0007` is validation-only.
- `alpha=1.0` corresponds to the frozen baseline alone, and `alpha=0.0` corresponds to the selected memory-only probabilities from `exp0006`.
- `val_mixed_probabilities.npy` stores the best-config mixed probabilities in validation row order.
- Tied settings are resolved conservatively in favor of larger alpha.

## Agent Handoff Text

```text
Use /workspace/scripts/07_evaluate_probability_mixing.py and the report /workspace/experiments/exp0023__source_probability_mixing_evaluation__xattn_long_e50_mix_val/recreation_report.md to recreate the validation-only probability-mixing stage that combines /workspace/experiments/exp0020__source_baseline_training__xattn_long_e50_baseline_bs4096 with /workspace/experiments/exp0022__source_memory_only_evaluation__xattn_long_e50_memory_val. Reconstruct the frozen baseline validation probabilities, mix them with the exp0006 memory probabilities across alpha in [0.0, 1.0], and verify the saved best_config.json, best_val_metrics.json, and val_mixed_probabilities.npy artifacts.
```
