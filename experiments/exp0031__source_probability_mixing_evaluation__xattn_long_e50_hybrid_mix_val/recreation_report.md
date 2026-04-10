# Source Probability-Mixing Recreation Report

## Scope

This report documents how to recreate the validation-only probability-mixing experiment stored at:

`/workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val`

The producing script is:

`/workspace/scripts/07_evaluate_probability_mixing.py`

Script SHA-256:

`f643aca68d35d4c2a6ae4b3255865748538f1646247af95c8d75e4c04de16588`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val`
- Experiment id: `exp0031`
- Operation label: `source_probability_mixing_evaluation`
- Memory-evaluation root: `/workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val`
- Baseline experiment: `/workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096`
- Query embedding root: `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat`
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
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096 --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --alpha-values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --seed 3407 --experiment-name exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096 --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --alpha-values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --seed 3407 --experiment-name source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val
```

## Preconditions

- The memory-only evaluation must already exist at `/workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val`.
- The baseline experiment must already exist at `/workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096`.
- The query embeddings must already exist at `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096/best.ckpt`
- Memory probabilities: `/workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val/val_probabilities.npy`
- Query embedding split: `/workspace/experiments/exp0027__fused_embedding_generation__xattn_long_e50_hybrid_concat/val`
- Validation rows: `11,219`
- FAISS note: `Not used directly in this stage; memory probabilities come from exp0006.`

## Sweep Summary

| alpha | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.718101 | 0.143574 | 0.011127 | 0.027371 |  |
| 0.1 | 0.758920 | 0.148864 | 0.033827 | 0.037516 |  |
| 0.2 | 0.763535 | 0.151206 | 0.065668 | 0.054879 |  |
| 0.3 | 0.765453 | 0.153231 | 0.097932 | 0.079373 |  |
| 0.4 | 0.766365 | 0.154779 | 0.130233 | 0.116427 |  |
| 0.5 | 0.766795 | 0.155553 | 0.162522 | 0.170536 |  |
| 0.6 | 0.766955 | 0.156347 | 0.194820 | 0.204543 | <- best |
| 0.7 | 0.766932 | 0.156777 | 0.227117 | 0.204378 |  |
| 0.8 | 0.766762 | 0.156773 | 0.259421 | 0.194773 |  |
| 0.9 | 0.766475 | 0.156307 | 0.291731 | 0.183561 |  |
| 1.0 | 0.766080 | 0.151865 | 0.324018 | 0.174507 |  |

## Best Configuration

- Best alpha: `0.6`
- Validation macro AUROC: `0.766955`
- Validation macro average precision: `0.156347`
- Validation macro ECE: `0.194820`
- Validation macro F1 @ 0.5: `0.204543`
- Diagnostic macro F1 @ tuned thresholds: `0.227688`

## Baseline Comparison

- Frozen baseline validation macro AUROC: `0.766080`
- Frozen baseline validation macro average precision: `0.151865`
- Best mixed minus baseline macro AUROC: `0.000875`
- Best mixed minus baseline macro average precision: `0.004482`
- Baseline reconstruction matches archived exp0004 forward metrics within 5e-4: `true`
- Baseline reconstruction max absolute metric delta: `0.000097508304`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `sweep_results.json`
- `best_config.json`
- `best_val_metrics.json`
- `val_mixed_probabilities.npy`
- `probability_mixing_selection.md`

## Output Sizes

- experiment_meta.json: `59.13K`
- sweep_results.json: `96.30K`
- best_config.json: `1.18K`
- best_val_metrics.json: `6.32K`
- val_mixed_probabilities.npy: `613.66K`
- probability_mixing_selection.md: `419B`
- Total output size: `777.00K`

## Final Artifact SHA-256

```text
76a7e91444d8175d365d1301e57a2b461ef14e711730b310802fa3a22ac3e809  /workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val/experiment_meta.json
f87e242457e50414b44becfe04ef04ea7b1724a2e1332a7248dd7a1680cc32f0  /workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val/sweep_results.json
e79dc34327ea7384f01e9d811f99f58fffb7a322011be70a9926f43e337d7e29  /workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val/best_config.json
4036f302e29c2ab2ba5b0c9064f0f9e336d2779a185196589c255648807531cc  /workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val/best_val_metrics.json
245e0888486a8609480b7bfc5ccef16c43bd8ac9c32b2a868f326578748e39c3  /workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val/val_mixed_probabilities.npy
b7f1211ae04c03851e5ec629a68eb969e1ccfd1c23808cf10ff34e4c0a9af14f  /workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val/probability_mixing_selection.md
```

## Important Reproduction Notes

- All selection in `exp0007` is validation-only.
- `alpha=1.0` corresponds to the frozen baseline alone, and `alpha=0.0` corresponds to the selected memory-only probabilities from `exp0006`.
- `val_mixed_probabilities.npy` stores the best-config mixed probabilities in validation row order.
- Tied settings are resolved conservatively in favor of larger alpha.

## Agent Handoff Text

```text
Use /workspace/scripts/07_evaluate_probability_mixing.py and the report /workspace/experiments/exp0031__source_probability_mixing_evaluation__xattn_long_e50_hybrid_mix_val/recreation_report.md to recreate the validation-only probability-mixing stage that combines /workspace/experiments/exp0028__source_baseline_training__xattn_long_e50_hybrid_baseline_bs4096 with /workspace/experiments/exp0030__source_memory_only_evaluation__xattn_long_e50_hybrid_memory_val. Reconstruct the frozen baseline validation probabilities, mix them with the exp0006 memory probabilities across alpha in [0.0, 1.0], and verify the saved best_config.json, best_val_metrics.json, and val_mixed_probabilities.npy artifacts.
```
