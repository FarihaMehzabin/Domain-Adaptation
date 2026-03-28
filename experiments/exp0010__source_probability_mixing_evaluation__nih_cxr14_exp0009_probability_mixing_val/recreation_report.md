# Source Probability-Mixing Recreation Report

## Scope

This report documents how to recreate the validation-only probability-mixing experiment stored at:

`/workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val`

The producing script is:

`/workspace/scripts/07_evaluate_probability_mixing.py`

Script SHA-256:

`f250dd26bb4f93f275809a90d9008782146e18f5277fc6b55a726d34f3ba5280`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val`
- Experiment id: `exp0010`
- Operation label: `source_probability_mixing_evaluation`
- Memory-evaluation root: `/workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val`
- Baseline experiment: `/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear`
- Query embedding root: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Evaluation split: `val`
- Validation rows: `11,219`
- Label count: `14`
- Label names: `atelectasis cardiomegaly consolidation edema pleural_effusion emphysema fibrosis hernia infiltration mass nodule pleural_thickening pneumonia pneumothorax`
- Selection metric: `macro_auroc`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- PyTorch: `2.4.1+cu124`
- Platform: `Linux-5.15.0-102-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val --baseline-experiment-dir /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --seed 3407 --experiment-name exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val --baseline-experiment-dir /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --seed 3407 --experiment-name source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val
```

## Preconditions

- The memory-only evaluation must already exist at `/workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val`.
- The baseline experiment must already exist at `/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear`.
- The query embeddings must already exist at `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear/best.ckpt`
- Memory probabilities: `/workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val/val_probabilities.npy`
- Query embedding split: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val`
- Validation rows: `11,219`
- FAISS note: `Not used directly in this stage; memory probabilities come from exp0009.`

## Sweep Summary

| alpha | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.682669 | 0.128676 | 0.009410 | 0.017704 |  |
| 0.1 | 0.742853 | 0.140025 | 0.033630 | 0.025260 |  |
| 0.2 | 0.754620 | 0.146865 | 0.068209 | 0.040358 |  |
| 0.3 | 0.759668 | 0.152161 | 0.103035 | 0.059053 |  |
| 0.4 | 0.762187 | 0.155088 | 0.137923 | 0.084636 |  |
| 0.5 | 0.763455 | 0.156332 | 0.172772 | 0.138347 |  |
| 0.6 | 0.764058 | 0.156689 | 0.207665 | 0.194591 |  |
| 0.7 | 0.764262 | 0.156619 | 0.242566 | 0.204701 | <- best |
| 0.8 | 0.764220 | 0.155943 | 0.277462 | 0.197358 |  |
| 0.9 | 0.764025 | 0.154383 | 0.312346 | 0.187389 |  |
| 1.0 | 0.763731 | 0.151466 | 0.347227 | 0.175977 |  |

## Best Configuration

- Best alpha: `0.7`
- Validation macro AUROC: `0.764262`
- Validation macro average precision: `0.156619`
- Validation macro ECE: `0.242566`
- Validation macro F1 @ 0.5: `0.204701`
- Diagnostic macro F1 @ tuned thresholds: `0.225547`

## Baseline Comparison

- Frozen baseline validation macro AUROC: `0.763731`
- Frozen baseline validation macro average precision: `0.151466`
- Best mixed minus baseline macro AUROC: `0.000531`
- Best mixed minus baseline macro average precision: `0.005153`
- Baseline reconstruction matches archived exp0006 forward metrics within 5e-4: `true`
- Baseline reconstruction max absolute metric delta: `0.000309800002`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `sweep_results.json`
- `best_config.json`
- `best_val_metrics.json`
- `val_mixed_probabilities.npy`
- `probability_mixing_selection.md`

## Output Sizes

- experiment_meta.json: `57.09K`
- recreation_report.md: `8.33K`
- sweep_results.json: `96.09K`
- best_config.json: `1.17K`
- best_val_metrics.json: `6.35K`
- val_mixed_probabilities.npy: `613.66K`
- probability_mixing_selection.md: `419B`
- Total output size: `783.11K`

## Final Artifact SHA-256

```text
dee84238b6aa781085c2211685ffab469a70a3dab0d589351262986cc104f879  /workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val/experiment_meta.json
e0cd7f0d39ec3af1c55db5036a1b5eceaa400cf7fa350bf5b691eb2f9cfaaf1a  /workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val/recreation_report.md
e68bb409923a5b1a71eb2360bfe2527bdc758084169bc3ad67bd5d77d602eeef  /workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val/sweep_results.json
6eb05390030de391f276a10ac89adce6103388e09eb32afb6542fd56af34e598  /workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val/best_config.json
1d76c0ecbaef8ceddd63cff05d1c9ef29ae7457e91b00b2d7f741ef64b3f3f01  /workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val/best_val_metrics.json
3b173a41fdf1dc44183b93922a2e4bcb8c8d75e873396f7958ee6db04ba9b292  /workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val/val_mixed_probabilities.npy
748c60a34261dab91fcc056aba24a76733eb2706700df23f97f69352870f3527  /workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val/probability_mixing_selection.md
```

## Important Reproduction Notes

- All selection in `exp0010` is validation-only.
- `alpha=1.0` corresponds to the frozen baseline alone, and `alpha=0.0` corresponds to the selected memory-only probabilities from `exp0009`.
- `val_mixed_probabilities.npy` stores the best-config mixed probabilities in validation row order.
- Tied settings are resolved conservatively in favor of larger alpha.

## Agent Handoff Text

```text
Use /workspace/scripts/07_evaluate_probability_mixing.py and the report /workspace/experiments/exp0010__source_probability_mixing_evaluation__nih_cxr14_exp0009_probability_mixing_val/recreation_report.md to recreate the validation-only probability-mixing stage that combines /workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear with /workspace/experiments/exp0009__source_memory_only_evaluation__nih_cxr14_exp0008_val. Reconstruct the frozen baseline validation probabilities, mix them with the exp0009 memory probabilities across alpha in [0.0, 1.0], and verify the saved best_config.json, best_val_metrics.json, and val_mixed_probabilities.npy artifacts.
```
