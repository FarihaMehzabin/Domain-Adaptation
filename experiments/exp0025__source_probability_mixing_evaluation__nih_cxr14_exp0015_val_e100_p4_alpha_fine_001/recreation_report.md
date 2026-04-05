# Source Probability-Mixing Recreation Report

## Scope

This report documents how to recreate the validation-only probability-mixing experiment stored at:

`/workspace/experiments/exp0025__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4_alpha_fine_001`

The producing script is:

`/workspace/scripts/07_evaluate_probability_mixing.py`

Script SHA-256:

`bc7835587290462e3f45065734b7633c287649b59f64a94c06863c631941356b`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0025__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4_alpha_fine_001`
- Experiment id: `exp0025`
- Operation label: `source_probability_mixing_evaluation`
- Memory-evaluation root: `/workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4`
- Baseline experiment: `/workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`
- Query embedding root: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`
- Manifest: `/workspace/manifest_nih_cxr14_all14.csv`
- Evaluation split: `val`
- Validation rows: `11,219`
- Label count: `14`
- Label names: `atelectasis cardiomegaly consolidation edema pleural_effusion emphysema fibrosis hernia infiltration mass nodule pleural_thickening pneumonia pneumothorax`
- Selection metric: `macro_auroc`
- Alpha values: `0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0`

## Environment

- Python: `3.11.10`
- NumPy: `1.26.3`
- PyTorch: `2.4.1+cu124`
- Platform: `Linux-6.8.0-85-generic-x86_64-with-glibc2.35`

## Exact Recreation Command

If you want to recreate the same directory name in place, use this command:

```bash
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4 --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --alpha-values 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 --seed 3407 --experiment-name exp0025__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4_alpha_fine_001 --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4 --baseline-experiment-dir /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --alpha-values 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 --seed 3407 --experiment-name source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4_alpha_fine_001
```

## Preconditions

- The memory-only evaluation must already exist at `/workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4`.
- The baseline experiment must already exist at `/workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`.
- The query embeddings must already exist at `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4/best.ckpt`
- Memory probabilities: `/workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4/val_probabilities.npy`
- Query embedding split: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2/val`
- Validation rows: `11,219`
- FAISS note: `Not used directly in this stage; memory probabilities come from exp0009.`

## Sweep Summary

| alpha | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.682669 | 0.128678 | 0.009410 | 0.017704 |  |
| 0.01 | 0.730846 | 0.135242 | 0.008775 | 0.019809 |  |
| 0.02 | 0.730862 | 0.135356 | 0.009371 | 0.019809 |  |
| 0.03 | 0.731581 | 0.135516 | 0.011305 | 0.019809 |  |
| 0.04 | 0.734402 | 0.136279 | 0.013899 | 0.020316 |  |
| 0.05 | 0.738087 | 0.137343 | 0.016378 | 0.020856 |  |
| 0.06 | 0.741618 | 0.138589 | 0.019036 | 0.021818 |  |
| 0.07 | 0.744792 | 0.139805 | 0.022118 | 0.022178 |  |
| 0.08 | 0.747577 | 0.140958 | 0.025148 | 0.024107 |  |
| 0.09 | 0.750036 | 0.142029 | 0.028277 | 0.025161 |  |
| 0.1 | 0.752218 | 0.143036 | 0.031488 | 0.025632 |  |
| 0.11 | 0.754134 | 0.143998 | 0.034580 | 0.026975 |  |
| 0.12 | 0.755862 | 0.144946 | 0.037826 | 0.029651 |  |
| 0.13 | 0.757423 | 0.145845 | 0.041048 | 0.030695 |  |
| 0.14 | 0.758812 | 0.146704 | 0.044292 | 0.031613 |  |
| 0.15 | 0.760062 | 0.147530 | 0.047502 | 0.034037 |  |
| 0.16 | 0.761190 | 0.148398 | 0.050800 | 0.035165 |  |
| 0.17 | 0.762230 | 0.149173 | 0.054060 | 0.036392 |  |
| 0.18 | 0.763165 | 0.149917 | 0.057327 | 0.038550 |  |
| 0.19 | 0.764038 | 0.150742 | 0.060599 | 0.039496 |  |
| 0.2 | 0.764820 | 0.151410 | 0.063887 | 0.041084 |  |
| 0.21 | 0.765556 | 0.152012 | 0.067146 | 0.043544 |  |
| 0.22 | 0.766224 | 0.152854 | 0.070422 | 0.045144 |  |
| 0.23 | 0.766842 | 0.153461 | 0.073641 | 0.047903 |  |
| 0.24 | 0.767429 | 0.153938 | 0.076896 | 0.049971 |  |
| 0.25 | 0.767972 | 0.154548 | 0.080170 | 0.052767 |  |
| 0.26 | 0.768461 | 0.155128 | 0.083374 | 0.056047 |  |
| 0.27 | 0.768916 | 0.155895 | 0.086671 | 0.056921 |  |
| 0.28 | 0.769342 | 0.156202 | 0.089912 | 0.058047 |  |
| 0.29 | 0.769732 | 0.156623 | 0.093162 | 0.059815 |  |
| 0.3 | 0.770096 | 0.157421 | 0.096443 | 0.061077 |  |
| 0.31 | 0.770440 | 0.157863 | 0.099717 | 0.063002 |  |
| 0.32 | 0.770764 | 0.158251 | 0.102977 | 0.066111 |  |
| 0.33 | 0.771062 | 0.158593 | 0.106253 | 0.068307 |  |
| 0.34 | 0.771339 | 0.158810 | 0.109518 | 0.070004 |  |
| 0.35 | 0.771598 | 0.159110 | 0.112788 | 0.073895 |  |
| 0.36 | 0.771843 | 0.159407 | 0.116063 | 0.075537 |  |
| 0.37 | 0.772073 | 0.159643 | 0.119343 | 0.079661 |  |
| 0.38 | 0.772283 | 0.159828 | 0.122611 | 0.083154 |  |
| 0.39 | 0.772485 | 0.160299 | 0.125912 | 0.086732 |  |
| 0.4 | 0.772664 | 0.160512 | 0.129148 | 0.091241 |  |
| 0.41 | 0.772845 | 0.160874 | 0.132411 | 0.095259 |  |
| 0.42 | 0.773008 | 0.160445 | 0.135665 | 0.100819 |  |
| 0.43 | 0.773165 | 0.160647 | 0.138935 | 0.106771 |  |
| 0.44 | 0.773308 | 0.160812 | 0.142215 | 0.114414 |  |
| 0.45 | 0.773431 | 0.160964 | 0.145468 | 0.121637 |  |
| 0.46 | 0.773561 | 0.161145 | 0.148748 | 0.127441 |  |
| 0.47 | 0.773677 | 0.161308 | 0.152031 | 0.132772 |  |
| 0.48 | 0.773783 | 0.161424 | 0.155296 | 0.140568 |  |
| 0.49 | 0.773875 | 0.161563 | 0.158570 | 0.149238 |  |
| 0.5 | 0.773982 | 0.161734 | 0.161834 | 0.154490 |  |
| 0.51 | 0.774067 | 0.161933 | 0.165106 | 0.160789 |  |
| 0.52 | 0.774157 | 0.162084 | 0.168379 | 0.171941 |  |
| 0.53 | 0.774233 | 0.162189 | 0.171645 | 0.180817 |  |
| 0.54 | 0.774301 | 0.162273 | 0.174912 | 0.185605 |  |
| 0.55 | 0.774364 | 0.162555 | 0.178190 | 0.191230 |  |
| 0.56 | 0.774417 | 0.162607 | 0.181461 | 0.197703 |  |
| 0.57 | 0.774467 | 0.162670 | 0.184732 | 0.204997 |  |
| 0.58 | 0.774520 | 0.162637 | 0.187992 | 0.204781 |  |
| 0.59 | 0.774565 | 0.162647 | 0.191270 | 0.208167 |  |
| 0.6 | 0.774614 | 0.162776 | 0.194548 | 0.210068 |  |
| 0.61 | 0.774657 | 0.162846 | 0.197801 | 0.209980 |  |
| 0.62 | 0.774687 | 0.162874 | 0.201059 | 0.209979 |  |
| 0.63 | 0.774717 | 0.163034 | 0.204329 | 0.212024 |  |
| 0.64 | 0.774740 | 0.163043 | 0.207606 | 0.214081 |  |
| 0.65 | 0.774768 | 0.163099 | 0.210876 | 0.214382 |  |
| 0.66 | 0.774791 | 0.163206 | 0.214146 | 0.213860 |  |
| 0.67 | 0.774809 | 0.163087 | 0.217410 | 0.212213 |  |
| 0.68 | 0.774823 | 0.163105 | 0.220687 | 0.211642 |  |
| 0.69 | 0.774831 | 0.163116 | 0.223961 | 0.211795 |  |
| 0.7 | 0.774842 | 0.163146 | 0.227235 | 0.211561 |  |
| 0.71 | 0.774849 | 0.163114 | 0.230505 | 0.209924 |  |
| 0.72 | 0.774856 | 0.163099 | 0.233772 | 0.209398 |  |
| 0.73 | 0.774863 | 0.163059 | 0.237033 | 0.208152 |  |
| 0.74 | 0.774865 | 0.162973 | 0.240306 | 0.208013 | <- best |
| 0.75 | 0.774864 | 0.163048 | 0.243573 | 0.207549 |  |
| 0.76 | 0.774859 | 0.162866 | 0.246843 | 0.206681 |  |
| 0.77 | 0.774853 | 0.162819 | 0.250116 | 0.205813 |  |
| 0.78 | 0.774854 | 0.162908 | 0.253388 | 0.205540 |  |
| 0.79 | 0.774848 | 0.162676 | 0.256661 | 0.204552 |  |
| 0.8 | 0.774841 | 0.162559 | 0.259937 | 0.203592 |  |
| 0.81 | 0.774832 | 0.162501 | 0.263215 | 0.202090 |  |
| 0.82 | 0.774825 | 0.162353 | 0.266484 | 0.200475 |  |
| 0.83 | 0.774811 | 0.162262 | 0.269759 | 0.199948 |  |
| 0.84 | 0.774801 | 0.162134 | 0.273015 | 0.199102 |  |
| 0.85 | 0.774783 | 0.162050 | 0.276288 | 0.198210 |  |
| 0.86 | 0.774770 | 0.161934 | 0.279560 | 0.197324 |  |
| 0.87 | 0.774751 | 0.161786 | 0.282834 | 0.195761 |  |
| 0.88 | 0.774736 | 0.161852 | 0.286111 | 0.194810 |  |
| 0.89 | 0.774717 | 0.161713 | 0.289383 | 0.193208 |  |
| 0.9 | 0.774697 | 0.161645 | 0.292654 | 0.192027 |  |
| 0.91 | 0.774680 | 0.161500 | 0.295924 | 0.191123 |  |
| 0.92 | 0.774656 | 0.161497 | 0.299196 | 0.189708 |  |
| 0.93 | 0.774636 | 0.161343 | 0.302463 | 0.188905 |  |
| 0.94 | 0.774611 | 0.161043 | 0.305725 | 0.188085 |  |
| 0.95 | 0.774581 | 0.160538 | 0.308995 | 0.187451 |  |
| 0.96 | 0.774558 | 0.160341 | 0.312268 | 0.185996 |  |
| 0.97 | 0.774534 | 0.160098 | 0.315538 | 0.185338 |  |
| 0.98 | 0.774504 | 0.159890 | 0.318810 | 0.183723 |  |
| 0.99 | 0.774477 | 0.159541 | 0.322082 | 0.182014 |  |
| 1.0 | 0.774450 | 0.159498 | 0.325353 | 0.181316 |  |

## Best Configuration

- Best alpha: `0.7`
- Validation macro AUROC: `0.774865`
- Validation macro average precision: `0.162973`
- Validation macro ECE: `0.240306`
- Validation macro F1 @ 0.5: `0.208013`
- Diagnostic macro F1 @ tuned thresholds: `0.234481`

## Baseline Comparison

- Frozen baseline validation macro AUROC: `0.774450`
- Frozen baseline validation macro average precision: `0.159498`
- Best mixed minus baseline macro AUROC: `0.000415`
- Best mixed minus baseline macro average precision: `0.003475`
- Baseline reconstruction matches archived exp0006 forward metrics within 5e-4: `true`
- Baseline reconstruction max absolute metric delta: `0.000316035093`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `sweep_results.json`
- `best_config.json`
- `best_val_metrics.json`
- `val_mixed_probabilities.npy`
- `probability_mixing_selection.md`

## Output Sizes

- experiment_meta.json: `60.77K`
- sweep_results.json: `764.20K`
- best_config.json: `1.18K`
- best_val_metrics.json: `6.36K`
- val_mixed_probabilities.npy: `613.66K`
- probability_mixing_selection.md: `420B`
- Total output size: `1.41M`

## Final Artifact SHA-256

```text
5a2e53735dd04fd79f4cc6990ab06ee41bad7252dc460bf294ff3687b85019c5  /workspace/experiments/exp0025__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4_alpha_fine_001/experiment_meta.json
8939c7deb6d1920e7a4680279f0a97325c63baa8f992b5bd6a3248b1c71c4c8d  /workspace/experiments/exp0025__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4_alpha_fine_001/sweep_results.json
5d479c68d398f218ee40011ec211f78bdadab4be49b1c1bc3b1726b16afdfa01  /workspace/experiments/exp0025__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4_alpha_fine_001/best_config.json
893dcd08905f1b9750ea6d47b8fda752b7ab2dccfd29efdcbf7a0238e0b205bd  /workspace/experiments/exp0025__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4_alpha_fine_001/best_val_metrics.json
f9802a1e48c96ff675a84b288033b1e4f00c4a7595c1c037362a27cf83f00c2f  /workspace/experiments/exp0025__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4_alpha_fine_001/val_mixed_probabilities.npy
ffaa02a210ba16ddd5fa9279aec550f0e09e66b8f66eab38f14b418d5cd33de8  /workspace/experiments/exp0025__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4_alpha_fine_001/probability_mixing_selection.md
```

## Important Reproduction Notes

- All selection in `exp0010` is validation-only.
- `alpha=1.0` corresponds to the frozen baseline alone, and `alpha=0.0` corresponds to the selected memory-only probabilities from `exp0009`.
- `val_mixed_probabilities.npy` stores the best-config mixed probabilities in validation row order.
- Tied settings are resolved conservatively in favor of larger alpha.

## Agent Handoff Text

```text
Use /workspace/scripts/07_evaluate_probability_mixing.py and the report /workspace/experiments/exp0025__source_probability_mixing_evaluation__nih_cxr14_exp0015_val_e100_p4_alpha_fine_001/recreation_report.md to recreate the validation-only probability-mixing stage that combines /workspace/experiments/exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4 with /workspace/experiments/exp0015__source_memory_only_evaluation__nih_cxr14_exp0014_val_e100_p4. Reconstruct the frozen baseline validation probabilities, mix them with the exp0009 memory probabilities across alpha in [0.0, 1.0], and verify the saved best_config.json, best_val_metrics.json, and val_mixed_probabilities.npy artifacts.
```
