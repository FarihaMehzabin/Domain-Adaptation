# Source Probability-Mixing Recreation Report

## Scope

This report documents how to recreate the validation-only probability-mixing experiment stored at:

`/workspace/experiments/exp0044__source_probability_mixing_evaluation__nih_cxr14_txtw075_val_e100_p4_alpha_fine_001`

The producing script is:

`/workspace/scripts/07_evaluate_probability_mixing.py`

Script SHA-256:

`bc7835587290462e3f45065734b7633c287649b59f64a94c06863c631941356b`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0044__source_probability_mixing_evaluation__nih_cxr14_txtw075_val_e100_p4_alpha_fine_001`
- Experiment id: `exp0044`
- Operation label: `source_probability_mixing_evaluation`
- Memory-evaluation root: `/workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4`
- Baseline experiment: `/workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4`
- Query embedding root: `/workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075`
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
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4 --baseline-experiment-dir /workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --alpha-values 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 --seed 3407 --experiment-name exp0044__source_probability_mixing_evaluation__nih_cxr14_txtw075_val_e100_p4_alpha_fine_001 --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4 --baseline-experiment-dir /workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --alpha-values 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 --seed 3407 --experiment-name source_probability_mixing_evaluation__nih_cxr14_txtw075_val_e100_p4_alpha_fine_001
```

## Preconditions

- The memory-only evaluation must already exist at `/workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4`.
- The baseline experiment must already exist at `/workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4`.
- The query embeddings must already exist at `/workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4/best.ckpt`
- Memory probabilities: `/workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4/val_probabilities.npy`
- Query embedding split: `/workspace/experiments/exp0029__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw075/val`
- Validation rows: `11,219`
- FAISS note: `Not used directly in this stage; memory probabilities come from exp0009.`

## Sweep Summary

| alpha | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.683231 | 0.127163 | 0.009080 | 0.013449 |  |
| 0.01 | 0.730715 | 0.133802 | 0.008167 | 0.015415 |  |
| 0.02 | 0.730726 | 0.133974 | 0.009225 | 0.015415 |  |
| 0.03 | 0.731412 | 0.134159 | 0.011419 | 0.015415 |  |
| 0.04 | 0.734197 | 0.134955 | 0.013314 | 0.016404 |  |
| 0.05 | 0.737940 | 0.136132 | 0.015671 | 0.017849 |  |
| 0.06 | 0.741537 | 0.137384 | 0.018394 | 0.018906 |  |
| 0.07 | 0.744781 | 0.138621 | 0.021429 | 0.019875 |  |
| 0.08 | 0.747628 | 0.139784 | 0.024410 | 0.022135 |  |
| 0.09 | 0.750136 | 0.140911 | 0.027573 | 0.022994 |  |
| 0.1 | 0.752354 | 0.141982 | 0.030757 | 0.024482 |  |
| 0.11 | 0.754317 | 0.142957 | 0.033812 | 0.025250 |  |
| 0.12 | 0.756087 | 0.143892 | 0.036932 | 0.026912 |  |
| 0.13 | 0.757667 | 0.144755 | 0.040186 | 0.028212 |  |
| 0.14 | 0.759093 | 0.145627 | 0.043464 | 0.029358 |  |
| 0.15 | 0.760378 | 0.146412 | 0.046788 | 0.031447 |  |
| 0.16 | 0.761541 | 0.147200 | 0.050010 | 0.032691 |  |
| 0.17 | 0.762590 | 0.147812 | 0.053210 | 0.034177 |  |
| 0.18 | 0.763558 | 0.148568 | 0.056405 | 0.036797 |  |
| 0.19 | 0.764429 | 0.149142 | 0.059602 | 0.037923 |  |
| 0.2 | 0.765233 | 0.149814 | 0.062825 | 0.039324 |  |
| 0.21 | 0.765975 | 0.150453 | 0.066024 | 0.041294 |  |
| 0.22 | 0.766651 | 0.151141 | 0.069341 | 0.042575 |  |
| 0.23 | 0.767272 | 0.151878 | 0.072619 | 0.044077 |  |
| 0.24 | 0.767852 | 0.152403 | 0.075854 | 0.047390 |  |
| 0.25 | 0.768397 | 0.152950 | 0.079142 | 0.048277 |  |
| 0.26 | 0.768907 | 0.153354 | 0.082374 | 0.049440 |  |
| 0.27 | 0.769370 | 0.153951 | 0.085631 | 0.050317 |  |
| 0.28 | 0.769794 | 0.154359 | 0.088871 | 0.052224 |  |
| 0.29 | 0.770187 | 0.154895 | 0.092132 | 0.054260 |  |
| 0.3 | 0.770565 | 0.155386 | 0.095406 | 0.056215 |  |
| 0.31 | 0.770924 | 0.155918 | 0.098614 | 0.059442 |  |
| 0.32 | 0.771248 | 0.156260 | 0.101866 | 0.062585 |  |
| 0.33 | 0.771548 | 0.156567 | 0.105120 | 0.063627 |  |
| 0.34 | 0.771827 | 0.156914 | 0.108388 | 0.066034 |  |
| 0.35 | 0.772085 | 0.157226 | 0.111651 | 0.069395 |  |
| 0.36 | 0.772336 | 0.157475 | 0.114912 | 0.072182 |  |
| 0.37 | 0.772571 | 0.157787 | 0.118135 | 0.074161 |  |
| 0.38 | 0.772787 | 0.158068 | 0.121392 | 0.078039 |  |
| 0.39 | 0.772987 | 0.158577 | 0.124657 | 0.080900 |  |
| 0.4 | 0.773172 | 0.158893 | 0.127929 | 0.085929 |  |
| 0.41 | 0.773347 | 0.159103 | 0.131198 | 0.091238 |  |
| 0.42 | 0.773512 | 0.159308 | 0.134461 | 0.096644 |  |
| 0.43 | 0.773670 | 0.159552 | 0.137721 | 0.104715 |  |
| 0.44 | 0.773814 | 0.159655 | 0.140975 | 0.110201 |  |
| 0.45 | 0.773948 | 0.159805 | 0.144227 | 0.118498 |  |
| 0.46 | 0.774072 | 0.159742 | 0.147491 | 0.125787 |  |
| 0.47 | 0.774189 | 0.159969 | 0.150754 | 0.130365 |  |
| 0.48 | 0.774305 | 0.160125 | 0.154005 | 0.137997 |  |
| 0.49 | 0.774405 | 0.160277 | 0.157250 | 0.145516 |  |
| 0.5 | 0.774503 | 0.160448 | 0.160520 | 0.151653 |  |
| 0.51 | 0.774591 | 0.160571 | 0.163777 | 0.161265 |  |
| 0.52 | 0.774677 | 0.160676 | 0.167040 | 0.171501 |  |
| 0.53 | 0.774745 | 0.160886 | 0.170298 | 0.181866 |  |
| 0.54 | 0.774817 | 0.160972 | 0.173548 | 0.187520 |  |
| 0.55 | 0.774881 | 0.161119 | 0.176814 | 0.190674 |  |
| 0.56 | 0.774943 | 0.161271 | 0.180081 | 0.197927 |  |
| 0.57 | 0.774999 | 0.161398 | 0.183340 | 0.203051 |  |
| 0.58 | 0.775050 | 0.161470 | 0.186603 | 0.204370 |  |
| 0.59 | 0.775105 | 0.161584 | 0.189863 | 0.207360 |  |
| 0.6 | 0.775147 | 0.161608 | 0.193114 | 0.208289 |  |
| 0.61 | 0.775183 | 0.161749 | 0.196372 | 0.210605 |  |
| 0.62 | 0.775221 | 0.161768 | 0.199632 | 0.211770 |  |
| 0.63 | 0.775254 | 0.161830 | 0.202894 | 0.212403 |  |
| 0.64 | 0.775281 | 0.161771 | 0.206164 | 0.213507 |  |
| 0.65 | 0.775310 | 0.161866 | 0.209422 | 0.213868 |  |
| 0.66 | 0.775332 | 0.161863 | 0.212684 | 0.214258 |  |
| 0.67 | 0.775352 | 0.161940 | 0.215932 | 0.213153 |  |
| 0.68 | 0.775371 | 0.161917 | 0.219183 | 0.213265 |  |
| 0.69 | 0.775392 | 0.161980 | 0.222443 | 0.211850 |  |
| 0.7 | 0.775408 | 0.162051 | 0.225709 | 0.211216 |  |
| 0.71 | 0.775416 | 0.161933 | 0.228963 | 0.210220 |  |
| 0.72 | 0.775427 | 0.161867 | 0.232228 | 0.209998 |  |
| 0.73 | 0.775429 | 0.161944 | 0.235490 | 0.209097 |  |
| 0.74 | 0.775435 | 0.161940 | 0.238747 | 0.208778 |  |
| 0.75 | 0.775438 | 0.161854 | 0.242007 | 0.208086 |  |
| 0.76 | 0.775438 | 0.161849 | 0.245266 | 0.207248 |  |
| 0.77 | 0.775439 | 0.161926 | 0.248529 | 0.206747 | <- best |
| 0.78 | 0.775437 | 0.161940 | 0.251788 | 0.205951 |  |
| 0.79 | 0.775431 | 0.161892 | 0.255051 | 0.204558 |  |
| 0.8 | 0.775425 | 0.161866 | 0.258318 | 0.203651 |  |
| 0.81 | 0.775419 | 0.161774 | 0.261580 | 0.202912 |  |
| 0.82 | 0.775412 | 0.161645 | 0.264833 | 0.201429 |  |
| 0.83 | 0.775406 | 0.161558 | 0.268093 | 0.200768 |  |
| 0.84 | 0.775396 | 0.161825 | 0.271351 | 0.200029 |  |
| 0.85 | 0.775384 | 0.161715 | 0.274611 | 0.198701 |  |
| 0.86 | 0.775367 | 0.161633 | 0.277879 | 0.197143 |  |
| 0.87 | 0.775357 | 0.161619 | 0.281140 | 0.195842 |  |
| 0.88 | 0.775344 | 0.161638 | 0.284404 | 0.194187 |  |
| 0.89 | 0.775330 | 0.161531 | 0.287667 | 0.193021 |  |
| 0.9 | 0.775321 | 0.161455 | 0.290929 | 0.192297 |  |
| 0.91 | 0.775305 | 0.161533 | 0.294188 | 0.191201 |  |
| 0.92 | 0.775283 | 0.161584 | 0.297453 | 0.190083 |  |
| 0.93 | 0.775265 | 0.161481 | 0.300705 | 0.188944 |  |
| 0.94 | 0.775246 | 0.161357 | 0.303965 | 0.188483 |  |
| 0.95 | 0.775224 | 0.161237 | 0.307225 | 0.187790 |  |
| 0.96 | 0.775202 | 0.160931 | 0.310489 | 0.186470 |  |
| 0.97 | 0.775181 | 0.160348 | 0.313749 | 0.184923 |  |
| 0.98 | 0.775159 | 0.160207 | 0.317009 | 0.184027 |  |
| 0.99 | 0.775137 | 0.160039 | 0.320275 | 0.182273 |  |
| 1.0 | 0.775113 | 0.159901 | 0.323539 | 0.181336 |  |

## Best Configuration

- Best alpha: `0.8`
- Validation macro AUROC: `0.775439`
- Validation macro average precision: `0.161926`
- Validation macro ECE: `0.248529`
- Validation macro F1 @ 0.5: `0.206747`
- Diagnostic macro F1 @ tuned thresholds: `0.234186`

## Baseline Comparison

- Frozen baseline validation macro AUROC: `0.775113`
- Frozen baseline validation macro average precision: `0.159901`
- Best mixed minus baseline macro AUROC: `0.000326`
- Best mixed minus baseline macro average precision: `0.002025`
- Baseline reconstruction matches archived exp0006 forward metrics within 5e-4: `true`
- Baseline reconstruction max absolute metric delta: `0.000447527411`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `sweep_results.json`
- `best_config.json`
- `best_val_metrics.json`
- `val_mixed_probabilities.npy`
- `probability_mixing_selection.md`

## Output Sizes

- experiment_meta.json: `62.05K`
- sweep_results.json: `763.78K`
- best_config.json: `1.18K`
- best_val_metrics.json: `6.36K`
- val_mixed_probabilities.npy: `613.66K`
- probability_mixing_selection.md: `420B`
- Total output size: `1.41M`

## Final Artifact SHA-256

```text
93b03ef311bfc20833d188e54ea01543db027216bac3b4703163c3d58ad97cc3  /workspace/experiments/exp0044__source_probability_mixing_evaluation__nih_cxr14_txtw075_val_e100_p4_alpha_fine_001/experiment_meta.json
f82399d9866fd1308d3341d9441ce763ec1ec3aba290a49b0cf73e66463d6e3c  /workspace/experiments/exp0044__source_probability_mixing_evaluation__nih_cxr14_txtw075_val_e100_p4_alpha_fine_001/sweep_results.json
4830f0439108c63b82036acbf5fd2831275d471c318120283cb0692dd8d9b5d9  /workspace/experiments/exp0044__source_probability_mixing_evaluation__nih_cxr14_txtw075_val_e100_p4_alpha_fine_001/best_config.json
2fc9d19fdfaa431431767e92e12316f9cfee1da876a89bf8e91f8e73e144f809  /workspace/experiments/exp0044__source_probability_mixing_evaluation__nih_cxr14_txtw075_val_e100_p4_alpha_fine_001/best_val_metrics.json
abfc94810263180132377e850cc697b3e23afeda11b27eff953f34d7c9b2805e  /workspace/experiments/exp0044__source_probability_mixing_evaluation__nih_cxr14_txtw075_val_e100_p4_alpha_fine_001/val_mixed_probabilities.npy
b9eed438f8a213d0c4d7fb3b92540f41a5153dd574de344225a166291d143cd8  /workspace/experiments/exp0044__source_probability_mixing_evaluation__nih_cxr14_txtw075_val_e100_p4_alpha_fine_001/probability_mixing_selection.md
```

## Important Reproduction Notes

- All selection in `exp0010` is validation-only.
- `alpha=1.0` corresponds to the frozen baseline alone, and `alpha=0.0` corresponds to the selected memory-only probabilities from `exp0009`.
- `val_mixed_probabilities.npy` stores the best-config mixed probabilities in validation row order.
- Tied settings are resolved conservatively in favor of larger alpha.

## Agent Handoff Text

```text
Use /workspace/scripts/07_evaluate_probability_mixing.py and the report /workspace/experiments/exp0044__source_probability_mixing_evaluation__nih_cxr14_txtw075_val_e100_p4_alpha_fine_001/recreation_report.md to recreate the validation-only probability-mixing stage that combines /workspace/experiments/exp0030__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw075_fused_linear_e100_p4 with /workspace/experiments/exp0043__source_memory_only_evaluation__nih_cxr14_txtw075_val_e100_p4. Reconstruct the frozen baseline validation probabilities, mix them with the exp0009 memory probabilities across alpha in [0.0, 1.0], and verify the saved best_config.json, best_val_metrics.json, and val_mixed_probabilities.npy artifacts.
```
