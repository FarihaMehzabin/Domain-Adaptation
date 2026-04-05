# Source Probability-Mixing Recreation Report

## Scope

This report documents how to recreate the validation-only probability-mixing experiment stored at:

`/workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001`

The producing script is:

`/workspace/scripts/07_evaluate_probability_mixing.py`

Script SHA-256:

`bc7835587290462e3f45065734b7633c287649b59f64a94c06863c631941356b`

## Final Experiment Identity

- Experiment directory: `/workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001`
- Experiment id: `exp0041`
- Operation label: `source_probability_mixing_evaluation`
- Memory-evaluation root: `/workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4`
- Baseline experiment: `/workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4`
- Query embedding root: `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050`
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
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4 --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --alpha-values 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 --seed 3407 --experiment-name exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001 --overwrite
```

If you want a fresh numbered run instead of overwriting the existing directory, use:

```bash
python /workspace/scripts/07_evaluate_probability_mixing.py --memory-eval-root /workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4 --baseline-experiment-dir /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4 --query-embedding-root /workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050 --manifest-csv /workspace/manifest_nih_cxr14_all14.csv --split val --batch-size 2048 --alpha-values 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 --seed 3407 --experiment-name source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001
```

## Preconditions

- The memory-only evaluation must already exist at `/workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4`.
- The baseline experiment must already exist at `/workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4`.
- The query embeddings must already exist at `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050/val`.
- The manifest must be present at `/workspace/manifest_nih_cxr14_all14.csv`.
- The required Python packages must be importable: `numpy`, `torch`.

## Input Summary

- Baseline checkpoint: `/workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4/best.ckpt`
- Memory probabilities: `/workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4/val_probabilities.npy`
- Query embedding split: `/workspace/experiments/exp0027__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2_txtw050/val`
- Validation rows: `11,219`
- FAISS note: `Not used directly in this stage; memory probabilities come from exp0009.`

## Sweep Summary

| alpha | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Best |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.683310 | 0.125202 | 0.009441 | 0.009712 |  |
| 0.01 | 0.731362 | 0.132772 | 0.008459 | 0.011520 |  |
| 0.02 | 0.731395 | 0.132988 | 0.008994 | 0.011520 |  |
| 0.03 | 0.732133 | 0.133352 | 0.010750 | 0.011520 |  |
| 0.04 | 0.735172 | 0.134305 | 0.012726 | 0.012075 |  |
| 0.05 | 0.739085 | 0.135636 | 0.014856 | 0.013601 |  |
| 0.06 | 0.742764 | 0.137059 | 0.017684 | 0.014440 |  |
| 0.07 | 0.746053 | 0.138403 | 0.020691 | 0.014971 |  |
| 0.08 | 0.748954 | 0.139744 | 0.023674 | 0.016759 |  |
| 0.09 | 0.751477 | 0.140995 | 0.026719 | 0.017072 |  |
| 0.1 | 0.753704 | 0.142172 | 0.029885 | 0.018025 |  |
| 0.11 | 0.755682 | 0.143262 | 0.033012 | 0.018711 |  |
| 0.12 | 0.757439 | 0.144290 | 0.036161 | 0.019958 |  |
| 0.13 | 0.759012 | 0.145223 | 0.039337 | 0.021439 |  |
| 0.14 | 0.760430 | 0.146150 | 0.042623 | 0.022456 |  |
| 0.15 | 0.761693 | 0.146961 | 0.045843 | 0.024807 |  |
| 0.16 | 0.762838 | 0.147811 | 0.049150 | 0.025564 |  |
| 0.17 | 0.763868 | 0.148499 | 0.052281 | 0.026493 |  |
| 0.18 | 0.764804 | 0.149016 | 0.055521 | 0.027691 |  |
| 0.19 | 0.765663 | 0.149657 | 0.058739 | 0.029192 |  |
| 0.2 | 0.766435 | 0.150255 | 0.061923 | 0.030826 |  |
| 0.21 | 0.767161 | 0.150906 | 0.065144 | 0.033724 |  |
| 0.22 | 0.767816 | 0.151560 | 0.068337 | 0.034905 |  |
| 0.23 | 0.768414 | 0.152286 | 0.071627 | 0.038055 |  |
| 0.24 | 0.768973 | 0.152784 | 0.074902 | 0.038923 |  |
| 0.25 | 0.769490 | 0.153632 | 0.078090 | 0.040833 |  |
| 0.26 | 0.769951 | 0.154050 | 0.081341 | 0.042522 |  |
| 0.27 | 0.770381 | 0.154360 | 0.084565 | 0.044305 |  |
| 0.28 | 0.770796 | 0.154728 | 0.087832 | 0.046000 |  |
| 0.29 | 0.771163 | 0.155079 | 0.091076 | 0.047064 |  |
| 0.3 | 0.771508 | 0.155477 | 0.094317 | 0.048387 |  |
| 0.31 | 0.771831 | 0.155813 | 0.097523 | 0.051289 |  |
| 0.32 | 0.772126 | 0.156067 | 0.100763 | 0.053083 |  |
| 0.33 | 0.772410 | 0.156414 | 0.104045 | 0.055250 |  |
| 0.34 | 0.772675 | 0.156679 | 0.107291 | 0.056957 |  |
| 0.35 | 0.772926 | 0.156926 | 0.110566 | 0.060154 |  |
| 0.36 | 0.773152 | 0.157115 | 0.113825 | 0.063396 |  |
| 0.37 | 0.773363 | 0.157376 | 0.117057 | 0.066722 |  |
| 0.38 | 0.773557 | 0.157577 | 0.120288 | 0.071154 |  |
| 0.39 | 0.773742 | 0.157814 | 0.123530 | 0.075072 |  |
| 0.4 | 0.773909 | 0.158013 | 0.126787 | 0.080193 |  |
| 0.41 | 0.774077 | 0.158174 | 0.130049 | 0.085696 |  |
| 0.42 | 0.774231 | 0.158405 | 0.133307 | 0.091603 |  |
| 0.43 | 0.774367 | 0.158521 | 0.136560 | 0.099635 |  |
| 0.44 | 0.774503 | 0.158662 | 0.139831 | 0.102353 |  |
| 0.45 | 0.774623 | 0.159183 | 0.143086 | 0.109030 |  |
| 0.46 | 0.774739 | 0.159447 | 0.146329 | 0.116375 |  |
| 0.47 | 0.774848 | 0.159616 | 0.149586 | 0.122335 |  |
| 0.48 | 0.774950 | 0.159748 | 0.152841 | 0.131242 |  |
| 0.49 | 0.775043 | 0.159874 | 0.156104 | 0.141589 |  |
| 0.5 | 0.775131 | 0.160029 | 0.159355 | 0.151295 |  |
| 0.51 | 0.775206 | 0.160132 | 0.162604 | 0.158767 |  |
| 0.52 | 0.775277 | 0.160184 | 0.165861 | 0.169272 |  |
| 0.53 | 0.775344 | 0.160281 | 0.169122 | 0.175919 |  |
| 0.54 | 0.775414 | 0.160399 | 0.172376 | 0.183685 |  |
| 0.55 | 0.775477 | 0.160459 | 0.175641 | 0.192464 |  |
| 0.56 | 0.775531 | 0.160547 | 0.178894 | 0.195134 |  |
| 0.57 | 0.775581 | 0.160603 | 0.182151 | 0.198767 |  |
| 0.58 | 0.775620 | 0.160699 | 0.185408 | 0.201496 |  |
| 0.59 | 0.775666 | 0.160754 | 0.188667 | 0.205532 |  |
| 0.6 | 0.775700 | 0.160727 | 0.191923 | 0.207285 |  |
| 0.61 | 0.775734 | 0.160873 | 0.195179 | 0.209639 |  |
| 0.62 | 0.775762 | 0.160904 | 0.198444 | 0.210714 |  |
| 0.63 | 0.775791 | 0.160936 | 0.201703 | 0.212897 |  |
| 0.64 | 0.775817 | 0.161034 | 0.204955 | 0.212641 |  |
| 0.65 | 0.775843 | 0.161050 | 0.208206 | 0.212961 |  |
| 0.66 | 0.775867 | 0.161295 | 0.211463 | 0.213756 |  |
| 0.67 | 0.775882 | 0.161299 | 0.214718 | 0.213995 |  |
| 0.68 | 0.775892 | 0.161344 | 0.217973 | 0.213053 |  |
| 0.69 | 0.775901 | 0.161374 | 0.221232 | 0.213024 |  |
| 0.7 | 0.775915 | 0.161359 | 0.224489 | 0.212179 |  |
| 0.71 | 0.775920 | 0.161404 | 0.227750 | 0.211162 |  |
| 0.72 | 0.775929 | 0.161445 | 0.231007 | 0.211151 |  |
| 0.73 | 0.775933 | 0.161496 | 0.234264 | 0.210507 |  |
| 0.74 | 0.775935 | 0.161408 | 0.237523 | 0.209025 | <- best |
| 0.75 | 0.775934 | 0.161402 | 0.240782 | 0.208783 |  |
| 0.76 | 0.775931 | 0.161342 | 0.244036 | 0.207606 |  |
| 0.77 | 0.775926 | 0.161273 | 0.247292 | 0.206506 |  |
| 0.78 | 0.775916 | 0.161227 | 0.250552 | 0.205662 |  |
| 0.79 | 0.775910 | 0.161194 | 0.253804 | 0.204349 |  |
| 0.8 | 0.775901 | 0.161149 | 0.257062 | 0.203798 |  |
| 0.81 | 0.775893 | 0.161090 | 0.260319 | 0.202733 |  |
| 0.82 | 0.775883 | 0.161127 | 0.263576 | 0.202518 |  |
| 0.83 | 0.775873 | 0.161106 | 0.266836 | 0.201875 |  |
| 0.84 | 0.775858 | 0.161009 | 0.270093 | 0.200602 |  |
| 0.85 | 0.775844 | 0.161054 | 0.273350 | 0.199090 |  |
| 0.86 | 0.775830 | 0.160991 | 0.276610 | 0.197676 |  |
| 0.87 | 0.775812 | 0.161095 | 0.279872 | 0.195755 |  |
| 0.88 | 0.775790 | 0.161045 | 0.283130 | 0.194214 |  |
| 0.89 | 0.775770 | 0.160971 | 0.286393 | 0.193386 |  |
| 0.9 | 0.775756 | 0.160999 | 0.289652 | 0.192556 |  |
| 0.91 | 0.775737 | 0.160823 | 0.292911 | 0.191549 |  |
| 0.92 | 0.775720 | 0.160741 | 0.296161 | 0.190327 |  |
| 0.93 | 0.775697 | 0.160556 | 0.299421 | 0.189767 |  |
| 0.94 | 0.775678 | 0.160489 | 0.302678 | 0.188164 |  |
| 0.95 | 0.775654 | 0.160307 | 0.305937 | 0.186890 |  |
| 0.96 | 0.775631 | 0.160256 | 0.309197 | 0.185647 |  |
| 0.97 | 0.775607 | 0.160192 | 0.312456 | 0.184606 |  |
| 0.98 | 0.775579 | 0.160182 | 0.315713 | 0.183643 |  |
| 0.99 | 0.775557 | 0.160191 | 0.318973 | 0.182442 |  |
| 1.0 | 0.775533 | 0.160133 | 0.322230 | 0.181852 |  |

## Best Configuration

- Best alpha: `0.7`
- Validation macro AUROC: `0.775935`
- Validation macro average precision: `0.161408`
- Validation macro ECE: `0.237523`
- Validation macro F1 @ 0.5: `0.209025`
- Diagnostic macro F1 @ tuned thresholds: `0.232870`

## Baseline Comparison

- Frozen baseline validation macro AUROC: `0.775533`
- Frozen baseline validation macro average precision: `0.160133`
- Best mixed minus baseline macro AUROC: `0.000402`
- Best mixed minus baseline macro average precision: `0.001275`
- Baseline reconstruction matches archived exp0006 forward metrics within 5e-4: `true`
- Baseline reconstruction max absolute metric delta: `0.000430121227`

## Expected Outputs

- `experiment_meta.json`
- `recreation_report.md`
- `sweep_results.json`
- `best_config.json`
- `best_val_metrics.json`
- `val_mixed_probabilities.npy`
- `probability_mixing_selection.md`

## Output Sizes

- experiment_meta.json: `61.96K`
- sweep_results.json: `763.91K`
- best_config.json: `1.18K`
- best_val_metrics.json: `6.36K`
- val_mixed_probabilities.npy: `613.66K`
- probability_mixing_selection.md: `420B`
- Total output size: `1.41M`

## Final Artifact SHA-256

```text
4819574e510bcda2637dd2ba1cf07bf0f5778a4b1122e1f88d114879d52f3450  /workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001/experiment_meta.json
d2710c889f6fd2d23ca1fd53d52f80581ef9527ebbb22de786bf5ed940e0f5b9  /workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001/sweep_results.json
97eed193a75de70f9a50d4e989db1c1b5e50db650b69aa618ccc015a9a972231  /workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001/best_config.json
74f2db7294312a15147ea15e2b5f9e2175f9c0eb01d4cd2265040c0af339b4e9  /workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001/best_val_metrics.json
de343e7f01629b77507df0adcc494f0e4ce2affd585037b1ffa8b38cb2788214  /workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001/val_mixed_probabilities.npy
bd32f488c1a325f6c011613f0eff4c61f13a358f462b0bdd00cf8423e5f6a6ad  /workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001/probability_mixing_selection.md
```

## Important Reproduction Notes

- All selection in `exp0010` is validation-only.
- `alpha=1.0` corresponds to the frozen baseline alone, and `alpha=0.0` corresponds to the selected memory-only probabilities from `exp0009`.
- `val_mixed_probabilities.npy` stores the best-config mixed probabilities in validation row order.
- Tied settings are resolved conservatively in favor of larger alpha.

## Agent Handoff Text

```text
Use /workspace/scripts/07_evaluate_probability_mixing.py and the report /workspace/experiments/exp0041__source_probability_mixing_evaluation__nih_cxr14_txtw050_val_e100_p4_alpha_fine_001/recreation_report.md to recreate the validation-only probability-mixing stage that combines /workspace/experiments/exp0028__source_baseline_training__nih_cxr14_exp0001_exp0002_concat_l2_txtw050_fused_linear_e100_p4 with /workspace/experiments/exp0040__source_memory_only_evaluation__nih_cxr14_txtw050_val_e100_p4. Reconstruct the frozen baseline validation probabilities, mix them with the exp0009 memory probabilities across alpha in [0.0, 1.0], and verify the saved best_config.json, best_val_metrics.json, and val_mixed_probabilities.npy artifacts.
```
