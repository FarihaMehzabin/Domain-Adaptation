# CXR Foundation Domain-Transfer RAG Summary

This summary covers the new retrieval-augmented rerun built on top of the existing CXR Foundation embedding export in `exp0014` and the linear probe in `exp0015`.

## Scope

- No chest X-ray report features were used.
- Existing embeddings were reused; no raw image redownload was required for the rerun.
- Retrieval memory was built from `d0_nih/train`.
- Hyperparameters were selected on `d0_nih/val`.
- Frozen evaluation was run on:
  - `d0_nih/test`
  - `d1_chexpert/val` as `d1_transfer`
  - `d2_mimic/test` as `d2_transfer`

## Selected Retrieval Settings

- Memory selection dir:
  - `exp0029__domain_transfer_source_memory_selection__cxr_foundation_general_avg_pilot5h_d0_val`
- Best memory config:
  - `k = 50`
  - `tau = 5.0`
- Mixing selection dir:
  - `exp0031__domain_transfer_probability_mixing_selection__cxr_foundation_general_avg_pilot5h_d0_val`
- Best mixing config:
  - `alpha = 1.0`

## Valid Result Directories

- Memory build:
  - `exp0028__domain_transfer_source_retrieval_memory_building__domain_transfer_source_retrieval_memory__cxr_foundation_general_avg_pilot5h_d0_train`
- Memory-only evaluation:
  - `exp0034__domain_transfer_source_memory_target_evaluation__cxr_foundation_general_avg_pilot5h_d0_test`
  - `exp0035__domain_transfer_source_memory_target_evaluation__cxr_foundation_general_avg_pilot5h_d1_transfer`
  - `exp0036__domain_transfer_source_memory_target_evaluation__cxr_foundation_general_avg_pilot5h_d2_transfer`
- Mixed evaluation:
  - `exp0037__domain_transfer_probability_mixing_target_evaluation__cxr_foundation_general_avg_pilot5h_d0_test`
  - `exp0031__domain_transfer_probability_mixing_target_evaluation__domain_transfer_probability_mixing_selection__cxr_foundation_general_avg_pilot5h_d0_val__d1_transfer`
  - `exp0031__domain_transfer_probability_mixing_target_evaluation__domain_transfer_probability_mixing_selection__cxr_foundation_general_avg_pilot5h_d0_val__d2_transfer`

The empty retry directories `exp0030__...`, `exp0037__...d1_transfer`, and `exp0038__...d2_transfer` can be ignored.

## Macro Metrics

| Split | Model | AUROC | AP | ECE | F1@0.5 | F1@tuned |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `d0_test` | baseline linear probe (`exp0015`) | 0.8455 | 0.2541 | 0.2667 | 0.2466 | 0.2769 |
| `d0_test` | memory-only | 0.8052 | 0.2235 | 0.0108 | 0.0724 | 0.2775 |
| `d0_test` | mixed | 0.8453 | 0.2531 | 0.4359 | 0.2485 | 0.2826 |
| `d1_transfer` | baseline linear probe (`exp0015`) | 0.8454 | 0.5430 | 0.2267 | 0.4610 | 0.3492 |
| `d1_transfer` | memory-only | 0.7697 | 0.4788 | 0.1338 | 0.0667 | 0.3673 |
| `d1_transfer` | mixed | 0.8452 | 0.5434 | 0.3072 | 0.4449 | 0.3514 |
| `d2_transfer` | baseline linear probe (`exp0015`) | 0.5007 | 0.1258 | 0.4232 | 0.1945 | 0.1357 |
| `d2_transfer` | memory-only | 0.5045 | 0.1259 | 0.0895 | 0.0275 | 0.1525 |
| `d2_transfer` | mixed | 0.5010 | 0.1258 | 0.3772 | 0.1948 | 0.1379 |

## Interpretation

- On NIH test, mixing preserved baseline AUROC/AP and slightly improved tuned-threshold F1.
- On CheXpert transfer, mixing matched baseline AUROC/AP closely, while memory-only lost ranking performance but slightly improved tuned-threshold F1.
- On MIMIC transfer, all methods remained near chance in AUROC/AP on this subset; memory-only improved calibration and tuned-threshold F1 slightly but not ranking quality.

## Optional Raw-Data Fallback

If raw MIMIC files ever need to be materialized for the manifest subset, the Kaggle dataset layout that matches the manifest is:

- `itsanmol124/mimic-cxr`

The selective downloader script has been updated to use that dataset as the default MIMIC source.
