# ResNet50 vs CXR Foundation RAG Comparison

This summary compares the same domain-transfer retrieval pipeline run on top of:

- ResNet50 embeddings from `exp0012` with baseline `exp0013`
- CXR Foundation embeddings from `exp0014` with baseline `exp0015`

Both branches use:

- no chest X-ray report features
- source retrieval memory built from `d0_nih/train`
- model selection on `d0_nih/val`
- frozen evaluation on `d0_nih/test`, `d1_chexpert/val`, and `d2_mimic/test`

## Selected Settings

### ResNet50

- memory selection dir:
  - `exp0041__domain_transfer_source_memory_selection__resnet50_default_avg_pilot5h_d0_val`
- best memory config:
  - `k = 50`
  - `tau = 5.0`
- mixing selection dir:
  - `exp0042__domain_transfer_probability_mixing_selection__resnet50_default_avg_pilot5h_d0_val`
- best mixing config:
  - `alpha = 0.7`

### CXR Foundation

- memory selection dir:
  - `exp0029__domain_transfer_source_memory_selection__cxr_foundation_general_avg_pilot5h_d0_val`
- best memory config:
  - `k = 50`
  - `tau = 5.0`
- mixing selection dir:
  - `exp0031__domain_transfer_probability_mixing_selection__cxr_foundation_general_avg_pilot5h_d0_val`
- best mixing config:
  - `alpha = 1.0`

## Macro Metrics

| Split | Model | AUROC | AP | ECE | F1@0.5 | F1@tuned |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `d0_test` | ResNet50 baseline | 0.7306 | 0.1263 | 0.4062 | 0.1720 | 0.1839 |
| `d0_test` | ResNet50 memory-only | 0.6781 | 0.1259 | 0.0114 | 0.0000 | 0.1664 |
| `d0_test` | ResNet50 mixed | 0.7353 | 0.1317 | 0.2821 | 0.1122 | 0.1769 |
| `d0_test` | CXR Foundation baseline | 0.8455 | 0.2541 | 0.2667 | 0.2466 | 0.2769 |
| `d0_test` | CXR Foundation memory-only | 0.8052 | 0.2235 | 0.0108 | 0.0724 | 0.2775 |
| `d0_test` | CXR Foundation mixed | 0.8453 | 0.2531 | 0.4359 | 0.2485 | 0.2826 |
| `d1_transfer` | ResNet50 baseline | 0.7218 | 0.3748 | 0.3091 | 0.3656 | 0.2946 |
| `d1_transfer` | ResNet50 memory-only | 0.6172 | 0.3135 | 0.1425 | 0.0000 | 0.2822 |
| `d1_transfer` | ResNet50 mixed | 0.7179 | 0.3670 | 0.1910 | 0.1097 | 0.3099 |
| `d1_transfer` | CXR Foundation baseline | 0.8454 | 0.5430 | 0.2267 | 0.4610 | 0.3492 |
| `d1_transfer` | CXR Foundation memory-only | 0.7697 | 0.4788 | 0.1338 | 0.0667 | 0.3673 |
| `d1_transfer` | CXR Foundation mixed | 0.8452 | 0.5434 | 0.3072 | 0.4449 | 0.3514 |
| `d2_transfer` | ResNet50 baseline | 0.4996 | 0.1278 | 0.4289 | 0.2064 | 0.1933 |
| `d2_transfer` | ResNet50 memory-only | 0.5215 | 0.1334 | 0.0680 | 0.0000 | 0.1505 |
| `d2_transfer` | ResNet50 mixed | 0.5042 | 0.1284 | 0.2865 | 0.0825 | 0.1986 |
| `d2_transfer` | CXR Foundation baseline | 0.5007 | 0.1258 | 0.4232 | 0.1945 | 0.1357 |
| `d2_transfer` | CXR Foundation memory-only | 0.5045 | 0.1259 | 0.0895 | 0.0275 | 0.1525 |
| `d2_transfer` | CXR Foundation mixed | 0.5010 | 0.1258 | 0.3772 | 0.1948 | 0.1379 |

## Main Takeaways

- The same RAG-style run is possible on ResNet50 embeddings and has now been executed.
- CXR Foundation is much stronger than ResNet50 on NIH and CheXpert, both before and after retrieval augmentation.
- On ResNet50, mixing gives only small ranking gains on NIH and MIMIC over the baseline and slightly improves tuned-threshold F1 on CheXpert, but it does not close the gap to CXR Foundation.
- On both embedding families, memory-only retrieval hurts ranking quality on NIH and CheXpert while improving calibration strongly.
- On MIMIC, both embedding families remain near chance in AUROC/AP; retrieval changes calibration and threshold behavior more than ranking quality.

## Practical Read

- If the goal is best overall transfer performance, the CXR Foundation branch remains the better choice.
- If the goal is to test whether the retrieval recipe itself transfers across embedding families, the answer is yes: the same pipeline runs on ResNet50 with the same manifest and split protocol.
