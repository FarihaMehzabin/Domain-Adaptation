# Supervisor-Facing Results Synthesis for the NIH CXR14 Source-Stage Experiments

Prepared on April 12, 2026 (UTC).

Most recent updates: the current active image-only domain-transfer pilot is summarized in Sections 0, 0A, 0B, and 0C below. If any older section later in this file conflicts with Sections 0, 0A, 0B, or 0C, treat Section 0C as the newest result for the present workspace.

## 0C. April 12, 2026 Update: Few-Shot CheXpert Target-Only Linear Probes on CXR Foundation

### Objective

After the NIH-source image-only baseline and the retrieval reruns, the next question was whether a small amount of labeled target-domain supervision on `CheXpert` could beat the existing `NIH -> CheXpert` linear baseline when the backbone remained frozen.

The specific question was:

- keep the same frozen `CXR Foundation` image embeddings
- train a new linear multilabel head on small labeled `CheXpert` target splits only
- evaluate on the same `234`-example `CheXpert valid` holdout that was used earlier as `D1 transfer`

### Experimental protocol

- Date:
  - `April 12, 2026 (UTC)`
- Backbone:
  - `CXR Foundation`
  - `general` image embeddings
  - `avg` token pooling
  - `768`-dim features
- Shared label space:
  - `atelectasis`
  - `cardiomegaly`
  - `consolidation`
  - `edema`
  - `pleural_effusion`
  - `pneumonia`
  - `pneumothorax`
- Target split construction:
  - `CheXpert train.csv` was split into disjoint target `train` and target `val`
  - `CheXpert valid.csv` was used unchanged as target `test`
- Shot settings:
  - `250 train / 250 val / 234 test`
  - `500 train / 500 val / 234 test`
  - `1000 train / 1000 val / 234 test`
- Model head:
  - frozen-backbone linear multilabel classifier
- Selection rule:
  - early stop on target `val` macro AUROC
  - tune thresholds on target `val`
  - report final results on target `test`
- Comparator:
  - the `NIH`-source `CXR Foundation` linear baseline from Section `0`
- Split identity check:
  - the earlier `D1 = CheXpert val` rows and the new target `test` rows were verified to have identical `row_id` sets and identical image paths

### Run lineage

- full manifest with CheXpert train enabled:
  - `/workspace/manifest_common_labels_nih_train_val_test_chexpert_mimic_with_train.csv`
- target manifests:
  - `/workspace/manifest_chexpert_target_250.csv`
  - `/workspace/manifest_chexpert_target_500.csv`
  - `/workspace/manifest_chexpert_target_1000.csv`
- `250-shot` target run:
  - embedding export: `exp0050__cxr_foundation_embedding_export__chexpert_target_250_cxr_foundation_avg_batch128`
  - head training: `exp0051__domain_transfer_head_training__chexpert_target_250_cxr_foundation_linear_gpu`
- `500-shot` target run:
  - embedding export: `exp0052__cxr_foundation_embedding_export__chexpert_target_500_cxr_foundation_avg_batch128`
  - head training: `exp0053__domain_transfer_head_training__chexpert_target_500_cxr_foundation_linear_gpu`
- `1000-shot` target run:
  - embedding export: `exp0054__cxr_foundation_embedding_export__chexpert_target_1000_cxr_foundation_avg_batch128`
  - head training: `exp0055__domain_transfer_head_training__chexpert_target_1000_cxr_foundation_linear_gpu`

### Main quantitative results

The earlier `NIH`-source comparator and the new target-only runs were evaluated on the same `234`-example `CheXpert` holdout.

| Training rule | Train rows | Val rows | Same CheXpert test AUROC | Same CheXpert test AP | Same CheXpert test F1@0.5 | Same CheXpert test F1@tuned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `NIH`-source linear baseline (`exp0015`) | `10,000 NIH` | `1,000 NIH` | `0.8454` | `0.5430` | `0.4610` | `0.3492` |
| `CheXpert` target-only linear (`250/250`) | `250` | `250` | `0.7532` | `0.4764` | `0.4521` | `0.4166` |
| `CheXpert` target-only linear (`500/500`) | `500` | `500` | `0.7611` | `0.4522` | `0.4356` | `0.3997` |
| `CheXpert` target-only linear (`1000/1000`) | `1000` | `1000` | `0.7702` | `0.4715` | `0.4448` | `0.4210` |

Target-only progression:

- AUROC improved from `0.7532` at `250` shots to `0.7702` at `1000` shots.
- AP was not monotonic:
  - `250`: `0.4764`
  - `500`: `0.4522`
  - `1000`: `0.4715`
- The best target-only fixed-threshold result also stayed at `250`, with `F1@0.5 = 0.4521`.

### Interpretation

- More target-only `CheXpert` supervision helped, but the gains were moderate:
  - `250 -> 1000` improved test macro AUROC by `+0.0170`
  - `250 -> 1000` changed test macro AP by `-0.0049`
  - `250 -> 1000` changed test macro `F1@0.5` by `-0.0074`
- The main comparison did not flip:
  - the `NIH`-source linear baseline remained ahead of the best target-only run by `+0.0752` AUROC
  - it also remained ahead by `+0.0715` AP
  - it remained slightly ahead on fixed-threshold `F1@0.5` by `+0.0162`
- Therefore, within this frozen-backbone linear-head setup, up to `1000` labeled target training examples were still not enough to beat the existing `NIH`-source `CXR Foundation` head on the same `CheXpert` holdout.

Thresholded interpretation:

- The target-only runs showed higher `F1@tuned` than the `NIH`-source baseline.
- However, that comparison should be treated cautiously because the target-only thresholds were tuned on a separate `CheXpert` target `val` split, whereas the original source baseline followed the earlier source-domain validation protocol.
- The cleaner primary comparison is therefore AUROC, AP, and fixed-threshold `F1@0.5`, where the `NIH`-source baseline still leads.

### Supervisor-facing message

The important supervisor-facing clarification is that this is not a split-mismatch artifact:

- the old source-transfer `CheXpert val` set and the new target-only `CheXpert test` set are the same `234` studies
- the comparison is therefore directly on the same held-out target examples

The defensible conclusion is:

1. `CXR Foundation` transfers strongly enough from `NIH` that a source-trained linear head remains hard to beat.
2. A fresh target-only linear head does improve as more `CheXpert` labels are added, but even `1000` target training examples do not yet surpass the `NIH`-source baseline on the main ranking metrics.
3. The next adaptation step should therefore not be framed as “replace source training with a tiny target-only head.” The more plausible next move is to warm-start from the strong `NIH`-source head and then adapt it with target supervision.

### Artifact locations for this update

- synthesis file:
  - `/workspace/SUPERVISOR_RESULTS_SYNTHESIS.md`
- target manifests:
  - `/workspace/manifest_chexpert_target_250.csv`
  - `/workspace/manifest_chexpert_target_500.csv`
  - `/workspace/manifest_chexpert_target_1000.csv`
- target-run experiment directories:
  - `/workspace/experiments/active/exp0051__domain_transfer_head_training__chexpert_target_250_cxr_foundation_linear_gpu`
  - `/workspace/experiments/active/exp0053__domain_transfer_head_training__chexpert_target_500_cxr_foundation_linear_gpu`
  - `/workspace/experiments/active/exp0055__domain_transfer_head_training__chexpert_target_1000_cxr_foundation_linear_gpu`

## 0B. April 12, 2026 Update: Retrieval-Augmented Domain-Transfer Comparison on CXR Foundation vs ResNet50

### Objective

After the image-only backbone comparison in Section 0 established that `CXR Foundation` was stronger than `ResNet50`, the next question was whether a retrieval-augmented pipeline could improve domain transfer when training still began from the same NIH source domain.

The goal of this update was to rerun the same retrieval recipe on both embedding families and compare them directly:

- `CXR Foundation` embeddings with a source-memory retrieval module
- `ResNet50` embeddings with the same source-memory retrieval module

The retrieval pipeline was designed to mimic the earlier source-memory experiments conceptually, but now under the domain-transfer pilot split and without chest X-ray reports.

### Experimental protocol

- Date:
  - `April 12, 2026 (UTC)`
- Data split protocol:
  - source memory built from `D0 = NIH train`
  - retrieval hyperparameters selected on `D0 = NIH val`
  - final evaluation on:
    - `D0 test = NIH test`
    - `D1 transfer = CheXpert val`
    - `D2 transfer = MIMIC test`
- Shared label space:
  - `atelectasis`
  - `cardiomegaly`
  - `consolidation`
  - `edema`
  - `pleural_effusion`
  - `pneumonia`
  - `pneumothorax`
- Manifest:
  - `/workspace/manifest_common_labels_pilot5h.csv`
- Sample counts:
  - `NIH train = 10,000`
  - `NIH val = 1,000`
  - `NIH test = 2,000`
  - `CheXpert val = 234`
  - `MIMIC test = 1,455`
- Input modality:
  - image embeddings only
  - no chest X-ray report features were used
- Data handling:
  - no raw-image redownload was needed for the reruns
  - the existing embedding exports were reused directly

### Retrieval method

For each backbone, the retrieval branch used the same three-stage procedure:

1. Build a source memory bank from `NIH train` embeddings and labels.
2. For each query image, retrieve nearest source examples from the memory bank and convert neighbor labels into a memory-derived multilabel probability vector.
3. Evaluate two retrieval variants:
   - `memory-only`: use only the retrieval-derived probability vector
   - `mixed`: interpolate between the baseline classifier probabilities and the retrieval probabilities

Operationally, the pipeline implemented:

- FAISS nearest-neighbor retrieval over the source embedding bank
- a temperature-style retrieval sharpness parameter `tau`
- a neighbor count `k`
- a probability mixing coefficient `alpha`

The selection rule was the same in both branches:

- choose `k` and `tau` on `NIH val` by macro AUROC
- then choose `alpha` on `NIH val` by macro AUROC

### Run lineage

#### CXR Foundation retrieval branch

- source memory build:
  - `exp0028__domain_transfer_source_retrieval_memory_building__domain_transfer_source_retrieval_memory__cxr_foundation_general_avg_pilot5h_d0_train`
- memory-only selection on NIH val:
  - `exp0029__domain_transfer_source_memory_selection__cxr_foundation_general_avg_pilot5h_d0_val`
- probability-mixing selection on NIH val:
  - `exp0031__domain_transfer_probability_mixing_selection__cxr_foundation_general_avg_pilot5h_d0_val`
- memory-only target evaluation:
  - `exp0034__domain_transfer_source_memory_target_evaluation__cxr_foundation_general_avg_pilot5h_d0_test`
  - `exp0035__domain_transfer_source_memory_target_evaluation__cxr_foundation_general_avg_pilot5h_d1_transfer`
  - `exp0036__domain_transfer_source_memory_target_evaluation__cxr_foundation_general_avg_pilot5h_d2_transfer`
- mixed target evaluation:
  - `exp0037__domain_transfer_probability_mixing_target_evaluation__cxr_foundation_general_avg_pilot5h_d0_test`
  - `exp0031__domain_transfer_probability_mixing_target_evaluation__domain_transfer_probability_mixing_selection__cxr_foundation_general_avg_pilot5h_d0_val__d1_transfer`
  - `exp0031__domain_transfer_probability_mixing_target_evaluation__domain_transfer_probability_mixing_selection__cxr_foundation_general_avg_pilot5h_d0_val__d2_transfer`
- compact branch summary:
  - `/workspace/experiments/complete/exp0039__domain_transfer_rag_summary__cxr_foundation_general_avg_pilot5h/summary.md`

#### ResNet50 retrieval branch

- source memory build:
  - `exp0040__domain_transfer_source_retrieval_memory_building__domain_transfer_source_retrieval_memory__resnet50_default_avg_pilot5h_d0_train`
- memory-only selection on NIH val:
  - `exp0041__domain_transfer_source_memory_selection__resnet50_default_avg_pilot5h_d0_val`
- probability-mixing selection on NIH val:
  - `exp0042__domain_transfer_probability_mixing_selection__resnet50_default_avg_pilot5h_d0_val`
- memory-only target evaluation:
  - `exp0043__domain_transfer_source_memory_target_evaluation__resnet50_default_avg_pilot5h_d0_test`
  - `exp0044__domain_transfer_source_memory_target_evaluation__resnet50_default_avg_pilot5h_d1_transfer`
  - `exp0044__domain_transfer_source_memory_target_evaluation__resnet50_default_avg_pilot5h_d2_transfer`
- mixed target evaluation:
  - `exp0046__domain_transfer_probability_mixing_target_evaluation__resnet50_default_avg_pilot5h_d0_test`
  - `exp0045__domain_transfer_probability_mixing_target_evaluation__resnet50_default_avg_pilot5h_d1_transfer`
  - `exp0046__domain_transfer_probability_mixing_target_evaluation__resnet50_default_avg_pilot5h_d2_transfer`
- compact cross-backbone summary:
  - `/workspace/experiments/complete/exp0047__resnet50_vs_cxr_foundation_rag_comparison__pilot5h/summary.md`

### Selected hyperparameters

| Backbone | Best `k` | Best `tau` | Best `alpha` | NIH val mixed macro AUROC | NIH val mixed macro AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| CXR Foundation | `50` | `5.0` | `1.0` | `0.8479` | `0.2643` |
| ResNet50 | `50` | `5.0` | `0.7` | `0.7396` | `0.1373` |

Interpretation:

- Both backbones preferred the same retrieval depth and temperature:
  - `k = 50`
  - `tau = 5.0`
- The CXR Foundation branch preferred full trust in retrieval at mixing time:
  - `alpha = 1.0`
- The ResNet50 branch preferred a partial blend:
  - `alpha = 0.7`

### Detailed results

#### NIH source test (`D0 test`)

| Backbone | Variant | AUROC | AP | ECE | F1@0.5 | F1@tuned |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CXR Foundation | baseline | `0.8455` | `0.2541` | `0.2667` | `0.2466` | `0.2769` |
| CXR Foundation | memory-only | `0.8052` | `0.2235` | `0.0108` | `0.0724` | `0.2775` |
| CXR Foundation | mixed | `0.8453` | `0.2531` | `0.4359` | `0.2485` | `0.2826` |
| ResNet50 | baseline | `0.7306` | `0.1263` | `0.4062` | `0.1720` | `0.1839` |
| ResNet50 | memory-only | `0.6781` | `0.1259` | `0.0114` | `0.0000` | `0.1664` |
| ResNet50 | mixed | `0.7353` | `0.1317` | `0.2821` | `0.1122` | `0.1769` |

NIH interpretation:

- `CXR Foundation` remained far stronger than `ResNet50` under all three variants.
- For `CXR Foundation`, retrieval mixing did not improve ranking quality meaningfully over the baseline:
  - AUROC stayed effectively unchanged
  - AP stayed effectively unchanged
  - tuned-threshold F1 increased slightly
- For `ResNet50`, the mixed variant produced a small AUROC and AP lift over the baseline, but the absolute performance level remained much lower than the CXR Foundation branch.
- Memory-only retrieval reduced ranking quality for both backbones on the source domain, although it sharply improved calibration.

#### CheXpert transfer (`D1 transfer`)

| Backbone | Variant | AUROC | AP | ECE | F1@0.5 | F1@tuned |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CXR Foundation | baseline | `0.8454` | `0.5430` | `0.2267` | `0.4610` | `0.3492` |
| CXR Foundation | memory-only | `0.7697` | `0.4788` | `0.1338` | `0.0667` | `0.3673` |
| CXR Foundation | mixed | `0.8452` | `0.5434` | `0.3072` | `0.4449` | `0.3514` |
| ResNet50 | baseline | `0.7218` | `0.3748` | `0.3091` | `0.3656` | `0.2946` |
| ResNet50 | memory-only | `0.6172` | `0.3135` | `0.1425` | `0.0000` | `0.2822` |
| ResNet50 | mixed | `0.7179` | `0.3670` | `0.1910` | `0.1097` | `0.3099` |

CheXpert interpretation:

- `CXR Foundation` remained substantially ahead of `ResNet50` on transfer to CheXpert.
- For `CXR Foundation`, the mixed retrieval variant was essentially tied with the baseline:
  - AUROC was flat
  - AP was flat
  - tuned-threshold F1 increased only slightly
- For `ResNet50`, the mixed variant did not improve ranking over the baseline and was slightly lower in AUROC and AP, although tuned-threshold F1 improved modestly.
- Memory-only retrieval again hurt ranking quality for both backbones.

#### MIMIC transfer (`D2 transfer`)

| Backbone | Variant | AUROC | AP | ECE | F1@0.5 | F1@tuned |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CXR Foundation | baseline | `0.5007` | `0.1258` | `0.4232` | `0.1945` | `0.1357` |
| CXR Foundation | memory-only | `0.5045` | `0.1259` | `0.0895` | `0.0275` | `0.1525` |
| CXR Foundation | mixed | `0.5010` | `0.1258` | `0.3772` | `0.1948` | `0.1379` |
| ResNet50 | baseline | `0.4996` | `0.1278` | `0.4289` | `0.2064` | `0.1933` |
| ResNet50 | memory-only | `0.5215` | `0.1334` | `0.0680` | `0.0000` | `0.1505` |
| ResNet50 | mixed | `0.5042` | `0.1284` | `0.2865` | `0.0825` | `0.1986` |

MIMIC interpretation:

- This remained the hardest domain by far.
- All variants for both backbones were effectively near chance in AUROC and AP.
- `CXR Foundation` no longer showed the large advantage it had on NIH and CheXpert.
- `ResNet50` memory-only had the highest AUROC/AP in this table, but the absolute values remained weak and the model collapsed at a fixed `0.5` threshold.
- The practical conclusion is that neither backbone plus simple retrieval solved the NIH-to-MIMIC domain shift.

### Cross-run synthesis

The main findings across the two retrieval runs are:

1. The same retrieval framework runs cleanly on both embedding families.
2. `CXR Foundation` remains the stronger representation on the NIH source domain and on CheXpert transfer.
3. Retrieval mixing does not materially improve the already-strong `CXR Foundation` baseline on NIH or CheXpert.
4. Retrieval mixing gives only modest, inconsistent changes on the weaker `ResNet50` branch.
5. Memory-only retrieval improves calibration strongly in both branches, but it usually hurts ranking quality.
6. The core unsolved problem remains `NIH -> MIMIC` transfer.

### Supervisor-facing interpretation

For presentation, the most coherent reading is:

- The original backbone comparison already showed that `CXR Foundation` is the correct image-only backbone to carry forward.
- The retrieval-augmented rerun confirms that this conclusion survives under a more elaborate inference setup.
- In other words, the backbone ranking does not flip once retrieval is added:
  - `CXR Foundation + RAG` is still much stronger than `ResNet50 + RAG` on NIH and CheXpert.
- However, retrieval does not create a breakthrough on the hardest transfer target:
  - `MIMIC` remains near chance for both families.

This is useful supervisor-facing evidence because it narrows the search space:

- switching from `ResNet50` to `CXR Foundation` was the right move
- adding a simple source-memory RAG layer is not enough to solve the MIMIC shift
- future adaptation work should focus on genuine domain-alignment strategies rather than expecting nearest-neighbor retrieval alone to fix transfer

### Recommended presentation message

One defensible presentation storyline is:

1. Start with the simple source-only linear probe comparison.
2. Show that `CXR Foundation` clearly dominates `ResNet50`.
3. Introduce the retrieval-augmented rerun as a stronger test of whether source-memory reasoning changes the conclusion.
4. Show that the conclusion remains the same:
   - `CXR Foundation` is still the better backbone
   - retrieval gives only marginal changes on NIH and CheXpert
   - MIMIC remains the failure case
5. Use that failure case to motivate the next stage of true domain adaptation.

### Artifact locations for this update

- Primary synthesis file:
  - `/workspace/SUPERVISOR_RESULTS_SYNTHESIS.md`
- CXR Foundation RAG branch summary:
  - `/workspace/experiments/complete/exp0039__domain_transfer_rag_summary__cxr_foundation_general_avg_pilot5h/summary.md`
- ResNet50 vs CXR Foundation RAG comparison:
  - `/workspace/experiments/complete/exp0047__resnet50_vs_cxr_foundation_rag_comparison__pilot5h/summary.md`

## 0. April 11, 2026 Update: Pilot Image-Only Domain-Transfer Comparison

### Objective

Run a controlled pilot comparison of two simple image-only source baselines before committing to long full-data runs:

- `ResNet50` image embeddings as the simple CNN baseline
- `CXR Foundation` image embeddings as the chest-X-ray-specific foundation baseline

The experimental goal was to train on NIH and then measure direct transfer to CheXpert and MIMIC under the same frozen linear-probe protocol.

### Experimental protocol

- Domains:
  - `D0 = NIH CXR14`
  - `D1 = CheXpert`
  - `D2 = MIMIC-CXR`
- Shared labels:
  - `atelectasis`
  - `cardiomegaly`
  - `consolidation`
  - `edema`
  - `pleural_effusion`
  - `pneumonia`
  - `pneumothorax`
- Pilot subset manifest:
  - `/workspace/manifest_common_labels_pilot5h.csv`
- Pilot subset sizes:
  - `D0 train = 10,000`
  - `D0 val = 1,000`
  - `D0 test = 2,000`
  - `D1 val = 234`
  - `D2 test = 1,455`
- Training rule:
  - train a frozen multilabel linear head on `D0 train` embeddings only
  - early stop on `D0 val` macro AUROC
  - evaluate the selected checkpoint on `D0 test`, `D1 val`, and `D2 test`
- Deployment interpretation:
  - this is still an image-only pipeline at inference time
  - no report input is required at deployment
  - for CXR Foundation, report information only enters indirectly through pretraining of the backbone

### Run lineage and implementation details

| Experiment | Purpose | Key settings | Main outcome |
| --- | --- | --- | --- |
| `exp0011` | ResNet50 batch-size sweep | `256` to `2048` on `D0 train` | highest successful batch size was `1536`; `1792` produced CUDA OOM |
| `exp0012` | ResNet50 pilot embedding export | torchvision `resnet50`, avg pooling, `2048`-dim embeddings, batch size `1536` | completed all `14,689` pilot images |
| `exp0013` | ResNet50 pilot linear probe | frozen linear probe, best epoch `50` | completed source and transfer evaluation |
| `exp0014` | CXR Foundation pilot embedding export | existing `/workspace/scripts/14_generate_cxr_foundation_embeddings.py` path, `general` embeddings, `avg` token pooling, `768`-dim embeddings, batch size `128` | completed all `14,689` pilot images |
| `exp0015` | CXR Foundation pilot linear probe | frozen linear probe, best epoch `49` | completed source and transfer evaluation |

Additional implementation notes:

- The CXR Foundation exporter in `exp0014` was patched to checkpoint every batch into `_partial_batches` and to update `split_progress.json` after each completed batch. This was done so that partial progress would be preserved if a long run was interrupted.
- The CXR Foundation export intentionally followed the existing `14_generate_cxr_foundation_embeddings.py` path rather than switching to an alternative feature path mid-study. That kept the comparison aligned with the original planned experiment.

### Runtime summary

- `exp0011` ResNet50 batch sweep:
  - started `2026-04-11T09:07:06Z`
  - finished `2026-04-11T09:12:23Z`
  - duration about `5m 17s`
- `exp0012` ResNet50 pilot embedding export:
  - started `2026-04-11T09:14:43Z`
  - finished `2026-04-11T09:19:51Z`
  - duration about `5m 08s`
- `exp0014` CXR Foundation pilot embedding export:
  - started `2026-04-11T09:20:33Z`
  - finished `2026-04-11T14:07:43Z`
  - duration about `4h 47m 10s`

The practical throughput difference was large:

- ResNet50 export was fast enough that the full pilot subset finished in a few minutes.
- CXR Foundation export required several hours on the same subset.
- The observed CXR Foundation throughput on this machine, using the existing exporter path, was roughly `1.43 seconds/image`.

### Main quantitative results

#### Macro metrics

| Model | D0 val AUROC | D0 test AUROC | D0 test AP | D1 transfer AUROC | D1 transfer AP | D2 transfer AUROC | D2 transfer AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet50 | `0.7376` | `0.7306` | `0.1263` | `0.7218` | `0.3748` | `0.4996` | `0.1278` |
| CXR Foundation | `0.8482` | `0.8455` | `0.2541` | `0.8454` | `0.5430` | `0.5007` | `0.1258` |
| Delta (`CXR - ResNet`) | `+0.1106` | `+0.1149` | `+0.1278` | `+0.1237` | `+0.1682` | `+0.0011` | `-0.0020` |

#### Key interpretation of the table

- On the NIH source test set, CXR Foundation substantially outperformed ResNet50.
- On direct transfer from NIH to CheXpert, CXR Foundation also substantially outperformed ResNet50.
- On direct transfer from NIH to MIMIC, neither model performed meaningfully above chance at the macro level.
- Therefore:
  - `CXR Foundation` is clearly the stronger source backbone
  - but a stronger source backbone alone is not enough to solve the NIH to MIMIC shift

### Label-level observations

#### NIH test (`D0`)

CXR Foundation improved AUROC over ResNet50 for all 7 labels:

- `atelectasis`: `0.7995` vs `0.7052`
- `cardiomegaly`: `0.8730` vs `0.7028`
- `consolidation`: `0.8062` vs `0.7318`
- `edema`: `0.9056` vs `0.8462`
- `pleural_effusion`: `0.8666` vs `0.7590`
- `pneumonia`: `0.8097` vs `0.6552`
- `pneumothorax`: `0.8579` vs `0.7138`

#### CheXpert validation transfer (`D1`)

CXR Foundation again improved AUROC over ResNet50 for all 7 labels:

- `atelectasis`: `0.8471` vs `0.7822`
- `cardiomegaly`: `0.7371` vs `0.5045`
- `consolidation`: `0.9145` vs `0.7790`
- `edema`: `0.8455` vs `0.7586`
- `pleural_effusion`: `0.9057` vs `0.7934`
- `pneumonia`: `0.8966` vs `0.8042`
- `pneumothorax`: `0.7716` vs `0.6305`

The largest practical gap on CheXpert was for `cardiomegaly`, where the ResNet50 baseline was weak and the CXR Foundation backbone materially improved transfer.

#### MIMIC test transfer (`D2`)

The MIMIC result was qualitatively different:

- ResNet50 macro AUROC: `0.4996`
- CXR Foundation macro AUROC: `0.5007`

Per-label MIMIC AUROCs for CXR Foundation stayed tightly clustered around chance:

- `atelectasis`: `0.5077`
- `cardiomegaly`: `0.4996`
- `consolidation`: `0.4972`
- `edema`: `0.5002`
- `pleural_effusion`: `0.5016`
- `pneumonia`: `0.5019`
- `pneumothorax`: `0.4968`

This means the main limitation is not simply that ResNet50 is too weak. The larger issue is that direct NIH to MIMIC transfer is a hard domain shift in this setup.

### Supervisor-facing interpretation

The strongest supervisor-facing conclusion is:

- We started with a simple, controlled image-only transfer study on a manageable pilot subset rather than running expensive full-data experiments blindly.
- Under the same data split and the same frozen linear-probe protocol, `CXR Foundation` clearly beat the old `ResNet50` image-only baseline on the NIH source task and on direct transfer to CheXpert.
- However, neither backbone solved direct transfer to MIMIC.
- This suggests that:
  - choosing the right chest-X-ray-specific source backbone matters
  - but for the hardest target domain, representation quality alone is not sufficient
  - actual domain adaptation is still needed for MIMIC

This gives a coherent staged story for presentation:

1. Start with a simple CNN source baseline.
2. Replace it with a chest-X-ray-specific foundation model.
3. Show that the stronger source backbone improves source    classification and easier cross-domain transfer.
4. Use the remaining failure on MIMIC to motivate the next adaptation step.

### Recommendation

Based on this pilot, the most defensible next move is:

- promote `CXR Foundation` as the primary image-only source backbone for subsequent adaptation experiments
- keep `ResNet50` as the simple baseline comparator
- use `MIMIC` as the main target domain for adaptation experiments, because it is the domain where simple direct transfer still fails

### Key artifact locations for presentation

- Pilot manifest:
  - `/workspace/manifest_common_labels_pilot5h.csv`
- ResNet50 embedding export:
  - `/workspace/experiments/complete/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h`
- ResNet50 transfer evaluation:
  - `/workspace/experiments/complete/exp0013__domain_transfer_linear_probe__resnet50_default_avg_pilot5h`
- CXR Foundation embedding export:
  - `/workspace/experiments/complete/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128`
- CXR Foundation transfer evaluation:
  - `/workspace/experiments/complete/exp0015__domain_transfer_linear_probe__cxr_foundation_general_avg_pilot5h`
- Main logs:
  - `/workspace/logs/exp0011__torch_image_batch_sweep__pilot5h_d0_train_resnet50.log`
  - `/workspace/logs/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h.log`
  - `/workspace/logs/exp0013__domain_transfer_linear_probe__resnet50_default_avg_pilot5h.log`
  - `/workspace/logs/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128.log`
  - `/workspace/logs/exp0015__domain_transfer_linear_probe__cxr_foundation_general_avg_pilot5h.log`

## 0A. April 11, 2026 Update: Source-Trained MLP Sweep on the Same Pilot Transfer Setup

### Objective

After establishing that `CXR Foundation` was the stronger image-only source backbone, the next controlled question was whether a slightly more expressive source-trained head could improve transfer further without changing the embedding backbone.

The specific comparison was:

- keep the same frozen pilot embeddings
- keep the same NIH-source training rule
- replace the linear multilabel head with a small one-hidden-layer MLP
- evaluate whether this improves:
  - `D0 = NIH` source classification
  - `D1 = CheXpert` direct transfer
  - `D2 = MIMIC-CXR` direct transfer

### Experimental protocol

- Date of sweep:
  - `April 11, 2026 (UTC)`
- Data:
  - reused the exact same pilot subset manifest from Section 0:
    - `/workspace/manifest_common_labels_pilot5h.csv`
- Embedding roots:
  - ResNet50 pilot embeddings:
    - `/workspace/experiments/complete/exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h`
  - CXR Foundation pilot embeddings:
    - `/workspace/experiments/complete/exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128`
- Head family:
  - one-hidden-layer MLP
  - `ReLU`
  - dropout `0.2`
  - hidden sizes swept:
    - `256`
    - `512`
    - `1024`
- Selection rule:
  - rank all candidate MLP runs only by `D0 val` macro AUROC
  - break ties by `D0 val` macro average precision
  - final tie-break by lower `D0 val` loss
- Training rule:
  - train on `NIH train` only
  - early stop on `NIH val` macro AUROC
  - evaluate on:
    - `NIH test`
    - `CheXpert val`
    - `MIMIC test`

This means the MLP sweep did not introduce any target-domain tuning. It stayed within the same NIH-source direct-transfer framing as the linear baseline.

### Run lineage

| Experiment | Purpose | Main outcome |
| --- | --- | --- |
| `exp0016` | pilot MLP sweep orchestrator | launched all 6 source-trained MLP runs and built the ranked leaderboard |
| `exp0017` | ResNet50 + MLP `256` | best ResNet MLP by `D0 val` |
| `exp0018` | ResNet50 + MLP `512` | completed |
| `exp0019` | ResNet50 + MLP `1024` | completed |
| `exp0020` | CXR Foundation + MLP `256` | completed |
| `exp0021` | CXR Foundation + MLP `512` | completed |
| `exp0022` | CXR Foundation + MLP `1024` | best overall MLP by `D0 val` |

### Main quantitative results

#### Linear baseline versus best MLP per backbone

| Backbone | Head | Selection experiment | D0 val AUROC | D0 test AUROC | D1 transfer AUROC | D2 transfer AUROC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ResNet50 | Linear | `exp0013` | `0.7376` | `0.7306` | `0.7218` | `0.4996` |
| ResNet50 | Best MLP (`256`) | `exp0017` | `0.7600` | `0.7590` | `0.7073` | `0.5037` |
| Delta (`MLP - Linear`) |  |  | `+0.0224` | `+0.0284` | `-0.0145` | `+0.0041` |
| CXR Foundation | Linear | `exp0015` | `0.8482` | `0.8455` | `0.8454` | `0.5007` |
| CXR Foundation | Best MLP (`1024`) | `exp0022` | `0.8527` | `0.8473` | `0.8369` | `0.5029` |
| Delta (`MLP - Linear`) |  |  | `+0.0045` | `+0.0018` | `-0.0085` | `+0.0022` |

#### Full MLP sweep summary

| Backbone | Hidden size | Experiment | D0 val AUROC | D0 test AUROC | D1 transfer AUROC | D2 transfer AUROC |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| ResNet50 | `256` | `exp0017` | `0.7600` | `0.7590` | `0.7073` | `0.5037` |
| ResNet50 | `512` | `exp0018` | `0.7571` | `0.7560` | `0.7037` | `0.5025` |
| ResNet50 | `1024` | `exp0019` | `0.7572` | `0.7587` | `0.7160` | `0.5063` |
| CXR Foundation | `256` | `exp0020` | `0.8513` | `0.8468` | `0.8201` | `0.5001` |
| CXR Foundation | `512` | `exp0021` | `0.8514` | `0.8438` | `0.8421` | `0.5056` |
| CXR Foundation | `1024` | `exp0022` | `0.8527` | `0.8473` | `0.8369` | `0.5029` |

### Key interpretation

The MLP result was mixed and is useful precisely because it was run in a controlled way.

- For `ResNet50`, all three MLP variants improved `NIH` source performance relative to the linear head.
- However, all three `ResNet50` MLP variants reduced direct transfer to `CheXpert`.
- For `CXR Foundation`, the best MLP by the official selection rule slightly improved `NIH` source performance relative to the linear head.
- But none of the `CXR Foundation` MLP variants beat the linear head on `CheXpert` transfer.
- For `MIMIC`, all changes were very small and remained effectively near chance overall.

The most important interpretation is:

- a more expressive source head can improve source fitting on `NIH`
- but that extra source expressiveness does not automatically improve cross-domain transfer
- in this pilot, the MLP tended to help `D0` more than `D1`

### Supervisor-facing conclusion

The supervisor-facing conclusion from the MLP sweep is narrower than the backbone conclusion in Section 0.

- The backbone change from `ResNet50` to `CXR Foundation` clearly improved both source performance and `CheXpert` transfer.
- The head change from `linear` to `small MLP` did not produce the same kind of transfer benefit.
- Therefore, the practical recommendation is:
  - keep `CXR Foundation` as the stronger source backbone
  - keep the `linear` head as the default direct-transfer head
  - treat the small MLP as an informative ablation showing that better source fit alone does not guarantee better domain transfer

This is useful for presentation because it shows the experimental sequence was not random:

1. start from a simple CNN and linear head
2. improve the backbone while keeping the head simple
3. separately test whether a more flexible head explains the gains
4. conclude that the main improvement came from the chest-X-ray-specific backbone, not from simply making the classifier head larger

### Recommendation

For the next stage, the most defensible default is still:

- `CXR Foundation` embeddings
- `linear` source-trained multilabel head
- adaptation experiments focused on the hardest transfer case, `NIH -> MIMIC`

The MLP sweep should be kept in the write-up as a negative-but-informative control:

- it improved source-domain NIH metrics
- it did not improve the main transfer result that matters most for the story

### Key artifact locations for presentation

- MLP sweep orchestrator:
  - `/workspace/experiments/complete/exp0016__domain_transfer_mlp_sweep__pilot5h_domain_transfer_mlp_sweep`
- MLP sweep leaderboard:
  - `/workspace/experiments/complete/exp0016__domain_transfer_mlp_sweep__pilot5h_domain_transfer_mlp_sweep/leaderboard.json`
  - `/workspace/experiments/complete/exp0016__domain_transfer_mlp_sweep__pilot5h_domain_transfer_mlp_sweep/leaderboard.csv`
  - `/workspace/experiments/complete/exp0016__domain_transfer_mlp_sweep__pilot5h_domain_transfer_mlp_sweep/summary.md`
- ResNet50 MLP runs:
  - `/workspace/experiments/complete/exp0017__domain_transfer_head_training__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h__pilot5h__head-mlp__hidden-256__dropout-0p2`
  - `/workspace/experiments/complete/exp0018__domain_transfer_head_training__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h__pilot5h__head-mlp__hidden-512__dropout-0p2`
  - `/workspace/experiments/complete/exp0019__domain_transfer_head_training__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h__pilot5h__head-mlp__hidden-1024__dropout-0p2`
- CXR Foundation MLP runs:
  - `/workspace/experiments/complete/exp0020__domain_transfer_head_training__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128__pilot5h__head-mlp__hidden-256__dropout-0p2`
  - `/workspace/experiments/complete/exp0021__domain_transfer_head_training__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128__pilot5h__head-mlp__hidden-512__dropout-0p2`
  - `/workspace/experiments/complete/exp0022__domain_transfer_head_training__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128__pilot5h__head-mlp__hidden-1024__dropout-0p2`
- Main logs:
  - `/workspace/logs/exp0016__domain_transfer_mlp_sweep__pilot5h_domain_transfer_mlp_sweep.log`
  - `/workspace/logs/exp0016__domain_transfer_mlp_sweep__pilot5h_domain_transfer_mlp_sweep__exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h__hidden-256.log`
  - `/workspace/logs/exp0016__domain_transfer_mlp_sweep__pilot5h_domain_transfer_mlp_sweep__exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h__hidden-512.log`
  - `/workspace/logs/exp0016__domain_transfer_mlp_sweep__pilot5h_domain_transfer_mlp_sweep__exp0012__embedding_generation__domain_transfer_resnet50_default_avg_pilot5h__hidden-1024.log`
  - `/workspace/logs/exp0016__domain_transfer_mlp_sweep__pilot5h_domain_transfer_mlp_sweep__exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128__hidden-256.log`
  - `/workspace/logs/exp0016__domain_transfer_mlp_sweep__pilot5h_domain_transfer_mlp_sweep__exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128__hidden-512.log`
  - `/workspace/logs/exp0016__domain_transfer_mlp_sweep__pilot5h_domain_transfer_mlp_sweep__exp0014__cxr_foundation_embedding_export__pilot5h_common7_general_avg_batch128__hidden-1024.log`

## 0B. April 11, 2026 Update: Image-Only ViT LoRA Pilot

### Objective

After the MLP sweep, the next question was whether a lightweight parameter-efficient backbone adaptation could beat the simple frozen-image baselines without requiring full backbone finetuning.

This pilot tested:

- a generic vision transformer backbone
- LoRA adapters on the attention projections
- the same `NIH -> CheXpert/MIMIC` direct-transfer protocol as Sections 0 and 0A

### Experimental protocol

- Model:
  - `google/vit-base-patch16-224-in21k`
- Adaptation method:
  - LoRA on attention `query` and `value` modules
  - `r = 8`
  - `alpha = 16`
  - `dropout = 0.1`
- Training setup:
  - image-only
  - `batch size = 256`
  - mixed precision on CUDA
  - max `12` epochs
  - early stopping patience `3`
- Data:
  - the same pilot manifest:
    - `/workspace/manifest_common_labels_pilot5h.csv`
- Selection rule:
  - train on `NIH train`
  - early stop on `NIH val` macro AUROC
  - evaluate on:
    - `NIH test`
    - `CheXpert val`
    - `MIMIC test`

### Run lineage

| Experiment | Purpose | Main outcome |
| --- | --- | --- |
| `exp0023` | smoke run | verified end-to-end LoRA training path |
| `exp0024` | batch fit test `64` | successful |
| `exp0025` | batch fit test `128` | successful |
| `exp0026` | batch fit test `256` | successful |
| `exp0027` | full pilot LoRA run | completed; best epoch `10` |

### Main quantitative results

#### ViT LoRA result

| Model | D0 val AUROC | D0 test AUROC | D1 transfer AUROC | D2 transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| ViT LoRA (`exp0027`) | `0.7938` | `0.7819` | `0.7299` | `0.4997` |

#### Comparison against the existing image-only baselines

| Model | D0 test AUROC | D1 transfer AUROC | D2 transfer AUROC |
| --- | ---: | ---: | ---: |
| ResNet50 linear (`exp0013`) | `0.7306` | `0.7218` | `0.4996` |
| CXR Foundation linear (`exp0015`) | `0.8455` | `0.8454` | `0.5007` |
| ViT LoRA (`exp0027`) | `0.7819` | `0.7299` | `0.4997` |

#### Delta view

- Compared with `ResNet50` linear:
  - `D0 test`: `+0.0513`
  - `D1 transfer`: `+0.0081`
  - `D2 transfer`: `+0.0001`
- Compared with `CXR Foundation` linear:
  - `D0 test`: `-0.0636`
  - `D1 transfer`: `-0.1155`
  - `D2 transfer`: `-0.0010`

### Key interpretation

The LoRA pilot was useful because it tested a genuine backbone-adaptation step rather than only changing the classifier head.

- The ViT LoRA model clearly beat the old `ResNet50` linear source baseline on `NIH test`.
- It also slightly improved `CheXpert` transfer relative to `ResNet50`.
- However, it remained far behind `CXR Foundation` on both `NIH` and `CheXpert`.
- It did not change the `MIMIC` story; transfer stayed effectively at chance overall.

So the most defensible interpretation is:

- generic vision-transformer LoRA is a meaningful improvement over a basic frozen ResNet baseline
- but chest-X-ray-specific pretraining still matters more than generic LoRA adaptation in this pilot
- and neither approach alone solves the `NIH -> MIMIC` domain gap

### Supervisor-facing conclusion

This LoRA result strengthens the overall narrative rather than weakening it.

- We did not assume that the gain from `CXR Foundation` came only from having a slightly more flexible head.
- We also tested a parameter-efficient backbone adaptation path using a standard vision transformer with LoRA.
- That LoRA path improved over the simple ResNet baseline, which is useful to show.
- But it still did not approach the `CXR Foundation` transfer result.

That supports a clear supervisor-facing message:

1. simple frozen ResNet is a weak starting point
2. generic backbone adaptation with LoRA helps somewhat
3. chest-X-ray-specific pretrained backbones still provide the strongest direct-transfer performance
4. the hardest remaining problem is still actual target adaptation, especially for `MIMIC`

### Recommendation

The best image-only default remains:

- `CXR Foundation` backbone
- `linear` source-trained head

The ViT LoRA branch should be kept as a useful comparison point:

- it is stronger than the old ResNet baseline
- it is weaker than `CXR Foundation`
- it does not solve `MIMIC`

### Key artifact locations for presentation

- Full pilot LoRA run:
  - `/workspace/experiments/complete/exp0027__domain_transfer_lora_training__vit_base_patch16_224_in21k_pilot5h`
- Full pilot log:
  - `/workspace/logs/exp0027__domain_transfer_lora_training__vit_base_patch16_224_in21k_pilot5h.log`

All statements below are derived from the local experiment artifacts only, including the original fused-lineage artifacts (`exp0007`-`exp0012`), the April 5, 2026 rerun and retrieval-refinement artifacts (`exp0013`-`exp0026`), and the April 5, 2026 fusion-weight sweep artifacts (`exp0027`-`exp0046`). No external sources were needed for this write-up. Metric values are reproduced exactly as reported in the experiment summaries and supporting artifacts.

## 1. Key findings

- Under the shared frozen-linear baseline protocol, the fused representation was the strongest of the three candidate embedding families. Its macro AUROC was `0.763730` on validation and `0.767933` on test, compared with `0.745876` and `0.746067` for image-only, and `0.698718` and `0.703594` for report-only.
- The fused baseline also led the other baselines on the other reported macro metrics. On test, it achieved macro average precision `0.152244`, macro ECE `0.347454`, macro `F1 @ 0.5` `0.175678`, and macro `F1 @ tuned thresholds` `0.209533`, all stronger than the image-only and report-only counterparts under the same evaluation setup.
- The margin over image-only was clear within this realized pipeline. Relative to image-only, the fused baseline improved test macro AUROC by `+0.021867`, test macro average precision by `+0.021729`, and test macro `F1 @ tuned thresholds` by `+0.024965`.
- Retrieval used as a memory-only predictor was clearly weaker than the fused supervised baseline. Validation selected `k=50` and `tau=1`, but the resulting memory-only system reached only `0.691239` macro AUROC and `0.130994` macro average precision on test, versus `0.767933` and `0.152244` for the fused baseline.
- Memory-only retrieval did show much lower macro ECE on test (`0.008359`) than the fused baseline (`0.347454`), but this coincided with extremely weak macro `F1 @ 0.5` (`0.015123`). That pattern does not support describing memory-only retrieval as simply better overall.
- Probability mixing between the fused baseline and retrieval produced a small macro-level improvement over the fused baseline alone. Validation selected `alpha=0.7`, with reported gains of `+0.000531` in macro AUROC and `+0.005153` in macro average precision relative to the fused baseline. On test, the mixed system achieved macro AUROC `0.768586`, macro average precision `0.153391`, macro ECE `0.242982`, macro `F1 @ 0.5` `0.204582`, and macro `F1 @ frozen val thresholds` `0.210987`. As reported in `exp0012`, the deltas versus the frozen baseline were `+0.000651` AUROC, `+0.001154` average precision, `-0.104474` ECE, `+0.028907` `F1 @ 0.5`, and `+0.001397` `F1 @ frozen val thresholds`.
- The mixed-model gains were not uniform across labels. On test, per-label AUROC improved for `10/14` labels and decreased for `4/14`; per-label average precision also improved for `10/14` labels and decreased for `4/14`. The macro improvement therefore should not be presented as a universal label-level gain.

## 2. Uncertainty, limitations, and cautions

- These results describe one realized source-stage pipeline on one fixed train/validation/test split of NIH CXR14 (`78,571` train, `11,219` val, `22,330` test). They do not by themselves establish robustness across seeds, resampling, or alternative splits.
- The baseline training artifacts use a single training seed, and the retrieval stages use a single realized validation-selection path. There are no confidence intervals, hypothesis tests, or multi-seed averages in the current evidence.
- The reported conclusions are limited to the active source-stage pipeline (`exp0004`-`exp0012`). They do not establish performance for any broader domain-adaptation stage beyond these artifacts.
- Threshold-tuned validation F1 should be treated as diagnostic only, because thresholds are selected on the same validation split used for reporting those validation summaries.
- On test, threshold-based F1 is frozen from validation for each method, but the threshold source differs by method: the fused baseline uses its own validation thresholds, memory-only uses the `exp0009` validation thresholds, and the mixed model uses the `exp0010` validation thresholds. For direct cross-method comparison, `F1 @ 0.5` is the cleaner threshold-controlled summary.
- Lower macro ECE should be described literally as lower macro ECE, not automatically as better calibration in a broader sense. In particular, the memory-only system combines very low ECE with near-zero `F1 @ 0.5`, so ECE should not be interpreted in isolation.
- The mixed model’s macro gain is small on discrimination metrics and not uniform across labels. Claims of broad superiority, strong robustness, or generalization beyond this setup would therefore be overstated.

## 3. Polished results write-up

The active source-stage experiments were conducted on NIH CXR14 using frozen image embeddings, frozen report embeddings, and their fused representation. The image branch produced `2048`-dimensional ResNet50 features, the report branch produced `128`-dimensional BiomedVLP CXR-BERT features, and the fused branch concatenated these into a `2176`-dimensional L2-normalized representation. A shared frozen linear multilabel probe was then trained separately on image-only, report-only, and fused embeddings, with validation macro AUROC used for checkpoint selection.

Within this common probe setting, the fused representation was the strongest of the three candidate baselines. It achieved validation macro AUROC `0.763730` and test macro AUROC `0.767933`, compared with `0.745876` and `0.746067` for image-only, and `0.698718` and `0.703594` for report-only. The fused baseline was also strongest on the other reported macro summaries, including average precision, macro ECE, `F1 @ 0.5`, and threshold-based F1. These results support selecting the fused representation as the canonical branch for downstream retrieval analysis within this source-stage pipeline.

Retrieval alone did not match the fused supervised baseline. The validation sweep over retrieval hyperparameters selected `k=50` and `tau=1`, and validation macro AUROC increased steadily with larger `k` at `tau=1`, from `0.529594` at `k=1` to `0.682669` at `k=50`. However, the final memory-only system remained substantially weaker than the fused baseline on held-out test data, with macro AUROC `0.691239` versus `0.767933`, macro average precision `0.130994` versus `0.152244`, and macro `F1 @ 0.5` `0.015123` versus `0.175678`. Although the memory-only model showed much lower macro ECE (`0.008359`), that result should be interpreted cautiously rather than as evidence that retrieval alone is preferable, because its fixed-threshold classification performance was weak.

Probability mixing provided a more favorable use of retrieval than memory-only prediction. Validation selected `alpha=0.7`, yielding a small increase in macro AUROC over the fused baseline (`0.764262` versus `0.763731`) and a larger increase in macro average precision (`0.156619` versus `0.151466`). On test, the mixed model again showed a modest macro-level improvement over the fused baseline, reaching macro AUROC `0.768586` and macro average precision `0.153391`, while also reducing macro ECE to `0.242982` and increasing macro `F1 @ 0.5` to `0.204582`. The most defensible interpretation is therefore not that retrieval replaces the supervised model, but that, within this specific frozen source-stage pipeline, retrieval contributes complementary signal when blended with the fused baseline, with a modest effect size on discrimination metrics and non-uniform label-level behavior.

## 4. Presentation recommendations

- Present the results in the following order: `Table 1` baseline comparison, one short paragraph explaining why fused was selected; `Table 2` retrieval augmentation comparison, one short paragraph explaining why memory-only is insufficient but mixing is still worth reporting; `Figure 1` validation alpha sweep; appendix tables for the full retrieval sweep and per-label deltas.
- Keep the main claim narrow: fused is the strongest representation family under the frozen-linear baseline protocol, memory-only retrieval is weaker than the fused baseline, and probability mixing yields a small improvement over the fused baseline on macro test metrics.
- Use the phrase `lower macro ECE` rather than `better calibrated` unless you add a separate calibration analysis.
- When discussing threshold-based F1 in captions or text, note that thresholds are validation-frozen but method-specific. For direct cross-method comparison, emphasize `F1 @ 0.5`.

### Suggested Table 1 Caption

Comparison of frozen linear multilabel baselines trained on image-only, report-only, and fused embeddings. The canonical branch is selected by validation macro AUROC; average precision, macro ECE, and F1 summaries are reported as supporting metrics.

### Suggested Table 1A. Validation Baseline Comparison

| Representation | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Macro F1 @ tuned thresholds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image-only | `0.745876` | `0.133964` | `0.358816` | `0.161989` | `0.199989` |
| Report-only | `0.698718` | `0.123513` | `0.394832` | `0.159970` | `0.191944` |
| Fused | `0.763730` | `0.151467` | `0.347225` | `0.175942` | `0.222298` |

### Suggested Table 1B. Test Baseline Comparison

| Representation | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Macro F1 @ tuned thresholds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image-only | `0.746067` | `0.130515` | `0.358840` | `0.160943` | `0.184568` |
| Report-only | `0.703594` | `0.129127` | `0.394382` | `0.159969` | `0.191192` |
| Fused | `0.767933` | `0.152244` | `0.347454` | `0.175678` | `0.209533` |

### Suggested Table 2 Caption

Test-set comparison of the selected fused baseline, memory-only retrieval, and probability mixing. Retrieval hyperparameters (`k=50`, `tau=1`) and the mixing weight (`alpha=0.7`) are selected on validation only. Threshold-based F1 values are validation-frozen but method-specific.

### Suggested Table 2. Retrieval Augmentation Comparison on Test

| Method | Selected config | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Macro F1 @ frozen val thresholds | Delta AUROC vs fused | Delta AP vs fused | Delta ECE vs fused |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fused baseline | supervised only | `0.767933` | `0.152244` | `0.347454` | `0.175678` | `0.209533` | `0.000000` | `0.000000` | `0.000000` |
| Memory-only | `k=50`, `tau=1` | `0.691239` | `0.130994` | `0.008359` | `0.015123` | `0.188889` | `-0.076694` | `-0.021250` | `-0.339095` |
| Mixed | `alpha=0.7` | `0.768586` | `0.153391` | `0.242982` | `0.204582` | `0.210987` | `+0.000651` | `+0.001154` | `-0.104474` |

Note: table entries are shown to six decimals, matching the summary artifacts. The mixed-model delta columns reproduce the reported `exp0012` deltas, so subtracting the displayed rounded absolute metrics may differ at the sixth decimal place.

### Suggested Figure 1

- Plot the validation alpha sweep from `exp0010`.
- Use `alpha` on the x-axis and macro AUROC on the y-axis.
- If space allows, add a second panel for macro average precision or macro ECE.

### Suggested Figure 1 Caption

Validation sweep for probability mixing between the fused supervised baseline and the memory-only retrieval model. The best setting is `alpha=0.7`, but the optimum is shallow rather than dramatic, indicating a modest preference for mixing over the baseline alone in this realized pipeline.

### Suggested Appendix Items

- `Appendix Table A1`: full `k`/`tau` sweep from `exp0009`, mainly to show that the retrieval result improves with larger `k` but remains well below the fused supervised baseline.
- `Appendix Table A2`: per-label mixed-versus-fused deltas on test, mainly to show that the macro gain is heterogeneous rather than universal.

## 5. Missing information or targeted follow-up questions

- No additional information is required to present the current source-stage results accurately to a supervisor.
- If you later want this adapted for a paper or slides, the only optional additions that would materially change the framing are a one-sentence project objective and any supervisor preference about whether calibration should stay in the main text or move to the appendix.

## 6. 2026-04-05 Rerun Update: 100-Epoch Fused-Baseline Rerun From `exp0003`

This addendum records a downstream rerun that started from the existing fused embedding root `exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2` and reran the source-stage pipeline from the baseline-training stage onward. No image embeddings or report embeddings were regenerated for this update.

### Rerun scope and experiment lineage

- `exp0013`: fused baseline rerun with `--epochs 100` and `--patience 4`
- `exp0014`: rebuilt train retrieval memory from the same fused `exp0003` embeddings
- `exp0015`: validation memory-only sweep
- `exp0016`: validation probability-mixing sweep
- `exp0017`: frozen memory-only test evaluation
- `exp0018`: frozen probability-mixing test evaluation

### What changed relative to the original fused lineage

- The upstream fused embeddings were unchanged. The rerun began from the existing `2176`-dimensional fused representation in `exp0003`.
- The main modeling change was the supervised baseline training budget. The original fused baseline in `exp0006` used `30` epochs; the rerun in `exp0013` used `100` epochs with `patience=4`.
- Because the embedding root was unchanged, the retrieval-only validation selection stayed effectively the same. The rerun again selected `k=50` and `tau=1` in `exp0015`.
- Probability mixing also kept the same validation-selected mixing weight. The rerun again selected `alpha=0.7` in `exp0016`.

### Comparison: original fused baseline vs 100-epoch fused baseline

| Metric | Original fused baseline (`exp0006`) | Rerun fused baseline (`exp0013`) | Delta |
| --- | ---: | ---: | ---: |
| Validation macro AUROC | `0.763730` | `0.774447` | `+0.010718` |
| Validation macro AP | `0.151467` | `0.159506` | `+0.008039` |
| Test macro AUROC | `0.767933` | `0.774642` | `+0.006708` |
| Test macro AP | `0.152244` | `0.160414` | `+0.008170` |

The rerun baseline improved materially over the original fused baseline on both validation and test discrimination metrics. The best epoch in the rerun was `100`, so the longer budget continued to help throughout the full run rather than stopping early at a much earlier checkpoint.

### Comparison: original mixed test result vs rerun mixed test result

| Metric | Original mixed test (`exp0012`) | Rerun mixed test (`exp0018`) | Delta |
| --- | ---: | ---: | ---: |
| Test macro AUROC | `0.768586` | `0.775239` | `+0.006653` |
| Test macro AP | `0.153391` | `0.159950` | `+0.006559` |
| Test macro ECE | `0.242982` | `0.227412` | `-0.015570` |
| Test macro F1 @ 0.5 | `0.204582` | `0.211526` | `+0.006944` |
| Test macro F1 @ frozen val thresholds | `0.210987` | `0.220846` | `+0.009859` |

The final mixed model therefore improved in absolute terms over the original `exp0012` result across all of the main reported macro summaries except that ECE decreased, which is favorable because lower macro ECE is better when stated literally as calibration error.

### Memory-only retrieval comparison

The memory-only branch was essentially unchanged because it depends on the same `exp0003` embeddings and the same validation-selected retrieval setting. The rerun again selected `k=50`, `tau=1`, and produced test macro AUROC `0.691239` and test macro AP `0.130995`, which is effectively identical to the original `exp0011` values up to rounding.

### Interpretation of the rerun

The strongest effect in the rerun came from improving the supervised fused baseline, not from changing the retrieval configuration. The new baseline in `exp0013` is clearly stronger than the original `exp0006`, and the final mixed system in `exp0018` is clearly stronger than the original `exp0012` in absolute terms.

However, the relationship between the rerun baseline and the rerun mixed model is more nuanced than in the original lineage. In the rerun, the mixed model still achieved a small test macro AUROC gain over the rerun baseline (`+0.000597`), but it was slightly lower on test macro average precision (`-0.000464`) than the rerun baseline alone. In other words, after strengthening the fused supervised baseline with the 100-epoch run, mixing still gave the best test AUROC, but no longer gave the best test AP.

The narrowest defensible supervisor-facing summary for the rerun is therefore:

- the 100-epoch rerun substantially strengthened the fused supervised baseline relative to the original 30-epoch baseline,
- the final mixed system also improved over the earlier mixed system in absolute terms,
- retrieval-only behavior was unchanged because the embedding root did not change,
- and in the rerun, mixing remained slightly best for test macro AUROC, while the rerun baseline alone was slightly best for test macro AP.

## 7. 2026-04-05 Expanded Retrieval-Sweep Update: Boundary Extension From The 100-Epoch Lineage

This addendum records a second downstream rerun that kept the same upstream ingredients as the `exp0013` to `exp0018` branch and changed only the retrieval-selection search space. The fused embedding root remained `exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`, and the canonical supervised baseline remained the 100-epoch fused run `exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`. No image embeddings, report embeddings, fusion recipe, or baseline-training budget were changed for this update.

### Expanded-sweep scope and experiment lineage

- `exp0019`: rebuilt the train retrieval memory from the same `exp0003` fused embeddings, aligned to the `exp0013` lineage
- `exp0020`: first expanded validation memory-only sweep over `k = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000]` and `tau = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 20, 40]`
- `exp0021`: refined validation memory-only sweep over the large-`k`, high-`tau` frontier with `k = [50, 100, 200, 500, 1000, 1500, 2000, 3000]` and `tau = [5, 10, 20, 40, 80, 160]`
- `exp0022`: validation probability-mixing sweep on top of `exp0021`
- `exp0023`: frozen memory-only test evaluation using the validation-selected `k` and `tau` from `exp0021`
- `exp0024`: frozen probability-mixing test evaluation using the validation-selected `alpha` from `exp0022`

### What changed relative to `exp0015` to `exp0018`

- The retrieval framework was preserved: FAISS nearest-neighbor lookup, the same probability construction from neighbor labels, validation-only hyperparameter selection, and frozen test application.
- The implementation of `06_evaluate_source_memory_only.py` was updated so that the sweep grid can be passed explicitly and `tau` is preserved as a float rather than being serialized through integer-only handling. This allowed direct testing of `tau < 1`.
- The first widened pass in `exp0020` already showed that the earlier `exp0015` optimum was boundary-limited: the best setting moved from `k=50, tau=1` to `k=1000, tau=40`.
- The refined pass in `exp0021` pushed the selected memory-only validation setting further to `k=3000, tau=40`.
- Despite explicitly testing smaller `tau` values than before, the validation winner did not move toward lower `tau`; in the expanded regime, the best completed setting used a much larger `k` and a relatively sharp `tau`.
- The validation-selected mixing weight remained `alpha=0.7` in `exp0022`.

### Comparison: 100-epoch retrieval branch before and after the expanded sweep

#### Memory-only validation selection

| Metric | Prior 100-epoch branch (`exp0015`) | Expanded-sweep branch (`exp0021`) | Delta |
| --- | ---: | ---: | ---: |
| Selected `k` | `50` | `3000` | `+2950` |
| Selected `tau` | `1` | `40` | `+39` |
| Validation macro AUROC | `0.682669` | `0.738898` | `+0.056230` |
| Validation macro AP | `0.128678` | `0.136206` | `+0.007527` |
| Validation macro ECE | `0.009410` | `0.006554` | `-0.002857` |
| Validation macro F1 @ 0.5 | `0.017704` | `0.001290` | `-0.016414` |

#### Memory-only test comparison

| Metric | Prior 100-epoch branch (`exp0017`) | Expanded-sweep branch (`exp0023`) | Delta |
| --- | ---: | ---: | ---: |
| Frozen `k` | `50` | `3000` | `+2950` |
| Frozen `tau` | `1` | `40` | `+39` |
| Test macro AUROC | `0.691239` | `0.745989` | `+0.054750` |
| Test macro AP | `0.130995` | `0.138152` | `+0.007157` |
| Test macro ECE | `0.008359` | `0.005927` | `-0.002432` |
| Test macro F1 @ 0.5 | `0.015123` | `0.000485` | `-0.014638` |
| Test macro F1 @ frozen val thresholds | `0.188889` | `0.191466` | `+0.002577` |

#### Probability-mixing validation comparison

| Metric | Prior 100-epoch branch (`exp0016`) | Expanded-sweep branch (`exp0022`) | Delta |
| --- | ---: | ---: | ---: |
| Selected `alpha` | `0.7` | `0.7` | `0.0` |
| Validation macro AUROC | `0.774842` | `0.774656` | `-0.000186` |
| Validation macro AP | `0.163146` | `0.160150` | `-0.002996` |
| Validation macro ECE | `0.227235` | `0.226923` | `-0.000311` |
| Validation macro F1 @ 0.5 | `0.211561` | `0.211274` | `-0.000287` |

#### Probability-mixing test comparison

| Metric | Prior 100-epoch branch (`exp0018`) | Expanded-sweep branch (`exp0024`) | Delta |
| --- | ---: | ---: | ---: |
| Frozen `alpha` | `0.7` | `0.7` | `0.0` |
| Test macro AUROC | `0.775239` | `0.775046` | `-0.000193` |
| Test macro AP | `0.159950` | `0.159121` | `-0.000829` |
| Test macro ECE | `0.227412` | `0.227055` | `-0.000357` |
| Test macro F1 @ 0.5 | `0.211526` | `0.211121` | `-0.000405` |
| Test macro F1 @ frozen val thresholds | `0.220846` | `0.221494` | `+0.000648` |

### Interpretation of the expanded sweep

The expanded search confirms that the earlier retrieval-only selection in `exp0015` was constrained by the original search boundary. Under the same frozen embedding root and the same 100-epoch fused baseline lineage, widening the sweep changed the memory-only validation winner from `k=50, tau=1` to `k=3000, tau=40`. That shift materially improved memory-only ranking metrics on both validation and test. The strongest headline gain is the memory-only test macro AUROC increase from `0.691239` to `0.745989`, with test macro AP increasing from `0.130995` to `0.138152`.

However, the expanded retrieval setting still does not make memory-only prediction a strong standalone decision rule. Its thresholded classification behavior remained poor and in some respects worsened: macro `F1 @ 0.5` on test fell from `0.015123` in `exp0017` to `0.000485` in `exp0023`, even though AUROC and AP improved. The most defensible reading is therefore that the broader retrieval sweep improved ranking quality and reduced reported macro ECE, but did not convert the retrieval-only branch into a practically useful fixed-threshold classifier.

The more important supervisor-facing result is that the broader retrieval sweep did not improve the final mixed system relative to the prior 100-epoch branch. The validation-selected mixing weight stayed at `alpha=0.7`, but both validation and test macro AUROC and macro AP for the mixed branch were slightly lower in `exp0022` and `exp0024` than in `exp0016` and `exp0018`. On held-out test data, the mixed model moved from macro AUROC `0.775239` and macro AP `0.159950` in `exp0018` to `0.775046` and `0.159121` in `exp0024`. The expanded-sweep mixed model still showed a small AUROC edge over the `exp0013` supervised baseline, but that edge remained narrow, and test macro AP remained below the baseline.

The narrowest defensible supervisor-facing summary for this update is therefore:

- widening the retrieval sweep beyond the original `k=50, tau=1` boundary materially improved memory-only AUROC and AP,
- the best completed memory-only setting moved to `k=3000, tau=40`, not to a smaller `tau`,
- the memory-only branch remained weak for fixed-threshold classification despite better ranking metrics,
- and the expanded retrieval sweep did not improve the final mixed system over the earlier 100-epoch branch in `exp0018`.

### Production recommendation for the main branch

For the production main branch, the retrieval configuration should remain the earlier 100-epoch setting selected in `exp0015`, namely `k=50` and `tau=1`, with `alpha=0.7` from `exp0016` for the mixed model. The reason is that the expanded retrieval sweep improved the retrieval-only branch in isolation, but it did not improve the final mixed system that is most relevant for deployment. Relative to the prior 100-epoch production branch in `exp0018`, the expanded-sweep mixed branch in `exp0024` was slightly lower on test macro AUROC (`0.775046` versus `0.775239`), test macro AP (`0.159121` versus `0.159950`), and test macro `F1 @ 0.5` (`0.211121` versus `0.211526`). The clearest supervisor-facing recommendation is therefore to keep the production retrieval configuration at `k=50`, `tau=1`, and `alpha=0.7`, while treating the expanded-sweep results as an analysis showing that stronger retrieval-only ranking performance does not necessarily translate into a better final mixed system.

One remaining caveat is that the selected memory-only validation setting in `exp0021` still lies on the tested `k` ceiling of `3000`. That means the retrieval frontier is not fully closed on `k`, even though the completed expanded sweep is already sufficient to show that the original `exp0015` selection was boundary-limited and that further retrieval-only improvement does not automatically translate into a better mixed system.

## 8. 2026-04-05 Fine Alpha-Sweep Update: Local Refinement On The 100-Epoch Production Branch

This addendum records a targeted follow-up run that kept the 100-epoch production retrieval branch fixed and refined only the validation-time probability-mixing weight. The upstream fused embedding root remained `exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`, the supervised baseline remained `exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`, and the frozen retrieval configuration remained the original 100-epoch production setting from `exp0015`, namely `k=50` and `tau=1`. No image embeddings, report embeddings, fusion recipe, baseline-training budget, or retrieval hyperparameters were changed for this update.

### Fine-alpha scope and experiment lineage

- `exp0025`: validation probability-mixing sweep on top of the original 100-epoch production retrieval branch, using the same memory-only validation artifact as `exp0016` but with a finer alpha grid
- `exp0026`: frozen probability-mixing test evaluation using the validation-selected alpha from `exp0025`

### What changed relative to `exp0016` to `exp0018`

- The retrieval inputs were held fixed at the earlier 100-epoch production configuration: `exp0015` for validation memory probabilities and `exp0017` for frozen test memory probabilities.
- The probability-mixing script was updated to accept an explicit alpha grid so that the selection stage no longer had to be restricted to `alpha = 0.0, 0.1, ..., 1.0`.
- The fine validation sweep in `exp0025` evaluated `alpha` on the full grid `0.00, 0.01, ..., 1.00` while preserving the same validation-only selection rule as before: highest macro AUROC, then higher macro AP, then larger alpha.
- No tuning was performed on test. `exp0026` simply applied the validation-selected alpha from `exp0025` to the held-out test split.

### Fine-alpha validation results

The coarse 100-epoch production branch in `exp0016` selected `alpha=0.7`. The finer sweep in `exp0025` moved the validation-selected alpha slightly upward to `0.74`.

#### Local validation ridge around the previous optimum

| Alpha | Validation macro AUROC | Validation macro AP |
| --- | ---: | ---: |
| `0.70` | `0.774842` | `0.163146` |
| `0.71` | `0.774849` | `0.163114` |
| `0.72` | `0.774856` | `0.163099` |
| `0.73` | `0.774863` | `0.163059` |
| `0.74` | `0.774865` | `0.162973` |
| `0.75` | `0.774864` | `0.163048` |

This confirms that the optimum is real but extremely shallow. Relative to the earlier `alpha=0.7` choice, the validation macro AUROC gain at `alpha=0.74` is only `+0.000023`.

#### Comparison: coarse alpha selection vs fine alpha selection on validation

| Metric | Prior 100-epoch branch (`exp0016`, `alpha=0.7`) | Fine-alpha branch (`exp0025`, `alpha=0.74`) | Delta |
| --- | ---: | ---: | ---: |
| Selected `alpha` | `0.70` | `0.74` | `+0.04` |
| Validation macro AUROC | `0.774842` | `0.774865` | `+0.000023` |
| Validation macro AP | `0.163146` | `0.162973` | `-0.000173` |
| Validation macro ECE | `0.227235` | `0.240306` | `+0.013071` |
| Validation macro F1 @ 0.5 | `0.211561` | `0.208013` | `-0.003548` |
| Diagnostic macro F1 @ tuned thresholds | `0.234437` | `0.234481` | `+0.000044` |

### Frozen test comparison

The fine-alpha branch was then frozen and evaluated on held-out test data in `exp0026`.

| Metric | Prior 100-epoch branch (`exp0018`, `alpha=0.7`) | Fine-alpha branch (`exp0026`, `alpha=0.74`) | Delta |
| --- | ---: | ---: | ---: |
| Frozen `alpha` | `0.70` | `0.74` | `+0.04` |
| Test macro AUROC | `0.775239` | `0.775224` | `-0.000014` |
| Test macro AP | `0.159950` | `0.160043` | `+0.000092` |
| Test macro ECE | `0.227412` | `0.240454` | `+0.013041` |
| Test macro F1 @ 0.5 | `0.211526` | `0.209309` | `-0.002217` |
| Test macro F1 @ frozen val thresholds | `0.220846` | `0.220639` | `-0.000207` |

### Interpretation of the fine-alpha update

The fine sweep shows that `alpha=0.7` was not the exact validation optimum on the 100-epoch production branch. Under the same frozen retrieval setting `k=50, tau=1`, the validation-selected alpha shifted slightly to `0.74` in `exp0025`.

However, the practical effect size is negligible. The improvement at validation time is only on the order of `2e-05` macro AUROC, and the held-out test comparison does not show a meaningful global win for the refined alpha. On test, `alpha=0.74` produced slightly higher macro AP than `alpha=0.7` (`0.160043` versus `0.159950`), but slightly lower macro AUROC (`0.775224` versus `0.775239`), worse macro ECE (`0.240454` versus `0.227412`), and slightly lower threshold-based F1.

The narrowest defensible supervisor-facing summary for this update is therefore:

- the original 100-epoch production choice `alpha=0.7` was near-optimal but not exactly optimal on the coarse validation grid,
- a finer sweep moved the validation-selected alpha to `0.74`,
- the local optimum around `0.7` to `0.75` is extremely shallow,
- and the refined alpha did not produce a clear test-set improvement over the original `alpha=0.7` production branch.

### Recommendation after the fine-alpha check

If the goal is strict adherence to validation-only selection, then `exp0025` supports using `alpha=0.74` as the updated validation winner on the original 100-epoch production retrieval branch. If the goal is stable production reporting with minimal complexity, `alpha=0.7` remains fully defensible because the fine-alpha sweep showed only a negligible validation gain and no convincing test-set improvement.

## 9. 2026-04-05 Fusion-Weight Sweep Update: Downweighting The Text Branch Before Final Normalization

This addendum records a targeted rerun that kept the fusion recipe fixed but swept the text-branch contribution before final L2 normalization. The copied upstream image and report embedding roots were reused directly: `exp0001__embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test` for image embeddings and `exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test` for report embeddings. Before execution, the copied `train`, `val`, and `test` embedding artifacts for both branches were explicitly checked to be present and readable, so no upstream image-embedding or report-embedding generation was rerun for this update.

### Fusion-weight sweep scope and experiment lineage

- `exp0027`, `exp0029`, `exp0031`, `exp0033`, `exp0035`, `exp0037`: fused embedding generations with image weight fixed at `1.0` and text weights `0.50`, `0.75`, `1.00`, `1.25`, `1.50`, and `2.00`
- `exp0028`, `exp0030`, `exp0032`, `exp0034`, `exp0036`, `exp0038`: corresponding 100-epoch frozen linear baselines for the six fusion settings
- `exp0039` to `exp0041`: retrieval-memory build, validation memory-only sweep, and fine alpha sweep for the validation-leading candidate `text weight = 0.50`
- `exp0042` to `exp0044`: the same retrieval and mixing evaluation path for the runner-up candidate `text weight = 0.75`
- `exp0045` and `exp0046`: frozen test evaluation for the validation-selected winner only

### What changed relative to the prior equal-weight branch

- The original fused embedding root `exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2` used equal branch weights. Its metadata records `image.weight = 1.0` and `report.weight = 1.0`.
- In this sweep, the image branch was held fixed at `1.0` and only the text branch weight was changed.
- The fusion recipe itself was unchanged: concatenation, image-referenced row alignment, and final L2 normalization.
- The downstream supervised baseline protocol was unchanged from the 100-epoch production line: `100` epochs with `patience = 4`.
- To keep cost moderate, the retrieval and probability-mixing stages were run only for the top two weights from the baseline screen, namely `0.50` and `0.75`.
- The retrieval validation grid remained the moderate production grid `k = [1, 3, 5, 10, 20, 50]` and `tau = [1, 5, 10, 20, 40]`, and the mixing stage again used the fine alpha grid `0.00, 0.01, ..., 1.00`.

### Stage-1 baseline sweep results

| Text weight | Validation macro AUROC | Validation macro AP | Test macro AUROC | Test macro AP |
| --- | ---: | ---: | ---: | ---: |
| `0.50` | `0.775530` | `0.160135` | `0.774630` | `0.160831` |
| `0.75` | `0.775114` | `0.159902` | `0.774719` | `0.160710` |
| `1.00` | `0.774447` | `0.159506` | `0.774642` | `0.160414` |
| `1.25` | `0.773663` | `0.158935` | `0.774430` | `0.160028` |
| `1.50` | `0.772827` | `0.158282` | `0.774126` | `0.159587` |
| `2.00` | `0.771124` | `0.157103` | `0.773377` | `0.158557` |

The baseline sweep was monotonic on validation: as the text branch was weighted more heavily, validation macro AUROC and validation macro AP both decreased. The equal-weight setting `text = 1.0` reproduced the existing 100-epoch equal-weight baseline almost exactly, which makes the sweep directly comparable to the prior production branch. Within this realized pipeline, the baseline evidence therefore supports a weaker text branch than the original equal-weight choice.

### Finalist validation comparison after retrieval and probability mixing

| Text weight | Selected `k` | Selected `tau` | Selected `alpha` | Validation mixed macro AUROC | Validation mixed macro AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.50` | `50` | `1` | `0.74` | `0.775935` | `0.161408` |
| `0.75` | `50` | `1` | `0.77` | `0.775439` | `0.161926` |

The validation winner on the primary selection metric was therefore `text weight = 0.50`. The runner-up `0.75` achieved slightly higher validation macro AP, but it was lower on the primary selection metric of validation macro AUROC.

### Comparison against the prior equal-weight branch

#### Baseline comparison: prior equal-weight branch vs sweep winner

| Metric | Prior equal-weight baseline (`exp0013`, text weight `1.0`) | Sweep winner baseline (`exp0028`, text weight `0.50`) | Delta |
| --- | ---: | ---: | ---: |
| Validation macro AUROC | `0.774447` | `0.775530` | `+0.001083` |
| Validation macro AP | `0.159506` | `0.160135` | `+0.000629` |
| Test macro AUROC | `0.774642` | `0.774630` | `-0.000012` |
| Test macro AP | `0.160414` | `0.160831` | `+0.000417` |

#### Final mixed comparison: prior equal-weight branch vs sweep winner

| Metric | Prior equal-weight mixed branch (`exp0026`, text weight `1.0`) | Sweep winner mixed branch (`exp0046`, text weight `0.50`) | Delta |
| --- | ---: | ---: | ---: |
| Validation macro AUROC | `0.774865` | `0.775935` | `+0.001070` |
| Validation macro AP | `0.162973` | `0.161408` | `-0.001565` |
| Test macro AUROC | `0.775224` | `0.774877` | `-0.000347` |
| Test macro AP | `0.160043` | `0.160280` | `+0.000237` |

### Interpretation of the fusion-weight sweep

The sweep gave a clear sensitivity result even though it did not produce a decisive new production winner. The strongest and most stable pattern is that equal weighting was not validation-optimal in this pipeline. On the baseline screen, reducing the text weight below `1.0` improved validation performance, and the best validation baseline occurred at `text weight = 0.50`. That pattern survived into the finalist retrieval-and-mixing stage, where `text weight = 0.50` also won on validation macro AUROC.

However, the held-out test result is more mixed than the validation result. Relative to the prior equal-weight mixed branch, the new `text weight = 0.50` branch was slightly lower on test macro AUROC (`0.774877` versus `0.775224`) but slightly higher on test macro AP (`0.160280` versus `0.160043`). The effect sizes are very small in both directions. The narrowest defensible interpretation is therefore that the sweep usefully established that the text branch should probably be somewhat weaker than the image branch in this concat-fusion setup, but it did not yield a clear held-out mixed-model improvement over the prior equal-weight production branch.

### Recommendation after the fusion-weight sweep

If the goal is strict adherence to validation-only model selection, then the sweep supports using `text weight = 0.50`, with `k = 50`, `tau = 1`, and `alpha = 0.74`, as the validation-selected winner of this branch. If the goal is conservative supervisor-facing reporting, the safer statement is that the sweep was informative rather than transformative: it showed that equal weighting was not optimal on validation, but it did not produce a decisive test-set improvement over the prior equal-weight mixed system. That makes the sweep worth presenting as a targeted fusion-sensitivity analysis, not as an unambiguous replacement of the earlier branch.

## 10. 2026-04-10 Correction Addendum: Actual `exp0010`-`exp0043` Chronology

This addendum supersedes the earlier post-`exp0009` chronology used in Sections `6` to `9` of this file. The local experiment directories show that the actual sequence after `exp0009` began with the one-epoch full-pipeline cross-attention training run `exp0010` on `2026-04-10`, then progressed through export, frozen-baseline, retrieval, longer cross-attention, hybrid concat, and gated-hybrid branches. All dates and times below are reported in UTC, and all metric values are copied from local artifacts only.

### Actual chronology overview

| Experiments | UTC timestamp(s) | Main action | Main output |
| --- | --- | --- | --- |
| `exp0010` | `2026-04-10T11:02:45Z` | one-epoch full-pipeline cross-attention training | first direct xattn classifier on the active source-stage split |
| `exp0011` to `exp0017` | `2026-04-10T11:21:05Z` to `2026-04-10T11:24:13Z` | export, frozen baseline, retrieval memory, validation selection, and frozen test evaluation | first completed xattn export-and-retrieval branch with `k=50`, `tau=20`, and `alpha=0.6` |
| `exp0018` to `exp0025` | `2026-04-10T13:57:50Z` to `2026-04-10T14:34:06Z` | longer xattn training plus the same downstream evaluation path | stronger 50-epoch-budget xattn branch with `k=50`, `tau=1`, and `alpha=0.8` |
| `exp0026` to `exp0033` | `2026-04-10T15:16:45Z` to `2026-04-10T15:49:07Z` | rebuilt image embeddings, generated hybrid concat embeddings, then reran baseline and retrieval stages | strongest completed mixed branch in the local artifacts |
| `exp0034` to `exp0035` | `2026-04-10T16:13:03Z` to `2026-04-10T16:13:19Z` | smoke gated-hybrid training and export | pipeline sanity check only, not a full-data comparison run |
| `exp0036` to `exp0043` | `2026-04-10T16:15:30Z` to `2026-04-10T17:57:48Z` | long gated-hybrid training plus downstream baseline and retrieval stages | completed gated-hybrid downstream branch, with partial training metadata for `exp0036` |

## 11. 2026-04-10 Full-Pipeline XAttn Branch: `exp0010`-`exp0017`

The first actual post-`exp0009` branch was the full-pipeline cross-attention line. `exp0010` trained the cross-attention encoder directly for one epoch, `exp0011` exported `512`-dimensional embeddings from that checkpoint, `exp0012` trained a frozen linear multilabel probe on those exported embeddings, and `exp0013` to `exp0017` evaluated retrieval-only and mixed variants on top of the exported representation.

### Branch lineage and timestamps

- `exp0010` at `2026-04-10T11:02:45Z`: direct full-pipeline cross-attention training with `epochs=1`, `patience=0`, `batch_size=32`, and `max_length=256`
- `exp0011` at `2026-04-10T11:21:05Z`: embedding export from the `exp0010` checkpoint with export batch size `160`
- `exp0012` at `2026-04-10T11:22:08Z`: frozen linear baseline trained on the exported `512`-dimensional embeddings
- `exp0013` at `2026-04-10T11:22:27Z`: FAISS train-memory build from the same `exp0011` embedding root
- `exp0014` at `2026-04-10T11:23:03Z`: memory-only validation sweep selecting `k=50` and `tau=20`
- `exp0015` at `2026-04-10T11:23:14Z`: validation probability-mixing sweep selecting `alpha=0.6`
- `exp0016` at `2026-04-10T11:24:01Z`: frozen memory-only test evaluation
- `exp0017` at `2026-04-10T11:24:13Z`: frozen mixed test evaluation

### Direct model versus frozen exported-embedding probe

| Model | Validation macro AUROC | Validation macro AP | Test macro AUROC | Test macro AP | Test macro ECE | Test macro F1 @ 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct xattn classifier (`exp0010`) | `0.718285` | `0.110125` | `0.722673` | `0.108092` | `0.386266` | `0.157584` |
| Frozen linear probe on exported xattn embeddings (`exp0012`) | `0.722824` | `0.117825` | `0.727785` | `0.111756` | `0.368609` | `0.155103` |

Relative to the direct one-epoch xattn classifier in `exp0010`, the frozen exported-embedding baseline in `exp0012` improved test macro AUROC by `+0.005111`, test macro AP by `+0.003665`, and test macro ECE by `-0.017656`, while slightly reducing test macro `F1 @ 0.5` by `-0.002480`.

### Validation-time retrieval and mixing selections

| Validation artifact | Selected config | Validation macro AUROC | Validation macro AP | Validation macro ECE | Validation macro F1 @ 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Memory-only selection (`exp0014`) | `k=50`, `tau=20` | `0.679425` | `0.111083` | `0.010496` | `0.004427` |
| Probability-mixing selection (`exp0015`) | `alpha=0.6` | `0.724728` | `0.119263` | `0.220573` | `0.141794` |

### Frozen test comparison within the branch

| Method | Frozen config | Test macro AUROC | Test macro AP | Test macro ECE | Test macro F1 @ 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen supervised baseline (`exp0012`) | supervised only | `0.727785` | `0.111756` | `0.368609` | `0.155103` |
| Memory-only retrieval (`exp0016`) | `k=50`, `tau=20` | `0.682572` | `0.107575` | `0.010428` | `0.003845` |
| Mixed model (`exp0017`) | `k=50`, `tau=20`, `alpha=0.6` | `0.729254` | `0.114845` | `0.221457` | `0.140398` |

Within this first xattn branch, the memory-only system was clearly weaker than the supervised baseline on ranking and fixed-threshold classification, despite much lower ECE. Relative to `exp0012`, the memory-only branch in `exp0016` changed test macro AUROC by `-0.045213`, macro AP by `-0.004182`, macro ECE by `-0.358181`, and macro `F1 @ 0.5` by `-0.151259`. Probability mixing in `exp0017` recovered a small ranking gain over the frozen baseline, with `+0.001470` test macro AUROC and `+0.003089` test macro AP, but it still remained below the frozen baseline on test macro `F1 @ 0.5` by `-0.014705`.

The narrowest defensible summary for the `exp0010` to `exp0017` branch is therefore:

- the full-pipeline xattn representation was viable, but the frozen exported-embedding baseline in `exp0012` was already slightly stronger than the direct one-epoch classifier in `exp0010`,
- retrieval-only prediction was not competitive as a standalone decision rule,
- and probability mixing delivered a small ranking improvement over the frozen baseline without producing the branch’s best fixed-threshold F1.

## 12. 2026-04-10 Long Cross-Attention Update: `exp0018`-`exp0025`

The next branch reran the xattn model with a substantially larger training budget. `exp0018` trained the direct cross-attention encoder with a `50`-epoch budget and `patience=5`, stopping early at best epoch `4`. `exp0019` exported embeddings from that stronger checkpoint, `exp0020` trained a frozen linear probe on the exported embeddings, and `exp0021` to `exp0025` repeated the retrieval-memory, validation-selection, and frozen-test stages.

### Branch lineage and timestamps

- `exp0018` at `2026-04-10T13:57:50Z`: long xattn training with `epochs=50`, `patience=5`, `batch_size=32`, and `max_length=256`; best epoch `4`
- `exp0019` at `2026-04-10T14:31:14Z`: embedding export from the `exp0018` checkpoint
- `exp0020` at `2026-04-10T14:32:09Z`: frozen linear baseline on exported `512`-dimensional long-xattn embeddings
- `exp0021` at `2026-04-10T14:32:29Z`: FAISS memory build for the long-xattn embedding root
- `exp0022` at `2026-04-10T14:32:59Z`: memory-only validation sweep selecting `k=50` and `tau=1`
- `exp0023` at `2026-04-10T14:33:10Z`: validation probability-mixing sweep selecting `alpha=0.8`
- `exp0024` at `2026-04-10T14:33:55Z`: frozen memory-only test evaluation
- `exp0025` at `2026-04-10T14:34:06Z`: frozen mixed test evaluation

### Direct long-xattn model versus frozen exported-embedding probe

| Model | Validation macro AUROC | Validation macro AP | Test macro AUROC | Test macro AP | Test macro ECE | Test macro F1 @ 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct long-xattn classifier (`exp0018`) | `0.755222` | `0.138868` | `0.759965` | `0.137380` | `0.360090` | `0.161734` |
| Frozen linear probe on exported long-xattn embeddings (`exp0020`) | `0.755293` | `0.139524` | `0.759056` | `0.137536` | `0.318471` | `0.168394` |

Relative to the direct model in `exp0018`, the frozen exported-embedding baseline in `exp0020` was nearly tied on test macro AUROC (`-0.000909`) while slightly improving macro AP (`+0.000156`), substantially lowering macro ECE (`-0.041619`), and improving macro `F1 @ 0.5` (`+0.006660`).

### Validation-time retrieval and mixing selections

| Validation artifact | Selected config | Validation macro AUROC | Validation macro AP | Validation macro ECE | Validation macro F1 @ 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Memory-only selection (`exp0022`) | `k=50`, `tau=1` | `0.702234` | `0.128844` | `0.011892` | `0.018889` |
| Probability-mixing selection (`exp0023`) | `alpha=0.8` | `0.755483` | `0.141482` | `0.254518` | `0.186476` |

### Frozen test comparison within the branch

| Method | Frozen config | Test macro AUROC | Test macro AP | Test macro ECE | Test macro F1 @ 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen supervised baseline (`exp0020`) | supervised only | `0.759056` | `0.137536` | `0.318471` | `0.168394` |
| Memory-only retrieval (`exp0024`) | `k=50`, `tau=1` | `0.700407` | `0.127088` | `0.011227` | `0.021696` |
| Mixed model (`exp0025`) | `k=50`, `tau=1`, `alpha=0.8` | `0.759259` | `0.139147` | `0.255039` | `0.186855` |

Relative to the long-xattn frozen baseline in `exp0020`, the memory-only branch in `exp0024` remained much weaker on discrimination and fixed-threshold F1, with `-0.058649` test macro AUROC, `-0.010448` test macro AP, and `-0.146699` test macro `F1 @ 0.5`, even though macro ECE dropped by `-0.307244`. The mixed branch in `exp0025` was only slightly better than the frozen baseline on ranking, with `+0.000204` test macro AUROC and `+0.001611` macro AP, but it also improved macro ECE by `-0.063432` and macro `F1 @ 0.5` by `+0.018461`.

### Comparison against the earlier full-pipeline xattn branch

| Comparison | Delta test macro AUROC | Delta test macro AP | Delta test macro ECE | Delta test macro F1 @ 0.5 |
| --- | ---: | ---: | ---: | ---: |
| Long-xattn frozen baseline (`exp0020`) minus first frozen xattn baseline (`exp0012`) | `+0.031271` | `+0.025779` | `-0.050138` | `+0.013291` |
| Long-xattn memory-only (`exp0024`) minus first memory-only (`exp0016`) | `+0.017835` | `+0.019513` | `+0.000799` | `+0.017851` |
| Long-xattn mixed (`exp0025`) minus first mixed system (`exp0017`) | `+0.030005` | `+0.024302` | `+0.033582` | `+0.046457` |

The most important change in this branch was the stronger supervised representation. Moving from the one-epoch full-pipeline xattn line to the longer xattn line materially improved both the frozen baseline and the final mixed system on test macro AUROC, macro AP, and macro `F1 @ 0.5`. It also changed the retrieval selection regime: the best validation memory-only setting moved from `tau=20` in `exp0014` to `tau=1` in `exp0022`, while `k` remained `50`.

## 13. 2026-04-10 Hybrid Concat Update: `exp0026`-`exp0033`

The hybrid branch augmented the stronger long-xattn export with separate image and report embeddings. `exp0026` rebuilt ResNet50 image embeddings, and `exp0027` concatenated three sources in order: exported long-xattn embeddings from `exp0019`, rebuilt image embeddings from `exp0026`, and report embeddings from `exp0002`. The resulting representation was `2688`-dimensional and L2-normalized after concatenation. `exp0028` then trained a frozen linear baseline on that hybrid representation, and `exp0029` to `exp0033` repeated the retrieval-memory and mixing stages.

### Branch lineage and timestamps

- `exp0026` at `2026-04-10T15:16:45Z`: rebuilt ResNet50 image embeddings for the hybrid branch
- `exp0027` at `2026-04-10T15:43:56Z`: generated the `2688`-dimensional L2-normalized hybrid concat embeddings
- `exp0028` at `2026-04-10T15:44:54Z`: frozen linear baseline on the hybrid embedding root
- `exp0029` at `2026-04-10T15:45:27Z`: FAISS memory build for the hybrid embedding root
- `exp0030` at `2026-04-10T15:46:37Z`: memory-only validation sweep selecting `k=50` and `tau=1`
- `exp0031` at `2026-04-10T15:46:49Z`: validation probability-mixing sweep selecting `alpha=0.6`
- `exp0032` at `2026-04-10T15:48:56Z`: frozen memory-only test evaluation
- `exp0033` at `2026-04-10T15:49:07Z`: frozen mixed test evaluation

### Validation-time retrieval and mixing selections

| Validation artifact | Selected config | Validation macro AUROC | Validation macro AP | Validation macro ECE | Validation macro F1 @ 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Memory-only selection (`exp0030`) | `k=50`, `tau=1` | `0.718101` | `0.143574` | `0.011127` | `0.027371` |
| Probability-mixing selection (`exp0031`) | `alpha=0.6` | `0.766955` | `0.156347` | `0.194820` | `0.204543` |

### Frozen test comparison within the branch

| Method | Frozen config | Test macro AUROC | Test macro AP | Test macro ECE | Test macro F1 @ 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen supervised baseline (`exp0028`) | supervised only | `0.770829` | `0.151510` | `0.324818` | `0.173262` |
| Memory-only retrieval (`exp0032`) | `k=50`, `tau=1` | `0.717359` | `0.142505` | `0.010017` | `0.029407` |
| Mixed model (`exp0033`) | `k=50`, `tau=1`, `alpha=0.6` | `0.772098` | `0.154132` | `0.195644` | `0.200968` |

### Comparison against the long-xattn branch

| Comparison | Delta test macro AUROC | Delta test macro AP | Delta test macro ECE | Delta test macro F1 @ 0.5 |
| --- | ---: | ---: | ---: | ---: |
| Hybrid frozen baseline (`exp0028`) minus long-xattn frozen baseline (`exp0020`) | `+0.011774` | `+0.013974` | `+0.006346` | `+0.004868` |
| Hybrid memory-only (`exp0032`) minus long-xattn memory-only (`exp0024`) | `+0.016952` | `+0.015417` | `-0.001211` | `+0.007711` |
| Hybrid mixed (`exp0033`) minus long-xattn mixed (`exp0025`) | `+0.012839` | `+0.014985` | `-0.059395` | `+0.014113` |

The hybrid branch was the strongest completed branch in the local artifacts. It improved the frozen baseline over the long-xattn baseline on test macro AUROC, macro AP, and macro `F1 @ 0.5`, although its baseline macro ECE was slightly higher. More importantly, the hybrid mixed system in `exp0033` was the strongest completed mixed model overall, reaching test macro AUROC `0.772098`, macro AP `0.154132`, macro ECE `0.195644`, and macro `F1 @ 0.5` `0.200968`.

## 14. 2026-04-10 Gated-Hybrid Update: `exp0034`-`exp0043`

### Smoke sanity check: `exp0034`-`exp0035`

Before the full gated-hybrid branch, the pipeline was sanity-checked with a smoke run. `exp0034` trained a gated-hybrid model with `max_samples_per_split=8`, `epochs=1`, `patience=0`, `batch_size=2`, and `max_length=64`. `exp0035` then exported embeddings from that smoke checkpoint.

| Smoke artifact | UTC timestamp | Key setup | Reported output |
| --- | --- | --- | --- |
| `exp0034` | `2026-04-10T16:13:03Z` | gated-hybrid smoke train on `8` samples per split | validation AUROC and AP were `null`; test macro AUROC `0.417143`, test macro AP `0.285556` |
| `exp0035` | `2026-04-10T16:13:19Z` | smoke export from `exp0034` checkpoint | confirmed the gated-hybrid export path produced `512`-dimensional embeddings |

These smoke artifacts are useful as pipeline checks only and should not be presented as comparable to the full-data branches.

### Long gated-hybrid branch: `exp0036`-`exp0043`

The full gated-hybrid branch then trained a longer model with `gated_hybrid=true`, exported embeddings, and repeated the frozen-baseline and retrieval path. Unlike the completed branches above, the training directory for `exp0036` contains `config.json`, `train_log.jsonl`, and `best.ckpt`, but it does not contain final `experiment_meta.json`, `val_metrics.json`, `test_metrics.json`, or `recreation_report.md`. The downstream export and evaluation artifacts `exp0037` to `exp0043` do exist and are internally consistent, so the branch is reportable with an explicit caveat that the training stage itself is only partially finalized.

#### `exp0036` logged training trajectory

| Logged epoch | Improved checkpoint | Validation macro AUROC | Validation macro AP | Validation macro ECE | Validation macro F1 @ 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| `1` | yes | `0.728602` | `0.119555` | `0.371746` | `0.157417` |
| `2` | yes | `0.748826` | `0.128147` | `0.400675` | `0.154009` |
| `3` | yes | `0.750791` | `0.135547` | `0.366147` | `0.162004` |
| `4` | yes | `0.754376` | `0.140957` | `0.340756` | `0.166335` |
| `5` | no | `0.749549` | `0.142428` | `0.329444` | `0.170396` |
| `6` | no | `0.750489` | `0.142064` | `0.346604` | `0.164609` |

The strongest logged validation AUROC in `exp0036` was therefore `0.754376` at epoch `4`, and the export stage `exp0037` explicitly points to the `exp0036` `best.ckpt` file. The defensible wording is that the gated-hybrid training run appears to have produced a usable checkpoint, but the final training metadata is incomplete.

#### Completed downstream lineage and timestamps

- `exp0036` config timestamp `2026-04-10T16:15:30Z`: long gated-hybrid training launched with `epochs=50`, `patience=5`, `batch_size=32`, and `max_length=256`
- `exp0037` at `2026-04-10T17:54:52Z`: embedding export from the `exp0036` checkpoint
- `exp0038` at `2026-04-10T17:55:42Z`: frozen linear baseline on exported gated-hybrid embeddings
- `exp0039` at `2026-04-10T17:56:04Z`: FAISS memory build for the gated-hybrid embedding root
- `exp0040` at `2026-04-10T17:56:37Z`: memory-only validation sweep selecting `k=50` and `tau=1`
- `exp0041` at `2026-04-10T17:56:50Z`: validation probability-mixing sweep selecting `alpha=0.6`
- `exp0042` at `2026-04-10T17:57:38Z`: frozen memory-only test evaluation
- `exp0043` at `2026-04-10T17:57:48Z`: frozen mixed test evaluation

### Validation-time retrieval and mixing selections

| Validation artifact | Selected config | Validation macro AUROC | Validation macro AP | Validation macro ECE | Validation macro F1 @ 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Memory-only selection (`exp0040`) | `k=50`, `tau=1` | `0.709453` | `0.132715` | `0.011119` | `0.017974` |
| Probability-mixing selection (`exp0041`) | `alpha=0.6` | `0.755682` | `0.142884` | `0.193870` | `0.194022` |

### Frozen test comparison within the branch

| Method | Frozen config | Test macro AUROC | Test macro AP | Test macro ECE | Test macro F1 @ 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen supervised baseline (`exp0038`) | supervised only | `0.759012` | `0.141954` | `0.323712` | `0.168837` |
| Memory-only retrieval (`exp0042`) | `k=50`, `tau=1` | `0.709685` | `0.132874` | `0.011057` | `0.023517` |
| Mixed model (`exp0043`) | `k=50`, `tau=1`, `alpha=0.6` | `0.759226` | `0.143809` | `0.194621` | `0.193147` |

### Comparison against the hybrid and long-xattn branches

| Comparison | Delta test macro AUROC | Delta test macro AP | Delta test macro ECE | Delta test macro F1 @ 0.5 |
| --- | ---: | ---: | ---: | ---: |
| Gated-hybrid frozen baseline (`exp0038`) minus hybrid frozen baseline (`exp0028`) | `-0.011817` | `-0.009556` | `-0.001106` | `-0.004426` |
| Gated-hybrid memory-only (`exp0042`) minus hybrid memory-only (`exp0032`) | `-0.007675` | `-0.009631` | `+0.001040` | `-0.005890` |
| Gated-hybrid mixed (`exp0043`) minus hybrid mixed (`exp0033`) | `-0.012872` | `-0.010323` | `-0.001024` | `-0.007821` |
| Gated-hybrid mixed (`exp0043`) minus long-xattn mixed (`exp0025`) | `-0.000033` | `+0.004662` | `-0.060419` | `+0.006292` |

The gated-hybrid branch therefore did not surpass the hybrid concat branch on any of the main completed downstream comparisons. Its mixed system `exp0043` was materially below the hybrid mixed system `exp0033` on test macro AUROC and macro AP. However, it was competitive with the long-xattn mixed branch `exp0025`: test macro AUROC was essentially tied, while macro AP, macro ECE, and macro `F1 @ 0.5` were modestly better in `exp0043`.

## 15. 2026-04-10 Cross-Branch Comparison And Supervisor Summary

### Direct learned-model checkpoints

| Branch | Training artifact | Status | Best epoch | Validation macro AUROC | Validation macro AP | Test macro AUROC | Test macro AP | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full-pipeline xattn | `exp0010` | complete | `1` | `0.718285` | `0.110125` | `0.722673` | `0.108092` | one-epoch direct cross-attention classifier |
| Long xattn | `exp0018` | complete | `4` | `0.755222` | `0.138868` | `0.759965` | `0.137380` | `50`-epoch budget, stopped early at epoch `4` |
| Long gated hybrid | `exp0036` | partial | best logged `4/6` | `0.754376` | `0.140957` | n/a | n/a | usable checkpoint and downstream export exist, but final training metadata is missing |

### Completed frozen baselines

| Branch | Baseline artifact | Representation | Test macro AUROC | Test macro AP | Test macro ECE | Test macro F1 @ 0.5 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| First xattn export branch | `exp0012` | exported xattn embeddings, `512` dims | `0.727785` | `0.111756` | `0.368609` | `0.155103` |
| Long xattn export branch | `exp0020` | exported long-xattn embeddings, `512` dims | `0.759056` | `0.137536` | `0.318471` | `0.168394` |
| Hybrid concat branch | `exp0028` | xattn + image + report concat, `2688` dims | `0.770829` | `0.151510` | `0.324818` | `0.173262` |
| Gated-hybrid export branch | `exp0038` | exported gated-hybrid embeddings, `512` dims | `0.759012` | `0.141954` | `0.323712` | `0.168837` |

### Completed mixed systems

| Branch | Mixed artifact | Selected config | Test macro AUROC | Test macro AP | Test macro ECE | Test macro F1 @ 0.5 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| First xattn export branch | `exp0017` | `k=50`, `tau=20`, `alpha=0.6` | `0.729254` | `0.114845` | `0.221457` | `0.140398` |
| Long xattn export branch | `exp0025` | `k=50`, `tau=1`, `alpha=0.8` | `0.759259` | `0.139147` | `0.255039` | `0.186855` |
| Hybrid concat branch | `exp0033` | `k=50`, `tau=1`, `alpha=0.6` | `0.772098` | `0.154132` | `0.195644` | `0.200968` |
| Gated-hybrid export branch | `exp0043` | `k=50`, `tau=1`, `alpha=0.6` | `0.759226` | `0.143809` | `0.194621` | `0.193147` |

### Completed memory-only systems

| Branch | Memory-only artifact | Selected config | Test macro AUROC | Test macro AP | Test macro ECE | Test macro F1 @ 0.5 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| First xattn export branch | `exp0016` | `k=50`, `tau=20` | `0.682572` | `0.107575` | `0.010428` | `0.003845` |
| Long xattn export branch | `exp0024` | `k=50`, `tau=1` | `0.700407` | `0.127088` | `0.011227` | `0.021696` |
| Hybrid concat branch | `exp0032` | `k=50`, `tau=1` | `0.717359` | `0.142505` | `0.010017` | `0.029407` |
| Gated-hybrid export branch | `exp0042` | `k=50`, `tau=1` | `0.709685` | `0.132874` | `0.011057` | `0.023517` |

### Supervisor-facing synthesis

- The actual experimental progression after `exp0009` was full-pipeline xattn (`exp0010` to `exp0017`), then a stronger long-xattn branch (`exp0018` to `exp0025`), then a hybrid concat branch (`exp0026` to `exp0033`), and finally a gated-hybrid branch (`exp0034` to `exp0043`).
- The main retrieval-selection change happened after the first branch: the memory-only winner used `tau=20` in `exp0014`, but every later completed branch selected `tau=1`. Across all completed branches, the selected `k` remained `50`.
- The strongest completed frozen baseline was the hybrid concat baseline `exp0028`, and the strongest completed mixed system was the hybrid concat mixed model `exp0033`.
- The gated-hybrid branch was informative but not dominant. Its downstream mixed result `exp0043` did not beat the hybrid mixed result `exp0033`, although it was competitive with the long-xattn mixed result `exp0025` and better than it on macro AP, macro ECE, and macro `F1 @ 0.5`.
- The incomplete training metadata for `exp0036` should be mentioned explicitly in any supervisor-facing presentation. The downstream gated-hybrid branch is still usable for comparison because `exp0037` to `exp0043` are present and internally coherent, but the training run itself should be described as partially finalized rather than fully documented.
