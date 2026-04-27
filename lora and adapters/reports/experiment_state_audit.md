# Experiment State Audit

## 1. Repository Overview

The repository is script-driven. `scripts/` contains the runnable training/evaluation pipeline, `manifests/` contains the derived NIH and MIMIC split CSVs, `reports/` contains both dry-run stage reports and real experiment outputs, `checkpoints/` stores model weights, `outputs/` stores prediction CSVs, and `logs/` stores a small amount of stdout logging. `src/cxr_transfer/` and `configs/` exist but are effectively placeholders: no source `.py` files or config files were found there.

Important folders and files detected:
- `checkpoints/`
- `configs/ (empty)`
- `data/`
- `logs/`
- `manifests/`
- `outputs/`
- `reports/`
- `runs/ (present, effectively unused)`
- `scripts/`
- `src/cxr_transfer/ (placeholder package directories only)`
- `tests/`
- Training scripts: `scripts/train_nih_2k_baseline.py`, `scripts/adapt_head_only_mimic.py`, `scripts/adapt_full_finetune_mimic.py`
- Evaluation script: `scripts/evaluate_nih_on_mimic.py`
- Dataset/preprocessing scripts: `scripts/check_mimic_common5.py`, `scripts/create_mimic_kshot_support.py`
- Config files: none found
- Notebooks: none found
- Result/log/checkpoint folders: `reports/`, `logs/`, `outputs/`, `checkpoints/`, `runs/`
- Artifact inventory written to `reports/discovered_artifacts.csv` with 153 indexed files

## 2. Research Goal Detected from Code

The active completed pipeline is a five-label NIH-to-MIMIC transfer benchmark: train a DenseNet-121 source model on a small NIH subset, evaluate it directly on a target MIMIC hospital split, then adapt with few-shot target support data using either head-only fine-tuning or full fine-tuning. A second, newer `common7` track exists in manifests/reports, but it is only partially prepared and has no completed model run.

## 3. Dataset and Label Setup

- NIH source data: ChestXray14 frontal AP/PA images only. Current completed source training uses `manifests/nih_dev_2k_{train,val,test}.csv` with 1400/200/400 images and zero patient overlap across splits.
- Additional NIH manifests exist for a 10k split and for a planned common7 label set: `manifests/nih_dev_10k_*` and `manifests/nih_dev_*_common7_*`.
- MIMIC target data: frontal AP/PA images only, derived from the official MIMIC split CSV. Current completed runs use `manifests/mimic_common5_train_pool.csv` (963 rows), `manifests/mimic_common5_val.csv` (958 rows), and `manifests/mimic_common5_test.csv` (596 rows) with zero subject/study/dicom overlap.
- Completed runs use the common5 label set: `Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Effusion`, where MIMIC `Pleural Effusion` is mapped to `Effusion`.
- A planned common7 setup also exists: `atelectasis`, `cardiomegaly`, `consolidation`, `edema`, `pleural_effusion`, `pneumonia`, `pneumothorax`.
- Uncertainty handling is inconsistent across repo branches. Current completed common5 manifests use U-zero logic in code: `-1` and blank/NaN are converted to `0`. Separate `raw`, `u_ignore`, `u_one`, and `u_zero` common7 manifests are present.
- Raw-label missingness is substantial in the current MIMIC common5 manifests: train_pool 40.08%, val 38.31%, test 21.98% of rows have all five raw target labels blank before U-zero conversion.
- K-shot support manifests currently used by completed adaptation runs are `manifests/mimic_support_k5_seed2027.csv` (7 images total) and `manifests/mimic_support_k20_seed2027.csv` (30 images total). The K value refers to minimum positive examples per label, not the number of support images.

## 4. Completed Experiments

Only runs with concrete model or prediction artifacts are listed below. Dry-run stage reports and scaffolding-only reports are excluded from this table.

| Run ID | Dataset Setup | Labels | Model | Adaptation Method | Split | Seed | Best Metric | Checkpoint | Trust Level |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mini_stage_b_train | NIH dev-2k common5 source baseline | 5 | DenseNet121 | source only | NIH patient split 1400/200/400 | 2027 | NIH test AUROC 0.6574 / AUPRC 0.1549 | checkpoints/nih_2k_densenet121_best.pt | MEDIUM |
| mini_stage_d_nih_to_mimic | NIH->MIMIC common5 no adaptation | 5 | DenseNet121 | none | MIMIC official frontal val/test | 2027 | MIMIC test AUROC 0.6103 / AUPRC 0.3312 | checkpoints/nih_2k_densenet121_best.pt | MEDIUM |
| head_only_k5_seed2027 | NIH->MIMIC common5 K=5 (7 support images) | 5 | DenseNet121 | head-only | support/train_pool + MIMIC val/test | 2027 | MIMIC test AUROC 0.6102 / AUPRC 0.3311 | checkpoints/head_only_k5_seed2027_best.pt | MEDIUM |
| head_only_k20_seed2027 | NIH->MIMIC common5 K=20 (30 support images) | 5 | DenseNet121 | head-only | support/train_pool + MIMIC val/test | 2027 | MIMIC test AUROC 0.6104 / AUPRC 0.3319 | checkpoints/head_only_k20_seed2027_best.pt | MEDIUM |
| full_ft_k5_seed2027 | NIH->MIMIC common5 K=5 (7 support images) | 5 | DenseNet121 | full fine-tune | support/train_pool + MIMIC val/test | 2027 | MIMIC test AUROC 0.6239 / AUPRC 0.3436 | checkpoints/full_ft_k5_seed2027_best.pt | MEDIUM |
| full_ft_k20_seed2027 | NIH->MIMIC common5 K=20 (30 support images) | 5 | DenseNet121 | full fine-tune | support/train_pool + MIMIC val/test | 2027 | Only val AUROC 0.6981 recovered from checkpoint; no completed test result | checkpoints/full_ft_k20_seed2027_best.pt | DO NOT USE |

## 5. Best Results So Far

Source/base model results:

| Run ID | Split | Macro AUROC | Macro AUPRC | Notes |
| --- | --- | --- | --- | --- |
| mini_stage_b_train | NIH val | 0.6843 | 0.1705 | epoch-1 val-loss checkpoint |
| mini_stage_b_train | NIH test | 0.6574 | 0.1549 | epoch-1 val-loss checkpoint |

Target hospital no-adaptation results:

| Run ID | Split | Macro AUROC | Macro AUPRC |
| --- | --- | --- | --- |
| mini_stage_d_nih_to_mimic | MIMIC val | 0.6790 | 0.2633 |
| mini_stage_d_nih_to_mimic | MIMIC test | 0.6103 | 0.3312 |

Fine-tuning/adaptation results:

| Run ID | Method | K-shot | Support Size | Macro AUROC (test) | Macro AUPRC (test) |
| --- | --- | --- | --- | --- | --- |
| head_only_k5_seed2027 | head-only | K=5 | 7 images | 0.6102 | 0.3311 |
| head_only_k20_seed2027 | head-only | K=20 | 30 images | 0.6104 | 0.3319 |
| full_ft_k5_seed2027 | full fine-tune | K=5 | 7 images | 0.6239 | 0.3436 |

Few-shot/K-shot results:

| Run ID | Support Manifest | Support Label Counts | Test Macro AUROC | Test Macro AUPRC |
| --- | --- | --- | --- | --- |
| head_only_k5_seed2027 | `manifests/mimic_support_k5_seed2027.csv` | 5/6/5/6/7 positives | 0.6102 | 0.3311 |
| head_only_k20_seed2027 | `manifests/mimic_support_k20_seed2027.csv` | 20/22/20/20/26 positives | 0.6104 | 0.3319 |
| full_ft_k5_seed2027 | `manifests/mimic_support_k5_seed2027.csv` | 5/6/5/6/7 positives | 0.6239 | 0.3436 |

## 6. Problems / Red Flags

- HIGH: Target-side common5 manifests use U-zero and convert blank/NaN MIMIC labels to 0. A large fraction of MIMIC rows have all five raw target labels blank before conversion: 40.08% train_pool, 38.31% val, 21.98% test. If blanks should be masked rather than treated as negatives, every target-side result changes.
- HIGH: The repository contains two incompatible label setups: common5 completed runs versus common7 staged manifests with multiple uncertainty policies. Comparing common5/U-zero results against common7 or masked-label results would be scientifically invalid.
- MEDIUM: Source and adaptation checkpoints are selected with inconsistent criteria. This weakens source-only versus adaptation comparisons, and AUROC-selected checkpoints are not necessarily AUPRC-optimal.
- MEDIUM: Target adaptation drops class-imbalance handling even on tiny support sets. K=5 has zero Effusion negatives and K=20 has only four, so head-only/full-ft adaptation can become badly biased.
- MEDIUM: No training augmentations are applied in any completed training script. This is a weak baseline for chest X-ray transfer and may exaggerate overfitting on tiny support sets.
- MEDIUM: Reproducibility metadata is incomplete. Seeds are saved, but there are no committed configs, environment snapshots, or run commands. Reproducing exactly from the repo alone is still manual.
- HIGH: full_ft_k20_seed2027 is incomplete and should not be used as a result. It has no trustworthy final val/test report or prediction files.

## 7. Experiments That Are Usable for Paper

- `mini_stage_b_train`: Valid source-domain baseline with checkpoint, predictions, and metrics that recompute exactly from saved CSVs.
- `mini_stage_d_nih_to_mimic`: Valid no-adaptation target-hospital baseline with saved predictions and reproducible metrics.
- `head_only_k5_seed2027`: Valid few-shot negative baseline showing head-only adaptation does not improve over source-only under the current setup.
- `head_only_k20_seed2027`: Second few-shot negative baseline; useful to show that simply adding more positive support without changing the method does not help.
- `full_ft_k5_seed2027`: Best completed target-side result with checkpoint, predictions, and exact metric consistency checks.

## 8. Experiments That Should Be Re-run

- `full_ft_k20_seed2027`: The run is incomplete: there are checkpoints and a training log, but no valid finished report or test predictions.
- `stage5_source_baseline`: The common7 source baseline stage is only PARTIAL and produced no checkpoint or metrics.
- `mini_stage_d_nih_to_mimic + head_only_k5_seed2027 + head_only_k20_seed2027 + full_ft_k5_seed2027`: Re-run these if the blank/uncertain MIMIC label policy changes after audit, because every target-side common5 metric depends on that preprocessing choice.

## 9. Missing Pieces Before Next Experiment

- A documented decision on whether blank MIMIC labels should be treated as negatives or masked out.
- A single official label space for the next round (`common5` versus `common7`) so runs are comparable.
- Consistent model-selection policy across baselines and adaptation runs if AUROC/AUPRC are the headline metrics.
- Actual saved experiment configs or run commands; right now JSON reports are the only source of hyperparameters.

## 10. Recommended Next Step

Decide and document the MIMIC blank/uncertain-label policy first, then regenerate the target manifests before trusting or extending any target-side result.
