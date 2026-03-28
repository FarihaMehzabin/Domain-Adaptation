# Source Baseline Selection

## Scope

This document records the source-baseline comparison for the current NIH CXR14 source stage and selects one canonical baseline for downstream retrieval-memory work.

Selection artifact directory:

`/workspace/experiments/exp0007__source_baseline_selection__nih_cxr14_source_baselines`

Compared baseline experiment directories:

- `/workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0001_image_only_linear`
- `/workspace/experiments/exp0005__source_baseline_training__nih_cxr14_exp0002_report_only_linear`
- `/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear`

Roadmap rule applied:

- do not assume the fused baseline wins automatically
- choose the canonical source baseline only after comparing image-only, report-only, and fused runs on held-out validation/test metrics

## Evaluation Protocol

All three baselines were trained with the same current trainer:

`/workspace/scripts/04_train_frozen_multilabel_baseline.py`

Shared training setup:

- task: NIH CXR14 multilabel classification with 14 labels
- model head: single linear layer on top of frozen embeddings
- loss: `BCEWithLogitsLoss`
- optimizer: `AdamW`
- batch size: `512`
- epoch budget: `30`
- learning rate: `1e-3`
- weight decay: `1e-4`
- patience: `5`
- seed: `1337`
- device: `cuda` via `--device auto`
- mixed precision: enabled with `--fp16-on-cuda`
- checkpoint selection: best validation macro AUROC
- threshold policy: per-label thresholds tuned on validation only, then reused for test

Input embedding roots:

- image-only baseline uses `/workspace/experiments/exp0001__image_embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test`
- report-only baseline uses `/workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test`
- fused baseline uses `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`

Input dimensionalities:

- image-only: `2048`
- report-only: `128`
- fused: `2176`

## Candidate Runs

### 1. Image-Only

- baseline experiment: `/workspace/experiments/exp0004__source_baseline_training__nih_cxr14_exp0001_image_only_linear`
- source embedding root: `/workspace/experiments/exp0001__image_embedding_generation__nih_cxr14_all14_resnet50_default_avg_train_val_test`
- best epoch: `30`

Validation macro metrics:

- AUROC: `0.745876`
- average precision: `0.133964`
- ECE: `0.358816`
- F1 @ 0.5: `0.161989`
- F1 @ tuned thresholds: `0.199989`

Test macro metrics:

- AUROC: `0.746067`
- average precision: `0.130515`
- ECE: `0.358840`
- F1 @ 0.5: `0.160943`
- F1 @ tuned thresholds: `0.184568`

### 2. Report-Only

- baseline experiment: `/workspace/experiments/exp0005__source_baseline_training__nih_cxr14_exp0002_report_only_linear`
- source embedding root: `/workspace/experiments/exp0002__report_embedding_generation__nih_cxr14_all14_biomedvlp_cxr_bert_specialized_auto_train_val_test`
- best epoch: `30`

Validation macro metrics:

- AUROC: `0.698718`
- average precision: `0.123513`
- ECE: `0.394832`
- F1 @ 0.5: `0.159970`
- F1 @ tuned thresholds: `0.191944`

Test macro metrics:

- AUROC: `0.703594`
- average precision: `0.129127`
- ECE: `0.394382`
- F1 @ 0.5: `0.159969`
- F1 @ tuned thresholds: `0.191192`

### 3. Fused

- baseline experiment: `/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear`
- source embedding root: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`
- best epoch: `30`

Validation macro metrics:

- AUROC: `0.763730`
- average precision: `0.151467`
- ECE: `0.347225`
- F1 @ 0.5: `0.175942`
- F1 @ tuned thresholds: `0.222298`

Test macro metrics:

- AUROC: `0.767933`
- average precision: `0.152244`
- ECE: `0.347454`
- F1 @ 0.5: `0.175678`
- F1 @ tuned thresholds: `0.209533`

## Side-by-Side Comparison

### Validation

| Baseline | Input dim | AUROC | AP | ECE | F1 @ 0.5 | F1 @ tuned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image-only | 2048 | 0.745876 | 0.133964 | 0.358816 | 0.161989 | 0.199989 |
| Report-only | 128 | 0.698718 | 0.123513 | 0.394832 | 0.159970 | 0.191944 |
| Fused | 2176 | 0.763730 | 0.151467 | 0.347225 | 0.175942 | 0.222298 |

### Test

| Baseline | Input dim | AUROC | AP | ECE | F1 @ 0.5 | F1 @ tuned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image-only | 2048 | 0.746067 | 0.130515 | 0.358840 | 0.160943 | 0.184568 |
| Report-only | 128 | 0.703594 | 0.129127 | 0.394382 | 0.159969 | 0.191192 |
| Fused | 2176 | 0.767933 | 0.152244 | 0.347454 | 0.175678 | 0.209533 |

## Ranking

### Validation ranking by macro AUROC

1. fused: `0.763730`
2. image-only: `0.745876`
3. report-only: `0.698718`

### Test ranking by macro AUROC

1. fused: `0.767933`
2. image-only: `0.746067`
3. report-only: `0.703594`

### Secondary metric check

The fused baseline is also best on the other macro metrics reported here:

- highest validation average precision
- highest test average precision
- lowest validation ECE
- lowest test ECE
- highest validation F1 @ 0.5
- highest test F1 @ 0.5
- highest validation F1 @ tuned thresholds
- highest test F1 @ tuned thresholds

## Delta Analysis

### Fused vs Image-Only

Validation deltas:

- macro AUROC: `+0.017854`
- macro average precision: `+0.017503`
- macro ECE: `-0.011592`
- macro F1 @ tuned thresholds: `+0.022309`

Test deltas:

- macro AUROC: `+0.021867`
- macro average precision: `+0.021729`
- macro ECE: `-0.011386`
- macro F1 @ tuned thresholds: `+0.024965`

### Fused vs Report-Only

Validation deltas:

- macro AUROC: `+0.065012`
- macro average precision: `+0.027954`
- macro ECE: `-0.047608`
- macro F1 @ tuned thresholds: `+0.030354`

Test deltas:

- macro AUROC: `+0.064340`
- macro average precision: `+0.023116`
- macro ECE: `-0.046929`
- macro F1 @ tuned thresholds: `+0.018341`

## Decision

The canonical source baseline is:

`/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear`

Decision label:

- selected baseline: `fused`
- selected experiment id: `exp0006`
- selected source embedding root: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`

## Rationale

The fused baseline wins on the primary selection metric, validation macro AUROC, and that advantage persists on test. The improvement is not marginal noise relative to the image-only baseline; it is roughly `+0.0179` on validation macro AUROC and `+0.0219` on test macro AUROC. It also improves average precision, calibration error, and both F1 summaries. That makes the choice defensible even under a stricter multi-metric review rather than a single-number pick.

The image-only baseline is the strongest non-fused fallback and materially outperforms report-only. That matters because it shows the visual branch is carrying most of the discriminative signal in this current setup. The report-only run is not useless, but under this exact frozen-linear protocol it is not competitive enough to justify choosing it as the canonical source baseline.

The roadmap explicitly warned not to assume fusion wins automatically. In this case the comparison does validate fusion, so the correct next-stage baseline for retrieval memory is the fused baseline, not the image-only run by default and not the report-only run because of the report-conditioned project framing.

## Operational Consequence

For the next source-stage milestone, any train-memory construction should use:

- chosen baseline experiment: `/workspace/experiments/exp0006__source_baseline_training__nih_cxr14_exp0003_fused_linear`
- chosen input embedding root: `/workspace/experiments/exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`
- chosen retrieval key family: fused embeddings derived from image + report features

If a later ablation needs a fallback baseline, use image-only from `exp0004` as the first comparator.

## Provenance

Relevant pushed commits:

- `216ab84c7` added the numbered workflow and image-only baseline
- `274221e9d` added the report-only baseline
- `5f1a9e7ac` added the fused baseline

Selection artifact creation date:

- UTC timestamp should be taken from this experiment directory commit

