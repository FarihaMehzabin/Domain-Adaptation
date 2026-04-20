# paper_v1

Clean continual domain-adaptation experiments for frozen chest X-ray embeddings.

This package is intentionally isolated from the legacy experiment code in the repo.
It reads existing manifests and embedding exports as assets, but it does not reuse
legacy trainer, evaluation, retrieval, or runner implementations.

## What Is Implemented

- Asset reconnaissance and reporting for manifests, split leakage, embedding coverage,
  embedding dimensionality, and known gaps.
- A strict manifest plus embedding-index data layer with explicit validation policies.
- Multilabel evaluation metrics, calibration, forgetting, parameter counts, FLOPs, and
  stage report generation.
- New model and training code for:
  - NIH source-only linear head
  - sequential fine-tune
  - LwF
  - L2-anchor
  - EWC
  - fixed-alpha prototype mixing
  - VQ summary replay
  - prototype-constrained residual logit adaptation
- Synthetic smoke tests and unit tests.

## Current Asset Assumptions

- NIH uses the current `manifest_common_labels_pilot5h.csv` plus the existing
  `exp0014` CXR Foundation export.
- MIMIC uses the current `exp0085` export, which covers 998 train rows; the two
  missing rows are reported explicitly.
- CheXpert stage-1 training is blocked until a refreshed patient-disjoint manifest
  and fresh embedding export are provided.

## Quick Start

Run reconnaissance:

```bash
cd /workspace/paper_v1
PYTHONPATH=src python -m paper_v1.runners.run_recon --config configs/experiments/recon_current_assets.json
```

Run smoke tests:

```bash
cd /workspace/paper_v1
PYTHONPATH=src python -m paper_v1.runners.run_smoke --config configs/experiments/smoke.json
python -m unittest discover -s tests -v
```

Run the real NIH Stage 0 baseline:

```bash
cd /workspace/paper_v1
PYTHONPATH=src python -m paper_v1.runners.run_stage0 --config configs/experiments/stage0_nih_current.json
```

Attempt NIH -> CheXpert:

```bash
cd /workspace/paper_v1
PYTHONPATH=src python -m paper_v1.runners.run_nih_to_chexpert --config configs/experiments/nih_to_chexpert_refresh_required.json
```

The Stage 1 runner will fail fast if the CheXpert manifest is not patient-disjoint.
