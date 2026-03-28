# Current Scripts

Use these numbered entrypoints for the current source-stage workflow:

1. `01_generate_nih_split_image_embeddings.py`
2. `02_generate_nih_split_report_embeddings.py`
3. `03_generate_split_fused_embeddings.py`
4. `04_train_frozen_multilabel_baseline.py`

Notes:

- Steps `01` to `03` are numbered entrypoints that delegate to the existing implementation files in this directory.
- Step `04` is the current baseline trainer for frozen embedding experiments.
- `scripts_old/` remains archived and is not part of the active workflow.
- After each experiment run, the expectation is:
  - the experiment directory contains a `recreation_report.md`,
  - the run is committed,
  - the commit is pushed before moving to the next experiment.

Compatibility:

- The original unnumbered generation scripts are still present so older references do not break.
- New runs should prefer the numbered entrypoints above.
