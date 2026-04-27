# Policy B Common5 Workspace Setup

## Folder Tree

```text
experiments/policyB_common5_v1/
  README.md
  workspace_manifest.json
  run_index.csv
  configs/
    experiment_defaults.json
  manifests/
    mimic_common5_policyB_train_pool.csv
    mimic_common5_policyB_val.csv
    mimic_common5_policyB_test.csv
    mimic_common5_policyB_support_k5_seed2027.csv
    mimic_common5_policyB_support_k20_seed2027.csv
    nih_dev_2k_train.csv
    nih_dev_2k_val.csv
    nih_dev_2k_test.csv
  checkpoints/
  outputs/
  reports/
    official_label_policy.md
    official_label_policy.json
    policyB_manifest_audit.md
    policyB_existing_predictions_eval.md
    workspace_setup_report.md
  logs/
  scripts_snapshot/
    experiment_namespace.py
    prepare_policyB_workspace.py
    evaluate_nih_on_mimic.py
    adapt_head_only_mimic.py
    adapt_full_finetune_mimic.py
    adapt_lastblock_mimic.py
    adapt_lora_mimic.py
    train_nih_2k_baseline.py
```

## Copied Manifests And Docs

- Policy B target manifests copied into `experiments/policyB_common5_v1/manifests/`:
  - `mimic_common5_policyB_train_pool.csv`
  - `mimic_common5_policyB_val.csv`
  - `mimic_common5_policyB_test.csv`
  - `mimic_common5_policyB_support_k5_seed2027.csv`
  - `mimic_common5_policyB_support_k20_seed2027.csv`
- NIH source manifests copied into `experiments/policyB_common5_v1/manifests/`:
  - `nih_dev_2k_train.csv`
  - `nih_dev_2k_val.csv`
  - `nih_dev_2k_test.csv`
- Policy documentation copied into `experiments/policyB_common5_v1/reports/`:
  - `official_label_policy.md`
  - `official_label_policy.json`
  - `policyB_manifest_audit.md`
  - `policyB_existing_predictions_eval.md`

## Scripts Patched

- `scripts/prepare_policyB_workspace.py` creates the isolated namespace, copies required inputs, and writes `workspace_manifest.json`.
- `scripts/experiment_namespace.py` centralizes namespace path resolution, Policy B manifest guards, and dry-run path printing.
- Patched training and evaluation entrypoints:
  - `scripts/evaluate_nih_on_mimic.py`
  - `scripts/adapt_head_only_mimic.py`
  - `scripts/adapt_full_finetune_mimic.py`
  - `scripts/adapt_lastblock_mimic.py`
  - `scripts/adapt_lora_mimic.py`
  - `scripts/train_nih_2k_baseline.py`
- Each patched entrypoint now accepts `--base_dir`, prints resolved manifests and output paths, and supports `--dry_run`.
- With `--label_policy uignore_blankzero`, the MIMIC eval/adaptation scripts reject the legacy `mimic_common5_{train_pool,val,test}.csv` target manifests and require the Policy B versions.

## Dry-Run Commands

```bash
python scripts/evaluate_nih_on_mimic.py \
  --base_dir experiments/policyB_common5_v1 \
  --run_name policyB_no_adaptation_eval_seed2027 \
  --source_checkpoint checkpoints/nih_2k_densenet121_best.pt \
  --label_policy uignore_blankzero \
  --dry_run

python scripts/adapt_head_only_mimic.py \
  --base_dir experiments/policyB_common5_v1 \
  --run_name policyB_head_only_k5_seed2027 \
  --source_checkpoint checkpoints/nih_2k_densenet121_best.pt \
  --support_manifest mimic_common5_policyB_support_k5_seed2027.csv \
  --source_only_report policyB_no_adaptation_eval_seed2027.json \
  --label_policy uignore_blankzero \
  --dry_run

python scripts/adapt_head_only_mimic.py \
  --base_dir experiments/policyB_common5_v1 \
  --run_name policyB_head_only_k20_seed2027 \
  --source_checkpoint checkpoints/nih_2k_densenet121_best.pt \
  --support_manifest mimic_common5_policyB_support_k20_seed2027.csv \
  --source_only_report policyB_no_adaptation_eval_seed2027.json \
  --label_policy uignore_blankzero \
  --dry_run

python scripts/adapt_full_finetune_mimic.py \
  --base_dir experiments/policyB_common5_v1 \
  --run_name policyB_full_ft_k5_seed2027 \
  --source_checkpoint checkpoints/nih_2k_densenet121_best.pt \
  --support_manifest mimic_common5_policyB_support_k5_seed2027.csv \
  --source_only_report policyB_no_adaptation_eval_seed2027.json \
  --label_policy uignore_blankzero \
  --dry_run

python scripts/adapt_full_finetune_mimic.py \
  --base_dir experiments/policyB_common5_v1 \
  --run_name policyB_full_ft_k20_seed2027 \
  --source_checkpoint checkpoints/nih_2k_densenet121_best.pt \
  --support_manifest mimic_common5_policyB_support_k20_seed2027.csv \
  --source_only_report policyB_no_adaptation_eval_seed2027.json \
  --label_policy uignore_blankzero \
  --dry_run
```

## Exact Next Command Suggestions

Run the official no-adaptation Policy B evaluation first:

```bash
python scripts/evaluate_nih_on_mimic.py \
  --base_dir experiments/policyB_common5_v1 \
  --run_name policyB_no_adaptation_eval_seed2027 \
  --source_checkpoint checkpoints/nih_2k_densenet121_best.pt \
  --label_policy uignore_blankzero
```

After that report exists at `experiments/policyB_common5_v1/reports/policyB_no_adaptation_eval_seed2027.json`, the first official adaptation rerun can be launched with:

```bash
python scripts/adapt_head_only_mimic.py \
  --base_dir experiments/policyB_common5_v1 \
  --run_name policyB_head_only_k5_seed2027 \
  --source_checkpoint checkpoints/nih_2k_densenet121_best.pt \
  --support_manifest mimic_common5_policyB_support_k5_seed2027.csv \
  --source_only_report policyB_no_adaptation_eval_seed2027.json \
  --label_policy uignore_blankzero
```

## Warnings

- No model training or evaluation was executed during setup.
- The adaptation dry-runs only pass once `--source_only_report` points to an existing file. For verification here, the patched scripts were checked with the legacy absolute source-only JSON, but the official commands above intentionally point to the future Policy B no-adaptation report inside this namespace.
- Root-level legacy folders were not moved, overwritten, or deleted.
