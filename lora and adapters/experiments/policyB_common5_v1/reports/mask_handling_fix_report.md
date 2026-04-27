# Policy B Mask Handling Fix Report

## Bug Summary
- The original Policy B no-adaptation evaluation treated `Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, and `Effusion` as dense binary labels and ignored the corresponding `*_mask` columns.
- That caused masked entries to be counted as negatives, label counts to omit masked totals, and macro metrics to be computed on the wrong row sets.
- The fix makes manifest validation, loss, metrics, prediction exports, and report rendering mask-aware when `--label_policy uignore_blankzero`.

## Files Patched
- `scripts/masked_multilabel_utils.py`
- `scripts/experiment_namespace.py`
- `scripts/evaluate_nih_on_mimic.py`
- `scripts/adapt_head_only_mimic.py`
- `scripts/adapt_full_finetune_mimic.py`
- `scripts/adapt_lastblock_mimic.py`
- `scripts/adapt_lora_mimic.py`

## Dry-Run Status
- `evaluate_nih_on_mimic.py --dry_run`: passed
- `adapt_head_only_mimic.py --dry_run` for `policyB_head_only_k5_seed2027`: passed
- `adapt_full_finetune_mimic.py --dry_run` for `policyB_full_ft_k5_seed2027`: passed

## Before vs After No-Adaptation Metrics
- Val macro AUROC: `0.6790065785` -> `0.6855828906`
- Val macro AUPRC: `0.2633158884` -> `0.2770682324`
- Test macro AUROC: `0.6102987726` -> `0.6183347016`
- Test macro AUPRC: `0.3311931404` -> `0.3540952717`

## Before vs After Label Counts
### Val
- Atelectasis: before `positives=194, negatives=764`; after `positives=194, negatives=712, masked=52, n_valid=906`
- Cardiomegaly: before `positives=205, negatives=753`; after `positives=205, negatives=724, masked=29, n_valid=929`
- Consolidation: before `positives=52, negatives=906`; after `positives=52, negatives=882, masked=24, n_valid=934`
- Edema: before `positives=131, negatives=827`; after `positives=131, negatives=767, masked=60, n_valid=898`
- Effusion: before `positives=249, negatives=709`; after `positives=249, negatives=682, masked=27, n_valid=931`

### Test
- Atelectasis: before `positives=170, negatives=426`; after `positives=170, negatives=394, masked=32, n_valid=564`
- Cardiomegaly: before `positives=159, negatives=437`; after `positives=159, negatives=424, masked=13, n_valid=583`
- Consolidation: before `positives=56, negatives=540`; after `positives=56, negatives=530, masked=10, n_valid=586`
- Edema: before `positives=128, negatives=468`; after `positives=128, negatives=424, masked=44, n_valid=552`
- Effusion: before `positives=219, negatives=377`; after `positives=219, negatives=350, masked=27, n_valid=569`

## Confirmation
- Per-label metrics now use only rows where the matching `*_mask == 1`.
- `positives + negatives + masked == row_count` is enforced for every label during manifest validation.
- A label is only macro-averaged if it still has at least one positive and one negative after masking.
- Masked loss now uses unreduced `BCEWithLogitsLoss` semantics with elementwise masking and normalization by `mask.sum()`.
- Micro AUROC and micro AUPRC are intentionally reported as `n/a` / `null`.
- Regenerated prediction CSVs now include the Policy B mask columns next to the true labels.

## Regenerated Artifacts
- `experiments/policyB_common5_v1/reports/policyB_no_adaptation_eval_seed2027.json`
- `experiments/policyB_common5_v1/reports/policyB_no_adaptation_eval_seed2027.md`
- `experiments/policyB_common5_v1/outputs/policyB_no_adaptation_eval_seed2027_val_predictions.csv`
- `experiments/policyB_common5_v1/outputs/policyB_no_adaptation_eval_seed2027_test_predictions.csv`

## Warnings
- Micro metrics remain disabled because they are not yet implemented in a clearly mask-aware way.
- The `k=5` support manifest contains some labels with only one class after masking. That is expected for few-shot data and affects metric validity, not masked loss correctness.
- Only the requested run artifacts inside `experiments/policyB_common5_v1/` were overwritten. Legacy root-level outputs were not touched.

## Final Status
- Bug fixed: `yes`
- Safe to start adaptation reruns: `yes`
