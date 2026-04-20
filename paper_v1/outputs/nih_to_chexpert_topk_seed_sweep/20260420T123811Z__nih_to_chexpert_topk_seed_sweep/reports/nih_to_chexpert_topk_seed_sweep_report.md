# NIH -> CheXpert Top-K Seed Sweep

## Manifest
- `/workspace/paper_v1/outputs/refresh_chexpert/20260420T081459Z__refresh_chexpert_full/manifest/manifest_current_plus_refreshed_chexpert.csv`

## Comparison Set
- `source_only`
- `harder_gate_clipping`
- `tiny_logit_correction`
- `topk_labelwise_trust_region_k1`
- `topk_labelwise_trust_region_k2`

## Seed Summary
- Per-seed results: `/workspace/paper_v1/outputs/nih_to_chexpert_topk_seed_sweep/20260420T123811Z__nih_to_chexpert_topk_seed_sweep/reports/seed_results.csv`
- Aggregate summary: `/workspace/paper_v1/outputs/nih_to_chexpert_topk_seed_sweep/20260420T123811Z__nih_to_chexpert_topk_seed_sweep/reports/seed_summary.csv`
- Candidate decision: `/workspace/paper_v1/outputs/nih_to_chexpert_topk_seed_sweep/20260420T123811Z__nih_to_chexpert_topk_seed_sweep/artifacts/promotion_decision.json`

## Sparse Diagnostics
- Active-label fraction: `/workspace/paper_v1/outputs/nih_to_chexpert_topk_seed_sweep/20260420T123811Z__nih_to_chexpert_topk_seed_sweep/reports/active_label_fraction.csv`
- Active-label fraction aggregate: `/workspace/paper_v1/outputs/nih_to_chexpert_topk_seed_sweep/20260420T123811Z__nih_to_chexpert_topk_seed_sweep/reports/active_label_fraction_aggregate.csv`
- Selection frequency by label: `/workspace/paper_v1/outputs/nih_to_chexpert_topk_seed_sweep/20260420T123811Z__nih_to_chexpert_topk_seed_sweep/reports/selection_frequency_by_label.csv`
- Selection frequency aggregate: `/workspace/paper_v1/outputs/nih_to_chexpert_topk_seed_sweep/20260420T123811Z__nih_to_chexpert_topk_seed_sweep/reports/selection_frequency_by_label_aggregate.csv`
- Selected-label count distribution: `/workspace/paper_v1/outputs/nih_to_chexpert_topk_seed_sweep/20260420T123811Z__nih_to_chexpert_topk_seed_sweep/reports/selected_label_count_distribution.csv`
- Selected-label count distribution aggregate: `/workspace/paper_v1/outputs/nih_to_chexpert_topk_seed_sweep/20260420T123811Z__nih_to_chexpert_topk_seed_sweep/reports/selected_label_count_distribution_aggregate.csv`
- Per-label AUROC delta: `/workspace/paper_v1/outputs/nih_to_chexpert_topk_seed_sweep/20260420T123811Z__nih_to_chexpert_topk_seed_sweep/reports/per_label_auroc_delta.csv`
- Per-label AUROC delta aggregate: `/workspace/paper_v1/outputs/nih_to_chexpert_topk_seed_sweep/20260420T123811Z__nih_to_chexpert_topk_seed_sweep/reports/per_label_auroc_delta_aggregate.csv`
- Representative delta diagnostics: `/workspace/paper_v1/outputs/nih_to_chexpert_topk_seed_sweep/20260420T123811Z__nih_to_chexpert_topk_seed_sweep/reports/topk_diagnostics`
