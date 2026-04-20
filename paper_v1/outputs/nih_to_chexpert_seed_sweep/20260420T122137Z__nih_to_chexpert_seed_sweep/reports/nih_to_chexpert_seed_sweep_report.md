# NIH -> CheXpert Seed Sweep

## Manifest
- `/workspace/paper_v1/outputs/refresh_chexpert/20260420T081459Z__refresh_chexpert_full/manifest/manifest_current_plus_refreshed_chexpert.csv`

## Seed Summary
- Per-seed results: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/seed_results.csv`
- Aggregate summary: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/seed_summary.csv`
- Promotion decision: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/artifacts/promotion_decision.json`

## Promotion Decision
- `promote_to_stage2_candidate`: `False`
- `mean_seen_average_above_reference`: `False`
- `mean_chexpert_above_source_only`: `True`
- `mean_forgetting_close_to_reference`: `False`
- `seen_average_variance_not_crazy`: `True`
- `chexpert_variance_not_crazy`: `True`

## Lead Diagnostics
- Per-seed gate summary: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/gate_by_label.csv`
- Aggregate gate summary: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/gate_by_label_aggregate.csv`
- Per-seed active-label fraction: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/active_label_fraction.csv`
- Aggregate active-label fraction: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/active_label_fraction_aggregate.csv`
- Per-seed residual summary: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/residual_by_label.csv`
- Aggregate residual summary: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/residual_by_label_aggregate.csv`
- Per-seed AUROC deltas: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/per_label_auroc_delta.csv`
- Aggregate AUROC deltas: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/per_label_auroc_delta_aggregate.csv`
- Representative AUROC delta CSVs: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/pilot_diagnostics`
- Representative diagnostics: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/pilot_diagnostics`

## Minimal Ablations
- Original vs harder gate summary: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/original_vs_harder_gate.csv`
- Gate/residual diagnostics: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/gate_ablation`
- Gate-cap sweep: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/gate_cap_sweep.csv`
