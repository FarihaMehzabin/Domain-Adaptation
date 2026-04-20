# NIH -> CheXpert Seed Sweep

## Manifest
- `/workspace/paper_v1/outputs/refresh_chexpert/20260420T081459Z__refresh_chexpert_full/manifest/manifest_current_plus_refreshed_chexpert.csv`

## Seed Summary
- Per-seed results: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T103712Z__nih_to_chexpert_seed_sweep/reports/seed_results.csv`
- Aggregate summary: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T103712Z__nih_to_chexpert_seed_sweep/reports/seed_summary.csv`
- Promotion decision: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T103712Z__nih_to_chexpert_seed_sweep/artifacts/promotion_decision.json`

## Promotion Decision
- `promote_to_stage2_candidate`: `True`
- `mean_seen_average_above_source_only`: `True`
- `mean_forgetting_near_zero`: `True`
- `max_forgetting_near_zero`: `True`
- `mean_chexpert_gain_present`: `True`
- `seen_average_variance_not_crazy`: `True`
- `chexpert_variance_not_crazy`: `True`

## Minimal Ablations
- Original vs harder gate summary: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T103712Z__nih_to_chexpert_seed_sweep/reports/original_vs_harder_gate.csv`
- Gate/residual diagnostics: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T103712Z__nih_to_chexpert_seed_sweep/reports/gate_ablation`
- Gate-cap sweep: `/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T103712Z__nih_to_chexpert_seed_sweep/reports/gate_cap_sweep.csv`
