# full_ft_k20 Verification

- status: `VERIFIED_COMPLETE`
- json report exists: yes
- markdown report exists: yes
- recomputed metrics available: yes

## Artifact Presence

| Artifact | Present | Path |
| --- | --- | --- |
| JSON report | yes | `/workspace/reports/full_ft_k20_seed2027.json` |
| Markdown report | yes | `/workspace/reports/full_ft_k20_seed2027.md` |
| val predictions | yes | `/workspace/outputs/full_ft_k20_seed2027_val_predictions.csv` |
| test predictions | yes | `/workspace/outputs/full_ft_k20_seed2027_test_predictions.csv` |

## Consistency Checks

- markdown checks: {'contains_run_name': True, 'contains_status_done': True, 'contains_safe_to_continue_yes': True}
- json checks: {'status_done': True, 'safe_to_continue': True, 'prediction_file_paths_match': True, 'manifest_paths_match': True, 'label_order_match': True}
- val metric checks: {'present': True, 'metrics_comparison': {'macro_auroc_matches': True, 'macro_auprc_matches': True, 'per_label': {'Atelectasis': {'positives_match': True, 'negatives_match': True, 'auroc_match': True, 'auprc_match': True, 'all_match': True}, 'Cardiomegaly': {'positives_match': True, 'negatives_match': True, 'auroc_match': True, 'auprc_match': True, 'all_match': True}, 'Consolidation': {'positives_match': True, 'negatives_match': True, 'auroc_match': True, 'auprc_match': True, 'all_match': True}, 'Edema': {'positives_match': True, 'negatives_match': True, 'auroc_match': True, 'auprc_match': True, 'all_match': True}, 'Effusion': {'positives_match': True, 'negatives_match': True, 'auroc_match': True, 'auprc_match': True, 'all_match': True}}, 'all_match': True}, 'match_method': 'dicom_id'}
- test metric checks: {'present': True, 'metrics_comparison': {'macro_auroc_matches': True, 'macro_auprc_matches': True, 'per_label': {'Atelectasis': {'positives_match': True, 'negatives_match': True, 'auroc_match': True, 'auprc_match': True, 'all_match': True}, 'Cardiomegaly': {'positives_match': True, 'negatives_match': True, 'auroc_match': True, 'auprc_match': True, 'all_match': True}, 'Consolidation': {'positives_match': True, 'negatives_match': True, 'auroc_match': True, 'auprc_match': True, 'all_match': True}, 'Edema': {'positives_match': True, 'negatives_match': True, 'auroc_match': True, 'auprc_match': True, 'all_match': True}, 'Effusion': {'positives_match': True, 'negatives_match': True, 'auroc_match': True, 'auprc_match': True, 'all_match': True}}, 'all_match': True}, 'match_method': 'dicom_id'}
