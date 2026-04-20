# NIH -> CheXpert Report

## Manifest
- `/workspace/paper_v1/outputs/refresh_chexpert/20260420T081459Z__refresh_chexpert_full/manifest/manifest_current_plus_refreshed_chexpert.csv`

## Leaderboard
- `source_only`: seen_avg=`0.8474` nih=`0.8448` chexpert=`0.8500` forgetting=`0.0000` avg_ece=`0.0652`
- `vq_summary_replay`: seen_avg=`0.8410` nih=`0.8262` chexpert=`0.8558` forgetting=`0.0185` avg_ece=`0.0643`
- `lwf`: seen_avg=`0.8405` nih=`0.8301` chexpert=`0.8510` forgetting=`0.0147` avg_ece=`0.0815`
- `l2_anchor`: seen_avg=`0.8383` nih=`0.8304` chexpert=`0.8462` forgetting=`0.0144` avg_ece=`0.0897`
- `ewc`: seen_avg=`0.8383` nih=`0.8304` chexpert=`0.8462` forgetting=`0.0144` avg_ece=`0.0897`
- `finetune_seq`: seen_avg=`0.8383` nih=`0.8304` chexpert=`0.8462` forgetting=`0.0144` avg_ece=`0.0897`
- `fixed_alpha_mix`: seen_avg=`0.8383` nih=`0.8309` chexpert=`0.8457` forgetting=`0.0139` avg_ece=`0.0883`
- `main_method`: seen_avg=`0.8362` nih=`0.8172` chexpert=`0.8551` forgetting=`0.0275` avg_ece=`0.0575`

## Pilot Gate
- Continue to Stage 2: `False`
- Best replay-free baseline by seen-average: `lwf`
- Similar-CheXpert forgetting reference: `vq_summary_replay`
- `seen_average_gain_vs_best_replay_free`: `-0.004384`
- `seen_average_gate_pass`: `False`
- `forgetting_delta_vs_similar_chex_reference`: `-0.009021`
- `forgetting_ratio_vs_similar_chex_reference`: `1.487089`
- `forgetting_gate_pass`: `False`
- `average_macro_ece_gain_vs_best_simpler_seen_average`: `0.006836`
- `calibration_gate_pass`: `False`
