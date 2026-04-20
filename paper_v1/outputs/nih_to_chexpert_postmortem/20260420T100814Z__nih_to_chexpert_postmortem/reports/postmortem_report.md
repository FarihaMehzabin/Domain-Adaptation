# NIH -> CheXpert Postmortem

## Verification
- `previous_requires_grad_false`: `True`
- `previous_grad_abs_sum_after_backward`: `0.0`
- `previous_logits_frozen`: `True`
- `memory_trainable_parameter_count`: `0`
- `old_bank_num_prototypes`: `42`
- `prototype_domain_counts`: `{'d0_nih': 42}`
- `current_domain_prototype_count`: `0`
- `retrieval_uses_current_domain_memory`: `False`
- `replay_active_each_epoch`: `True`
- `zero_penalty_active_each_epoch`: `True`
- `forgetting_metric_correct`: `True`

## Main Method Diagnostics
- Gate histogram CSVs: `/workspace/paper_v1/outputs/nih_to_chexpert_postmortem/20260420T100814Z__nih_to_chexpert_postmortem/reports/diagnostics/main_method__nih_test__gate_histogram.csv`, `/workspace/paper_v1/outputs/nih_to_chexpert_postmortem/20260420T100814Z__nih_to_chexpert_postmortem/reports/diagnostics/main_method__chexpert_test__gate_histogram.csv`
- Residual histogram CSVs: `/workspace/paper_v1/outputs/nih_to_chexpert_postmortem/20260420T100814Z__nih_to_chexpert_postmortem/reports/diagnostics/main_method__nih_test__residual_norm_histogram.csv`, `/workspace/paper_v1/outputs/nih_to_chexpert_postmortem/20260420T100814Z__nih_to_chexpert_postmortem/reports/diagnostics/main_method__chexpert_test__residual_norm_histogram.csv`
- Per-label AUROC deltas: `/workspace/paper_v1/outputs/nih_to_chexpert_postmortem/20260420T100814Z__nih_to_chexpert_postmortem/reports/diagnostics/main_method_vs_source_only__nih_test__per_label_auroc_delta.csv`, `/workspace/paper_v1/outputs/nih_to_chexpert_postmortem/20260420T100814Z__nih_to_chexpert_postmortem/reports/diagnostics/main_method_vs_source_only__chexpert_test__per_label_auroc_delta.csv`
- Main-method training curves: `/workspace/paper_v1/outputs/nih_to_chexpert_postmortem/20260420T100814Z__nih_to_chexpert_postmortem/20260420T100821Z__main_method/artifacts/main_method_history.csv`

## Rescue Comparison
- `harder_gate_clipping` seen_avg=`0.8484` chex_delta_vs_source=`0.0033` ece_delta_vs_source=`0.0013`
- `lwf_prototype_replay` seen_avg=`0.8375` chex_delta_vs_source=`0.0007` ece_delta_vs_source=`-0.0038`
- `smaller_bottleneck` seen_avg=`0.8464` chex_delta_vs_source=`0.0035` ece_delta_vs_source=`-0.0015`
- `stronger_replay_zero` seen_avg=`0.8251` chex_delta_vs_source=`-0.0033` ece_delta_vs_source=`-0.0123`
- `tiny_logit_correction` seen_avg=`0.8474` chex_delta_vs_source=`0.0000` ece_delta_vs_source=`-0.0011`

## Recommendation
- `continue_with_rescue_method`: `True`
- `viable_rescues`: `['harder_gate_clipping', 'tiny_logit_correction']`
- `pivot_to_protocol_benchmark_story`: `False`
- `best_seen_average_method`: `harder_gate_clipping`
