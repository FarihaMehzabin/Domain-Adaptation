# Policy B LoRA k20 Diagnostic

## Final Diagnosis
- final diagnosis: `EFFECTIVE_NO_OP_DUE_TO_MODEL_SELECTION`
- recommended next action: `change LoRA learning rate/rank and rerun`
- overall mean prediction diff vs source-only: 0.001747
- overall max prediction diff vs source-only: 0.006116
- LoRA tensors nonzero: yes
- train loss decreased: yes

## Prediction Diff: val
- join key: `dicom_id`
- matched rows: 958
- overall mean abs diff: 0.001725
- overall max abs diff: 0.005742
- count abs diff > 1e-6: 4789
- count abs diff > 1e-4: 4494
- count abs diff > 1e-3: 2548
- effectively source-only flag: no

| Label | Mean Abs Diff | Max Abs Diff | Pearson | Spearman | >1e-6 | >1e-4 | >1e-3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 0.000244 | 0.001100 | 0.999997 | 0.999991 | 957 | 705 | 2 |
| Cardiomegaly | 0.002825 | 0.004548 | 0.999993 | 0.999992 | 958 | 958 | 950 |
| Consolidation | 0.001061 | 0.002405 | 0.999998 | 0.999995 | 958 | 955 | 527 |
| Edema | 0.000629 | 0.001626 | 0.999999 | 0.999997 | 958 | 918 | 111 |
| Effusion | 0.003866 | 0.005742 | 0.999991 | 0.999987 | 958 | 958 | 958 |

## Prediction Diff: test
- join key: `dicom_id`
- matched rows: 596
- overall mean abs diff: 0.001769
- overall max abs diff: 0.006116
- count abs diff > 1e-6: 2980
- count abs diff > 1e-4: 2793
- count abs diff > 1e-3: 1682
- effectively source-only flag: no

| Label | Mean Abs Diff | Max Abs Diff | Pearson | Spearman | >1e-6 | >1e-4 | >1e-3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 0.000223 | 0.000936 | 0.999996 | 0.999986 | 596 | 425 | 0 |
| Cardiomegaly | 0.002934 | 0.004283 | 0.999992 | 0.999986 | 596 | 596 | 593 |
| Consolidation | 0.001172 | 0.002240 | 0.999997 | 0.999988 | 596 | 596 | 406 |
| Edema | 0.000650 | 0.001509 | 0.999999 | 0.999995 | 596 | 580 | 87 |
| Effusion | 0.003865 | 0.006116 | 0.999989 | 0.999980 | 596 | 596 | 596 |

## Checkpoint Inspection
- checkpoint keys: ['adaptation_method', 'architecture', 'best_metric_name', 'best_metric_value', 'epoch', 'image_size', 'label_names', 'model_state_dict', 'num_classes', 'parameter_summary', 'source_checkpoint', 'train_loss', 'val_loss']
- LoRA tensors present: yes
- LoRA tensor count: 66
- LoRA all-zero tensor count: 0
- LoRA near-zero tensor count: 0
- LoRA aggregate norm: 6.645872
- non-LoRA original tensors changed vs source: 0
- classifier weight comparison: [{'checkpoint_key': 'classifier.base_layer.weight', 'source_key': 'classifier.weight', 'shape': [5, 1024], 'max_abs_diff': 0.0, 'mean_abs_diff': 0.0, 'changed': False}, {'checkpoint_key': 'classifier.base_layer.bias', 'source_key': 'classifier.bias', 'shape': [5], 'max_abs_diff': 0.0, 'mean_abs_diff': 0.0, 'changed': False}]

## Training History
- history length: 11
- selected best epoch: 1
- stopped early: yes
- early stopping reason: Inferred patience stop after 10 consecutive non-improving epochs following best epoch 1.
- train loss decreased: yes
- val macro AUROC improved after epoch 1: no
- val macro AUPRC improved after epoch 1: no
- best epoch 1 source-like: yes
- later epochs changed predictions but were not selected: yes
- last vs source val mean/max abs diff: 0.059745 / 0.178309
- last vs source test mean/max abs diff: 0.058757 / 0.186016
- last vs best val mean/max abs diff: 0.058037 / 0.173697
- last vs best test mean/max abs diff: 0.057019 / 0.179899

## Trainability Audit
- target module count: 33 (expected 33)
- trainable parameters: 87060 (expected 87060)
- trainable parameter tensors: 66 (expected 66)
- trainable names LoRA-only: yes
- classifier base accidentally trainable: no
- denseblock base accidentally trainable: no

## Support Set
| Label | Positives | Negatives | Masked | n_valid | Positive Fraction | Warnings |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Atelectasis | 20 | 8 | 2 | 28 | 0.714 | none |
| Cardiomegaly | 21 | 9 | 0 | 30 | 0.700 | none |
| Consolidation | 20 | 10 | 0 | 30 | 0.667 | none |
| Edema | 21 | 8 | 1 | 29 | 0.724 | none |
| Effusion | 26 | 4 | 0 | 30 | 0.867 | negatives < 5, positive fraction >= 0.8 |

## Decision Notes
- Later epochs changed predictions materially, but validation selected epoch 1.
