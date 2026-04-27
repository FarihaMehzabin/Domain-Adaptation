# Hospital Adapters V1

This experiment adds a hospital-specific residual adapter on the pooled DenseNet-121 feature vector instead of using LoRA.

The baseline design is:

- load a pretrained DenseNet-121 multilabel checkpoint
- freeze the DenseNet backbone
- freeze the classifier head by default
- extract the pooled 1024-d feature
- apply a zero-initialized residual bottleneck adapter
- add an optional hospital-specific bias vector

The adapter starts as an identity map because the final `Up` layer is zero-initialized and the bias starts at zero. Before training, the wrapped model should match the source-only checkpoint almost exactly.

## Why This Is Different From LoRA

LoRA modifies layer internals. This experiment does not touch DenseNet blocks at all by default. It adapts only the pooled feature representation with one tiny hospital-specific residual module plus an optional bias term.

## Files

- `train_hospital_adapter.py`: trains one adapter for one target hospital
- `eval_hospital_adapter.py`: evaluates source-only and adapter checkpoints on one manifest
- `models/hospital_adapter.py`: adapter modules, parameter freezing helpers, adapter checkpoint save/load
- `common.py`: manifest loading, masked loss, metrics, predictions, run-folder helpers

## Outputs

Training writes to:

- `experiments/hospital_adapters_v1/runs/<run_name>/adapter_best.pt`
- `experiments/hospital_adapters_v1/runs/<run_name>/adapter_last.pt`
- `experiments/hospital_adapters_v1/runs/<run_name>/config.json`
- `experiments/hospital_adapters_v1/runs/<run_name>/train_log.csv`
- `experiments/hospital_adapters_v1/runs/<run_name>/target_val_report.json`
- `experiments/hospital_adapters_v1/runs/<run_name>/target_test_report.json`
- `experiments/hospital_adapters_v1/runs/<run_name>/source_eval_report.json` when source manifests are provided
- `experiments/hospital_adapters_v1/runs/<run_name>/predictions_target_val.csv`
- `experiments/hospital_adapters_v1/runs/<run_name>/predictions_target_test.csv` when test manifest is provided

Evaluation writes to:

- `experiments/hospital_adapters_v1/runs/<run_name>/eval_report.json`
- `experiments/hospital_adapters_v1/runs/<run_name>/predictions_source_only.csv`
- `experiments/hospital_adapters_v1/runs/<run_name>/predictions_adapter_<name>.csv` for each adapter checkpoint

## Commands

Source-only evaluation on the MIMIC Policy-B validation manifest:

```bash
python experiments/hospital_adapters_v1/eval_hospital_adapter.py \
  --run-name source_only_on_mimic_val \
  --checkpoint /workspace/checkpoints/nih_2k_densenet121_best.pt \
  --manifest /workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_val.csv \
  --label-policy uignore_blankzero \
  --out-dir /workspace/experiments/hospital_adapters_v1/runs/source_only_on_mimic_val
```

Source-only evaluation on the MIMIC Policy-B test manifest:

```bash
python experiments/hospital_adapters_v1/eval_hospital_adapter.py \
  --run-name source_only_on_mimic_test \
  --checkpoint /workspace/checkpoints/nih_2k_densenet121_best.pt \
  --manifest /workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_test.csv \
  --label-policy uignore_blankzero \
  --out-dir /workspace/experiments/hospital_adapters_v1/runs/source_only_on_mimic_test
```

Train an NIH-to-MIMIC hospital adapter on the `k20` support set:

```bash
python experiments/hospital_adapters_v1/train_hospital_adapter.py \
  --run-name nih_base_to_mimic_adapter_k20_seed2027 \
  --source-hospital nih \
  --target-hospital mimic \
  --checkpoint /workspace/checkpoints/nih_2k_densenet121_best.pt \
  --support-csv /workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_support_k20_seed2027.csv \
  --val-csv /workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_val.csv \
  --test-csv /workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_test.csv \
  --label-policy uignore_blankzero \
  --adapter-bottleneck 128 \
  --adapter-dropout 0.1 \
  --adapter-scale-init 0.001 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --epochs 100 \
  --patience 15 \
  --batch-size 16 \
  --seed 2027 \
  --out-dir /workspace/experiments/hospital_adapters_v1/runs
```

Evaluate the trained adapter on the MIMIC Policy-B test manifest:

```bash
python experiments/hospital_adapters_v1/eval_hospital_adapter.py \
  --run-name eval_nih_base_to_mimic_adapter_k20_seed2027 \
  --checkpoint /workspace/checkpoints/nih_2k_densenet121_best.pt \
  --adapter-checkpoint /workspace/experiments/hospital_adapters_v1/runs/nih_base_to_mimic_adapter_k20_seed2027/adapter_best.pt \
  --manifest /workspace/experiments/policyB_common5_v1/manifests/mimic_common5_policyB_test.csv \
  --label-policy uignore_blankzero \
  --out-dir /workspace/experiments/hospital_adapters_v1/runs/eval_nih_base_to_mimic_adapter_k20_seed2027
```

Reverse direction is supported as long as the checkpoint and manifests use the same label order.

## Notes

- `--train-classifier-head` is off by default.
- `--use-pos-weight` is on by default and is computed from the support set with a max clamp.
- `--debug-overfit-one-batch` runs a short smoke test and restores the initial adapter state before real training starts.
- `target_val_report.json` and `target_test_report.json` include both adapted metrics and source-only comparison metrics.

