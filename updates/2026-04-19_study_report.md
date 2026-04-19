# Oracle, Transfer, and Naive Continual Baselines

- Manifest: `/workspace/manifest/manifest_pilot5h_binary_mimic.csv`
- Campaign: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h`

## Model Size And Compute

- Trainable linear head only: `5.4k` params
- Trainable linear head only: about `10.8k` FLOPs per image
- Frozen `CXR Foundation` backbone FLOPs were not logged because these runs used pre-exported embeddings

## 1. Independent Single-Domain Oracle Baselines

- `NIH` only (`exp0001`): `NIH test` AUROC `0.8461`
- `CheXpert` only (`exp0004`): `CheXpert test` AUROC `0.7703`
- `MIMIC` only (`exp0005`): `MIMIC test` AUROC `0.7619`

## 2. Cross-Domain Transfer Baseline

- Train on `NIH`, test directly with no adaptation:
- `CheXpert test` AUROC `0.8486`
- `MIMIC test` AUROC `0.7370`
- Relative to oracle:
- `NIH -> CheXpert` vs `CheXpert` oracle: `+0.0783`
- `NIH -> MIMIC` vs `MIMIC` oracle: `-0.0249`

## 3. Naive Continual Baseline

- Stage A `NIH` source (`exp0001`): `NIH 0.8461`, `CheXpert 0.8486`, `MIMIC 0.7370`
- Stage B after `CheXpert` fine-tune (`exp0002`): `NIH 0.8210`, `CheXpert 0.7882`
- Stage C after `MIMIC` fine-tune (`exp0003`): `NIH 0.8221`, `CheXpert 0.8135`, `MIMIC 0.7770`

## Main Deltas

- `NIH` forgetting after `CheXpert` fine-tune: `-0.0251`
- final `NIH` vs original `NIH` source: `-0.0240`
- `CheXpert` after Stage B vs original `NIH -> CheXpert` zero-shot: `-0.0604`
- `CheXpert` after Stage C vs Stage B: `+0.0253`
- `MIMIC` after Stage C vs original `NIH -> MIMIC` zero-shot: `+0.0400`

## Readout

- The current embedding space is strongest on `NIH`.
- Raw `NIH -> CheXpert` transfer is already stronger than the `CheXpert`-only oracle on this pilot holdout.
- The main measurable forgetting happens at the `NIH -> CheXpert` step.
- `MIMIC` adaptation is still useful because it improves `MIMIC` while adding little extra `NIH` damage.
