# Oracle, Transfer, Naive Continual, LwF, and MAS on `NIH -> CheXpert -> MIMIC`

Prepared on `April 19, 2026 (UTC)`.

## Objective

This report consolidates the current `CXR Foundation` pilot into the five comparison blocks that matter for the continual-learning story:

1. independent single-domain oracle baselines
2. `NIH` cross-domain transfer with no adaptation
3. the naive continual baseline
4. sequential rehearsal-free `LwF`
5. sequential rehearsal-free `MAS`

The `LwF` and `MAS` methods here are strictly head-only, frozen-embedding, rehearsal-free baselines. They are not full-backbone continual-learning methods.

## Shared setup

- Backbone and features:
  - frozen `CXR Foundation`
  - `general` embeddings
  - `avg` token pooling
  - `768`-dimensional features
- Trainable model:
  - multilabel linear head only
  - `5.4k` trainable parameters
  - about `10.8k` FLOPs per image for the trainable `768 -> 7` classifier
  - full backbone FLOPs were not logged because all runs used pre-exported embeddings
- Optimization:
  - `AdamW`
  - learning rate `1e-3`
  - weight decay `1e-4`
  - batch size `512`
  - early stopping patience `5`
- Label space:
  - common `7` labels
- Data scale:
  - `NIH`: `10000 train / 1000 val / 2000 test`
  - `CheXpert`: `1000 train / 1000 val / 234 test`
  - `MIMIC`: `1000 train / 1000 val / 676 test`
  - `MIMIC` train embeddings contain `998` rows because `2` corrupt JPGs were skipped during export

## 1. Independent Single-Domain Oracle Baselines

| Train rule | Run | Best epoch | Same-domain test AUROC |
| --- | --- | ---: | ---: |
| `NIH` only | `exp0001` | `24` | `0.8461` |
| `CheXpert` only | `exp0004` | `50` | `0.7703` |
| `MIMIC` only | `exp0005` | `24` | `0.7619` |

These are the present same-domain ceilings of the frozen embedding space under a simple linear head.

## 2. Cross-Domain Transfer Baseline

| Train domain | Test domain | Run | Test AUROC |
| --- | --- | --- | ---: |
| `NIH` | `NIH` | `exp0001` | `0.8461` |
| `NIH` | `CheXpert` | `exp0001` | `0.8486` |
| `NIH` | `MIMIC` | `exp0001` | `0.7370` |

Relative to oracle:

- `NIH -> CheXpert` minus `CheXpert` oracle: `+0.0783`
- `NIH -> MIMIC` minus `MIMIC` oracle: `-0.0249`

The asymmetry remains important:

- `CheXpert` is already highly aligned with the `NIH` source head
- `MIMIC` still shows a real transfer gap

## 3. Naive Continual Baseline

The baseline chain is:

1. train a source head on `NIH`
2. fine-tune on `CheXpert`
3. fine-tune that head on `MIMIC`

The original Stage-B run had been missing the report-only `MIMIC` cell. That has now been backfilled from the saved Stage-B checkpoint with no retraining, using the same saved thresholds and the current all-domain evaluation plan.

| Stage | Run | Best epoch | NIH test | CheXpert test | MIMIC test |
| --- | --- | ---: | ---: | ---: | ---: |
| Stage A: `NIH` source | `exp0001` | `24` | `0.8461` | `0.8486` | `0.7370` |
| Stage B: after `CheXpert` fine-tune | `exp0002` | `24` | `0.8210` | `0.7882` | `0.7638` |
| Stage C: after `MIMIC` fine-tune | `exp0003` | `6` | `0.8221` | `0.8135` | `0.7770` |

Key deltas:

| Quantity | Delta AUROC |
| --- | ---: |
| `NIH` after `CheXpert` vs Stage A | `-0.0251` |
| final `NIH` vs Stage A | `-0.0240` |
| `CheXpert` after Stage B vs Stage A zero-shot | `-0.0604` |
| `CheXpert` after Stage C vs Stage A zero-shot | `-0.0351` |
| `CheXpert` after Stage C vs Stage B | `+0.0253` |
| `MIMIC` after Stage B vs Stage A zero-shot | `+0.0268` |
| `MIMIC` after Stage C vs Stage A zero-shot | `+0.0400` |

So the main forgetting event is still the `NIH -> CheXpert` step. The later `MIMIC` step is useful for `MIMIC` and does not add much extra `NIH` damage.

## 4. Sequential Rehearsal-Free `LwF`

`LwF` here is the strict current-input version:

- Stage B trains on `CheXpert` only and distills teacher `A` on `CheXpert` inputs only
- Stage C trains on `MIMIC` only and distills teacher `B` on `MIMIC` inputs only
- no old-domain replay is used

Sweep grid:

- `alpha in {0.25, 0.5, 1.0}`
- `temperature in {2, 4, 8}`

| Setting | Stage B NIH test | Stage B CheXpert test | Stage B MIMIC test | Stage C NIH test | Stage C CheXpert test | Stage C MIMIC test | Val Pareto | Test Pareto |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `alpha=0.25, T=2` | `0.8211` | `0.7879` | `0.7642` | `0.8155` | `0.8055` | `0.7709` | no | no |
| `alpha=0.25, T=4` | `0.8216` | `0.7884` | `0.7643` | `0.8167` | `0.8076` | `0.7717` | no | no |
| `alpha=0.25, T=8` | `0.8217` | `0.7884` | `0.7643` | `0.8172` | `0.8079` | `0.7722` | no | no |
| `alpha=0.5, T=2` | `0.8267` | `0.7963` | `0.7650` | `0.8218` | `0.8102` | `0.7748` | no | no |
| `alpha=0.5, T=4` | `0.8273` | `0.7965` | `0.7648` | `0.8230` | `0.8132` | `0.7751` | yes | no |
| `alpha=0.5, T=8` | `0.8275` | `0.7964` | `0.7648` | `0.8235` | `0.8143` | `0.7754` | yes | no |
| `alpha=1.0, T=2` | `0.8334` | `0.8092` | `0.7649` | `0.8296` | `0.8206` | `0.7761` | no | no |
| `alpha=1.0, T=4` | `0.8339` | `0.8097` | `0.7649` | `0.8311` | `0.8227` | `0.7767` | yes | no |
| `alpha=1.0, T=8` | `0.8341` | `0.8095` | `0.7649` | `0.8316` | `0.8230` | `0.7768` | yes | yes |

Readout:

- The strongest test setting is `alpha=1.0, T=8`.
- Final test AUROC at that point is `NIH 0.8316`, `CheXpert 0.8230`, `MIMIC 0.7768`.
- Relative to the naive final stage, that is:
  - `NIH +0.0095`
  - `CheXpert +0.0095`
  - `MIMIC -0.0002`
- Relative to the naive Stage-B checkpoint, the same setting improves:
  - `NIH +0.0131`
  - `CheXpert +0.0212`
  - `MIMIC +0.0011`

So `LwF` does help. It reduces forgetting and improves `CheXpert`, but it does not materially improve the final `MIMIC` outcome relative to the naive chain.

## 5. Sequential Rehearsal-Free `MAS`

`MAS` here is strict saved-state online accumulation:

- after Stage A, compute `omega_NIH` on `NIH train`, save `anchor_A + omega_NIH`
- Stage B trains on `CheXpert` only with a penalty to `anchor_A`
- after Stage B, compute `omega_CheXpert` on `CheXpert train`, normalize, accumulate, save `anchor_B + omega_total`
- Stage C trains on `MIMIC` only using the saved accumulated state
- no old-domain data are reopened after a stage is finished

Sweep grid:

- `lambda in {0.1, 0.3, 1.0, 3.0, 10.0, 30.0}`

| Lambda | Stage B NIH test | Stage B CheXpert test | Stage B MIMIC test | Stage C NIH test | Stage C CheXpert test | Stage C MIMIC test | Val Pareto | Test Pareto |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `0.1` | `0.8280` | `0.7978` | `0.7687` | `0.8236` | `0.8102` | `0.7771` | yes | no |
| `0.3` | `0.8359` | `0.8125` | `0.7683` | `0.8341` | `0.8225` | `0.7791` | yes | yes |
| `1.0` | `0.8426` | `0.8319` | `0.7632` | `0.8414` | `0.8356` | `0.7722` | yes | yes |
| `3.0` | `0.8454` | `0.8404` | `0.7546` | `0.8453` | `0.8397` | `0.7622` | yes | yes |
| `10.0` | `0.8465` | `0.8450` | `0.7453` | `0.8468` | `0.8449` | `0.7502` | yes | yes |
| `30.0` | `0.8461` | `0.8472` | `0.7411` | `0.8465` | `0.8475` | `0.7444` | yes | yes |

Readout:

- All six `MAS` settings lie on the validation Pareto frontier because the sweep traces a clean retention-versus-adaptation tradeoff.
- The best balanced point is `lambda=0.3`.
- Final test AUROC at `lambda=0.3` is `NIH 0.8341`, `CheXpert 0.8225`, `MIMIC 0.7791`.
- Relative to the naive final stage, `lambda=0.3` gives:
  - `NIH +0.0120`
  - `CheXpert +0.0090`
  - `MIMIC +0.0020`

The retention-first end of the curve is also informative:

- At `lambda=30`, final test AUROC is `NIH 0.8465`, `CheXpert 0.8475`, `MIMIC 0.7444`.
- Relative to the naive final stage, that is:
  - `NIH +0.0244`
  - `CheXpert +0.0340`
  - `MIMIC -0.0326`
- Relative to the original Stage-A source model, `lambda=30` is still:
  - `NIH +0.0003`
  - `CheXpert -0.0011`
  - `MIMIC +0.0074`

So high-`lambda` `MAS` can almost perfectly preserve the original `NIH` source model and the original `NIH -> CheXpert` transfer behavior, but it does so by giving up much of the extra `MIMIC` adaptation.

## Main interpretation

The five-block picture is now clean:

1. The oracle ceilings remain `NIH 0.8461`, `CheXpert 0.7703`, `MIMIC 0.7619`.
2. Raw `NIH -> CheXpert` transfer is already unusually strong at `0.8486`.
3. The naive continual chain forgets mostly at the `NIH -> CheXpert` step, then gains on `MIMIC`.
4. Rehearsal-free `LwF` improves over naive, but only modestly.
5. Rehearsal-free `MAS` gives the stronger and more interpretable tradeoff curve.

If one practical rehearsal-free baseline has to be emphasized, the best choice is `MAS lambda=0.3`:

- it improves all three final test domains versus the naive final stage
- it beats the best `LwF` point on `NIH` and `MIMIC`
- it stays essentially tied with the best `LwF` point on `CheXpert`

If a retention-first baseline is needed, `MAS lambda=30` is the right point to show:

- it nearly restores the original Stage-A `NIH` performance
- it nearly restores the original `NIH -> CheXpert` zero-shot transfer
- it still remains above the original `NIH -> MIMIC` zero-shot baseline, although below the naive adapted `MIMIC` result

## Artifact locations

- naive baseline campaign:
  - `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h`
- regularizer sweep campaign:
  - `/workspace/experiments/campaigns/10_sequential_regularizer_sweeps_pilot5h`
- merged manifest:
  - `/workspace/manifest/manifest_pilot5h_binary_mimic.csv`
- trainer and study scripts:
  - `/workspace/scripts/15_train_domain_transfer_linear_probe.py`
  - `/workspace/scripts/36_run_sequential_warmstart_forgetting_study.py`
  - `/workspace/scripts/37_run_sequential_regularizer_sweeps.py`
  - `/workspace/scripts/38_compute_online_mas_state.py`
  - `/workspace/scripts/39_backfill_domain_transfer_checkpoint_eval.py`
