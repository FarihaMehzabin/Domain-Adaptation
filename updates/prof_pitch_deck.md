# Professor Pitch Deck

Prepared for April 21, 2026.

---

# Slide 1

## Title

**Controlled Domain Adaptation For Chest X-Ray Classification**

## Subtitle

**From NIH to CheXpert, then extension to MIMIC, using frozen CXR Foundation embeddings**

## On-slide bullets

- Goal: adapt across hospital domains without destroying earlier-domain performance.
- Core question: is target-only retraining enough, or do we need structured adaptation?
- Main result: forgetting is the key bottleneck, and selective correction improves the tradeoff.

## Speaker notes

“The main story is not that I trained one more classifier. I set up a controlled adaptation study. I first tested whether target-only training on CheXpert was enough. Then I tested warm-start adaptation, then distillation-based adaptation, and finally a selective correction model designed to reduce forgetting. I also extended the line to a third domain, MIMIC, to stress-test the methods.”

---

# Slide 2

## Title

**Experimental Setup**

## On-slide bullets

- Backbone: frozen `CXR Foundation` image embeddings.
- Labels: `7` shared findings.
- Domains:
- `D0 = NIH`
- `D1 = CheXpert`
- `D2 = MIMIC`
- Core evaluation:
- target performance: macro AUROC
- retention: source / old-domain forgetting
- sequential test: `NIH -> CheXpert -> MIMIC`

## On-slide table

| Domain | Role | Size used in current clean line |
| --- | --- | ---: |
| `NIH` | source train / retention test | train `10000`, val `1000`, test `2000` |
| `CheXpert` | target adaptation | train `1000`, val `1000`, test `234` |
| `MIMIC` | third-domain stress test | test `676` in `paper_v1` |

## Speaker notes

“I kept the backbone fixed so the question stayed focused on domain adaptation, not feature learning. The source domain is NIH, the first target is CheXpert, and then I extend the chain to MIMIC. The two main metrics I care about are target macro AUROC and forgetting on prior domains.”

---

# Slide 3

## Title

**What Each Method Actually Is**

## On-slide bullets

- `source` / `source_only`
- frozen embeddings -> single linear multilabel head
- train on `NIH` only
- `target_only`
- same linear head, random init
- train on `CheXpert` only
- `warm_start`
- same linear head
- initialize from the trained `NIH` head, then continue training on `CheXpert`
- `LwF`
- same warm-started linear head
- add a distillation loss to preserve old `NIH` predictions
- `main_method`
- frozen previous `NIH` head + prototype memory + residual adapter + learned gate
- final logits = `previous_logits + gate * residual`
- `harder_gate_clipping`
- same as `main_method`, but cap the gate on old-like examples

## On-slide equation

`main_method`: `logits = previous_logits + gate * residual`

## On-slide definitions

- **source** = the old model you are trying to preserve
- **main** = the first full continual-adaptation architecture in `paper_v1`

## Speaker notes

“This slide is here so the method names are interpretable. `source_only` is just the NIH linear head on frozen embeddings. `target_only`, `warm_start`, and `LwF` all use that same simple head, they differ in initialization and losses. The big architecture change happens at `main_method`: I freeze the old NIH head, retrieve old-domain prototype information, predict a residual correction, and use a gate to decide how much correction to apply. `harder_gate_clipping` is not a new model from scratch, it is a safety rule added to that main model.”

---

# Slide 4

## Title

**Phase A: Does Domain Adaptation Actually Help?**

## Subtitle

**Pilot adaptation ladder from the original experiment line**

## On-slide table

| Method | CheXpert AUROC | NIH AUROC | Main reading |
| --- | ---: | ---: | --- |
| `NIH source-only direct transfer` (`exp0015`) | `0.8454` | `0.8455` | strong source baseline |
| `CheXpert target-only` (`exp0055`) | `0.7702` | not evaluated | target-only is much weaker |
| `NIH -> CheXpert warm-start` (`exp0075`) | `0.7757` | `0.8112` | adaptation helps, but forgets |
| `NIH -> CheXpert + LwF` (`exp0084`) | `0.7991` | `0.8394` | best pilot adaptation result |

## On-slide takeaways

- Target-only training is not enough.
- Warm-start adaptation is better than retraining from scratch.
- The first clear failure mode is forgetting.
- `LwF` improves target AUROC and source retention at the same time.

## Speaker notes

“This is the causal ladder. The important point is that I did not jump straight to a complicated method. First I showed that target-only training underused the source model. Then I showed that warm-start adaptation helped a bit, but damaged NIH performance. Then `LwF` improved both the target result and source retention. So by this point, the case for domain adaptation was already real.”

## Artifact references

- [exp0015](/workspace/experiments/by_id/exp0015__domain_transfer_linear_probe__cxr_foundation_general_avg_pilot5h)
- [exp0055](/workspace/experiments/by_id/exp0055__domain_transfer_head_training__chexpert_target_1000_cxr_foundation_linear_gpu)
- [exp0075](/workspace/experiments/by_id/exp0075__domain_transfer_head_training__nih_warmstart_adaptation__chexpert_target_1000__cxr_foundation_linear_gpu)
- [exp0084](/workspace/experiments/by_id/exp0084__domain_transfer_head_training__nih_warmstart_lwf__chexpert_target_1000__alpha-1p0__temp-8__cxr_foundation_linear_gpu)

---

# Slide 5

## Title

**Phase B: Cleaner Rerun And Failure-Mode Diagnosis**

## Subtitle

**`paper_v1` rerun on refreshed patient-disjoint CheXpert assets**

## On-slide bullets

- I rebuilt the line in `paper_v1` with cleaner assets and a smaller, controlled method set.
- The first rich residual-gating model improved CheXpert, but underperformed on seen-average because it over-corrected.
- Mechanistic evidence:
- mean gate on `NIH test`: `0.9993`
- mean gate on `CheXpert test`: `0.9996`

## On-slide table

| Method | NIH AUROC | CheXpert AUROC | Seen-average |
| --- | ---: | ---: | ---: |
| `source_only` | `0.8448` | `0.8500` | `0.8474` |
| `main_method` | `0.8172` | `0.8551` | `0.8362` |

## On-slide takeaway

**The model was almost always opening the correction gate, even on old-like source examples.**

## Speaker notes

“This was the key diagnosis. The richer model was not failing because adaptation was a bad idea. It was failing because it corrected too aggressively. The gate was basically saturated near one on both NIH and CheXpert. That made the next question very specific: can I keep the correction idea, but suppress it on examples that already look like old-domain cases?”

## Artifact references

- [postmortem_report.md](/workspace/paper_v1/outputs/nih_to_chexpert_postmortem/20260420T100814Z__nih_to_chexpert_postmortem/reports/postmortem_report.md)
- [main_method NIH gate summary](/workspace/paper_v1/outputs/nih_to_chexpert_postmortem/20260420T100814Z__nih_to_chexpert_postmortem/reports/diagnostics/main_method__nih_test__gate_summary.json)
- [main_method CheXpert gate summary](/workspace/paper_v1/outputs/nih_to_chexpert_postmortem/20260420T100814Z__nih_to_chexpert_postmortem/reports/diagnostics/main_method__chexpert_test__gate_summary.json)

---

# Slide 6

## Title

**Selective Correction Fix: Harder Gate Clipping**

## Subtitle

**Cap corrections on old-like examples instead of correcting everything**

## On-slide bullets

- Rule: if retrieval similarity is high enough, cap the gate to `0.1`.
- Intuition: preserve old-domain behavior unless there is strong evidence that a correction is needed.
- Architecture reminder:
- frozen old NIH head + prototype memory + residual adapter + learned gate
- `harder_gate_clipping` only changes the gate behavior on old-like cases

## On-slide table

| Method | NIH AUROC mean | CheXpert AUROC mean | Seen-average mean | NIH forgetting mean |
| --- | ---: | ---: | ---: | ---: |
| `source_only` | `0.8448` | `0.8500` | `0.8474` | `0.0000` |
| `harder_gate_clipping` | `0.8403` | `0.8547` | `0.8475` | `0.0045` |
| `labelwise_trust_region` | `0.8357` | `0.8560` | `0.8459` | `0.0090` |
| `lwf` | `0.8303` | `0.8507` | `0.8405` | `0.0145` |

## On-slide bottom line

**`harder_gate_clipping` was the promoted stage-1 candidate because it was the only nontrivial method that improved CheXpert while preserving seen-average.**

## Speaker notes

“This is the core result I would emphasize tomorrow. Across three seeds, `harder_gate_clipping` slightly improved seen-average over `source_only` and raised CheXpert AUROC by about `+0.0047`. `labelwise_trust_region` got a slightly higher CheXpert number, but it lost too much on NIH and seen-average. So if I want the most defensible stage-1 adaptation candidate, it is `harder_gate_clipping`.”

## Artifact references

- [seed_summary.csv](/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/seed_summary.csv)
- [promotion_decision.json](/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/artifacts/promotion_decision.json)
- [original_vs_harder_gate.csv](/workspace/paper_v1/outputs/nih_to_chexpert_seed_sweep/20260420T122137Z__nih_to_chexpert_seed_sweep/reports/original_vs_harder_gate.csv)

---

# Slide 7

## Title

**Extension To A Third Domain: MIMIC**

## Subtitle

**Sequential `NIH -> CheXpert -> MIMIC` stress test**

## On-slide table

| Method | NIH AUROC | CheXpert AUROC | MIMIC AUROC | NIH forgetting | CheXpert forgetting |
| --- | ---: | ---: | ---: | ---: | ---: |
| `source_only` | `0.8448` | `0.8500` | `0.7344` | `0.0000` | `0.0000` |
| `LwF` | `0.8252` | `0.8471` | `0.7843` | `0.0049` | `0.0039` |
| `harder_gate_clipping` | `0.8364` | `0.8615` | `0.7565` | `0.0066` | `0.0000` |
| `vq_summary_replay` | `0.8221` | `0.8544` | `0.7607` | `0.0042` | `0.0014` |

## On-slide takeaways

- All three adaptation methods beat the `source_only` MIMIC baseline.
- `LwF` had the strongest new-domain `MIMIC` AUROC: `0.7843`.
- `harder_gate_clipping` had the strongest `CheXpert` AUROC: `0.8615`.
- `harder_gate_clipping` did **not** have zero NIH forgetting here; the zero is on **CheXpert** forgetting.
- So in stage 2, `LwF` is better for pushing into MIMIC, while `harder_gate_clipping` is better for protecting the intermediate CheXpert domain.

## Speaker notes

“I also pushed the best candidates one step further to a third domain. This matters because it shows the work was not limited to one pair of datasets. The ranking changes here: `LwF` is best on the new domain itself, which tells me distillation is still strong when moving into MIMIC. `harder_gate_clipping` is not zero-forgetting on NIH here; the zero is on CheXpert forgetting. That means it is especially good at protecting the intermediate domain while adapting onward. So my reading is: the gate idea is strongest for preservation, and MIMIC is where the next hybrid method should combine `LwF`-style new-domain strength with gate-based protection.”

## Artifact references

- [nih_to_chexpert_to_mimic_report.md](/workspace/paper_v1/outputs/nih_to_chexpert_to_mimic/20260420T165345Z__nih_to_chexpert_to_mimic_harder_gate_only/reports/nih_to_chexpert_to_mimic_report.md)
- [nih_to_chexpert_to_mimic_summary.csv](/workspace/paper_v1/outputs/nih_to_chexpert_to_mimic/20260420T165345Z__nih_to_chexpert_to_mimic_harder_gate_only/reports/nih_to_chexpert_to_mimic_summary.csv)

---

# Slide 8

## Title

**Take-Home Message**

## On-slide bullets

- I showed that domain adaptation is real in this pipeline.
- Target-only retraining was not enough.
- Warm-start and `LwF` proved that adaptation can improve target performance.
- The main bottleneck was forgetting.
- `harder_gate_clipping` is my strongest stage-1 result because it is mechanistically motivated and preserves seen-average while improving CheXpert.
- The MIMIC extension shows the line scales to a third domain, but it also reveals the next problem: combining `LwF`-style new-domain strength with gate-based preservation.

## Final line to say out loud

**“My contribution is not just another leaderboard point. It is a controlled adaptation study, a diagnosis of the forgetting mechanism, and a selective correction rule that gives the best stage-1 tradeoff in the cleaner rerun.”**

## Speaker notes

“If I have to leave the room with one sentence, it is that I did not just try adaptation, I structured and de-risked it. I now know that target-only training is insufficient, that forgetting is the central failure mode, that distillation helps, and that selective correction is the most defensible next method in the cleaner stage-1 setting. The MIMIC result shows the work is moving beyond a single transfer pair.”

---

# Optional Q&A Backup

## If asked: why does `LwF` beat `harder_gate_clipping` on MIMIC?

- Because stage 2 emphasizes moving into a new domain while retaining two earlier ones.
- `LwF` remains very strong at pushing the newest domain.
- `harder_gate_clipping` remains strongest at protecting the already-adapted intermediate domain.
- The next natural method is a hybrid that combines distillation with selective gate control.

## If asked: why is `harder_gate_clipping` still your headline method?

- It is the best promoted candidate in the 3-seed stage-1 rerun.
- It is directly tied to a diagnosed mechanism.
- It improves CheXpert without giving away seen-average.
- It is easier to defend scientifically than just saying “this method got the highest number somewhere.”
