# Experiment Storyline

This document organizes the experiment results into a coherent narrative rather than a single mixed leaderboard.

## Main Principle

The clean story is the current **7-label image-only domain-transfer line** on the fixed `pilot5h` split:

1. pick the right frozen image backbone
2. test whether target-only supervision is enough
3. test whether warm-start adaptation from the source head is better than target-only
4. test whether `LwF` preserves the warm-start benefit while reducing forgetting

Everything else should be treated as either:

- a control
- an exploratory side branch
- or appendix material

## Why This Ordering Works

This ordering makes each stage answer one concrete question:

1. **Backbone selection**
   Which frozen image representation is worth carrying forward?
2. **Target-only learning**
   If we keep that backbone fixed, can a fresh target-domain head solve CheXpert with limited labels?
3. **Warm-start adaptation**
   Is it better to start from the strong NIH head instead of throwing it away?
4. **LwF**
   Can we keep the benefit of warm-starting without paying the forgetting penalty?

That makes the comparisons causal and interpretable. The later runs are then motivated by the earlier ones.

## Main Story

### 1. Choose The Source Backbone First

Use:

- `exp0013`: `ResNet50` image-only frozen linear probe
- `exp0015`: `CXR Foundation` image-only frozen linear probe

Shared setup:

- manifest: `manifest_common_labels_pilot5h.csv`
- labels: 7 shared labels
  - `atelectasis`
  - `cardiomegaly`
  - `consolidation`
  - `edema`
  - `pleural_effusion`
  - `pneumonia`
  - `pneumothorax`
- domains:
  - `D0 = NIH`
  - `D1 = CheXpert`
  - `D2 = MIMIC`
- split sizes:
  - `D0 train = 10,000`
  - `D0 val = 1,000`
  - `D0 test = 2,000`
  - `D1 val = 234`
  - `D2 test = 1,455`

Backbone details:

- `exp0013`
  - embeddings: frozen `ResNet50`, `avg` pooled, `2048`-d
- `exp0015`
  - embeddings: frozen `CXR Foundation`, `general`, `avg` pooled, `768`-d

Key results:

- NIH test AUROC: `0.7306 -> 0.8455`
- CheXpert transfer AUROC: `0.7218 -> 0.8454`
- MIMIC transfer AUROC: `0.4996 -> 0.5007`

Meaning:

- `CXR Foundation` is the correct source image backbone.
- The stronger backbone clearly matters for NIH and CheXpert.
- MIMIC remains unsolved even with the better backbone.

Supervisor-facing message:

- do not spend time optimizing the weak `ResNet50` line as the primary path
- carry `CXR Foundation` forward as the main frozen image representation

### 2. Test Target-Only Learning On The Chosen Backbone

Use:

- `exp0051`: CheXpert target-only `250/250`
- `exp0053`: CheXpert target-only `500/500`
- `exp0055`: CheXpert target-only `1000/1000`

Shared setup:

- frozen `CXR Foundation` image embeddings
- `general`, `avg`, `768`-d
- train a fresh linear multilabel head on target labels only
- evaluate on the same `234`-example CheXpert holdout

Target split sizes:

- `250 train / 250 val / 234 test`
- `500 train / 500 val / 234 test`
- `1000 train / 1000 val / 234 test`

Key results:

- CheXpert test AUROC:
  - `250-shot`: `0.7532`
  - `500-shot`: `0.7611`
  - `1000-shot`: `0.7702`
- progression:
  - `0.7532 -> 0.7702`

Meaning:

- more target labels help
- but the gains are moderate
- even `1000` target examples do not beat the source-trained `CXR Foundation` NIH head on the same CheXpert holdout

Why this step must come before warm-start adaptation:

- it creates the correct baseline for adaptation
- otherwise `exp0075` has no clean comparison target

Supervisor-facing message:

- a fresh target-only head is not enough
- the correct next move is to adapt the strong source head rather than replace it

### 3. Test Warm-Start Adaptation From The NIH Head

Use:

- `exp0055`: best target-only baseline
- `exp0075`: NIH-to-CheXpert warm-start adaptation

Shared setup:

- frozen `CXR Foundation` image embeddings
- `general`, `avg`, `768`-d
- start from the existing NIH-trained source head from `exp0015`
- adapt on:
  - `CheXpert train = 1000`
  - `CheXpert val = 1000`
  - `CheXpert test = 234`
- extra evaluation on `NIH test = 2000`

Key results:

- CheXpert test AUROC: `0.7702 -> 0.7757`
- CheXpert test AP: `0.4715 -> 0.4770`
- NIH test AUROC: `0.8455 -> 0.8112`
- NIH test AP: `0.2541 -> 0.2188`

Meaning:

- warm-starting from the strong NIH head is better than retraining a fresh CheXpert-only head
- but the gain is small
- and it comes with clear source forgetting

This is the pivot point in the story:

- adaptation is directionally right
- but naive adaptation is not good enough

Supervisor-facing message:

- warm-start adaptation is better than target-only replacement
- the real problem is not whether to adapt
- the real problem is how to adapt without forgetting

### 4. Use LwF To Reduce Forgetting While Preserving Adaptation Gains

Use:

- `exp0075`: warm-start no-LwF baseline
- `exp0076` to `exp0084`: warm-start + `LwF`

Shared setup:

- same frozen `CXR Foundation` embeddings
- same NIH-trained source head initialization
- same CheXpert `1000/1000/234` adaptation split
- same NIH source supervision for the distillation branch

Sweep:

- `alpha in {0.25, 0.5, 1.0}`
- `temperature in {2, 4, 8}`

Best results in the sweep:

- best target AUROC:
  - `exp0083`: CheXpert test AUROC `0.7991`
- best retention / near-tied target:
  - `exp0084`: CheXpert test AUROC `0.7991`, NIH test AUROC `0.8394`

Main comparison against `exp0075`:

- CheXpert test AUROC: `0.7757 -> 0.7991`
- CheXpert test AP: `0.4770 -> 0.4814` at best
- NIH test AUROC: `0.8112 -> 0.8394`
- NIH test AP: `0.2188 -> 0.2447`

Meaning:

- `LwF` is the first adaptation improvement that clearly moves both target performance and source retention in the right direction at the same time
- the main weakness of warm-start adaptation was not adaptation itself
- the main weakness was forgetting

Important limitation:

- even the best `LwF` runs still do not beat the original direct-transfer `exp0015` source model on the main CheXpert ranking metrics

Supervisor-facing message:

- `LwF` is a real improvement over naive warm-start adaptation
- it materially reduces the forgetting penalty
- it is the best current adaptation recipe in this line
- but it is not yet the final answer

## The Clean Main Conclusion

The most coherent storyline is:

1. `CXR Foundation` is the right source backbone.
2. Target-only training improves with more labels, but is still not enough.
3. Warm-starting from the NIH head is better than target-only replacement.
4. The main failure mode of warm-start adaptation is forgetting.
5. `LwF` substantially fixes that failure mode and gives the best current adaptation results.
6. MIMIC remains essentially unsolved in this source-only pilot framing.

## Where The Other Runs Belong

### Useful Control, But Not Main Story

`exp0027` ViT LoRA pilot should be presented only as a control.

Why:

- it helps show that the `CXR Foundation` gain is not just because the baseline was weak
- but it does not change the main conclusion

Key numbers:

- NIH test AUROC:
  - `ResNet50 linear`: `0.7306`
  - `ViT LoRA`: `0.7819`
  - `CXR Foundation linear`: `0.8455`

Meaning:

- generic ViT + LoRA is better than weak frozen ResNet
- but still clearly below `CXR Foundation`

Recommendation:

- keep this as a side control or appendix figure
- do not let it interrupt the main adaptation narrative

### Appendix Only

The older **14-label source-stage multimodal / retrieval / xattn / hybrid** line should not be mixed into the main 7-label domain-transfer story.

Why:

- different task framing
- different label space
- different data split
- different representation families
- different goal

If included at all, present it as a separate appendix storyline:

1. fused beats image-only
2. longer xattn beats first xattn
3. hybrid beats long xattn
4. gated-hybrid is informative but not best

That branch is exploratory and useful, but it does not belong in the core adaptation narrative.

## What To Exclude From The Main Story

Do not center the story on:

- memory-only retrieval runs
- validation-only wins
- tiny AUROC changes that do not change interpretation
- the 1-epoch xattn branch
- mixed comparisons across incompatible setups

These runs can remain in the repo and appendix, but they should not define the headline.

## Suggested Supervisor Slide Order

1. **Backbone choice**
   `ResNet50` vs `CXR Foundation`
2. **Target-only scaling**
   `250 -> 500 -> 1000` CheXpert labels
3. **Warm-start adaptation**
   `exp0055 -> exp0075`
4. **LwF sweep**
   `exp0075 -> exp0083/exp0084`
5. **Takeaway / open problem**
   strong CheXpert adaptation progress, MIMIC still unsolved

## One-Sentence Version

The meaningful story is: choose the right source backbone, show that target-only training is insufficient, show that warm-start adaptation is better but forgets, then show that `LwF` is the first method that improves target performance and source retention at the same time.
