# Policy B Common5 Transfer: Systematic Failure of Lightweight Adaptation on Test AUROC

## Executive Summary

To evaluate whether lightweight hospital-specific adaptation could improve transfer from NIH to MIMIC under Policy B, I ran a controlled **official k=20 comparison series** using the same source checkpoint, the same held-out MIMIC test set, and the same five-label task. The result is negative but informative:

**none of the adaptation variants improved test macro AUROC over the source-only baseline.**

This is not a case where one weak run failed. I tested progressively more flexible adaptation strategies, increasing the trainable budget from **5,125 parameters** to **201,221 parameters**, and expanding the adaptation path from **classifier only** to **LoRA on the last dense block**, and finally to **LoRA + classifier-head adaptation + BatchNorm affine adaptation**. Despite that progression, **0 of 4 adaptation runs beat the source-only model on held-out test macro AUROC**.

To keep this comparison clean, I am showing only the **official k=20 series** here and omitting the earlier k=5 exploratory run.

## Experimental Setting

### Model

- Backbone: **DenseNet-121**
- Transfer setting: **NIH-trained source model evaluated/adapted on MIMIC-CXR**
- Task: **5-label multilabel chest X-ray classification**
- Labels: **Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion**
- Input resolution: **224 x 224**
- Base model size: **6,958,981 parameters**
- Base forward-pass compute: **5.666 GFLOPs per image**

### Source Dataset: NIH ChestXray14 dev-2k

The source checkpoint used in every transfer run was trained on a **2,000-image NIH subset** for the same five-label problem.

- Source train split: **1,400 images**, **395 patients**
- Source validation split: **200 images**, **66 patients**
- Source test split: **400 images**, **82 patients**
- Split integrity: **0 patient overlap** and **0 image overlap** across train, validation, and test
- Views: **frontal AP/PA only**
- Source train view mix: **924 PA**, **476 AP**
- Source validation view mix: **157 PA**, **43 AP**
- Source test view mix: **216 AP**, **184 PA**

Source label counts:

- Train: Atelectasis **129**, Cardiomegaly **25**, Consolidation **75**, Edema **30**, Effusion **174**
- Validation: Atelectasis **29**, Cardiomegaly **13**, Consolidation **4**, Edema **7**, Effusion **29**
- Test: Atelectasis **31**, Cardiomegaly **7**, Consolidation **30**, Edema **14**, Effusion **86**

The source checkpoint itself was not trivial or degenerate on NIH:

- NIH validation macro AUROC: **0.6843**
- NIH test macro AUROC: **0.6574**

This matters because the downstream MIMIC comparison is not starting from an untrained or obviously broken model. It starts from a source model that has already learned the intended five-label NIH task to a reasonable degree, then asks whether lightweight target adaptation can improve cross-hospital transfer.

### Target Dataset: MIMIC-CXR Policy B Common5

- Support set: **30 images**, **29 subjects**, **30 studies**
- Validation set: **958 images**, **358 subjects**, **958 studies**
- Test set: **596 images**, **268 subjects**, **596 studies**
- Label policy: **Policy B (`uignore_blankzero`)**
- Evaluation is **mask-aware**, so metrics are computed only on valid label observations

Across the held-out **test set**, there are **2,854 valid label observations** out of **2,980 possible label slots**, with **126 masked observations**.

Test-set label composition after masking:

- Atelectasis: **170 positive / 394 negative** among **564 valid**
- Cardiomegaly: **159 positive / 424 negative** among **583 valid**
- Consolidation: **56 positive / 530 negative** among **586 valid**
- Edema: **128 positive / 424 negative** among **552 valid**
- Effusion: **219 positive / 350 negative** among **569 valid**

### Training Protocol

- Same NIH source checkpoint for every adaptation run
- Same official Policy B k=20 support subset for adaptation
- Same held-out MIMIC test split for final comparison
- Maximum **50 epochs**
- **Patience 10**
- **Batch size 8**
- **Seed 2027**

Checkpoint selection was validation-based during training. The head-only and pure-LoRA runs selected the best checkpoint by **validation macro AUROC**, whereas the final LoRA + BN-affine + classifier-head variant selected by **validation macro AUPRC**. The comparison below is reported **only in terms of held-out test macro AUROC**.

## Adaptation Attempts and Test AUROC

| Configuration | What Was Adapted | Key Settings | Trainable Params | Trainable % | Total Params | Forward FLOPs | Test Macro AUROC | Delta vs Source-Only |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Source-only baseline** | No target adaptation | Direct NIH-to-MIMIC transfer | 0 | 0.000% | 6,958,981 | 5.666 G | **0.618335** | **0.000000** |
| **Head-only adaptation** | Classifier head only | Full classifier update on k=20 support set | 5,125 | 0.074% | 6,958,981 | 5.666 G | 0.617968 | -0.000367 |
| **Pure LoRA adaptation** | LoRA on DenseNet denseblock4 + classifier LoRA | rank 4, alpha 4, dropout 0.0 | 87,060 | 1.236% | 7,046,041 | 5.674 G | 0.618292 | -0.000043 |
| **Higher-capacity LoRA adaptation** | LoRA on DenseNet denseblock4 + classifier LoRA | rank 8, alpha 16, lr 3e-4, dropout 0.0 | 174,120 | 2.441% | 7,133,101 | 5.683 G | 0.618236 | -0.000099 |
| **LoRA + BN-affine + classifier-head adaptation** | LoRA on denseblock4, full classifier head, BN affine in denseblock4 and norm5 | rank 8, alpha 16, dropout 0.05, BN stats frozen, differential learning rates | 201,221 | 2.824% | 7,124,869 | 5.683 G | 0.618156 | -0.000178 |

Notes on the final configuration:

- LoRA parameters: **165,888**
- Classifier parameters: **5,125**
- BN affine parameters: **30,208**
- Original non-LoRA convolution weights remained frozen

In ranking terms, the held-out test macro AUROC order is:

1. **Source-only baseline**: 0.618335
2. **Pure LoRA adaptation**: 0.618292
3. **Higher-capacity LoRA adaptation**: 0.618236
4. **LoRA + BN-affine + classifier-head adaptation**: 0.618156
5. **Head-only adaptation**: 0.617968

## What This Table Shows

The important observation is not simply that the runs failed. The important observation is that they failed **after increasing both adaptation flexibility and trainable capacity**.

The sequence was:

1. **Minimal adaptation**: classifier-only updating
2. **Structured low-rank adaptation**: LoRA on the last DenseNet block
3. **Higher-capacity low-rank adaptation**: larger LoRA rank and learning-rate adjustment
4. **More permissive lightweight adaptation**: LoRA plus full classifier-head updates plus BN affine adaptation

Even after moving from **0.074% trainable parameters** to **2.824% trainable parameters**, the best adaptation result still remained **below the source-only baseline on test AUROC**.

The closest any adaptation came to matching the baseline was the pure LoRA setting, but even that remained **slightly below baseline by 0.000043 test AUROC**. The more permissive LoRA + BN-affine + classifier variant still did not cross the baseline, despite explicitly allowing adaptation of the classifier and normalization pathway that pure LoRA had left frozen.

## Why This Is a Real Result, Not Just an Unlucky Run

This negative result is credible for several reasons:

- The comparison is not based on a single adaptation attempt; it spans **four distinct adaptation configurations** in the official k=20 setting.
- The adaptation budgets range from **5 thousand** to **201 thousand** trainable parameters.
- The compute budget also increases slightly across LoRA variants, from **5.666 GFLOPs** to about **5.683 GFLOPs**, so the conclusion is not tied to one exact parameterization.
- The last variant explicitly opens additional adaptation paths by allowing **classifier-head changes** and **BatchNorm affine changes**, yet still does not improve test AUROC.

For the LoRA family specifically, prior diagnostics also indicated that the adapters were functioning as intended: the implementation was valid, LoRA tensors were nonzero, training loss decreased, and predictions changed across epochs. In other words, the failure is unlikely to be explained by a broken training path alone.

## Interpretation

The cleanest interpretation is:

> Under Policy B, lightweight adaptation methods were tried repeatedly and systematically, but none of them improved held-out test macro AUROC over the NIH source model.

This is a useful research result because it narrows the search space. It suggests that the limitation is not simply "we forgot to tune learning rate once" or "we only tried one weak baseline." Instead, the evidence now points to a more substantive conclusion:

**for this transfer setup, small or moderately sized adaptation routes appear too constrained to deliver a reliable test-AUROC gain.**

This is especially important because earlier full fine-tuning experiments indicated that adaptation is possible in principle. That makes the present negative result more meaningful: it suggests that the issue is not whether domain adaptation can help at all, but whether these **restricted adaptation mechanisms** are expressive enough for the hospital shift in this problem.

## Bottom-Line Message for Discussion

If this needs to be stated plainly in a meeting:

> I ran a controlled official k=20 transfer series on Policy B using a DenseNet-121 source model and four different lightweight adaptation strategies. None of the adaptation strategies beat the source-only baseline on held-out test macro AUROC, even after increasing trainable capacity from 5,125 to 201,221 parameters and expanding the adaptation path to include LoRA, classifier-head updates, and BatchNorm affine adaptation.

That is a failed outcome from an optimization standpoint, but it is a strong result from an experimental standpoint: it shows that the failure is **systematic**, not anecdotal.
