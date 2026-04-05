# Supervisor-Facing Results Synthesis for the NIH CXR14 Source-Stage Experiments

All statements below are derived from the local experiment artifacts only, including the original fused-lineage artifacts (`exp0007`-`exp0012`) and the April 5, 2026 rerun artifacts (`exp0013`-`exp0018`). No external sources were needed for this write-up. Metric values are reproduced exactly as reported in the experiment summaries and supporting artifacts.

## 1. Key findings

- Under the shared frozen-linear baseline protocol, the fused representation was the strongest of the three candidate embedding families. Its macro AUROC was `0.763730` on validation and `0.767933` on test, compared with `0.745876` and `0.746067` for image-only, and `0.698718` and `0.703594` for report-only.
- The fused baseline also led the other baselines on the other reported macro metrics. On test, it achieved macro average precision `0.152244`, macro ECE `0.347454`, macro `F1 @ 0.5` `0.175678`, and macro `F1 @ tuned thresholds` `0.209533`, all stronger than the image-only and report-only counterparts under the same evaluation setup.
- The margin over image-only was clear within this realized pipeline. Relative to image-only, the fused baseline improved test macro AUROC by `+0.021867`, test macro average precision by `+0.021729`, and test macro `F1 @ tuned thresholds` by `+0.024965`.
- Retrieval used as a memory-only predictor was clearly weaker than the fused supervised baseline. Validation selected `k=50` and `tau=1`, but the resulting memory-only system reached only `0.691239` macro AUROC and `0.130994` macro average precision on test, versus `0.767933` and `0.152244` for the fused baseline.
- Memory-only retrieval did show much lower macro ECE on test (`0.008359`) than the fused baseline (`0.347454`), but this coincided with extremely weak macro `F1 @ 0.5` (`0.015123`). That pattern does not support describing memory-only retrieval as simply better overall.
- Probability mixing between the fused baseline and retrieval produced a small macro-level improvement over the fused baseline alone. Validation selected `alpha=0.7`, with reported gains of `+0.000531` in macro AUROC and `+0.005153` in macro average precision relative to the fused baseline. On test, the mixed system achieved macro AUROC `0.768586`, macro average precision `0.153391`, macro ECE `0.242982`, macro `F1 @ 0.5` `0.204582`, and macro `F1 @ frozen val thresholds` `0.210987`. As reported in `exp0012`, the deltas versus the frozen baseline were `+0.000651` AUROC, `+0.001154` average precision, `-0.104474` ECE, `+0.028907` `F1 @ 0.5`, and `+0.001397` `F1 @ frozen val thresholds`.
- The mixed-model gains were not uniform across labels. On test, per-label AUROC improved for `10/14` labels and decreased for `4/14`; per-label average precision also improved for `10/14` labels and decreased for `4/14`. The macro improvement therefore should not be presented as a universal label-level gain.

## 2. Uncertainty, limitations, and cautions

- These results describe one realized source-stage pipeline on one fixed train/validation/test split of NIH CXR14 (`78,571` train, `11,219` val, `22,330` test). They do not by themselves establish robustness across seeds, resampling, or alternative splits.
- The baseline training artifacts use a single training seed, and the retrieval stages use a single realized validation-selection path. There are no confidence intervals, hypothesis tests, or multi-seed averages in the current evidence.
- The reported conclusions are limited to the active source-stage pipeline (`exp0004`-`exp0012`). They do not establish performance for any broader domain-adaptation stage beyond these artifacts.
- Threshold-tuned validation F1 should be treated as diagnostic only, because thresholds are selected on the same validation split used for reporting those validation summaries.
- On test, threshold-based F1 is frozen from validation for each method, but the threshold source differs by method: the fused baseline uses its own validation thresholds, memory-only uses the `exp0009` validation thresholds, and the mixed model uses the `exp0010` validation thresholds. For direct cross-method comparison, `F1 @ 0.5` is the cleaner threshold-controlled summary.
- Lower macro ECE should be described literally as lower macro ECE, not automatically as better calibration in a broader sense. In particular, the memory-only system combines very low ECE with near-zero `F1 @ 0.5`, so ECE should not be interpreted in isolation.
- The mixed model’s macro gain is small on discrimination metrics and not uniform across labels. Claims of broad superiority, strong robustness, or generalization beyond this setup would therefore be overstated.

## 3. Polished results write-up

The active source-stage experiments were conducted on NIH CXR14 using frozen image embeddings, frozen report embeddings, and their fused representation. The image branch produced `2048`-dimensional ResNet50 features, the report branch produced `128`-dimensional BiomedVLP CXR-BERT features, and the fused branch concatenated these into a `2176`-dimensional L2-normalized representation. A shared frozen linear multilabel probe was then trained separately on image-only, report-only, and fused embeddings, with validation macro AUROC used for checkpoint selection.

Within this common probe setting, the fused representation was the strongest of the three candidate baselines. It achieved validation macro AUROC `0.763730` and test macro AUROC `0.767933`, compared with `0.745876` and `0.746067` for image-only, and `0.698718` and `0.703594` for report-only. The fused baseline was also strongest on the other reported macro summaries, including average precision, macro ECE, `F1 @ 0.5`, and threshold-based F1. These results support selecting the fused representation as the canonical branch for downstream retrieval analysis within this source-stage pipeline.

Retrieval alone did not match the fused supervised baseline. The validation sweep over retrieval hyperparameters selected `k=50` and `tau=1`, and validation macro AUROC increased steadily with larger `k` at `tau=1`, from `0.529594` at `k=1` to `0.682669` at `k=50`. However, the final memory-only system remained substantially weaker than the fused baseline on held-out test data, with macro AUROC `0.691239` versus `0.767933`, macro average precision `0.130994` versus `0.152244`, and macro `F1 @ 0.5` `0.015123` versus `0.175678`. Although the memory-only model showed much lower macro ECE (`0.008359`), that result should be interpreted cautiously rather than as evidence that retrieval alone is preferable, because its fixed-threshold classification performance was weak.

Probability mixing provided a more favorable use of retrieval than memory-only prediction. Validation selected `alpha=0.7`, yielding a small increase in macro AUROC over the fused baseline (`0.764262` versus `0.763731`) and a larger increase in macro average precision (`0.156619` versus `0.151466`). On test, the mixed model again showed a modest macro-level improvement over the fused baseline, reaching macro AUROC `0.768586` and macro average precision `0.153391`, while also reducing macro ECE to `0.242982` and increasing macro `F1 @ 0.5` to `0.204582`. The most defensible interpretation is therefore not that retrieval replaces the supervised model, but that, within this specific frozen source-stage pipeline, retrieval contributes complementary signal when blended with the fused baseline, with a modest effect size on discrimination metrics and non-uniform label-level behavior.

## 4. Presentation recommendations

- Present the results in the following order: `Table 1` baseline comparison, one short paragraph explaining why fused was selected; `Table 2` retrieval augmentation comparison, one short paragraph explaining why memory-only is insufficient but mixing is still worth reporting; `Figure 1` validation alpha sweep; appendix tables for the full retrieval sweep and per-label deltas.
- Keep the main claim narrow: fused is the strongest representation family under the frozen-linear baseline protocol, memory-only retrieval is weaker than the fused baseline, and probability mixing yields a small improvement over the fused baseline on macro test metrics.
- Use the phrase `lower macro ECE` rather than `better calibrated` unless you add a separate calibration analysis.
- When discussing threshold-based F1 in captions or text, note that thresholds are validation-frozen but method-specific. For direct cross-method comparison, emphasize `F1 @ 0.5`.

### Suggested Table 1 Caption

Comparison of frozen linear multilabel baselines trained on image-only, report-only, and fused embeddings. The canonical branch is selected by validation macro AUROC; average precision, macro ECE, and F1 summaries are reported as supporting metrics.

### Suggested Table 1A. Validation Baseline Comparison

| Representation | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Macro F1 @ tuned thresholds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image-only | `0.745876` | `0.133964` | `0.358816` | `0.161989` | `0.199989` |
| Report-only | `0.698718` | `0.123513` | `0.394832` | `0.159970` | `0.191944` |
| Fused | `0.763730` | `0.151467` | `0.347225` | `0.175942` | `0.222298` |

### Suggested Table 1B. Test Baseline Comparison

| Representation | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Macro F1 @ tuned thresholds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image-only | `0.746067` | `0.130515` | `0.358840` | `0.160943` | `0.184568` |
| Report-only | `0.703594` | `0.129127` | `0.394382` | `0.159969` | `0.191192` |
| Fused | `0.767933` | `0.152244` | `0.347454` | `0.175678` | `0.209533` |

### Suggested Table 2 Caption

Test-set comparison of the selected fused baseline, memory-only retrieval, and probability mixing. Retrieval hyperparameters (`k=50`, `tau=1`) and the mixing weight (`alpha=0.7`) are selected on validation only. Threshold-based F1 values are validation-frozen but method-specific.

### Suggested Table 2. Retrieval Augmentation Comparison on Test

| Method | Selected config | Macro AUROC | Macro AP | Macro ECE | Macro F1 @ 0.5 | Macro F1 @ frozen val thresholds | Delta AUROC vs fused | Delta AP vs fused | Delta ECE vs fused |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fused baseline | supervised only | `0.767933` | `0.152244` | `0.347454` | `0.175678` | `0.209533` | `0.000000` | `0.000000` | `0.000000` |
| Memory-only | `k=50`, `tau=1` | `0.691239` | `0.130994` | `0.008359` | `0.015123` | `0.188889` | `-0.076694` | `-0.021250` | `-0.339095` |
| Mixed | `alpha=0.7` | `0.768586` | `0.153391` | `0.242982` | `0.204582` | `0.210987` | `+0.000651` | `+0.001154` | `-0.104474` |

Note: table entries are shown to six decimals, matching the summary artifacts. The mixed-model delta columns reproduce the reported `exp0012` deltas, so subtracting the displayed rounded absolute metrics may differ at the sixth decimal place.

### Suggested Figure 1

- Plot the validation alpha sweep from `exp0010`.
- Use `alpha` on the x-axis and macro AUROC on the y-axis.
- If space allows, add a second panel for macro average precision or macro ECE.

### Suggested Figure 1 Caption

Validation sweep for probability mixing between the fused supervised baseline and the memory-only retrieval model. The best setting is `alpha=0.7`, but the optimum is shallow rather than dramatic, indicating a modest preference for mixing over the baseline alone in this realized pipeline.

### Suggested Appendix Items

- `Appendix Table A1`: full `k`/`tau` sweep from `exp0009`, mainly to show that the retrieval result improves with larger `k` but remains well below the fused supervised baseline.
- `Appendix Table A2`: per-label mixed-versus-fused deltas on test, mainly to show that the macro gain is heterogeneous rather than universal.

## 5. Missing information or targeted follow-up questions

- No additional information is required to present the current source-stage results accurately to a supervisor.
- If you later want this adapted for a paper or slides, the only optional additions that would materially change the framing are a one-sentence project objective and any supervisor preference about whether calibration should stay in the main text or move to the appendix.

## 6. 2026-04-05 Rerun Update: 100-Epoch Fused-Baseline Rerun From `exp0003`

This addendum records a downstream rerun that started from the existing fused embedding root `exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2` and reran the source-stage pipeline from the baseline-training stage onward. No image embeddings or report embeddings were regenerated for this update.

### Rerun scope and experiment lineage

- `exp0013`: fused baseline rerun with `--epochs 100` and `--patience 4`
- `exp0014`: rebuilt train retrieval memory from the same fused `exp0003` embeddings
- `exp0015`: validation memory-only sweep
- `exp0016`: validation probability-mixing sweep
- `exp0017`: frozen memory-only test evaluation
- `exp0018`: frozen probability-mixing test evaluation

### What changed relative to the original fused lineage

- The upstream fused embeddings were unchanged. The rerun began from the existing `2176`-dimensional fused representation in `exp0003`.
- The main modeling change was the supervised baseline training budget. The original fused baseline in `exp0006` used `30` epochs; the rerun in `exp0013` used `100` epochs with `patience=4`.
- Because the embedding root was unchanged, the retrieval-only validation selection stayed effectively the same. The rerun again selected `k=50` and `tau=1` in `exp0015`.
- Probability mixing also kept the same validation-selected mixing weight. The rerun again selected `alpha=0.7` in `exp0016`.

### Comparison: original fused baseline vs 100-epoch fused baseline

| Metric | Original fused baseline (`exp0006`) | Rerun fused baseline (`exp0013`) | Delta |
| --- | ---: | ---: | ---: |
| Validation macro AUROC | `0.763730` | `0.774447` | `+0.010718` |
| Validation macro AP | `0.151467` | `0.159506` | `+0.008039` |
| Test macro AUROC | `0.767933` | `0.774642` | `+0.006708` |
| Test macro AP | `0.152244` | `0.160414` | `+0.008170` |

The rerun baseline improved materially over the original fused baseline on both validation and test discrimination metrics. The best epoch in the rerun was `100`, so the longer budget continued to help throughout the full run rather than stopping early at a much earlier checkpoint.

### Comparison: original mixed test result vs rerun mixed test result

| Metric | Original mixed test (`exp0012`) | Rerun mixed test (`exp0018`) | Delta |
| --- | ---: | ---: | ---: |
| Test macro AUROC | `0.768586` | `0.775239` | `+0.006653` |
| Test macro AP | `0.153391` | `0.159950` | `+0.006559` |
| Test macro ECE | `0.242982` | `0.227412` | `-0.015570` |
| Test macro F1 @ 0.5 | `0.204582` | `0.211526` | `+0.006944` |
| Test macro F1 @ frozen val thresholds | `0.210987` | `0.220846` | `+0.009859` |

The final mixed model therefore improved in absolute terms over the original `exp0012` result across all of the main reported macro summaries except that ECE decreased, which is favorable because lower macro ECE is better when stated literally as calibration error.

### Memory-only retrieval comparison

The memory-only branch was essentially unchanged because it depends on the same `exp0003` embeddings and the same validation-selected retrieval setting. The rerun again selected `k=50`, `tau=1`, and produced test macro AUROC `0.691239` and test macro AP `0.130995`, which is effectively identical to the original `exp0011` values up to rounding.

### Interpretation of the rerun

The strongest effect in the rerun came from improving the supervised fused baseline, not from changing the retrieval configuration. The new baseline in `exp0013` is clearly stronger than the original `exp0006`, and the final mixed system in `exp0018` is clearly stronger than the original `exp0012` in absolute terms.

However, the relationship between the rerun baseline and the rerun mixed model is more nuanced than in the original lineage. In the rerun, the mixed model still achieved a small test macro AUROC gain over the rerun baseline (`+0.000597`), but it was slightly lower on test macro average precision (`-0.000464`) than the rerun baseline alone. In other words, after strengthening the fused supervised baseline with the 100-epoch run, mixing still gave the best test AUROC, but no longer gave the best test AP.

The narrowest defensible supervisor-facing summary for the rerun is therefore:

- the 100-epoch rerun substantially strengthened the fused supervised baseline relative to the original 30-epoch baseline,
- the final mixed system also improved over the earlier mixed system in absolute terms,
- retrieval-only behavior was unchanged because the embedding root did not change,
- and in the rerun, mixing remained slightly best for test macro AUROC, while the rerun baseline alone was slightly best for test macro AP.

## 7. 2026-04-05 Expanded Retrieval-Sweep Update: Boundary Extension From The 100-Epoch Lineage

This addendum records a second downstream rerun that kept the same upstream ingredients as the `exp0013` to `exp0018` branch and changed only the retrieval-selection search space. The fused embedding root remained `exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`, and the canonical supervised baseline remained the 100-epoch fused run `exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`. No image embeddings, report embeddings, fusion recipe, or baseline-training budget were changed for this update.

### Expanded-sweep scope and experiment lineage

- `exp0019`: rebuilt the train retrieval memory from the same `exp0003` fused embeddings, aligned to the `exp0013` lineage
- `exp0020`: first expanded validation memory-only sweep over `k = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000]` and `tau = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 20, 40]`
- `exp0021`: refined validation memory-only sweep over the large-`k`, high-`tau` frontier with `k = [50, 100, 200, 500, 1000, 1500, 2000, 3000]` and `tau = [5, 10, 20, 40, 80, 160]`
- `exp0022`: validation probability-mixing sweep on top of `exp0021`
- `exp0023`: frozen memory-only test evaluation using the validation-selected `k` and `tau` from `exp0021`
- `exp0024`: frozen probability-mixing test evaluation using the validation-selected `alpha` from `exp0022`

### What changed relative to `exp0015` to `exp0018`

- The retrieval framework was preserved: FAISS nearest-neighbor lookup, the same probability construction from neighbor labels, validation-only hyperparameter selection, and frozen test application.
- The implementation of `06_evaluate_source_memory_only.py` was updated so that the sweep grid can be passed explicitly and `tau` is preserved as a float rather than being serialized through integer-only handling. This allowed direct testing of `tau < 1`.
- The first widened pass in `exp0020` already showed that the earlier `exp0015` optimum was boundary-limited: the best setting moved from `k=50, tau=1` to `k=1000, tau=40`.
- The refined pass in `exp0021` pushed the selected memory-only validation setting further to `k=3000, tau=40`.
- Despite explicitly testing smaller `tau` values than before, the validation winner did not move toward lower `tau`; in the expanded regime, the best completed setting used a much larger `k` and a relatively sharp `tau`.
- The validation-selected mixing weight remained `alpha=0.7` in `exp0022`.

### Comparison: 100-epoch retrieval branch before and after the expanded sweep

#### Memory-only validation selection

| Metric | Prior 100-epoch branch (`exp0015`) | Expanded-sweep branch (`exp0021`) | Delta |
| --- | ---: | ---: | ---: |
| Selected `k` | `50` | `3000` | `+2950` |
| Selected `tau` | `1` | `40` | `+39` |
| Validation macro AUROC | `0.682669` | `0.738898` | `+0.056230` |
| Validation macro AP | `0.128678` | `0.136206` | `+0.007527` |
| Validation macro ECE | `0.009410` | `0.006554` | `-0.002857` |
| Validation macro F1 @ 0.5 | `0.017704` | `0.001290` | `-0.016414` |

#### Memory-only test comparison

| Metric | Prior 100-epoch branch (`exp0017`) | Expanded-sweep branch (`exp0023`) | Delta |
| --- | ---: | ---: | ---: |
| Frozen `k` | `50` | `3000` | `+2950` |
| Frozen `tau` | `1` | `40` | `+39` |
| Test macro AUROC | `0.691239` | `0.745989` | `+0.054750` |
| Test macro AP | `0.130995` | `0.138152` | `+0.007157` |
| Test macro ECE | `0.008359` | `0.005927` | `-0.002432` |
| Test macro F1 @ 0.5 | `0.015123` | `0.000485` | `-0.014638` |
| Test macro F1 @ frozen val thresholds | `0.188889` | `0.191466` | `+0.002577` |

#### Probability-mixing validation comparison

| Metric | Prior 100-epoch branch (`exp0016`) | Expanded-sweep branch (`exp0022`) | Delta |
| --- | ---: | ---: | ---: |
| Selected `alpha` | `0.7` | `0.7` | `0.0` |
| Validation macro AUROC | `0.774842` | `0.774656` | `-0.000186` |
| Validation macro AP | `0.163146` | `0.160150` | `-0.002996` |
| Validation macro ECE | `0.227235` | `0.226923` | `-0.000311` |
| Validation macro F1 @ 0.5 | `0.211561` | `0.211274` | `-0.000287` |

#### Probability-mixing test comparison

| Metric | Prior 100-epoch branch (`exp0018`) | Expanded-sweep branch (`exp0024`) | Delta |
| --- | ---: | ---: | ---: |
| Frozen `alpha` | `0.7` | `0.7` | `0.0` |
| Test macro AUROC | `0.775239` | `0.775046` | `-0.000193` |
| Test macro AP | `0.159950` | `0.159121` | `-0.000829` |
| Test macro ECE | `0.227412` | `0.227055` | `-0.000357` |
| Test macro F1 @ 0.5 | `0.211526` | `0.211121` | `-0.000405` |
| Test macro F1 @ frozen val thresholds | `0.220846` | `0.221494` | `+0.000648` |

### Interpretation of the expanded sweep

The expanded search confirms that the earlier retrieval-only selection in `exp0015` was constrained by the original search boundary. Under the same frozen embedding root and the same 100-epoch fused baseline lineage, widening the sweep changed the memory-only validation winner from `k=50, tau=1` to `k=3000, tau=40`. That shift materially improved memory-only ranking metrics on both validation and test. The strongest headline gain is the memory-only test macro AUROC increase from `0.691239` to `0.745989`, with test macro AP increasing from `0.130995` to `0.138152`.

However, the expanded retrieval setting still does not make memory-only prediction a strong standalone decision rule. Its thresholded classification behavior remained poor and in some respects worsened: macro `F1 @ 0.5` on test fell from `0.015123` in `exp0017` to `0.000485` in `exp0023`, even though AUROC and AP improved. The most defensible reading is therefore that the broader retrieval sweep improved ranking quality and reduced reported macro ECE, but did not convert the retrieval-only branch into a practically useful fixed-threshold classifier.

The more important supervisor-facing result is that the broader retrieval sweep did not improve the final mixed system relative to the prior 100-epoch branch. The validation-selected mixing weight stayed at `alpha=0.7`, but both validation and test macro AUROC and macro AP for the mixed branch were slightly lower in `exp0022` and `exp0024` than in `exp0016` and `exp0018`. On held-out test data, the mixed model moved from macro AUROC `0.775239` and macro AP `0.159950` in `exp0018` to `0.775046` and `0.159121` in `exp0024`. The expanded-sweep mixed model still showed a small AUROC edge over the `exp0013` supervised baseline, but that edge remained narrow, and test macro AP remained below the baseline.

The narrowest defensible supervisor-facing summary for this update is therefore:

- widening the retrieval sweep beyond the original `k=50, tau=1` boundary materially improved memory-only AUROC and AP,
- the best completed memory-only setting moved to `k=3000, tau=40`, not to a smaller `tau`,
- the memory-only branch remained weak for fixed-threshold classification despite better ranking metrics,
- and the expanded retrieval sweep did not improve the final mixed system over the earlier 100-epoch branch in `exp0018`.

### Production recommendation for the main branch

For the production main branch, the retrieval configuration should remain the earlier 100-epoch setting selected in `exp0015`, namely `k=50` and `tau=1`, with `alpha=0.7` from `exp0016` for the mixed model. The reason is that the expanded retrieval sweep improved the retrieval-only branch in isolation, but it did not improve the final mixed system that is most relevant for deployment. Relative to the prior 100-epoch production branch in `exp0018`, the expanded-sweep mixed branch in `exp0024` was slightly lower on test macro AUROC (`0.775046` versus `0.775239`), test macro AP (`0.159121` versus `0.159950`), and test macro `F1 @ 0.5` (`0.211121` versus `0.211526`). The clearest supervisor-facing recommendation is therefore to keep the production retrieval configuration at `k=50`, `tau=1`, and `alpha=0.7`, while treating the expanded-sweep results as an analysis showing that stronger retrieval-only ranking performance does not necessarily translate into a better final mixed system.

One remaining caveat is that the selected memory-only validation setting in `exp0021` still lies on the tested `k` ceiling of `3000`. That means the retrieval frontier is not fully closed on `k`, even though the completed expanded sweep is already sufficient to show that the original `exp0015` selection was boundary-limited and that further retrieval-only improvement does not automatically translate into a better mixed system.

## 8. 2026-04-05 Fine Alpha-Sweep Update: Local Refinement On The 100-Epoch Production Branch

This addendum records a targeted follow-up run that kept the 100-epoch production retrieval branch fixed and refined only the validation-time probability-mixing weight. The upstream fused embedding root remained `exp0003__fused_embedding_generation__nih_cxr14_exp0001_exp0002_concat_l2`, the supervised baseline remained `exp0013__source_baseline_training__nih_cxr14_exp0003_fused_linear_e100_p4`, and the frozen retrieval configuration remained the original 100-epoch production setting from `exp0015`, namely `k=50` and `tau=1`. No image embeddings, report embeddings, fusion recipe, baseline-training budget, or retrieval hyperparameters were changed for this update.

### Fine-alpha scope and experiment lineage

- `exp0025`: validation probability-mixing sweep on top of the original 100-epoch production retrieval branch, using the same memory-only validation artifact as `exp0016` but with a finer alpha grid
- `exp0026`: frozen probability-mixing test evaluation using the validation-selected alpha from `exp0025`

### What changed relative to `exp0016` to `exp0018`

- The retrieval inputs were held fixed at the earlier 100-epoch production configuration: `exp0015` for validation memory probabilities and `exp0017` for frozen test memory probabilities.
- The probability-mixing script was updated to accept an explicit alpha grid so that the selection stage no longer had to be restricted to `alpha = 0.0, 0.1, ..., 1.0`.
- The fine validation sweep in `exp0025` evaluated `alpha` on the full grid `0.00, 0.01, ..., 1.00` while preserving the same validation-only selection rule as before: highest macro AUROC, then higher macro AP, then larger alpha.
- No tuning was performed on test. `exp0026` simply applied the validation-selected alpha from `exp0025` to the held-out test split.

### Fine-alpha validation results

The coarse 100-epoch production branch in `exp0016` selected `alpha=0.7`. The finer sweep in `exp0025` moved the validation-selected alpha slightly upward to `0.74`.

#### Local validation ridge around the previous optimum

| Alpha | Validation macro AUROC | Validation macro AP |
| --- | ---: | ---: |
| `0.70` | `0.774842` | `0.163146` |
| `0.71` | `0.774849` | `0.163114` |
| `0.72` | `0.774856` | `0.163099` |
| `0.73` | `0.774863` | `0.163059` |
| `0.74` | `0.774865` | `0.162973` |
| `0.75` | `0.774864` | `0.163048` |

This confirms that the optimum is real but extremely shallow. Relative to the earlier `alpha=0.7` choice, the validation macro AUROC gain at `alpha=0.74` is only `+0.000023`.

#### Comparison: coarse alpha selection vs fine alpha selection on validation

| Metric | Prior 100-epoch branch (`exp0016`, `alpha=0.7`) | Fine-alpha branch (`exp0025`, `alpha=0.74`) | Delta |
| --- | ---: | ---: | ---: |
| Selected `alpha` | `0.70` | `0.74` | `+0.04` |
| Validation macro AUROC | `0.774842` | `0.774865` | `+0.000023` |
| Validation macro AP | `0.163146` | `0.162973` | `-0.000173` |
| Validation macro ECE | `0.227235` | `0.240306` | `+0.013071` |
| Validation macro F1 @ 0.5 | `0.211561` | `0.208013` | `-0.003548` |
| Diagnostic macro F1 @ tuned thresholds | `0.234437` | `0.234481` | `+0.000044` |

### Frozen test comparison

The fine-alpha branch was then frozen and evaluated on held-out test data in `exp0026`.

| Metric | Prior 100-epoch branch (`exp0018`, `alpha=0.7`) | Fine-alpha branch (`exp0026`, `alpha=0.74`) | Delta |
| --- | ---: | ---: | ---: |
| Frozen `alpha` | `0.70` | `0.74` | `+0.04` |
| Test macro AUROC | `0.775239` | `0.775224` | `-0.000014` |
| Test macro AP | `0.159950` | `0.160043` | `+0.000092` |
| Test macro ECE | `0.227412` | `0.240454` | `+0.013041` |
| Test macro F1 @ 0.5 | `0.211526` | `0.209309` | `-0.002217` |
| Test macro F1 @ frozen val thresholds | `0.220846` | `0.220639` | `-0.000207` |

### Interpretation of the fine-alpha update

The fine sweep shows that `alpha=0.7` was not the exact validation optimum on the 100-epoch production branch. Under the same frozen retrieval setting `k=50, tau=1`, the validation-selected alpha shifted slightly to `0.74` in `exp0025`.

However, the practical effect size is negligible. The improvement at validation time is only on the order of `2e-05` macro AUROC, and the held-out test comparison does not show a meaningful global win for the refined alpha. On test, `alpha=0.74` produced slightly higher macro AP than `alpha=0.7` (`0.160043` versus `0.159950`), but slightly lower macro AUROC (`0.775224` versus `0.775239`), worse macro ECE (`0.240454` versus `0.227412`), and slightly lower threshold-based F1.

The narrowest defensible supervisor-facing summary for this update is therefore:

- the original 100-epoch production choice `alpha=0.7` was near-optimal but not exactly optimal on the coarse validation grid,
- a finer sweep moved the validation-selected alpha to `0.74`,
- the local optimum around `0.7` to `0.75` is extremely shallow,
- and the refined alpha did not produce a clear test-set improvement over the original `alpha=0.7` production branch.

### Recommendation after the fine-alpha check

If the goal is strict adherence to validation-only selection, then `exp0025` supports using `alpha=0.74` as the updated validation winner on the original 100-epoch production retrieval branch. If the goal is stable production reporting with minimal complexity, `alpha=0.7` remains fully defensible because the fine-alpha sweep showed only a negligible validation gain and no convincing test-set improvement.
