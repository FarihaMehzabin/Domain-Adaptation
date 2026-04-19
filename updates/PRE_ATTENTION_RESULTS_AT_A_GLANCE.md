# Pre-Attention Results Summary

Scope:
- This table summarizes the dated result updates that appear before the attention sections begin in `/workspace/SUPERVISOR_RESULTS_SYNTHESIS.md`.
- Attention-related sections start at `## 11. 2026-04-10 Full-Pipeline XAttn Branch`, so everything from `## 11` onward is excluded.
- I also excluded `## 10` because it is only a chronology correction, not a result update.
- The rows are not perfectly apples-to-apples. There are three main settings in the file:
  - full `NIH CXR14` source-stage, `14` labels
  - `pilot5h` image-only transfer setup, common `7` labels
  - target-domain `CheXpert` or `MIMIC` adaptation subsets

## Best Result At A Glance

| What you mean by "best" | Best result | Why |
| --- | --- | --- |
| Highest pre-attention AUROC overall | `0.8455` on `NIH test` and `0.8454` on `CheXpert` transfer | Strong `CXR Foundation` image-only source baseline on the `pilot5h` common-7 setup |
| Best pre-attention `CheXpert` adaptation result | `0.7991` `CheXpert test AUROC`, best AP `0.4814` | `Learning without Forgetting` was the strongest adaptation recipe before attention models |
| Best pre-attention `MIMIC` number | `0.5215` `MIMIC AUROC`, `0.1334` AP | This came from a memory-only retrieval variant, but `F1@0.5 = 0.0000`, so it is not a practically convincing winner |
| Best pre-attention full source-stage result | `0.775239` test macro AUROC, `0.159950` test macro AP | Strongest mixed result in the older full `NIH CXR14` 14-label source-stage line |

## Date-Ordered Table

| Date | Update | Setting | Clear result summary | Main takeaway |
| --- | --- | --- | --- | --- |
| `2026-04-05` | 100-epoch fused-baseline rerun | Full `NIH CXR14` source-stage, `14` labels | Baseline improved to `0.774642` test macro AUROC and `0.160414` AP. Final mixed system reached `0.775239` AUROC and `0.159950` AP. | This was the strongest AUROC result in the older full source-stage line before attention models. |
| `2026-04-05` | Expanded retrieval sweep | Full `NIH CXR14` source-stage, `14` labels | Memory-only retrieval improved a lot to `0.745989` AUROC and `0.138152` AP, but the final mixed system was slightly worse than the earlier 100-epoch branch at `0.775046` AUROC and `0.159121` AP. | Better retrieval-only ranking did not translate into a better final mixed model. |
| `2026-04-05` | Fine alpha sweep | Full `NIH CXR14` source-stage, `14` labels | Refining the mixing weight produced `0.775224` test macro AUROC and `0.160043` AP, which was essentially tied with the earlier branch. | The effect was negligible. |
| `2026-04-05` | Fusion-weight sweep | Full `NIH CXR14` source-stage, `14` labels | Downweighting text improved validation. The mixed winner reached `0.774877` test macro AUROC and `0.160280` AP. The best baseline AP in this sweep was `0.160831`. | Informative sensitivity check, but not a decisive upgrade over the earlier 100-epoch branch. |
| `2026-04-11` | Pilot image-only domain-transfer comparison | `pilot5h`, common `7` labels, source-trained direct transfer | `CXR Foundation` clearly beat `ResNet50`: `NIH test AUROC 0.8455` vs `0.7306`, `CheXpert AUROC 0.8454` vs `0.7218`. `MIMIC` stayed near chance for both: `0.5007` vs `0.4996`. | Strongest simple image-only baseline; good on `NIH` and `CheXpert`, not enough for `MIMIC`. |
| `2026-04-11` | Source-trained MLP sweep | Same `pilot5h` common-7 setup | MLPs improved source fitting a bit, but they did not beat the linear `CXR Foundation` result on `CheXpert`. Best CXR MLP reached `0.8473` on `NIH` and `0.8369` on `CheXpert`; `MIMIC` stayed near chance. | A more expressive head helped source fit more than transfer. |
| `2026-04-11` | Image-only ViT LoRA pilot | Same `pilot5h` common-7 setup | ViT LoRA reached `0.7819` on `NIH`, `0.7299` on `CheXpert`, and `0.4997` on `MIMIC`. | Better than the old `ResNet50` baseline, still clearly weaker than `CXR Foundation`. |
| `2026-04-12` | Retrieval-augmented comparison on `CXR Foundation` vs `ResNet50` | Same `pilot5h` common-7 setup, retrieval added | `CXR Foundation` baseline and mixed variants stayed strongest on `NIH` and `CheXpert`, both essentially tied around `0.845`. Best `MIMIC` number in the table was `0.5215` AUROC and `0.1334` AP, but it came from `ResNet50` memory-only retrieval with `F1@0.5 = 0.0000`. | Retrieval did not materially improve the already-strong `CXR Foundation` transfer story, and it did not solve `MIMIC`. |
| `2026-04-12` | Few-shot `CheXpert` target-only linear probes | `CheXpert` target subsets, same `234`-example holdout | Best `CheXpert` AUROC was `0.7702` with `1000/1000` train/val. Best AP was `0.4764` with `250/250`. All target-only runs stayed below the source-trained `CXR Foundation` baseline at `0.8454` AUROC and `0.5430` AP. | More `CheXpert` labels helped, but not enough to beat the source model. |
| `2026-04-13` | Incremental `NIH -> CheXpert` warm-start adaptation | `CheXpert 1000/1000/test` plus `NIH` retention evaluation | Warm-start adaptation improved over target-only training to `0.7757` `CheXpert` AUROC and `0.4770` AP, but `NIH` retention fell to `0.8112` AUROC and `0.2188` AP. | Better than training a fresh target-only head, but it introduced clear source forgetting. |
| `2026-04-13` | `Learning without Forgetting` sweep on warm-started adaptation | Same `CheXpert 1000/1000/test` setup plus `NIH` source batches | Best `CheXpert` AUROC was `0.7991`, best `CheXpert` AP was `0.4814`, and best retained `NIH` result was `0.8394` AUROC and `0.2447` AP. | Best pre-attention adaptation result. It clearly improved both target performance and source retention relative to plain warm-start adaptation, but it still did not beat the original `CXR Foundation` source baseline on `CheXpert`. |

## Bottom Line

| Question | Answer |
| --- | --- |
| Strongest overall result before the attention sections? | The strongest direct-transfer result was still the `CXR Foundation` image-only source baseline from `2026-04-11`, with `0.8455` on `NIH` and `0.8454` on `CheXpert`. |
| Strongest adaptation result before the attention sections? | The `2026-04-13` `Learning without Forgetting` update was best, reaching `0.7991` `CheXpert` AUROC while also reducing the forgetting seen in plain warm-start adaptation. |
| What stayed weak the whole time? | `MIMIC` transfer. Across direct transfer and retrieval-era baselines, results stayed near chance in this pre-attention summary slice. |
