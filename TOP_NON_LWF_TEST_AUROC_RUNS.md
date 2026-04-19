# Top Non-LwF Test-AUROC Improvements

Filtered to:
- exclude everything tied to the first 1-epoch mixed xattn system
- exclude validation-only improvements
- report held-out test AUROCs as `before -> after`

| Run | AUROC before -> after | Embeddings / input | Dataset split | Params | FLOPs |
| --- | --- | --- | --- | ---: | --- |
| `exp0015` CXR Foundation linear probe | `0.7306 -> 0.8455` on NIH test | Frozen `CXR Foundation`, `general`, `avg`, `768`-d | `manifest_common_labels_pilot5h.csv`: NIH `10000/1000/2000`, CheXpert `234`, MIMIC `1455`, 7 labels | `5,383` | Not logged; head-only forward is about `10,752` ops/sample |
| `exp0027` ViT LoRA pilot | `0.7306 -> 0.7819` on NIH test | Raw images into `google/vit-base-patch16-224-in21k`, `CLS` pooled, hidden size `768` | Same `pilot5h` split as `exp0015`, 7 labels | `86,691,079` total, `301,831` trainable | Not logged |
| Original fused baseline family | `0.746067 -> 0.767933` on test macro AUROC | Frozen fused `2176`-d embeddings = ResNet50 image + CXR-BERT report | `NIH CXR14 all14`: `78,571/11,219/22,330` | `30,478` for the `2176 -> 14` linear head | Not logged; head-only forward is about `60,928` ops/sample |
| `exp0013` 100-epoch fused-baseline rerun in the synthesis | `0.767933 -> 0.774642` on test macro AUROC | Frozen fused `2176`-d embeddings = ResNet50 image + CXR-BERT report | `NIH CXR14 all14`: `78,571/11,219/22,330` | `30,478` for the `2176 -> 14` linear head | Not logged; head-only forward is about `60,928` ops/sample |
| `exp0033` hybrid mixed branch in the synthesis | `0.759259 -> 0.772098` on test macro AUROC | `2688`-d L2-normalized hybrid concat = long-xattn export + rebuilt ResNet50 image embeddings + report embeddings | Fixed NIH CXR14 source-stage split used in synthesis Sections 11-15: train `78,571`, val `11,219`, test `22,330`, 14 labels; retrieval built from train, selected on val, reported on test | Mixed stage has `0` learned params beyond upstream baseline; matching `2688 -> 14` baseline head would be `37,646` params | Not logged |

Notes:
- The older source-stage entries in the synthesis use IDs that do not map cleanly to today's `experiments/by_id` folders, so those rows are based on the synthesis plus surviving metadata.
- FLOPs are mostly not recorded anywhere in the repo; only the trivial linear-head counts were derived directly.
