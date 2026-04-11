# Current Scripts

Use these numbered entrypoints for the current source-stage workflow:

1. `01_generate_nih_split_image_embeddings.py`
2. `02_generate_nih_split_report_embeddings.py`
3. `03_generate_split_fused_embeddings.py`
4. `04_train_frozen_multilabel_baseline.py`
5. `05_build_source_retrieval_memory.py`
6. `06_evaluate_source_memory_only.py`
7. `07_evaluate_probability_mixing.py`
8. `08_evaluate_source_memory_test.py`
9. `09_evaluate_probability_mixing_test.py`
10. `10_train_cross_attention_encoder.py`
11. `11_export_cross_attention_embeddings.py`
12. `12_build_common_label_manifest.py`
13. `13_build_domain_transfer_manifest.py`
14. `14_generate_cxr_foundation_embeddings.py`
15. `15_train_domain_transfer_linear_probe.py`
16. `16_build_subset_manifest.py`
17. `17_generate_domain_transfer_image_embeddings.py`
18. `18_benchmark_cxr_foundation_batch_sizes.py`
19. `19_benchmark_torch_image_batch_sizes.py`
20. `20_run_domain_transfer_mlp_sweep.py`
21. `21_train_image_only_lora_transfer.py`

Notes:

- Steps `01` to `03` are the pooled frozen-embedding baseline path.
- Step `04` is the current baseline trainer for frozen embedding experiments.
- Step `05` builds the train-time retrieval memory from the selected frozen embedding source.
- Step `06` evaluates the frozen train memory on held-out validation queries before any baseline fusion.
- Step `07` evaluates probability mixing between the frozen baseline and the selected memory-only validation probabilities.
- Step `08` evaluates the frozen train memory on the held-out test split using the validation-selected `k` and `tau`.
- Step `09` evaluates frozen probability mixing on the held-out test split using the validation-selected `alpha`.
- Step `10` trains the new multimodal image-report cross-attention branch with frozen ResNet50 and frozen CXR-BERT backbones.
  By default this now uses a gated hybrid embedding that mixes:
  - pooled legacy image features,
  - projected legacy report features,
  - the learned cross-attention embedding.
  Use `--disable-gated-hybrid` only if you explicitly want the older pure cross-attention export path.
- Step `11` exports the trained cross-attention model into a standard split-aware `embeddings.npy` root so steps `04` to `09` can run unchanged on the new branch.
- Step `12` builds the original common-label manifest used for the NIH-train to CheXpert/MIMIC transfer view.
- Step `13` builds the new D0/D1/D2 manifest with NIH `train/val/test`, CheXpert `val`, and MIMIC `val/test`.
- Step `14` exports image-only CXR Foundation embeddings into domain/split-sharded `embeddings.npy` roots plus a root-level `embedding_index.csv`.
- Step `15` trains the image-only multilabel linear or MLP head on `D0 train`, early-stops on `D0 val` macro AUROC, and evaluates `D0 test` plus direct transfer to `D1` and `D2`.
- Step `16` builds smaller subset manifests such as the pilot NIH-source transfer slice used to compare backbones quickly before full runs.
- Step `17` exports domain-aware torch image embeddings for NIH/CheXpert/MIMIC using the same basic CNN path as the original NIH-only ResNet workflow.
- Step `18` benchmarks CXR Foundation batch sizes on a chosen subset before committing to a long export run.
- Step `19` benchmarks torch image-encoder batch sizes on a chosen subset before the real ResNet export.
- Step `20` sweeps small source-trained MLP heads on the existing pilot embedding roots and ranks them by `D0 val` macro AUROC.
- Step `21` trains an image-only Hugging Face vision transformer with LoRA directly on NIH images and evaluates source-only transfer to CheXpert and MIMIC without pre-exporting embeddings.
- `scripts_old/` remains archived and is not part of the active workflow.
- After each experiment run, the expectation is:
  - the experiment directory contains a `recreation_report.md`,
  - the run is committed,
  - the commit is pushed before moving to the next experiment.
- The CXR Foundation path has separate Python requirements:
  - install with `python -m pip install -r /workspace/scripts/requirements_cxr_foundation.txt`
  - accept the model terms for `google/cxr-foundation` on Hugging Face and provide `HF_TOKEN` when required
- The image-only LoRA ViT path has separate Python requirements:
  - install with `python -m pip install -r /workspace/scripts/requirements_vision_lora.txt`

Compatibility:

- The original unnumbered generation scripts are still present so older references do not break.
- New runs should prefer the numbered entrypoints above.
- The cross-attention branch is additive. Retrieval and probability mixing remain downstream of the exported 2D embedding root, not inside the attention model itself.
- The CXR Foundation branch is additive. It does not replace the existing NIH-only ResNet workflow.
