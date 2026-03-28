# Scripts Guide

This directory contains setup utilities, data-prep helpers, embedding generators, frozen-classifier trainers, retrieval-memory evaluation scripts, and a one-shot generation script.

Most of the ML scripts are tightly coupled to the NIH CXR14 workflow in this workspace:

- They assume paths under `/workspace`.
- Many defaults target the 5-label setup: `atelectasis`, `cardiomegaly`, `consolidation`, `edema`, and `pleural_effusion`.
- Several scripts default to `/workspace/manifest_nih_cxr14 .csv` with a space before `.csv`. If your manifest is named differently, pass `--manifest-csv` or `--nih-manifest` explicitly.
- The ResNet50 retrieval-memory evaluation scripts are staged and depend on outputs from earlier stages.

## Quick chooser

Use this section first if you just need to know which script to run.

| If you need to... | Start with... |
| --- | --- |
| Download NIH, CheXpert-small, or PadChest-small raw data | `data_setup.py` |
| Install GitHub CLI auth for pushes | `setup_github_auth.sh` |
| Create runtime virtualenvs for FlashAttention or long-generation work | `setup_runtime_envs.sh` |
| Snapshot the whole workspace to a Git remote | `backup_to_git.sh` |
| Auto-commit and auto-push only generated reports or outputs | `report_autopush.sh` |
| Generate generic Hugging Face vision embeddings from NIH split CSVs | `generate_hf_nih_split_image_embeddings.py` |
| Reproduce the DINOv2 val/test embedding run used here | `generate_dinov2_nih_val_test_image_embeddings.py` |
| Generate ResNet50 image embeddings for train/val/test | `generate_resnet50_nih_split_image_embeddings.py` |
| Concatenate image embeddings with report embeddings | `generate_fused_nih_embeddings.py` |
| Fuse ResNet50 image embeddings with CLS report embeddings | `generate_fused_nih_embeddings_resnet50_cls.py` |
| Fuse ViT-base image embeddings with CLS report embeddings | `generate_fused_nih_embeddings_vit_cls.py` |
| Train a frozen linear classifier on DINOv2 or fused embeddings | `train_nih_frozen_dinov2_linear.py` |
| Train a frozen linear classifier on report-only embeddings | `train_nih_frozen_report_linear.py` |
| Train the shared frozen classifier on ResNet50 fused CLS embeddings | `train_nih_frozen_resnet50_fused_linear_cls.py` |
| Train the shared frozen classifier on ViT fused CLS embeddings | `train_nih_frozen_vit_fused_linear_cls.py` |
| Build the train-time FAISS memory for the ResNet50 fused CLS setup | `build_resnet50_fused_cls_retrieval_memory.py` |
| Evaluate retrieval memory only on validation | `evaluate_resnet50_fused_cls_val_memory_only.py` |
| Mix baseline probabilities with memory probabilities on validation | `evaluate_resnet50_fused_cls_val_probability_mixing.py` |
| Apply memory-driven logit correction on validation | `evaluate_resnet50_fused_cls_val_logit_correction.py` |
| Run the single frozen Stage 5A configuration on held-out test | `evaluate_resnet50_fused_cls_test_frozen_stage5a.py` |
| Generate one-shot MedGemma reports from `prompt.txt` | `run_medgemma_nih_one_shot.py` |

## Typical workflow order

For the main NIH fused-classifier experiments in this directory, the scripts usually chain together in this order:

1. `data_setup.py`
2. One of the image embedding generators
3. One of the fusion scripts
4. One of the training scripts
5. `build_resnet50_fused_cls_retrieval_memory.py`
6. `evaluate_resnet50_fused_cls_val_memory_only.py`
7. `evaluate_resnet50_fused_cls_val_probability_mixing.py`
8. `evaluate_resnet50_fused_cls_val_logit_correction.py`
9. `evaluate_resnet50_fused_cls_test_frozen_stage5a.py`

The evaluation scripts are not interchangeable. Stage 5A depends on Stage 4 outputs. Stage 5B depends on Stage 5A outputs. The held-out test script expects the frozen Stage 5A winner to already be fixed.

## Setup and repo automation

### `backup_to_git.sh`

What it does:
- Stages every file under `/workspace`, commits the result, and pushes it to `origin main`.
- Initializes a git repository if one does not already exist.
- Creates or updates the `origin` remote to the URL you pass as the first argument.
- Sets fallback git identity values if `user.name` and `user.email` are missing.
- Runs `git lfs install --local` if Git LFS is available.

Choose this when:
- You want a quick full-workspace backup.
- You do not need fine-grained control over what gets committed.

Inputs:
- Argument 1: remote repo URL. Default: `https://github.com/FarihaMehzabin/Domain-Adaptation`
- Argument 2: commit message. Default: `backup: <UTC timestamp>`

Outputs:
- A new commit on `main`, pushed to `origin`.

Important notes:
- It stages everything with `git add -A`.
- If there are no staged changes, it exits cleanly with `No changes to commit.`
- This is the broadest and least selective git script in the directory.

### `report_autopush.sh`

What it does:
- Commits and pushes only a chosen reports path, which defaults to `outputs`.
- Can run once or as a background loop.
- Tracks its background PID in `.git-report-autopush/pid` and logs to `.git-report-autopush/report-autopush.log`.
- Skips a cycle if `.git/index.lock` exists.

Choose this when:
- You want periodic auto-sync for generated outputs without backing up the whole workspace.
- You want something safer and narrower than `backup_to_git.sh`.

Commands:
- `run-once`: stage, commit, and push one time
- `start`: run `run-once` every 1800 seconds by default
- `stop`: stop the background loop
- `status`: show whether the loop is running

Environment overrides:
- `REPORT_AUTOPUSH_PATH`
- `REPORT_AUTOPUSH_INTERVAL_SECONDS`
- `REPORT_AUTOPUSH_REMOTE`
- `REPORT_AUTOPUSH_BRANCH`
- `REPORT_AUTOPUSH_DIR`
- `REPORT_AUTOPUSH_MESSAGE`

Outputs:
- Git commits affecting only the chosen path.
- Background runtime files under `.git-report-autopush/`.

Important notes:
- Unlike `backup_to_git.sh`, this script expects an existing git repository.
- It only stages and commits the configured reports path, not the full tree.

### `setup_github_auth.sh`

What it does:
- Installs GitHub CLI (`gh`) if needed.
- Authenticates to GitHub either with `GH_TOKEN` or `GITHUB_TOKEN`, or by launching the normal web login flow.
- Runs `gh auth setup-git` so git can use GitHub CLI credentials.

Choose this when:
- Pushes are failing because the machine is not authenticated with GitHub.
- You want one script to both install and configure `gh`.

Inputs:
- `--skip-install`
- `--hostname`
- `--git-protocol`

Outputs:
- An authenticated `gh` setup and git integration for the chosen host.

Important notes:
- On Linux it tries `apt-get`, `dnf`, `yum`, `pacman`, or `zypper`.
- On macOS it requires Homebrew.
- If a token is present, the auth flow is non-interactive.

### `setup_runtime_envs.sh`

What it does:
- Builds one or both helper virtualenvs:
- `.venv-fa2` for `flash-attn`
- `.venv-longgen` for long-generation work with CUDA 12.4 PyTorch 2.6 and Transformers

Choose this when:
- You need a clean environment for FlashAttention.
- You need the long-generation environment used by the MedGemma-style workloads.

Inputs:
- Positional target: `flash-attn`, `longgen`, or `all`

Outputs:
- `/workspace/.venv-fa2`
- `/workspace/.venv-longgen`

Important notes:
- The FlashAttention environment uses `--system-site-packages`.
- The long-generation environment installs:
- `torch==2.6.0`
- `torchvision==0.21.0`
- `torchaudio==2.6.0`
- `transformers>=5.3.0`
- `accelerate`
- `sentencepiece`
- `pillow`

## Data setup

### `data_setup.py`

What it does:
- Downloads raw datasets from Kaggle into `/workspace/data/.../raw`.
- Supports `nih_cxr14`, `chexpert_small`, and `padchest_small`.
- Can clean the `raw/` directory before download.
- Extracts downloaded archives and performs dataset-specific post-processing.
- For NIH, also writes `train.csv`, `val.csv`, and `test.csv` split files from the manifest.

Choose this when:
- You are preparing a fresh workspace.
- Raw data folders are missing or incomplete.
- NIH split CSVs need to be regenerated from the manifest.

Inputs:
- `--choice` quick selector for all, NIH, PadChest, or CheXpert
- `--datasets` explicit dataset names
- `--data-root`
- `--kaggle-config`
- `--clean-raw`
- `--force`
- `--keep-archives`
- `--dry-run`
- `--nih-manifest`
- `--split-output-dir`

Outputs:
- Downloaded raw files under:
- `/workspace/data/nih_cxr14/raw`
- `/workspace/data/chexpert_small/raw`
- `/workspace/data/padchest_small/raw`
- NIH split CSVs under `/workspace/data/nih_cxr14/splits` unless overridden

Important notes:
- Requires Kaggle CLI and a valid `kaggle.json`.
- It only manages `raw/` contents. It does not touch existing manifests, docs, or non-raw folders.
- `chexpert_small` is flattened out of the nested `CheXpert-v1.0-small` directory.
- `padchest_small` extracts nested archives inside `PC/`.
- The script comment block still mentions `setup_data.py`, but the actual file here is `data_setup.py`.

## Embedding generation

### `generate_hf_nih_split_image_embeddings.py`

What it does:
- Runs a generic Hugging Face vision encoder over NIH image split CSVs.
- Loads images listed in split CSVs, embeds them, applies configurable pooling, and L2 normalizes the result.
- Supports `train`, `val`, and `test`.

Choose this when:
- You want a model-agnostic image embedding script.
- You want to swap to a different Hugging Face vision model without rewriting the pipeline.
- You want optional train split support in the generic HF path.

Default behavior:
- Model: `facebook/dinov2-large`
- Default splits: `val test`
- Pooling: `auto`

Inputs:
- `--data-root`
- `--split-csv-dir`
- `--output-root`
- `--splits`
- `--model-id`
- `--pooling`
- `--batch-size`
- `--num-workers`
- `--max-images`
- `--extensions`
- `--device`
- `--fp16-on-cuda`

Outputs per split:
- `embeddings.npy`
- `image_manifest.csv`
- `image_paths.txt`
- `failed_images.jsonl` if any images fail
- `run_meta.json`

Important notes:
- For `train`, the output directory is the `output-root` itself.
- For `val` and `test`, the output directories are `output-root/val` and `output-root/test`.
- If you want strict reproduction of the existing DINOv2 val/test setup here, use `generate_dinov2_nih_val_test_image_embeddings.py` instead.

### `generate_dinov2_nih_val_test_image_embeddings.py`

What it does:
- Runs the DINOv2 large model on NIH validation and test images.
- Hard-codes the reproduction choices used in this workspace:
- model `facebook/dinov2-large`
- CLS pooling
- L2 normalization

Choose this when:
- You want the explicit DINOv2 val/test embedding pipeline used by the current experiments.
- You do not need train split generation from this script.

Inputs:
- `--data-root`
- `--split-csv-dir`
- `--output-root`
- `--splits` limited to `val` and `test`
- `--model-id`
- `--pooling`
- `--batch-size`
- `--num-workers`
- `--max-images`
- `--extensions`
- `--device`
- `--fp16-on-cuda`

Outputs per split:
- `embeddings.npy`
- `image_manifest.csv`
- `image_paths.txt`
- `failed_images.jsonl` if needed
- `run_meta.json`

Important notes:
- This is effectively a self-contained specialization of the generic HF embedding script.
- It only supports `val` and `test`.

### `generate_resnet50_nih_split_image_embeddings.py`

What it does:
- Generates NIH image embeddings with torchvision ResNet50.
- Uses the model's `avgpool` representation and L2 normalizes it.
- Reads rows from the manifest rather than split CSV files.
- Resolves image paths against several likely roots and fails loudly if the manifest paths are inconsistent.

Choose this when:
- You are building the ResNet50 branch of the fused-classifier pipeline.
- You need train, val, and test image embeddings under a split-aware directory structure.

Inputs:
- `--manifest-csv`
- `--data-root`
- `--output-root`
- `--splits`
- `--weights`
- `--batch-size`
- `--num-workers`
- `--max-images-per-split`
- `--extensions`
- `--device`
- `--fp16-on-cuda`
- `--bootstrap-legacy-train-root`
- `--overwrite`

Outputs per split:
- `embeddings.npy`
- `image_manifest.csv`
- `image_paths.txt`
- `failed_images.jsonl` if needed
- `run_meta.json`

Important notes:
- It defaults to `/workspace/image_embeddings/resnet50`.
- If legacy train artifacts already exist directly under the output root, the script can copy them into `output-root/train` so later scripts see a consistent split layout.
- It skips existing split outputs unless `--overwrite` is enabled.

### `generate_fused_nih_embeddings.py`

What it does:
- Concatenates image embeddings with report embeddings for each split.
- Aligns image rows and report rows by `Path(image_path).stem == report_id`.
- Writes one fused embedding matrix per split.

Choose this when:
- You already have image embeddings and report embeddings and want a fused representation.
- You want the shared fusion logic used by the ResNet50 and ViT CLS wrappers.

Inputs:
- `--image-root`
- `--report-train-dir`
- `--report-val-dir`
- `--report-test-dir`
- `--output-root`

Outputs per split:
- `embeddings.npy`
- `image_paths.txt`
- `run_meta.json`

Important notes:
- It fails if any image is missing a matching report embedding.
- It checks for Git LFS pointer files in report embeddings and stops if the real arrays were not fetched.
- It does not normalize or reweight the concatenated vector. It simply concatenates image and report features.

### `generate_fused_nih_embeddings_resnet50_cls.py`

What it does:
- Thin wrapper around `generate_fused_nih_embeddings.py`.
- Uses ResNet50 image embeddings and CLS-style report embeddings by default.

Choose this when:
- You are on the ResNet50 fused CLS branch and want the standard paths without retyping them.

Default inputs:
- Image root: `/workspace/image_embeddings/resnet50`
- Report roots:
- `/workspace/report_embeddings_cls/train/microsoft__BiomedVLP-CXR-BERT-specialized`
- `/workspace/report_embeddings_cls/val/microsoft__BiomedVLP-CXR-BERT-specialized`
- `/workspace/report_embeddings_cls/test/microsoft__BiomedVLP-CXR-BERT-specialized`

Outputs:
- Fused split directories under `/workspace/fused_embeddings_cls/resnet50`

Important notes:
- Unknown CLI flags are forwarded to the shared fusion script.

### `generate_fused_nih_embeddings_vit_cls.py`

What it does:
- Thin wrapper around `generate_fused_nih_embeddings.py`.
- Uses ViT-base image embeddings and CLS-style report embeddings by default.

Choose this when:
- You are on the ViT fused CLS branch and want the standard paths without retyping them.

Default inputs:
- Image root: `/workspace/image_embeddings/vit-base-patch16-224`
- Report roots:
- `/workspace/report_embeddings_cls/train/microsoft__BiomedVLP-CXR-BERT-specialized`
- `/workspace/report_embeddings_cls/val/microsoft__BiomedVLP-CXR-BERT-specialized`
- `/workspace/report_embeddings_cls/test/microsoft__BiomedVLP-CXR-BERT-specialized`

Outputs:
- Fused split directories under `/workspace/fused_embeddings_cls/vit-base-patch16-224`

Important notes:
- Unknown CLI flags are forwarded to the shared fusion script.

## Training

### `train_nih_frozen_dinov2_linear.py`

What it does:
- Trains a linear multilabel classifier on frozen embeddings.
- Uses deterministic seeding, `AdamW`, and `BCEWithLogitsLoss` with per-label `pos_weight`.
- Tunes the largest safe micro-batch automatically, then derives an effective batch size in the 512-1024 range with gradient accumulation if needed.
- Uses validation macro AUROC for early stopping and model selection.
- Tunes F1 thresholds on validation only, then evaluates test exactly once with the selected checkpoint.
- Saves reliability and calibration artifacts as both JSON and PNG.

Choose this when:
- You want the main shared trainer for frozen embedding experiments.
- Your inputs are image-or-fused embeddings aligned to train/val/test image-path files.
- You want to reuse the same training logic as the ResNet50 and ViT fused wrappers.

Inputs:
- `--train-embeddings`
- `--val-embeddings`
- `--test-embeddings`
- `--train-split-csv`
- `--val-split-csv`
- `--test-split-csv`
- `--manifest-csv`
- `--train-image-paths`
- `--val-image-paths`
- `--test-image-paths`
- `--output-root`
- `--run-name`
- `--device`
- `--learning-rate`
- `--weight-decay`
- `--max-epochs`
- `--patience`
- `--seed`

Outputs in a timestamped or named run directory:
- `config.json`
- `best.ckpt`
- `history.jsonl`
- `val_metrics.json`
- `test_metrics.json`
- `val_f1_thresholds.json`
- `calibration/val/*.json`
- `calibration/val/*.png`
- `calibration/test/*.json`
- `calibration/test/*.png`

Important notes:
- If the manifest path exists, the script prefers manifest-based alignment over the split CSV files.
- The run directory name is `--run-name` or a UTC timestamp like `20260324T091149Z`.
- This script is more general than the filename suggests. The ResNet50 and ViT fused wrapper scripts both reuse it.

### `train_nih_frozen_report_linear.py`

What it does:
- Trains the same style of frozen linear multilabel classifier, but on report embeddings instead of image or fused embeddings.
- Uses the manifest to break rows into train, val, and test.
- Aligns rows through `report_ids.json` rather than `image_paths.txt`.

Choose this when:
- You want a text-only baseline using precomputed report embeddings.

Inputs:
- `--manifest-csv`
- `--train-embeddings`
- `--val-embeddings`
- `--test-embeddings`
- `--train-report-ids`
- `--val-report-ids`
- `--test-report-ids`
- `--output-root`
- `--run-name`
- `--device`
- `--learning-rate`
- `--weight-decay`
- `--max-epochs`
- `--patience`
- `--seed`

Outputs in a timestamped or named run directory:
- `config.json`
- `best.ckpt`
- `history.jsonl`
- `val_metrics.json`
- `test_metrics.json`
- `val_f1_thresholds.json`
- `calibration/val/*.json`
- `calibration/val/*.png`
- `calibration/test/*.json`
- `calibration/test/*.png`

Important notes:
- Architecturally this is the report-only sibling of `train_nih_frozen_dinov2_linear.py`.

### `train_nih_frozen_resnet50_fused_linear_cls.py`

What it does:
- Thin wrapper around `train_nih_frozen_dinov2_linear.py`.
- Injects the default ResNet50 fused CLS train, val, and test paths.

Choose this when:
- You want the standard ResNet50 fused CLS training run without manually wiring every path.

Default inputs:
- `/workspace/fused_embeddings_cls/resnet50/train/embeddings.npy`
- `/workspace/fused_embeddings_cls/resnet50/val/embeddings.npy`
- `/workspace/fused_embeddings_cls/resnet50/test/embeddings.npy`
- Matching `image_paths.txt` files for each split

Default output root:
- `/workspace/outputs/nih_cxr14_frozen_fused_linear_cls_resnet50`

Important notes:
- Extra CLI flags are forwarded to the shared trainer, so you can still override learning rate, patience, run name, and so on.

### `train_nih_frozen_vit_fused_linear_cls.py`

What it does:
- Thin wrapper around `train_nih_frozen_dinov2_linear.py`.
- Injects the default ViT-base fused CLS train, val, and test paths.

Choose this when:
- You want the standard ViT fused CLS training run without manually wiring every path.

Default inputs:
- `/workspace/fused_embeddings_cls/vit-base-patch16-224/train/embeddings.npy`
- `/workspace/fused_embeddings_cls/vit-base-patch16-224/val/embeddings.npy`
- `/workspace/fused_embeddings_cls/vit-base-patch16-224/test/embeddings.npy`
- Matching `image_paths.txt` files for each split

Default output root:
- `/workspace/outputs/models/nih_cxr14/fused/linear_cls_vit_base`

Important notes:
- Extra CLI flags are forwarded to the shared trainer.

## Retrieval memory and staged evaluation

### `build_resnet50_fused_cls_retrieval_memory.py`

What it does:
- Builds the train-time retrieval memory for the ResNet50 fused CLS setup.
- Loads fused train embeddings, aligned labels, raw image embeddings, and report CLS embeddings.
- Verifies alignment and fusion consistency.
- L2 normalizes the fused train embeddings.
- Builds a FAISS `IndexFlatIP` index over the normalized vectors.
- Runs internal sanity checks:
- self-retrieval
- label-agreement sampling
- qualitative nearest-neighbor inspection

Choose this when:
- You are preparing the retrieval store used by Stage 4 and later evaluation scripts.

Inputs:
- `--fused-embeddings`
- `--fused-image-paths`
- `--fused-run-meta`
- `--manifest-csv`
- `--image-embeddings-dir`
- `--text-embeddings-dir`
- `--baseline-config`
- `--output-dir`
- `--self-retrieval-sample-size`
- `--label-agreement-queries`
- `--qualitative-queries`
- `--top-k`
- `--seed`

Outputs:
- `embeddings.npy`
- `labels.npy`
- `image_embeddings.npy`
- `text_embeddings.npy`
- `example_ids.json`
- `image_paths.txt`
- `items.jsonl`
- `index.faiss`
- `metadata.json`
- `sanity_report.json`
- `qualitative_neighbors.json`

Important notes:
- This script is specific to the ResNet50 fused CLS branch.
- Later stage scripts assume this memory directory exists and is internally consistent.

### `evaluate_resnet50_fused_cls_val_memory_only.py`

What it does:
- Stage 4 of the ResNet50 fused CLS evaluation chain.
- Uses the train retrieval memory to produce validation-time memory probabilities.
- Sweeps over `k` and `tau` to find a good retrieval configuration.
- Saves the default configuration outputs plus a sweep table and a chosen best config.
- Writes a success or failure `report.md`.

Choose this when:
- You want to know how far pure retrieval gets before mixing it with the baseline classifier.
- You need Stage 4 outputs for Stage 5A.

Default sweep:
- `k` in `[1, 3, 5, 10, 20, 50]`
- `tau` in `[1, 5, 10, 20, 40]`
- Default saved config: `k=5`, `tau=10`

Inputs:
- `--train-memory-root`
- `--val-embeddings`
- `--val-image-paths`
- `--val-run-meta`
- `--manifest-csv`
- `--output-dir`

Outputs:
- `run_config.json`
- `val_example_ids.json`
- `val_labels.npy`
- `val_p_mem_default.npy`
- `val_neighbor_indices_default.npy`
- `val_neighbor_scores_default.npy`
- `default_metrics.json`
- `sweep_results.csv`
- `best_memory_config.json`
- `qualitative_neighbors_val.json`
- `sanity_checks.json`
- `val_threshold_diagnostics.json`
- `report.md`

Possible extra debugging artifacts:
- `resolved_val_manifest_rows.jsonl`
- `dropped_val_rows.json`

Important notes:
- This is validation only. It should not touch the test split.
- Later scripts treat this directory as the canonical Stage 4 result.

### `evaluate_resnet50_fused_cls_val_probability_mixing.py`

What it does:
- Stage 5A of the ResNet50 fused CLS chain.
- Compares the frozen baseline classifier against retrieval memory on validation.
- Evaluates probability mixing:
- `p_mix = alpha * p_base + (1 - alpha) * p_mem`
- Searches over `alpha` on a coarse grid.
- Compares two memory setups:
- primary: `k=50, tau=1`
- reference: `k=5, tau=10`
- Can optionally run a local refinement search after the coarse grid.

Choose this when:
- You want to test whether combining the baseline classifier with retrieval beats either alone.
- You need the chosen Stage 5A config before running the held-out test script.

Inputs:
- `--baseline-run-root`
- `--baseline-run-root-fallback`
- `--stage4-output-dir`
- `--train-memory-root`
- `--val-embeddings`
- `--val-image-paths`
- `--val-run-meta`
- `--manifest-csv`
- `--output-dir`
- `--batch-size`
- `--run-optional-refinement`

Outputs:
- Aligned arrays:
- `aligned_val_example_ids.json`
- `aligned_val_labels.npy`
- `p_base_val.npy`
- `p_mem_k50_tau1.npy`
- `p_mem_k5_tau10.npy`
- `p_mix_best.npy`
- Search summaries:
- `mixing_results_coarse.csv`
- `mixing_results_refined.csv` if refinement runs
- Selection artifacts:
- `run_config.json`
- `best_mixing_config.json`
- `best_metrics.json`
- `qualitative_mixing_cases.json`
- `sanity_checks.json`
- `report.md`

Important notes:
- This script is still validation-only.
- The later held-out test script expects the chosen config from `best_mixing_config.json`.
- The selection rule prioritizes validation macro AUROC, then macro average precision, then larger alpha, then the primary memory config.

### `evaluate_resnet50_fused_cls_val_logit_correction.py`

What it does:
- Stage 5B of the ResNet50 fused CLS chain.
- Starts from baseline logits and memory probabilities instead of mixing probabilities directly.
- Evaluates two formulations:
- `primary_multilabel_logit`
- `pdf_literal_logprob`
- Sweeps a correction weight `beta`.
- Tracks calibration using ECE in addition to ranking and threshold metrics.

Choose this when:
- You want to test a stronger memory-informed correction than plain probability interpolation.
- You specifically want the validation-only logit-correction analysis.

Inputs:
- `--baseline-run-root`
- `--baseline-run-root-fallback`
- `--stage5a-output-dir`
- `--stage4-output-dir`
- `--train-memory-root`
- `--val-embeddings`
- `--val-image-paths`
- `--val-run-meta`
- `--manifest-csv`
- `--output-dir`
- `--batch-size`
- `--eps`
- `--ece-bins`
- `--run-optional-refinement`

Outputs:
- Aligned arrays:
- `aligned_val_example_ids.json`
- `aligned_val_labels.npy`
- `z_base_val.npy`
- `p_base_val.npy`
- `p_mem_k50_tau1.npy`
- `z_mem_primary.npy`
- `z_mem_pdf_literal.npy`
- `p_corr_best.npy`
- `z_corr_best.npy`
- Search summaries:
- `logit_correction_results_coarse.csv`
- `logit_correction_results_refined.csv` if refinement runs
- Selection artifacts:
- `run_config.json`
- `best_logit_correction_config.json`
- `best_metrics.json`
- `qualitative_logit_correction_cases.json`
- `sanity_checks.json`
- `report.md`

Important notes:
- This script is still validation-only and should not be used as the final held-out test result.
- Its selection rule uses AUROC, then average precision, then lower ECE, then smaller `beta`.

### `evaluate_resnet50_fused_cls_test_frozen_stage5a.py`

What it does:
- Runs the single held-out test evaluation for the frozen Stage 5A winner.
- Uses the fixed config:
- `alpha = 0.80`
- `k = 50`
- `tau = 1`
- Recomputes test retrieval probabilities from the train memory.
- Compares baseline-only against the frozen Stage 5A mixed system.
- Includes leakage and split-overlap checks and verifies archived baseline metrics.

Choose this when:
- Stage 5A is finalized and you are ready for the true held-out test run.
- You want the final test comparison for the frozen Stage 5A probability-mixing system.

Inputs:
- `--baseline-run-root`
- `--baseline-run-root-fallback`
- `--stage5a-output-dir`
- `--train-memory-root`
- `--test-embeddings`
- `--test-image-paths`
- `--test-run-meta`
- `--manifest-csv`
- `--train-split-csv`
- `--val-split-csv`
- `--test-split-csv`
- `--output-dir`
- `--batch-size`
- `--ece-bins`

Outputs:
- Aligned arrays:
- `aligned_test_example_ids.json`
- `aligned_test_labels.npy`
- `z_base_test.npy`
- `p_base_test.npy`
- `p_mem_test_k50_tau1.npy`
- `test_neighbor_indices_k50_tau1.npy`
- `test_neighbor_scores_k50_tau1.npy`
- `p_mix_test_alpha0p80.npy`
- Metrics and summaries:
- `test_metrics_baseline.json`
- `test_metrics_stage5a_frozen.json`
- `test_comparison_summary.json`
- `test_metrics_frozen_thresholds.json` if available
- `qualitative_test_cases.json`
- `sanity_checks.json`
- `run_config.json`
- `report.md`

Important notes:
- This is the only script in the staged chain that is meant to consume the held-out test split.
- It should be run after the validation-only selection work is complete.
- It explicitly checks that there is no train/test or val/test leakage.

## Generation

### `run_medgemma_nih_one_shot.py`

What it does:
- Runs `google/medgemma-4b-it` in one-shot mode on NIH images.
- Reads the user prompt from `/workspace/prompt.txt`.
- Loads image rows from a split CSV, builds multimodal chat inputs, and writes one report text file per image.
- Sanitizes some known awkward output patterns and appends `7) DOMAIN VECTOR` if missing.

Choose this when:
- You want quick sample generations from MedGemma against NIH images.
- You want a smoke test before building something larger.

Default behavior:
- Default split CSV: `/workspace/data/nih_cxr14/splits/train.csv`
- Default output dir: `/workspace/outputs/fresh_medgemma_4b_it_one_shot_smoke`
- If you do not pass image IDs and also do not set `--limit` or `--offset`, it uses three hard-coded smoke-test image IDs:
- `00000002_000`
- `00000005_001`
- `00000013_014`

Inputs:
- `--split-csv`
- `--output-dir`
- `--image-id` repeatable
- `--limit`
- `--offset`
- `--overwrite`
- `--batch-size`
- `--max-new-tokens`
- `--attn-implementation`

Outputs:
- One `<image_id>.txt` file per generated image.

Important notes:
- Requires `HF_TOKEN` and accepted access terms for `google/medgemma-4b-it`.
- Uses `device_map="auto"` and `torch.bfloat16`.
- If concurrent generation OOMs on CUDA, the script tells you to reduce `--batch-size` or change the attention implementation.

## Practical advice

If someone is new to this directory, the safest way to choose a script is:

1. Decide whether you are doing setup, embedding generation, training, staged evaluation, or report generation.
2. Prefer wrapper scripts when they match your exact branch, because they encode the expected default paths.
3. Prefer the generic scripts when you want to swap models, roots, or artifact locations.
4. Treat the `evaluate_resnet50_fused_cls_*` scripts as a fixed chain, not as independent utilities.

If you are unsure where to start for the main NIH fused workflow, start with `data_setup.py`, then pick the embedding generator that matches your backbone, and only move into the evaluation stages after the training and memory artifacts exist.
