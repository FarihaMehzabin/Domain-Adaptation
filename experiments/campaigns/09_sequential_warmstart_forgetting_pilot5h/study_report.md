# Sequential Warm-Start Forgetting Study

- Manifest: `/workspace/experiments/campaigns/09_sequential_warmstart_forgetting_pilot5h/manifest/manifest_pilot5h_binary_mimic.csv`
- Embedding view: `/tmp/cxr_sequential_forgetting_study/embedding_views/pilot5h_nih_chexpert_mimic_cxr_foundation`

## Stage AUROC

- NIH source: NIH `0.8461346978442619`, CheXpert `0.8485879868021924`, MIMIC `0.737037083174961`
- After CheXpert train: NIH `0.821012785227422`, CheXpert `0.788215306650718`, MIMIC `0.7638197470260162`
- After MIMIC train: NIH `0.8220945772214833`, CheXpert `0.8134939813280871`, MIMIC `0.7770320135451925`

## Forgetting

- NIH after CheXpert delta: `-0.025121912616839825`
- NIH after MIMIC delta vs NIH source: `-0.024040120622778605`
- NIH after MIMIC delta vs CheXpert stage: `0.0010817919940612208`
- CheXpert after MIMIC delta vs CheXpert stage: `0.025278674677369106`

## Transfer

- NIH to CheXpert zero-shot: `0.8485879868021924`
- NIH to MIMIC zero-shot: `0.737037083174961`
- CheXpert finetune gain vs NIH zero-shot: `-0.060372680151474345`
- MIMIC finetune gain vs NIH zero-shot: `0.039994930370231496`
