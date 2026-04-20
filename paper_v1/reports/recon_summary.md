# Recon Summary

- Manifest: `/workspace/manifest/manifest_common_labels_pilot5h.csv`
- Label order: `atelectasis, cardiomegaly, consolidation, edema, pleural_effusion, pneumonia, pneumothorax`
- Embedding dimension: `768`

## Split Counts
- `d0_nih/test`: `2000`
- `d0_nih/train`: `10000`
- `d0_nih/val`: `1000`
- `d1_chexpert/test`: `234`
- `d1_chexpert/train`: `1000`
- `d1_chexpert/val`: `1000`
- `d2_mimic/test`: `676`
- `d2_mimic/train`: `1000`
- `d2_mimic/val`: `1000`

## View Counts
- `d0_nih/test/FRONTAL`: `2000`
- `d0_nih/train/FRONTAL`: `10000`
- `d0_nih/val/FRONTAL`: `1000`
- `d1_chexpert/test/FRONTAL`: `202`
- `d1_chexpert/test/LATERAL`: `32`
- `d1_chexpert/train/FRONTAL`: `857`
- `d1_chexpert/train/LATERAL`: `143`
- `d1_chexpert/val/FRONTAL`: `850`
- `d1_chexpert/val/LATERAL`: `150`
- `d2_mimic/test/FRONTAL`: `596`
- `d2_mimic/test/LATERAL`: `10`
- `d2_mimic/test/UNKNOWN`: `70`
- `d2_mimic/train/FRONTAL`: `963`
- `d2_mimic/train/LATERAL`: `14`
- `d2_mimic/train/UNKNOWN`: `23`
- `d2_mimic/val/FRONTAL`: `958`
- `d2_mimic/val/LATERAL`: `14`
- `d2_mimic/val/UNKNOWN`: `28`

## Manifest Issues
- none

## Alignment Issues
- `missing_embedding_file`: embedding file does not exist: /workspace/experiments/by_id/exp0085__cxr_foundation_embedding_export__mimic_target_1000_cxr_foundation_avg_batch128/d2_mimic/train/embeddings.npy
- `missing_embedding`: missing embedding for row_id mimic_cxr__p10855616__s55442288__acc6c734-64303ed5-4d69ca7d-1d75440d-6da5c2dc
- `missing_embedding`: missing embedding for row_id mimic_cxr__p10862862__s57816531__e07b2ae9-8f5ac6ac-a6fe93af-b5fe92ca-53b20a75

## Notes
- Dropped rows under current policy: `mimic_cxr__p10003400__s51777274__67eafcfb-841cb6fa-38d704c7-29fc44fc-bb89535e, mimic_cxr__p10010867__s57662143__e4a3fe20-cf6c133d-7cb2e0e0-04ecbee2-d8911acc, mimic_cxr__p10029291__s51934618__6d00654c-94387330-275cff2b-f94acdd9-d980c90d, mimic_cxr__p10038999__s50971552__1fa9e818-e87b1da0-a236c55f-2b940f25-eb8769bd, mimic_cxr__p10065383__s53295124__331145d6-d982c8c1-772c4dac-f0da5016-52726c05, mimic_cxr__p10074434__s57980377__892a4da8-e50b49d1-b331571f-0ed1ddb4-de4f6cb1, mimic_cxr__p10076526__s59199952__4a2d9d8c-205ac898-29948077-fe0683cb-0a6f51f9, mimic_cxr__p10083833__s51949859__83bff72b-dd475c82-4a3f55be-73303f01-b21c450a, mimic_cxr__p10095258__s54774968__ad749fb9-c3a837ab-f91ad06b-f1a54f51-9b6d5998, mimic_cxr__p10109899__s51270735__128dc4c3-beb071cf-8fdef3d3-5d71fbbe-8ba6d7bc`
