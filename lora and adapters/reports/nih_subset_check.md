# NIH Subset Check

## Executive Summary

```json
{
  "safe_to_proceed": true,
  "stage": "stage2_nih_subset_validation",
  "status": "DONE"
}
```

## Paths Used

```json
{
  "metadata_csv": "/workspace/data/nih_chest_xray14/Data_Entry_2017.csv",
  "nih_root": "/workspace/data/nih_chest_xray14"
}
```

## Metadata Validation

```json
{
  "age_summary": {
    "max": 414.0,
    "mean": 46.90146271851587,
    "median": 49.0,
    "min": 1.0,
    "missing_or_invalid": 0,
    "suspicious_gt_120": 16
  },
  "gender_counts": {
    "F": 48780,
    "M": 63340
  },
  "optional_columns_present": [
    "Follow-up #",
    "OriginalImage[Width",
    "Height]",
    "OriginalImagePixelSpacing[x",
    "y]"
  ],
  "required_columns": [
    "Image Index",
    "Finding Labels",
    "Patient ID",
    "Patient Age",
    "Patient Gender",
    "View Position"
  ],
  "view_position_counts": {
    "AP": 44810,
    "PA": 67310
  }
}
```

## Image Matching Validation

```json
{
  "duplicate_file_basenames": 0,
  "duplicate_image_ids": 0,
  "matched_images": 112120,
  "metadata_rows": 112120,
  "missing_images": 0,
  "unique_images": 112120,
  "unique_patients": 30805
}
```

## Duplicate Checks

```json
{
  "duplicate_file_basenames": {},
  "duplicate_image_ids": 0
}
```

## Label Parsing Validation

```json
{
  "label_counts": {
    "Atelectasis": 11559,
    "Cardiomegaly": 2776,
    "Consolidation": 4667,
    "Edema": 2303,
    "Effusion": 13317,
    "Emphysema": 2516,
    "Fibrosis": 1686,
    "Hernia": 227,
    "Infiltration": 19894,
    "Mass": 5782,
    "Nodule": 6331,
    "Pleural_Thickening": 3385,
    "Pneumonia": 1431,
    "Pneumothorax": 5302
  },
  "label_issue_rows": [],
  "no_finding_count": 60361
}
```

## Patient-Level Split Validation

```json
{
  "leakage": {
    "image_overlap": {
      "train_test": [],
      "train_val": [],
      "val_test": []
    },
    "patient_overlap_train_test": [],
    "patient_overlap_train_val": [],
    "patient_overlap_val_test": []
  },
  "split_counts": {
    "test": 22353,
    "train": 78707,
    "val": 11060
  },
  "split_label_counts": {
    "test": {
      "Atelectasis": 2068,
      "Cardiomegaly": 611,
      "Consolidation": 878,
      "Edema": 490,
      "Effusion": 2475,
      "Emphysema": 486,
      "Fibrosis": 383,
      "Hernia": 66,
      "Infiltration": 3995,
      "Mass": 1180,
      "Nodule": 1224,
      "Pleural_Thickening": 712,
      "Pneumonia": 312,
      "Pneumothorax": 928
    },
    "train": {
      "Atelectasis": 8352,
      "Cardiomegaly": 1846,
      "Consolidation": 3320,
      "Edema": 1603,
      "Effusion": 9438,
      "Emphysema": 1711,
      "Fibrosis": 1172,
      "Hernia": 150,
      "Infiltration": 13966,
      "Mass": 3994,
      "Nodule": 4458,
      "Pleural_Thickening": 2333,
      "Pneumonia": 997,
      "Pneumothorax": 3892
    },
    "val": {
      "Atelectasis": 1139,
      "Cardiomegaly": 319,
      "Consolidation": 469,
      "Edema": 210,
      "Effusion": 1404,
      "Emphysema": 319,
      "Fibrosis": 131,
      "Hernia": 11,
      "Infiltration": 1933,
      "Mass": 608,
      "Nodule": 649,
      "Pleural_Thickening": 340,
      "Pneumonia": 122,
      "Pneumothorax": 482
    }
  }
}
```

## Dev 2k Subset Data Card

```json
{
  "image_counts": {
    "test": 400,
    "train": 1400,
    "val": 200
  },
  "label_counts": {
    "test": {
      "Atelectasis": 31,
      "Cardiomegaly": 7,
      "Consolidation": 30,
      "Edema": 14,
      "Effusion": 86,
      "Emphysema": 5,
      "Fibrosis": 4,
      "Hernia": 1,
      "Infiltration": 111,
      "Mass": 15,
      "Nodule": 25,
      "Pleural_Thickening": 28,
      "Pneumonia": 11,
      "Pneumothorax": 27
    },
    "train": {
      "Atelectasis": 129,
      "Cardiomegaly": 25,
      "Consolidation": 75,
      "Edema": 30,
      "Effusion": 174,
      "Emphysema": 18,
      "Fibrosis": 15,
      "Hernia": 2,
      "Infiltration": 219,
      "Mass": 110,
      "Nodule": 71,
      "Pleural_Thickening": 50,
      "Pneumonia": 21,
      "Pneumothorax": 70
    },
    "val": {
      "Atelectasis": 29,
      "Cardiomegaly": 13,
      "Consolidation": 4,
      "Edema": 7,
      "Effusion": 29,
      "Emphysema": 8,
      "Fibrosis": 3,
      "Hernia": 0,
      "Infiltration": 24,
      "Mass": 7,
      "Nodule": 4,
      "Pleural_Thickening": 1,
      "Pneumonia": 0,
      "Pneumothorax": 8
    }
  },
  "patient_counts": {
    "test": 82,
    "train": 395,
    "val": 66
  },
  "total_label_counts": {
    "Atelectasis": 189,
    "Cardiomegaly": 45,
    "Consolidation": 109,
    "Edema": 51,
    "Effusion": 289,
    "Emphysema": 31,
    "Fibrosis": 22,
    "Hernia": 3,
    "Infiltration": 354,
    "Mass": 132,
    "Nodule": 100,
    "Pleural_Thickening": 79,
    "Pneumonia": 32,
    "Pneumothorax": 105
  },
  "warnings": []
}
```

## Dev 10k Subset Data Card

```json
{
  "image_counts": {
    "test": 2000,
    "train": 7000,
    "val": 1000
  },
  "label_counts": {
    "test": {
      "Atelectasis": 196,
      "Cardiomegaly": 47,
      "Consolidation": 99,
      "Edema": 35,
      "Effusion": 260,
      "Emphysema": 35,
      "Fibrosis": 28,
      "Hernia": 3,
      "Infiltration": 341,
      "Mass": 105,
      "Nodule": 96,
      "Pleural_Thickening": 48,
      "Pneumonia": 12,
      "Pneumothorax": 96
    },
    "train": {
      "Atelectasis": 769,
      "Cardiomegaly": 156,
      "Consolidation": 278,
      "Edema": 140,
      "Effusion": 919,
      "Emphysema": 120,
      "Fibrosis": 87,
      "Hernia": 9,
      "Infiltration": 1208,
      "Mass": 309,
      "Nodule": 324,
      "Pleural_Thickening": 192,
      "Pneumonia": 112,
      "Pneumothorax": 297
    },
    "val": {
      "Atelectasis": 118,
      "Cardiomegaly": 18,
      "Consolidation": 32,
      "Edema": 13,
      "Effusion": 134,
      "Emphysema": 16,
      "Fibrosis": 7,
      "Hernia": 3,
      "Infiltration": 139,
      "Mass": 47,
      "Nodule": 42,
      "Pleural_Thickening": 41,
      "Pneumonia": 1,
      "Pneumothorax": 28
    }
  },
  "patient_counts": {
    "test": 567,
    "train": 1851,
    "val": 312
  },
  "total_label_counts": {
    "Atelectasis": 1083,
    "Cardiomegaly": 221,
    "Consolidation": 409,
    "Edema": 188,
    "Effusion": 1313,
    "Emphysema": 171,
    "Fibrosis": 122,
    "Hernia": 15,
    "Infiltration": 1688,
    "Mass": 461,
    "Nodule": 462,
    "Pleural_Thickening": 281,
    "Pneumonia": 125,
    "Pneumothorax": 421
  },
  "warnings": []
}
```

## Batch Loading Sanity Check

```json
{
  "batch_size": 16,
  "finite_values": true,
  "image_dtype": "torch.float32",
  "image_max": 0.9686274528503418,
  "image_mean": 0.5624061226844788,
  "image_min": 0.0,
  "image_shape": [
    16,
    1,
    96,
    96
  ],
  "label_binary": true,
  "label_shape": [
    16,
    14
  ],
  "montage_path": "reports/nih_dev_2k_batch.png",
  "stats_path": "reports/nih_dev_2k_batch_stats.json"
}
```

## Warnings

```json
[]
```

## Failure Conditions

```json
[]
```

## Safe-To-Proceed Recommendation

```json
{
  "safe_to_proceed": true
}
```
