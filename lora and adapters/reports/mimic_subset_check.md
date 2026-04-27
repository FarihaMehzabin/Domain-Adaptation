# MIMIC Subset Check

## Executive Summary

```json
{
  "safe_to_proceed": true,
  "stage": "stage3_mimic_subset_verification",
  "status": "DONE"
}
```

## Paths And Files Used

```json
{
  "chexpert_csv": "/workspace/data/mimic_cxr/metadata/mimic-cxr-2.0.0-chexpert.csv.gz",
  "image_filenames": "data/mimic_cxr/metadata/IMAGE_FILENAMES",
  "metadata_csv": "/workspace/data/mimic_cxr/metadata/mimic-cxr-2.0.0-metadata.csv.gz",
  "mimic_root": "/workspace/data/mimic_cxr",
  "split_csv": "/workspace/data/mimic_cxr/metadata/mimic-cxr-2.0.0-split.csv.gz"
}
```

## MIMIC Format Detected

jpg

## Metadata Validation

```json
{
  "chexpert_columns": [
    "subject_id",
    "study_id",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices"
  ],
  "chexpert_rows": 227827,
  "duplicate_dicoms_in_local_images": {},
  "image_filenames_entries": 377110,
  "local_image_count": 2676,
  "matched_images": 2676,
  "metadata_columns": [
    "dicom_id",
    "subject_id",
    "study_id",
    "PerformedProcedureStepDescription",
    "ViewPosition",
    "Rows",
    "Columns",
    "StudyDate",
    "StudyTime",
    "ProcedureCodeSequence_CodeMeaning",
    "ViewCodeSequence_CodeMeaning",
    "PatientOrientationCodeSequence_CodeMeaning"
  ],
  "metadata_rows": 377110,
  "method": "recursive_jpg_index",
  "missing_paths": 0,
  "preferred_candidates_checked": 754220,
  "split_columns": [
    "dicom_id",
    "study_id",
    "subject_id",
    "split"
  ],
  "split_rows": 377110
}
```

## Split File Validation

```json
{
  "split_counts": {
    "test": 676,
    "train": 1000,
    "validate": 1000
  },
  "split_method": "official_split"
}
```

## Image Path Resolution

```json
{
  "chexpert_columns": [
    "subject_id",
    "study_id",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices"
  ],
  "chexpert_rows": 227827,
  "duplicate_dicoms_in_local_images": {},
  "image_filenames_entries": 377110,
  "local_image_count": 2676,
  "matched_images": 2676,
  "metadata_columns": [
    "dicom_id",
    "subject_id",
    "study_id",
    "PerformedProcedureStepDescription",
    "ViewPosition",
    "Rows",
    "Columns",
    "StudyDate",
    "StudyTime",
    "ProcedureCodeSequence_CodeMeaning",
    "ViewCodeSequence_CodeMeaning",
    "PatientOrientationCodeSequence_CodeMeaning"
  ],
  "metadata_rows": 377110,
  "method": "recursive_jpg_index",
  "missing_paths": 0,
  "preferred_candidates_checked": 754220,
  "split_columns": [
    "dicom_id",
    "study_id",
    "subject_id",
    "split"
  ],
  "split_rows": 377110
}
```

## Image/Study/Subject Counts

```json
{
  "duplicate_dicoms": 0,
  "metadata_rows": 377110,
  "missing_images": 0,
  "unique_dicoms": 2676,
  "unique_studies": 2676,
  "unique_subjects": 1639,
  "verified_images": 2676
}
```

## View Position Distribution

```json
{
  "AP": 1624,
  "LATERAL": 2,
  "LL": 36,
  "MISSING": 121,
  "PA": 893
}
```

## Label Value Distributions

```json
{
  "Atelectasis": {
    "missing": 1926,
    "negative": 20,
    "positive": 591,
    "uncertain": 139
  },
  "Cardiomegaly": {
    "missing": 1822,
    "negative": 193,
    "positive": 589,
    "uncertain": 72
  },
  "Consolidation": {
    "missing": 2369,
    "negative": 74,
    "positive": 178,
    "uncertain": 55
  },
  "Edema": {
    "missing": 1807,
    "negative": 302,
    "positive": 399,
    "uncertain": 168
  },
  "Enlarged Cardiomediastinum": {
    "missing": 2387,
    "negative": 70,
    "positive": 111,
    "uncertain": 108
  },
  "Fracture": {
    "missing": 2576,
    "negative": 7,
    "positive": 89,
    "uncertain": 4
  },
  "Lung Lesion": {
    "missing": 2536,
    "negative": 8,
    "positive": 118,
    "uncertain": 14
  },
  "Lung Opacity": {
    "missing": 1930,
    "negative": 43,
    "positive": 656,
    "uncertain": 47
  },
  "No Finding": {
    "missing": 1953,
    "negative": 0,
    "positive": 723,
    "uncertain": 0
  },
  "Pleural Effusion": {
    "missing": 1535,
    "negative": 305,
    "positive": 748,
    "uncertain": 88
  },
  "Pleural Other": {
    "missing": 2608,
    "negative": 1,
    "positive": 50,
    "uncertain": 17
  },
  "Pneumonia": {
    "missing": 1907,
    "negative": 278,
    "positive": 229,
    "uncertain": 262
  },
  "Pneumothorax": {
    "missing": 1962,
    "negative": 561,
    "positive": 137,
    "uncertain": 16
  },
  "Support Devices": {
    "missing": 1758,
    "negative": 40,
    "positive": 874,
    "uncertain": 4
  }
}
```

## Uncertainty Policy Preview

```json
{
  "Atelectasis": {
    "u_ignore": 0.967266775777414,
    "u_ignore_mask_coverage": 0.22832585949177878,
    "u_one": 0.27279521674140506,
    "u_zero": 0.22085201793721973
  },
  "Cardiomegaly": {
    "u_ignore": 0.7531969309462916,
    "u_ignore_mask_coverage": 0.2922272047832586,
    "u_one": 0.2470104633781764,
    "u_zero": 0.22010463378176381
  },
  "Consolidation": {
    "u_ignore": 0.7063492063492064,
    "u_ignore_mask_coverage": 0.09417040358744394,
    "u_one": 0.08707025411061285,
    "u_zero": 0.06651718983557549
  },
  "Edema": {
    "u_ignore": 0.5691868758915835,
    "u_ignore_mask_coverage": 0.2619581464872945,
    "u_one": 0.21188340807174888,
    "u_zero": 0.1491031390134529
  },
  "Pleural Effusion": {
    "u_ignore": 0.7103513770180437,
    "u_ignore_mask_coverage": 0.39349775784753366,
    "u_one": 0.312406576980568,
    "u_zero": 0.2795216741405082
  },
  "Pneumonia": {
    "u_ignore": 0.4516765285996055,
    "u_ignore_mask_coverage": 0.18946188340807174,
    "u_one": 0.18348281016442453,
    "u_zero": 0.08557548579970105
  },
  "Pneumothorax": {
    "u_ignore": 0.19627507163323782,
    "u_ignore_mask_coverage": 0.2608370702541106,
    "u_one": 0.05717488789237668,
    "u_zero": 0.051195814648729444
  }
}
```

## Verified Manifest Summary

```json
{
  "k1": "manifests/mimic_target_support_k1_seed2027.csv",
  "k10": "manifests/mimic_target_support_k10_seed2027.csv",
  "k20": "manifests/mimic_target_support_k20_seed2027.csv",
  "k5": "manifests/mimic_target_support_k5_seed2027.csv",
  "k50": "manifests/mimic_target_support_k50_seed2027.csv",
  "label_value_counts": "manifests/mimic_label_value_counts.json",
  "target_query": "manifests/mimic_target_query.csv",
  "target_test": "manifests/mimic_target_test.csv",
  "target_train_pool": "manifests/mimic_target_train_pool.csv",
  "verified_all": "manifests/mimic_verified_all.csv",
  "verified_frontal": "manifests/mimic_verified_frontal.csv"
}
```

## Target Train/Query/Test Split Summary

```json
{
  "target_query_counts": {
    "dicoms": 958,
    "images": 958,
    "studies": 958,
    "subjects": 358
  },
  "target_test_counts": {
    "dicoms": 596,
    "images": 596,
    "studies": 596,
    "subjects": 268
  },
  "target_train_pool_counts": {
    "dicoms": 963,
    "images": 963,
    "studies": 963,
    "subjects": 942
  }
}
```

## K-Shot Support Set Summary

```json
{
  "achieved_positives_per_label": {
    "k1": {
      "Atelectasis": 2,
      "Cardiomegaly": 3,
      "Consolidation": 1,
      "Edema": 1,
      "Pleural Effusion": 2,
      "Pneumonia": 3,
      "Pneumothorax": 1
    },
    "k10": {
      "Atelectasis": 23,
      "Cardiomegaly": 26,
      "Consolidation": 17,
      "Edema": 20,
      "Pleural Effusion": 37,
      "Pneumonia": 14,
      "Pneumothorax": 12
    },
    "k20": {
      "Atelectasis": 52,
      "Cardiomegaly": 51,
      "Consolidation": 26,
      "Edema": 33,
      "Pleural Effusion": 67,
      "Pneumonia": 31,
      "Pneumothorax": 21
    },
    "k5": {
      "Atelectasis": 13,
      "Cardiomegaly": 14,
      "Consolidation": 5,
      "Edema": 13,
      "Pleural Effusion": 20,
      "Pneumonia": 7,
      "Pneumothorax": 6
    },
    "k50": {
      "Atelectasis": 99,
      "Cardiomegaly": 97,
      "Consolidation": 50,
      "Edema": 72,
      "Pleural Effusion": 131,
      "Pneumonia": 57,
      "Pneumothorax": 39
    }
  },
  "infeasible_labels_per_K": {
    "k1": [],
    "k10": [],
    "k20": [],
    "k5": [],
    "k50": [
      "Pneumothorax"
    ]
  },
  "per_K_counts": {
    "k1": {
      "dicoms": 7,
      "images": 7,
      "studies": 7,
      "subjects": 7
    },
    "k10": {
      "dicoms": 66,
      "images": 66,
      "studies": 66,
      "subjects": 65
    },
    "k20": {
      "dicoms": 129,
      "images": 129,
      "studies": 129,
      "subjects": 127
    },
    "k5": {
      "dicoms": 35,
      "images": 35,
      "studies": 35,
      "subjects": 35
    },
    "k50": {
      "dicoms": 265,
      "images": 265,
      "studies": 265,
      "subjects": 259
    }
  }
}
```

## Leakage Checks

```json
{
  "image_overlap": {
    "query_test": [],
    "train_pool_query": [],
    "train_pool_test": []
  },
  "study_overlap": {
    "query_test": [],
    "train_pool_query": [],
    "train_pool_test": []
  },
  "subject_overlap": {
    "query_test": [],
    "train_pool_query": [],
    "train_pool_test": []
  },
  "support_query_overlap": {
    "k1": {
      "images": [],
      "studies": [],
      "subjects": []
    },
    "k10": {
      "images": [],
      "studies": [],
      "subjects": []
    },
    "k20": {
      "images": [],
      "studies": [],
      "subjects": []
    },
    "k5": {
      "images": [],
      "studies": [],
      "subjects": []
    },
    "k50": {
      "images": [],
      "studies": [],
      "subjects": []
    }
  },
  "support_test_overlap": {
    "k1": {
      "images": [],
      "studies": [],
      "subjects": []
    },
    "k10": {
      "images": [],
      "studies": [],
      "subjects": []
    },
    "k20": {
      "images": [],
      "studies": [],
      "subjects": []
    },
    "k5": {
      "images": [],
      "studies": [],
      "subjects": []
    },
    "k50": {
      "images": [],
      "studies": [],
      "subjects": []
    }
  }
}
```

## Warnings

```json
[
  "668 images have all common CheXpert labels missing.",
  "K=50 infeasible for labels: Pneumothorax"
]
```

## Repair Plan If Needed

```json
[]
```

## Safe-To-Proceed Recommendation

```json
{
  "safe_to_proceed": true
}
```
