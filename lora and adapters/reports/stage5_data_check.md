# Stage 5 Data Integrity Check

## Executive Summary

```json
{
  "safe_to_proceed": true,
  "status": "DONE"
}
```

## Selected Task and Label Order

```json
{
  "canonical_label_order": [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax"
  ],
  "name": "common7"
}
```

## Dataset and Split Integrity

```json
{
  "label_binary_check": true,
  "no_image_leakage": true,
  "no_patient_leakage": true,
  "no_target_data_used": true,
  "path_resolution": {
    "test": {
      "checked": 400,
      "missing": 0,
      "missing_examples": []
    },
    "train": {
      "checked": 1400,
      "missing": 0,
      "missing_examples": []
    },
    "val": {
      "checked": 200,
      "missing": 0,
      "missing_examples": []
    }
  },
  "split_overlaps": {
    "train_test": {
      "image_ids": 0,
      "paths": 0,
      "patients": 0
    },
    "train_val": {
      "image_ids": 0,
      "paths": 0,
      "patients": 0
    },
    "val_test": {
      "image_ids": 0,
      "paths": 0,
      "patients": 0
    }
  },
  "test_patients": 82,
  "test_rows": 400,
  "train_patients": 395,
  "train_rows": 1400,
  "val_patients": 66,
  "val_rows": 200
}
```

## Class Counts

```json
{
  "test": {
    "atelectasis": 31,
    "cardiomegaly": 7,
    "consolidation": 30,
    "edema": 14,
    "pleural_effusion": 86,
    "pneumonia": 11,
    "pneumothorax": 27
  },
  "train": {
    "atelectasis": 129,
    "cardiomegaly": 25,
    "consolidation": 75,
    "edema": 30,
    "pleural_effusion": 174,
    "pneumonia": 21,
    "pneumothorax": 70
  },
  "val": {
    "atelectasis": 29,
    "cardiomegaly": 13,
    "consolidation": 4,
    "edema": 7,
    "pleural_effusion": 29,
    "pneumonia": 0,
    "pneumothorax": 8
  }
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
