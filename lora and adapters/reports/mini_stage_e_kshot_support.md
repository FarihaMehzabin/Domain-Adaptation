# Mini-Stage E K-shot Support

## Goal

Create clean K-shot support sets from MIMIC train_pool.

## Input Files

```json
{
  "test_csv": "/workspace/manifests/mimic_common5_test.csv",
  "train_pool_csv": "/workspace/manifests/mimic_common5_train_pool.csv",
  "val_csv": "/workspace/manifests/mimic_common5_val.csv"
}
```

## Label Names

```json
[
  "Atelectasis",
  "Cardiomegaly",
  "Consolidation",
  "Edema",
  "Effusion"
]
```

## K-shot Definition

K-shot means we try to select at least K positive examples for each disease. This is multi-label data, so one image may count for more than one disease.

## Leakage Checks

```json
{
  "dicom_overlap": {
    "train_pool_vs_test": {
      "count": 0,
      "examples": []
    },
    "train_pool_vs_val": {
      "count": 0,
      "examples": []
    },
    "val_vs_test": {
      "count": 0,
      "examples": []
    }
  },
  "image_overlap": {
    "train_pool_vs_test": {
      "count": 0,
      "examples": []
    },
    "train_pool_vs_val": {
      "count": 0,
      "examples": []
    },
    "val_vs_test": {
      "count": 0,
      "examples": []
    }
  },
  "image_path_columns": {
    "test": "abs_path",
    "train_pool": "abs_path",
    "val": "abs_path"
  },
  "label_binary": {
    "test": {
      "Atelectasis": {
        "invalid_count": 0,
        "invalid_values": []
      },
      "Cardiomegaly": {
        "invalid_count": 0,
        "invalid_values": []
      },
      "Consolidation": {
        "invalid_count": 0,
        "invalid_values": []
      },
      "Edema": {
        "invalid_count": 0,
        "invalid_values": []
      },
      "Effusion": {
        "invalid_count": 0,
        "invalid_values": []
      }
    },
    "train_pool": {
      "Atelectasis": {
        "invalid_count": 0,
        "invalid_values": []
      },
      "Cardiomegaly": {
        "invalid_count": 0,
        "invalid_values": []
      },
      "Consolidation": {
        "invalid_count": 0,
        "invalid_values": []
      },
      "Edema": {
        "invalid_count": 0,
        "invalid_values": []
      },
      "Effusion": {
        "invalid_count": 0,
        "invalid_values": []
      }
    },
    "val": {
      "Atelectasis": {
        "invalid_count": 0,
        "invalid_values": []
      },
      "Cardiomegaly": {
        "invalid_count": 0,
        "invalid_values": []
      },
      "Consolidation": {
        "invalid_count": 0,
        "invalid_values": []
      },
      "Edema": {
        "invalid_count": 0,
        "invalid_values": []
      },
      "Effusion": {
        "invalid_count": 0,
        "invalid_values": []
      }
    }
  },
  "study_overlap": {
    "train_pool_vs_test": {
      "count": 0,
      "examples": []
    },
    "train_pool_vs_val": {
      "count": 0,
      "examples": []
    },
    "val_vs_test": {
      "count": 0,
      "examples": []
    }
  },
  "subject_overlap": {
    "train_pool_vs_test": {
      "count": 0,
      "examples": []
    },
    "train_pool_vs_val": {
      "count": 0,
      "examples": []
    },
    "val_vs_test": {
      "count": 0,
      "examples": []
    }
  }
}
```

## Support Set Summary: K=5

```json
{
  "reached_k": {
    "Atelectasis": true,
    "Cardiomegaly": true,
    "Consolidation": true,
    "Edema": true,
    "Effusion": true
  },
  "total_images_selected": 7,
  "total_studies_selected": 7,
  "total_subjects_selected": 7
}
```

```json
{
  "Atelectasis": 5,
  "Cardiomegaly": 6,
  "Consolidation": 5,
  "Edema": 6,
  "Effusion": 7
}
```

## Support Set Summary: K=20

```json
{
  "reached_k": {
    "Atelectasis": true,
    "Cardiomegaly": true,
    "Consolidation": true,
    "Edema": true,
    "Effusion": true
  },
  "total_images_selected": 30,
  "total_studies_selected": 30,
  "total_subjects_selected": 29
}
```

```json
{
  "Atelectasis": 20,
  "Cardiomegaly": 22,
  "Consolidation": 20,
  "Edema": 20,
  "Effusion": 26
}
```

## Adaptation Validation Set

Use the existing validation manifest only: `/workspace/manifests/mimic_common5_val.csv`

## Warnings

```json
[]
```

## Final Decision

```json
{
  "failure_reasons": [],
  "safe_to_continue": true,
  "status": "DONE"
}
```
