# NIH 2k Manifest Check

## train
- images: 1400
- patients: 395
- image path column: `abs_path`
- patient column: `Patient ID`
- image id column: `image_id`
- label counts:
  - Atelectasis: 129
  - Cardiomegaly: 25
  - Consolidation: 75
  - Edema: 30
  - Effusion: 174

## val
- images: 200
- patients: 66
- image path column: `abs_path`
- patient column: `Patient ID`
- image id column: `image_id`
- label counts:
  - Atelectasis: 29
  - Cardiomegaly: 13
  - Consolidation: 4
  - Edema: 7
  - Effusion: 29

## test
- images: 400
- patients: 82
- image path column: `abs_path`
- patient column: `Patient ID`
- image id column: `image_id`
- label counts:
  - Atelectasis: 31
  - Cardiomegaly: 7
  - Consolidation: 30
  - Edema: 14
  - Effusion: 86

## Overlap
- train_vs_val: patient_overlap=0, image_overlap=0
- train_vs_test: patient_overlap=0, image_overlap=0
- val_vs_test: patient_overlap=0, image_overlap=0
