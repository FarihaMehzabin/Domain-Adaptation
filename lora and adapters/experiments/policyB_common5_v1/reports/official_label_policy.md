# Official MIMIC Label Policy

The official primary MIMIC label policy for the common 5 target labels is `uignore_blankzero`.

## Mapping

| Raw value | Target label | Mask |
| --- | ---: | ---: |
| `1` | `1` | `1` |
| `0` | `0` | `1` |
| `-1` | `0` | `0` |
| blank / `NaN` | `0` | `1` |

## Rationale

Uncertain labels (`-1`) are masked because they do not provide reliable positive-or-negative supervision for the target task. Keeping them as valid negatives would force an arbitrary decision into both adaptation and evaluation.

Blank labels are currently treated as zero so that the target-side manifests preserve practical coverage across the common 5 labels. This is a modeling assumption rather than a ground-truth guarantee, and `blank-zero` will be documented as a limitation anywhere final target-side results are reported.

Policy C (`uignore_blankignore`) was rejected as the primary policy because blank-ignore caused large coverage loss in the MIMIC common5 target evaluation pool. It remains useful as a sensitivity analysis, but not as the official primary target-side manifest policy.

Policy A (`current_uzero_blankzero`) old results are preliminary only. They were produced before the official masking decision for uncertain labels and should not be treated as final target-side numbers.

Final target-side validation, test, and adaptation results must use the Policy B manifests:

- `manifests/mimic_common5_policyB_train_pool.csv`
- `manifests/mimic_common5_policyB_val.csv`
- `manifests/mimic_common5_policyB_test.csv`
