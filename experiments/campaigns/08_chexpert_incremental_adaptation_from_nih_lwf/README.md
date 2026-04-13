# 08 CheXpert Incremental Adaptation From NIH LwF

This campaign contains the first `NIH -> CheXpert` warm-start adaptation sweep with Learning without Forgetting (`LwF`) on frozen `CXR Foundation` embeddings.

Included here:

- warm-started `CheXpert 1000-shot` head-adaptation runs
- frozen `NIH` teacher preservation via `LwF` on `NIH train`
- alpha and temperature sweep runs
- post-adaptation evaluation on both `CheXpert test` and `NIH test`

Primary range:

- `exp0076` to `exp0084`
