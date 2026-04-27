# Policy B Common5 Workspace

This folder contains the official Policy B common5 experiments.

The legacy root-level folders such as `checkpoints/`, `outputs/`, `reports/`, `logs/`, and `manifests/` are provisional or historical and must remain untouched.

All new official checkpoints, outputs, logs, and reports must be written inside this namespace: `experiments/policyB_common5_v1/`.

Adaptation runs from the old policy must be rerun here before they are treated as official Policy B results.

The no-adaptation NIH source model can be reevaluated for Policy B comparison, but final adaptation comparisons require Policy B-trained adapters or Policy B full fine-tunes produced inside this namespace.
