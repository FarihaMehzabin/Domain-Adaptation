# Experiments Layout

This directory is organized into one canonical storage view and several human-facing views.

## Canonical Storage

- `by_id/`
  - The one true location for every experiment directory.
  - New experiment references should prefer this path.

## Human-Facing Views

- `campaigns/`
  - Numbered scientific groupings of experiments.
  - Each campaign folder contains symlinks back to `by_id/`.
- `index.csv`
  - Flat registry of all experiments, their campaign assignment, and their canonical path.

## Operational View

- `running/`
  - Reserved for truly live jobs if you want a temporary operational view.
  - It is intentionally separate from the canonical experiment archive.

## Notes

- Some historical experiment IDs were reused with different suffixes.
  - `exp0061`
  - `exp0068`
- Because of that, the full directory name is the real unique identifier, not the short `expNNNN` prefix alone.
