---
name: hydra-ablation-sweep
description: >
  Sets up and runs a single-hyperparameter ablation sweep for I-JEPA-AD (tau, lambda,
  predictor depth, loss type, per PRD §7.4) as Hydra config overrides with multiple
  seeds, writing results to a consistent CSV schema. Use this whenever setting up or
  running any ablation from the PRD's ablation matrix.
---

# Hydra Ablation Sweep

1. Express the swept hyperparameter as a single config field override — never hardcode
   the sweep values in a script; use Hydra multirun (`-m`) or an explicit loop over
   config overrides.
2. Hold every other config field fixed for the duration of the sweep — confirm this
   before running, since accidental cross-contamination between ablations is a common
   research-code bug.
3. Run each configuration with at least 3 different seeds (per
   IMPLEMENTATION_PLAN.md §5); do not report a single-seed result as an ablation point.
4. Write results to results/ablations/<sweep_name>.csv with one row per
   (config_value, seed) pair, and columns matching the schema used elsewhere so
   generate_paper_assets.py can consume it without a one-off parser.
5. After the sweep, compute mean/std (or bootstrap CI) per config_value before
   reporting a trend — a single point estimate is not an ablation result.
6. Log the sweep's config and result file path in FLOW.md's LLD data-schema section
   if it introduces a new results schema.
