# Uncertainty Blockers

## Resolved In This Pass

- Figure 3 now uses existing run-level rows from `results/runs/harvest_invasion/curated/harvest_oversight_gap_stageA_runs.csv` to add approximate 95% confidence intervals.
- Figure 6 now uses existing LLM sampled-population rows from `*_map_samples.csv` to add approximate 95% confidence intervals.
- LLM uncertainty summaries are written to `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_uncertainty_summary.csv`.
- Full threshold sensitivity for the local safety margin and global patch-health threshold is now available from the recovered 9,000-row sharded sweep.

## Remaining Blockers

### Per-step threshold replay without reruns

The current full Stage A logs store derived local/global safety metrics at the configured threshold. They do not store per-step request and patch-state traces for every episode. Because of that, new threshold definitions cannot be recomputed offline across the full matrix without reruns.

Required fix:

- log per-step `max_requested_frac`, `mean_patch_health_after`, `failed_patch_fraction_after`, and local/global predicate flags for all target cells; or
- rerun selected/full matrix cells with alternative threshold values if a new safety definition is needed.

### More case-level uncertainty

The current paper has one extracted local-pass/global-fail case trace. That is enough for explanation, not for estimating how traces vary.

Required fix:

- extract multiple cases across conditions and capability gaps; or
- store per-step traces during targeted reruns.

### Overseer-limit attribution

The current overseer capability axis bundles detection recall, delay, target capacity, and cost. Existing Stage A data cannot isolate which component drives which outcome.

Required fix:

- run an ablation where one overseer limit varies at a time while the others stay at the strong setting.
