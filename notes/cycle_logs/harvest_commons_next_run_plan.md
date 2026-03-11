# Harvest Commons next run plan

## Goal
Move Harvest Commons from a tuned prototype into a Study 2 evidence matrix that is comparable in discipline to the fishery Study 1b runs.

## What is already done
- Batch 8 is the current reference batch.
- Hybrid now reacts to local aggressive neighborhoods.
- Bottom-up signals affect later harvest behavior.
- Local mechanism metrics are logged.

Reference artifacts:
- `results/runs/harvest/curated/harvest_commons_batch8_neighbor_targeted_table.csv`
- `results/runs/showcase/curated/harvest_commons_batch8_neighbor_targeted_readout.md`
- `results/runs/showcase/curated/harvest_commons_batch8_neighbor_targeted_hybrid_vs_topdown_deltas.png`

## Next run block: Harvest matrix v1
This is the direct analogue of the fishery matrix.

### Axis 1: social composition
- `cooperative_heavy`
- `mixed_pressure`
- `adversarial_heavy`

### Axis 2: governance condition
- `none`
- `top_down_only`
- `bottom_up_only`
- `hybrid`

### Axis 3: harvest environment tier
Add three Harvest Commons presets next:
- `easy_h1`
- `medium_h1`
- `hard_h1`

Suggested meaning:
- `easy_h1`: higher regeneration, lower weather noise, lower neighbor externality
- `medium_h1`: current default tuned setting
- `hard_h1`: lower regeneration, higher weather noise, stronger neighbor externality

## Runs to execute after tier presets exist
### 1. Pilot matrix
Purpose: check that ranking does not collapse under the new tier axis.

Recommended settings:
- `n_runs=6`
- all 3 social mixes
- all 4 governance conditions
- all 3 tiers

### 2. High-power decision matrix
Purpose: paper-grade evidence for `hybrid` versus `top_down_only`.

Recommended settings:
- `n_runs=20`
- social mixes: `mixed_pressure,adversarial_heavy`
- conditions: `top_down_only,hybrid`
- tiers: `medium_h1,hard_h1`

Main outputs to compare:
- `mean_patch_health`
- `mean_welfare`
- `mean_max_local_aggression`
- `mean_neighborhood_overharvest`
- `mean_targeted_agent_fraction`

### 3. Bottom-up validation block
Purpose: prove that the bottom-up channel matters and is not cosmetic.

Recommended settings:
- `n_runs=12`
- social mixes: all 3
- conditions: `none,bottom_up_only,hybrid`
- tier: `medium_h1`

Main outputs to compare:
- `total_credit_transferred`
- `mean_welfare`
- `mean_aggressive_request_fraction`
- `mean_neighborhood_overharvest`

## Command skeletons
### Current tuned batch reference
```bash
python -m experiments.run_harvest_study \
  --n-runs 12 \
  --social-mixes cooperative_heavy,mixed_pressure,adversarial_heavy \
  --government-trigger 15.5 \
  --strict-cap-frac 0.18 \
  --relaxed-cap-frac 0.32 \
  --soft-trigger 17.5 \
  --deterioration-threshold 0.25 \
  --activation-warmup 3 \
  --aggressive-request-threshold 0.7 \
  --aggressive-agent-fraction-trigger 0.34 \
  --local-neighborhood-trigger 0.67 \
  --output-prefix results/runs/harvest/harvest_commons_batch8_neighbor_targeted
```

### Plot current tuned batch
```bash
MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp \
python -m experiments.plot_harvest_commons \
  --table-csv results/runs/harvest/curated/harvest_commons_batch8_neighbor_targeted_table.csv \
  --output-prefix results/runs/showcase/curated/harvest_commons_batch8_neighbor_targeted
```

## Immediate implementation work before the matrix
1. Add Harvest Commons tier presets.
2. Add `--conditions` filtering to `run_harvest_study.py`, like the fishery side already has.
3. Add one summary script for Harvest Commons matrices, similar to `summarize_study1b.py` but simpler.

## Decision rule for the next cycle
Harvest Commons is ready to become a real Study 2 matrix only after tiers exist.
Until then, use batch 8 as the mechanism demonstration, not as the full generalization claim.
