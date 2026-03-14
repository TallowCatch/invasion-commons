# harvest search_mutation next plan

## Summary
The next Study 2 evidence cycle should not be another live LLM sweep. The next cycle should strengthen Harvest Commons with a more adversarial non-LLM baseline that stays inside the current invasion setup:

- `mutation`
- `search_mutation`

This gives a tighter comparison than the current live LLM run, avoids runtime waste, and tells us whether the Harvest governance-architecture result survives a stronger attacker than plain mutation.

## Why this is next
- Harvest LLM Stage B worked technically, but it is still too noisy and costly to justify Stage C right away.
- The current non-LLM Harvest evidence is already strong.
- `search_mutation` is the cleanest next stressor because it remains:
  - non-LLM
  - strategy-level
  - inside the same population turnover logic

## Core question
Does the Harvest Commons result, especially the advantage of `hybrid` over `top_down_only`, survive a stronger search-based invader rather than plain mutation?

## Experimental sequence

### Stage A: search_mutation smoke
Run one small matched cell:
- tier: `medium_h1`
- partner mix: `balanced`
- conditions: `top_down_only`, `hybrid`
- injectors: `mutation`, `search_mutation`
- adversarial pressure: `0.3`
- runs: `1`
- generations: `4`
- train/test seeds: `8`

Success criteria:
- run completes cleanly
- `search_mutation` behaves deterministically under the same seed
- summaries and paired deltas generate without manual edits

### Stage B: narrow pilot matrix
Run:
- tiers: `medium_h1`, `hard_h1`
- partner mixes: `balanced`, `adversarial_heavy`
- conditions: `top_down_only`, `hybrid`
- injectors: `mutation`, `search_mutation`
- adversarial pressure: `0.3`, `0.5`
- runs per cell: `3`
- generations: `12`
- train/test seeds per generation: `24`

Goal:
- determine whether `search_mutation` materially weakens the current Harvest result
- identify which rows deserve high-power follow-up

### Stage C: high-power decision matrix
Only if Stage B is useful.

Run:
- tiers: `medium_h1`, `hard_h1`
- partner mixes: `balanced`, `adversarial_heavy`
- conditions: `top_down_only`, `hybrid`
- injectors: `search_mutation`
- adversarial pressure: `0.3`, `0.5`
- runs per cell: `8`
- generations: `15`
- train/test seeds per generation: `32`

Goal:
- produce a confidence-backed Harvest stress test stronger than plain mutation

## Main outputs to compare
- `test_mean_patch_health`
- `test_mean_welfare`
- `test_mean_max_local_aggression`
- `test_mean_neighborhood_overharvest`
- `test_mean_prevented_harvest`
- `per_regime_health_survival_over_generations`

## Decision rule
Advance the new result into the paper only if:
1. `hybrid` remains competitive or better than `top_down_only` in most pilot cells
2. patch health and neighborhood overharvest remain directionally favorable
3. the welfare penalty stays comparable to, or better than, the current mutation-only Harvest result

If `search_mutation` breaks the result badly, that is still useful:
- it becomes evidence that the current Study 2 conclusion is baseline-sensitive
- and it tells us the next improvement should target mechanism design, not LLMs

## Expected artifacts
- `results/runs/harvest_invasion/curated/harvest_invasion_search_stageB_*`
- `results/runs/showcase/curated/harvest_invasion_search_stageB_*`
- if promoted:
  - `results/runs/harvest_invasion/curated/harvest_invasion_search_stageC_*`
  - `results/runs/showcase/curated/harvest_invasion_search_stageC_*`

## Command sketch
Smoke:
```bash
python -m experiments.run_harvest_invasion_matrix \
  --tiers medium_h1 \
  --partner-mixes balanced \
  --conditions top_down_only,hybrid \
  --injector-modes mutation,search_mutation \
  --adversarial-pressures 0.3 \
  --n-runs 1 \
  --generations 4 \
  --seeds-per-generation 8 \
  --test-seeds-per-generation 8 \
  --output-prefix results/runs/harvest_invasion/curated/harvest_invasion_search_stageA_smoke \
  --experiment-tag harvest_invasion_search_stageA
```

Pilot:
```bash
python -m experiments.run_harvest_invasion_matrix \
  --tiers medium_h1,hard_h1 \
  --partner-mixes balanced,adversarial_heavy \
  --conditions top_down_only,hybrid \
  --injector-modes mutation,search_mutation \
  --adversarial-pressures 0.3,0.5 \
  --n-runs 3 \
  --generations 12 \
  --seeds-per-generation 24 \
  --test-seeds-per-generation 24 \
  --output-prefix results/runs/harvest_invasion/curated/harvest_invasion_search_stageB \
  --experiment-tag harvest_invasion_search_stageB
```

Summary:
```bash
python -m experiments.summarize_harvest_invasion \
  --runs-csv results/runs/harvest_invasion/curated/harvest_invasion_search_stageB_runs.csv \
  --output-prefix results/runs/showcase/curated/harvest_invasion_search_stageB
```

Plots:
```bash
MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp \
python -m experiments.plot_harvest_invasion \
  --ci-csv results/runs/showcase/curated/harvest_invasion_search_stageB_ci.csv \
  --output-prefix results/runs/showcase/curated/harvest_invasion_search_stageB
```
