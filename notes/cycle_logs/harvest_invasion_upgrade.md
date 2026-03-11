# Harvest invasion upgrade

## Purpose
Upgrade Harvest Commons from a fixed-composition governance comparison into a true invasion study with the same population turnover logic used in Fishery Commons.

## New code paths
- `fishery_sim/harvest.py`
- `fishery_sim/harvest_benchmarks.py`
- `fishery_sim/harvest_evolution.py`
- `experiments/run_harvest_invasion.py`
- `experiments/run_harvest_invasion_matrix.py`
- `experiments/summarize_harvest_invasion.py`
- `experiments/plot_harvest_invasion.py`
- `notebooks/06_harvest_invasion.ipynb`

## Study 2 framing
- Study 1: central governance signals under invasion pressure in Fishery Commons.
- Study 2: governance architecture under invasion pressure in Harvest Commons.

## Conditions
- `none`
- `top_down_only`
- `bottom_up_only`
- `hybrid`

## Injectors
- `random`
- `mutation`
- `adversarial_heuristic`

## Tier presets
- `easy_h1`
- `medium_h1`
- `hard_h1`

## Held-out regime pack per tier
- `noisy_weather`
- `strong_externality`
- `low_init`
- `slow_regen`

## Default run outputs
Matrix runner writes:
- `*_runs.csv`
- `*_generation_history.csv`
- `*_strategy_history.csv`

Summarizer writes:
- `*_table.csv`
- `*_ranking.csv`
- `*_delta.csv`
- `*_ci.csv`
- `*_summary.md`

Plotter writes:
- `*_main.png`
- `*_mechanisms.png`

## Stage names
- Stage A smoke: `harvest_invasion_stageA`
- Stage B pilot: `harvest_invasion_stageB_pilot_v2`
- Stage C high-power: `harvest_invasion_stageC_highpower_v2`
