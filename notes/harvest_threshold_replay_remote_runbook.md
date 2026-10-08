# Harvest Threshold Replay Remote Runbook

## Purpose

This runbook covers the full threshold-robustness check for the scalable-oversight Harvest paper. The full grid is too large for local single-worker execution, so the intended path is a sharded remote workflow.

Current paper status:

- targeted replay implemented;
- reduced high-actor / weak-overseer threshold sweep completed;
- full-matrix threshold robustness completed and recovered from the aggregate bundle.

## Sharding Scheme

The full logical grid is:

- 2 scenarios
- 5 local safety margins
- 5 global patch-health thresholds
- 3 actor capability levels
- 3 overseer capability levels

This gives:

- 450 logical shards

Each logical shard runs:

- one scenario
- one local margin
- one global threshold
- one actor level
- one overseer level
- all four oversight architectures
- all paper seed settings

## Remote Workflow

Workflow:

- `.github/workflows/harvest-threshold-replay.yml`
- `.github/threshold_replay_request.json`

Core helper scripts:

- `experiments/threshold_replay_shards.py`
- `experiments/run_threshold_replay_shard_group.py`
- `experiments/merge_threshold_replay_shards.py`
- `experiments/analyze_threshold_replay_grid.py`

Recommended default:

- `group_size = 6`

That turns 450 logical shards into:

- 75 GitHub Actions jobs

This is large but realistic for remote execution. If needed, use:

- `max_groups`

to run a partial batch first.

## Suggested First Remote Run

Run the smoke validation batch first by committing and pushing this request file:

- `.github/threshold_replay_request.json`

Smoke settings:

- `group_size = 6`
- `max_groups = 1`
- `n_runs = 1`
- `generations = 2`
- `seeds_per_generation = 4`
- `test_seeds_per_generation = 4`
- `run_name_suffix = smoke`

That executes 6 logical shards with reduced run settings and checks that:

- shard jobs complete cleanly;
- artifacts upload correctly;
- merge succeeds;
- analysis outputs are produced in the expected format.

Then launch the full remote sweep by updating the same request file to:

- `group_size = 6`
- `max_groups = null`
- `n_runs = 5`
- `generations = 15`
- `seeds_per_generation = 32`
- `test_seeds_per_generation = 32`
- `run_name_suffix = ""`

The workflow also supports manual `workflow_dispatch` inputs, but the request-file path is the fallback when the GitHub CLI is unavailable locally.

## Expected Outputs

Merged outputs:

- `results/runs/threshold_replay/curated/harvest_oversight_gap_threshold_replay_full_grid_runs.csv`
- `results/runs/threshold_replay/curated/harvest_oversight_gap_threshold_replay_full_grid_generation_history.csv`
- `results/runs/threshold_replay/curated/harvest_oversight_gap_threshold_replay_full_grid_strategy_history.csv`
- `results/runs/threshold_replay/curated/harvest_oversight_gap_threshold_replay_full_grid_agent_history.csv`

Replay summary:

- `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid.csv`

Analysis outputs:

- `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid_pair_summary.csv`
- `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid_winner_summary.csv`
- `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid_condition_summary.csv`
- `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid_summary.md`
- `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid_table.tex`

## What To Tell The Supervisor

Use this wording:

“The full threshold robustness check was too large for local single-worker execution, so I moved it to a sharded remote workflow. The recovered full-grid summary now covers 25 threshold pairs, all actor and overseer capability settings, both stress settings, and all four oversight architectures.”

## What Not To Claim Yet

Claim carefully:

- the completed sweep supports robustness to local safety margin and global patch-health threshold variation;
- it does not test every possible global safety definition;
- it does not vary the failed-patch threshold or broader environment dynamics.
