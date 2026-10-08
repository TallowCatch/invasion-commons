# Validation Status

## Completed From Existing Logs

- Stage A condition means, actor-capability validation, and trace-level threshold sensitivity were recomputed successfully with:

```bash
python -m experiments.analyze_harvest_oversight_stageA \
  --table-csv results/runs/showcase/curated/harvest_oversight_gap_stageA_table.csv \
  --case-trace-csv results/runs/showcase/curated/harvest_oversight_gap_stageA_oversight_case_trace.csv \
  --strategy-history-csv results/runs/harvest_invasion/curated/harvest_oversight_gap_stageA_strategy_history.csv \
  --output-prefix results/runs/showcase/curated/harvest_oversight_gap_stageA
```

- LLM bridge uncertainty was computed from existing sampled-population rows:

```bash
python -m experiments.analyze_llm_bridge_uncertainty
```

- Stage A held-out stress-regime summaries were computed from generation history:

```bash
python -m experiments.analyze_stagea_stress_regimes
```

- Paper input checks pass:

```bash
python -m experiments.check_paper_inputs
```

## Reduced Threshold Sweep Completed

A reduced but meaningful threshold sweep was run to test whether the local-pass/global-fail metric survives nearby threshold changes without launching the full Stage A matrix:

```bash
python -m experiments.run_targeted_threshold_replay \
  --scenarios community_irrigation,forest_co_management \
  --conditions none,bottom_up_only,top_down_only,hybrid \
  --actor-capability-levels high_actor \
  --overseer-capability-levels weak_overseer \
  --local-margins 0.00,0.025,0.05,0.075,0.10 \
  --global-thresholds 9.0,9.5,10.0,10.5,11.0 \
  --full-grid \
  --n-runs 1 \
  --generations 3 \
  --seeds-per-generation 8 \
  --test-seeds-per-generation 8 \
  --output-dir results/runs/threshold_replay/reduced_full_grid \
  --summary-csv results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_reduced_full_grid.csv \
  --execute

python -m experiments.analyze_threshold_replay_grid
```

Outputs:

- `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_reduced_full_grid.csv`
- `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_reduced_full_grid_condition_summary.csv`
- `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_reduced_full_grid_pair_summary.csv`
- `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_reduced_full_grid_winner_summary.csv`
- `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_reduced_full_grid_summary.md`
- `paper/paper_v5_scalable_oversight_commons/tables/table_reduced_threshold_sweep.tex`

Scope:

- both stress settings: `community_irrigation` and `forest_co_management`;
- all four oversight architectures;
- one actor level: `high_actor`;
- one overseer level: `weak_overseer`;
- all 25 threshold pairs;
- one run;
- three generations;
- eight train and eight test seeds per generation.

Main result:

- in the moderate-coupling setting, local-pass/global-fail remains effectively zero across all 25 threshold pairs for local, global-signal, and hybrid oversight;
- in the high-coupling setting, local-pass/global-fail remains positive in all 25 threshold pairs for no oversight, local oversight, global signal, and hybrid oversight;
- the patch-health ranking is stable inside this slice: hybrid wins all 25 moderate-coupling pairs, while global signal wins all 25 high-coupling pairs.

This reduced slice was useful as a pipeline and analysis check before launching the full sharded run.

## Full Threshold Robustness Completed

The full threshold sweep has now been completed through the sharded GitHub Actions workflow and aggregate recovery path, then verified with:

```bash
python -m experiments.audit_threshold_sweep_completeness
```

It covers:

- both stress settings;
- all 25 local/global threshold pairs;
- all three actor-capability levels;
- all three overseer-capability levels;
- all four oversight architectures;
- five runs per cell.

Main full-grid result:

- the recovered summary has 9,000 rows;
- the completeness audit found 9,000 expected run keys, 9,000 observed unique run keys, no missing run keys, no duplicate run keys, and no bad run-count cells;
- the default-threshold subset exactly matches the original Stage A run-level defaults on the checked metrics;
- across 450 threshold-capability decision cells, hybrid wins 375 by patch health and the global signal wins 75;
- no oversight and local oversight never rank first by patch health in the full threshold sweep;
- local-pass/global-fail remains measurable across many threshold-capability cells, especially in the high-coupling setting.

Files:

- `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid_recovered.csv`
- `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid_recovered_summary.md`
- `notes/THRESHOLD_SWEEP_COMPLETENESS_AUDIT.md`
- `paper/paper_v5_scalable_oversight_commons/tables/table_threshold_robustness_sweep.tex`
- `paper/paper_v5_scalable_oversight_commons/figures/fig07_threshold_robustness.pdf`

## Still Requires Larger Validation

- Additional LLM model or larger strategy bank if the LLM bridge is meant to be more than a pilot.
- More case traces if the paper wants to characterize local-pass/global-fail qualitatively beyond one selected episode.
- Additional safety-definition sensitivity, especially failed-patch threshold variation or alternative aggregate-state predicates.

## Full Matrix Execution Status

The full threshold-robustness check was too large for local single-worker execution and was moved to a sharded remote workflow.

Local status:

- the replay wrapper now supports resume and incremental summary writes;
- a local full-grid launch was started and then terminated deliberately;
- the full paper-setting grid is too large for practical local single-worker execution.

Remote status:

- a sharded workflow completed the full sweep;
- the logical grid is split by scenario, local margin, global threshold, actor capability, and overseer capability;
- each logical shard runs all four oversight architectures under the paper settings;
- GitHub Actions infrastructure handles grouped shard execution, artifact merge, and summary analysis;
- aggregate recovery was added to merge the run-level robustness summary without concatenating all large history files.

See:

- `notes/harvest_threshold_replay_remote_runbook.md`
- `.github/workflows/harvest-threshold-replay.yml`
- `.github/threshold_replay_request.json`

## Reduced Overseer-Limit Ablation Completed

The reduced high-coupling ablation was run successfully with:

```bash
python -m experiments.run_overseer_limit_ablation --execute
```

Outputs:

- `results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced.csv`
- `results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced_compact.csv`
- `results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced_summary.md`
- `results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced_table.tex`
- `results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced.pdf`
- `results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced_audit.md`

Completeness check:

- expected run rows: 28;
- observed run rows: 28;
- missing run keys: 0;
- duplicate run rows: 0;
- null values in checked metrics: 0.

Reduced-run mechanism summary:

- isolated delay is the first single overseer limitation that produces nonzero unsafe rate and local-pass/global-fail rate;
- isolated recall and isolated capacity raise missed-target rates but do not by themselves produce unsafe outcomes in this reduced check;
- isolated cost appears mainly as burden;
- the bundled weak overseer causes the largest overall degradation.

This remains a mechanism check, not a full robustness result, because it uses one stress setting, one actor-capability level, two protected architectures, two runs, and reduced seed counts.

## Full Threshold Sensitivity Command Template

This is intentionally not run by default:

```bash
python -m experiments.run_targeted_threshold_replay \
  --scenarios community_irrigation,forest_co_management \
  --conditions none,bottom_up_only,top_down_only,hybrid \
  --actor-capability-levels low_actor,medium_actor,high_actor \
  --overseer-capability-levels strong_overseer,limited_overseer,weak_overseer \
  --local-margins 0.00,0.025,0.05,0.075,0.10 \
  --global-thresholds 9.0,9.5,10.0,10.5,11.0 \
  --full-grid \
  --n-runs 5 \
  --generations 15 \
  --seeds-per-generation 32 \
  --test-seeds-per-generation 32 \
  --execute
```

This would be expensive because it repeats the Stage A matrix across threshold settings. The reduced sweep above is the current evidence-bearing compromise. A better long-term route is to log per-step traces once and recompute thresholds offline.

## Reduced Overseer-Limit Ablation Command Template

Implemented and now executed in reduced form:

```bash
python -m experiments.run_overseer_limit_ablation --execute
```

Dry-run command generated by the script:

```bash
python -m experiments.run_harvest_invasion_matrix \
  --scenario-presets forest_co_management \
  --conditions top_down_only,hybrid \
  --actor-capability-levels high_actor \
  --overseer-capability-levels strong_overseer,recall_limited_only,delay_limited_only,capacity_limited_only,cost_limited_only,limited_overseer,weak_overseer \
  --n-runs 2 \
  --generations 8 \
  --population-size 6 \
  --seeds-per-generation 16 \
  --test-seeds-per-generation 16 \
  --replacement-fraction 0.2 \
  --max-workers 1 \
  --output-prefix results/runs/overseer_limit_ablation/reduced_ablation \
  --experiment-tag harvest_overseer_limit_ablation \
  --no-progress
```
