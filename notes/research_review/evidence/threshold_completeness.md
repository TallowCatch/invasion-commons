# Threshold Sweep Completeness Audit

## Files Checked

- Summary CSV: `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid_recovered.csv`
- Raw recovered runs CSV: `results/runs/threshold_replay/curated/harvest_oversight_gap_threshold_replay_full_grid_recovered_runs.csv`
- Stage A default runs CSV: `results/runs/harvest_invasion/curated/harvest_oversight_gap_stageA_runs.csv`
- GitHub artifact bundle: `results/artifacts/github/harvest_oversight_gap_threshold_replay_full_grid_recovered-bundle.zip`

## Expected Grid

- Stress settings: 2
- Local safety margins: 5
- Global patch-health thresholds: 5
- Actor-capability levels: 3
- Overseer-capability levels: 3
- Oversight architectures: 4
- Runs per cell: 5
- Expected run rows: 9000
- Expected architecture cells before run expansion: 1800
- Expected patch-health decision cells: 450

## Observed Counts

- Observed run rows: 9000
- Observed unique run keys: 9000
- Observed architecture cells before run expansion: 1800
- Duplicate run rows: 0
- Bad run-count cells: 0
- Missing architecture cells: 0
- Missing run keys: 0
- Unexpected run keys: 0

## Metric Null Check

- `test_global_unsafe_rate_mean` null values: 0
- `test_local_pass_global_fail_rate_mean` null values: 0
- `test_mean_patch_health_mean` null values: 0
- `test_garden_failure_mean` null values: 0
- `test_mean_welfare_mean` null values: 0
- `test_governance_budget_spent_mean` null values: 0

## Winner Count Check

- Observed patch-health decision cells: 450

| Stress setting | Winner | Patch-health winning cells |
| --- | --- | ---: |
| Moderate coupling | Hybrid | 175 |
| Moderate coupling | Global signal | 50 |
| High coupling | Hybrid | 200 |
| High coupling | Global signal | 25 |

| Winner | Total patch-health winning cells |
| --- | ---: |
| Hybrid | 375 |
| Global signal | 75 |

## Missing Or Duplicate Cells

- Missing architecture cells: None.
- Missing run keys: None.
- Unexpected run keys: None.
- Cells with non-five run count: None.

## Default-Threshold Consistency

- Default-threshold replay exactly matches the Stage A run-level defaults on checked metrics.

| Metric | Maximum absolute difference vs Stage A defaults |
| --- | ---: |
| `test_global_unsafe_rate_mean` | 0 |
| `test_local_pass_global_fail_rate_mean` | 0 |
| `test_mean_patch_health_mean` | 0 |
| `test_garden_failure_mean` | 0 |
| `test_mean_welfare_mean` | 0 |
| `test_governance_budget_spent_mean` | 0 |

## Citation Safety

Safe to cite in the paper: yes. The recovered summary contains the complete expected grid, no duplicate run keys, five runs per cell, no nulls in checked metrics, and exact default-threshold agreement with the Stage A run-level defaults.

Note: this audit verifies completeness of the recovered merged artifact. It does not re-audit the GitHub Actions UI logs shard by shard. A silently skipped shard would show up here as a missing expected run key or a non-five run-count cell.
