# Overseer-Limit Ablation Plan

## Purpose

The next technical question is which overseer limitation drives failure: detection recall, delay, targeting capacity, or intervention cost. The current overseer capability axis bundles these limitations into strong, limited, and weak overseer presets. That is acceptable for a first benchmark, but a reviewer can reasonably ask which part of the bundle matters.

The ablation should vary one overseer limitation at a time and compare each isolated limitation against a strong overseer baseline and the bundled limited/weak overseer presets.

## Overseer settings

| Setting | Detection recall | Delay | Max target share | Budget cost | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `strong_overseer` | 1.0 | 0 | 1.0 | 0.00 | Baseline with no imposed overseer limitation. |
| `recall_limited_only` | 0.5 | 0 | 1.0 | 0.00 | Isolates missed detections. |
| `delay_limited_only` | 1.0 | 2 | 1.0 | 0.00 | Isolates delayed intervention. |
| `capacity_limited_only` | 1.0 | 0 | 0.33 | 0.00 | Isolates limited targeting capacity. |
| `cost_limited_only` | 1.0 | 0 | 1.0 | 0.04 | Isolates intervention burden. |
| `limited_overseer` | 0.7 | 1 | 0.5 | 0.02 | Bundled moderate limitation used in the main benchmark. |
| `weak_overseer` | 0.5 | 2 | 0.33 | 0.04 | Bundled severe limitation used in the main benchmark. |

## Metrics

The ablation should report:

- global unsafe rate;
- local-pass/global-fail rate;
- mean patch health;
- garden failure rate;
- welfare;
- oversight burden;
- missed-target rate;
- delayed-intervention count.

The interpretation should separate ecological safety from cost. A limitation can increase missed targets or burden without immediately increasing global unsafe rate. That is still useful mechanism evidence.

## Reduced version

The reduced version is designed to answer whether the ablation mechanism works before launching a larger grid.

Cells:

- stress setting: `forest_co_management` only;
- actor capability: `high_actor`;
- oversight architectures: `top_down_only`, `hybrid`;
- overseer settings: all seven settings listed above;
- runs: `2`;
- generations: `8`;
- population size: `6`;
- train/test seeds per generation: `16/16`;
- replacement fraction: `0.2`.

This is 28 run jobs:

```text
1 stress setting × 1 actor level × 2 architectures × 7 overseer settings × 2 runs = 28 jobs
```

Recommended command:

```bash
python -m experiments.run_overseer_limit_ablation --execute
python -m experiments.analyze_overseer_limit_ablation
python -m experiments.audit_overseer_limit_ablation \
  --runs-csv results/runs/overseer_limit_ablation/reduced_ablation_runs.csv \
  --scenarios forest_co_management \
  --conditions top_down_only,hybrid \
  --actor-capability-levels high_actor \
  --overseer-levels strong_overseer,recall_limited_only,delay_limited_only,capacity_limited_only,cost_limited_only,limited_overseer,weak_overseer \
  --n-runs 2
```

Expected runtime:

- local laptop: roughly 10 to 30 minutes, depending on current load;
- GitHub Actions or remote CPU: usually safer for reproducibility, but not necessary for the reduced version.

Allowed claims if only the reduced version is run:

- "This reduced mechanism check suggests which isolated overseer limitation is most damaging in the high-coupling/high-actor slice."
- "This is not a full robustness result across the benchmark."
- "The result motivates a full overseer-limit ablation."

Do not claim from the reduced version:

- that one limitation is universally the main driver;
- that all stress settings behave the same;
- that the bundled overseer axis is fully explained.

## Full version

The full version should use the same paper settings as the main Stage A benchmark and vary isolated overseer limitations across the main actor/stress grid.

Cells:

- stress settings: `community_irrigation`, `forest_co_management`;
- actor capability: `low_actor`, `medium_actor`, `high_actor`;
- oversight architectures: `top_down_only`, `hybrid`;
- overseer settings: all seven settings listed above;
- runs: `5`;
- generations: `15`;
- population size: `6`;
- train/test seeds per generation: `32/32`;
- replacement fraction: `0.2`.

This is 420 run jobs:

```text
2 stress settings × 3 actor levels × 2 architectures × 7 overseer settings × 5 runs = 420 jobs
```

Recommended command:

```bash
python -m experiments.run_overseer_limit_ablation \
  --scenario community_irrigation,forest_co_management \
  --conditions top_down_only,hybrid \
  --actor-capability-level high_actor,medium_actor,low_actor \
  --n-runs 5 \
  --generations 15 \
  --seeds-per-generation 32 \
  --test-seeds-per-generation 32 \
  --output-prefix results/runs/overseer_limit_ablation/full_ablation \
  --summary-csv results/runs/showcase/curated/harvest_overseer_limit_ablation_full.csv \
  --execute

python -m experiments.analyze_overseer_limit_ablation \
  --summary-csv results/runs/showcase/curated/harvest_overseer_limit_ablation_full.csv \
  --output-prefix results/runs/showcase/curated/harvest_overseer_limit_ablation_full

python -m experiments.audit_overseer_limit_ablation \
  --runs-csv results/runs/overseer_limit_ablation/full_ablation_runs.csv \
  --scenarios community_irrigation,forest_co_management \
  --conditions top_down_only,hybrid \
  --actor-capability-levels low_actor,medium_actor,high_actor \
  --overseer-levels strong_overseer,recall_limited_only,delay_limited_only,capacity_limited_only,cost_limited_only,limited_overseer,weak_overseer \
  --n-runs 5
```

Expected runtime:

- local laptop single worker: likely several hours and not recommended as the first attempt;
- GitHub Actions or remote sharded execution: preferred if the full version is launched;
- recommended workflow: run one dry/smoke slice first, then shard by `scenario × actor_capability × overseer_setting`.

Allowed claims if the full version is run:

- "The overseer capability axis was decomposed into detection, delay, targeting capacity, and cost components."
- "Within this benchmark, the largest degradation comes from [observed limitation], conditional on the completed results."
- "The bundled weak overseer effect can be interpreted through its isolated components."

Do not claim even after the full version:

- that the limitation ranking generalizes outside the Harvest benchmark;
- that the ablation covers every possible overseer limitation;
- that capability imbalance is fully characterized by these four variables.

## Output paths

Reduced version:

- raw run CSV: `results/runs/overseer_limit_ablation/reduced_ablation_runs.csv`;
- summary CSV: `results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced.csv`;
- compact CSV: `results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced_compact.csv`;
- summary note: `results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced_summary.md`;
- paper-style table: `results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced_table.tex`;
- figure: `results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced.png/pdf/svg`;
- completeness audit: `results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced_audit.md`.

Full version:

- raw run CSV: `results/runs/overseer_limit_ablation/full_ablation_runs.csv`;
- summary CSV: `results/runs/showcase/curated/harvest_overseer_limit_ablation_full.csv`;
- compact CSV: `results/runs/showcase/curated/harvest_overseer_limit_ablation_full_compact.csv`;
- summary note: `results/runs/showcase/curated/harvest_overseer_limit_ablation_full_summary.md`;
- table: `results/runs/showcase/curated/harvest_overseer_limit_ablation_full_table.tex`;
- figure: `results/runs/showcase/curated/harvest_overseer_limit_ablation_full.png/pdf/svg`;
- completeness audit: `results/runs/showcase/curated/harvest_overseer_limit_ablation_full_audit.md`.

## Figure and table design

The main figure should be compact:

- x-axis: overseer limitation;
- two lines or grouped bars: global signal and hybrid;
- panels: unsafe rate, local-pass/global-fail rate, patch health, burden;
- use the table for welfare, missed-target rate, and delayed-intervention count if the figure becomes crowded.

The table should keep all metrics because this ablation is primarily explanatory. It should report global and hybrid side by side.

## Interpretation template

Use this structure:

1. Start from the strong overseer baseline.
2. Compare each isolated limitation against the baseline.
3. Identify whether the limitation affects safety, resource health, cost, or targeting metrics.
4. Compare the isolated limitations to bundled limited and weak overseer settings.
5. State whether the bundled result appears additive, dominated by one limitation, or qualitatively different.

The paper should only use this ablation after the completeness audit passes.
