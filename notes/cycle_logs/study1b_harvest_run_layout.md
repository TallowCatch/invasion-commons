# study1b + harvest commons run layout

## purpose
This batch does two things:
1. Strengthen Study 1 on the fishery by testing whether governance ranking is stable across harder partner mixes and stronger invasion pressure.
2. Start Study 2 with a second commons substrate, renamed from orchard to harvest commons, to compare top-down, bottom-up, and hybrid governance.

## study1b first execution batch
This is the first non-smoke execution batch, not the final high-power paper run.

### matrix
- injectors: `mutation`, `adversarial_heuristic`, `search_mutation`
- benchmark packs: `easy_v1`, `medium_v1`
- partner mixes: `balanced`, `adversarial_heavy`
- adversarial pressure: `0.1`, `0.3`, `0.5`, `0.7`
- governance conditions: `none`, `monitoring`, `monitoring_sanctions`, `adaptive_quota`, `temporary_closure`
- runs per cell: `2`
- generations: `10`
- population size: `12`
- train seeds per generation: `12`
- test seeds per generation: `12`

### why this matrix
- `easy_v1` checks whether governance ordering is already visible when collapse is not saturated.
- `medium_v1` is the decision-critical regime from the earlier paper cycle.
- `balanced` tests nominal mixed populations.
- `adversarial_heavy` tests robustness under hostile starting populations.
- the three injectors test whether conclusions depend on how invaders are created.

## harvest commons first execution batch
### matrix
- conditions: `none`, `top_down_only`, `bottom_up_only`, `hybrid`
- runs per condition: `24`
- seed range: `0..23`

### goal
- establish whether the second substrate already separates governance families cleanly enough for a Study 2 setup note.
- check whether bottom-up credit sharing moves welfare and patch health without central caps.

## follow-up decision rule
After this batch:
1. If governance ranking is stable in `medium_v1` across injectors, move to a higher-power rerun only for the most decision-critical cells.
2. If `temporary_closure` dominates only by collapsing welfare, treat it as a stress-policy rather than a main governance winner.
3. If harvest commons shows weak separation, tune only the communication and credit-transfer regime before adding more complexity.

## executed in this session
The full first batch above was too wide for an interactive run, so the completed execution slice was narrowed to:

- `Study 1b`
  - benchmark pack: `medium_v1`
  - partner mixes: `balanced`, `adversarial_heavy`
  - pressure levels: `0.3`, `0.5`
  - injectors: `mutation`, `adversarial_heuristic`, `search_mutation`
  - governance conditions: all five (`none`, `monitoring`, `monitoring_sanctions`, `adaptive_quota`, `temporary_closure`)
  - `n_runs=1`, `generations=4`, `population_size=10`, `seeds_per_generation=4`, `test_seeds_per_generation=4`

- `Harvest commons`
  - conditions: `none`, `top_down_only`, `bottom_up_only`, `hybrid`
  - `n_runs=24`

Artifacts from the completed slice:
- `results/runs/ablation/study1b_medium_slice_*`
- `results/runs/harvest/harvest_commons_batch1_*`
- `results/runs/showcase/study1b_harvest_batch1_readout.md`
