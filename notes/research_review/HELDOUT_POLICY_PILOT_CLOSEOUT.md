# Held-out Policy Oversight: Development Closeout

23 September 2026. This follows the decision-case coverage check. **No paper
claim was changed.** These are local development runs, not a confirmatory
multi-game result. The original `useful`/`stress` run and the first mixed run
remain untouched. The mixed retry corrected a duplicate-case reference-seed
defect without changing policies, episodes, monitor decisions or outcomes.

## What was run

The original held-out run sampled new cooperative and aggressive threshold
policies in Fishery and structured policies in Harvest: 160 complete episodes,
1,741 original no-intervention proposals, and 6,964 matched monitor decisions.
This exposed a sampling problem: initially safe Harvest states under the
cooperative population had 1,280 safe and zero risky proposals across both
renewal settings; the aggressive population had zero safe and 16 risky. Both
groups were retained and reported. These cells cannot jointly assess missed
harm and unnecessary restriction within one policy population.

A new, explicitly exploratory repair used two fixed mixtures: `mix2` (two
aggressive agents) and `mix4` (four aggressive agents), with fresh population
seeds. It completed 160 episodes, 1,762 original proposals, and 7,048 matched
monitor decisions. The first attempt found one identical Harvest physical
case with a safe/unresolved label disagreement because reference seeds were
based on trace IDs. The allowed engineering retry keyed draws to case content.
The corrected run has 1,746 distinct physical cases, 16 repeated occurrences
and **zero conflicting labels**. Closed-loop episode outcomes were byte-identical
across the two mixed attempts. The corrected record is `mixed_policy_repair_v1b`.

## Coverage of original proposals

Counts below start from globally safe states, use original policy proposals,
and keep unresolved Monte Carlo references visible. Repeated weather-stream
occurrences are not independent policy contexts.

| Game / population / renewal | Safe | Risky | Unresolved |
| --- | ---: | ---: | ---: |
| Fishery / mix2 | 320 | 0 | 0 |
| Fishery / mix4 | 27 | 4 | 0 |
| Harvest / mix2 / base | 439 | 80 | 60 |
| Harvest / mix2 / slower | 283 | 25 | 31 |
| Harvest / mix4 / base | 2 | 8 | 4 |
| Harvest / mix4 / slower | 0 | 10 | 0 |

The minimum two-class gate passed in Fishery `mix4` and Harvest `mix2` in
both renewal settings. It failed in Fishery `mix2` and Harvest `mix4` under
slower renewal. Harvest `mix4` base has only two safe observations, so its
safe-rejection fraction is especially unstable. These failures remain in the
tables; no seed or mixture was selected after seeing a monitor win.

## What the monitors did

On initially safe states in Fishery `mix4`, joint monitoring approved **0/4**
risky proposals and rejected **0/27** safe proposals. Conservative local
monitoring also approved **0/4** risky proposals but rejected **6/27** safe
ones. Optimistic local monitoring approved **4/4** risky proposals. This is
consistent with shared total demand avoiding an equal-share bound's needless
restriction on these cases, but four harmful cases from four contexts are a
thin basis for generalization.

In Harvest `mix2` base renewal, joint and weather-aware local monitoring each
approved **0/80** resolved-risky proposals; joint rejected **145/439**
resolved-safe requests, versus **121/439** for weather-aware local. Under slower
renewal both approved **0/25** risky requests; joint rejected **132/283** safe
requests, versus **122/283** for weather-aware local. The conservative local
variant rejected **415/439** and **283/283** safe requests, respectively.
The reference's 60 and 31 unresolved cases are not counted as safe or risky.
Thus these data do **not** show a joint-information advantage over the strongest
local Harvest rule on decision quality.

Closed-loop results give a different, longer-horizon view. In Harvest `mix2`,
both joint and weather-aware local monitoring had zero unsafe steps in both
renewal settings and near-equal population return (base: 737.77 joint, 741.16
local; slower: 641.21 joint, 640.60 local). No intervention had unsafe-step
fractions 0.097 and 0.477. In Fishery `mix4`, joint and conservative local both
avoided unsafe steps, but conservative local retained more stock and earned
more total return over the episode (699.42 versus 516.34). A one-step-safe
approval can still reduce later stock and return. These are paired-context
descriptions, not claims that one method is generally optimal.

Some no-intervention and optimistic-local episodes ended early. The exported
`global_unsafe_rate` is over **observed** steps; it is not a fixed-80-step
occupancy measure. Treating terminal collapse as absorbing unsafe status for
the remaining planned steps gives, as a post hoc diagnostic, Fishery `mix4`
unsafe fractions 0.916 (none) and 0.913 (optimistic local), and Harvest `mix4`
0.991 (base none) and 0.997 (slower none). The corresponding observed-step
fractions were 0.428, 0.431, 0.897 and 0.964. The absorbing convention is a
reporting assumption, not a simulated recovery trajectory. Onset count,
terminal failure and episode length should accompany any future occupancy plot.

## Validity and next decision

- Engineering: both predeclared matrices completed; all monitor arms had
  identical policy parameters and game configuration within each paired cell;
  the corrected mixed run's artifact verifier passed. Full repository tests:
  204 passed, ten optional native Cleanup tests skipped in the ordinary
  environment. No paid or remote compute, LLM calls or training occurred.
- Statistical: four independent policy contexts per population are too few
  for a publication-level generalization. Weather streams, timesteps and
  repeated physical cases are nested. The one-step labels use finite Monte
  Carlo draws and leave boundary cases unresolved.
- Scope: these policies vary behavioral mixture, not measured actor ability.
  Candidate limits bound repair search, not reviewer forecast competence.
  Harvest local reports are pooled by a global combiner; no independent local
  institution is modeled. The same safety target and intervention menu make
  the evidence-access comparison interpretable, but not a full scalable-
  oversight validation.

**Stop this development phase.** The next main experiment should manipulate
reviewer information and actual checking budget separately on new, held-out
contexts. Its design must include enough safe and harmful original proposals,
match intervention authority and policy source, report decision errors plus
resource/return and cost, and choose replication from a precision target. The
`mix2`/`mix4` data can inform that design but cannot serve as untouched
confirmation. Cleanup remains behind its separate admission gate: the current
productive controller sees the full hidden simulator state. It needs a useful
observation-limited cleaner and a damaging free-rider policy before any
cross-game monitor comparison. No fifth game or additional LLM is needed to
repair these gaps.

Artifacts: `results/runs/heldout_policy_v1_pilot/`,
`results/runs/mixed_policy_repair_v1/` (first attempt), and
`results/runs/mixed_policy_repair_v1b/` (corrected source of reported mixed
numbers). The `v1b/analysis/` folder has per-episode, context, contrast,
coverage and decision CSVs plus `experiment.md`. The first mixed attempt's
generated `experiment.md` has an incorrect protocol filename; its manifest
and source snapshots identify the correct design. The corrected record fixes
that report-template error.
