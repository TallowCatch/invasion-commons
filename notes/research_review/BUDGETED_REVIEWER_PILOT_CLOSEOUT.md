# Budgeted Reviewer Pilot: Development Closeout

23 September 2026. The predeclared local CPU pilot in
`BUDGETED_REVIEWER_PILOT_PROTOCOL.md` completed. This is a development result,
not a new main-paper claim or confirmatory study. The paper was not changed.

## Execution and validity

The smoke completed 14 episodes. The full matrix completed **400 episodes**,
**1,716 original no-intervention proposals**, and **15,444 matched reviewer
decisions** in 23.76 wall / 23.62 CPU seconds. Raw blocks, source snapshots,
frozen cases and labels, decision records and analysis deliverables have
checksums; the completed-artifact verifier passed. There are 1,700 distinct
physical cases and 16 repeated occurrences with zero reference-label
conflicts. Inspection subsets are nested for `k=0,3,6`, fixed by case identity
and shared across methods. Unit tests verify that changing a hidden request
cannot alter the monitor's decision. Full ordinary test suite: **223 passed,
11 skipped** (optional native Cleanup tests require its isolated runtime).
An independent post-run audit checked every one of the 15,444 frozen decisions
and 27,958 monitored episode steps against the declared deterministic
inspection schedule: zero subset/count mismatches and zero cross-method
disagreements on the inspected subset for a shared case and budget.
No paid/cloud compute, LLM inference or policy training was used. Tracking is
the explicit local-file manifest and result tables; no MLflow run is claimed.

The reference corpus has these safe/risky/unresolved *original* proposals from
initially safe states:

| Game / mixture / renewal | Safe | Risky | Unresolved |
| --- | ---: | ---: | ---: |
| Fishery / mix2 | 261 | 1 | 0 |
| Fishery / mix4 | 36 | 4 | 0 |
| Harvest / mix2 / base | 317 | 10 | 7 |
| Harvest / mix2 / slower | 55 | 99 | 41 |
| Harvest / mix4 / base | 0 | 9 | 4 |
| Harvest / mix4 / slower | 0 | 10 | 0 |

The `mix4` Harvest cells cannot estimate safe-rejection rates. Fishery harmful
sample sizes are small. Repeated timesteps and weather streams do not create
additional independent policy contexts; there are four per mixture.

## Decision trade-off

Every method uses the same one-step resource-safety target, transition form,
five-scale intervention menu and five-candidate cap. The varied resource is
the number of **current requests inspected**, not model intelligence.
Uninspected requests have an explicit upper bound of 1. The executor retains
the full proposed vector only to apply the chosen uniform multiplier.

In Fishery `mix4`, joint review approved **0/4** risky requests at each budget.
Its safe rejections fell from **14/36** at `k=0`, to **8/36** at `k=3`, to
**0/36** at `k=6`. Bounded local review also approved **0/4** risky requests;
it rejected **14/36**, **14/36**, and **7/36** safe requests. Optimistic local
review missed 3/4, 3/4 and 4/4 risky requests respectively. Shared demand
information therefore improved this immediate Fishery decision trade-off on
the sampled cases. It did not add a new authority or safety target.

In Harvest `mix2` base renewal, every method rejected all 317 resolved-safe
requests at `k=0` and `k=3`. At `k=6`, joint rejected **163/317**, bounded
local **317/317**, and optimistic weather-aware local **129/317**. Each approved
**0/10** risky requests. Under slower renewal, all 55 resolved-safe requests
were rejected at `k=0` and `k=3`; at `k=6`, joint rejected **51/55**, bounded
local **55/55**, and optimistic local **47/55**, while all approved **0/99**
risky requests. The 7 and 41 unresolved references remain separate. This
shows that conservative handling of missing requests can protect the one-step
target while suppressing useful activity. It does **not** show joint review
outperforming the strongest local comparator in Harvest.

## Complete episodes and mechanism limit

The main episode result is conditional. In Fishery `mix4`, joint and bounded
local review had no unsafe steps. Yet at `k=0`, they earned mean 80-step return
1,109.36; at `k=6`, joint earned 527.44 and bounded local 817.76. With less
information they intervened more conservatively, leaving more stock for
subsequent harvest. This illustrates a real distinction between the monitor's
**one-step objective** (retain the largest immediately safe request) and the
agents' **long-run return**. Better one-step discrimination need not improve
the long-run outcome under this policy and renewal model.

In Harvest `mix2`, joint and both local modes had zero unsafe closed-loop
steps across the three budgets in both renewal settings. This is a safety
floor, not evidence of equivalence. No-intervention unsafe fractions over the
fixed 80-step horizon were 0.484 (base) and 0.705 (slower). For early terminal
failure, the fixed-horizon measure counts remaining steps as unsafe under an
explicit absorbing-failure convention; observed-step rate, onset, terminal
event, episode length, return and extraction are also saved separately.

## Stop and next paper gate

Stop this development run. The cost/information diagnostic works, but the
comparison is not yet publication-grade: Fishery has four harmful cases in
its most informative mixture, Harvest `mix4` lacks safe cases, four population
contexts give poor precision, and the known-transition/weather-bound model is
much simpler than a weaker learned overseer. The checker knows public patch
health; local Harvest predictions are pooled by a global combiner. Inspections,
component evaluations, transmissions and wall time are separate proxies; no
scalar actor-minus-overseer capability score is validated.

Before a main-paper inference, specify whether the primary question is
**decision quality at fixed immediate risk** or **long-run resource/welfare
control**. Both can be reported, but a method can rank differently. Then use
new population contexts, a precision target and credible strong-local control
to confirm a predeclared contrast. A larger seed sweep on the current pilot
without that choice would not fix the conceptual gap. Cleanup is not admitted
for a cross-game comparison; see its separate closeout. LLM bank expansion and
additional game families remain unnecessary for this stage.

Artifacts: `results/runs/budgeted_reviewer_v1_smoke/` and
`results/runs/budgeted_reviewer_v1_pilot/`. The pilot's `analysis/` directory
contains episode, context, outcome, coverage, decision-quality and paired
contrast CSVs plus `experiment.md`.
