# Budgeted Reviewer Development Pilot

23 September 2026. Prospective local contract, before any outcomes from the
new seed cohort. This is the next stage of
`literature_20260921/OVERSIGHT_PROTOCOL_DRAFT.md`, informed by the completed
held-out policy pilot. It is not a confirmatory test or a proof that stronger
models can oversee one another.

## Question and contrast

When six agents share a resource, how do three information rules trade missed
harm against blocked useful activity as the reviewer inspects 0, 3 or 6
current requests? `local_optimistic` uses own-request patch reports and omits
neighbour extraction. `local_bounded` uses own-request patch reports but bounds
unseen neighbour extraction. `joint` aggregates inspected requests and bounds
uninspected requests. The same subset is inspected by all modes in a cell.
All methods know the simulator transition form and current stock/patch health;
Harvest local reports are pooled into one global check. The experiment varies
request information/aggregation, not political authority or model intelligence.

Prediction: more request access can reduce unnecessary restriction for joint
review while retaining safety. A strong local rule may match or outperform it;
the optimistic rule may miss spillovers. Either result is acceptable. A flat
result means this policy/context sample did not expose information value.

## Fixed mechanics and population

- The shared target is the existing Fishery stock-safe or Harvest global
  mean-health/failure-share predicate. All methods apply the same five
  uniform request multipliers `[1, .75, .5, .25, 0]`, test at most five
  candidates and use the same zero-action fallback. The objective is to retain
  the largest requested activity among predicted-safe menu actions. A method
  may be infeasible; rejection is not a safety guarantee.
- Public state is current Fishery stock or Harvest patch-health vector.
  Uninspected requests are replaced with the known upper bound 1 for safety
  prediction. Local optimistic explicitly assumes no neighbour damage; it is
  a non-conservative comparator. Harvest methods share the same Gaussian
  weather margin; no method sees the evaluator's future disturbance or risk
  label. The executor alone retains the full proposed action to scale it.
- Inspection subsets are a deterministic permutation keyed by game, policy
  mixture, regime, context, weather and time. The first `k` slots are visible;
  thus 0/3/6 subsets are nested and identical across monitor modes. The
  monitor interface accepts a masked request vector and must be invariant to
  any change to hidden requests.
- Populations use the fixed `mix2` and `mix4` policy generators from the
  mixed-policy repair, but **new seeds**: population bases 300,000,000 and
  310,000,000; weather base 320,000,000; reference base 330,000,000. These
  mixtures test behavioral pressure, not a validated capability ordering.
  Four independent policy contexts per mixture, two Harvest renewal settings,
  two nested weather streams, 80-step horizon. No policy search or tuning.
- One no-intervention trajectory per context/weather cell supplies original
  frozen proposals. Evaluate each with all nine mode/inspection combinations
  before interpreting decisions. Exact Fishery reference and 128-draw Harvest
  reference with 0.05 risk threshold and Wilson95 intervals; unresolved
  labels remain unresolved. Duplicate physical cases share reference draws.

Full matrix: Harvest 2 mixtures x 2 renewal settings x 4 contexts x 2 weather
streams x (1 no-intervention + 9 monitors) = **320 episodes**. Fishery 2 mixes
x 4 contexts x (1 + 9) = **80 episodes**. Smoke: one mix2 context, one base
Harvest weather stream, `k=0,6` and three modes plus no intervention in each
game, at most 12 steps: 14 episodes. No replicated no-intervention arm is
counted as a fresh sample.

## Outcomes and decision rule

Primary frozen-decision quantities by game/mixture/regime/k/mode and initial
safety: risky original proposals approved, safe originals rejected, unresolved
share, approval and abstention coverage. Closed-loop outcomes: unsafe onset,
terminal failure, observed-step unsafe fraction, fixed-horizon absorbing-failure
fraction (explicit convention), resource health, population return and actual
extraction. Checking resources: inspected current requests, candidate/model
component evaluations, transmitted scalars and wall time. The latter is
machine-dependent. Count total reviewer work across all local reports.

Pair episode arms by independent population context; average weather streams
within context. Report raw context differences and ranges, not p-values or
population confidence intervals with only four contexts. Show all k levels,
including missing safe/risky class coverage; do not select a winning cell.
Fishery and Harvest retain native units. A one-step-safe approval can still
harm later stock; decision and episode metrics answer different questions.
No scalar actor-minus-reviewer rank is constructed.

## Resource cap and stop

One local CPU worker, one smoke plus one 400-episode development run, maximum
900 wall/CPU seconds per run and 250 MB of artifacts. No cloud, package
installation, training or LLM calls. Run invariant/unit tests first. Save
frozen cases/labels, matched decisions, raw traces, per-context summaries,
source/config hashes and an experiment record. Use explicit local-file
tracking; missing MLflow logging must never trigger a scientific rerun. One
engineering retry is allowed only for a code defect, in a new directory, with
the failed attempt preserved. Stop after checking measurement coverage and
the information-cost trade-off. A confirmatory study needs a separate
precision/sample-size contract and fresh population contexts.
