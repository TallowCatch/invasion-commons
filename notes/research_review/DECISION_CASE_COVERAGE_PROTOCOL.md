# Decision-Case Coverage Repair

22 September 2026. Prospective development protocol, written after inspecting
the preceding pilot, before generating or comparing this case suite. This is
not a confirmatory study or a change to the manuscript's results.

## Question and Reason

Can the matched oversight task distinguish risky approvals from unnecessary
rejections on an explicit range of safe, boundary and already-damaged states?
The previous Fishery sample contained no risky proposals. More repetitions of
that sample would not fix the measurement problem.

Expected mechanism: joint requests should permit less conservative decisions
than worst-case neighbour bounds, while ignoring weather can miss risk near a
boundary. Strong nominal and weather-aware local controls remain included.
No joint-method win is required. Analytic safety rules and benchmark grids are
our design choices, not newly discovered safety mechanisms.

The existing literature rationale is in
`literature_20260921/LITERATURE_AND_GAME_DECISION.md` and
`literature_20260921/OVERSIGHT_PROTOCOL_DRAFT.md`. No fresh novelty claim or broad
literature search is needed for this measurement repair. The native simulator
code determines transition semantics; sources do not justify the numerical
grid or reference-draw count below.

## Two Separately Reported Sources

**Recorded proposals:** reuse the completed matched pilot's no-intervention
blocks at steps 0, 5, 15, 30, 50 and 70 where present, using original requests
only. Do not regenerate policies or trajectories, and do not include the old
scaled-request variants as naturally occurring decisions. Preserve occurrences
and their block/context/weather/step identities. Identical numerical cases may
occur more than once; report content fingerprints, not extra independent trials.
These are descriptive development data already seen, not a held-out sample.

**Structural challenges:** construct a fixed grid before monitor evaluation.
These are admissible simulator states, not claimed to occur with any measured
probability or to be reachable under the sampled policies.

- Fishery stock: 5, 8, 10, 12, 20, 35, 70; below-threshold history zero,
  collapse flag false. Mean requested fractions: 0, .1, .25, .5, .75, 1.
  Allocations: uniform, concentrated into successive agents, reverse concentrated.
  Each allocation has the same total demand at a given mean.
- Harvest: existing base and 0.85-regrowth configurations. Health parameters:
  8, 10, 11, 14, 18. Profiles: uniform, alternating +/-2, clustered +/-2, and
  half-depleted (first three zero, last three twice the parameter, clipped at
  native maximum). Record actual mean and failed-patch share, not merely the
  nominal health parameter. Same demand means/allocation shapes as Fishery.
- Deduplicate identical structural state/proposal pairs within game/regime,
  keeping design aliases. Retain every remaining case regardless of label or
  monitor performance. Never fill missing label quotas by searching for cases
  where a preferred method wins.

This is balanced *design coverage*, not a promised equal safe/risky class mix.
Observed label counts and any missing cells are primary outputs. Recovery
states and initially safe states are reported separately throughout.

## Frozen Labels and Common Comparison

Save cases and source/config/code hashes before reference labeling; save labels
and their checksum before evaluating monitors. Harvest reference: 128 fresh
one-step weather draws, risk tolerance .05, Wilson 95% interval; unresolved
intervals remain unresolved. Fishery uses the exact native deterministic step.
No label or evaluator seed is passed to a monitor.

Validate Harvest's prediction against a native one-step replay from every case
state at an independent recorded seed. The only simulator change is an optional
initial-patch vector for replay; default episode behavior and transition physics
must remain identical. Verify actual extraction as well as resource state.

Use the existing eight Harvest and four Fishery methods. Same target, scale
menu, policies/proposals and default checking budget; no actor-capability ranks
or new monitor variants. A baseline applying a fixed cutoff remains a secondary
rule comparator, not the primary evidence-access contrast.

For each decision record original risk label, approval/rejection/abstention,
selected scale, infeasibility, checking and communication quantities, and
post-intervention reference risk. Reuse the reference disturbance stream across
scales for paired counterfactual estimates; monitors never see it. Record actual
one-step extraction retained, with zero-original-extraction ratios unavailable.
This measures immediate activity, not long-run welfare or beneficial activity
unless the proposal's reference label is safe.

## Analysis and Gates

Report counts and denominators by game, regime, source and initial safety.
Do not pool natural and designed cases into a headline accuracy, or pool games'
resource units. Distinguish risky approvals, safe rejections, unresolved share,
retained extraction on safe cases, and risky/unresolved executed actions.
Report checking work, requested actions inspected, and transmitted scalars.
These are proxies, not measured intelligence or hardware-normalized costs.

Synthetic-grid proportions are finite-suite descriptions, not estimates of
deployment prevalence. No binomial or timestep confidence intervals for method
success. Reference Wilson intervals describe stochastic outcome estimation
within a case, not between-case generalization. Natural contexts remain nested;
do not inflate replication using related snapshots.

Coverage gate: each game's/regime's initially safe structural set must include
at least 10 resolved-safe and 10 resolved-risky cases to support a development
confusion table. This is a diagnostic minimum, not a power calculation. All
recovery and unresolved cases remain reported. Missing coverage yields an
incomplete gate, not an automatic extra run. A difference of 5 percentage points
in risky approvals or safe rejections is a practical *follow-up signal* for this
suite, not statistical significance or a publishable population claim.

Engineering gates: deterministic case IDs; verified source blocks; matching
cases/labels across methods; native parity; immutable completed artifacts;
all planned judgements present; no leakage to monitor inputs. Failure invalidates
affected comparisons and stops execution until repaired.

## Budget and Stop

At most 1,600 cases and 12,800 monitor judgments, one CPU process, no training,
cloud, model calls or package installation. One fixed suite run, capped at 900
CPU seconds and 900 wall seconds; generated artifacts capped at 250 MB. Up to
one engineering retry with unchanged design is allowed; preserve any failed
attempt. Unit tests use small fixtures, not an extra scientific sweep.

Save manifests, frozen cases/labels, decision table, coverage table, plots and
`experiment.md`; log the completed run to existing local MLflow only. A logging
failure never triggers a scientific rerun. Preserve all previous pilot outputs.

Stop after coverage analysis and review. Cleanup policy/monitor validation and
resource-limited judgments follow as separately specified stages. This task
does not silently start a three-game sweep or rewrite the paper around a pilot.
