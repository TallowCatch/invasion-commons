# Held-out Policy Oversight Pilot

23 September 2026. Prospective local development contract. The constructed
decision-case results are known; no outcomes from this new policy sample have
been inspected. This is not confirmatory evidence or a new capability ladder.

## Question and prediction

Do the decision and resource trade-offs seen on fixed challenge cases persist
when newly seeded policies act for complete episodes? Shared-request information
may reduce missed harmful proposals or avoid unnecessary restrictions relative
to local reports, but a strong local rule may match or outperform it. A failure
of the joint method to improve the trade-off is an admissible result.

The literature and rationale remain those in
`literature_20260921/OVERSIGHT_PROTOCOL_DRAFT.md`. No new novelty claim is made.

## Fixed design

- Two separately reported policy strata: cooperative/useful and aggressive/stress.
  Harvest uses the existing cooperative and adversarial structured-strategy
  generators. Fishery uses independent threshold draws, with low/high harvest
  fractions from [0.05, 0.20]/[0.20, 0.40] for useful and
  [0.45, 0.65]/[0.75, 0.95] for stress. Thresholds are uniform on [25, 65].
  These are design choices, not measured actor-capability levels.
- Four fresh population contexts per stratum, disjoint from the prior pilot's
  population seeds. Harvest has two renewal settings (base, 0.85 times base)
  and two weather streams nested per context. Fishery is deterministic.
- Each Harvest stratum/regime/context/weather evaluates `none`,
  `local_uncertain`, `local_conservative_uncertain`, and `joint_uncertain`.
  Fishery evaluates `none`, `local_nominal`, `local_conservative`, and
  `joint_nominal`. This is 128 Harvest and 32 Fishery episodes, maximum 80
  steps each. The old fixed cutoff is historical, not the primary local arm.
- All arms use the existing global safe set, five uniform action scales,
  five-candidate limit, same transition model and response authority.
  Harvest local reports are pooled by a public combiner; this is not fully
  independent local governance. Compare candidate and component checks and
  transmitted scalars as separate proxies, not equivalent monetary costs.
- On every no-intervention step, freeze the original state and proposed action.
  Judge each with all applicable methods. Fishery uses exact one-step labels.
  Harvest uses 128 independent weather draws, Wilson 95% uncertainty and a
  0.05 risk threshold. Unresolved labels remain visible. Initially safe and
  already unsafe states are reported separately. No scaled counterfactual
  requests are called naturally occurring.

## Outcomes and analysis

Primary descriptive contrast is joint versus the weather-aware local rule
within stratum and renewal setting: unsafe-step fraction, safety-onset count,
population return, mean resource health and realized extraction. The
conservative local rule is an additional strong missing-neighbour control.
Frozen decisions report risky approvals, safe rejections, unresolved share,
approval coverage and abstentions. Also report candidate/component work and
communication separately. For closed-loop contrasts, average nested weather
streams within a context, then compute paired context differences. Four
contexts are too few for a precise population inference; report raw context
values, mean differences and ranges, no p-values or confidence intervals.
Fishery and Harvest keep their native units and are never averaged together.

Check that each stratum has both useful safe and harmful risky original
proposals among initially safe no-intervention states. If a stratum lacks one
class, mark that decision rate unavailable; do not search for favorable seeds.
Unsafe steps in already damaged states are occupancy, not fresh onset. A
rejection is not automatically a safe executed action. The same numeric seed
must generate identical policy parameters and disturbance sequence across
monitor arms; verify this in tests and saved blocks.

## Execution and stop

One local worker, at most the 160 declared episodes, 80 steps each, 128
reference draws per Harvest frozen state, 900 wall and CPU seconds, 250 MB
artifacts, no cloud, training, package installation or new model. First run
unit tests. The software smoke uses **one** useful
context, one renewal setting, one weather stream and all four methods in each
game, maximum 12 steps, in a separate directory. It is a plumbing check, not
an extra sample. One engineering retry is permitted only for a code defect,
with the failed attempt retained and a new output directory after source
changes. Never select seeds or settings based on a preferred winner. Save
source hashes, policy parameters, raw traces, labels, summaries and a local
experiment record. Local MLflow is optional and must remain local.

Stop after this development analysis. A confirmatory run requires a new
precision target and sample-size rationale, held-out seeds, and separate
reviewer-resource manipulation. Cleanup has its own policy-admission gate.
