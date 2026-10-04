# Coupled-local information check (24 September 2026)

## Question

Does the apparent advantage of joint review over bounded local review survive
when local patch calculations use *the same inspected requests* and known
neighbour dynamics? This is a diagnostic of the existing comparator, not a new
independent confirmation cohort. The 64 population contexts per game and their
reference labels were already analyzed on 23 September.

## Frozen comparison

Replay every original proposal in the completed
`budgeted_reviewer_confirmation_v1` artifact at inspection budgets 0, 3, and 6.
Keep public state, the exact nested inspected subset, upper bound 1 for missing
requests, one-step global safety target, weather margin, five uniform scaling
options, and five candidate checks unchanged. Compare the existing `joint`
decision with a `local_coupled` decision:

- Fishery: sum one bounded extraction contribution per user. A local maximum
  multiplied by population size is **not** used.
- Harvest: each patch report uses its own health and extraction plus the two
  neighbouring excess-extraction contributions. Reports are pooled for the
  same global predicate. The only request values entering these contributions
  are inspected values or the declared upper bound.

The model predicts exact decision equality for these deterministic one-step
calculations at all three budgets. Any disagreement is an implementation or
information-leakage failure, not a favorable effect. Log the number of paired
cases, disagreement count, reference-label counts, and logical scalar-report
counts. Check equality on hand-built vectors and on masked vectors before
interpreting the replay. Do not use a case-level confidence interval to turn
identical decisions into an equivalence claim about unseen settings.

## Interpretation and resource cap

This is **post-hoc exploratory**, because the new comparison was motivated by
the already-seen joint-versus-bounded-local gap. It cannot prove a deployable
decentralized monitor: patch reports are assumed truthful, the transition
model is known, and communication cost is an accounting convention rather
than a network implementation. It can show whether information aggregation,
rather than physical centralization, explains the observed difference.

Use one local CPU process, no paid/cloud resource, no new simulations or model
inference, at most one complete replay, a 180-second wall-time cap, and a
50 MB output cap. Preserve existing results untouched. Stop if source coverage
or paired identities fail. Save this protocol, output manifest, aggregate
summary, and the exact command. The next *new-data* experiment, if warranted,
requires its own protocol and fresh population contexts.
