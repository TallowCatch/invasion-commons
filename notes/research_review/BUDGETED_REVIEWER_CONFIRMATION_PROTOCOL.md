# Budgeted Reviewer Confirmation: Frozen Local Protocol

23 September 2026. Written after the development pilot and before looking at
the fresh cohort below. This is a local prospective protocol, not a public
preregistration. The earlier pilot selected the two informative settings and
the intervention rules; its outcomes are not part of this analysis.

## Question and primary outcome

Can a reviewer of six interacting agents retain safe proposed activity while
avoiding approval of harmful joint requests when inspection is limited? The
primary outcome is the **two-dimensional immediate decision trade-off** among
original proposals from initially safe states: (1) fraction of resolved-risky
requests approved and (2) fraction of resolved-safe requests restricted. A
reviewer is not called better from either fraction alone. The unit of decision
is a joint request vector, not an individual agent action. The reference is
exact for deterministic Fishery and 128-draw one-step Monte Carlo with a
Wilson-interval 0.05 risk threshold for Harvest; unresolved references remain
outside both denominators and are reported.

The principal paired comparison at six inspected requests is `joint` versus
`local_bounded`, the conservative local rule. `local_optimistic` is a required
strong activity-retention comparator. Budgets 0 and 3 characterize how the
decision trade-off changes with information, not extra independent tests. The
prediction is that joint review blocks fewer safe original proposals than
bounded local review without approving more harmful ones. A null, reversed,
or mixed result is valid. In particular, no universal joint advantage over
the optimistic local rule is predicted.

## Controlled comparison

All three modes judge the same global one-step safety target, current public
state, five uniform action multipliers `[1,.75,.5,.25,0]`, candidate limit 5,
and zero-action fallback. All see the same nested 0/3/6 inspected subset in a
case. Uninspected requests use the known upper bound 1. Harvest modes use the
same weather margin, and no mode sees future weather or reference labels.
Local Harvest patch reports are pooled; this is not autonomous local control.
Resource accounting separately reports requests inspected, component checks,
communications, and wall time. No scalar capability gap is constructed.

Pilot-informed target settings, with no claim about other game populations:

| Game | Policy mixture | Resource setting | Independent contexts | Weather streams |
| --- | --- | --- | ---: | ---: |
| Fishery | `mix4` | deterministic stock | 64 | 1 |
| Harvest | `mix2` | slow renewal | 64 | 2, averaged within context |

Each context has one no-intervention reference trajectory and nine monitored
episodes. Total: 1,920 episodes (640 Fishery, 1,280 Harvest). Original
proposals are frozen from the no-intervention trajectory and evaluated under
all nine reviewer settings. Fresh population seed bases are 500,000,000 for
Fishery and 510,000,000 for Harvest; Harvest weather base is 520,000,000 and
reference base is 530,000,000. The horizon is 80. Contexts, not steps,
weather streams, agents or repeated cases, are the independent units.

## Analysis and decision

For each game and mode at each budget, report resolved safe/risky/unresolved
counts, harmful approvals, safe restrictions, and their separate rates. At
budget 6, report paired `joint - local_bounded` and `joint -
local_optimistic` differences in both rates. Ratios are case-weighted within
game; uncertainty resamples entire population contexts, retaining both
Harvest weather streams and all paired methods. Use 4,000 context-bootstrap
resamples with fixed analysis seed 20260923 and percentile 95% two-sided
intervals. No pooled Fishery/Harvest estimate, p-values, or claim of
equivalence. Report the number of independent contexts containing each
resolved class. The practical difference worth discussing is 5 percentage
points; it is not an automatic significance or success cutoff. If fewer than
20 contexts contain risky or safe proposals in a game, the corresponding
effect is descriptive only. Missing denominators yield unavailable rates.
This deliberately leaves the uncertainty of rare harmful proposals visible.

Closed-loop fixed-80-step unsafe fraction, onset, terminal failure, resource
health, total return and realized extraction are secondary. Terminal collapse
is counted unsafe for the remaining horizon and also reported separately.
These outcomes may rank monitors differently from immediate decision quality.
Do not select only favorable budgets or settings. The two game settings are
separate tests of transfer, not interchangeable replicates.

## Execution boundary

One local CPU worker, no paid/remote compute, no model training or LLM calls.
Run tests and a small smoke before the full fresh-seed matrix. At most one
smoke and one 1,920-episode run, each with 900 wall/CPU seconds and 250 MB
artifact cap. No outcome-triggered retries or parameter changes. A single
engineering repair may use a new directory with the failed attempt retained.
Stop after the planned analysis even if an effect looks promising. Preserve
source snapshots, raw blocks, labels, decisions, run manifest and checksums.
The paper is not updated automatically: first assess coverage, uncertainty,
baseline fairness and whether the new result answers the stated question.
