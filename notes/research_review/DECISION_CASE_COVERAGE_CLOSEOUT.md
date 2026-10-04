# Decision-Case Coverage: Completed Development Check

23 September 2026. This follows the prospective
`DECISION_CASE_COVERAGE_PROTOCOL.md`. The manuscript was not changed.

## The Question

The previous Fishery pilot had no harmful proposals, so a monitor's ability to
catch harm could not be measured. This check asks whether a fixed set of
proposed joint actions includes both safe and risky choices, and what each
monitor approves or blocks when its safety target and action menu are held
fixed. For the designed cases, the reported quantities are finite-suite
fractions, not estimates of how often such cases occur during deployment.

## Execution and Coverage

The completed, single-process run used saved no-intervention states and
original proposals from the previous pilot, plus a fixed grid of stock/patch
states and demand allocations. It generated **934 cases**, of which **360 were
recorded proposals** and **574 were constructed challenges**, and **6,984 matched
monitor decisions**. Repeated case content used the same reference draws;
there were 878 unique case contents. The full run took **7.01 wall seconds / 4.76
CPU seconds** before tracking overhead, inside the declared 900-second cap.

On initially safe *constructed* states, all three coverage checks passed:

| Setting | Safe proposals | Risky proposals | Unresolved |
| --- | ---: | ---: | ---: |
| Fishery, shared stock | 42 | 28 | 0 |
| Harvest, base renewal | 122 | 44 | 2 |
| Harvest, slower renewal | 117 | 49 | 2 |

The Fishery proposals recorded from actual episodes remained 24 safe, zero
risky. The constructed Fishery grid fixes that **measurement gap** without
establishing that harmful actions are common under the sampled policies. The
recorded slow-renewal Harvest cases were likewise all safe when starting from
an initially safe state. Already damaged states are in separate rows of the
coverage table, with recovery/persistence rather than fresh failure as their
question.

## What the Monitors Did

Every method judged the same cases. A risky approval is an original proposal
that the monitor accepted unmodified and that the one-step reference labeled
risky. A safe rejection is an original safe proposal that the monitor reduced.

| Initially safe constructed cases | Risky approvals | Safe rejections |
| --- | ---: | ---: |
| Fishery, joint exact-accounting check | 0/28 | 0/42 |
| Fishery, conservative local check | 0/28 | 14/42 |
| Fishery, optimistic local check | 13/28 | 0/42 |
| Harvest base, joint with weather margin | 0/44 | 11/122 |
| Harvest base, local with weather margin | 0/44 | 1/122 |
| Harvest base, conservative local with weather margin | 0/44 | 38/122 |
| Harvest slower renewal, joint with weather margin | 0/49 | 12/117 |
| Harvest slower renewal, local with weather margin | 1/49 | 4/117 |
| Harvest slower renewal, conservative local with weather margin | 0/49 | 41/117 |

The Fishery result isolates an interpretable case where sharing total demand
allows the exact one-step check to avoid an equal-share bound's unnecessary
restriction. On these Harvest cases, shared requests were **not consistently
better** than a strong local check with a weather margin. The conservative
local rule caught risky requests but often restricted safe ones. These are
mechanism diagnostics on a selected grid, not a general comparison of
decentralized and global governance.

Post-intervention risk, retained one-step extraction and checking/communication
proxies are in `analysis/method_summary.csv` and `analysis/decisions.csv`. A
rejection does not imply a safe resulting action. For example, the unmodeled
weather and infeasible states remain explicit in the records. No long-run
welfare comparison was made in this check.

## Verification and Limits

- All **197 repository tests passed**; ten optional native Cleanup tests were
  skipped in the ordinary environment. Earlier isolated Cleanup native tests
  passed separately and were not rerun for this check.
- Each source block's embedded identity is checked against its filename and
  completed-run checksums. Harvest reference dynamics were checked against the
  native one-step transition and actual extraction from every unique case and
  selected intervention scale. Completed cases, labels, decisions, source
  snapshots and analysis deliverables have checksums. The artifact verifier
  passed on 934 cases and 6,984 decisions.
- Harvest risk labels use 128 weather draws and pointwise Wilson intervals;
  near-threshold cases remain unresolved. The constructed grid is not a
  probability distribution or a held-out policy sample. Case counts, timesteps
  and interventions are not independent population replications.
- The local reports in Harvest are pooled into one global predicate. They are
  not autonomous local institutions. The monitor sees the simulator's known
  transition form. Candidate evaluations, inspections and transmitted scalars
  are accounting proxies, not measured human labor or hardware-normalized cost.
- A one-step decision test cannot establish that this oversight protects a
  long-running commons. It also does not validate an actor-overseer capability
  ladder, or show that model-generated strategies add a distinct challenge.

Saved result: `results/runs/oversight_decision_coverage_v1/`. Read
`analysis/coverage.csv`, `analysis/method_summary.csv`, the matching figure
(`analysis/decision_coverage.pdf`), and the generated `analysis/experiment.md`.
The original archived pilot remained unchanged. Local MLflow experiment
`commons-decision-coverage`, run `b9f3837d29044fb991e63911dc75c069`,
finished with 934 cases and 6,984 decisions.

## Next Research Gate

**Do not present this as the final paper experiment.** The next comparison
should use *held-out policy populations and states* chosen independently of
these fixed challenge cases. Before a larger run, specify a proposal generator
that produces damaging and useful behavior in both Fishery and Harvest, then
test whether the monitor trade-off persists over complete episodes with the
same target, authority, actor policies and matched reviewer resources. Compare
the strong local weather-aware rule directly with the joint weather-aware rule;
do not use the optimistic local rule as the only local baseline. Treat the
multi-axis actor/reviewer resource settings separately rather than subtracting
rank labels.

Cleanup is a distinct admission gate. Its native transition/replay works, but
the only productive controller evaluated so far sees the full simulator state.
First validate observation-limited cleaning and harvesting policies, plus
damaging proposals, before constructing comparable Cleanup monitors. Action
suppression cannot supply missing cleaning, so the intervention question will
need to say what monitor action can actually restore maintenance. A further
game or LLM model would not repair these present coverage and policy gaps.

For Yali: "I changed the benchmark so monitors judge the same full proposed
action and can make the same intervention. The new case suite verifies that
we can measure both missed harmful decisions and unnecessary restrictions.
Fishery gives a clean shared-information example. Harvest shows that a good
local rule can sometimes be as effective, which is why the next test must use
held-out populations and explicit reviewer budgets rather than claim a general
winner from constructed examples."
