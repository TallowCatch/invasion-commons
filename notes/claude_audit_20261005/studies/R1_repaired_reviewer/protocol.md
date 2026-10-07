# Protocol R1: the reviewer experiment, repaired

**Frozen:** 2026-10-04T23:57Z (00:57 London), before any R1 outcome was
generated.

**Status:** written down locally before the run; not a public preregistration.

**Rule:** this file is not edited after results exist. Changes go under
"Amendments" at the end, with a reason.

**Already known when this was written:** the outcomes of the 23 September
confirmation (`budgeted_reviewer_confirmation_v1`) and the post hoc
re-analysis in the September reviewer write-up. R1 is motivated by those, so it is a **repair and
replication with new seeds**, not an independent discovery.

## 1. Question

With the measurement problems from the design-issues note fixed, how do the three reviewers
compare on two things?

- **Unsafe approvals:** letting through a joint request whose one-step risk
  is above 5%.
- **Usefulness loss:** cutting a joint request that was safe.

And does changing the safety *target* in Fishery remove the long-run
reversal, where better information led to less total harvest?

Literature terms, from `literature_ledger.md`:

- We call the per-step rate of approving a risky joint request the
  **unsafe approval rate**. In the AI-control literature (Greenblatt et al.
  2023/24) this kind of failure is what "safety" measures, although their
  estimand is defined over a whole deployment.
- Cutting a safe request is **usefulness loss**, which matches the
  shielding literature's "minimal interference" goal (Elsayed-Aly et al. 2021).
- The weather is **environmental stochasticity** (May et al. 1978).
- The reviewer's 5% rule is a **chance constraint** (Charnes & Cooper 1959).
- The sustainability target is the logistic **MSY level**: stock K/2 left
  after harvest (Schaefer 1954, secondary source).

## 2. What changes from the 23 September design

| Weak point (the design-issues note) | R1 change |
| --- | --- |
| A1: oversized Harvest weather allowance | **All reviewers use a chance constraint.** Each one approves the largest scale whose *estimated* one-step risk is ≤ 5%. The estimate comes from 400 simulated weather draws applied to that reviewer's own prediction. The rule covers both safety conditions; there is no fixed buffer. |
| B3: unseen requests assumed to be the maximum | **Fill is a factor.** Either `max` (the old rule, 1.0) or `previous` (that agent's request from the previous step, which is observed after the fact; 1.0 at step 0). |
| B4: borderline cases dropped | **More reference draws.** Labels use 4,000 draws for the open-loop cases and 2,000 for the closed-loop cases (Wilson 95% interval at the 5% threshold). Unresolved counts are reported. |
| A5: only open-loop scoring | **Closed-loop scoring added.** Every step a reviewer actually faces in closed loop is also labelled and scored. |
| A2: Fishery safety line far below the productive level | **Fishery target is a factor.** Either `one_step` (next stock ≥ 10, the old line) or `msy` (at least 50 left after harvest, i.e. K/2). The reference label uses the same target as the reviewer. |
| A4: bounded-local reviewer description | Unchanged on purpose, for comparability. As before, it bounds all other agents. The coupled local reviewer is not rerun: it is algebraically identical to joint (the September reviewer write-up). |

Kept from the original design:

- the games and policy generators;
- Fishery `mix4` with deterministic stock;
- Harvest `mix2` in forest co-management with slow renewal (×0.85);
- the action menu {1, 0.75, 0.5, 0.25, 0};
- the nested random inspection order;
- inspection budgets k ∈ {0, 3, 6};
- the 80-step horizon.

## 3. Conditions and sample

**Independent unit:** the population context. There are 64 fresh contexts
per game, with new seed bases:

| Game | Population | Weather | Reviewer draws | Reference draws |
| --- | --- | --- | --- | --- |
| Fishery | 600,000,000 | none | none | none |
| Harvest | 610,000,000 | 620,000,000 | 630,000,000 | 640,000,000 |

Harvest uses one weather stream per context. The earlier run used two; the
context is the unit either way.

**Fishery:** 64 contexts × [1 no-reviewer run + 3 reviewers × 3 budgets ×
2 fills × 2 targets] = 64 × 37 = 2,368 episodes. The no-reviewer run is
shared by both targets.

**Harvest:** 64 contexts × [1 + 3 × 3 × 2] = 64 × 19 = 1,216 episodes.

**Open-loop cases:** every request recorded in each no-reviewer run that
starts from a currently safe state. Each is evaluated by every reviewer ×
budget × fill (and × target in Fishery).

## 4. Outcomes

**Primary**, reported separately and never combined into one score:

1. Unsafe approval rate: resolved-risky requests approved unchanged ÷
   resolved-risky requests.
2. Usefulness loss: resolved-safe requests scaled down ÷ resolved-safe
   requests. Also the mean fraction of safe requests retained.
3. Long-run outcomes: total 80-step harvest, mean stock or patch health,
   and the share of the fixed horizon spent unsafe.

Outcomes 1 and 2 are computed both open loop and closed loop.

**Secondary:** unresolved counts; component checks per step.

## 5. Hypotheses, each with what would count against it

- **H1 (Harvest, k = 6, open loop).**
  - Prediction: joint review has usefulness loss < 5% and unsafe approval
    rate < 1%, and optimistic local has unsafe approval rate > 3%.
  - Falsifier: joint usefulness loss ≥ 5% or unsafe approval ≥ 1%, or
    optimistic local ≤ 3%.
  - Note: joint review with full inspection is close to the reference by
    construction (same model, same 5% rule, different random draws). H1
    therefore mainly checks the repair.
- **H2 (Harvest, k = 3, fill = `previous`, open loop).**
  - Prediction: usefulness loss is below 90% for joint review (no longer
    saturated), and joint and bounded local differ by more than 10
    percentage points.
  - Falsifier: either fails.
- **H3 (Fishery target).**
  - Prediction: under the `msy` target, joint review's total 80-step harvest
    at k = 6 is higher than under `one_step` at k = 6, and does not fall
    as k rises (k = 6 ≥ k = 0 − 2%).
  - Falsifier: either fails.
  - Note: this tests the post hoc explanation from the September reviewer write-up §6d.
- **H4 (safety in closed loop).**
  - Prediction: joint and bounded local have a fixed-horizon unsafe fraction
    ≤ 1% in both games, and optimistic local in Fishery has more than 10%.
  - Falsifier: either fails.

## 6. Analysis

- Counts and rates per game × reviewer × budget × fill (× target).
- Paired differences between reviewers at k = 6, and between targets, with
  95% percentile intervals from 4,000 bootstrap resamples of whole contexts
  (seed 20261005).
- No p-values and no pooling across games.
- A rate is labelled descriptive if fewer than 20 contexts contain the
  relevant class.

## 7. Execution and stop rules

- **Engineering checks first.** Unit tests must pass. Then a smoke run of 2
  contexts with a short horizon, which must show:
  - the chance-constrained joint reviewer at k = 6 agrees with the
    reference label on at least 95% of resolved open-loop cases;
  - identical outputs on rerun with the same seeds.
- **Then the full run, once.** One CPU worker, with a 30-minute cap.
- **Integrity rules:** no parameter changes after seeing full results, and
  no retries for "better" results. A crash may be repaired once, in a new
  directory, keeping the failed attempt.

## Amendments

(none yet)
