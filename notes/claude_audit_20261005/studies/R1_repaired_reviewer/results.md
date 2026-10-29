# Results R1: the reviewer experiment, repaired

- **Protocol:** `studies/R1_repaired_reviewer/protocol.md`, frozen 2026-10-04T23:57Z
  before the run, with no amendments.
- **Run:** `results/runs/claude_r1_repaired_reviewer_v1`. It took 171
  seconds on one CPU and produced 3,648 episode records.
- **Analysis:** `experiments/analyze_r1_repaired_reviewer.py`, with outputs
  in `.../analysis/`.

Every number below is **verified**: it was computed by those scripts from
the run's saved data. An independent check (the verification log, §8) corrected some
statements in an earlier draft of this file. Engineering checks passed before the full run:
- unit tests 4/4;
- the smoke run reproduced byte-for-byte;
- joint review at k = 6 matched the reference on every smoke case.

## 1. Question

Once the design weak points from the design-issues note are fixed, what do the three
reviewers actually trade off? The two quantities are:
- **unsafe approvals:** letting through a request whose chance of making the
  system unsafe next step is above 5%;
- **usefulness loss:** cutting a request that was safe.

And does the Fishery long-run reversal ("better information → less total
harvest") disappear when the safety target is set at the level where the
stock is most productive?

## 2. What was done

**Same as 23 September:**
- games and agent types: Fishery with 4 of 6 aggressive agents; Harvest with
  2 of 6 aggressive, slow renewal;
- the action menu (scale all requests by 1, 0.75, 0.5, 0.25 or 0);
- random inspection of 0, 3 or 6 requests;
- an 80-step horizon.

**Fixed:**
- **No hand-set weather buffer.** Every reviewer now uses a **chance
  constraint** (Charnes & Cooper 1959). It approves the largest scale whose
  estimated one-step risk is ≤ 5%, estimated from 400 simulated weather draws
  run through that reviewer's own prediction.
- **Unseen requests** are either assumed to be the maximum (`max`, the old
  rule) or set to that agent's request from the previous step (`previous`).
- **Labels** use 4,000 weather draws (open loop) or 2,000 (closed loop).
  Unresolved Harvest cases fell from 923 (15%) to 43 (1.6%).
- **Every step a reviewer actually faced in closed loop is scored too,** not
  only requests from unregulated runs.
- **Fishery target is a factor:**
  - `one_step`: stock stays ≥ 10 next step, the old line;
  - `msy`: at least 50 left after harvest. This is half the carrying
    capacity, where logistic regrowth is fastest (the MSY level; Schaefer
    1954, secondary source).

**Sample:**
- 64 new population contexts per game (the independent unit).
- 2,368 Fishery episodes (plus one duplicate no-reviewer run per context)
  and 1,216 Harvest episodes.

## 3. What came out

### 3a. Harvest decisions, scored on requests from unregulated runs (open loop)

There were 2,016 safe, 720 risky and 43 unresolved requests from safe states.

| Inspected | Unseen filled with | Reviewer | Risky approved | Safe cut | Avg. share of safe request kept |
| --- | --- | --- | ---: | ---: | ---: |
| 6 | – | Joint | 4 / 720 (0.6%) | 4 / 2,016 (0.2%) | 99.9% |
| 6 | – | Bounded local | 0 / 720 | 1,807 / 2,016 (89.6%) | 68.6% |
| 6 | – | Optimistic local | **91 / 720 (12.6%)** | 2 / 2,016 (0.1%) | 99.98% |
| 3 | max | Joint | 0 / 720 | 1,944 / 2,016 (96.4%) | 53.5% |
| 3 | previous | Joint | **169 / 720 (23.5%)** | 215 / 2,016 (10.7%) | 97.1% |
| 3 | previous | Bounded local | 0 / 720 | 1,810 / 2,016 (89.8%) | 68.5% |
| 3 | previous | Optimistic local | 218 / 720 (30.3%) | 140 / 2,016 (6.9%) | 98.2% |
| 0 | previous | Joint | 262 / 720 (36.4%) | 334 / 2,016 (16.6%) | 94.9% |

Paired differences at k = 6, with 95% intervals over contexts:
- joint − optimistic local, unsafe approvals: −12.1 points [−17.1, −8.5];
- joint − bounded local, usefulness loss: −89.4 points [−96.7, −80.6].

### 3b. Fishery decisions (open loop, one-step target)

There were 704 safe and 62 risky requests.

**Full inspection (k = 6):**

| Reviewer | Risky approved | Safe cut |
| --- | ---: | ---: |
| Joint | 0 | 0 |
| Bounded local | 0 | 103 / 704 (14.6%) |
| Optimistic local | 62 / 62 | 0 |

These match the 23 September results closely.

**Joint reviewer with fewer inspections:**

| Inspected | Unseen filled with | Risky approved | Safe cut |
| --- | --- | ---: | ---: |
| 0 | max | 0 | 206 / 704 |
| 0 | previous | 0 | 15 / 704 |

**Under the `msy` target,** all 766 requests from unregulated runs were
"risky": the aggressive agents always ask for more than the sustainable
amount. So usefulness loss cannot be measured there. Joint and bounded local
approve none of them; optimistic local approves 169–243.

### 3c. Long-run outcomes (closed loop, mean over 64 contexts)

Unless a row says otherwise, unseen requests are filled with the maximum
(`max`). Harvest uses one weather stream per context. The fill matters a lot
in Fishery, so both fills are shown in the second table.

| Game, target | Reviewer, inspected | Total harvest (80 steps) | Mean stock / health | Share of horizon unsafe |
| --- | --- | ---: | ---: | ---: |
| Fishery, one-step | Joint, 0 | 1,073 | 38.1 | 0 |
| Fishery, one-step | Joint, 3 | 840 | 27.7 | 0 |
| Fishery, one-step | Joint, 6 | 555 | 17.1 | 0 |
| Fishery, one-step | Bounded local, 6 | 784 | 25.6 | 0 |
| Fishery, one-step | Optimistic local, 6 | 225 | 26.2 | 0.86 |
| Fishery | No reviewer | 223 | 26.8 | 0.86 |
| **Fishery, MSY** | **Joint, 0** | **1,313** | 78.2 | 0 |
| **Fishery, MSY** | **Joint, 3** | **1,359** | 74.8 | 0 |
| **Fishery, MSY** | **Joint, 6** | **1,393** | 70.4 | 0 |
| Fishery, MSY | Bounded local, 6 | 1,344 | 76.1 | 0 |
| Harvest, one-step | Joint, 6 | 675.6 | 10.58 | 0.005 |
| Harvest, one-step | Bounded local, 6 | 640.1 | 11.28 | 0 |
| Harvest, one-step | Optimistic local, 6 | 677.2 | 10.55 | 0.015 |
| Harvest, one-step | Joint, 0, previous fill | 673.5 | 10.66 | 0.026 |
| Harvest | No reviewer | 700.1 | 10.07 | 0.46 |

The sustainable maximum in Fishery is 17.5 per step, about 1,400 over 80
steps.

**Fishery joint review, total harvest by fill:**

| Target, fill | k = 0 | k = 3 | k = 6 |
| --- | ---: | ---: | ---: |
| one-step, `max` | 1,073 | 840 | 555 |
| one-step, `previous` | 556 | 555 | 555 |
| MSY, `max` | 1,313 | 1,359 | 1,393 |
| MSY, `previous` | 1,392 | 1,392 | 1,393 |

Paired differences, with 95% context-bootstrap intervals:
- **Fishery joint, k = 6:** MSY target − one-step target = +838 total
  harvest [+800, +865].
- **Fishery, MSY target:** joint k = 6 − k = 0 = +80 [+78, +81].
- **Fishery, one-step target:** joint k = 6 − k = 0 = −518 [−541, −489].
- **Harvest, k = 6:** joint − bounded local = +35.5 total harvest
  [+27.0, +43.7].

### 3d. Hypothesis verdicts (criteria fixed in the protocol)

| | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| H1 | Harvest k = 6: joint usefulness loss < 5% and unsafe approval < 1%; optimistic > 3% | 0.2%, 0.6%; 12.6% | Supported |
| H2 | Harvest k = 3, `previous` fill: joint usefulness loss < 90%, and differs from bounded by > 10 points | 10.7% vs 89.8% | Supported, **but** see §4b |
| H3 | Fishery MSY target: joint k = 6 beats one-step k = 6, and harvest does not fall with more inspection | +838; +80 (`max` fill; with `previous` fill it is flat) | Supported |
| H4 | Closed-loop unsafe share ≤ 1% for joint and bounded; > 10% for optimistic in Fishery | 0–0.5% at k = 6; 86% | Supported at k = 6 (the analysis checked only k = 6). With `previous` fill, Harvest joint reaches 1.6% (k = 3) and 2.6% (k = 0). |

## 4. What it means

### 4a. The Harvest picture was mostly the buffer, as suspected

With a calibrated chance constraint, the result in Harvest is:
- joint review is nearly exact;
- the bounded local rule cuts 90% of safe requests;
- the optimistic local rule lets through 1 in 8 risky requests.

That is the same ordering as in Fishery. This **confirms the post hoc
re-analysis** (the September reviewer write-up §6b) on fresh data, with a protocol written before
the run.

**True by construction:** joint review with full inspection uses the same
model and the same 5% rule as the label. Only the random draws differ. Its
near-perfect score is therefore expected, not discovered. The informative
parts are how far the two local approximations fall from it, and what
happens when inspection is limited.

### 4b. Limited checking now shows a real trade-off

The old rule (assume unseen requests are the maximum) made almost every
request look dangerous. Assuming instead that each unseen agent repeats its
last request removes most unnecessary cuts. Joint review at k = 3 goes from
cutting 96% of safe requests to 10.7%.

But this is **not free**: it lets through 23.5% of risky requests, and 36.4%
at k = 0. In closed loop, Harvest joint review with no inspections and the
`previous` fill was unsafe for 2.6% of the horizon, against 0.5% with full
inspection.

**What it shows:** what an overseer assumes about what it did not check
decides whether it errs towards unsafe approvals or towards usefulness loss.
This is the "limited checking" part of the original question, now measurable
instead of saturated.

**What it does not show:** the best way to fill in unchecked requests. Only
two simple fills were tried.

### 4c. The Fishery long-run reversal: the target matters, and "information" partly meant "less pessimism"

This section was corrected after the independent check. The earlier draft
said the reversal "came from the target, not the information". The fuller
picture:

- **The target decides the level of long-run harvest.** At full inspection
  the same reviewer harvests 1,393 under the MSY target and 555 under the
  old line: +838 [+800, +865]. That holds whatever the fill. This tests and
  supports the post hoc explanation in the September reviewer write-up §6d, with the prediction
  fixed in the protocol.
- **The reversal needs two ingredients together:**
  - the old line, **and**
  - assuming unseen requests are the maximum.

  That pessimism acts as an accidental conservation buffer:
  - **under the old line**, the buffer was the only thing holding the stock
    up, so inspecting more (removing the buffer) lowered harvest;
  - **under the MSY target**, the target itself conserves, so the buffer
    only wastes harvest, and inspecting more raises it slightly
    (1,313 → 1,393).
- **With the realistic fill,** each unseen agent assumed to repeat its last
  request, inspection makes almost no difference under either target.

So what looked like "better information hurts" was really "less pessimism
about unchecked requests hurts *when the target is set too low*".

Partly by construction: an exact reviewer enforcing "leave 50 after
harvest" is the textbook MSY harvest rule, so reaching about 1,400 is
expected.

### 4d. Reviewers still cost harvest in Harvest

No reviewer beat "no reviewer" on total Harvest harvest within 80 steps
(700.1 vs at most 677.2). But "no reviewer" spent 46% of the horizon unsafe,
against at most 1.5% for the reviewers.

So the cost of safety in Harvest is about 3–4% of total harvest for joint or
optimistic review, and 9% for bounded local review. Here the safety line
almost equals the most productive level, so this cost is a real price for
avoiding unsafe states, not an artefact of a bad target.

## 5. Design caveats

- **Still honest agents and a known model.** Nobody lies, the reviewer knows
  the exact game rules, and agents do not adapt. S1 relaxes the first of
  these.
- **The settings were chosen in earlier pilots,** and R1 itself was
  motivated by looking at the 23 September data. It is a repair and
  replication on fresh seeds, not an independent discovery.
- **The reference labels share the reviewer's model.** They are one-step
  Monte Carlo estimates, not long-run truth.
- **The MSY target is only tested in Fishery.** In Harvest the safety line
  is already close to the most productive level (about 95%).
- **Harvest used one weather stream per context** (the 23 September run used
  two). The context is the independent unit either way.
- **A small detail at full inspection.** Harvest joint review approves
  slightly *more* risky requests than bounded local review: +0.56 points
  [+0.13, +1.11]. Its draws are an estimate, so a few borderline cases go the
  other way.
- **The Fishery no-reviewer run was recorded once per target.** The two
  records are identical, and the analysis counts it once.
- **"Previous request" assumes the reviewer learns each agent's last
  request after the fact.** That is a modelling assumption, not
  something tested.

## 6. Where this leaves us

**What we now believe, with reasonable confidence in these two settings:**
- with honest agents and a known model, reviewer errors follow from how
  requests are combined and what is assumed about unchecked requests;
- a calibrated chance constraint removes the artefacts of a hand-set buffer;
- a safety target at the productive level, rather than at the brink, gives
  far more long-run harvest (+838 in Fishery);
- what looked like a penalty for better information was the side effect of
  pessimistic assumptions about unchecked requests under a target set too
  low.

**Still untested:** what happens when the reports themselves cannot be
trusted. That is S1, next file.
