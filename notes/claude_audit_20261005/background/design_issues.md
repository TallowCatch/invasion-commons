# Design issues in the experiments

These are problems in **how the experiments were set up**, not in how the
paper is worded. Each one has:

- **What:** the problem in plain words.
- **Evidence:** where it shows in the code or data.
- **Why it matters:** what conclusion it weakens.
- **Fix:** how a future design would avoid it.

Each is also labelled "acknowledged" if your own notes already raise it,
or "new" if I found no acknowledgement.

Severity:

- **A** changes the interpretation of the current main result.
- **B** limits what the current main result can show.
- **C** affects historical results only. It matters only if those results
  stay in the paper.

---

## A. Issues that change how the current main result should be read

### A1. The Harvest uncertainty allowance is about 3.6× larger than the binding condition needs (new)

- **What.** Reviewers subtract 2.39 standard deviations (1.005) from
  *every* patch's prediction. This is a valid "union bound" that protects
  both parts of the safety rule:
  - the average must stay at or above 10;
  - fewer than half of the patches may fall below 4.

  In these cases, however, the second condition never comes close to
  binding: the third-lowest predicted patch is never below 8.4. Only the
  average condition matters, and the noise on an average of six patches is
  much smaller. A 5% one-sided allowance for the average is about 0.28.
- **Evidence.**
  - The margin is set at `fishery_sim/budgeted_oversight.py:77`.
  - My re-analysis is in `scripts/margin_sweep.py` **[post hoc]**, checked by
    an independent re-implementation. With the 0.28 allowance:
    - joint review's unnecessary restrictions fall from 2,521/3,687 to 0;
    - the optimistic local reviewer approves 95/1,368 risky requests
      instead of 0.
  - The result is sensitive to the allowance; the full range is in the September reviewer write-up
    §6b. For example, at 0.35 joint restricts 21 safe requests and the
    optimistic local reviewer approves 42 risky ones.
- **Why it matters.** The Harvest headline numbers mostly measure the
  allowance, not information. The "optimistic local is competitive in
  Harvest" finding holds only when the allowance is large, at about 0.79 or
  more.
- **Fix.** Before the run, size the allowance so that every reviewer's
  approval rule has about the same risk as the label's 5% tolerance, for
  *both* safety conditions. Check this on held-out synthetic cases. Report
  results across a small range of allowances, because the comparison
  depends on it.

### A2. The one-step safety line is far from the resource's productive level (partly acknowledged)

- **What.** In Fishery, "safe" means the stock stays above 10. The stock
  regrows fastest at 50. Approving everything that keeps the stock above 10
  pushes it to a low-productivity state.
- **Evidence** **[post hoc]**:
  - All 694 "safe" Fishery requests leave less than 50 after harvest; 340
    leave less than 30.
  - In closed loop, the fully informed joint reviewer leaves a median of
    about 8 after harvest and harvests 6.9 per step.
  - The uninformed one leaves about 25 and harvests 13.6 per step.
  - The sustainable maximum is 17.5 per step (50 left after harvest).
- **Why it matters.**
  - The primary metric ("safe requests restricted") counts as *errors* some
    restrictions that actually raise long-run harvest.
  - "Retaining safe activity" is not the same as "retaining useful
    activity".
  - Your notes acknowledge that one-step and long-run outcomes disagree, but
    not that the likely cause is where the safety line sits.
  - This explanation has not been tested, because no run varied the line.
- **Fix.**
  - Decide what "harm" means for the paper.
  - Either add a sustainability-aware target, such as not dropping below the
    most productive level or a multi-step lookahead, or report long-run
    harvest and stock as *co-primary* outcomes.
  - In Harvest, the safety line is already close to the most productive
    level (about 95% of maximum regrowth), which makes Harvest the better
    test bed for this question.

### A3. Several reviewer results are true by construction (partly acknowledged)

- **What.**
  - In Fishery at full inspection, the joint reviewer computes exactly the
    label function, so it cannot err.
  - The bounded local reviewer always over-counts, so it can never approve
    harm.
  - The optimistic local reviewer always under-counts, so it can only miss
    harm.
  - The coupled local reviewer is algebraically the joint reviewer, so zero
    disagreements was guaranteed.
- **Evidence.** `budgeted_oversight.py:113-127` against the Fishery label in
  `fishery_oversight.py:83-85`.
- **Why it matters.** The direction of every result was predictable without
  simulation. The experiment measures *how big* the errors of two chosen
  approximations are on this case mix. It does not discover *whether*
  information helps. Your notes call the replay an "implementation check",
  which is right, but the main contrast has the same character.
- **Fix.** Real uncertainty has to come from somewhere other than the
  reviewer's arithmetic:
  - reports that can be false;
  - a reviewer that does not know the model exactly;
  - agents that respond to the reviewer.

  See the what-we-learned note.

### A4. "Bounded local" discards most of the inspected information (partly acknowledged)

- **What.** The method description says it bounds *unseen* neighbours. In
  the code, it bounds *all* other agents, seen or not:
  - Fishery uses 6 × the largest request, so other agents' inspected
    values enter only through that maximum;
  - Harvest assumes every neighbour takes the maximum, so neighbours'
    inspected values never enter at all.
- **Evidence.** `budgeted_oversight.py:121-122`;
  `oversight_protocol.py:81-91` (`local_patch_report` takes no neighbour
  input). The paper's limitations section (`main.tex:324`) already says
  the rule "discards useful inspected information"; the method description
  is what is inaccurate.
- **Why it matters.**
  - In Harvest, inspection budget only affects this reviewer through each
    agent's own request.
  - The joint vs bounded-local gap measures how information is *used*, not
    what is *available*. The replay later made exactly this point.
- **Fix.** Describe it accurately. For a fair "local information" arm,
  give local reviewers their neighbours' inspected requests and nothing else.

### A5. The scored requests come from runs nobody was regulating (new)

- **What.** The decision results are scored on requests recorded from
  unregulated runs (open loop).
  - In Fishery's unregulated runs the stock crosses below 10 at a median of
    step 8, so only the early downhill phase is scored.
  - Each Fishery context contributes exactly one risky request: the step
    that crosses the line.
  - Under a reviewer, the game spends 80 steps in quite different states.
    About 86% of the requests the joint reviewer faces there are risky,
    against 8% in the scoring set.
- **Why it matters.** The case mix being scored is not the one the reviewer
  faces in operation. The 63 Fishery risky cases are 63 copies of one type
  of event. For the joint reviewer the error rates happen to carry over (it
  cut only risky requests in closed loop); for the other reviewers this is
  unknown.
- **Fix.** Score decisions on the states the reviewer actually encounters in
  closed loop, labelling each request against the same reference. Report the
  case mix: how far from the line, and which phase of the run.

---

## B. Issues that limit what the current main result can show

### B1. Agents never react to the reviewer (acknowledged in general)

- **What.** Every policy is a fixed threshold rule. Nobody misreports,
  splits requests, waits for a lax reviewer, or learns.
- **Why it matters.** Oversight is hard mainly because actors respond to it.
  Without that, the scalable-oversight link is only motivation.
- **Fix.** Add reports that can be false, audits that cost something, and
  actors that choose their behaviour knowing the reviewer protocol. See
  the what-we-learned note.

### B2. The reviewer knows the exact game rules (acknowledged)

- **Why it matters.** A known model plus true inputs makes the joint
  reviewer an oracle.
- **Fix.** Add model error to the reviewer, for example a wrong regrowth
  rate, or require it to estimate the model from data.

### B3. Inspection budgets 0 and 3 mostly test "assume the maximum" (new)

- **What.** Unseen requests are set to 1.0, the largest possible, so with 2
  or more unseen requests almost everything looks dangerous. In Harvest, all
  reviewers restrict about 100% of safe requests at k = 0 and 3. Fishery is
  milder: joint restricts 33% at k = 0 and 15% at k = 3.
- **Why it matters.** In Harvest, the inspection-budget curve, the "limited
  checking" part of the question, is saturated and uninformative.
- **Fix.** Fill unseen requests with a realistic estimate (the population's
  typical request, or the agent's previous request), plus a calibrated
  allowance. Or let the reviewer choose *which* requests to inspect (risk
  targeting).

### B4. The hardest cases are left out of the scoring (new)

- **What.** Harvest requests whose risk is close to 5% are labelled
  "unresolved" and dropped: 923 cases, about 15%.
- **Evidence** **[post hoc]**, from 4,000-draw risk estimates: unresolved
  cases have median risk 3.2%, and 70% are below 5%.
- **Why it matters.** The dropped band is exactly where reviewers would
  disagree. What remains is easy: risky cases have median risk 35%.
- **Fix.** Use more weather draws for the label (thousands are cheap), so
  almost no case is unresolved. Or report a risk-weighted score that does
  not need a hard label.

### B5. The two settings were chosen after a pilot (acknowledged)

- **What.** Fishery with 4 aggressive agents and Harvest with 2 under slow
  renewal were chosen because they produced both safe and risky requests.
- **Why it matters.** Fresh seeds protect the confirmation from simple
  overfitting. Generality, however, rests on two hand-picked points.
- **Fix.** Predeclare a small grid of settings, for example aggressiveness
  × regrowth, and report all of them.

### B6. Communication cost is a bookkeeping convention (acknowledged)

- **What.** Local reviewers are charged about 11.5–17 "transmitted values"
  per step even with zero inspections (depending on game and rule); joint
  review is charged 0.
- **Fix.** Define the cost model before the run, and test cost as a factor
  if it is part of the claim.

---

## C. Issues in the historical (pre-September) experiments

These only matter if those results stay in the paper.

1. **Fishery's static quota was unsustainable even with full compliance**
   (new). It allows 7% of the stock per agent per step; with 12 agents
   that is 84%. "Adaptive quota wins" is largely "a sustainable limit beats
   an unsustainable one" (`run_governance_ablation.py:73`).
2. **The penalty for collapse did not affect which strategies survived**
   (new). It is subtracted equally from everyone in the population, and
   survival is decided by rank (`evolution.py:587-590`). The same holds for
   Harvest's garden-failure penalty. Two exceptions:
   - Harvest's search-based entrant generator scores each candidate in its
     own episode, so there the penalty *does* influence which mutant enters
     (`harvest_evolution.py:832`);
   - the Fishery language-model prompt shows the penalised fitness.
3. **Confidence intervals with 5 runs were too narrow** (new).
   - Stage A uses 1.96 × SE (`summarize_harvest_invasion.py:120`); with 5
     runs the correct multiplier is 2.78, so the intervals are about 30%
     too narrow.
   - Fishery used a percentile bootstrap over 5 runs, which also
     under-covers.
4. **The "stress" test regimes were weaker than labelled** (new; code bug).
   - `get_harvest_regime_pack` builds its numbers from the base tier, then
     applies them to scenario configs that have different base values
     (`harvest_benchmarks.py:267-297`).
   - Community-irrigation "slow regrowth" was −7%, not −15%.
   - "Strong spillover" was +4% to +6%, not +30%.
   - This affects Stage A, the frictions study, and the September
     frozen-population checks (the two "slow regrowth" columns are not the
     same manipulation).
5. **The central governor is almost always on** (new). It triggers below
   average health 16, far above the safety line of 10. Under the
   strong-overseer setting, its zero unsafe time follows from the tuning.
   Under limited or weak overseers it still had 3–14% unsafe time.
6. **Stage A's "local" package has no overseer, and the capability-gap score
   is not a measurement** (acknowledged). New details:
   - only 48 of the 72 rows are distinct;
   - the "gap" effect comes entirely from the overseer setting;
   - the actor ladder runs backwards in every condition without an
     overseer: the "weakest" actor setting causes the most unsafe time.
7. **About half of hybrid's 13 wins are ties at zero unsafe time** (partly
   acknowledged). In those cases hybrid won on health differences of 0.01
   to 0.06.
8. **Language-model strategies were mostly shaped by the prompt numbers and
   by post-processing** (acknowledged). New details: very low diversity
   (Llama used only 5 distinct values for one cooperative field), and the
   Fishery arm added greedy drift after generation.
9. **Obedience is hard-coded** (new as a design point). Every Harvest policy
   lowers its request when neighbours announce restraint and caps itself at
   any announced cap plus at most 0.25. Results about messages and
   announcements say nothing about agents that could ignore them.
10. **The PPO reward was tuned on the evaluation metrics** (new). The
    training reward included bonuses and penalties for the same health and
    aggression measures used to judge it.
11. **"Confirmation" runs before September reused pilot seeds** (new). All
    stages start from seed 0, so they are not fresh draws. The September
    confirmation *did* use fresh seed bases.

---

## What was done well

These are worth keeping in every future experiment.

- **Protocols written before the runs, with failures kept.** Clean Up and
  the actor-pressure pilot are recorded as failures, not hidden.
- **Same target and same action menu across reviewers** in the confirmation.
  This is the right instinct and fixed the earlier unfair comparisons.
- **Independent units chosen correctly** in the confirmation: 64 fresh
  contexts, resampled as whole contexts.
- **Hashes, manifests and a committed archive.** The main result is fully
  reproducible: tests pass, the replay reruns, and the figure regenerates
  identically.
