# The reviewer experiment, explained and re-analysed

This file covers the fresh-seed confirmation run of 23 September 2026
(`budgeted_reviewer_confirmation_v1`). It is the main independent evidence in
the current paper. The protocol was written before the run (locally, not
publicly), and the settings were chosen from an earlier pilot.

## 1. The question in one sentence

Six agents ask to take from a shared resource. A reviewer can see some of the
requests and may scale them all down. **How does what the reviewer sees (and
how it combines that information) change two kinds of mistake:**

- letting through a request that will make the system unsafe next step
  (*missed harm*), and
- cutting a request that was actually safe (*unnecessary restriction*)?

## 2. The two games, with the exact numbers used

**Fishery** (one shared stock)

- 6 agents. The stock starts at 70 and can grow to at most 100.
- Each agent may take up to 6 per step.
- After harvest, the remaining stock R regrows by 0.7 × R × (1 − R/100).
- "Unsafe" means the stock is below 10.
- No randomness, so the next stock can be computed exactly.
- Regrowth is fastest when R = 50. Leaving 50 after every harvest (so the
  stock is 67.5 at the start of each step) allows **17.5 units per step
  forever** (the MSY).

**Harvest** (six patches in a ring)

- Each patch belongs to one agent, starts at 12.5 and can reach at most 20.
- Regrowth rate is 0.476 (the "slow renewal" setting).
- Taking more than 2.1 from your patch also damages each neighbour's patch,
  by 0.22 × the excess.
- Weather adds random noise to every patch each step (standard deviation 0.42).
- "Unsafe" means the average patch is below 10, or at least half the patches
  are below 4.
- Regrowth per patch is fastest at R = 10, giving 2.38 per patch per step.
  So here the safety line sits close to the most productive level: a patch
  held exactly at the line regrows about 2.26 per step, about 95% of the
  maximum.

**Agents.** In both games the agents are fixed threshold rules: "if my
resource is above X, request a high fraction, otherwise a low one".
- Fishery used a mix with 4 aggressive agents out of 6; Harvest a mix with 2.
- Agents never change their behaviour in response to the reviewer, and they
  never lie, because there is nothing to lie about: the reviewer sees the
  true requests.

## 3. How the experiment was run

1. **Generate 64 fresh populations ("contexts") per game.** These are the
   independent units.
2. **Run each population with no reviewer.** Fishery ran once per context;
   Harvest ran under 2 separate weather streams per context. Record every
   joint request along the way. That gave 11,249 requests in total (1,009 Fishery,
   10,240 Harvest).
3. **Label each recorded request** as safe, risky or unresolved one step
   ahead. Only requests made from a currently safe state are scored: 757
   Fishery and 5,978 Harvest.
4. **Show every recorded request to every reviewer** at inspection budgets
   0, 3 and 6, and score its decision against the label. This is the
   *open-loop* part, and the primary result.
5. **Separately, run each population for 80 steps with each reviewer actually
   in charge** (*closed loop*). Record stock, total harvest and unsafe time.
   These are the secondary results.

Total: 1,920 episodes, 101,241 reviewer decisions.

## 4. A worked example (Fishery, context 0, step 14)

- **Situation.** The stock is 21.21. The six requests, as fractions of the
  maximum 6, are 0.568, 0.092, 0.574, 0.558, 0.159 and 0.499. Together they
  ask for 14.70 units.
- **Joint reviewer.** It adds them up: 21.21 − 14.70 = 6.51 left, which
  regrows to 10.77. That is above 10, so it approves.
- **Bounded local reviewer.** It assumes everyone might take as much as the
  biggest request: 6 × 0.574 × 6 = 20.65 units. That would leave the stock
  near 0, so it halves everyone's request.
- **Optimistic local reviewer.** It counts only one agent's largest request
  (3.44 units). It sees no problem and approves.

All three saw the same six numbers. They differ only in how they combine
them. **This is the whole mechanism of the experiment.**

## 5. Results as reported (verified against the data)

Full inspection (k = 6):

| Game | Reviewer | Risky approved | Safe restricted |
| --- | --- | ---: | ---: |
| Fishery | Joint | 0 / 63 | 0 / 694 |
| Fishery | Bounded local | 0 / 63 | 100 / 694 |
| Fishery | Optimistic local | 63 / 63 | 0 / 694 |
| Harvest | Joint | 0 / 1,368 | 2,521 / 3,687 |
| Harvest | Bounded local | 0 / 1,368 | 3,687 / 3,687 |
| Harvest | Optimistic local | 0 / 1,368 | 2,407 / 3,687 |

Lower inspection budgets:

- **Fishery joint reviewer:** it restricts 229 safe requests at k = 0,
  107 at k = 3 and 0 at k = 6.
- **Harvest, all three reviewers:** they restrict essentially every safe
  request at k = 0 and k = 3.

Closed loop (80 steps, k = 6):

"Unsafe time" here is the share of the 80 steps spent unsafe, with every step
after a collapse counted as unsafe. Counting only steps actually played, the
unsafe share is 0.37 for Fishery optimistic local and 0.35 for no reviewer.

| Game | Reviewer | Total harvest | Average stock or patch health | Unsafe time |
| --- | --- | ---: | ---: | ---: |
| Fishery | Joint | 551.5 | 16.9 | 0 |
| Fishery | Bounded local | 784.7 | 25.6 | 0 |
| Fishery | Optimistic local | 221.3 | 26.2 | 0.86 |
| Fishery | No reviewer | 218.9 | 27.1 | 0.86 |
| Harvest | Joint | 653.7 | 11.0 | 0 |
| Harvest | Bounded local | 625.7 | 11.7 | 0 |
| Harvest | Optimistic local | 653.2 | 10.9 | 0 |
| Harvest | No reviewer | 701.7 | 10.1 | 0.42 |

The coupled local replay checked all 11,249 requests at all three budgets:
0 disagreements with the joint reviewer out of 33,747 decisions. I reran it
myself and got the same result.

## 6. What these numbers actually mean

### 6a. Much of the Fishery result follows from the arithmetic

- **Joint reviewer.** In Fishery, at full inspection, the joint reviewer
  computes *exactly the same thing as the label*: same total, same regrowth
  formula, same threshold. It cannot make either mistake. Its "0 and 0" is
  true by construction, not measured.
- **Bounded local reviewer.** It always assumes total demand is at least
  the true total, so it can never approve a risky request. It can only
  over-restrict.
- **Optimistic local reviewer.** It always under-counts, so it can only
  miss harm.
- **Coupled local reviewer.** It is the joint calculation written
  differently. Zero disagreements was guaranteed. The replay is a check
  that the code is correct, not evidence about the world.

So the Fishery experiment measures *how loose two chosen approximations are
on this particular mix of requests*. That is a legitimate measurement, but
its direction was known before running anything.

### 6b. In Harvest, the size of the uncertainty allowance drives most of the result [post hoc]

**What the allowance is.** The Harvest reviewers protect against weather by
lowering *every patch's* predicted value by 1.005 (2.39 standard deviations)
before checking safety. That is a valid "union bound" for both parts of the
safety rule at once:

- the average patch must stay at or above 10;
- fewer than half of the patches may fall below 4.

**Why it is too big for these cases.** In the 5,978 scored Harvest cases, the
second condition never comes close to binding. The third-lowest predicted
patch is never below 8.4, and no weather draw produced a patch failure. So in
practice only the *average* condition matters. The average of six patches is
much less noisy than any single patch: a one-sided 5% allowance for the
average needs only 0.282. The allowance used is therefore about 3.6 times
larger than what the only binding condition requires.

I reran the reviewers' decisions on the saved requests with smaller
allowances (`scripts/margin_sweep.py`, and an independent re-implementation
during verification). Rerunning with the original allowance reproduces the
saved counts exactly, which shows the script is correct.

| Harvest, k = 6 | Allowance in the run (1.005) | Allowance for the average (0.282) |
| --- | --- | --- |
| Joint: safe restricted | 2,521 / 3,687 | **0 / 3,687** |
| Joint: risky approved | 0 / 1,368 | **4 / 1,368** |
| Bounded local: safe restricted | 3,687 / 3,687 | 3,529 / 3,687 |
| Optimistic local: safe restricted | 2,407 / 3,687 | 0 / 3,687 |
| Optimistic local: risky approved | 0 / 1,368 | **95 / 1,368** |

The result depends strongly on the allowance, so here is the whole range
(full inspection, my own re-implementation):

| Allowance | Joint: safe restricted | Joint: risky approved | Optimistic: safe restricted | Optimistic: risky approved |
| ---: | ---: | ---: | ---: | ---: |
| 0.282 | 0 | 4 | 0 | 95 |
| 0.35 | 21 | 0 | 8 | 42 |
| 0.50 | 477 | 0 | 350 | 17 |
| 0.79 | 1,780 | 0 | 1,649 | 0 |
| 1.005 (used) | 2,521 | 0 | 2,406 | 0 |

What this means:

- **The joint reviewer's 68% unnecessary restriction rate came from the
  allowance, not from missing information.** With an allowance between 0.28
  and 0.35 it is nearly perfect. That is expected: like Fishery's joint
  reviewer, it is the label's own calculation shifted by an offset.
- **The "optimistic local is as good as joint in Harvest" finding depends
  on the large allowance.** The extra cushion compensates for the neighbour
  damage that the optimistic reviewer ignores. With any allowance below
  about 0.79 it starts approving risky requests. That is the same direction
  as in Fishery (where it approved all 63), though much smaller.
- **Harvest then tells the same story as Fishery:** joint is close to exact,
  bounded local over-restricts, and optimistic local misses harm.
- **The 4 risky requests that slip through at 0.282 are borderline.** Their
  true risk is about 3.8–5.0%, but the 128-draw label called them risky.
  They reflect label noise at the 5% line, not a gap in the allowance.
- **Caveats.**
  - This is post hoc, so it must be redone prospectively (see file 04).
  - A prospective design should still protect both safety conditions, but
    with an allowance sized to the actual risk.
  - The 0.282 allowance also approves 662 of the 923 "unresolved" cases.

I also estimated each Harvest request's true one-step risk with 4,000 weather
draws instead of 128 **[post hoc]**:

- **"Safe" requests:** median risk 0%. 99.97% are below 5%.
- **"Risky" requests:** median risk 35%. 36% have risk above 50%. They are
  mostly clearly risky, not borderline.
- **"Unresolved" requests:** median risk 3.2%. 70% are actually below 5%.
  They sit exactly in the borderline band, where reviewers would disagree
  most. Leaving them out removes the hardest cases from the scoring.

### 6c. Inspection budgets 0 and 3 mostly test the "assume the worst" rule (Harvest)

Any request the reviewer did not see is assumed to be the maximum (1.0). In
Harvest, with two or three unseen requests at maximum, almost any joint
request looks dangerous. That is why every Harvest reviewer restricts nearly
everything at k = 0 and k = 3.

In Fishery the effect is milder. The joint reviewer restricts 33% of safe
requests at k = 0 and 15% at k = 3.

In Harvest, the budget curve therefore mainly shows how harsh this
worst-case assumption is, not how useful partial information could be. A
reviewer that filled unseen requests with something realistic would give a
more informative curve. That could be the population's typical request, or
the agent's last request.

### 6d. The long-run reversal most likely comes from the safety target

The Fishery reviewers in closed loop:

| Fishery reviewer | Median stock at start of step | Median left after harvest | Share of steps it cut requests | Average scaling | Harvest per step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint, k = 0 | 38 | 25 | 77% | 0.77 | 13.6 |
| Bounded local, k = 6 | 22 | 14 | 88% | 0.61 | 9.8 |
| Joint, k = 6 | 13 | 8 | 86% | 0.43 | 6.9 |

Regrowth depends on what is left after harvest. It is fastest when 50 is
left (17.5 per step). It is about 13 when 25 is left, about 8 when 14 is
left, and about 5 when 8 is left.

What this shows:

- **The best-informed reviewer allows the most extraction early.** It
  approves anything that keeps the stock just above 10. After that, only
  about 8 is left after each harvest, which regrows slowly. So the reviewer
  has to cut requests most of the time, and long-run harvest is low.
- **The least-informed reviewer is the most cautious, by accident.** About
  25 is left after each harvest, so regrowth is much faster and total
  harvest is highest: 1,084 over 80 steps.
- **None of the reviewers gets near the sustainable maximum.** Leaving 50
  after every harvest would give 17.5 per step, about 1,400 over 80 steps.
- **The one-step "safe" label is far from sustainable** **[post hoc]**.
  All 694 Fishery requests labelled safe leave less than 50 after harvest:
  340 leave less than 30, and 158 less than 20. Even after regrowth, 409
  (59%) leave the stock below 50.

The most likely cause is that the safety line (stock ≥ 10) sits far below
the level where the resource is most productive. Approving more "safe"
requests is then not the same as allowing more useful work, so **the main
metric counts as mistakes some restrictions that actually improve the
long-run outcome.**

This is an explanation, not a tested fact. No run varied the safety line, so
the only measured contrast is information (k) under one fixed line. Varying
the line is part of the proposed next step.

In Harvest the safety line is close to the most productive level, so this
problem should be milder. Still, no reviewer beat "no reviewer" on total
harvest within 80 steps (701.7 vs at most 653.7). A likely contributor is the
oversized allowance, which makes every reviewer cut too much.

### 6e. The scored requests have a different mix from what the reviewer faces in operation

The scored requests come from runs where *nobody intervened*. In Fishery:

- The stock in those runs crosses below 10 at a median of step 8, and the
  run ends four steps later.
- So the scored requests come only from the first, downhill part of an
  unregulated run. The median stock across the 757 scored requests was 47.4.
- The 63 risky requests are one per context. Each is the single step where
  the stock crossed below 10, exactly four recorded steps before the end of
  that context's run. So the "risky" class is really 63 copies of one kind
  of event: the step that tips the stock over the line.

Under a reviewer, the game runs for 80 steps in a different region, with the
stock at 13–38. There the mix of cases is reversed:

- In closed loop, about 86% of the requests the fully informed joint
  reviewer faced were one-step risky.
- In the open-loop scoring set, only 8% (63 of 757) were.
- The joint reviewer handled both correctly: every one of its 4,426 cuts in
  closed loop was a risky request, and it never cut a safe one.

For the joint reviewer, the error rates carry over from scoring to
operation. For the other reviewers this was not checked. Either way, the
scoring set says little about the situations the reviewer actually spends its
time in. Because the agents
keep asking for too much, the reviewer has to cut almost every step.

## 7. What the experiment does establish

- **The protocol works.** Same target, same set of possible actions, same
  inspected requests, different ways of combining them. The data are
  complete, hashed and reproducible; I reran the tests, the replay and the
  figure.
- **How a reviewer *combines* information matters,** and the direction of
  the error follows from the approximation: over-counting causes unnecessary
  restriction, under-counting causes missed harm.
- **Good one-step decisions do not guarantee good long-run outcomes.** The
  most likely reason here is where the safety line sits relative to the most
  productive stock level.

## 8. What it does not establish

- Anything about agents who lie, hide, or adapt to the reviewer. None do.
- Anything about reviewers that do not know the rules of the game.
- That information access (as opposed to how it is used) is what limits
  oversight. Once inspected numbers are combined correctly, the local and
  joint reviewers are identical.
- Generality beyond these two settings. They were chosen after a pilot
  because they produced both safe and risky requests.
