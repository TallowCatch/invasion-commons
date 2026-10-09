# Propositions: the results that are arithmetic, stated once (5 October 2026)

Several findings follow from simple arithmetic about the agents and reviewers.
Each is stated here once, with its conditions, so the simulations are needed
only for what is *not* arithmetic. That covers:

- whether strategies chosen on training contexts generalise;
- noisy liars;
- the resource dynamics and collapse.

This answers caveat 1 (two games only) in `background/caveats_assessment.md`:
a general statement plus worked examples, instead of claims tied to two games.

Notation:

| Symbol | Meaning |
| --- | --- |
| q | probability that a given agent is audited in a step |
| s | probability that an audit detects an over-take |
| F | flat fine per catch |
| e = q s F | expected fine per agent-step |
| g | a cheater's expected gain from one over-take step at its chosen level |

---

## P1. The deterrence break-even

**Statement.** Take a risk-neutral cheater, facing a flat fine, whose
over-takes do not change its later gains much. It over-takes in a step if and
only if g > e. So deterrence starts at e* = g, whatever the mix of q, s and F
behind e.

**Conditions:**
- risk neutrality;
- a flat fine (not growing with the over-take);
- audits independent across steps and unpredictable;
- no other consequence of a catch (no memory, no exclusion);
- gains roughly additive across steps.

**Evidence:**
- S3: e* was within 0.06 of the 0.356 break-even, for every audit rate and
  detection rate tested.
- R2: the threshold matched in all 3 computable Fishery settings **once g
  was measured per over-take step at the cheaters' own preferred level
  [post hoc]**:

  | Setting | e* | g |
  | --- | ---: | ---: |
  | 2 stress agents, r = 0.5 | 1.38 | 1.29 |
  | 4 stress agents, r = 0.7 | 0.373 | 0.373 |
  | 4 stress agents, r = 0.9 | 0.245 | 0.227 |

  R2's frozen definition (a level of 0.75, per horizon step) failed in 3 of
  4 testable settings. That is a fault of the stated definition, not of P1.

- **R3 (prospective):** g\* was fixed on training runs before any search. The threshold fell within one grid step of g\* in 5 of 5 testable settings, with g\* ranging from 0.19 to 1.20 (`studies/R3_breakeven_confirmation/results.md`).

**What breaks it:** risk aversion, fines that scale with the over-take,
predictable audits (P2), and memory (P4).

## P2. A known audit schedule cannot deter

**Statement.** Suppose the cheater knows which steps are audited, and its gain
per unaudited step is positive. Then it over-takes on every unaudited step and
complies on audited ones. Its expected fine is 0 for every F.

**Evidence:** S3 Part C. Timed cheaters were never caught. Honest agents
ended worse off than with no checks: 28.8 against 40.9.

## P3. Memory against a fixed liar

**Statement.** Take a liar who always under-reports by the same fraction d.
- One catch reveals d exactly.
- From then on, a reviewer with memory corrects that liar's reports
  exactly.
- The number of steps before the first catch is geometric with mean 1/q.
- So the harm that remains is about the per-step harm of uncorrected lies
  times 1/q, per liar.

Against a liar who draws a fresh fraction each step, one catch reveals only
one draw. The correction is then right only on average, and harm remains.

**Evidence:**
- S3 Part D: Fishery, from 69% of steps unsafe to 1.5% with fixed liars, and
  to 32% with noisy liars.
- R2 grid: memory was safer than no memory in 9 of 9 testable settings.

## P4. Memory alone does not deter

**Statement.** Suppose a catch only changes the agent's future allowance,
which can never go below 0, and no penalty pushes its payoff below
complying. Then over-taking weakly dominates complying. A reviewer with memory
but no sanction can reduce harm, but cannot make cheating unprofitable.

**Evidence:** S4. The chosen level was never 0 under memory alone, at any
audit rate. With memory plus a fine, deterrence started at half the audit
rate needed with the fine alone.

## P5. Joint review with a known model and true inputs is exact

**Statement.** Suppose a reviewer:
- adds up all true requests;
- predicts with the environment's own model;
- checks the same chance constraint used to label harm.

Then its decisions match the labels, up to Monte Carlo error. Local rules err
in fixed directions:
- assuming others take the maximum over-restricts;
- ignoring others misses harm.

**Evidence:**
- R1: joint approved 0.6% of risky requests.
- C1: joint approved 0.3%, against 100% for optimistic local, in a game with
  compositional harm.

**What breaks it:**
- **a wrong model (R2 Part B).** Underestimating weather noise by half raised
  joint's risky approvals from 0.9% to 26%. Overestimating regrowth by 25%
  raised them to 82%.
- **false inputs (S1).**

## Conjecture C1, revised after T1. A deterministic report-based aim leaves most agents unchecked

**Statement.** If audits go to the agent whose *reported* request most
changes the predicted harm, an under-reporter lowers its own chance of audit
by under-reporting.

**Evidence:** C1 Part B. Sensitivity-targeted audits gave 3.4 points more
unsafe steps than random audits [1.8, 5.3].

In S1 Harvest, targeted audits used a physical signal (patch state) rather
than the report, and helped. This is post hoc and needs a direct test: compare
report-based targeting with signal-based targeting.

**T1 (7 Oct) revises this conjecture.** On new seeds, aiming at the largest report was worse than random in Fishery and River. But it landed on liars *more* often than random audits did (100% and 72%, against 67% and 34%).

The failure is **coverage**. The aim keeps returning to one agent that memory has already corrected, so most liars are never audited: only 25% of liars are ever caught in Fishery and 76% in River, against 100% at random **[post hoc diagnostic]**.

**Corrected statement.** With memory, an audit adds information only about agents not yet corrected. A deterministic aim that keeps choosing the same agent wastes the budget. Random audits, or audits aimed at a physical signal of harm, reach every agent. This is the same lesson as P2: audits must be unpredictable and cover everyone.
