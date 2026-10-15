# Results R3: deterrence starts at the break-even in every testable Fishery setting

**Protocol:** `protocol.md` in this folder, frozen at 2026-10-07T11:31Z (commit `49aef43`) before any R3 code existed. The code and tests were committed as `90ce85b` after the gates and before the full run. There are no amendments.

**Run:** `results/runs/claude_r3_v1/`, one full run in about 1 minute. Tables are in `notes/claude_audit_20261005/runs/claude_r3_v1/`.

**Status of the numbers.** Verified. The intervals are 95% paired context-bootstrap intervals (4,000 resamples).

## Question

A cheater facing random audits and a flat fine should stop once the expected fine per step, e, exceeds g\*, its gain from one step of over-taking (Proposition P1). R2 matched this only after redefining g in light of the data. R3 tests that definition prospectively, in all six Fishery settings.

## What was done

- **The prediction came first.** For each setting, g\* was computed on 32 training populations with no checks:

  g\* = max over d of [gain from cheating at level d] ÷ [steps over-taken at level d]

- **Then the search.** For each expected fine on a grid from 0 to 2 g\* in steps of 0.2 g\*, the cheaters chose their best level on the same 32 training populations.
- **Fixed throughout:** audit rate 1/6, perfect detection, MSY target.
- **Test:** 64 new populations at each chosen level.

## What came out

| Setting | Predicted g\* | Observed threshold e\* | e\* ÷ g\* | Level chosen below e\* |
| --- | ---: | ---: | ---: | ---: |
| 2 greedy, regrowth 0.5 | 1.20 | 1.44 | 1.2 | 1.0 |
| 2 greedy, regrowth 0.7 | 0.61 | 0.73 | 1.2 | 1.0 |
| 2 greedy, regrowth 0.9 | 0 | — | not testable | cheating never pays |
| 4 greedy, regrowth 0.5 | 0.29 | 0.34 | 1.2 | 0.5 |
| 4 greedy, regrowth 0.7 (pilot) | 0.36 | 0.43 | 1.2 | 0.75 |
| 4 greedy, regrowth 0.9 | 0.19 | 0.19 | 1.0 | 0.75 |

In every setting the choice was all or nothing. The group cheated at one level for every e below the threshold, and stopped completely above it.

**Hypotheses:**
- **R3-H1 (P1): supported, 5 of 5 testable settings.** The threshold falls within one grid step of the prediction.
  - In four settings the group still cheated at exactly e = g\* and stopped at the next grid value (1.2 g\*).
  - At exactly e = g\*, cheating and complying pay the same on average, so either choice is consistent with P1.
- **R3-H2 (the switch is not premature): falsified as stated**, in 4 of 5 settings. The held-out gain at the last cheating level has an interval that includes 0. For example, the pilot setting gives +0.04 [−1.70, +1.78].
  - That last level is exactly e = g\* in those four settings, where the gain is zero by definition. The hypothesis was badly posed; this is not evidence against P1.
  - In the fifth setting (the last cheating level at 0.8 g\*) the gain is +2.04 [+1.57, +2.56], as predicted.
- **R3-H3 (the chosen level never rises with the fine): supported** in all 5.

**R2's definition agreed with this one.** The gain per over-take step at the cheaters' preferred no-check level gave the same g\* in all five testable settings. This time the definition was fixed before seeing any results.

## What it means

**What it shows:**
- The break-even rule predicts the deterrence threshold to within one grid step (20% of g\*), across settings where the predicted threshold varies six-fold (0.19 to 1.20).
- The prediction comes only from no-check runs on training populations.
- Claim 4's threshold is now confirmed prospectively, not only after the fact.

**What it does not show:**
- cheaters who are not risk-neutral;
- fines that grow with the size of the over-take;
- any setting outside Fishery.

**Design caveat.** The grid step is 0.2 g\*, so "within one step" means within 20%. A finer grid near g\* would locate the threshold more precisely, but cannot separate e = g\* from just above it: at that point the two choices pay the same.

## What we now believe

The deterrence threshold is the per-over-take break-even. We are confident about this for risk-neutral fixed-rule cheaters in Fishery.
