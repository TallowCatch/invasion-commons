# Results R2: a predeclared settings grid, and a reviewer whose model is wrong

**Protocol:** `studies/R2_robustness_and_reviewer_model/protocol.md`, frozen at
12:00Z before any code. Amendment 1 (14:32Z) was added before the full run.

**Run:**
- `results/runs/claude_r2_v1/{partA,partB,analysis}`: one full run, no
  reruns, about 20 minutes on one core.
- Gates passed:
  - 333 tests;
  - R2's `exact` path reproduces R1's code decision by decision on 4
    sampled contexts;
  - a full R1 rerun reproduces all 54 rows of R1's saved tables;
  - the smoke run is byte-identical on rerun.

**Status of the numbers:** verified, read from
`analysis/hypothesis_tests.csv`, `A_*.csv` and `B_*.csv`. Intervals are 95%
paired context-bootstrap intervals (4,000 resamples). Holm correction is
applied within families A and B. **[post hoc]** marks readings made after
seeing the data.

## Question

1. Do the main findings hold in settings that were never piloted?
2. What happens when the reviewer's model of the game is wrong?

## What was done

**Part A.** A grid of 10 settings, 8 never piloted:
- Fishery: 2 or 4 stress agents × regrowth 0.5, 0.7 or 0.9;
- Harvest: 2 or 4 stress agents × regrowth multiplier 0.85 or 1.0.

Per setting it reran:
- R1's reviewer comparison;
- S3 Part D's memory comparison;
- in Fishery, S3's deterrence search over 11 fine levels.

**Part B.** The reviewer's copy of the model was made wrong in several ways:
- weather noise ×0.5 or ×2;
- regrowth ×0.75 or ×1.25;
- carrying capacity ×0.75 or ×1.25;
- learned online, starting from an optimistic model;
- the wrong form: the environment has a hidden collapse point (an Allee
  effect, A = 20) that the reviewer does not know about.

Part B used the piloted setting of each game and one unpiloted setting. There
were 64 test contexts per cell.

## What came out

### Part A (settings grid)

| Hypothesis | Settings passed | Note |
| --- | --- | --- |
| A-H1: joint < 2% risky approved, optimistic local above joint (Harvest) | 3 of 4 | All 4 at the point estimate. In `harvest_s2_m1.0` the interval crosses 2%: few risky cases. |
| A-H2: MSY target beats the one-step line on harvest (Fishery) | 5 of 6 | Fails in `fishery_s2_r0.9` only, where both give 1,380.3: the one-step line never binds. |
| A-H3: memory safer than no memory | **9 of 9** testable | `fishery_s2_r0.9` not testable: nothing is ever unsafe. |
| A-H4: deterrence starts at break-even G | 1 of 4 testable | The definition of G was wrong; see below. |

**Fishery harvest, MSY target vs one-step line** (joint reviewer, all 6
inspected):

| Setting | MSY | One-step |
| --- | ---: | ---: |
| 2 stress, r = 0.5 | 1,003 | 614 |
| 4 stress, r = 0.5 | 1,000 | 391 |
| 4 stress, r = 0.7 (piloted) | 1,394 | 544 |

**Deterrence.** The protocol set G at a cheating level of 0.75, per horizon
step. In unpiloted settings the cheaters preferred another level, or did not
over-take every step.

**[post hoc]** Measured as the gain per over-take step at the cheaters' own
preferred level, the threshold matched in all 3 computable settings (see
`propositions.md`, P1):

| Setting | e* | g |
| --- | ---: | ---: |
| 2 stress, r = 0.5 | 1.38 | 1.29 |
| 4 stress, r = 0.7 | 0.373 | 0.373 |
| 4 stress, r = 0.9 | 0.245 | 0.227 |

### Part B (reviewer model)

**Harvest, open loop, all 6 inspected.** Percentages are risky requests
approved / safe requests cut.

| Reviewer model | Joint (piloted) | Optimistic local (piloted) | Joint (unpiloted) |
| --- | --- | --- | --- |
| exact | 0.9% / 0.3% | 9.1% / 0.1% | 0% / 0% |
| noise × 0.5 | **25.6%** / 0% | 40.1% / 0% | 7.8% / 0% |
| noise × 2 | 0% / **25.0%** | 0.6% / 21.0% | 0% / 78.6% |
| regrowth × 1.25 | **82.3%** / 0% | 90.2% / 0% | 58.4% / 0% |
| regrowth × 0.75 | 0% / 41.6% | 0% / 37.9% | 0% / 100% |
| learned | 9.2% / 0.2% | 18.9% / 0.1% | 62.3% / 0% |

**Closed loop** (joint reviewer, all 6 inspected):
- Harvest with regrowth × 1.25: 18% of steps unsafe, against 0.5% with the
  true model.
- Fishery one-step target with regrowth × 1.25: the stock collapsed in 41%
  of contexts.

**The learned reviewer:**
- Harvest risky approvals fell from 16.4% (steps 0–19) to 2.5% (steps 60–79)
  in the piloted setting, and from 4.3% to 0% in the unpiloted one.
- Harvest was within 1% of the exact reviewer.
- The open-loop numbers above use a model learned from a no-reviewer run,
  which is why they look worse.

**The Allee environment** (the reviewer believes the logistic model):

| Setting | Target | Collapsed | Harvest |
| --- | --- | ---: | ---: |
| piloted Fishery | one-step | **78%** | 520 |
| piloted Fishery | MSY | **0%** | 1,902 |
| unpiloted (2 stress agents, r = 0.9) | either | 0% | — |

In the unpiloted setting nothing pushes the stock towards the collapse point.

**Hypotheses:**
- **B-H1 (noise errors):** holds at the point estimate in both cells. In the
  unpiloted cell the noise_low part fails after Holm (7.8% [0.7, 11.6]).
- **B-H2 (joint below optimistic local in every condition):** fails in one
  component only. With regrowth × 0.75, both rules approve 0% and tie.
- **B-H3 (the MSY decision ignores regrowth):** passes. This was a code check.
- **B-H4 (learning improves):** passes in both Harvest cells. In Fishery
  nothing risky was ever approved, so "late below early" cannot hold.
- **B-H5 (Allee):** passes in the piloted cell. It fails in the unpiloted
  cell, where nothing threatens the stock.

## What it means

**What it shows:**
- **The memory result is robust.** It held in every testable setting.
- **The target result is real but conditional.** The MSY target matters
  wherever the resource is under pressure, and not where it is not.
- **Joint review's exactness depended on knowing the model.** Moderate errors
  in noise or regrowth take its missed-harm rate from under 1% to 26–82%.
  Joint still does better than optimistic local under the same error.
- **A reviewer that learns the model recovers within about 60 steps.**
- **The MSY target makes the reviewer robust to one kind of wrong model
  form**, because it keeps the stock far from the hidden collapse point.

**What it does not show:**
- errors in more than one parameter at once;
- a reviewer that hedges against its own model uncertainty;
- learned models in settings that change over time.

## Design caveats

- **Several hypotheses had few risky cases in unpiloted settings.** These
  are flagged as descriptive only when fewer than 20 contexts had a risky
  case.
- **A-H4's failure comes from a wrongly stated quantity in the protocol.** The
  corrected reading is post hoc.
- **The learned Harvest noise estimate is probably biased low** by clipping
  at 0 and at the patch maximum.
- **The Allee growth form scales growth by (R/A − 1).** Above R = 2A = 40,
  growth is faster than logistic, which is why MSY harvest under Allee
  (1,902) exceeds the exact case (1,394). The collapse contrast does not
  depend on this, but the harvest numbers do. A form that only removes growth
  below A would be a cleaner test.
