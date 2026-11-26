# Protocol R2: a predeclared settings grid, and a reviewer whose model is wrong

**Frozen:** 2026-10-05T12:00Z, before any R2 code was written or run.

This is a local protocol, not a public preregistration. It is not edited after
results exist; changes go under "Amendments". It was written knowing the
results of R1, S1, S1b, S2, S3 and S4 (the R1 results–18). It addresses two of the
four figure caveats (`background/caveats_assessment.md`):

- **caveat 4**: every experiment so far used two settings chosen after pilots;
- **caveat 3**: every reviewer predicts with the environment's own
  configuration.

---

## Part A: does the main pattern hold across a predeclared grid of settings?

### Question

Do the R1 reviewer findings, the S3 Part D memory finding and the S3
deterrence threshold hold in settings that were never piloted?

### Grid (declared now; every cell is reported)

| Game | Factor | Levels | Piloted level |
| --- | --- | --- | --- |
| Fishery | stress agents (of 6) | 2, 4 | 4 |
| Fishery | regrowth rate r | 0.5, 0.7, 0.9 | 0.7 |
| Harvest (`forest_co_management`) | stress agents (of 6) | 2, 4 | 2 |
| Harvest | regrowth multiplier | 0.85, 1.0 | 0.85 |

That is 6 Fishery cells and 4 Harvest cells. Exactly one cell per game was
piloted; the other 8 were never run before.

### Measures per cell

1. **R1 open loop, all 6 inspected:** usefulness loss (safe requests cut)
   and unsafe approvals (risky requests approved) for `joint`,
   `local_bounded` and `local_optimistic`. Fishery uses both targets
   (one-step and MSY); Harvest uses the one-step chance constraint.
2. **R1 closed loop, joint reviewer:** k ∈ {0, 3, 6}, fill = previous
   request, 80 steps. Outcomes: total harvest, share of steps unsafe, and
   collapse. Fishery under both targets.
3. **S3 Part D memory (Harvest, and Fishery MSY):** trust, audit without
   memory, audit with memory. Two random audits per step; misreporters are
   the stress agents and under-report by half (fixed) or by a fresh random
   fraction with mean one half (noisy). Outcome: share of steps unsafe
   (Harvest) or breaking the target (Fishery).
4. **Deterrence threshold (Fishery cells only), testing Proposition P1**
   (the propositions note): first measure each cell's no-check gain per cheater per step,
   G (cheaters at level 0.75; the stress agents are the cheaters). Then run
   S3 Part A's search with q = 1/6 and perfect audits over 11 values of
   e = q × F evenly spaced in [0, 2 × G], and record e*, the first e with
   chosen level 0.

### Hypotheses (confirmatory family A)

- **A-H1.** In every Harvest cell, joint review approves under 2% of risky
  requests, and optimistic local approves more risky requests than joint.
  - Falsifier: any cell where either fails.
- **A-H2.** In every Fishery cell, under the one-step target, the joint
  reviewer at k = 6 gives less long-run harvest than under the MSY target.
  - Falsifier: the paired interval for (MSY − one-step) includes or is below
    0 in any cell.
- **A-H3.** In every cell with fixed liars, memory has fewer unsafe or
  target-breaking steps than audit without memory.
  - Falsifier: the paired interval includes or exceeds 0 in any cell.
- **A-H4 (Proposition P1).** In every Fishery cell, e* lies within one grid
  step of the break-even G.
  - Falsifier: any cell where |e* − G| is more than one grid step.

Cells where a measure cannot be computed (no risky requests, or no gain
from cheating) are reported as "not testable", not as passes.

---

## Part B: a reviewer whose model is wrong

### Question

How fast do each reviewer's errors grow when its model of the game is wrong,
estimated from data, or wrong in form? Does the advantage of joint review
survive?

### Cells

The piloted cell of each game, plus one unpiloted cell declared now:
Fishery (2 stress agents, r = 0.9) and Harvest (4 stress agents, multiplier
1.0).

### Reviewer model conditions

The environment always uses its true configuration. Only the reviewer's copy
changes.

| Code | Harvest | Fishery |
| --- | --- | --- |
| `exact` | true model | true model |
| `noise_low` | weather noise × 0.5 | — (no noise in Fishery) |
| `noise_high` | weather noise × 2 | — |
| `regen_low` | regrowth × 0.75 | regrowth × 0.75 |
| `regen_high` | regrowth × 1.25 | regrowth × 1.25 (capped at 1) |
| `K_low` | — | carrying capacity × 0.75 |
| `K_high` | — | carrying capacity × 1.25 |
| `learned` | starts at regrowth × 1.25 and noise × 0.5 (optimistic). After each step it re-estimates regrowth by least squares on the logistic form from all transitions seen, and noise from the residual standard deviation (once 5 steps are seen). | starts at regrowth × 1.25 and carrying capacity × 1.25. Re-estimates both by least squares on the logistic form after each step (once 5 steps are seen). |
| `allee` (Fishery only) | — | The **environment** has critical depensation: growth = r R (1 − R/K)(R/A − 1) with A = 20. The reviewer keeps believing the logistic model. |

### Measures

- Open loop at k = 6 for all three reviewers: usefulness loss and unsafe
  approvals, scored against the *true* model's labels.
- Closed loop for the joint reviewer at k = 6 (fill = previous): total
  harvest, share of steps unsafe, collapse. Fishery under both targets.
- `learned`: unsafe-approval rate in steps 0–19 vs steps 60–79 (closed loop).

### Hypotheses (confirmatory family B)

- **B-H1.** Under `noise_low`, joint review's unsafe-approval rate in Harvest
  rises above 2% (R1: 0.6%). Under `noise_high`, its usefulness loss rises by
  at least 10 points over `exact`.
- **B-H2.** In every condition and cell, joint approves fewer risky requests
  than optimistic local.
  - Falsifier: any condition where joint's rate is at least optimistic
    local's.
- **B-H3.** In Fishery, `regen_low` / `regen_high` change the joint reviewer's
  decisions under the one-step target but **not** under the MSY target
  (whose check does not use the regrowth rate). `K_low` / `K_high` change
  decisions under both targets.
  - Falsifier: any MSY decision that differs between `exact` and a regrowth
    condition. This hypothesis is true by construction and is included as a
    code check.
- **B-H4.** `learned`: the unsafe-approval rate in steps 60–79 is lower than
  in steps 0–19, and long-run harvest is within 5% of `exact`.
- **B-H5 (`allee`, Fishery).** Under the one-step target, the joint reviewer
  at k = 6 lets the stock collapse in at least 25% of contexts. Under the MSY
  target, collapse is below 5% (MSY keeps the stock at 50, above A = 20).

---

## Common design

- **Independent unit:** the context (population + seeds); 64 test contexts
  per cell.
- **Seed bases (new):** Fishery population 1,200,000,000; Harvest population
  1,210,000,000; weather 1,220,000,000; reviewer 1,230,000,000; reference
  1,240,000,000; audit 1,245,000,000; deterrence training population
  1,250,000,000 (32 training contexts).
- **Analysis:** paired context-bootstrap contrasts, 4,000 resamples, seed
  20261012, 95% percentile intervals.
- **Multiplicity:** families A and B are each corrected with Holm's method at
  0.05 overall, using bootstrap p-values (the share of resamples on the wrong
  side of 0, doubled). Uncorrected intervals are also reported.
- **Exploratory:** everything not listed as a hypothesis is labelled
  exploratory.

## Engineering gates (before the full run)

1. `pytest -q tests` passes, with new tests for:
   - reviewer configuration copies;
   - the least-squares estimator (recovers r and K from noise-free logistic
     data);
   - the Allee transition with A = 0 equals the existing transition.
2. **Replication gate.** In the piloted cells with `exact`, the R1 code path
   reproduces R1's decisions for 4 sampled contexts when given R1's seeds.
3. A smoke run (2 contexts, 20 steps) is byte-identical on rerun.

## Run budget and stop rules

- One full run per part, with no reruns after seeing results.
- A crash may be repaired once, in a new directory, keeping the failed
  attempt.
- Gate failures may be fixed before the full run.

## Amendments

### Amendment 1 (2026-10-05T14:32Z, before any full run)

Written after the engineering gates and two smoke runs (2 contexts, 20
steps), before any full run. Nothing above this heading has changed.
Hypotheses, cells, conditions, seeds bases, bootstrap and Holm families are
as declared. These are choices the protocol left open.

**Gates.**
- **Replication gate.** R1's per-decision outputs were never saved; only its
  aggregate tables were. So, as in S3 Amendment 1, the gate compares against
  **R1's unchanged code run on this machine** with R1's seeds.
  - Contexts 26, 29, 37 and 53 were drawn with seed 20261012.
  - On these, R2's code with `exact` gives the same decisions, labels and
    episode outcomes as R1's code. This holds for every Harvest and Fishery
    closed-loop run (k = 0, 3, 6, fill = previous, both Fishery targets)
    and every open-loop decision at k = 6.
  - Extra check: R1's code was rerun in full here (64 contexts). It
    reproduces all 54 rows of R1's saved open-loop and closed-loop decision
    tables exactly.
- **Smoke run.** All output files of both parts, including `manifest.json`,
  are byte-identical on rerun. Wall-clock time now goes in a separate
  `timing.json`, which is excluded from the comparison.

**Part A.**
- **Open loop** follows R1:
  - requests come from one no-reviewer run per context, scored only at
    steps whose state is safe;
  - k = 6, so every request is inspected and the fill rule plays no role;
  - the reviewer seed is R1's open-loop seed with fill = "max";
  - Harvest labels use 4,000 weather draws (2,000 in the closed loop);
    the reviewer always uses 400.
- **Memory (S3 Part D):**
  - the misreporters are the cell's stress agents;
  - fixed liars under-report by 0.5;
  - noisy liars draw a fresh uniform [0, 1) fraction for each agent and
    step, from a new seed base of 1,247,000,000 (the protocol names none);
  - two random audits per step, no sanction; the joint reviewer acts on the
    reports;
  - the Harvest outcome is the share of the 80 steps that are unsafe
    (steps after a garden failure count as unsafe);
  - the Fishery outcome is the share of steps with a safe state whose
    executed action breaks the MSY target, as in S3 Part D.
- **Deterrence:**
  - "Gain per step" means per step of the 80-step horizon:
    (cheater payoff at d = 0.75 − payoff at d = 0) ÷ number of cheaters ÷ 80.
    This is the mean over the 32 training contexts, with allowances under
    the MSY target and no checks. It is the same reading as the S3 results
    (28.5 ÷ 80).
  - The grid is e_j = j × 2G / 10 for j = 0, …, 10, with fine F = e / q.
  - At each e, S3 Part A's search over d ∈ {0, 0.25, 0.5, 0.75, 1} runs on
    the same 32 training contexts, with ties going to the smaller d.
  - Detection is perfect, so the detect seed (1,246,000,000) is unused.
  - If no grid value gives d* = 0, A-H4 **fails** for that cell (e* > 2G).
  - If G ≤ 0, the cell is "not testable".
- **Deterrence, exploratory additions** (never used for A-H4):
  - the cheaters' best level with no checks, d_allow;
  - the break-even per step on which a cheater over-takes, at both 0.75
    and d_allow;
  - when no grid value deters, the same search continues past 2G in the
    same step, up to 10G, to find where deterrence actually starts.
  - Reason: in the smoke run, the cheaters in some unpiloted cells
    preferred d = 1 to 0.75, or over-took on only some steps. In those
    cells G at 0.75 per horizon step is not the true break-even.

**Part B.**
- **Open loop.** Requests and states come from no-reviewer runs.
  - For `allee`, the no-reviewer run uses the Allee environment, and labels
    use the Allee model.
  - For `learned`, the reviewer uses the model it learned from that
    no-reviewer trajectory up to the step being judged.
- **`learned`, Fishery:**
  - estimation is least squares without intercept on S' − R = rR − (r/K)R²;
  - transitions that end in the environment's collapse reset are skipped;
  - if the residual stock never varies, r ≤ 0, or the fitted curve does not
    bend downward, the previous model is kept;
  - r is capped at 1.
- **`learned`, Harvest:**
  - the patch maximum is known;
  - r is fitted by least squares on all patches' transitions, including
    patches clipped at 0 or at the maximum (the reviewer does not model
    clipping), and r is clipped to [0, 1];
  - the noise is the residual standard deviation (ddof = 1).
- **Both `learned` cases:** the model is re-estimated after every observed
  step, once 5 steps have been seen.
- **B-H1, B-H2:** open loop at k = 6.
- **B-H4 windows:** closed loop, joint, k = 6, steps with a safe state only.
  "Within 5%" is tested as 0.05 − |mean harvest under `learned` ÷ mean
  harvest under `exact` − 1| > 0.

**Analysis.**
- Rates are per-context ratios of sums, resampled by context. All tests
  with the same number of contexts use the same resample matrix.
- Holm's method runs over the component tests in each family (cell ×
  component). A-H4 and B-H3 are exact checks with no p-value and are left
  out of Holm.
- A hypothesis fails if any testable component fails.
- Each hypothesis gets four verdicts. The **primary** verdict requires the
  estimate on the predicted side and a Holm-adjusted p below 0.05. The other
  three are reported alongside:
  - the uncorrected 95% interval entirely on the predicted side;
  - the point estimate alone on the predicted side;
  - the falsifier read literally: the interval for A-H2 and A-H3, the point
    estimate for the rest.
- B-H3's verdict is its declared falsifier only: no MSY decision, open or
  closed loop, differs between `exact` and a regrowth condition. Its other
  statements are reported on a separate secondary line. These are: regrowth
  changes one-step decisions, and K changes decisions under both targets.

**Runtime.** No reduction is needed: the full run is estimated at about 20
to 40 minutes on one core. All Monte Carlo draws are as declared. Part A and
Part B run one after the other in one background script, into `partA/` and
`partB/`, and are then analysed together.
