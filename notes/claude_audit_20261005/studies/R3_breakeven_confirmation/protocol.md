# Protocol R3: does deterrence start at the break-even in every Fishery setting?

**Frozen:** 2026-10-07T11:31Z, before any R3 code was written or run.

This is a local protocol, not a public preregistration. It is not edited after
results exist; changes go under "Amendments". It confirms one finding that was
reached after the fact, and adds nothing new to the paper's scope (paper plan,
claim 4).

## Question

Proposition P1 (`propositions.md`) says this. A cheater facing random audits
and a flat fine stops cheating once the expected fine per step,
e = q × s × F, exceeds its gain from one step of cheating.

- **S3** confirmed P1 in the pilot setting.
- **R2** found the threshold in the unpiloted settings only after redefining
  the gain *after* seeing the data: per step on which the cheater actually
  over-takes, at the cheaters' own preferred level.

R3 tests that corrected definition prospectively, in all six Fishery
settings.

## The prediction, fixed before any search

A cheater group choosing level d compares its gain G(d) with its expected
fines e × n(d), where n(d) is the number of steps on which it over-takes. It
stops cheating at every level once e > G(d) / n(d) for all d. The predicted
threshold is therefore:

  **g\* = max over d > 0 of G(d) / n(d)**

- G(d) is the cheaters' mean net gain per cheater over complying, on the
  training contexts with no checks.
- n(d) is the mean number of over-take steps per cheater in those runs.
- d ranges over {0.25, 0.5, 0.75, 1}, and only levels with G(d) > 0 count.

g\* is computed on the training contexts **before** any search with fines. A
setting is *testable* if g\* > 0.

R2's post-hoc definition (the gain per over-take step at the cheaters'
preferred no-check level) is reported next to it. It need not be equal.

## Design

**Settings:** the six Fishery settings of the R2 grid:
- 2 or 4 greedy agents;
- regrowth rate 0.5, 0.7 or 0.9.

**Kept from S2/S3:**
- the MSY target;
- the cheating model: a cheater takes allowance + d × (request − allowance);
- audit rate q = 1/6 per agent per step (Bernoulli), detection s = 1;
- a flat fine F = e / q per catch;
- no memory and no exclusion.

**Search.** For each e, one common level d\* is chosen from
{0, 0.25, 0.5, 0.75, 1}. It maximises the cheaters' mean net payoff on 32
training contexts, with ties going to the smaller d.

**Grid.** e_j = j × 0.2 g\* for j = 0 … 10, so from 0 to 2 g\*. If no grid
value gives d\* = 0, the grid is extended to j = 20 (4 g\*). Extended values
are reported, but count as a failure for H1.

**Test.** 64 fresh test contexts at each e's d\*, plus the all-comply and
no-check runs.

**Seed bases (new):**

| Seed | Base |
| --- | --- |
| Fishery population | 1,700,000,000 |
| Audit | 1,745,000,000 |
| Detection | 1,746,000,000 |
| Training population | 1,750,000,000 |

## Hypotheses

- **R3-H1 (P1).** In every testable setting, the smallest grid e with
  d\* = 0 (call it e\*) lies within one grid step of g\*, that is between
  0.8 g\* and 1.2 g\*.
  - Falsifier: any testable setting outside that band.
  - The fraction of settings inside is also reported.
- **R3-H2 (the switch is not premature).** At the largest grid e with
  d\* > 0, the held-out cheater gain over complying is positive.
  - Falsifier: the 95% interval includes 0 or lies below it, in any
    testable setting.
- **R3-H3 (monotone).** Within each setting, d\* never increases as e rises.
  - Falsifier: any increase.

## Analysis

- 95% paired context-bootstrap intervals: 4,000 resamples, seed 20261017.
- No pooling across settings.
- e\*, g\* and d\* are reported as found.

## Engineering gates (before the full run)

1. `pytest -q tests` passes.
2. A smoke run (2 contexts × 20 steps) is byte-identical on rerun.
3. With S3's seeds, in the pilot setting (4 greedy, regrowth 0.7), the R3
   episode equals `s3.fishery_episode` for 8 sampled contexts × 2 levels.

## Run budget and stop rules

- One full run, with a 2-hour cap.
- No changes after seeing the full results. A crash may be repaired once, in
  a new directory, keeping the failed attempt.

## Amendments

(none yet)
