# Protocol T1: do audits aimed at the largest report miss under-reporters?

**Frozen:** 2026-10-07T11:31Z, before any T1 code was written or run.

This is a local protocol, not a public preregistration. It is not edited after
results exist; changes go under "Amendments". It confirms one finding that was
reached after the fact (C1 Part B; Conjecture C1 in `propositions.md`), for
claim 5 of the paper plan. It adds nothing else.

## Question

A natural way to aim a limited audit budget is at the agent whose request
matters most, which in practice means its *report*. But an under-reporter
makes its own report small.

- **C1 (River)** found that report-aimed audits gave more unsafe steps than
  random ones: +3.4 points [1.8, 5.3].
- **S1 (Forest)** found that audits aimed by a *physical* signal (a plot doing
  worse than predicted) helped.

T1 tests both on fresh seeds, in all three games.

## Design

**Games and settings:**

| Game | Setting | Harm measure |
| --- | --- | --- |
| Fishery | 4 greedy agents, regrowth 0.7, MSY target | steps breaking the MSY limit |
| Forest | the code's Harvest game; 2 greedy agents, regrowth × 0.85, 5% chance constraint | share of steps unsafe |
| River | C1's frozen parameters, compositional game | share of steps unsafe |

In each game, the greedy agents under-report by half (d = 0.5, a fixed lie),
as in S1 and C1. The reviewer is joint and **remembers each caught lie** (the
S3 Part D memory rule, which C1 also used). There are no fines and no
exclusion. One audit per step.

**Arms:**

| Arm | Who is audited |
| --- | --- |
| `trust` | nobody (reference) |
| `random` | one agent chosen uniformly at random each step |
| `report` | the agent whose believed request (its report, corrected by memory) is largest. In River this is C1's rule: the largest effect on predicted damage. |
| `signal` (Forest only) | S1's `targ1` rule: the agent whose plot fell furthest below the reviewer's prediction (decaying sum, factor 0.8) |

Fishery and River have no per-agent physical signal: the stock and the river
are shared. So `signal` cannot be defined there.

**Contexts:** 64 test contexts per game. Arms are paired by context.

**Seed bases (new):**

| Seed | Base |
| --- | --- |
| Fishery population | 1,800,000,000 |
| Forest population | 1,810,000,000 |
| Forest weather | 1,820,000,000 |
| Reviewer | 1,830,000,000 |
| Reference | 1,840,000,000 |
| Audit | 1,845,000,000 |
| River | C1's seed dictionary with every base shifted to 1,860,000,000 + (C1 base − 1,400,000,000) |

**Outcomes:**
- harm, as defined per game above;
- the share of audits that land on a liar (mechanism);
- liars caught per context;
- honest agents' harvest.

## Prediction stated now

Whether report-aiming fails depends on how small a liar's halved report is
compared with honest agents' reports.

- **River:** liars' halved reports are usually not the largest (C1).
- **Fishery:** greedy requests are 0.75–0.95 when the stock is high, so their
  halved reports (about 0.38–0.48) can still exceed honest requests
  (0.2–0.4). Report-aiming may therefore still find liars.

So H1 is a real test, not a foregone conclusion.

## Hypotheses

- **T1-H1 (claim 5).** `report` gives more harm than `random`, with the
  paired 95% interval of the difference above 0, in at least 2 of the 3
  games.
  - Falsifier: fewer than 2 games.
- **T1-H2 (mechanism).** The share of audits landing on a liar is lower
  under `report` than under `random`, in at least 2 of the 3 games.
  - Falsifier: fewer than 2.
- **T1-H3 (Forest).** `signal` gives no more harm than `random`.
  - Falsifier: the difference (signal − random) has a 95% interval entirely
    above 0.

## Analysis

- 95% paired context-bootstrap intervals: 4,000 resamples, seed 20261018.
- No pooling across games.

## Engineering gates (before the full run)

1. `pytest -q tests` passes.
2. A smoke run is byte-identical on rerun.
3. **Code equivalence on this machine:**
   - With S1's seeds, T1's Forest `random` arm with memory equals S3
     Part D's `harvest_episode_d(..., "rand1", "memory", "fixed")`, and
     T1's Forest `signal` arm equals S3 Part D's `targ1`, for 8 sampled
     contexts.
   - With C1's own seeds, T1's River arms equal C1 Part B's `episode`, for
     8 sampled contexts.

## Run budget and stop rules

- One full run, with a 2-hour cap.
- No changes after seeing the full results. A crash may be repaired once, in
  a new directory, keeping the failed attempt.

## Amendments

(none yet)
