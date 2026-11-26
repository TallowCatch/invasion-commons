# Protocol R4: does memory make audits work in River, as in Fishery and Forest?

**Frozen:** 2026-10-09, before any R4 code was written or run.

**Why this study.** Ameer asked for consistent testing across the three games. The memory result in Figure 2(a) rests on a grid of settings in Fishery (5 testable) and Forest (4), each compared under no audits, audits without memory, and audits with memory (R2, A-H3). River was not in that grid. It has one fixed setting, and audits without memory were never run there. R4 runs River on a grid like Forest's.

This is a local protocol, not a public preregistration. Changes go under "Amendments". It adds nothing outside claim 3. It was added after the main experiment list was frozen, at Ameer's request, and it involves no language models.

## Question

With under-reporting agents in River, does a reviewer that remembers each caught lie reduce the share of rounds below half capacity:
- compared with trusting reports;
- compared with audits that correct only the current decision?

## Design

- **Game:** River, the compositional game with C1's frozen parameters (κ, λ, σ, ranges) and 80 rounds.
- **Reviewer:** joint, with a 5% chance constraint, as in C1. Its target is **quality ≥ 50 (half capacity)**, not C1's 30. In all three games, the reviewer then aims at the same line the harm is measured against. In Fishery that is the half-capacity target; in Forest it is the 5% chance constraint at plot health 10 of 20.
- **Settings (the grid, as in Forest):**
  - **greedy agents** (stressed agents who under-report): 2 (one X and one Y) or 4 (two X and two Y);
  - **regrowth multiplier** on C1's recovery rate r: 1.0 or 0.85.

  That gives 4 settings, labelled like the other games: 2g 1.0, 2g 0.85, 4g 1.0, 4g 0.85.
- **Lie:** each greedy agent reports half of its request (LIE = 0.5, as in C1, T1 and R2's fixed liars).
- **Conditions:**
  - `trust`: no audits, and reports are believed;
  - `memoryless`: one agent per round is audited at random, and its true request replaces its report for that round only;
  - `memory`: the same audits, and the reviewer also records each caught lie and corrects that agent's later reports. This is C1 Part B's random arm, using S3 Part D's memory rule, `memory_belief`.
- **Populations:** 64 test populations per setting, from a new population base (1,880,000,000) and new weather, reviewer and audit seed bases. All three conditions use the same populations, weather and audit draws within a setting.

## Outcome

- **Primary:** the share of rounds whose next-round quality is below 50.
- **Secondary:**
  - the share below 30;
  - total discharge (efficiency against the maximum sustainable discharge);
  - the share of liars ever caught.

## Hypotheses (per setting)

A setting is **testable** if the trust condition has a primary-outcome share above 0.

- **R4-H1:** `memory` gives a lower share than `memoryless`.
  - The paired difference across populations (memory − memoryless) has a 95% bootstrap interval below 0.
- **R4-H2:** `memory` gives a lower share than `trust`. Same test.
- **R4-H3 (descriptive, matching R2's finding):** `memoryless` removes less than half of the trust harm. Reported, not tested.

**Supported for the paper** if H1 holds in every testable setting, as R2's A-H3 held in 9 of 9.

## Analysis

- Paired bootstrap over populations, 4,000 resamples, seed 20261019.
- All settings are reported, including untestable ones.

## Gates

- `pytest -q tests` passes. The tests must include a check that `memory` with no caught lies equals `memoryless`, and that the 2g, 1.0 setting with C1's population base reproduces C1's per-round quality for the random arm when the target is 30.
- A smoke run uses 2 populations and 20 rounds.

## Amendments
