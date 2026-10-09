# Results R4: memory makes audits work in River, as in Fishery and Forest

**Protocol:** `protocol.md` in this folder, frozen on 2026-10-09 before any code (commit `6cc5578`). The code and gate tests were committed before the full run. There are no amendments.

**Run:** `results/runs/claude_r4_v1/`, one full run in 7 seconds: 4 settings × 64 populations × 3 conditions. The summary is in `notes/claude_audit_20261005/runs/claude_r4_v1/r4_summary.json`.

**Gates:** all passed.
- `pytest` passes: 360 passed, 11 skipped.
- With C1's seeds, 2 greedy agents, regrowth 1.0 and the 30 line, R4 reproduces C1's River quality round by round.
- With no lies, memory equals memoryless.
- The smoke run passed.

**Status of the numbers:** verified (`run_r4_river_grid.py`). Intervals are 95% paired bootstrap intervals over the 64 populations (4,000 resamples).

## Question

With under-reporting agents in River, does a reviewer that remembers each caught lie reduce the share of rounds below half capacity, compared with trusting reports and with audits that correct only the current decision?

## What was done

- River with C1's frozen parameters and 80 rounds.
- The joint reviewer aims at quality ≥ 50 (half capacity) with a 5% chance constraint, so the reviewer's target and the harm line are the same, as in Fishery and Forest.
- Grid:
  - 2 or 4 greedy agents, who under-report by half (two-agent: one of each reagent; four-agent: two of each);
  - River's recovery rate × 1.0 or × 0.85.
- Conditions:
  - no audits;
  - one random audit per round without memory;
  - the same audits with memory (S3 Part D's rule).

## What came out

**Share of rounds below half capacity:**

| Setting | No audits | Audits without memory | Audits with memory | Memory − without memory [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| 2g 1.0 | 38.4% | 32.6% | **2.8%** | −29.8 points [−33.0, −26.5] |
| 2g 0.85 | 34.9% | 30.2% | **2.9%** | −27.3 points [−30.3, −24.3] |
| 4g 1.0 | 66.2% | 58.7% | **4.9%** | −53.7 points [−57.5, −49.7] |
| 4g 0.85 | 66.1% | 57.9% | **4.8%** | −53.2 points [−56.6, −49.5] |

**Hypotheses:**
- **R4-H1** (memory below memoryless): holds in **4 of 4** settings.
- **R4-H2** (memory below trust): holds in **4 of 4**.
- **R4-H3** (descriptive): audits without memory removed only 11–15% of the no-audit harm.

**Secondary outcomes:**
- Below quality 30 (the old line), memory gave 0.1–1.5% of rounds, against 1.6–21.1% with no audits.
- Every liar was caught at least once in both audited conditions (100%). So the difference is what the reviewer does with a catch, not whether liars are caught.

## What it means

- **Claim 3 now holds in all three games.** With consequences (memory), audits remove most of the harm. Without them, audits barely help. Across 13 testable settings, memory beats audits without memory in 13 of 13: 9 from R2 in Fishery and Forest, and 4 from R4 in River.
- **Why C1's River result looked weak at half capacity** (53% → 39% of rounds below 50): C1's reviewer aimed at quality 30, so it allowed quality to sit between 30 and 50 by design. When the reviewer aims at the half-capacity line, memory removes almost all of the harm. This is the lesson of R1: the reviewer's goal matters as much as its information.

## Design caveats

- R4 changes the River reviewer's target to 50, so its numbers are not directly comparable with C1's 30-line results; C1's results stand as reported.
- The 4-greedy-agent populations are drawn with their own generator (two of each reagent type). The two-agent populations are exactly C1's construction, from a new population base.
- Regrowth 0.85 gave slightly less harm than 1.0 with two greedy agents (34.9% against 38.4% with no audits). The reviewer cuts more when recovery is slower, which offsets some of the harm. This is descriptive and was not tested.
