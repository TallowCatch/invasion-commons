# Protocol R5: the Fishery memory test on the same 2×2 grid as Forest and River

**Frozen:** 2026-10-09, before any R5 code was written or run.

**Why this study.** Ameer asked for one design across the three games. The memory test in Figure 2(a) currently uses different grids:
- Fishery uses R2's absolute regrowth rates (0.5, 0.7, 0.9) × 2 or 4 greedy agents, which gives 5 testable settings.
- Forest (R2) and River (R4) use a 2×2 grid: 2 or 4 greedy agents × a regrowth *multiplier* of 1.0 or 0.85 on each game's base rate.

R5 runs Fishery on that same 2×2 grid, so every game has 4 settings with identical labels.
- R2's 5-setting Fishery result stays as reported, and moves to the supplement as a wider robustness check.
- This protocol was added after the experiment freeze, at Ameer's request. It uses programmed agents only.

This is a local protocol, not a public preregistration. Changes go under "Amendments".

## Design

- **Game:** Fishery, R2's setup (`fishery_setup_r2`):
  - 6 agents, stock starting at 70 of 100, 80 rounds;
  - the joint reviewer with the half-capacity (MSY) target.
- **Grid:**
  - greedy (under-reporting) agents: 2 or 4;
  - regrowth multiplier 1.0 or 0.85 on Fishery's base rate r = 0.7, so r = 0.7 or 0.595.
  - Labels: 2g 1.0, 2g 0.85, 4g 1.0, 4g 0.85.
- **Lie and audits:**
  - fixed liars, who report half of their request (LIE = 0.5);
  - two random audits per round (`rand2`).
  - This is exactly R2's memory test (`memory_episode`).
- **Conditions:** trust (no audits), audits without memory, audits with memory (R2's `MEMORY_MODES`).
- **Populations and seeds:**
  - 64 test populations per setting, from new seed bases (population 1,890,000,000; reviewer 1,891,000,000; reference 1,892,000,000; audit 1,893,000,000).
  - All three conditions share populations and audit draws within a setting.

## Outcome and hypotheses

- **Outcome:** R2's harm measure for Fishery. It is the pooled share of rounds in which the executed catch breaks the MSY target (`exec_risky / scored_steps`), the quantity plotted in Figure 2(a).
- **Testable:** a setting is testable if the trust share is above 0.
- **R5-H1:** memory gives a lower share than audits without memory.
  - The paired difference over populations has a 95% bootstrap interval below 0.
- **R5-H2:** memory gives a lower share than trust. Same test.
- **R5-H3** (descriptive): audits without memory remove less than half of the trust harm.
- **Supported** if H1 holds in every testable setting.

## Analysis

Paired bootstrap over populations of the per-population harm share, 4,000 resamples, seed 20261019.

## Gates

- `pytest -q tests` passes.
- **Reproduction test.** With R2's seeds, R5's episode function reproduces R2's saved memory episodes (`exec_risky`, `scored_steps`) for setting 4g at r = 0.7, contexts 0–1, all three conditions, fixed liars.
- A smoke run uses 2 populations and 20 rounds.

## Amendments
