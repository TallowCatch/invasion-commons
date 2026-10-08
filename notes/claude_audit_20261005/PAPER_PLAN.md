# Paper plan (7 October 2026)

The full plan is a shared doc:
https://claude.ai/code/artifact/af1a091c-1c78-49da-aed2-442f9cdba821

This file is the short version, kept in the repository so the plan travels
with the code.

**Working title:** *What makes audits work: oversight with a limited audit
budget when agents can hide what they take from a shared resource.*

## The six claims

| # | Claim | Evidence | Status |
| --- | --- | --- | --- |
| 1 | The goal matters more than inspection: an MSY target beats the one-step line wherever the resource is under pressure | R1, R2 | tested |
| 2 | A wrong model breaks an exact reviewer; a learning reviewer recovers; MSY protects against a wrong model form | R2 Part B | tested |
| 3 | Audits help only through consequences: memory or sanctions | S1b, S3 Part D, R2 (9 of 9 settings) | tested |
| 4 | Deterrence starts at the expected-fine break-even; memory plus a fine halves the audit rate needed; this survives richer attackers | S3, S4, S5, R2, **R3** (5 of 5 settings, prospective) | tested |
| 5 | How audits are aimed matters: audits must be unpredictable and reach every agent; a known schedule or a fixed report-based aim leaves agents unchecked | S3 Part C, **T1** (Fishery, River; 25% of liars ever caught under report-aiming) | tested |
| 6 | LLM agents follow the same logic: those that over-take stop only when the expected fine exceeds their gain | L1 pilot, **L2** (H1 in 3 of 3 frozen families; 0% over-taking at e = 6 t, 51–92% at e ≤ 1.33 t for gpt-oss and Nemotron) | tested (memory cell H3 running) |

## The only remaining experiments

Each one gets a protocol frozen before the code, one full run and a results
note.

| Experiment | What it is | Notes (protocol → results) |
| --- | --- | --- |
| R3 | Confirm the break-even, measured per over-take step, in all 6 Fishery settings | 28 → 31 |
| T1 | Random vs report-targeted vs signal-targeted audits against under-reporters, in 3 games | 29 → 32 |
| L2 | LLM agents, 3 frozen families + Mistral (added) × fines 0–36 t, plus memory, plus wording controls | protocol and results in `studies/L2_llm_agents/` (done except EM) |

## Parked for the next paper

Anything not needed for claims 1–6 goes here:

- capability ladders;
- more games;
- better RL attackers;
- Harvest deterrence;
- hedging reviewers;
- LLM free-text talk and collusion;
- the Act 1 governance studies (appendix only).

## Phases

1. Confirm: R3 and T1.
2. L2.
3. Figures and draft (`paper/paper_v6`).
4. Freeze the repository:
   - update `00_README`;
   - package the data bundle;
   - run the tests;
   - push.

## Exhibits

Draft figures and tables for claims 1–5 are in `paper/paper_v6/` (`exhibits_preview.pdf`, README).
- **Game names in the paper:** Fishery, Forest (the code's Harvest) and River (C1).
- **Composite figures:** Fig. 2 covers the reviewer's target and model; Fig. 3 covers what makes audits work.
- **Fig. 4 (LLM agents, L2)** is drawn: over-taking and harm against the expected fine, the wording controls, and what an over-take is (post hoc). A memory panel can be added when EM finishes.
