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
| 4 | Deterrence starts at the expected-fine break-even; memory plus a fine halves the audit rate needed; this survives richer attackers | S3, S4, S5, R2 | needs **R3** |
| 5 | How audits are aimed matters: a known schedule is worse than none; report-targeting misses under-reporters | S3 Part C, River Part B | needs **T1** |
| 6 | LLM agents follow the same logic | L1 pilot | needs **L2** |

## The only remaining experiments

Each one gets a protocol frozen before the code, one full run and a results
note.

| Experiment | What it is | Notes (protocol → results) |
| --- | --- | --- |
| R3 | Confirm the break-even, measured per over-take step, in all 6 Fishery settings | 28 → 31 |
| T1 | Random vs report-targeted vs signal-targeted audits against under-reporters, in 3 games | 29 → 32 |
| L2 | LLM agents, 2 model families × fines 0–36 t, plus memory, plus framing controls | 30 → 33 |

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
- **Still to come:** Fig. 4 (LLM agents) waits for L2. R3 and T1 add panels to Fig. 3.
