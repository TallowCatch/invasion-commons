# L04 Audits, fines and memory for programmed agents (S1-S5, R2-R5, T1, C1)

Local audit source, 14 Oct 2026. Read only.

## Code constants

| Study | File:line | Value |
|---|---|---|
| S1 | `experiments/oversight/run_s1_reporting_audit.py:28-29` | protocols `report, rand1, rand2, targ1, targ2, peer, peer_collude` |
| S1 | `:32` | `EXCLUSION_STEPS = 5  # caught agents get 0 this step and the next 5` |
| S1 | `:33` | `FINE = 12.0  # Amendment 1: fine per catch (two steps of one agent's maximum harvest)` |
| S1 | `:75-77` | `rand` protocols audit `k` agents: `rng.permutation(active)[:k]` (rand1 = 1 per round, rand2 = 2 per round) |
| S2 | `run_s2_compliance_deterrence.py:33` | `FINES = (0.0, 6.0, 12.0, 24.0)` |
| S3 | `run_s3_threshold_timing_memory.py:33` | `E_GRID = (0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0)` |
| S3 | `:34-36` | `PART_A_Q = (1 / 6, 0.10, 0.05, 0.02)`, `PART_B_S = (0.5, 0.25)`, `PART_C_FINES = (0.0, 6.0, 24.0)` |
| S4 | `run_s4_adaptive_memory_cost.py:34-36` | `FINE = 6.0`, regimes `fine, memory, fine+memory`, `RATES = (0.01, 0.02, 0.05, 0.10, 1 / 6, 1 / 3)` |
| S4 | `:44-48` | memory rule `targeted_allowance(scale, dbar, p)`: `max(0, (scale - dbar)/(1 - dbar)) * p` |
| S5 | `run_s5_stronger_attackers.py:35-38` | `FINE = 6.0`, regimes incl. `memory_cut`, `memory_cap`, `RATES = (0.02, 0.05, 0.10, 1 / 6)` |
| R2 | `run_r2_robustness_reviewer_model.py:52-58` | `MEMORY_PROTOCOL = "rand2"`, `LIE = 0.5`, `DETER_Q = 1 / 6`, `N_E = 11` |
| R3 | `run_r3_breakeven.py:22` | `Q = r2.DETER_Q  # 1/6` |
| R5 | `run_r5_fishery_grid.py:19-21` | `BASE_R = 0.7`, `GREEDY = (2, 4)`, `MULT = (1.0, 0.85)`; uses R2's `memory_episode` (rand2) |
| T1 | `run_t1_audit_targeting.py:33-34` | `LIE = 0.5`, arms `trust: report, random: rand1, report: rep1, signal: targ1` |
| C1/R4 | `run_c1_compositional_harm.py:123`; `run_r4_river_grid.py:79-80` | one agent audited per round: `integers(0, TR.N_AGENTS)` |
| River memory | `fishery_sim/two_reagent.py:239-247` | `memory_belief`: report rescaled by `1/(1 - mean d_hat)` |

## Recorded justifications (verbatim)

- q = 1/6 origin: inspection budgets k ∈ {0, 3, 6} of 6 agents (`R1_repaired_reviewer/protocol.md:59`). `literature_ledger.md:84`: "The "2% audit budget" is a different kind of quantity. It is a share of tasks audited by expensive trusted labour. Our k of 6 per step is a much larger share (17–100%) with perfect verification. Results at k/6 are not comparable to the 2% regime."
- S1 fine: `S1_reporting_and_audits/protocol.md:197-211` (Amendment 1, after the smoke run): "In the Harvest smoke run, a caught agent's later payoff was sometimes *higher* than an honest agent's. ... the exclusion acts like resting a field rather than a sanction. **Why it matters.** A penalty that can benefit the offender cannot test deterrence (Becker 1968; Allingham & Sandmo 1972 ...). ... pays a fine of 12 units (two steps of one agent's maximum harvest)". The post-results note (`:222-228`) says those two papers were checked only for identifiers.
- S3 e-grid: `S3_threshold_timing_memory/protocol.md:25-27`: "In S2, any tested fine (6, 12 or 24) with one random audit per step stopped ... threshold near a fine of 2, below the grid." So the S3 grid was set from S2's outcome.
- S4 fine: `S4_adaptive_cheaters_audit_cost/protocol.md:98-99`: "**Fines at F = 6:** from S3, deterrence starts at e = q × 6 ≈ 0.36, so cheaters should comply at q ≥ 0.10 and cheat at q ≤ 0.05."
- S3 break-even: `S3_threshold_timing_memory/results.md:45-49`: "each gained 28.5 over complying, held out. The expected-fine break-even is therefore 28.5 ÷ 80 = **0.356**."
- R2's gain definition changed after the data: `R3_breakeven_confirmation/protocol.md:18-20`: "**R2** found the threshold in the unpiloted settings only after redefining the gain *after* seeing the data". `PROJECT_HISTORY.md:111`: "The deterrence formula as written failed; it matched once measured correctly **[post hoc]**."
- Memory concept: paper `body.tex:24`: "records of past offences give enforcement leverage \citep{harrington1988enforcement}". The memory *rules* themselves are the project's own (S3 Part D; S4 targeted allowance).
- Two audits per round in the memory grid: `R2_robustness_and_reviewer_model/protocol.md:43-44`: "S3 Part D memory ... Two random audits per step; misreporters are the stress agents and under-report by half". No reason recorded for 2 rather than 1. River (R4) uses one per round (`R4_river_grid/protocol.md:27`: "one agent per round is audited at random").
- Exclusion of 5 steps (S1): no reason recorded.
- Grounding of the deterrence logic: paper `body.tex:10` and `:51` cite Becker 1968, Allingham & Sandmo 1972, Avenhaus et al. 2002. Ledger `:87`: "Becker and Allingham–Sandmo were checked at metadata depth only."

## Flags

- **Paper/code mismatch:** supplementary.tex:22 says "Audits random with probability $1/6$ per agent and round (T1 and C1: one audit per round)". The memory grid behind Figure 2a (R2, R5) audits two agents per round in Fishery and Forest (rate 1/3 per agent), and River (R4) one per round. S1/S1b also used rand1 and rand2.
- **Chosen after results:** S3's e-grid (from S2), S4/S5's F = 6 (from S3), R2's gain definition (post hoc), R3 (prospective re-test of the post hoc definition).

## Verdicts

| Setting | Value | Verdict |
|---|---|---|
| Audit rate (main) | 1/6 per agent-round (one of six) | C (inherited from k of 6; not comparable to AI-control 2%) |
| Audit-rate grids | S3 {1/6, .10, .05, .02}; S4 {.01 ... 1/3}; S5 {.02 ... 1/6} | C |
| Detection s | 1, 0.5, 0.25 | C |
| Audits in memory grid | 2 per round (Fishery, Forest), 1 (River) | U (no reason; paper misstates) |
| S1 fine | 12 + 5-step exclusion | C (fine); U (5 steps) |
| S2 fines | 0, 6, 12, 24 | U |
| S3 e-grid | 0-1 | C (set from S2 result) |
| S4/S5 fine | 6 | C (set from S3 result) |
| R2/R3 e-grid | j × 0.2 g*, j = 0..10 | C |
| Memory rules | rescale by mean lie; targeted allowance; tighter shared cut | C (concept cited: Harrington 1988) |
