# L12 Where the settings came from: pilot history, after-the-fact choices, and paper/code mismatches

Local audit source, 14 Oct 2026. Read only.

## 1. Lineage of the game settings

| Date | Event | Source (verbatim) |
|---|---|---|
| 24-25 Feb 2026 | Fishery created | `PROJECT_HISTORY.md:66`: "Created the repository and a minimal Fishery simulation: one renewable stock, agents requesting harvest, noisy stock observations, collapse and population-composition sweeps." |
| 27-28 Feb | Hard settings tuned | `PROJECT_HISTORY.md:67`: "Calibrated easy/hard conditions ... Harsh conditions were tuned because near-certain collapse cannot distinguish interventions." |
| 11-16 Mar | Harvest (Forest) added | `PROJECT_HISTORY.md:69`: "Fishery's single stock could not test the difference between what a participant sees locally and what happens across patches." |
| 2-21 Apr | Scenario presets (incl. `forest_co_management`, externality 0.22) | `PROJECT_HISTORY.md:70`: "the scenario labels were parameter packages, not validated irrigation or forestry field models." |
| 22 Sep | Matched-oversight pilot fixes Fishery 70/100/0.7/6/10 | `MATCHED_OVERSIGHT_PILOT_PROTOCOL.md:57-59`: "These are declared pilot choices, not externally calibrated values." |
| 23 Sep | Held-out policy pilot fixes ranges and U(25, 65) | `HELDOUT_POLICY_PILOT_PROTOCOL.md:25`: "These are design choices, not measured actor-capability levels." |
| 23 Sep | Confirmation picks mix4 Fishery and mix2 slow-renewal Forest | `BUDGETED_REVIEWER_CONFIRMATION_PROTOCOL.md:5-6`: "The earlier pilot selected the two informative settings and the intervention rules". |
| 4-5 Oct | R1 keeps them, adds MSY target | `R1_repaired_reviewer/protocol.md:52-60` ("Kept from the original design"); `:120` "this tests the post hoc explanation" |
| 5 Oct | River calibrated | C1 Amendment 1 (rule targeting ~45% risky steps) |
| 5 Oct | R2 grid of never-piloted settings | `R2 protocol:33`: "Exactly one cell per game was piloted; the other 8 were never run before." |
| 6 Oct | LLM interface (L1): horizon 20, temperature 0.7, 4 LLM agents | `L1 protocol:37`, `:81` |
| 8 Oct | Harm = half capacity; GovSim/Perolat metrics | `runs/social_metrics/README.md:1` "(8 October 2026, post hoc)" |
| 9 Oct | R4, R5 put all games on one 2×2 grid | R4/R5 protocols ("at Ameer's request") |
| 12 Oct | L4, L5, L6 (freeze lifted) | L4 protocol `:3` |

`background/caveats_assessment.md:148-154` summarises the chain: "the MSY target and the dropping of exclusion were both decided after seeing S1 (the S1b/S2 results); S4 was written knowing S1b, S2 and S3 (the S4 protocol); every experiment inherited the two game settings. This sequential way of working is normal and useful."

## 2. Settings chosen after seeing results (search words: pilot, chosen, tuned, post hoc, amended)

| Setting | What was seen first | Source |
|---|---|---|
| mix4 Fishery, mix2 Forest, Forest ×0.85 | September pilot outcomes (both safe and risky requests) | `design_issues.md:199-204` |
| Fishery MSY target as reviewer default | September post hoc re-analysis; R1 results | `R1 protocol:120`; `S1b_S2 protocol:110` |
| Half-capacity harm line in all games; efficiency/equality/survival | S-series, C1, T1 and L2 results | `PAPER_PLAN.md:56`; `social_metrics/README.md:1` |
| River κ, r, σ, ranges | calibration rule plus an unsaved coarse look | C1 Amendment 1 |
| S1 fine 12 | smoke run (exclusion rewarded the offender) | S1 Amendment 1 |
| S3 e-grid 0-1 | S2's threshold below its grid | `S3 protocol:25-27` |
| S4/S5 fine 6 | S3's break-even | `S4 protocol:98-99` |
| Programmed gain definition (R2) | R2 data ("redefining the gain *after* seeing the data") | `R3 protocol:18-20` |
| 32 training contexts (R2+) | S3 Part C over-fitting | `caveats_assessment.md:135-137, 180` |
| LLM explicit permission wording | L1: 0% over-taking without it | L1 results decision table |
| LLM catch uncapped; units in tonnes | L1 v1/v2 faults | L1 Amendments 2, 3 |
| LLM fine grids (L2, L3), L4's e = 2, 3 | L1, L2, L3 curves | L1 results; L3 `:7`; L4 `:61-64` |
| Mistral Large 3, DeepSeek-only L5 | token cost of reasoning models | L2 Amendment 4; L5 Amendments 2-4 |
| L6 fine = 0 | L6 pilot over-take rate 4% | L6 Amendment 1; `f_choice.json` |
| One-round gain g1 = 6 − allowance; "realised whole-game gain" | all LLM results | `supplementary.tex:266` "Both analyses were chosen after all results were seen." |

Each of these is dated and states what was seen, which is better practice than most commons papers. The exposure is that the *values* were chosen to make the games informative for the reviewer comparisons, not taken from outside data.

## 3. Mismatches between the paper text and the code or protocols

1. **Audit rate in the memory grid.** `supplementary.tex:22`: "Audits random with probability $1/6$ per agent and round (T1 and C1: one audit per round)." The Figure 2a memory grid uses `MEMORY_PROTOCOL = "rand2"` (`run_r2_robustness_reviewer_model.py:52`; R5 reuses it), two audits per round in Fishery and Forest (1/3 per agent), and one per round in River (`run_r4_river_grid.py:79-80`). S1 and S1b used rand1 and rand2.
2. **Training populations.** `body.tex:67`: "a search chooses to maximise their payoff on 32 training populations". S1, S1b, S2, S3 and S4 used 8 (`run_s2_compliance_deterrence.py:44`, reused by S3 and S4). The pilot threshold result (`body.tex:96`) is S3 Part A (8).
3. **Forest harm.** `body.tex:41` gives "a mean Forest plot health below 10 of 20 (as a 5% chance constraint)"; the code also counts "at least half the plots below 4" as unsafe (`calibrated_oversight.py:35-37`).
4. **"Sustainable 2.1".** `body.tex:45`. The per-plot MSY under the game's own regrowth is 2.38; 2.1 is the engine's default `sustainable_harvest_frac = 0.35` × 6 (`harvest.py:21`).
5. **Tolerance naming.** The task brief's "tolerance 0.06" is the LLM over-take margin (0.06 t, `run_l2_llm_agents.py:60`); the reviewer's chance-constraint tolerance is 0.05 (`calibrated_oversight.py:52`).
6. **L6 fine.** The paper's limitations say "language-model versions of Forest and River would test whether over-taking spreads" (`body.tex:131`); L6 has now run, but with F = 0 (no consequence), which any Forest claim must state.
