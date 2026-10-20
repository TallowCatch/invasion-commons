# L11 Populations, seeds and analysis settings

Local audit source, 14 Oct 2026. Read only.

## Values and where set

| Item | Value | Where |
|---|---|---|
| Programmed test populations | 64 per cell | `run_r2_robustness_reviewer_model.py:66` `contexts=64`; `run_s1_reporting_audit.py:40`; `run_c1_compositional_harm.py:41`; `run_r4_river_grid.py:36` |
| Programmed training populations | 8 (S1-S4) or 32 (R2, R3, S5) | see L02 |
| River calibration contexts | 8 | `run_c1_compositional_harm.py:30` |
| LLM populations | 10 (L2, L3, L4 threshold/transfer, L5, L6); 20 (L4 matched) | `run_l2_llm_agents.py:37`; `run_l4_llm_identification.py:38` `CTX10, CTX20` |
| Sampled games per population | 1 (temperature 0.7, fixed seeds) | L2 results `:156` |
| Population definition | seeded draw of the 6 policies (+ weather, audits) from `pop_base + context` | `claude_oversight_common.py:36`; supplementary.tex:59 |
| Seed bases | one new base per study (e.g. L2 pop 1,500,000,000; L4 1,960,000,000; L6 1,980,000,000; R4 1,880,000,000) | each runner's `SEEDS` |
| Bootstrap | 4,000 resamples of populations, paired; seeds 20261005 (R1), 20261014 (C1), 20261019 (L-series, R4, R5) | protocols; `run_r4_river_grid.py:29` |
| Multiple comparisons | Holm (C1, L4, L6; memory comparisons in the paper) | protocols; `body.tex:77` |
| Token caps | L2 20 M per model (Amendment 1); L4 40 M; L5 5 M; L6 15 M | `run_l2_llm_agents.py:50`; `run_l4_llm_identification.py:37`; `run_l6_llm_forest.py:41` |

## Recorded justifications (verbatim)

- 64: first used in the 23 Sep confirmation (`background/september_reviewer_experiment.md:54-55`: "**Generate 64 fresh populations ("contexts") per game.** These are the independent units."). No power or precision argument for 64 was found in the notes.
- 10 LLM populations: `L2_llm_agents/protocol.md:69-77`: "**Contexts:** 10 per cell, the same 10 in every cell and for every model ... **Cost.** About 113,000 tokens per episode". Paper `body.tex:77`: "With 10 populations, the smallest paired difference in over-take rate detectable with 80\% power is 6--15 points" (computed after the runs).
- 20 for L4 matched cells: power argument (see L10).
- One game per population: `L2_llm_agents/results.md:156`: "Replies use temperature 0.7 with a fixed seed per call, so each game is one draw from the model. The intervals resample contexts, the independent unit; they do not separately show how muc[h] ..." Limitation in `body.tex:131`: "10 populations per condition, with one sampled game per population."
- Fresh seeds vs setting choice: `background/caveats_assessment.md:129-131`: "Fresh seeds protect against luck *within* a setting. They do not protect against choosing *which* settings to run."

## Verdicts

| Setting | Value | Verdict |
|---|---|---|
| 64 test populations | | U (no power reasoning recorded) |
| 10 LLM populations | | C (cost; power computed afterwards) |
| 20 LLM populations (L4 matched) | | C (power argument) |
| 8 calibration contexts (River) | | C |
| Seeds and bootstrap | 4,000 paired resamples, fixed seeds | C |
| One sampled game per population | | C (cost; stated as a limitation) |
