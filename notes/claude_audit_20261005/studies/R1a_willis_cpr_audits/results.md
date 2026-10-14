# Results R1a: do random audits and fines stop Selfish strategies taking over in Willis et al.'s commons?

**Status: pending: grid running remotely.** As of 14 October 2026, the grid runs on GitHub Actions (workflow `r1a-willis-cpr.yml`, results branch `r1a-results`). Done so far:
- the code and tests;
- the prediction, the pilot and the choice of R (Amendments 1–4).

No evolution result exists or has been looked at, so H0–H3, model shares and figures are pending. This file will be completed with `PYTHONPATH=.:external/emergent_llm/src python3 -m experiments.oversight.analyze_r1a --runs <checkout of r1a-results>/claude_r1a_v1/runs`.

**Protocol:** `protocol.md` in this folder, frozen 14 October 2026 before any R1a code existed, with four amendments written before any grid run:
1. how the frozen text is implemented (what strategies see, the seed key, the tests);
2. e_pred(4) turned out negative, so n = 4 uses the n = 64 fine scale and H1 at n = 4 is re-stated;
3. R = 20 from the timed pilot, with q = 0.25 and e = 0.75, 1.5 × e_pred dropped by the protocol's rule;
4. the grid runs remotely.

**Code:**
- `overseer/adapters/emergent_llm.py` (the audited game and evolution, subclassing their code);
- `experiments/oversight/run_r1a_willis_audits.py` (prediction, pilot, timing, sharded grid);
- `experiments/oversight/analyze_r1a.py` (H0–H3, tables, figure);
- `tests/test_r1a.py` (22 tests, all pass). The full suite passes: 454 passed, 11 skipped (before the last two runner tests were added; `test_r1a.py` alone, 22 passed).

**No LLM calls.** The strategies are the authors' committed Python classes: 3 models (Claude Haiku 4.5, Gemini 3.1 Flash Lite, GPT-5.4 Mini) × 8 attitude words × 128 = 3,072. The unused claude-fable-5 strategies are excluded.

## Question

Willis, Zhao, Du and Leibo (arXiv:2602.16662) found that when strategies spread by payoff-biased copying (cultural evolution), Selfish strategies take over groups of 64 in their commons game (CPR) and welfare efficiency falls to 15%, against 91% in groups of 4. They call for "cooperation-supporting institutions" but test none. R1a adds one: an overseer that audits each agent each round with probability q and fines a caught defection F.

## What was done so far

### The audited game (verified by tests)

- **Their game and evolution, unchanged:** C takes S/(2n), D takes S/n, logistic regrowth at rate 2 capped at k = 4n, 20 rounds; 512 agents, 4 games each per generation, Fermi imitation with β = 1, mutation 0.25%, 2,000 generations.
- **The overseer:** `UniformRandom(q)` audits, `Perfect` detection of D, `Fine(F)`. The final round is exempt (primary arm).
- **What the tests check** (`tests/test_r1a.py`):
  - **The stock follows extraction, never the net payoff.** With the same actions, fines of 0, 7.5 and 10⁶ give identical stock paths, both with hand-coded players and with their LLM-written strategies, at n = 4 and 64. A hand calculation of the stock matches round by round.
  - **The strategies are blind to fines.** The history they read holds what they took, not the fine (Amendment 1). This matters because 54 of the 3,072 strategies read their own payoff history.
  - **Final-round exemption.** No audit and no fine in round 20; rounds 1–19 are drawn identically in both arms.
  - **Audits.** They equal `default_rng(stable_seed(26101440, "audit", run, generation, game, round)).random(n) < q`. They are deterministic. Audits at q = 0.1 are a subset of those at q = 0.25. They never touch the generators the strategies use.
  - **e = 0 is their game, bit for bit.** One game: identical actions, payoffs and stock. Whole evolution runs (n = 4 and 64, several generations): identical genotype histories, identical final populations, identical mean payoffs, and Python's and numpy's generators end in the same state as in their unmodified `CulturalEvolution`.
  - **Bookkeeping.** Net payoff = extraction − fines; fines = F × catches; a C is never caught; every audited D is caught.
  - **Their imitation uses fitness differences only.** Adding a constant to every fitness leaves the copying draw unchanged, so keeping their fine-free normalisation bounds keeps selection strength the same in every condition.
  - **The CI runner.** Shards partition the job list; existing result files are never overwritten.

### The prediction (verified; computed before any evolution run)

Source: `runs/claude_r1a_v1/prediction.json`, computed 2026-10-14 15:27:35 UTC (16 s). It used 2,000 fine-free games per n at the evolution's starting mix (every seat a uniformly drawn genotype and strategy, seed 26101440 + 900).

| | n = 64 | n = 4 |
|---|---:|---:|
| Mean payoff per game, Selfish-genotype agent | 18.77 | 26.82 |
| Mean payoff per game, Collective-genotype agent | 15.56 | 29.18 |
| ΔΠ (Selfish − Collective) | **+3.21** | **−2.36** |
| Non-final-round defections per game, Selfish / Collective | 10.00 / 6.66 | 6.69 / 2.65 |
| ΔD | 3.34 | 4.03 |
| **e_pred = ΔΠ ÷ ΔD** (expected fine per defection that equalises the two) | **0.962** | **−0.586** |
| Descriptive 95% interval (bootstrap over games, not in the protocol) | 0.933 to 0.995 | −0.723 to −0.441 |
| Welfare efficiency of these games | 0.35 | 0.63 |
| Myopic benchmark: one-round gain from D at full stock | 2.0 (F\* = 20 at q = 0.1) | 2.0 |
| Forward-looking gain of one mid-game D among cooperators (rundown §4.5) | 1.97 | 1.30 |

**What this says.**
- **In groups of 64**, a Selfish strategy earns 3.21 more per game than a Collective one at the start, and defects 3.34 more times before the last round. So an expected fine of about 0.96 per defection should cancel its advantage at the starting mix.
  - That is under half the myopic benchmark of 2. Selfish strategies defect most when the stock is already low, where a defection gains little.
- **In groups of 4, Selfish strategies already earn less than Collective ones before any fine** (−2.36 per game). Their extra defections drain a stock shared with only three others, so they hurt themselves.
  - So the selection argument predicts that Collective strategies are favoured at n = 4 with no oversight at all. That agrees with the paper's 56% Collective at n = 4.
  - The frozen fine levels at n = 4 would have been negative (a reward for defecting). Amendment 2 therefore uses the n = 64 scale at both sizes (e ∈ {0, 0.481, 0.962, 1.202, 1.924} after Amendment 3). It re-states H1 at n = 4 as "Collective > 0.5 at every tested e, including 0".

### The pilot and the choice of R (verified; Amendment 3)

- **Pilot:** 2 runs × 200 generations at n = 64, e = 0: 63.5 s and 45.0 s, so 0.271 s per generation (`runs/claude_r1a_v1/pilot_timing.json`).
- **Probes** of the other run types (`timing_probe.json`): an n = 4 run with audits costs about 2.1× an n = 64 run, because the overseer draws once per agent-round (9,728 rounds per generation at n = 4, 608 at n = 64).
- **Cost of the full 28-condition grid:** about 126 CPU-hours at R = 20 and 627 at R = 100. The protocol's budget is 24 CPU-hours.
- **The protocol's rule therefore gives R = 20.** It drops q = 0.25 first and then the 0.75 and 1.5 × e_pred levels. The analysed grid is 12 conditions × 20 runs = 240 runs, about 50 CPU-hours here, still above the budget; the rule has no further step.
- **What is lost:**
  - H2 can be checked at q = 0.1 only, not "for both q";
  - H1 is tested at 2 cells instead of 4.
- *Other reading, recorded in Amendment 3:* if the 24 h meant wall-clock on three 4-core lanes, the full grid would have fit at R = 30.

## What came out

**Pending: grid running remotely.** No evolution result exists yet. A local grid was started at 15:32 UTC and stopped at 15:37 UTC on instruction, before any run finished (Amendment 4).

When the runs are in, `analyze_r1a.py` writes:
- `runs/claude_r1a_v1/r1a_summary.json`, `r1a_conditions.csv` and `r1a_runs.csv`;
- `figures/fig_r1a_willis_audits.{pdf,png}`, a composite with four panels:
  - (a) share of Collective strategies against e ÷ e_pred(64), for n = 4 and 64;
  - (b) welfare efficiency on the same axis, with fines counted as transfers (points) and as lost (bars);
  - (c) final genotype shares by model and attitude;
  - (d) share Collective over generations at n = 64.

The analysis and figure were checked end to end on a 4-generation smoke run (not analysed, not kept).

## What it will and will not be able to show

**True by construction, with blind strategies:**
- A fine never changes what any strategy does in a game. The strategies cannot see it, and the stock path given the actions is identical.
- Every effect of oversight therefore comes through **which genotypes get copied**.
- A large enough fine lowers a Selfish genotype's payoff mechanically. "Collective wins at a high enough fine" is built in.

**Not built in, and what R1a tests:**
- whether the switch happens near e_pred(64) = 0.96, given frequency-dependent payoffs and weak, noisy selection (β = 1 on fitness normalised to about 0–1);
- whether welfare in groups of 64 recovers by at least 0.2 (H2);
- which model's strategies survive;
- whether the welfare gain is worth the audit cost (H3, break-even cost per audit).

**What R1a cannot show:**
- **Whether strategies respond to audits within a game.** That needs strategies regenerated with the audit rule in the prompt (R1b).
- **The welfare result counting fines as lost.** In groups of 64 at the predicted fine this count can fall below the no-oversight level even if the Collective share rises. Both counts are reported.

## Design caveats

- **e_pred is computed at the starting mix only.** As the population changes, the payoff gap changes too, so the prediction is a first-order guide, not an equilibrium threshold.
- **n = 4 uses another group size's scale** (Amendment 2), chosen after seeing the prediction but before any evolution run. H1 at n = 4 as re-stated is close to a replication claim: it predicts that Collective wins at e = 0.
- **Shared seeds.** Run i starts from the same population in every condition. The intervals still resample conditions independently (conservative).
- **Final-window statistics** follow their batch summary: the last 100 played generations.
- **Software.** Their package imports LLM client libraries, `pathos` and matplotlib at import time. The adapter replaces any that are missing with placeholders that are never called; their code is unmodified.

## What we now believe, what is untested, next step

- **Believed (verified):**
  - The audited CPR is correct: fines never leak into the stock, and e = 0 is the authors' code bit for bit.
  - At the starting mix, defection pays in groups of 64 (by 0.96 per defection) and already costs in groups of 4.
- **Untested:** every evolution hypothesis (H0–H3).
- **Next:** run the 240 jobs on CI lanes (about 50 CPU-hours on an M1 core, perhaps 60–75 on hosted runners; for example 8 lanes × 4 processes, about 2–2.5 h each). Then run `analyze_r1a.py` and fill in this file. No LLM cost.
