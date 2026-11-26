# Results R1a: do random audits and fines stop Selfish strategies taking over in Willis et al.'s commons?

**Status: complete:** 560 of 560 runs (the pre-registered 240 plus the 320 in the Amendment 4 extension). The last two extension runs (`n4_q0.25_e0.75`, runs 8–9) were re-run on 15–16 October (GitHub Actions run 38760318444) and the analysis repeated on 19 October. Only that extension point changed: 0.72 Collective [0.65, 0.79], against 0.71 on 18 runs.
- The analysis was run on 15 October 2026 on the `r1a-results` branch (GitHub Actions run 38591376955; code `b86d620`).
- Every number below comes from `runs/claude_r1a_v1/r1a_summary.json` and `r1a_conditions.csv` **[verified]**.

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

### H0: the paper replicates (holds)

| | Ours, no oversight (20 runs) | Paper |
|---|---|---|
| n = 64, share Collective | **0.15** [0.12, 0.18] | 0.19 |
| n = 64, welfare efficiency | **0.13** [0.11, 0.15] | 0.15 |
| n = 4, share Collective | **0.51** [0.43, 0.59] | 0.56 |
| n = 4, welfare efficiency | **0.91** [0.90, 0.91] | 0.91 |

### H1 at n = 64: Collective strategies take over just above the predicted fine (holds)

Share of Collective strategies in the last 100 generations (q = 0.1; Holm-corrected p = 0.001):

| Expected fine ÷ e_pred (0.96) | 0 | 0.5 | 0.75\* | 1.0 | 1.25 | 1.5\* | 2.0 |
|---|---|---|---|---|---|---|---|
| Share Collective | 0.15 | 0.20 | 0.22 | 0.34 | **0.60** [0.54, 0.65] | 0.72 | 0.84 |

\* Extension point (Amendment 4), not part of the test.

- **Verdict.** Below half at every level ≤ 0.75 e_pred, and above half at 1.25 and 2.0, as predicted.
- **The switch** lies between 1.0 and 1.25 times the fine predicted, before any run, from the starting mix. At exactly the predicted fine the share is still 0.34. The prediction is a payoff balance at the *starting* mix, and evolution needs a small surplus to tip the population.
- **Extension at q = 0.25**, the same expected fines made of more frequent and smaller fines: 0.17, 0.22, 0.38, 0.56, 0.71, 0.82. This is almost the same curve. With blind strategies this is expected: only the expected fine reaches selection.

### H1 at n = 4 (as amended): fails, but only at e = 0

- **Verdict.** H1 at n = 4 required the Collective share above half at every e, *including e = 0*. With no fine the share is 0.51 [0.43, 0.59]; the interval includes 0.5 (p = 0.38), so H1 fails there.
- **With any fine**, it is above half: 0.64 to 0.87 at q = 0.1, and 0.58 to 0.84 at q = 0.25.
- The payoff disadvantage of Selfish strategies at n = 4 (F13) is too small to win under weak selection on its own. A fine makes it decisive.

### H2: oversight rescues the large group (holds)

At n = 64, q = 0.1, e = 1.25 e_pred, compared with no oversight:
- welfare efficiency, fines counted as transfers: **0.13 → 0.57**, +0.44 [+0.38, +0.49];
- fines counted as lost: 0.13 → 0.39, +0.27 [+0.20, +0.33].

At 2 × e_pred, efficiency is 0.87 (transfers) and 0.74 (lost), close to the n = 4 level of 0.91.

### H3: break-even cost per audit (descriptive)

c\* = welfare gain over no oversight ÷ audits per game. Welfare is in payoff units; one agent's cooperative take per round at full stock is 2.

| n = 64, e ÷ e_pred | 0.5 | 1.0 | 1.25 | 2.0 |
|---|---|---|---|---|
| q = 0.1 (122 audits per game): c\*, fines as transfers / as lost | 0.04 / 0.01 | 0.07 / 0.02 | 0.14 / 0.08 | 0.23 / 0.19 |
| q = 0.25 (304 audits per game) | 0.02 / 0.00 | 0.03 / 0.01 | 0.05 / 0.03 | 0.10 / 0.08 |

**At the same expected fine, rare audits with large fines break even at 2.3–2.8× the audit cost of frequent audits with small fines.** They deliver the same deterrence with 60% fewer audits. This is Becker's cost argument, and it matches the direction of L4: at e = 2 t, a rare audit with a large fine deterred more.

### Secondary arm: fining the final round too (n = 64, q = 0.1)

| | Final round exempt | Final round fined |
|---|---|---|
| e = 1.25 e_pred: share Collective | 0.60 | **0.66** |
| e = 1.25 e_pred: efficiency (transfers) | 0.57 | **0.66** |
| e = 2.0 e_pred: share Collective | 0.84 | 0.86 |
| e = 2.0 e_pred: efficiency (transfers) | 0.87 | 0.90 |

Fining the final round adds deterrence against Selfish genotypes, and that outweighs the welfare lost by deterring the welfare-raising last-round grab. The design trap of F12 did not bite here.

### Which models' strategies survive (descriptive)

| | Largest shares (last 100 generations) |
|---|---|
| n = 64, no oversight | GPT-5.4 Mini Selfish 0.38, Claude Haiku Selfish 0.35, Gemini Selfish 0.12 (Collective 0.15 in total) |
| n = 64, e = 1.25 e_pred | GPT Collective 0.25, Claude Selfish 0.26, Claude Collective 0.22 |
| n = 64, e = 2.0 e_pred | **Claude Collective 0.39**, Gemini Collective 0.25, GPT Collective 0.20 |
| n = 4, any condition | **Gemini dominates** (0.85–0.88 of the population, both attitudes) |

Oversight changes not only how many strategies are Collective but **which developers' strategies win**. In large groups without oversight, Selfish GPT and Claude strategies take over. With strong oversight, Collective Claude strategies lead.

## What it means

1. **Willis et al.'s institution gap, filled for the first time.** An overseer with random audits and fines reverses the takeover by Selfish strategies in groups of 64. It raises welfare efficiency from 0.13 to 0.57 just above the predicted fine, and to 0.87 at twice it.
2. **The fine needed was predicted before any run** from fine-free games: a payoff balance of 0.96 per defection. The switch happened between 1.0 and 1.25 times it. This is the same kind of claim as L3/L4 (a frozen, then confirmed, threshold), now at the scale of hundreds of agents with cultural evolution.
3. **Cost.** For the same deterrence, rare audits with large fines are cheaper per unit of welfare than frequent small ones (H3). This connects the monitoring-cost feedback to Becker's result.

## What it does not show

- **Built in, with blind strategies.** Strategies cannot react to audits, so every effect runs through selection. "A large enough fine makes Collective win" is mechanical. What is not mechanical is that the switch fell close to the predicted balance point, given frequency dependence and weak selection, and that welfare recovered as much as it did.
- **Whether LLM-written strategies would respond to audits if told the rules.** That is R1b.
- **Equal results at q = 0.1 and q = 0.25 for the same e** are expected for blind strategies. They say nothing about how LLMs weigh certainty against severity; L4 tested that.

## Design caveats

- **"Time to switch" is uninformative.** Every run starts at 50% Collective, so noise crosses 0.5 in the first generation (median 1.0 in every condition). It is not reported as a result. The over-time curves (figure, panel d) show the dynamics instead.
- **e_pred is a starting-mix quantity** (see above). Its closeness to the switch point is an empirical finding, not a guarantee.
- **The amended n = 4 test** was close to a replication claim, and it failed only at e = 0.
- **Figure, panels c and d:** the x-axis labels and the legend overlap, to fix.

## What we now believe, what is untested, next step

- **Believed** (verified, one game, blind strategies, 20 runs per condition):
  - audits with fines reverse the Selfish takeover in large groups just above a fine predicted in advance;
  - rare audits with large fines are the cheaper way to get there.
- **Untested:**
  - in-game responses of strategies that know the rules (R1b, about $1 of Ollama credit);
  - an LLM-written overseer at this scale (R1d).
- **Next:** R1b, regenerating the strategies with the audit rule in the prompt (pilot of 20 strategies first).
