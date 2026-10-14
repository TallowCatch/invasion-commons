# R1a protocol: do random audits and fines stop Selfish strategies taking over in Willis et al.'s commons?

**Frozen:** 14 October 2026, before any R1a code exists. Amendments are appended below and never edit this text.

**Background.**
- Willis, Zhao, Du and Leibo (arXiv:2602.16662; code `willis-richard/emergent_llm`, commit `be78ff9`, MIT licence) found that under payoff-biased imitation, Selfish LLM-written strategies take over large groups in their Common Pool Resource (CPR) game: welfare efficiency is 15% at n = 64, against 91% at n = 4.
- They call for "cooperation-supporting institutions" but test none (research note `research/commons_platform/findings/willis_rundown.md`).
- R1a adds this project's overseer (`overseer/`) to their CPR: random audits that detect a defection, and a fine.

**What R1a can and cannot show.**
- The existing strategies are **blind** to audits. They were written without any audit rule, so a fine cannot change what a strategy does in a game. It can only change **which strategies win and get copied** under cultural evolution.
- R1a therefore tests deterrence **through selection**, not through in-game responses. In-game responses are R1b's question, with strategies regenerated with the rules in the prompt.
- Because a fine lowers a Selfish strategy's payoff mechanically, part of the result is true by construction. What is *not* built in:
  - where the switch happens, given frequency-dependent payoffs and noisy selection;
  - whether welfare in large groups recovers;
  - which model's strategies win;
  - whether the welfare gain exceeds the audit cost.

## Setting (their code, unchanged)

**Game.** Their CPR, unchanged:
- capacity k = 4n; r = 20 rounds;
- C takes S/(2n) and D takes S/n;
- logistic regrowth at rate 2, capped at k.

**Strategies.** Their committed CPR pool: 3 models × 2 attitudes (Collective, Selfish), 512 strategies per (model, attitude). The genotype = (model, attitude), as in their cultural evolution.

**Evolution settings.** Their paper's values:
- 512 agents;
- G = 4 games per agent per generation;
- Fermi imitation with β = 1;
- mutation 0.25% per gene;
- 2,000 generations (the paper's value; their script defaults to 1,000);
- statistics averaged over the final 100 generations;
- group sizes n = 4 and n = 64.

**Seeds.** Their code sets none. R1a seeds Python's and numpy's generators per run from `R1A_BASE = 26101440` + run index. Audit draws use `overseer.seeding.stable_seed(R1A_BASE, "audit", run, generation, game, round)`.

## The overseer (this project's `overseer/` package)

- **Audits:** `UniformRandom(q)`, one draw per agent per round.
- **Detection:** perfect. A caught D is a D; a C is never caught.
- **Sanction:** `Fine(F)`, subtracted from the payoff only. **The stock is reduced by the extraction, never by the net payoff.**
- **No allowances.** The C extraction is the implicit allowance.
- **Final round: exempt (primary).** A final-round D raises welfare in their game, and their welfare maximum assumes all-D then. Not exempt is a secondary arm (below).
- **Fitness normalisation:** their original bounds from the game without fines. The same normalisation in every condition, so selection strength is the same. If their `_imitate` uses anything other than fitness differences, this is amended before any run.

## The prediction (computed before any run with a fine)

For each n, from fine-free games among strategies sampled uniformly across the six genotypes (the evolution's starting mix; 2,000 games per n, seed `R1A_BASE + 900`):
- ΔΠ = mean payoff of a Selfish-genotype agent − mean payoff of a Collective-genotype agent, per game;
- ΔD = the mean difference in the number of non-final-round defections, per game;
- **e_pred = ΔΠ ÷ ΔD**, the expected fine per audited-eligible defection at which the two genotypes' mean payoffs are equal at the starting mix.

This is written to `runs/claude_r1a_v1/prediction.json` with a timestamp before the grid runs. Also recorded: the myopic benchmark at full stock (2 per D, so F\* = 2/q) and the forward-looking deviation gain (1.30 at n = 4, 1.97 at n = 64; Willis rundown §4.5).

## Conditions

- **Audit probability:** q ∈ {0.1, 0.25}.
- **Expected fine per defection:** e = qF ∈ {0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0} × e_pred(n), so F = e/q.
- **Group size:** n ∈ {4, 64}.
- e = 0 is the no-oversight replication. With q = 0.1 and q = 0.25 at e = 0, it is run once per n, not twice.
- **Secondary arm, final round not exempt:** n = 64, q = 0.1, e ∈ {1.25, 2.0} × e_pred.

## Hypotheses and falsifiers

- **H0 (replication).**
  - With no oversight at n = 64, the final-window Collective frequency is < 0.5 and welfare efficiency (fines as transfers) is < 0.3. The paper reports 19% and 15%.
  - At n = 4, welfare efficiency is > 0.5 (the paper reports 91%).
  - *Fails* if either is outside these bounds. A failure is reported as non-replication, and H1–H3 are then interpreted against our own baseline.
- **H1 (selection deterrence at the predicted fine).** At each n and q, the final-window Collective frequency is < 0.5 at e ≤ 0.75 e_pred and > 0.5 at e ≥ 1.25 e_pred.
  - *Fails* at an (n, q) if either side is violated (the interval of the run mean must exclude 0.5).
- **H2 (oversight rescues the large group).** At n = 64, welfare efficiency at e = 1.25 e_pred is at least 0.2 above the e = 0 baseline, for both q.
  - *Fails* if the interval of the difference includes 0.2 from below.
- **H3 (monitoring cost, descriptive).** The break-even cost per audit is c\* = (W_cell − W_none) ÷ (audits per game), per W1's definitions. Both welfare counts are reported: fines as transfers and fines as lost.
- **Descriptive outcomes:**
  - model shares in the final window under each condition (which model's strategies survive oversight);
  - fines paid per game;
  - the time to switch (the first generation where the Collective frequency exceeds 0.5, if ever).

## Unit, runs, intervals

- **Unit:** the independent evolution run.
- **Number of runs per condition, R:** fixed after a timed pilot by this rule. Pilot: 2 runs of 200 generations at n = 64, e = 0, not analysed.
  - R = 100 (the paper's value) if the full grid fits in 24 h of total CPU on the GitHub Actions lanes; otherwise the largest of {50, 30, 20} that fits.
  - R is never below 20. If even 20 does not fit, drop q = 0.25 first and then the e = 0.75 and 1.5 levels, by amendment, before any full run.
- **Intervals:** 95% bootstrap over runs, 4,000 resamples, seed 26101441.
- **Multiplicity:** H1 is tested at 4 (n, q) cells, Holm-corrected one-sided tests at 0.05.

## Outputs

- **Vendored code:** `external/emergent_llm` at commit `be78ff9`, unmodified.
- **Our code:**
  - `overseer/adapters/emergent_llm.py`: an audited CPR built by subclassing their game in our code;
  - `experiments/oversight/run_r1a_willis_audits.py`;
  - `experiments/oversight/analyze_r1a.py`;
  - `tests/test_r1a.py`. Tests check that the stock is unaffected by fines, the final-round exemption, audit determinism, and that e = 0 is identical to their unaudited game on fixed seeds.
- **Results:**
  - raw files in `results/runs/claude_r1a_v1/`;
  - a summary in `notes/claude_audit_20261005/runs/claude_r1a_v1/`;
  - `studies/R1a_willis_cpr_audits/results.md`.
- **Figures:** Collective frequency and welfare efficiency against e/e_pred, for n = 4 and 64; model shares.
- **Findings log:** append an entry to `FINDINGS_AND_IMPLICATIONS.md`.

**Stop rule.** No reruns for significance. A failed replication (H0) is reported and does not trigger changes to their settings.

## Amendments

### Amendment 1 (14 October 2026, 15:27 UTC; written after reading their code and writing the adapter and tests, before the prediction and before any evolution run)

Clarifications of how the frozen text is implemented. None changes a hypothesis, a condition or a threshold.

1. **What the strategies see.** Their game records the payoff in the history that strategies read (`PlayerHistory.my_payoffs`), and 54 of the 3,072 strategies read it. The audited game records the **extraction** there, not the net payoff, and keeps fines in a separate per-player, per-round array. So the strategies are exactly blind to audits and fines, as "What R1a can and cannot show" assumes. Fitness (their `total_payoff`) is the net payoff, extraction minus fines.
2. **Detection in share units.** The overseer's ledger is in units of the C share: taken = 1 for C and 2 for D, allowance = 1, `Perfect()` detection. An audited D is therefore caught at any stock, including stock 0.
3. **Final-round exemption.** In the exempt arm the overseer is not called in round 20: no draw, no audit, no fine. Draws in rounds 1–19 are identical in both arms. "Audits per game" counts the audits actually made.
4. **Audit seed key.** `stable_seed(R1A_BASE, "audit", run, generation, game, round)` with game = repetition × (512/n) + match, in their play order, and agent = seat in that game. q and F are not in the key, so at the same (run, generation, game, round) the agents audited at q = 0.1 are a subset of those at q = 0.25. The audit generator is separate from Python's and numpy's global generators, so auditing never shifts the random streams their code and the strategies use. Run i uses seed R1A_BASE + i in every condition, so run i starts from the same population everywhere.
5. **Baselines.** e = 0 is run with no audits (q = 0), once per n. With F = 0 audits cannot change anything (tested: an F = 0 run with audits is bit-for-bit their unaudited run).
6. **Their `_imitate`** uses differences of normalised fitness, `(π − π_min)/(π_max − π_min)` with their fine-free bounds, in a Fermi rule. That is an affine map with the same constants in every condition, so selection acts on fitness differences only (tested: adding a constant to all fitnesses leaves the imitation draw unchanged). No amendment of the normalisation is needed.
7. **Reported statistics follow their batch summary.** Final window = the last 100 *played* generations (1,901–2,000). Collective frequency = the window mean of the population that played. Welfare per player per game is averaged over the window, then turned into efficiency with their analytic bounds (`min_player_welfare`, `max_player_welfare`). Model shares are window means.
8. **Prediction sampling.** Each seat draws one of the 24 fine (model, sub-attitude) genes uniformly and then one of its 128 strategies, which is their starting mix (uniform over the six base genotypes).
9. **Tests, made precise.**
   - Intervals: each condition's runs are resampled independently (4,000 resamples, seed 26101441), percentile intervals.
   - H1 at an (n, q) cell: "e ≤ 0.75 e_pred" includes the shared e = 0 baseline, as written. For each level, a one-sided bootstrap p-value against 0.5 (the share of resampled means on the wrong side, (k + 1)/(B + 1)); the cell's p-value is the largest over its levels (all must hold); Holm over the cells tested, at 0.05. The 95% intervals are reported too.
   - H2: the 95% interval of the difference (independent resampling of the two conditions); H2 holds at a q if its lower end is above 0.2.
   - H3: W per game = the sum over the n players of extraction (fines as transfers) or of net payoff (fines as lost); c\* = (W_cell − W_none) ÷ audits per game, final-window means. Descriptive.
10. **Software.** Their package imports LLM client libraries (anthropic, openai, ollama, google-genai) and `pathos`, which are not installed and which R1a never calls. The adapter puts empty placeholder modules in `sys.modules` for them; anything that called one would raise. Their code is not modified.

### Amendment 2 (14 October 2026, 15:28 UTC; written after the prediction, before the pilot and before any evolution run)

**What was seen.** The prediction (`runs/claude_r1a_v1/prediction.json`, computed 15:27:35 UTC) gives:
- n = 64: ΔΠ = +3.21 per game, ΔD = 3.34 non-final defections per game, **e_pred = 0.962** (descriptive bootstrap over games 0.933–0.995).
- n = 4: ΔΠ = **−2.36**, ΔD = 4.03, **e_pred = −0.586** (−0.723 to −0.441). At the starting mix a Selfish-genotype agent in a group of 4 already earns *less* than a Collective one: its extra defections deplete a stock it shares with only three others.

A negative e_pred makes the frozen fines at n = 4, F = (multiple × e_pred)/q, negative: a payment for each caught defection. That is not an audit-and-fine institution.

**Change.**
1. **n = 4 uses the n = 64 scale.** At n = 4 the expected fines are e = multiple × e_pred(64), the same absolute values as at n = 64: e ∈ {0, 0.481, 0.721, 0.962, 1.202, 1.443, 1.924}. So the two group sizes get identical fines, and n is the only difference between them. Figures plot n = 4 against e ÷ e_pred(64) and say so.
2. **H1 at n = 4, re-stated.** The frozen H1's lower side ("< 0.5 at e ≤ 0.75 e_pred") is undefined at n = 4 and is reported as *not testable as frozen*. The same selection argument predicts that, because every e ≥ 0 exceeds e_pred(4), the Collective frequency is **> 0.5 at every tested e at n = 4, including e = 0**. This is tested as H1(n = 4) instead, with the same one-sided bootstrap and Holm procedure as Amendment 1, item 9. It is labelled "amended before the runs".
3. H1 at n = 64, H0, H2 and H3 are unchanged. The secondary arm is unchanged (n = 64).

### Amendment 3 (14 October 2026, 15:32 UTC; R fixed from the timed pilot, before any grid run)

**Pilot** (`results/runs/claude_r1a_v1/pilot/pilot_timing.json`): 2 runs × 200 generations, n = 64, e = 0, 2 processes on this machine (Apple M1, 8 cores): 63.5 s and 45.0 s, so **0.271 s per generation**, about 540 s per 2,000-generation run. Short probes of the other condition types (`timing_probe/timing_probe.json`, 50 generations each, 6 at once) cost, relative to an n = 64 fined run: n = 4 baseline 1.24×, n = 4 fined 2.08× (one audit generator per agent-round, 9,728 rounds per generation at n = 4 against 608 at n = 64), n = 64 fined 1.03×. Per 2,000-generation run (single-process CPU): n = 64 about 550 s, n = 4 baseline 670 s, n = 4 fined 1,130 s.

**Applying the rule.** "Total CPU" is read as summed CPU time, with a GitHub Actions core assumed as fast as one core here (optimistic: hosted runners are usually slower).

| Grid | Conditions | R = 100 | 50 | 30 | 20 |
|---|---|---|---|---|---|
| Full | 28 | 627 h | 314 h | 188 h | 126 h |
| q = 0.25 dropped | 16 | 346 h | 173 h | 104 h | 69 h |
| … and e = 0.75, 1.5 × e_pred dropped | 12 | 252 h | 126 h | 76 h | **50 h** |

None fits in 24 h. The rule's last step is reached, so:
1. **R = 20** runs per condition (the floor).
2. **q = 0.25 is dropped, then e = 0.75 and 1.5 × e_pred are dropped.** The analysed grid is 12 conditions, 240 runs: the two baselines; n ∈ {4, 64} at q = 0.1 with e ∈ {0.5, 1.0, 1.25, 2.0} × e_pred(64); and the secondary arm (n = 64, q = 0.1, e ∈ {1.25, 2.0} × e_pred, final round not exempt).
3. Even this grid needs about 50 CPU-hours, twice the budget. The protocol has no further step, so nothing more is dropped; the runs are split between this machine and remote runs instead.
4. **Consequences for the tests.** H1 is tested at 2 cells, (n, q) = (64, 0.1) and (4, 0.1), Holm over 2. H2 is tested at q = 0.1 only; "for both q" cannot be checked. H3 and the descriptive outcomes use the 10 fined conditions.
5. The dropped conditions remain in the runner (`--include-dropped`). If they are run later, they are reported as an extension outside this pre-registered grid, not as part of H1–H3.

*Other reading, recorded for transparency:* if "24 h of total CPU on the GitHub Actions lanes" meant 24 h of wall clock on three 4-core lanes (288 core-hours), the full grid would fit at R = 30 (188 h) and not at R = 50. The literal reading (summed CPU time) is used.

**Run split.** This machine runs the analysed grid with 7 processes in priority order (baselines; n = 64 at q = 0.1; n = 4 at q = 0.1; secondary arm) and stops starting new runs after 5.5 h (estimated 8.2 h for the whole analysed grid at this load). Whatever has not finished is listed in `remaining_runs.md` with ready commands and reported in `results.md` as pending.

### Amendment 4 (14 October 2026, before any grid run finished; no grid result exists)

The local grid launched under Amendment 3 stopped before completing any run (`results/runs/claude_r1a_v1/grid_log.jsonl` is empty). The grid now runs on GitHub Actions instead (workflow `r1a-willis-cpr.yml`; results branch `r1a-results`).

The repository is public, so CI CPU time is free.

- **The confirmatory grid and the tests stay exactly as in Amendment 3:** 12 conditions, R = 20, with H1 at (64, 0.1) and (4, 0.1) and H2 at q = 0.1.
- **The conditions Amendment 3 dropped** (q = 0.25; e = 0.75 and 1.5 × e_pred) are also run, R = 20. As Amendment 3, item 5 already states, they are reported as an **extension outside the pre-registered tests**, not as part of H1–H3.

Jobs are one condition × 10 runs (runs 0–9 and 10–19), with 4 processes each, and every run is its own file. Runs are skipped if their file already exists, so lanes cannot overwrite each other (lesson from the L4 data note, F8).
