# LLM inspectors inside Harvest RL: results (23 Nov 2026)

Each run trains 7 IPPO agents for 1e8 steps against a random overseer whose checks make one LLM's measured mistakes, used as fixed coin flips: α is the chance a check misses a violation, and β_eat the chance it flags an honest harvest. The values are the w = 1 numbers from the inspector ladder (`SUMMARY.md`, `results/tables.md`). Job list: `experiments/llm_overseer/runs_llm.tsv` (40 runs, all done). Design and predictions: `protocol.md` §4 and §6, `README.md` §4.

- Penalty per catch P = 7c/k, so a perfect inspector gives an average penalty of c apples per over-harvest.
- Each value is the mean of the last 10 updates for one seed; tables show the mean over seeds [seed range]; n = 3 seeds, except Qwen 7B direct (n = 1 by design, a no-deterrence control).
- Baselines, same trainer and seeds: the perfect inspector (stage 1, `B_random_*`) and the mistakes test (`M_miss50_*`, `F_alarm25_*`; `../socialjax/phase2_ippo/results.md`).
- **C1** predicts an inspector deters like a perfect one at e = (1 − α − β)·c. It classes a cell "protects" if e ≥ 0.75 and "fails" if e ≤ 0.5.
- **"Fines per over-harvest"** = P × catches / over-harvests: what an over-harvest actually costs. With these inspectors it equals (1 − α)·c.

## Main table

Apples a game [seed range]; orchard alive is the share of games with apples left at the end (1.00 unless shown).

| Inspector | α | β_eat | n | C1 e at c = 1 | k = 1, c = 1 | k = 7, c = 1 | C1 e at c = 2 | k = 1, c = 2 | k = 7, c = 2 |
|---|---|---|---|---|---|---|---|---|---|
| perfect (stage 1) | 0 | 0 | 3 | 1.00 | 441 [411, 457] | 477 [468, 483] | 2.00 | 515 [506, 530] | 530 [522, 536] |
| gpt-oss-20B, raw maps | 0.10 | 0.077 | 3 | 0.82, protects | **210 [177, 255], alive 0.63 [0.34, 0.90]** | 361 [336, 393], alive 0.99 | 1.65, protects | 520 [503, 537] | 538 [532, 545] |
| Qwen 3B + reasoning, summary | 0.00 | 0.231 | 3 | 0.77, protects | 380 [336, 426] | 480 [462, 492] | 1.54, protects | 486 [470, 507] | 521 [519, 522] |
| Qwen 7B + reasoning, raw maps | 0.467 | 0.038 | 3 | 0.50, fails | 69, dead | 68, dead | 0.99, protects | 389 [359, 437] | 494 [487, 498] |
| Qwen 7B, raw maps | 1.00 | 0.038 | 1 | −0.04, fails | 68, dead | 68, dead | −0.08, fails | 68, dead | 67, dead |
| mistakes test: misses 50% | 0.5 | 0 | 3 | 0.50 | 69, dead | 68, dead | 1.00 | 342 [310, 361] | 479 [474, 481] |
| mistakes test: false alarms 25% | 0 | 0.25 | 3 | 0.75 | 382 [357, 401] | 482 [466, 497] | 1.50 | 470 [457, 488] | 509 [500, 516] |

"Dead" means the orchard died in every game of every seed (69 or fewer apples, against 68 with no overseer).

Other measures, mean over seeds:

| Inspector | Fines per over-harvest, c = 1 / c = 2 | Over-harvests as % of eats, k = 1 / k = 7 at c = 1 | Wrong flags a game, k = 1 / k = 7 at c = 1 | Wrongful fines per legal apple, c = 1 / c = 2 |
|---|---|---|---|---|
| perfect | 1.00 / 2.00 | 11% / 7% | 0 / 0 | 0 / 0 |
| gpt-oss-20B, raw | 0.90 / 1.80 | 32% / 17% | 2 / 23 | 0.08 / 0.15 |
| Qwen 3B + reasoning, summary | 1.00 / 2.00 | 17% / 8% | 10 / 102 | 0.23 / 0.46 |
| Qwen 7B + reasoning, raw | 0.53 / 1.07 | 43% / 41% | 0 / 2 | 0.04 / 0.08 |
| Qwen 7B, raw (n = 1) | 0 / 0 | 48% / 46% | 0 / 1 | 0.04 / 0.07 |

## Did C1's protect/fail classes hold?

**In 15 of 16 cells, yes.** The protocol predicted that every "protects" cell keeps the orchard and every "fails" cell ends near 68 apples with the orchard dead.

| Prediction | Cells | Held |
|---|---|---|
| "fails" ends dead at about 68 apples | Qwen 7B + reasoning at c = 1; Qwen 7B direct at c = 1 and 2; each at k = 1 and 7 (6 cells) | 6 of 6 |
| "protects" keeps the orchard | gpt-oss-20B and Qwen 3B + reasoning at c = 1 and 2; Qwen 7B + reasoning at c = 2; each at k = 1 and 7 (10 cells) | 9 of 10. **gpt-oss-20B at k = 1, c = 1 failed:** 210 apples, orchard alive in 63% of games (34–90% by seed). At k = 7 it held (alive 97–100%), but at 361 apples, well below the perfect inspector's 477 |
| at the same (α, β), k = 1 collects clearly fewer apples than k = 7 | the 5 surviving pairs | 4 of 5 (210 vs 361; 380 vs 480; 486 vs 521; 389 vs 494). gpt-oss-20B at c = 2 is the exception: 520 vs 538, with seed ranges overlapping |

The one failure is informative. **C1 ranks gpt-oss-20B (e = 0.82) above Qwen 3B + reasoning (e = 0.77), but the learners rank them the other way round, at both k:** 210 vs 380 apples at k = 1, and 361 vs 480 at k = 7. Misses and false alarms do not trade one for one.
- **A miss** lowers what an over-harvest costs: gpt-oss-20B's 10% misses make it 0.90 apples instead of 1.
- **A false alarm** fines an honest harvest. It does not make over-harvesting cheaper. Qwen 3B + reasoning never misses, so an over-harvest still costs the full 1 apple, and its 23% false alarms only fine legal eating. A plausible reason this barely weakens deterrence: in Harvest the alternative to taking an apple illegally is often to wait, not to take a legal one, so a fine on legal eating doesn't push agents towards the illegal apple. This is an interpretation; nothing here tests it directly.
- **The same split is in the mistakes test, read again.** False alarms of 25% at c = 1 (C1 e = 0.75) kept the orchard because over-harvests still cost the full 1 apple. Qwen 3B + reasoning (β 0.23) reproduces that arm almost exactly: 380 vs 382 at k = 1, and 480 vs 482 at k = 7. Qwen 7B + reasoning (α 0.47) reproduces the 50%-miss arm: dead at c = 1; at c = 2, 389 vs 342 at k = 1 and 494 vs 479 at k = 7, slightly better, as its slightly lower miss rate predicts.

**So the protect/fail line is better drawn on the fine an over-harvest actually costs, (1 − α)·c, with a threshold that rises as checks get fewer and bigger:**

| Fine per over-harvest | k = 1 (penalty 7c per catch) | k = 7 (penalty c per catch) |
|---|---|---|
| 0–0.53 | dead (Qwen 7B, 7B + reasoning at c = 1, misses 50% at c = 1) | dead (same) |
| 0.90 (gpt-oss-20B, c = 1) | **partly dead** (alive 34–90%), 210 apples | alive, 361 apples |
| 1.00 and above | alive in every run | alive in every run |

The decentralised runs place the same line between 0.75 and 0.88 at k = 2–4 (`../socialjax/DECENTRALISED_RESULTS.md`).

## The cost to honest agents (C2)

Wrongful fines per legal apple equal β_eat × c at both k = 1 and k = 7: 0.08 / 0.15 for gpt-oss-20B, 0.23 / 0.46 for Qwen 3B + reasoning, at c = 1 / 2. That is built in, because the false alarm is a coin flip on each checked honest harvest. What learners add is how they respond. With Qwen 3B + reasoning at c = 2 the agents still collect 486–521 apples and keep the orchard, but wrongful fines take about 45% of what they collect (0.46 apples per legal apple). Protection survives a high β; the agents' welfare does not.

## Compared with the stage 1 and mistakes-test baselines

- **gpt-oss-20B on raw maps is a perfect inspector at c = 2** (520 and 538 apples, against 515 and 530). At c = 1 it costs 231 apples at k = 1 and 116 at k = 7. In the ladder it was the smaller of the two models rated "protects" on raw maps. In RL it protects only with penalty headroom.
- **Qwen 3B + reasoning on summaries is as good as perfect at k = 7** (480 vs 477 at c = 1; 521 vs 530 at c = 2). At k = 1 it falls behind (380 vs 441; 486 vs 515), because its large, frequent wrongful fines make the reward noisy.
- **Qwen 7B + reasoning on raw maps** behaves like the 50%-miss arm: it needs c = 2 to protect.
- **Qwen 7B direct on raw maps** is no overseer at all (n = 1, as designed).

## Run notes

- All 40 runs reached 390 updates; 18–42 min each (1,050–2,535 s).
- gpt-oss-20B at k = 1, c = 1 varies a lot across seeds (orchard alive 0.34 / 0.66 / 0.90). Only `summary.json` was pulled, not the learning curves, so whether these runs were still improving at 1e8 steps is unknown.
- In every run agents 0 and 1 collect about 5 apples more than the other five (for example 40, 40, 35, 35, …). This also shows when the orchard is dead (14 vs 8), so it is a spawn-position effect, not anything the inspector does.
- These runs use α and β as fixed coin flips. A real LLM misses the hard records more often, and learners could find and exploit that (`protocol.md` §8). That is not tested here. **Update 23 Nov:** the pattern is measured in `error_patterns/results.md`. On learners' records gpt-oss-20B's average α is 0.215–0.288, not 0.10, and a 36-run job list tests the pattern against matched coin flips.
