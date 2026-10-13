# Results L4: expected fine vs fine size, out-of-sample threshold, transfer to reworded rules, no-audit control

- **Protocol:** `protocol.md` in this folder (Note 1; Amendment 1: H2 for gpt-oss-120b only).
- **Predictions:** frozen before any stage-2 game (`prediction.json`). F* = 30 t for both models and both wordings.
- **Data:** `claude_l4_v1/` on the `l2-results` branch. gpt-oss: 190 games. Nemotron: 70 planned games, plus 3 stray matched-cell games from before the budget cut, which are excluded.
- **Analysis:** `experiments/oversight/analyze_l4.py`, written before any game and run on 13 Oct after every planned game was complete. Summary: `runs/claude_l4_v1/l4_summary.json`.

## Over-take rate by cell

The over-take rate is the share of a model's decisions above its allowance, with 95% population-bootstrap intervals. All populations are new: the L4 seed bases, not L2/L3's.

| Cell | q | F (t) | e (t) | gpt-oss-120b | Nemotron 3 Super |
|---|---|---:|---:|---:|---:|
| N0: nobody audits | — | — | — | 30.2% [26.6, 33.8] | 46.0% [43.5, 48.5] |
| E0 | 1/6 | 0 | 0 | 57.5% [53.2, 61.5] | 64.3% [60.3, 67.8] |
| P0 (paraphrase) | 1/6 | 0 | 0 | 75.2% [73.1, 77.8] | 65.0% [62.3, 67.6] |
| Q3_6 | 1/3 | 6 | 2 | 40.7% [38.7, 43.0] | — |
| E12 | 1/6 | 12 | 2 | 32.1% [29.2, 34.7] | — |
| Q12_24 | 1/12 | 24 | 2 | 27.9% [24.9, 30.8] | — |
| Q3_9 | 1/3 | 9 | 3 | 9.3% [7.1, 11.8] | — |
| E18 | 1/6 | 18 | 3 | 9.2% [7.2, 11.2] | — |
| Q12_36 | 1/12 | 36 | 3 | 10.9% [9.1, 12.8] | — |
| E24 | 1/6 | 24 | 4 | 0.9% [0.4, 1.4] | 7.4% [4.2, 12.3] |
| E30 | 1/6 | 30 | 5 | 0.2% [0.0, 0.6] | 1.8% [1.0, 2.5] |
| P24 | 1/6 | 24 | 4 | 2.2% [1.4, 3.2] | 19.9% [11.4, 29.4] |
| P30 | 1/6 | 30 | 5 | 1.1% [0.5, 1.9] | 1.9% [1.5, 2.2] |

## Hypotheses

| | gpt-oss-120b | Nemotron 3 Super |
|---|---|---|
| H1a: deterred at F* = 30 t (≤ 5%) | **holds** (0.2%) | **holds** (1.8%) |
| H1b: not deterred one step below (24 t > 5%) | **fails** (0.9%): stops one step early again, as in L3 | **holds** in point estimate (7.4%; Holm p = 0.17) |
| H3a: under the paraphrase, deterred at 30 t | **holds** (1.1%) | **holds** (1.9%) |
| H3b: under the paraphrase, not deterred at 24 t | **fails** (2.2%), same pattern as the explicit wording | **holds** (19.9%) |
| **H2, e = 2 (primary)**: d = rate(q = 1/12) − rate(q = 1/3) | **severity**: d = −12.9 points [−16.4, −9.3]; minimum detectable effect 5.2 | not run (Amendment 1) |
| H2, e = 3 (secondary) | **expected fine**: d = +1.6 points [−1.4, +4.5], inside ±10 | not run |
| H4 (descriptive): N0 − E0 | −27.3 points [−33.2, −21.1] | −18.3 points [−23.3, −12.8] |

## What it means

1. **The prediction replicated on new games, and transferred to reworded rules.** Both models stopped by the predicted 30 t on populations the prediction never saw, under both wordings. The fine detail replicated too: gpt-oss again stopped one step early, at 24 t, under both wordings, while Nemotron stopped at 30 t.
2. **Expected fine, mostly, with a pull from fine size.** At an expected fine of 3 t, the three mixes of audit probability and fine gave the same over-taking (9–11%). At 2 t, the same expected fine deterred more when it came as a rare audit with a large fine (28%) than as a frequent audit with a small fine (41%), with the 1-in-6 mix in between (32%). So the expected fine sets the level, but where deterrence is partial, a larger fine counts for more than a higher audit chance. This is the opposite of the "certainty beats severity" pattern often reported for people. It fits risk aversion (a large, unlikely fine weighs more) or the salience of a large number. One more factor at e = 2: the frequent-audit cell has the smallest fine (6 t), and small fines were seen to raise over-taking in L2 (fine as price). This test cannot separate these explanations.
3. **Mentioning audits with a 0 t fine raised over-taking.** With rules that never mention audits, over-taking was 30% (gpt-oss) and 46% (Nemotron). With audits that carry no fine, it was 58% and 64%. The earlier "silent" control (which still mentioned a 0 t fine) therefore was not a no-oversight baseline. Announcing a check with a zero penalty looks like a signal that over-taking is acceptable, in line with the fine-as-price effect.
4. **Calibration was stable.** On the new populations, the observed excess was 4.45 t for gpt-oss and 4.68 t for Nemotron (L2: 4.48 and 4.74), and the no-fine over-take rate was 57.5% and 64.3% (L2: 55% and 65%).

## Caveats

- H2 rests on one model (gpt-oss), with 20 populations.
- The e = 2 result is the pre-registered primary; e = 3 is secondary.
- Nemotron's three stray matched-cell games (population 7) are not analysed.
