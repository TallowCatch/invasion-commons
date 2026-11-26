# X10 step 1: scripted bots in Clean Up under the stricter rule W = 25 (26 Nov 2026)

Written before any X10 learner run was submitted. Prediction being checked (`../05_formulation.md` §5.6,
Proposition 5.11): the X10 learner prediction applies only if agents that keep the rule keep the orchard alive to
the end of the episode in at least 90% of games. Under the old rule (W = 100) this was 38%.

**Rule.** An agent breaks the rule if it eats an apple while pollution is above 0.2 and it has not cleaned (fired
the cleaning beam and removed dirt) in the last W steps. W = 25 asks for cleaning four times as often as W = 100.

**How measured.** `python -m experiments.socialjax.measure_lpgf --bots --bot-env cleanup --clean-window 25
--episodes 64` (laptop CPU, about 2–3 minutes per population; the `--clean-window` flag was added to the bot path
for X10). The scripted bots are those of H5 (`overseer/socialjax/game_bots.py`, `cleanup_policy`): the honest bot
reads W from the rule, so under W = 25 it goes to clean whenever the river is polluted above the threshold and,
while eating would break the rule, does not step onto apples. No overseer. 1,000-step episodes. Brackets are 95%
bootstrap intervals over episodes. "Pollution crossings" are the times pollution rises to 0.4, where apples stop
regrowing. LPGF (a_W) is the share of those crossings with no violation by anyone in the rule's own W-step window.

The W = 100 rows were run again with the same code (32 episodes) as a check: they reproduce H5 exactly (38% alive
for 7 honest bots).

| population | rule | games | violations per game | orchard alive at end | pollution crossings per game | share of steps polluted (≥ 0.4) | mean pollution | LPGF (a_W) |
|---|---|---|---|---|---|---|---|---|
| 7 honest | W = 25 | 64 | 2.6 [2.2, 3.1] | **100% (64 of 64)** | 0.13 [0, 0.38] | 1.0% | 0.16 | 1.00 (8 of 8 crossings) |
| 1 greedy + 6 honest | W = 25 | 64 | 137 [133, 141] | 100% (64 of 64) | 0 | 0.7% | 0.17 | no crossings |
| 7 greedy | W = 25 | 64 | 0 (no apples ever grow) | 0% | 0 (polluted from the start) | 100% | 0.83 | — |
| 7 honest | W = 100 | 32 | 0 | 38% [22, 53] | 6.1 [4.7, 7.7] | 30% | — | 1.00 |
| 1 greedy + 6 honest | W = 100 | 32 | 65 | 0% | 4.5 | 72% | — | 0.06 |
| 7 greedy | W = 100 | 32 | 0 | 0% | 0 | 100% | — | — |

**Answer.** The condition holds. Under W = 25 the rule-keeping population keeps the orchard to the end in all 64
games (95% interval for the true rate, Wilson: 94–100%), against 38% under W = 100. Pollution reaches the
no-regrowth level only 8 times in 64 games (0.13 per game, against 6.1). Every one of those 8 crossings still had
no violation in the 25-step window, so local-pass/global-fail is not zero, but it is now rare and does not cost
the orchard. So the X10 learner prediction applies: learners at ρ ≥ ρ* under W = 25 should keep the orchard.

**Two things to keep in mind.**
- **The honest bots are almost, not exactly, rule-keeping under W = 25:** 2.6 violations per game (at most 8),
  against 0 under W = 100. A run with the bots' 5% random moves turned off still gave 1.9 per game (16 games), so
  these are not random steps onto apples. They are most likely edge cases where pollution crosses 0.2 during the
  step in which the bot eats (the bot decides on the state before the step). That is at most a few eats in a game
  of several hundred, so the population is a fair stand-in for "agents who keep the rule".
- **Under W = 25 one free-rider no longer kills the orchard.** With 1 greedy + 6 honest bots the orchard survived
  every game (0% under W = 100), because six agents cleaning every 25 steps keep pollution low (mean 0.17) even
  with one agent never cleaning. The greedy bot breaks the rule 137 times a game, so the overseer still has
  something to catch, but the commons is far more forgiving under the stricter rule. For the learner runs this
  means survival will be easier to reach under W = 25 than under W = 100 even at modest compliance; the learner
  comparison should read compliance (violations, caught share) as well as survival.

Summaries: `bots_summary.json` (here). Raw per-episode output: `results/x10/bots_W25.json`, `results/x10/bots_W100.json` (laptop, not committed); the honest-bot
check without random moves: `results/x10/honest_explore.log`.
