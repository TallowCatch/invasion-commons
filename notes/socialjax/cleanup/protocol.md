# Clean Up with central and decentralised overseers: scripted-bot protocol

Written 22 Nov 2026, **before the sweep ran**. Code: `experiments/socialjax/run_decentralised_bots.py --game cleanup`; rule in `overseer/socialjax/games.py` (`CleanupRules`), bots in `game_bots.py` (`cleanup_policy`).

## The game

SocialJax Clean Up (`clean_up.py` at commit 18e2171), 7 agents, individual rewards (an apple pays 7 to its eater), 19 x 28 map:
- the orchard is the bottom 4 rows and the river and its banks are the top 6 rows, about 10 rows apart;
- each step, apples regrow on empty orchard cells with chance 5% x (1 − pollution / 0.4), so none regrow once pollution (the share of the 167 river cells that are dirty) reaches 0.4;
- after step 50, a new piece of dirt appears with chance 50% per step. At the start 79 of 167 cells (47%) are dirty, so nothing grows until someone cleans;
- the cleaning beam (action 8) removes dirt from the 4 cells in front of the agent. Cleaning pays nothing; eating pays.

So cleaning is a public good and eating without cleaning is free-riding.

## The rule, and why

**Don't eat an apple while pollution is above 0.2 unless you cleaned (fired the beam and removed at least one piece of dirt) in the last W = 100 steps.**

- **It targets free-riding**, the dilemma in Clean Up: someone who eats but leaves the cleaning to others.
- **It is checkable from an agent's record.** The record holds the agent's eating, its beams, and the river's pollution when it ate. Whether a beam removed dirt is read from the map before and after the step; dirt only ever leaves by cleaning.
- **The 0.2 threshold:** below it the orchard regrows at more than half speed and cleaning is not urgent, so eating is never a violation there.
- **W = 100:** an honest agent needs about 10–15 steps to walk between the orchard and the river each way, plus time to clean. W = 100 leaves time for a full round trip and a long spell of eating. A tighter W would make honest agents break the rule by accident. These two numbers were fixed before any overseer was run and are not tuned on the results.
- **This is the "hard to see in one check" case (claim C4 in `../../phd_plan/2026-11-21_general_formulation.md`).** One check sees an agent eat, but proving the violation needs the 100 steps before it.
  - A check that reads w < W steps can never prove a violation.
  - With `presume=True`, the overseer flags any eating whose visible part of those 100 steps shows no cleaning. That catches every free-rider but also flags honest agents who cleaned before the look began.
  - Each check costs 1 + 0.01 w units, so a longer look costs more.

## Bots

Same five kinds as the other games (`game_bots.py`):
- **Honest:** when pollution is above 0.2 and it last cleaned 70 or more steps ago, it walks to the river and cleans until it has removed dirt 5 times, then goes back to eat. It never steps on an apple while eating would break the rule.
- **Greedy:** never cleans; eats the nearest apple.
- **Sneaky:** free-rides (stops cleaning, eats) while no overseer has checked it for 20 steps.
- **Migrator:** free-rides only in the strip whose overseer has not checked it for 20 steps.

**Seen in development runs, not predictions** (3–5 seeds, no overseer unless stated):
- 7 honest: about 860 apples a game (range 690–1,080), mean pollution about 0.35, about 55 effective cleans each.
- 1 greedy + 6 honest: 190–420 apples, and the greedy bot breaks the rule 40–80 times.
- 7 greedy: 0 apples. Nobody cleans, so nothing ever grows.
- 7 sneaky under 1 random check per step: 0 violations, but 627 apples. Their free-riding spells are shorter than 100 steps, so they never break the rule, but they skip cleaning.

## Conditions

999 steps, perfect inspector, a caught agent is frozen for 50 steps (it can't eat).

- **Main sweep, window w = 150** (a check can prove eating up to 50 steps old):
  - none;
  - central random and central history, at k = 1 and 2;
  - R = 2 vertical strips: random, history, history + shared suspicion, and history + shared suspicion and coverage. Each strip holds half the orchard and half the river.
- **How long a look has to be:** central random, k = 2.
  - w = 1, 50, 100, 150, 200, proving only;
  - w = 10, 50, 100 with `presume=True`.

20 conditions, 10 seeds. Populations: 7 honest; 1 greedy + 6 honest; 7 sneaky; 7 migrator.

## Predictions (written before the runs)

- **U1, the look must cover the rule's window.**
  - Proving only, with 1 greedy + 6 honest at k = 2: no violation is caught for w < 100.
  - At w = 100, about 29% are caught (only eating in the current step can be proven, and a given agent is checked with chance 2/7).
  - At w ≥ 150, nearly all are caught (a violation stays provable for 50 steps, and the chance of going unchecked that long is (5/7)^51, under 0.1%).
  - Check cost rises from 1.01 to 3 units per check across this range.
- **U2, presuming guilt catches everything and punishes honest cleaners.**
  - With `presume` at w = 10, every free-ride is caught.
  - Most of the honest agents' eating while polluted is also flagged (more than 100 wrong flags a game).
  - Freezing them cuts apples below the level with no overseer.
  - At w = 100, wrong flags fall a lot (honest bots clean every 70–100 steps) but stay above 0.
- **U3, the overseer can stop eating, not make anyone clean.** With 1 greedy + 6 honest, a w = 150 overseer cuts the greedy bot's apples by more than half. Total apples rise by less than 25%, because the river stays as dirty.
- **U4, shirking within the rule.**
  - Sneaky and migrator bots break the rule close to 0 times at every k ≥ 1, as in development: checks come more often than every 100 steps.
  - Their harm is skipped cleaning, which shows as fewer apples than 7 honest bots. The rule misses it.
  - This is a limit of a "clean once per W steps" rule, not of the overseer.
- **U5, decentralised as in Harvest.** At w = 150:
  - R = 2 random catches about as many as central random;
  - independent history catches fewer than central history;
  - sharing suspicion recovers most of the gap.
- **U6, honest bots are never flagged when the overseer only acts on proof** (0 wrong flags without `presume`).
