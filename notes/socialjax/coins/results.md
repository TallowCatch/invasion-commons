# Coins with central and decentralised overseers: scripted-bot results

**Setup.** Run on 22 Nov 2026 under `protocol.md`, which was written and committed before the sweep.
- Raw runs (git-ignored): `results/runs/socialjax_decentralised/coins.json`.
- Figure: `fig_decentralised.png`.
- Code: `experiments/socialjax/run_decentralised_bots.py --game coins`.

The sweep took 73 s on a laptop CPU.

The bots are scripted and do not learn, so this is a preview, not a paper result. Each cell is 10 seeds. Numbers are in coins: a coin pays its taker 1, and a stolen coin also costs its owner 2. A caught player is frozen for 20 steps. The 2 regions are the top and bottom halves of the map. Each condition at a given k has the same total budget of k checks per step.

## The short version

1. **Coins is a clear dilemma, and the rule is cleanly checkable.**
   - With no overseer, a greedy player nets 117 coins against an honest partner's −21 (the pair: 97). Two honest players get 165, and two greedy players get −1.
   - The wrapper reads thefts off the rewards exactly (unit test).
2. **With only 2 players, splitting the overseer wastes about a quarter of the budget.** When both players stand in the same half, the other half's check has nobody to look at: 24–31% of checks go unused.
3. **That waste is enough to make theft pay again.** At an average penalty of 1 coin per theft (break-even for a central random overseer):
   - with central random checks, the greedy player nets 81 coins, the same as being honest (83);
   - with 2 regional overseers at the same budget it nets 86–88, so theft pays a little again.
4. **Repeat-offenders-first checks work centrally and much less when split.** At k = 1:
   - central catches 89% of thefts and leaves the thief 62 coins;
   - 2 independent halves catch 41%;
   - sharing suspicion lifts that only to 43–47%. With one player per half, there is rarely a neighbour with anything to share.
5. **Migrators steal more under split overseers**, but only a few times a game. Central: 0.8 thefts a game at k = 1; split: 4–7.
   - Here too, sharing coverage makes it worse at k = 2: 0.9 → 4.3 thefts.

## Thefts caught, and the thief's coins

1 greedy + 1 honest. "Thief's net" is the greedy player's coins after an average penalty of 1 coin per theft, worked out from the catches (2/k coins per catch). Each player gets about 83 when both are honest.

| Overseer | k = 1: caught | k = 1: thief's net | k = 1: pair's coins | k = 2: caught | k = 2: thief's net | k = 2: pair's coins |
|---|---|---|---|---|---|---|
| none | 0% | 117 | 97 | – | – | – |
| central, random | 46% | 81 | 111 | 100% | 78 | 122 |
| central, repeat offenders first | 89% | 62 | 120 | 100% | 78 | 122 |
| 2 halves, random | 37% | 86 | 109 | 67% | 88 | 114 |
| 2 halves, independent | 41% | 85 | 114 | 83% | 82 | 118 |
| 2 halves, share suspicion | 43% | 83 | 110 | 96% | 78 | 120 |
| 2 halves, share suspicion + coverage | 47% | 82 | 115 | 97% | 79 | 123 |

Unused checks with 2 halves: 26–28% of the budget.

With k = 2, a central overseer checks both players every step. Two half-overseers with one check each match that only when the players are in different halves.

## Other populations

| Overseer | 2 greedy: pair's coins | 2 migrators: thefts per game | 2 sneaky: thefts per game |
|---|---|---|---|
| none | −1 | 84 | 84 |
| central random, k = 1 | 0 | 0.8 | 0.8 |
| 2 halves random, k = 1 | 5 | 4.2 | 0.8 |
| 2 halves, independent, k = 1 | 1 | 5.8 | 0.8 |
| 2 halves, share suspicion + coverage, k = 1 | 3 | 7.3 | 0.8 |
| central random, k = 2 | −5 | 0.8 | 0.8 |
| 2 halves random, k = 2 | 0 | 1.8 | 0.8 |
| 2 halves, independent, k = 2 | −2 | 0.9 | 1.3 |
| 2 halves, share suspicion + coverage, k = 2 | −4 | 4.3 | 1.3 |

- Sneaky bots steal only after 20 unchecked steps. With 2 players and at least one check every other step, that almost never happens, centrally or not. The 0.8 thefts are coins appearing under a step, which honest bots also take by accident.
- Two greedy players: the overseer freezes them in turn, and the pair's total stays near 0 whatever the overseer. With both stealing, the freeze only moves coins between them.

## Predictions against results

| Prediction | Result |
|---|---|
| K1: two thieves are worse than one; each nets less than a lone thief | **Yes.** Pair −1 vs 97; each about 0 vs 117 |
| K2: central random k = 1 catches about half; history over 80%; k = 2 all | **Yes.** 46%, 89%, 100% |
| K3: split k = 1 catches 35–45% with 20–30% of checks unused; split k = 2 catches 70–85% | **Mostly.** k = 1: 37% with 26% unused (yes). k = 2: 67% (just below the range) |
| K4: migrators steal more under split overseers | **Yes**, but few: 4–7 vs 0.8 a game at k = 1 |
| K5: welfare rises with the catch share | **Yes.** Pair's coins: 97 with no overseer, 109–115 at 37–47% caught, 120–123 at 89–100% |
| K6: theft stops paying at an average penalty of 1 coin per theft | **Yes centrally** (thief 81 vs honest 83). **Not quite with split overseers** (86–88), because of the wasted checks |

## What this means

- Coins is the cleanest test of the rule-and-penalty part of the overseer (claim C1 in `../../phd_plan/2026-11-21_general_formulation.md`). One theft gains exactly 1 coin, so the break-even average penalty is known in advance: 1 coin per theft.
- **Decentralisation is the wrong design for 2 players.** It wastes a quarter of the budget. For learning agents, the interesting Coins runs are central random against central history, and the break-even penalty. The GPU list still includes the split arms at the same budget, so the waste can be measured with learners.
- Coins is limited to 2 players in SocialJax, so it can't test more agents per region.
