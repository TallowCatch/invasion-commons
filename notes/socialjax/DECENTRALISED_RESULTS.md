# Decentralised overseers with learning agents: results (23 Nov 2026)

IPPO agents trained for 1e8 steps against a central overseer or against 2 or 4 regional overseers that share the same total budget of k checks per step. Job lists: `experiments/socialjax/decentralised_runs.py` (`harvest`, `coins`). Design and commands: `DECENTRALISED_README.md`. Bot preview and predictions: `DECENTRALISED_SUMMARY.md`, `harvest/protocol.md`, `coins/protocol.md`, and tracker findings 12–16.

> **Read this first: these regional overseers hand records over for free.** In `overseer/socialjax/decentralised.py` an agent's record travels with it, and any regional overseer's check reads all of it, including steps spent in other regions. So every regional arm below measures only **where checks can go** (a fixed budget per region while agents crowd unevenly). None of it measures evidence lost when an agent crosses a boundary. That needs `LocalRecordOverseer(records="local")` (`overseer/socialjax/local_records.py`, `locality/results.md` §0), which has not been trained against.

## What ran

| List | Runs | Done | Notes |
|---|---|---|---|
| `harvest` (7 agents) | 69 | 67 | still running: `DH_R2_random_k4_c1.0_seed1`, `DH_R2_history_k4_c1.0_seed1`, so those two cells have 2 seeds |
| `coins` (2 players) | 45 | 45 | |
| `harvest_large`, `cleanup` | 21, 24 | not in this pull | |

- Central random at every (k, c) and central "repeat offenders first" at k = 2, c = 1 come from stage 1 (`B_random_*`, `D_history_k2_c1.0`): the same code path and seeds, so the job list did not repeat them.
- Penalty per catch P = N c / k, so random central checks give an average penalty of c per violation.
- Each value is the mean of the last 10 updates for one seed. Tables give the mean over seeds with the seed range in brackets; n is the number of seeds.
- Harvest runs took 36–42 min each, Coins runs 7–15 min. All runs reached 390 updates.
- **"Fines per over-harvest"** is the fine the agents actually paid per violation: penalty × catches / violations. With a perfect inspector this is P times the share caught. It is the number that matters below.
- Coins reuses Harvest's metric names (checked in `train_ippo_overseer.py` and `games.py`): `apples_eaten` = coins taken, `violations` = thefts (taking the other player's coin), `penalty_apples` = fines in coins. `orchard_alive` means nothing in Coins and is not reported. "Pair's coins after thefts" = coins taken − 2 × thefts, because a theft gains the thief 1 coin and costs the owner 2.
- These runs used code from before per-agent logging, so there is no per-agent or per-agent-per-region data. Per-region totals (checks, catches, violations) exist.

## Harvest

**Average penalty 1 apple (c = 1)**

k = 2 (penalty per catch 3.5 apples):

| Overseer | n | Apples | Orchard alive | Over-harvests, % of eats | % caught | Fines per over-harvest | Checks unused |
|---|---|---|---|---|---|---|---|
| central, random | 3 | 458 [441, 470] | 1.00 | 9% | 28% | 1.00 | 0% |
| central, repeat offenders first | 3 | 470 [460, 483] | 1.00 | 7% | 29% | 1.00 | 0% |
| 2 strips, random | 3 | 438 [406, 471] | 1.00 | 10% | 27% | 0.95 | 0% |
| 2 strips, repeat offenders first, independent | 3 | 420 [405, 442] | 1.00 | 10% | 25% | 0.88 | 0% |
| 2 strips, repeat offenders first, sharing | 3 | 457 [444, 464] | 1.00 | 8% | 27% | 0.94 | 0% |
| **4 strips, random** | 3 | **76 [75, 77]** | **0.00** | 41% | 18% | **0.65** | 12% |
| **4 strips, repeat offenders first, sharing** | 3 | **175 [149, 211]** | **0.13 [0.02, 0.24]** | 29% | 20% | **0.71** | 9% |

k = 4 (penalty per catch 1.75 apples):

| Overseer | n | Apples | Orchard alive | Over-harvests, % of eats | % caught | Fines per over-harvest | Checks unused |
|---|---|---|---|---|---|---|---|
| central, random | 3 | 476 [454, 493] | 1.00 | 8% | 57% | 1.00 | 0% |
| central, repeat offenders first | 3 | 484 [460, 497] | 1.00 | 6% | 57% | 1.00 | 0% |
| 2 strips, random | 2 | 423 [397, 449] | 1.00 | 12% | 53% | 0.93 | 0% |
| 2 strips, repeat offenders first, independent | 2 | 439 [432, 446] | 1.00 | 9% | 53% | 0.92 | 0% |
| 2 strips, repeat offenders first, sharing | 3 | 444 [414, 464] | 1.00 | 9% | 54% | 0.94 | 0% |
| **4 strips, random** | 3 | **86 [73, 95]** | **0.00** | 37% | 38% | **0.67** | 11% |
| **4 strips, repeat offenders first, sharing** | 3 | **224 [185, 263]** | **0.29 [0.02, 0.67]** | 25% | 41% | **0.72** | 9% |

**Average penalty 2 apples (c = 2):** every arm keeps the orchard in every seed.

| Overseer | k = 2 apples | k = 2 fines per over-harvest | k = 4 apples | k = 4 fines per over-harvest |
|---|---|---|---|---|
| central, random | 527 [517, 540] | 2.01 | 524 [516, 534] | 2.00 |
| central, repeat offenders first | 517 [514, 521] | 2.03 | 522 [515, 531] | 2.01 |
| 2 strips, random | 519 [510, 531] | 1.94 | 522 [518, 527] | 1.88 |
| 2 strips, repeat offenders first, independent | 512 [506, 520] | 1.90 | 524 [520, 526] | 1.89 |
| 2 strips, repeat offenders first, sharing | 518 [510, 533] | 1.92 | 519 [507, 531] | 1.89 |
| 4 strips, random | 510 [495, 520] | 1.59 | 519 [511, 525] | 1.64 |
| 4 strips, repeat offenders first, sharing | 515 [511, 518] | 1.58 | 516 [510, 525] | 1.57 |

n = 3 in every c = 2 cell. Over-harvests are 1–2% of eats everywhere; 4 strips waste 4–8% of checks.

**What it says.**
1. **Four regional overseers lose the orchard at an average penalty of 1 apple, at both k = 2 and k = 4.** Central overseers at the same budget keep it in every seed (458–484 apples). Four strips with random checks end at 76–86 apples, near the no-overseer level of 68. This confirms the early single-seed glance, now on 3 seeds and at k = 4 as well.
2. **The cause is a weaker deterrent, not anything the agents exploit cleverly.** Four strips catch a smaller share of over-harvests (18% vs 28% at k = 2; 38% vs 57% at k = 4), so an over-harvest costs about 0.65–0.72 apples instead of 1. That puts the realised penalty below the learners' threshold.
   - Only part of the loss is wasted checks (9–12% unused). The rest is crowding. Each strip gets k/4 checks whatever the number of agents in it, and over-harvests happen where agents gather. Per-strip catch rates are 18–21% while checks cover 25% of agent-steps.
3. **The cells line up on one scale.** Across these runs, the fine an over-harvest actually costs sorts the outcome:
   - 0.64–0.69 apples: orchard dead in every seed (6 runs);
   - 0.68–0.75: mostly dead, alive in 2–67% of games (6 runs; the best seed, at 0.75, is alive in 67%);
   - 0.88 and above: alive in every game (all 2-strip and central runs).
4. **Two strips cost little.** They lose 1–4 points of catch share. At c = 1 they collect up to about 60 apples less: 420–457 at k = 2 against 458–470 centrally, and 423–444 at k = 4 against 476–484. Seed ranges overlap for random and sharing at k = 2. At c = 2 there is no visible cost. They waste no checks at all: learners never leave a strip empty. The bots wasted 6–20%.
5. **Sharing suspicion: mixed.**
   - At k = 2, c = 1, 2 strips with sharing (457 [444, 464]) beat independent repeat-offenders-first (420 [405, 442]) and match central (470). The seed ranges barely separate.
   - At k = 4 there is no difference (444 vs 439, n = 2 for the independent arm).
   - With 4 strips, sharing plus repeat-offenders-first did better than random (175 vs 76 and 224 vs 86 apples). But no 4-strip independent repeat-offenders-first arm exists, so this cannot be pinned on sharing rather than on the chooser.
6. **Repeat offenders first is no better than random here, centrally or regionally.** All 7 agents share one network, so there is no lasting individual cheater to find (as in stage 1, finding 6).
7. **Communication cost is exactly as designed:** 500 units a game for 2 strips and 1,500 for 4 (0.25 per message, 2 or 6 messages per step).

## Coins

**No overseer** (n = 3): 68.6 [68.0, 69.3] thefts a game. The pair keeps 30 [29, 31] coins after thefts, out of 167 taken.

**Average penalty 1 coin per theft (c = 1), the break-even point** (one theft gains the thief exactly 1 coin):

| Overseer | n | Thefts | % caught | Fines per theft (coins) | Pair's coins after thefts | Checks unused |
|---|---|---|---|---|---|---|
| central, random (k = 1) | 3 | 7.8 [7.5, 8.1] | 50% | 1.00 | 151 | 0% |
| central, repeat offenders first (k = 1) | 3 | 6.7 [6.2, 7.5] | 50% | 1.00 | 153 | 0% |
| **2 halves, random (k = 1)** | 3 | **29.4 [29.1, 29.7]** | 34% | **0.68** | 108 | 31% |
| 2 halves, repeat offenders first, independent (k = 1) | 3 | 25.7 [25.0, 26.2] | 34% | 0.69 | 115 | 28% |
| 2 halves, repeat offenders first, sharing (k = 1) | 3 | 27.1 [26.5, 28.0] | 34% | 0.68 | 112 | 29% |
| central, both checked (k = 2) | 3 | 8.8 [8.0, 9.6] | 100% | 1.00 | 149 | 0% |
| **2 halves, random (k = 2)** | 3 | **28.3 [28.1, 28.6]** | 67% | **0.67** | 110 | 29% |

**c = 2:** every arm cuts thefts to 0.9–1.5 a game (central 0.9; split 1.4–1.5). The split arms catch 35% (k = 1) and 70% (k = 2), so the fine per theft is 1.38–1.40 coins, above break-even. The pair keeps 163–164 coins in every arm. n = 3 everywhere.

**What it says.**
1. **At break-even, one overseer works and two don't.** Centrally, thefts fall from 69 to 7–9 a game. With two half-map overseers at the same budget, thefts stay at 26–29, about 3.5 times as many. This confirms the single-seed glance (25–30 against 7–8) on 3 seeds.
2. **It is the waste, almost exactly.** With 2 players, both are often in the same half, so 28–31% of checks go unused. The share caught falls by about that much (50% → 34%; 100% → 67%), and the fine per theft drops to 0.67–0.69 coins. Theft pays again, and learners find it.
3. **At twice the break-even penalty, splitting costs nothing that matters.** The split overseer's realised 1.4 coins per theft is still above 1.
4. **The chooser and sharing make no difference.** All split arms catch 34–35%. With 2 players sharing one network, there is no individual thief to target.

## Predictions from the bot runs, scored against learning agents

These are the bot protocols' predictions and the bot findings. Neither was written for learners, so this is a comparison, not a pre-registered test.

| Bot prediction or finding | Learning agents | Verdict |
|---|---|---|
| H1 / finding 12: splitting wastes budget (bots: Harvest 4 strips 22–29%, 2 strips 6–20%; Coins 24–31%) | Harvest 4 strips 4–13%, 2 strips 0–1%; Coins 26–32% | held in Coins; held in direction but smaller in Harvest (learners spread out more evenly than bots) |
| H2 / finding 12: random checks lose about the wasted share of catches | Coins: yes (34% vs 50% caught, 31% wasted). Harvest 4 strips: loses 36% (k = 2) and 33% (k = 4) of catches with 11–12% wasted; 2 strips within 1–4 points | held for Coins and 2 strips; failed for 4 strips, where crowding costs more than the waste |
| Bot point 4: splitting hurts even agents that don't exploit regions (sneaky bots 7 → 21 over-harvests) | 4 strips at c = 1 kill the orchard (76–86 apples vs 458–476) | held, and much larger with learners: they respond to the weaker realised penalty |
| H5 / finding 13: migrators over-harvest 5× more under 2 regional random overseers (k = 2) | 2 strips, random: 42 vs 39 over-harvests at k = 2 (1.1×), 49 vs 39 at k = 4 (1.3×, n = 2) | the bot-scale gain did not appear. Learners don't see checks and share one network, so this design can't tell migration apart from crowding |
| H3 / finding 14: independent regional targeting loses to central; sharing suspicion recovers most of it | k = 2, c = 1: 420 → 457 with sharing (central 470). k = 4: no gap closed (439 vs 444; central 484) | partly held (k = 2 only, small n) |
| Bot point 3: at k = 4, local targeting matches or beats central | 2 strips, repeat offenders first, k = 4, c = 1: 439 vs central 484 | failed (no lasting individual cheater with one shared network) |
| H7: messages cost 0.5 per step (2 strips) and 1.5 (4 strips) | 500 and 1,500 a game | held exactly |
| K2: central random k = 1 catches about half the thefts; central history over 80% | 50% and 50% | half held (history gains nothing with one shared network) |
| K3: split random k = 1 catches 35–45%, wastes 20–30%; split k = 2 catches 70–85% | 34%, wastes 30–32%; k = 2 catches 67–70% | held at the edges of the predicted ranges |
| K6 / finding 16: theft stops paying at an average penalty of 1 coin centrally, not under split overseers | central: 69 → 7–9 thefts; split: 26–29 | held, and larger than with bots (bots: thief 86 vs 81 coins) |

## What this means

- **With learning agents, decentralisation at a fixed budget fails exactly when it pushes the realised fine per violation below the threshold.** For Harvest this happened with 4 strips at an average penalty of 1 apple; for Coins with 2 halves at break-even. Where the nominal penalty has headroom (c = 2), splitting costs a few apples at most.
- **A practical rule:** set the penalty from the catch rate the regional design actually achieves (fines per violation), not from k/N. Equivalently, regional budgets that follow where the agents are would remove the crowding loss. That is the "budgets that follow agents" test proposed in `locality/results.md`.
- **The threshold reading agrees with the LLM-inspector runs** (`../llm_overseer/rl_results.md`). There, too, the fine an over-harvest actually expects sorts the outcomes: about 0.65–0.75 apples fails at k = 2–4, and 0.88 or more survives. At k = 1 even 0.90 is only partly enough.
- **What this does not show.** Evidence lost at boundaries (records stay free to read across regions). Whether learners migrate on purpose (no per-agent region logs; checks are hidden; one shared network). The separate-networks and worst-case lists (`worstcase/protocol.md`) and a local-records arm are the tests for those.
