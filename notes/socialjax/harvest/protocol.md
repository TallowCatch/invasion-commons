# Harvest with decentralised overseers: scripted-bot protocol

Written 22 Nov 2026, **before the sweep ran**. Code: `experiments/socialjax/run_decentralised_bots.py --game harvest` and `--game harvest_large`; overseer in `overseer/socialjax/decentralised.py`, bots in `overseer/socialjax/game_bots.py` (Harvest bots in `harvest_bots.py`).

## The question

The overseer so far is central: it checks k of the 7 agents anywhere on the map. Here the map is split into R regions with one overseer each, and the overseers share the same total budget of k checks per step (k/R each; when k is not a multiple of R, the spare checks go to randomly chosen regions each step). An overseer can only check agents standing in its region. Does splitting the overseer cost catches, and can agents exploit it?

## Setting

- SocialJax Harvest Open, 7 agents, individual rewards, 999 steps, the rule m = 3 (don't eat an apple that leaves fewer than 3 within two squares).
- Each check sees only this step (window 1); a perfect inspector (no misses, no false alarms).
- Consequence: a caught agent is **frozen for 50 steps** (bots ignore reward penalties, so the freeze is the consequence they feel). Whether rule-breaking pays under a reward penalty is worked out afterwards from the catches, at an average penalty of 1 apple per violation (a penalty of 7/k apples per catch).
- Regions: vertical strips (the map is mirror-symmetric, so 2 strips are two copies of the same orchard). R = 2 and R = 4.
- Larger map (`harvest_large`): the map tiled twice side by side (16 x 44), 14 agents, the same rule. R = 2 strips are two full copies of the normal map.

## Conditions (at each k, the same total budget)

| Name | Who checks | How it chooses |
|---|---|---|
| none | nobody | – |
| central random | one overseer, whole map | k agents at random |
| central history | one overseer | agents caught before, then the longest unchecked |
| R random | one per region | k/R of its own agents at random |
| R history | one per region, independent | its own catches and its own checks only |
| R history+share | one per region | as above, and every step sends its suspicion scores to neighbouring regions (0.25 cost units per message); each keeps the higher score |
| R history+share+coverage | one per region | also shares when it last checked each agent; each overseer keeps the most recent check |

- Harvest: k = 1, 2, 4; R = 2, 4; 31 conditions, 10 seeds each.
- Harvest large: k = 2, 4 (the same checks per agent as k = 1, 2 on the small map); R = 2, 4; 21 conditions, 5 seeds each.

## Populations

- 7 honest (restrained);
- 1 greedy + 6 honest (one fixed cheater);
- 7 sneaky (greedy once no overseer has checked it for 20 steps);
- 7 migrator (new): greedy only in regions whose own overseer has not checked it for 20 steps, honest elsewhere, and drawn to apples in those regions. Under a central overseer it is the sneaky bot (a test checks this).

Large map: 14 honest; 2 greedy + 12 honest; 14 sneaky; 14 migrator.

## Measured

Apples eaten, orchard alive at the end, violations, the share of violations caught, checks used and checks left unused (a region with fewer agents than its budget), check cost, communication cost, and per region: checks, catches, violations, detections and misses (by the region the violation happened in).

## Predictions (written before the runs)

- **H1, wasted budget.** Split overseers waste checks when their region has fewer agents than checks. With 7 agents: under 5% of the budget for R = 2, and 10–30% for R = 4 at k = 4. Central overseers never waste a check.
- **H2, random checks lose little.** Against 1 greedy + 6 honest, decentralised random checks catch about the same share of violations as central random (k/7: 14%, 29%, 57% at k = 1, 2, 4), within 5 percentage points, apart from the wasted checks.
- **H3, targeting suffers from local memory.** Central history checks find the greedy agent and keep catching it (week 2: k = 2 saved the orchard in 5 of 5 runs). Independent regional history checks each have to catch it themselves after it crosses into their strip, so they catch a smaller share than central history and save the orchard less often. Sharing suspicion with neighbours recovers most of the gap (within 10 points of central).
- **H4, sneaky bots don't care.** Sneaky bots react to being checked by anyone, so decentralised and central overseers give the same result at the same budget (week 2: random k ≥ 2 or history k ≥ 1 keeps the orchard alive).
- **H5, migrators exploit local overseers.** Migrators break the rule at least twice as often under decentralised random checks as under central random checks at k = 2, because a region that has not seen an agent for a while has not checked it either. Independent history checks blunt this (a newcomer is "long unchecked" in its new region, so it is checked first). **Sharing coverage makes it worse**: neighbours tell the new region that the migrator was just checked, so it is skipped, which is exactly when the migrator cheats.
- **H6, honest bots.** No difference between conditions: about 500 apples, orchard alive, violations near 0.
- **H7, communication.** R = 2 strips: 2 messages per step (0.5 cost units per step); R = 4 strips: 6 messages (1.5 per step). Checks cost k per step, so sharing costs 12–150% of the check budget, depending on k and R.
- **H8, larger map.** The same direction, with less wasted budget (more agents per region).

## What would change the plan

- If H2 fails (random checks lose much more than the wasted share), the split budget itself is the problem, not memory, and the GPU runs should include an R = 2 random arm at every k.
- If H5 fails (migrators gain nothing), local memory is not exploitable by this bot, and the learned-agent test (do RL agents learn to migrate?) becomes the key GPU question.
