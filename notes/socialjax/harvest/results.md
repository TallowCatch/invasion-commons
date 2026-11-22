# Harvest with decentralised overseers: scripted-bot results

**Setup.** Run on 22 Nov 2026 under `protocol.md`, which was written and committed before the sweep.
- Raw runs (git-ignored): `results/runs/socialjax_decentralised/harvest.json` and `harvest_large.json`.
- Summaries: `*_summary.json` in the same folder.
- Figures: `fig_decentralised.png` (7 agents), `fig_decentralised_large_map.png` (14 agents).
- Code: `experiments/socialjax/run_decentralised_bots.py`, `analyze_decentralised.py`, `make_decentralised_figures.py`.

The sweep took 6.5 min (7 agents) and 4.7 min (14 agents) on a laptop CPU.

The bots are scripted and do not learn, so this is a preview for designing the learning-agent runs, not a paper result. Each cell is 10 seeds on the 7-agent map and 5 on the 14-agent map. "Caught" pools violations and catches over seeds.

**Every condition at a given k has the same total budget:** k checks per step, summed over the regions. Regions are vertical strips. A caught agent is frozen for 50 steps.

## The short version

1. **Splitting the budget wastes checks.** A strip with fewer agents than checks leaves the spare checks unused:
   - 6–20% of the budget with 2 strips;
   - 22–29% with 4 strips, and up to 46% when migrators crowd into one strip.

   Random checks lose roughly that share of their catches.
2. **Agents that cheat where the local overseer has not looked (migrators) gain a lot.** At 2 checks per step:
   - central random checks keep them to 7 over-harvests a game, with the orchard alive in 10/10 runs;
   - 2 independent random overseers allow 37, with the orchard alive in 1/10.
3. **Sharing helps or hurts depending on what is shared:**
   - **suspicion** (who was caught) helps against a fixed cheater: catches rise from 53% to 68% at k = 2;
   - **coverage** (when each agent was last checked) helps a bit more against a fixed cheater (75%);
   - but coverage makes migrators worse: 12 → 27 over-harvests at k = 2, and 1.9 → 17 at k = 4. Neighbours tell the new region that the migrator was "just checked", so it is skipped exactly when it cheats.
4. **Decentralisation hurts even bots that don't exploit regions (sneaky bots).** A crowded strip checks each of its agents less often than a central overseer would, so gaps of 20+ steps without any check appear.
   - k = 2 random: 21 over-harvests a game, against 7 for central random.
   - With repeat-offenders-first checks at k = 1: the orchard survives in 3/10 runs, against 10/10.
5. **With enough budget, local targeting is as good as central targeting or better.** At k = 4, 2 independent strips catch 93% of the greedy agent's over-harvests (central 87%).
   - On the 14-agent map with one greedy agent per half, 2 independent strips beat the central overseer: 77% against 74% caught, and 717 against 623 apples.
   - Each strip's overseer gets to know its own cheater.

## 7 agents: one fixed cheater (1 greedy + 6 honest)

Share of the greedy agent's over-harvests caught, and orchard alive at the end (of 10 runs). Fig. (a)–(c).

| Overseer | k = 1 | k = 2 | k = 4 |
|---|---|---|---|
| central, random | 15% (0/10) | 25% (0/10) | 58% (6/10) |
| 2 strips, random | 11% (0/10) | 30% (0/10) | 51% (4/10) |
| 4 strips, random | 12% (0/10) | 21% (0/10) | 47% (1/10) |
| central, repeat offenders first | 56% (7/10) | 81% (10/10) | 87% (10/10) |
| 2 strips, independent | 20% (0/10) | 53% (5/10) | 93% (10/10) |
| 2 strips, share suspicion | 28% (0/10) | 68% (8/10) | 94% (10/10) |
| 2 strips, share suspicion + coverage | 32% (0/10) | 75% (9/10) | 95% (10/10) |
| 4 strips, independent | 16% (0/10) | 27% (0/10) | 71% (9/10) |
| 4 strips, share suspicion | 19% (0/10) | 35% (0/10) | 82% (10/10) |
| 4 strips, share suspicion + coverage | 18% (0/10) | 39% (1/10) | 88% (10/10) |

**Does over-harvesting pay?** At an average penalty of 1 apple per over-harvest (7/k apples per catch), worked out from the catches; the bots ignore penalties:
- With 2 random checks, the greedy agent nets 11.5 apples centrally and 5.4 with 2 strips.
- The honest agents in the same group get 20 and 27.
- Without an overseer the greedy agent gets 35 and the honest ones 8.

## 7 agents: migrators and sneaky bots

Over-harvests per game, apples eaten, and orchard alive (of 10). Fig. (d)–(e).

| Overseer | 7 migrators, k = 2 | 7 migrators, k = 4 | 7 sneaky, k = 2 |
|---|---|---|---|
| none | 28, 71 apples, 0/10 | – | 28, 71, 0/10 |
| central, random | 6.9, 484, 10/10 | 2.4, 497, 10/10 | 6.9, 484, 10/10 |
| 2 strips, random | 37, 195, 1/10 | 16, 397, 10/10 | 21, 395, 10/10 |
| 4 strips, random | 30, 100, 0/10 | 36, 196, 6/10 | 34, 243, 3/10 |
| central, repeat offenders first | 2.6, 505, 10/10 | 2.7, 499, 10/10 | 2.6, 505, 10/10 |
| 2 strips, independent | 12, 392, 9/10 | 1.9, 461, 10/10 | 8.1, 452, 9/10 |
| 2 strips, share suspicion | 17, 362, 5/10 | 1.9, 461, 10/10 | 7.8, 448, 9/10 |
| 2 strips, share suspicion + coverage | 27, 130, 3/10 | 17, 345, 8/10 | 16, 372, 8/10 |
| 4 strips, independent | 28, 116, 0/10 | 5.2, 210, 10/10 | 10, 452, 10/10 |
| 4 strips, share suspicion + coverage | 30, 108, 0/10 | 27, 99, 0/10 | 12, 445, 8/10 |

Under one central overseer a migrator is the sneaky bot, so the central rows are identical in both columns (as the test `test_migrator_is_sneaky_with_one_overseer` requires).

Why suspicion sharing changes little against migrators and sneaky bots: they are rarely caught (being watched deters them), so there is little suspicion to share.

## 7 agents: what decentralisation costs to run

**Unused checks** (share of the budget), 1 greedy + 6 honest, random checks. Fig. (f).

| Regions | k = 1 | k = 2 | k = 4 |
|---|---|---|---|
| 2 strips | 7% | 6% | 20% |
| 4 strips | 22% | 24% | 29% |

**Communication:** sharing every step at 0.25 cost units per message.
- 2 strips: 2 messages a step, 500 units a game.
- 4 strips: 6 messages a step, 1,498 a game.
- Checks cost 1 unit each, k × 999 a game. So sharing costs as much as half a check per step with 2 strips, and 1.5 checks per step with 4.

7 honest bots: 497–506 apples and the orchard alive in every condition. Their 2.4–2.9 accidental over-harvests a game come from neighbours taking an apple in the same step.

## 14 agents on the doubled map

5 seeds. 2 strips are two copies of the normal map. k = 2 and 4 give each agent the same chance of a check as k = 1 and 2 with 7 agents. Fig. `fig_decentralised_large_map.png`.

| Overseer, k = 4 | 2 greedy + 12 honest: caught, apples, alive | 14 migrators: over-harvests, apples, alive | 14 sneaky: over-harvests, apples, alive |
|---|---|---|---|
| central, random | 27%, 356, 0/5 | 9.6, 974, 5/5 | 9.6, 974, 5/5 |
| 2 strips, random | 27%, 354, 0/5 | 50, 565, 4/5 | 22, 897, 5/5 |
| 4 strips, random | 31%, 376, 0/5 | 55, 254, 0/5 | 60, 580, 2/5 |
| central, repeat offenders first | 74%, 623, 5/5 | 4.2, 1026, 5/5 | 4.2, 1026, 5/5 |
| 2 strips, independent | 77%, 717, 5/5 | 1.4, 618, 5/5 | 5.4, 994, 5/5 |
| 2 strips, share suspicion | 87%, 728, 5/5 | 1.6, 589, 5/5 | 5.2, 997, 5/5 |
| 2 strips, share suspicion + coverage | 87%, 760, 5/5 | 48, 387, 2/5 | 3.2, 1024, 5/5 |
| 4 strips, independent | 49%, 623, 2/5 | 31, 615, 5/5 | 26, 877, 4/5 |
| 4 strips, share suspicion + coverage | 73%, 713, 4/5 | 53, 208, 0/5 | 26, 797, 5/5 |

- Unused checks: 0–9% with 2 strips and 5–19% with 4 (up to 28% with migrators). That is less than on the 7-agent map, as predicted.
- 14 honest bots: 994–1,026 apples, alive in 5/5, in every condition.
- At k = 2, sneaky bots under repeat-offenders-first checks:
  - central: 4.6 over-harvests, 1,003 apples, 5/5 alive;
  - 2 strips: 32, 704, 3/5;
  - 4 strips: 56, 313, 0/5.
- Migrators on 2 strips with independent overseers keep the orchard but eat only 618 apples (central: 1,026). Their pull towards unwatched strips bunches them together. That costs harvest even when they do not break the rule.

## Predictions against results

| Prediction | Result |
|---|---|
| H1: unused checks under 5% (2 strips) and 10–30% (4 strips, k = 4) | **Partly.** 4 strips: 22–29% (yes). 2 strips: 6–20% on the 7-agent map, more than predicted; 0–9% on the 14-agent map |
| H2: decentralised random catches within 5 points of central | **Mostly.** 2 strips: −4, +5, −7 points at k = 1, 2, 4. 4 strips: −3, −4, −11. The loss tracks the wasted checks |
| H3: independent targeting catches less than central; sharing recovers most of the gap | **Yes when the budget is tight (k ≤ 2), no at k = 4.** k = 2: central 81%, independent 53%, sharing 68–75% (most of the gap with 2 strips, not with 4). At k = 4, 2 independent strips beat central (93% vs 87%) |
| H4: sneaky bots don't care about decentralisation | **No.** Uneven coverage leaves longer gaps between checks. k = 2 random: 6.9 → 21 (2 strips) → 34 (4 strips) over-harvests; the orchard survives 3/10 with 4 strips |
| H5: migrators cheat at least twice as much under decentralised random; independent targeting blunts it; coverage sharing makes it worse | **Yes, all three.** 6.9 → 37 (×5.4) at k = 2; independent targeting 12; with coverage sharing 27 (k = 2) and 17 vs 1.9 (k = 4). Same on the 14-agent map |
| H6: honest bots unaffected | **Yes** (497–506 apples, 10/10 alive) |
| H7: communication 0.5 and 1.5 units per step | **Yes** (by construction: 500 and 1,498 a game) |
| H8: same on the larger map, with less waste | **Yes**, plus a new result: with one cheater per half, local targeting beats central targeting |

## What this means for the learning-agent runs (GPU)

1. **Decentralisation's cost has two parts.**
   - *Waste:* the budget can't follow the agents. This hits random checks and needs no exploiting agent.
   - *Local memory:* each overseer knows only its own checks. This hits agents that learn where they are unwatched.

   The GPU job list (`../DECENTRALISED_README.md`) puts central and decentralised overseers at the same budget and average penalty, so learned agents can show which part matters.
2. **The key learning-agent question:** do IPPO agents learn to migrate, that is, to over-harvest in strips whose overseer has not checked them? The bots show it pays: 5× more over-harvests at k = 2.
3. **Share what helps against the expected adversary:**
   - suspicion helps against fixed cheaters and is harmless otherwise;
   - coverage helps against fixed cheaters but invites migration.

   The GPU list uses suspicion sharing only.
4. **Caveats:**
   - bots don't respond to penalties, see the whole map, and know exactly when each overseer last checked them;
   - learned agents see an 11 x 11 window and see the overseer only through their rewards.
