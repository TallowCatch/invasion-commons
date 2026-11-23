# Decentralised overseers in SocialJax: summary (22 Nov 2026)

Branch `claude/decentralised-games`, from `claude/r1-s1-oversight-experiments` at 55a07b6. Code and commands: `DECENTRALISED_README.md`.

**The question.** So far the overseer has been central: it checks k of N agents anywhere on the map each step. Here the map is split into R regions with one overseer each. The overseers share the same total budget, and each can only check agents inside its region.
- Does splitting cost catches?
- Can agents exploit it?
- Does it help for the overseers to talk?

And does the overseer carry over to other SocialJax games?

## What was built

1. **Decentralised overseers** (`overseer/socialjax/decentralised.py`, `regions.py`).
   - **Regions:** strips, bands or a grid. The total budget is k checks per step, split k/R per region, with any spare check going to a random region each step.
   - **Modes:**
     - (a) independent overseers;
     - (b) overseers that share suspicion scores with neighbouring regions each step, at 0.25 cost units per message (a check costs 1);
     - optionally, (b) plus sharing when each agent was last checked ("coverage").
   - **Logs, per region:** agents present, checks, unused checks, catches, wrong flags, and violations, detections and misses by the region they happened in; also check cost and communication cost.
   - **R = 1 is the central overseer:** it reproduces the existing `HarvestOverseer` step for step under six configurations, including an episode reset (test).
2. **Rules for three games** (`games.py`). Each wrapper has the same interface and logs.

   | Game | Agents | The rule | Why it's checkable |
   |---|---|---|---|
   | Harvest | 7, or 14 on a doubled 16 x 44 map | don't eat an apple that leaves fewer than 3 within two squares | one step of record |
   | Coins | 2 (SocialJax hard-codes two players) | don't take a coin of the other player's colour | one step; read exactly off the rewards (test) |
   | Clean Up | 7 | don't eat while pollution > 0.2 unless you cleaned (removed dirt) in the last 100 steps | needs 100 steps of record; a shorter look can't prove it, or must presume guilt |
3. **Bots for every game** (`game_bots.py`): honest, greedy, sneaky (cheats once nobody has checked it for 20 steps) and the new **migrator** (cheats only in regions whose own overseer has not checked it for 20 steps, and is drawn to them). Under one overseer the migrator is exactly the sneaky bot (test).
4. **Tests:** `tests/test_decentralised_overseer.py`, 27 tests, including one that parses every GPU job line with the trainer. The original 15 in `tests/test_socialjax_overseer.py` still pass.
5. **Bot sweeps**, with protocols and predictions committed before the runs:
   - Harvest: 31 conditions × 4 populations × 10 seeds;
   - Harvest on the doubled map: 21 × 4 × 5;
   - Coins: 13 × 5 × 10;
   - Clean Up: 20 × 4 × 10.

   About 20 min of laptop CPU in all.
6. **The GPU trainer** now takes `--env`, `--regions`, `--layout`, `--share`, `--share-coverage`, `--region-k`, `--presume` and Clean Up's rule settings. It ran end to end on CPU for every game.

## Bot results (scripted bots: a preview for the learning-agent runs, not a paper result)

Full tables, figures and predictions against results: `harvest/results.md`, `coins/results.md`, `cleanup/results.md`.

**1. Splitting the budget wastes checks the agents' positions can't use.**

| Setting | Checks unused | Effect |
|---|---|---|
| Harvest, 7 agents, 2 strips | 6–20% | random checks catch within 4–7 points of central |
| Harvest, 7 agents, 4 strips | 22–29% | random checks lose 3–11 points (k = 4: 47% vs 58% of over-harvests caught) |
| Harvest, 14 agents, 2 / 4 strips | 0–9% / 5–19% | less waste with more agents per region |
| Coins, 2 players, 2 halves | 24–31% | theft pays again at the penalty that makes it break even centrally: the thief nets 86 coins vs 81 (an honest player gets 83) |

**2. Migrators exploit the overseers' local memory.**

Harvest, 7 migrators, 2 checks per step:

| Overseer | Over-harvests per game | Apples | Orchard alive |
|---|---|---|---|
| one central, random | 6.9 | 484 | 10/10 |
| 2 regional, random | 37 | 195 | 1/10 |
| 2 regional, repeat offenders first, independent | 12 | 392 | 9/10 |
| ... sharing suspicion | 17 | 362 | 5/10 |
| ... sharing suspicion and coverage | 27 | 130 | 3/10 |

- The same holds on the 14-agent map: 9.6 → 50 over-harvests at k = 4.
- It holds in Coins too, though there the numbers are small (0.8 → 4–7 thefts a game).

**3. Sharing helps against one adversary and hurts against another.** Against a fixed cheater in Harvest (1 greedy + 6 honest, k = 2, repeat offenders first):

| Overseer | Over-harvests caught | Orchard alive |
|---|---|---|
| central | 81% | 10/10 |
| 2 regional, independent | 53% | 5/10 |
| ... sharing suspicion | 68% | 8/10 |
| ... sharing suspicion and coverage | 75% | 9/10 |
| 4 regional, sharing suspicion and coverage | 39% | 1/10 |

- Sharing coverage helps here but invites migration (point 2).
- **At k = 4, local targeting matches or beats central:** 93% vs 87% caught.
- On the 14-agent map with one cheater per half, 2 independent overseers beat the central one: 77% vs 74% caught, and 717 vs 623 apples. Each overseer gets to know its own cheater.

**4. Splitting hurts even agents that don't exploit regions.**
- Sneaky bots react to any check. But a crowded strip checks each of its agents less often, so long gaps between checks appear: 6.9 → 21 over-harvests (k = 2, random, 2 strips).
- With repeat-offenders-first checks at k = 1, the orchard survives in 3/10 runs, against 10/10 centrally.

**5. Clean Up: a check has to read the rule's whole window.** With 1 free-rider among 6 honest agents and 2 random checks per step:

| Look | Free-rides caught | Wrong flags on honest agents | Apples (all 7) |
|---|---|---|---|
| 1 or 50 steps | 0% | 0 | 374 |
| 100 steps | 32% | 0 | 334 |
| 150 or 200 steps | 100% | 0 | 343 |
| 10 steps, presuming guilt | 98% | 35 | 130 |

- Presuming guilt on a 10-step look also wrecks a fully honest group: 45 wrong flags and 196 apples instead of 908. Freezing a few cleaners tips the river past the point where apples stop growing.
- The overseer stops the free-rider eating (85 → 14 apples) but can't make it clean, so total apples don't rise.
- Sneaky bots and migrators shirk cleaning in short spells and never break the 100-step rule.
  - Under 1 central random check per step they still cost the group 25% of its apples (680 vs 908).
  - With 2 regional overseers, migrators cost it 43% (517).
  - That is a limit of the rule, not of the overseer.

**Predictions** (20, written before the runs): 12 held, 6 held in part, 1 failed and 1 couldn't be tested.
- **The failure:** "sneaky bots don't care about splitting". They do (point 4).
- **The most important partial:** "regional targeting is worse than central" held only at k ≤ 2. At k = 4 it was as good or better (point 3).

Every prediction is checked in the results files.

## Learning-agent results (IPPO, 23 Nov): what changes

Full tables: `DECENTRALISED_RESULTS.md`. Harvest 67 of 69 runs (2 still running), Coins 45 of 45, 3 seeds per cell. **Same caveat as above:** records are handed over for free, so these runs measure check placement only.

- **Splitting fails exactly when it pushes the fine a violation actually costs below the learners' threshold.**
  - Harvest, 4 strips, average penalty 1 apple: the orchard dies at k = 2 and k = 4 (76–86 apples with random checks, against 458–476 centrally). Fewer over-harvests are caught (18% vs 28% at k = 2), so an over-harvest costs 0.65–0.72 apples instead of 1.
  - Coins, 2 halves, at break-even: 26–29 thefts a game against 7–9 centrally.
  - At twice the penalty (c = 2), every split arm works, within about 20 apples or 1 theft a game of central.
- **The bots' waste overstated it in Harvest and got it right in Coins.** Learners leave 0–1% of checks unused with 2 strips (bots 6–20%) and 4–13% with 4 strips (bots 22–29%). But with 4 strips the catch share falls by a third, more than the waste, because over-harvests happen where agents crowd. In Coins the waste (28–31%) matches the lost catches.
- **The bots' migration effect did not show** (2 strips, random: 1.1–1.3× central over-harvests, against 5× for migrator bots). The design can't separate deliberate migration from crowding: checks are hidden, there is one shared network and no per-agent region logs.
- **Targeting and sharing matter little with one shared network.** Repeat offenders first is no better than random, centrally or regionally. Sharing suspicion helped only at k = 2, c = 1 with 2 strips (420 → 457 apples; central 470), and only slightly.
- **Bot point 3 (local targeting beats central at k = 4) failed** with learners: 439 vs 484 apples.
- **Change to the conclusions:** the main cost of decentralising is a weaker realised deterrent from fixed per-region budgets. Set the penalty from the achieved catch rate, or let budgets follow the agents. Migration and evidence loss remain bot-only results until the worst-case and local-records runs exist.

## What's ready for the GPU

The job lists come from `experiments/socialjax/decentralised_runs.py`: 159 runs, 1e8 steps, 3 seeds. Each compares central and decentralised overseers at the same total budget, with penalty per catch = N × c / k so that random checks give an average penalty of c.

| List | Runs | Content |
|---|---|---|
| `pilot` | 4 | one 2e7-step run per game (check time and memory first) |
| `harvest` | 69 | k = 2, 4 × average penalty 1 and 2 apples (as in stage 1): 2 strips random / independent targeting / sharing, 4 strips random / sharing, central targeting where stage 1 lacks it |
| `harvest_large` | 21 | 14 agents, k = 4, c = 1: none, central random, central targeting, 2 and 4 strips random and sharing |
| `coins` | 45 | none; k = 1 at c = 1 and 2 (central random and targeting; 2 halves random, independent, sharing); k = 2 central and split |
| `cleanup` | 24 | none; central random with a 150-step look at c = 1, 2; central targeting; a 50-step look (can't prove anything); a 50-step look presuming guilt; 2 strips random and sharing |

**Exact commands** (details in `DECENTRALISED_README.md`). On a CREATE login node:

```bash
WORK=/scratch/users/$USER/socialjax_work
git clone -b claude/decentralised-games https://github.com/TallowCatch/invasion-commons.git $WORK/invasion-commons-dec
cp $WORK/invasion-commons-dec/scripts/create/ippo_overseer_array.sbatch ~/ippo/ippo_overseer_array_dec.sbatch
cd $WORK/invasion-commons-dec
for L in pilot harvest harvest_large coins cleanup; do
  python3 -m experiments.socialjax.decentralised_runs $L > ~/ippo/runs_dec_$L.tsv
done
# pilot first
sbatch --array=1-4 --time=04:00:00 \
  --export=ALL,RUNS=$HOME/ippo/runs_dec_pilot.tsv,REPO=$WORK/invasion-commons-dec ~/ippo/ippo_overseer_array_dec.sbatch
# then each full list (set --time from the pilot's train_seconds x 5 + 5 min)
for L in harvest harvest_large coins cleanup; do
  N=$(wc -l < ~/ippo/runs_dec_$L.tsv)
  sbatch --array=1-$N%8 --time=04:00:00 \
    --export=ALL,RUNS=$HOME/ippo/runs_dec_$L.tsv,REPO=$WORK/invasion-commons-dec ~/ippo/ippo_overseer_array_dec.sbatch
done
```

- The array script now reads an optional `REPO`, so these jobs use their own checkout and the stage 2–3 jobs are untouched.
- Outputs go to `/scratch/users/$USER/ippo_overseer/<DH_|DL_|DC_|DU_...>/`.

## What's left

1. **Run the GPU lists** (pilot first) and write the learning-agent results. *Done 23 Nov for Harvest (67 of 69) and Coins (see above); the 14-agent map and Clean Up lists are not in yet.*
   - The key question is whether IPPO agents learn to migrate: the bots show it pays (5× more over-harvests).
   - Learned agents see only an 11 x 11 window and see the overseer only through their rewards, so they may not find it.
2. **A third game.** Assessed, not built:

   | Game | A clear, checkable rule? | Verdict |
   |---|---|---|
   | Territory (9 agents) | yes: don't zap another player (two hits remove a player for good); visible in one step | **best next port.** Its claim mechanics (walls as resources, claims stored as special grid codes) need a few hours of bot work |
   | PD Arena (2 agents, 50 steps) | yes: don't pick up a defect resource | rewards come only through a zap-beam interaction between players, so bots need that mechanic; only 2 agents |
   | Gift (6 agents) | "gift before you consume" is a history rule like Clean Up's | adds little beyond Clean Up |
   | Mushrooms (6 agents) | no clean rule: eating orange hurts everyone including the eater, so there is no temptation; "don't eat red" is not clearly harmful | skip |
   | Coop Mining (4 agents) | abandoning a partner at gold is a coordination failure, not a temptation (gold pays more) | skip |
3. **The Clean Up rule** misses intermittent shirking (point 5). A rule on each agent's share of the cleaning would be the next rule to test.
4. **Bot caveats.**
   - Bots ignore penalties: whether rule-breaking pays is worked out from the catches.
   - Bots know exactly when each overseer last checked them.
   - Bots see the whole map.
   - The 0.25 cost per message and the 20-step "unwatched" threshold are choices, not measurements.
5. **Trackers updated:**
   - `notes/OVERSEER_TRACKER.md` (findings 12–16);
   - `FINDINGS_AND_IMPLICATIONS.md` (F19);
   - `what_we_learned_and_next.md` (Update 16);
   - `PROJECT_HISTORY.md`;
   - `notes/socialjax/README.md`;
   - `experiments/README.md`.

**Caveat on findings 12–16 and the 159 decentralised GPU runs (found 22 Nov).** In the current regional overseer (`overseer/socialjax/decentralised.py`), an agent's record of its last w steps travels with it, and any overseer's check reads all of it, including steps spent in other regions. So these results are for regional overseers that hand records over to each other for free. They measure only the loss from checks being in the wrong place, not the loss from evidence left behind when an agent crosses a boundary. The latter (the formulation's P_0 and claim D1) needs the local-records overseer being built on branch `claude/locality-scale`.
