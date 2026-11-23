# Overseer tracker: what we've found, in plain English

One page for the budgeted-overseer work: an overseer that can only check a few agents at a time. Each finding has its numbers, how sure we are, and where the details are. Newest changes are in the log at the bottom.

**How sure we are:**
- **Solid:** clear in every run.
- **Lead:** seen, but in few runs or still running.
- **Prediction:** not tested yet.

**The setting so far.** SocialJax Harvest Open:
- 7 agents collect apples for 1,000 steps;
- apples only regrow near other apples;
- each agent keeps its own apples.

**The rule.** Don't take an apple with fewer than 3 others within two squares.

**The overseer.** Checks k of the 7 agents each step, at random unless stated, and takes a penalty off anyone it catches. The **average penalty** is the chance of being checked (k out of 7) times the penalty, in apples.

## The findings at a glance

| # | Finding | Key numbers | How sure | Details |
|---|---|---|---|---|
| 1 | Harvest is a real social dilemma, and very fragile | all hold back: 71.5 apples each; all greedy: 9.6; one greedy bot kills the orchard in 10 of 10 runs | solid (bots) | `socialjax/week2_bots/results.md` |
| 2 | With no overseer, learning agents always strip the orchard | 68 apples a game, orchard dead in every run, 45% of apples taken break the rule | solid | `socialjax/phase2_ippo/results.md` |
| 3 | The average penalty decides, not the number of checks | 0.5 apples: nothing changes even with all 7 checked; 1 apple: the orchard always survives (441–477 apples); 2 apples: 515–530 | solid (3 runs each) | same |
| 4 | One check per step is almost as good as checking everyone (with a perfect inspector) | 515 apples with 1 check against 530 with 7, at 2 apples average penalty | solid | same |
| 5 | Freezing caught agents doesn't work | 77 apples, orchard dead | solid | same |
| 6 | Checking repeat offenders first is faster, not better, when agents start from scratch | orchard safe by about 13M steps (random: 14–57% of games); both end at 460–470 apples | solid | same |
| 7 | Targeting on a bad signal is worse than random | checking by location catches 5% of over-harvests, against 28% for random | solid | same |
| 8 | **Remove the overseer and over-harvesting comes straight back** | every setting falls back to 68 apples and a dead orchard within about 5–12M steps | solid (18 of 18 runs) | below |
| 9 | **Fixing agents that already over-harvest needs more checks, or targeting** | after 50M steps: 1 check 149 apples, 2 checks 197, 4 checks 274, 7 checks 325; repeat offenders first with 2 checks 348, and very consistent | solid (3 runs each) | below |
| 10 | **An overseer that misses half the time needs twice the penalty** | misses 50%: 1 apple average penalty now fails (68 apples, dead), 2 apples works (479 at k = 7, matching the perfect inspector at 1 apple) | solid (3 runs each) | `socialjax/phase2_ippo/results.md` |
| 11 | **False alarms cost honest agents the same per honest apple whatever the number of checks, but with a mistaken overseer, few big checks do worse than many small ones** | about 0.25 apples wrongly fined per legal apple at k = 1 and k = 7; apples at the same average penalty, k = 1 vs 7: 342 vs 479 (misses), 382 vs 482 and 470 vs 509 (false alarms); the threshold for learners is between 0.5 and 0.75 | solid (3 runs each) | same |
| 12 | **Splitting the overseer by region wastes budget** that can't follow the agents | Harvest, 4 strips: 22–29% of checks unused; 2 players in Coins: 26%; random checks lose about that share of catches | solid (bots) | `socialjax/harvest/results.md`, `socialjax/coins/results.md` |
| 13 | **Agents that cheat where the local overseer hasn't looked gain a lot** | migrators, 2 checks per step: 7 over-harvests a game under one overseer, 37 under 2 regional ones (orchard alive 10/10 vs 1/10) | solid (bots) | `socialjax/harvest/results.md` |
| 14 | **What overseers share matters:** sharing who was caught helps; sharing who was just checked invites migration | fixed cheater, k = 2: 53% caught alone, 68% sharing suspicion, 75% also sharing coverage; migrators: 12 → 27 over-harvests with coverage sharing | solid (bots) | same |
| 15 | **A check has to read the rule's whole window** (Clean Up: no eating unless you cleaned in the last 100 steps) | free-rides caught: 0% for looks under 100 steps, 32% at 100, 100% at 150+; presuming guilt on a 10-step look flags 45 honest agents a game and cuts apples from 908 to 196 | solid (bots) | `socialjax/cleanup/results.md` |
| 16 | In Coins, theft stops paying at exactly the predicted penalty under one overseer, not under split ones | average penalty 1 coin per theft: thief nets 81 coins (honest 83) centrally, 86–88 split | solid (bots) | `socialjax/coins/results.md` |
| 17 | **BazaarBench (recorded LLM traders): random checks catch about k/20; repeat offenders first is worse than random when nearly everyone plans unsafe acts** | exploit instructions, k = 5 of 20: random 28.5%, repeat offenders 24.6% (past-plan failures: 45% vs 25%) | lead (1 market pair, offline) | `bazaarbench/offline_replay/results.md` |
| 18 | **BazaarBench: most failures exist only in private reasoning, so what a check can read decides what it catches** | exploit: 998 of 1,051 failures never leave reasoning; a check without reasoning sees at most 5%, record checks 0.2% | lead (GPT-5 labels, 1 tested model) | same |
| 19 | **BazaarBench: daily checks are too late for harm; fixed local budgets waste a small budget** | everyone checked daily catches 2 and 5 of 15 carried-through failures first; 4 independent overseers at 1 unit catch 0.3–1.2% vs 6–7% central; a 500-token shared board recovers it | lead | same |
| 20 | **Locality: crossings and uneven crowding decide what splitting costs, not where harm lands; the existing regional overseer shares records for free** | ξ_loc ranks catches lost to local records (ρ = 0.80) but not the total gap (0.42); P_0 off by up to 0.73 when agents return (Markov form within 0.003); fixed groups at N = 256 lose ≤ 0.01 | lead (bots, 7 maps) | `socialjax/locality/results.md` |
| 21 | **LLM inspectors: a step at 7B, and what the record shows matters as much as size** | at w = 1, Qwen ≤ 3B useless, Qwen 7B and gpt-oss perfect on summaries; on raw maps only gpt-oss protects (120B α 0.03); long records hurt (α + β_eat up to 0.31 at w = 50) | lead (600 fixed items) | `llm_overseer/SUMMARY.md` |
| 22 | **With learning agents, splitting the overseer fails when it pushes the fine a violation actually costs below the threshold** (records handed over for free, so placement only) | Harvest, 4 strips, c = 1: orchard dead at k = 2 and 4 (76–86 apples vs 458–476 central; 0.65–0.72 apples per over-harvest instead of 1); Coins, 2 halves, at break-even: 26–29 thefts vs 7–9 central; 2 strips and c = 2 cost little; no sign of bot-scale migration (1.1–1.3×) | solid (3 seeds; 2 Harvest cells have 2) | `socialjax/DECENTRALISED_RESULTS.md` |
| 23 | **LLM inspectors in RL: C1's protect/fail classes held in 15 of 16 cells, but misses and false alarms don't trade one for one** | gpt-oss-20B on raw maps (C1 e 0.82) at k = 1, c = 1: 210 apples, orchard alive in 63% of games; Qwen 3B + reasoning (e 0.77, no misses) keeps it (380). What matters is the fine an over-harvest actually costs, (1 − α)·c: 0.90 is partly enough at k = 1, enough at k = 7 | solid (3 seeds; Qwen 7B direct 1) | `llm_overseer/rl_results.md` |
| 24 | **Trained agents cross region boundaries about half as often as bots; harm is as local** | χ 0.011–0.015 vs 0.028 (2 strips), 0.027–0.044 vs 0.073 (4 strips); ζ within 0.06 of bots; ξ_loc at w = 50 on 2 strips 0.27–0.47 vs 0.70; band crossings rise 2.5× once the orchard lives | lead (1 seed per setting, 32 episodes, central overseers) | `socialjax/locality/results.md` §8 |
| 25 | **A real LLM inspector's misses depend on the situation; its false alarms barely do** | gpt-oss-20B on raw maps, 2,463 trained-agent eats: α 0.00 (no apples around, few in the outer cells) to 0.48 (2 around, 2+ outer); held-out gain over a constant rate +0.128 nats per item [+0.103, +0.153]. β 0.06 at exactly 3 apples, about 0.01 elsewhere; no gain over a constant. Bank averages α 0.215 (overseen agents' records 0.288), β 0.039, against the ladder's 0.10 / 0.08. Qwen 7B + reasoning, 179 items: α 0.57, β 0.10, same direction, no held-out gain | verified offline; RL untested (36-run job list written) | `llm_overseer/error_patterns/results.md` |

**Caveat on findings 12–16 and the 159 decentralised GPU runs (found 22 Nov).** In the current regional overseer (`overseer/socialjax/decentralised.py`), an agent's record of its last w steps travels with it, and any overseer's check reads all of it, including steps spent in other regions. So these results are for regional overseers that hand records over to each other for free. They measure only the loss from checks being in the wrong place, not the loss from evidence left behind when an agent crosses a boundary. The latter (the formulation's P_0 and claim D1) needs the local-records overseer being built on branch `claude/locality-scale`.

The same patterns had already shown up in the earlier non-spatial games:
- **Fishery with LLM agents:** the expected fine decides (L4).
- **Fishery:** always checking the biggest reporter caught 25% of liars, against 100% for random checks (T1).
- **Willis et al.'s commons:** strategies obey while audited but don't change (R1b, F16).

Details are in `claude_audit_20261005/FINDINGS_AND_IMPLICATIONS.md`.

## Finding 3 in a table

Apples collected per game, average of 3 runs, at the end of training:

| Average penalty | 1 check | 2 checks | 4 checks | 7 checks | Orchard survives? |
|---|---|---|---|---|---|
| none (no overseer) | 68 | 68 | 68 | 68 | never |
| 0.5 apples | 68 | 68 | 68 | 68 | never |
| 1 apple | 441 | 458 | 476 | 477 | always |
| 2 apples | 515 | 527 | 528 | 530 | always |

For comparison, seven well-behaved scripted bots collect 500.

## Findings 8 and 9: switching the overseer off and on (stages 2 and 3, still filling in)

**Switching it off** (agents trained with the overseer, then 50M more steps without it; 18 of 18 runs):

| Trained with | Apples before | Apples after | Orchard |
|---|---|---|---|
| random checks, 1, 2, 4 or 7 per step | 408–482 | 68 | dead in every run |
| repeat offenders first, 2 checks | 466 | 68 | dead |
| freeze | 70 | 68 | dead (it never worked) |

**Switching it on** (agents that learned to over-harvest with no overseer, then 50M steps with it, average penalty 1 apple, 3 runs each):

| Overseer switched on | Apples at the end | Orchard survives |
|---|---|---|
| none | 68 | 0% |
| 1 random check | 149 (120–172) | 58% |
| 2 random checks | 197 (125–293) | 80% |
| 4 random checks | 274 (241–315) | 100% |
| 7 random checks | 325 (244–403) | 100% |
| 2 checks, repeat offenders first | 348 (346–352) | 100% |
| 2 checks, 50-step freeze | 76 | 0% |

**What it means.** The average penalty decides whether the overseer can work in the long run (finding 3). But how quickly it fixes agents that already misbehave depends on how often it checks, and on targeting. Frequent small penalties are a clearer signal to learn from than rare big ones. Some runs were still improving at the end, so the gap may narrow with longer training.

## Findings 12–16: decentralised overseers and two more games (scripted bots, 22 Nov)

Several overseers, each covering one region of the map, share the same total budget of k checks per step. Each can only check agents in its region. They are either independent or share scores with neighbouring regions, at a cost per message. Code in `overseer/socialjax/decentralised.py`. Games: Harvest (7 agents, and 14 on a doubled map), Coins (2 players) and Clean Up (7 agents).

| | Central overseer | 2 regional overseers | Notes |
|---|---|---|---|
| Fixed cheater in Harvest, k = 2, repeat offenders first | 81% caught, orchard alive 10/10 | 53% (5/10); sharing suspicion 68% (8/10) | at k = 4 the split overseers catch as many or more (93% vs 87%) |
| Migrators in Harvest, k = 2, random | 6.9 over-harvests, 484 apples | 37, 195 apples | they cheat where the local overseer hasn't checked them recently |
| Sneaky bots in Harvest, k = 2, random | 6.9 over-harvests | 21 | even bots that don't exploit regions gain: a crowded strip checks each agent less often |
| Thief in Coins, k = 1, random, average penalty 1 coin | nets 81 (honest: 83) | nets 86 | a quarter of the checks are wasted when both players are in one half |

**Clean Up** gives the "hard to see in one check" case (claim C4 in the general formulation), with a dial: the look a check needs equals the rule's window.

**Bots only, so all of this is a lead for the learning-agent runs.** GPU job lists and commands: `socialjax/DECENTRALISED_README.md`. Summary: `socialjax/DECENTRALISED_SUMMARY.md`.

## What the overseer costs

So far cost is counted in checks per game: 1 check per step is 1,000 checks a game, and checking everyone is 7,000. With a perfect inspector, 1,000 checks recover about 97% of the gap between no overseer and checking everyone (finding 4). That's measured as η = (515 − 68) / (530 − 68).

## Running now (CREATE)

| What | Runs | Job |
|---|---|---|
| Decentralised Harvest, last 2 cells (`DH_R2_random_k4_c1.0_seed1`, `DH_R2_history_k4_c1.0_seed1`) | 2 | from 38810497/98/513/515 |
| Decentralised 14-agent Harvest and Clean Up lists | 45 | not in the 23 Nov pull; status to check |

Stages 2–3, the mistakes test, the 40 LLM-inspector runs, the Coins list and the trained-agent locality job are done (as of 23 Nov).

Nodes comp035 and comp040 have broken GPUs and are excluded.

## Next

- **BazaarBench offline test:** done 21 Nov (findings 17–19; merged 22 Nov). Next there: a cheap open-model judge for real miss and false-alarm rates, and a tested model that acts on more of its plans.
- **The plan for using both testbeds:** `phd_plan/2026-11-21_way_forward.md` (written 21 Nov).

## Log

- **23 Nov (evening).** Experiments paused to define the overseer properly (`formulation/README.md`). Worst-case runs held on CREATE (separate networks undertrained at 1e8 steps). Phase 1 literature map done (`formulation/01_literature_map.md`). Phase 2 draft definition written: the overseer as a player whose action is look, judge, respond, share; June ideas restored (two levels of harm and local-pass/global-fail, observation channels, capability vector, capability gap).
- **23 Nov (afternoon).** Patterned LLM inspector errors (finding 25 here, F26 in the findings log), branch `claude/llm-error-patterns`. Situation bank of 13,159 eats from trained agents; gpt-oss-20B judged 2,463 (2.42M cloud tokens of a 3M cap, about $0.77) and Qwen 7B + reasoning 179. Trainer option `--inspector-table` (off by default; default path tested identical) and a 36-run CREATE list, not submitted: `llm_overseer/error_patterns/`.
- **23 Nov.** Three sets of GPU results written up (findings 22–24 here, F23–F25 in the findings log).
  - Decentralised IPPO, Harvest 67 of 69 runs (`DH_R2_random_k4_c1.0_seed1` and `DH_R2_history_k4_c1.0_seed1` still running) and Coins 45 of 45: `socialjax/DECENTRALISED_RESULTS.md`.
  - 40 LLM-inspector RL runs: `llm_overseer/rl_results.md`.
  - Trained-agent locality: `socialjax/locality/results.md` §8.
  - Across all three, the fine a violation actually costs (catch share × penalty) sorts the outcomes. It needs to be above about 0.75–0.9 apples, and more at k = 1. False alarms fine honest agents but barely weaken deterrence.
- **22 Nov (night).** LLM inspector ladder done (finding 21 here, F22 in the findings log): Qwen 0.5B–7B and gpt-oss 20B/120B on 600 fixed items; cloud spend 1.1M tokens of the 1.5M cap. Branch `claude/llm-overseer` merged. 40 RL runs with LLM inspectors listed, not submitted.
- **22 Nov (late).** Separate networks and the worst case (agents see who is checked) built on branch `claude/separate-nets-worstcase`: `--separate-nets`, `--reveal-checks`, the `mixed` chooser, `check_prob`, and `just_checked` and `dodger` bots. Protocol with predictions written first (`socialjax/worstcase/protocol.md`). Bot preview (`socialjax/worstcase/results.md`): with checks visible, a dodger is never caught by round robin or history, cannot gain from random checks at c = 1, is caught 7–13% of the time under mixed (ε = 0.3), and finds crowded strips under 2-strip random checks. Cheating still kills the orchard and leaves the dodger with half the apples it gets by holding back. 33 GPU runs listed (15 separate-nets, 18 worst case), not submitted.
- **22 Nov (evening).** Locality and scale done (finding 20 here, F21 in the findings log): crossing and harm locality measured on 11 layouts, the P_0 formula tested against simulation, D1 tested, Willis commons at N = 64 and 256. Branch `claude/locality-scale` merged. Trained-agent locality job prepared, not submitted.
- **22 Nov (afternoon).** Stages 2–3 complete (findings 8–9 now solid). Mistakes test in (findings 10–11). Decentralised GPU runs submitted: pilot 38810496, then 159 runs that start only if the pilot succeeds (38810497/98/513/515).

- **22 Nov.** Decentralised overseers built, plus Coins and Clean Up ports. Bot sweeps run under protocols written first (findings 12–16). 159 GPU runs listed (`socialjax/DECENTRALISED_README.md`). Branch `claude/decentralised-games`.
- **21 Nov (later).** BazaarBench offline replay on two markets (GPT-5.5 base, gpt-oss-120b tested, ordinary vs exploit instructions): findings 17–19. Predictions in `bazaarbench/offline_replay/protocol.md` were written before the replay; 2 of 11 held in full, the other 9 partly or mixed (scorecard in `results.md`).
- **21 Nov.** Stages 2 and 3 partly in (findings 8 and 9). The mistakes test was submitted. False alarms were redefined to fall only on honest harvests, before any run used them. This tracker was started.
- **16 Nov.** Stage 1 results (findings 2–7). Wording corrected: the average penalty is (k/7) × penalty, not k × penalty.
- **15 Nov.** Bots (finding 1) and the break-even point (greed stops paying at about 1 apple of average penalty).
