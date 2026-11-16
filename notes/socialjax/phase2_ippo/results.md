# Phase 2 with IPPO learners: results (stage 1)

Run on CREATE on 15–16 Nov 2026, under `protocol.md` (with Amendment 1).
- **Jobs:**
  - stage 1 array 38656035;
  - reruns 38662888, for tasks 8 and 26 (see "Run problems" below).
- **Runs:** `results/runs/socialjax_phase2/` (git-ignored; params stay on CREATE).
- **Summary:** `summary.json`.
- **Figures:** `fig_checks_by_penalty.png`, `fig_learning_curves.png`.
- **Code:**
  - `experiments/socialjax/train_ippo_overseer.py`;
  - `phase2_runs.py`;
  - `analyze_phase2.py`.

**Status:** stage 1 is complete except two reruns. B random k = 4 at c = 0.5 and c = 2.0 have 2 of 3 seeds so far. Stages 2 and 3 (Q5, Q6) are queued.

**How to read the numbers:**
- Each value is the mean of the last 10 updates (2,560 episodes) for one seed. The table gives the mean over 3 seeds, with the seed range in brackets.
- One episode is 1,000 steps with 7 agents.
- Seven restrained scripted bots eat 500 apples per episode (week 2).

## Main table

| Overseer | k × P (apples) | Apples eaten | Orchard alive | Over-harvests | Share of eats that over-harvest | Caught | Apples left at end |
|---|---|---|---|---|---|---|---|
| None (arm A) | – | 68 [67, 68] | 0.00 | 30 | 45% | – | 0 |
| Random, c = 0.5, any k | 3.5 | 68 [68, 68] | 0.00 | 27–29 | 40–42% | 4–27 | 0 |
| Random k = 1, c = 1 | 7 | 441 [411, 457] | 1.00 | 50 | 11% | 7 | 23 |
| Random k = 2, c = 1 | 7 | 458 [441, 470] | 1.00 | 39 | 9% | 11 | 24 |
| Random k = 4, c = 1 | 7 | 476 [454, 493] | 1.00 | 39 | 8% | 22 | 26 |
| Random k = 7, c = 1 | 7 | 477 [468, 483] | 1.00 | 35 | 7% | 35 | 26 |
| Random k = 1, c = 2 | 14 | 515 [506, 530] | 1.00 | 7 | 1.4% | 1 | 29 |
| Random k = 2, c = 2 | 14 | 527 [517, 540] | 1.00 | 6 | 1.2% | 2 | 29 |
| Random k = 4, c = 2 (2 seeds) | 14 | 528 [523, 534] | 1.00 | 5 | 1.0% | 3 | 29 |
| Random k = 7, c = 2 | 14 | 530 [522, 536] | 1.00 | 5 | 1.0% | 5 | 29 |
| Random k = 2, freeze 50 (arm C) | – | 77 [75, 78] | 0.00 | 32 | 41% | 9 | 0 |
| History k = 2, c = 1 (arm D) | 7 | 470 [460, 483] | 1.00 | 32 | 7% | 9 | 26 |
| Location k = 2, c = 1 (arm D) | 7 | 150 [122, 179] | 1.00 | 77 | 51% | 4 | 6 |

- **"Over-harvests" is a count**, and it can rise when the orchard lives, because there are more apples to eat. The share column (over-harvests ÷ apples eaten) is the fairer comparison.
- **"Orchard alive" means at least one apple is left at the end.** The location arm passes that test with about 6 apples left.

## Answers to the protocol's questions

**Q1. Do IPPO learners with no overseer over-harvest?**
- **Yes, more strongly than predicted.** In all 3 seeds the orchard dies in every env within about 3M steps (update 10–15) and never recovers.
- The agents settle at 68 apples per episode, against 500 for restrained bots. 45% of their eats break the m = 3 rule.
- The setting therefore has a problem for the overseer to fix, so the "what would change the plan" condition is not triggered.

**Q2. Which k × P protects the orchard, and is it the expected penalty that matters?**
- **It is a sharp threshold set by k × P, as predicted.**
  - At k × P = 3.5 apples, nothing changes at any k (68 apples, orchard dead). That holds even at k = 7, where every over-harvest is caught with a 0.5-apple penalty.
  - At k × P = 7 every run saves the orchard (441–477 apples).
  - At 14 the agents eat 515–530 apples, more than seven restrained bots (500), and only about 1% of their eats break the rule.
- **The learners' threshold lies between 3.5 and 7 apples.** This is consistent with week 2's break-even of about 7 apples for scripted bots, though this sweep cannot place it more finely than that. A finer sweep (k × P = 4, 5, 6) would.
- **The cheapest working overseer is k = 1 with a 14-apple penalty:** 1,000 checks per episode, 515 apples. Checking all 7 agents every step (7,000 checks) at the same expected penalty adds only 15 apples.
- **Second-order effect of k at k × P = 7:** more checks with smaller penalties do a little better (k = 1: 441 [411, 457]; k = 7: 477 [468, 483]; the seed ranges do not overlap). A plausible reason is that a learner sees a steadier signal when catches are frequent and small. With 3 seeds this is a lead, not a result.

**Q3. Freeze or penalty?**
- **The penalty wins clearly.** Random k = 2 with a 50-step freeze does nothing useful: 77 apples, orchard dead in every seed, and 41% of eats still break the rule.
- This matches week 2, where k = 2 with a 50-step freeze also failed to save the orchard against one greedy bot.
- As predicted, a freeze changes no reward directly. The learners do not learn to avoid it, and it is too weak to protect the orchard by itself.
- Caveat: the freeze arm is not matched to the penalty arms in cost to the agent, so this compares two specific settings rather than the two kinds of consequence in general.

**Q4. Random vs targeted at k = 2 (c = 1)?**
- **History targeting reaches the same end state as random,** 470 [460, 483] against 458 [441, 470], **but gets there much faster.**
  - At update 50 (12.8M steps) the orchard was alive in 98–100% of envs under history, against 14–57% under random. See the learning curves.
  - Early on, history catches 44% of over-harvests against random's 29%, because it keeps checking agents it has caught before. It settles at the same 28% once all agents behave alike. They share one network, so there is no persistent cheater to single out.
- **Location targeting fails.** This contradicts the protocol's prediction ("location similar").
  - It catches about 5% of over-harvests, against 28% for random.
  - Over-harvesting stays at half of all eats, and the agents eat 150 apples with the orchard barely alive (6 apples left, against 24–26 for random).
- **Why location fails: the targeting rule is a bad proxy, not mainly that agents learned to dodge it.**
  - The chooser checks the agents standing where the fewest apples are left. An agent that has just over-harvested is, by the rule's definition, standing beside one or two apples. Idle agents on bare ground outrank it.
  - The catch rate is already 4% at update 0, before any learning. So most of the failure is built into the rule.
  - It rises to about 22% around update 20 and then falls back to 5–6%. That later fall may be learned exploitation, but this design cannot separate the two.
  - This is the same lesson as Fishery T1 (always checking the largest reporter caught 25% of liars, against 100% for random) and the week 2 sneaky bots: **targeting on a proxy can be worse than random, even when agents cannot see the checks.**

**Q5 and Q6 (stages 2 and 3)** are queued and will be added here.

## Run problems

- **Bad node:** arc-hpc-comp035 had no visible GPU ("No devices were found"), so JAX silently ran on the CPU.
  - Task 8 (B random k = 4, c = 0.5, seed 1) hit the 8 h limit.
  - Task 26 (B random k = 4, c = 2.0, seed 2) was cancelled after 6 h 22 min.
  - Both were resubmitted as job 38662888, excluding that node. Stages 2 and 3 were updated to exclude it as well.
  - The job script now stops at once if JAX does not see a GPU.
- **No run used results from a partial run.** The reruns start fresh.
- **Speed:** stage 1 runs took 18 min (H200) to 38 min (A100) each.

## Caveats

- 3 seeds per arm, and the means come from the last 10 updates. Effects that do not separate the seed ranges are leads only.
- **Parameter sharing:** all 7 agents use one network, so there is no persistent individual cheater. This limits what history targeting can gain, and it is why phase 4 needs heterogeneous agents or agents that can see the checks.
- The inspector is perfect (no misses, no false alarms). With false alarms, the large penalties that make k = 1 work (14 apples) would also hit honest agents. That trade-off is phase 5.
- Agents cannot see who is checked, so targeting cannot be dodged deliberately here. That is phase 4.

## What this means for the next steps

1. **Phase 3 (the overseer sets its own penalty) has a clear target.** k × P must be above about 7 apples per over-harvest, and the overseer can learn it by watching. A finer sweep at k × P = 4–6 would pin down the threshold for learners.
2. **Phase 4 is well motivated by the location result.** A targeted rule can be worse than random even before agents adapt. The red-team loop should include a proxy-targeted overseer as a known weak baseline and test whether history targeting stays ahead once agents can see checks.
3. **Freeze is dropped as the main consequence.** Phases 3–5 will use the reward penalty.
