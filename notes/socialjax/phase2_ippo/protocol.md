# Phase 2: IPPO learners under the budgeted overseer in Harvest

Written 15 Nov 2026, before any CREATE run.
- Code: `experiments/socialjax/train_ippo_overseer.py` (SocialJax's IPPO at commit 18e2171 plus the overseer wrapper).
- Job list: `experiments/socialjax/phase2_runs.py`. Job script: `scripts/create/ippo_overseer_array.sbatch`.
- Builds on week 2 with scripted bots (`notes/socialjax/week2_bots/results.md`).

## Setting

- **Game:** SocialJax Harvest Open, 7 agents, individual rewards (one apple pays 7), 1,000-step episodes.
- **Rule:** m = 3. An eat is judged on the map the agent saw. Each check sees only the current step, and the inspector is perfect.
- **Learners:** IPPO with SocialJax's hyperparameters (`ippo_base.yaml`): parameter sharing, CNN actor-critic, 256 parallel envs, 1e8 steps (390 updates), lr 5e-4 annealed, γ = 0.99.
- **What agents see:** only their own 11×11 view. They cannot see who is checked.
- **Penalties** are in apples: a penalty of P apples removes 7P reward from the caught agent. A frozen agent's actions are replaced by `stay`.

## Runs

**Stage 1: from scratch, 1e8 steps, seeds 1–3.**

| Arm | Overseer | Runs |
|---|---|---|
| A | none (k = 0), the baseline, whose final policy is also the stage 2 start | 3 |
| B | random checks, k ∈ {1, 2, 4, 7} × level c ∈ {0.5, 1, 2}, with penalty per catch P = 7c / k apples. Each level fixes k × P, the expected penalty per over-harvest, around week 2's break-even k × P ≈ 7 apples. | 36 |
| C | random k = 2, freeze 50 steps, no penalty | 3 |
| D | targeted k = 2 at c = 1 (P = 3.5): checking by catch history, and checking by location | 6 |

**Stage 2: switching the overseer on for agents that already over-harvest.**
- Start from arm A's final policy, same seed, 5e7 steps.
- Arms:
  - random c = 1 for each k ∈ {1, 2, 4, 7};
  - the freeze arm;
  - history c = 1;
  - a no-overseer continuation (the control for extra training).
- 7 × 3 = 21 runs.
- Stage 2 starts only after all three arm A runs finish.

## Measures

Each update is one episode in each of the 256 envs.

- **Main outcomes:**
  - apples eaten per episode (welfare);
  - orchard alive at the end of the episode (share of envs);
  - apples left at the end.
- **Overseer measures:**
  - over-harvesting (violations), caught, missed;
  - normal harvesting wrongly stopped (0 by construction with a perfect inspector);
  - penalties paid (apples) and checks used (the cost).
- **Reported value:** the mean of the last 10 updates, per seed. Then the mean over the 3 seeds, with the seed range (3 seeds is too few for intervals).
- **Switch-off test:** the final policy plays 64 episodes with the overseer on and 64 with it off, with no learning. Over-harvesting that comes back with the overseer off means the agents complied but did not change.

## Questions and predictions, written before the runs

| Question | Prediction |
|---|---|
| Q1. Do IPPO learners with no overseer over-harvest? | Yes. Fewer apples than 7 restrained bots (500), and the orchard often dies, as the SocialJax paper reports for individual-reward IPPO in Harvest Open. |
| Q2. Which k × P protects the orchard, and is it the expected penalty that matters? | c ≥ 1 cuts violations and raises apples eaten over arm A; c = 0.5 does less. Rows with the same c look alike across k. Learning agents discount the future, so the level needed may differ from the bots' 7 apples; this is the test. |
| Q3. Freeze or penalty? | A 50-step freeze also protects the orchard directly, but learners may not learn to avoid it, since it changes no reward until the agent misses apples. |
| Q4. Random vs targeted at k = 2? | Agents cannot see checks, so they cannot exploit targeting. History targeting should catch more and protect better than random; location similar. (Targeting being exploited is phase 4, when agents can see checks.) |
| Q5. Does switching on the overseer fix agents that already over-harvest? | Yes, with c ≥ 1, but more slowly than starting with it. |
| Q6. Switch-off | Some over-harvesting returns when the overseer is removed, as R1b found in Fishery. |

**What would change the plan:**
- If arm A does not over-harvest (orchard alive, violations near the restrained bots' level), there is no problem for the overseer to fix. Then we change the setting (e.g. Harvest with fewer apples) before phases 3–4.
- If no arm beats arm A, check the penalty scale before going on.

A pilot of two runs (arm A and arm B k = 2, c = 1, seed 1, 2e7 steps) checks the GPU path and logging first. Its results are not analysed.

## Amendment 1 (15 Nov, after the pilot, before stage 1)

- **Pilot** (seed 1, 2e7 steps, H200, about 4 min each; not analysed as a result): both runs completed and logged correctly. The no-overseer learners stripped the orchard within 10 updates.
- **The planned switch-off test cannot work for penalty-only overseers.** With learning stopped, a penalty changes rewards but not actions, so "overseer on" and "overseer off" episodes are identical. The pilot showed exactly this.
- **Replaced by stage 3, switch-off training:**
  - start from the final policy of each stage 1 arm: random c = 1 for every k, the freeze arm, and history;
  - train 5e7 steps with **no overseer**;
  - 6 arms × 3 seeds = 18 runs.
  - Q6 becomes: does over-harvesting return once the agents keep learning without the overseer?
- The end-of-run evaluation is kept, because it still matters for the freeze arm, where a freeze changes actions.
- **Timing:** a 1e8-step run takes about 20 min on an H200, so all stages fit in a day.

## Amendment 2 (16 Nov, after stage 1; wording only)

- The protocol called k × P "the expected penalty per over-harvest". That is wrong by a factor of 7. Under random checks an over-harvest is checked with chance k/7, so the expected penalty is (k/7) × P = c apples.
- The three levels are therefore an expected 0.5, 1 and 2 apples per over-harvest (k × P = 3.5, 7, 14). Week 2's break-even k × P ≈ 7 is an expected penalty of about 1 apple.
- The runs, the arms and the predictions are unchanged. Only the label changes. `results.md` uses the corrected wording.
