# The overseer in SocialJax Harvest (phase 1)

Built 15 Nov 2026, as weeks 1–2 of `notes/phd_plan/2026-11-15_phased_plan.md`.

## Set up and run (laptop, CPU)

```bash
bash scripts/socialjax_local_setup.sh   # clones SocialJax into external/SocialJax (commit 18e2171), makes .venv-socialjax
.venv-socialjax/bin/python -m pytest -q tests/test_socialjax_overseer.py
.venv-socialjax/bin/python -m experiments.socialjax.run_bots_preview --out results/runs/socialjax_bots_preview
```

One 1,000-step episode takes about 1.6 s on an M1 CPU. The SocialJax environment id is `harvest_common_open`; training scripts call it `harvest_open`.

## What the code does

| File | What it is |
|---|---|
| `overseer/socialjax/harvest_overseer.py` | The wrapper. Each step: record over-harvesting, check k of 7 agents, apply a consequence, log what was caught, missed and wrongly stopped. |
| `overseer/socialjax/harvest_bots.py` | Scripted players: restrained, greedy, sneaky, random. |
| `overseer/socialjax/rollout.py` | Runs one episode with `jax.lax.scan` and summarises it. |
| `tests/test_socialjax_overseer.py` | 15 tests; they skip outside `.venv-socialjax`. |
| `experiments/socialjax/run_bots_preview.py` | First look: every bot mix × every way of choosing whom to check. |

**The rule.** Eating an apple that leaves fewer than m apples (default 3) within two squares is over-harvesting.
- Harvest regrows an empty cell at 2.5%, 0.5% or 0.1% per step when 3+, 2 or 1 apples are near it, and never when none are (`harvest_open.py`, `regrow_apple`). So this is the move that kills patches.
- The rule is judged on the map the agent saw when it moved. An earlier version judged it after everyone moved, and then restrained bots "broke" it about 30 times an episode only because another agent took a neighbouring apple in the same step.
- Rewards are individual (`shared_rewards=False`), so it is a social dilemma. One apple pays 7.

**The overseer's dials.** This is where Yali's feedback is built in.

| Dial | Setting | Meaning |
|---|---|---|
| Sight | `k`, `window` | How many agents it checks per step; how many past steps a check reveals (1 = it must catch the act) |
| Whom | `chooser` | none, all, random, round_robin, history (caught before, else longest unchecked), location (standing where fewest apples are left) |
| Smarts (scripted overseer) | `miss_rate`, `false_alarm_rate` | A weaker inspector misses real over-harvesting or flags normal harvesting. For a learned or LLM overseer this becomes model size. |
| Spend | `cost_per_check`, `cost_per_step_seen` | Each check costs a base amount plus an amount per step of history read. This is the token cost of a longer look. |
| Consequence | `penalty`, `freeze_steps` | Reward removed from a caught agent, and/or steps it is held still |

**What is logged:** over-harvesting, checks, catches, false alarms (normal harvesting wrongly stopped), violations detected and missed, apples on the map, penalties and cost. Every violation ends up either detected or missed; a test checks this.

## First look with scripted bots (preview, not a frozen study)

Setup: 5 seeds per row, 999 steps, k = 2 of 7, window 1, a caught agent is frozen for 50 steps, no reward penalty.

| Bots | Who gets checked | Over-harvesting caught / total | Apples eaten | Orchard alive at the end (of 5) |
|---|---|---|---|---|
| 1 greedy + 6 restrained | none | 0.0 / 27.2 | 84 | 0 |
| 1 greedy + 6 restrained | random | 7.2 / 30.4 | 152 | 0 |
| 1 greedy + 6 restrained | round robin | 8.4 / 28.2 | 175 | 0 |
| 1 greedy + 6 restrained | history | 19.4 / 24.4 | 337 | 5 |
| 1 greedy + 6 restrained | location | 9.8 / 30.8 | 207 | 0 |
| 1 greedy + 6 restrained | all | 19.8 / 19.8 | 362 | 5 |
| 7 sneaky | none | 0.0 / 28.0 | 70 | 0 |
| 7 sneaky | random | 1.2 / 7.4 | 483 | 5 |
| 7 sneaky | round robin | 0.6 / 2.0 | 509 | 5 |
| 7 sneaky | history | 0.8 / 2.8 | 512 | 5 |
| 7 sneaky | location | 8.4 / 32.4 | 141 | 0 |
| 7 sneaky | all | 3.2 / 3.2 | 499 | 5 |
| 7 greedy | none | 0.0 / 30.8 | 68 | 0 |
| 7 greedy | random | 10.0 / 31.2 | 70 | 0 |
| 7 greedy | round robin | 8.2 / 30.2 | 72 | 0 |
| 7 greedy | history | 2.4 / 30.4 | 69 | 0 |
| 7 greedy | location | 6.2 / 30.4 | 68 | 0 |
| 7 greedy | all | 42.2 / 42.2 | 105 | 0 |
| 7 restrained | none | 0.0 / 2.2 | 508 | 5 |
| 7 restrained | random | 0.6 / 1.8 | 513 | 5 |
| 7 restrained | round robin | 0.6 / 2.0 | 509 | 5 |
| 7 restrained | history | 0.8 / 2.8 | 512 | 5 |
| 7 restrained | location | 0.2 / 2.2 | 513 | 5 |
| 7 restrained | all | 3.2 / 3.2 | 499 | 5 |

What it suggests (scripted bots only; they don't learn, and greedy bots ignore consequences):
- **One greedy agent wrecks the orchard** with no overseer (84 apples eaten, dead in all 5), as the SocialJax paper reports for Harvest Open.
- **Targeting by catch history saves it with 2 checks per step:** 337 apples, alive in 5/5, close to checking all 7 every step (362). Random and round-robin checks at the same budget don't save it.
- **Sneaky bots** (greedy once unchecked for 20 steps) **beat location targeting.** It keeps checking agents in bare patches and leaves the rest unwatched (141 apples, 0/5 alive). Random, round-robin and history keep them honest. Note that "history" works like round-robin here because no one gets caught. This is the Harvest version of our Fishery T1 result: predictable targeting can be exploited.
- **Seven greedy bots strip the orchard whatever the overseer does.** A freeze after the fact is too late when everyone defects.

Next (week 2): a Schelling diagram with and without the overseer, choose m, the heatmap of k × consequence, and sneaky bots with different patience. Then IPPO on CREATE.
