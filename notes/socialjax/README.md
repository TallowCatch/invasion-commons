# The overseer in SocialJax Harvest (phase 1)

Built 15 Nov 2026, as weeks 1–2 of `notes/phd_plan/2026-11-15_phased_plan.md`.

## Set up and run (laptop, CPU)

```bash
bash scripts/socialjax_local_setup.sh   # clones SocialJax into external/SocialJax (commit 18e2171), makes .venv-socialjax
.venv-socialjax/bin/python -m pytest -q tests/test_socialjax_overseer.py
.venv-socialjax/bin/python -m experiments.socialjax.run_bots_preview --out results/runs/socialjax_bots_preview
```

One 1,000-step episode takes about 1.6 s on an M1 CPU. The SocialJax environment id is `harvest_common_open`; training scripts call it `harvest_open`.

## CREATE (GPU training)

- **Smoke job 38280051** (`scripts/create/smoke_harvest.sbatch`, 28 Oct):
  - Completed with exit code 0 in 7 min 46 s on an NVIDIA A40 (`interruptible_gpu`, node erc-hpc-comp127), at about 6 GB of memory.
  - It was IPPO on Harvest Open, 8 parallel envs, 200k steps; the training loop, GIF saving and JAX on the GPU all worked.
  - CREATE runs the same SocialJax commit as the laptop (18e2171).
- **Reward setting:** SocialJax's Harvest IPPO config defaults to `reward: common` (one shared team reward), which removes the dilemma. Our runs must pass **`reward=individual`**. The smoke job used the default.
- **Throughput** (`scripts/create/throughput_harvest.sbatch`): IPPO, individual rewards, the default 256 parallel envs. One update = 256 envs x 1,000 steps; a 1e8-step run is about 390 updates.

  | Job | GPU | Updates | Wall time (including start-up) |
  |---|---|---|---|
  | 38654061 | — | — | Failed in 5 s: node without the python module (fixed) |
  | 38654092 | A100 80GB | 2 | Preempted on `interruptible_gpu` after 8 min |
  | 38654239 | A100 40GB | 1 / 3 | 445 s / 356 s (start-up varies more than the work, so no per-update number) |
  | 38654514 | H200 | 21 | 421 s |

- **What this means:**
  - Start-up (compiling, loading, saving a GIF) takes several minutes and varies between runs.
  - On an H200, an update costs at most 20 s even if start-up were free. So a full 1e8-step run takes **at most about 2.2 h, and probably about 1 h**. That is at least about 12,800 environment steps per second (5.4M steps in 421 s).
  - On an A100 40GB, the 3-update run bounds it at 120 s per update, but that bound is loose. A full run is probably a few hours.
  - The `gpu` partition hands out different GPUs (A100 40GB, then H200). Pin one type with `--constraint` when timing matters.
  - Use `gpu`, not `interruptible_gpu`, for long runs; otherwise they need to checkpoint and resume.

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

Week 2 (done 15 Nov): `week2_bots/results.md`: m = 3, Schelling diagrams, k × freeze and k × penalty heatmaps. Next: IPPO on CREATE.

Phase 2 with IPPO learners (stage 1 done 16 Nov): `phase2_ippo/results.md`. With no overseer the learners strip the orchard in every env (68 apples). Random checks save it once an over-harvest costs an expected 1 apple (chance of a check k/7 × penalty), at any k, and 0.5 apples does nothing. A 50-step freeze fails. History targeting matches random at the end but learns faster. Location targeting fails (5% of over-harvests caught). Stages 2 (overseer switched on late) and 3 (switched off) are queued.

CREATE note (16 Nov): node arc-hpc-comp035 shows no GPU, so jobs there silently ran on the CPU. The array script now stops if JAX sees no GPU, and jobs exclude that node.
