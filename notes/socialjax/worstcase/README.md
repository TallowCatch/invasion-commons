# Separate networks and the worst case: code and GPU jobs

Built 22–23 Nov 2026 on branch `claude/separate-nets-worstcase`. Protocol and predictions: `protocol.md`. Bot preview: `results.md`.

## What the code does

| File | What it is |
|---|---|
| `overseer/socialjax/reveal.py` | The `reveal_checks` observation: 3 channels after the game's own (who was checked last step, in the agent's rotated view; "I was checked last step"; "steps since I was last checked", capped at 20, scaled to 0–1). `topk_chance` gives each agent's exact chance of being among the top k with random tie-breaks |
| `overseer/socialjax/harvest_overseer.py`, `decentralised.py` | `--chooser mixed` (each step, with chance `mixed_eps` the checks are random, otherwise by history; each regional overseer flips its own coin); `reveal_checks`; `obs_shape()`; `check_prob(state)`, each agent's chance of being checked in the coming step from public information (the rule, budgets and record, not the coin flips). R = 1 still reproduces the central overseer step for step |
| `overseer/socialjax/harvest_bots.py` | `just_checked` (cheats in the 2 steps after its own check) and `dodger` (cheats when `check_prob` is below `dodge_below`; 0.999 / P cheats exactly when the expected penalty is below the apple gained) |
| `experiments/socialjax/train_ippo_overseer.py` | `--separate-nets` (one network and Adam state per agent, vmapped; each agent's minibatches hold only its own transitions; `--init-params` may be a shared run's params, copied to every agent), `--reveal-checks`, `--mixed-eps`; per-agent measures (`apples_agent`, `violations_agent`, `catches_agent`, `net_reward_agent`) in every run. The default path is unchanged |
| `experiments/socialjax/decentralised_runs.py` | Job lists `worstcase_pilot` (2), `separate` (15), `worstcase` (18) |
| `experiments/socialjax/run_worstcase_bots.py` | The CPU bot preview |
| `tests/test_worstcase.py` | 23 tests: the default trainer path gives the same parameters and measures as the trainer at fca047d; separate-nets shapes and per-agent minibatches; a shared run's params copied into separate nets give the same policy; the reveal map lines up with the game's channels in Harvest, Coins and Clean Up; R = 1 equals the central overseer for `mixed` and with `reveal_checks`; `check_prob` matches how often agents are checked (central and 2 regions); dodgers are never caught by round robin; the job lists parse |

**R = 2 caveat.** In `DecentralisedOverseer` an agent's record of its last w steps travels with it and any overseer's check reads all of it, including steps spent in other regions. So the R = 2 arms are regional overseers with a free, full hand-over of records. They measure where checks land, not evidence lost at boundaries. These runs use w = 1, so there is nothing to carry across anyway.

## Local (CPU)

```bash
.venv-socialjax/bin/python -m pytest -q tests/test_worstcase.py tests/test_decentralised_overseer.py
.venv-socialjax/bin/python -m experiments.socialjax.run_worstcase_bots --out results/runs/socialjax_worstcase_bots
.venv-socialjax/bin/python -m experiments.socialjax.train_ippo_overseer --out results/runs/train_smoke/worstcase_1e5 \
   --separate-nets --reveal-checks --k 2 --chooser mixed --penalty-apples 3.5 --total-steps 1e5 --num-envs 4 --eval-envs 4
```

**CPU check, 23 Nov** (the command above: separate nets, visible checks, mixed checks; 25 updates of 4 envs, M1 laptop, sharing the CPU with the test suite for its first 5 min):
- It ran end to end in 3,810 s of training plus the two evaluations. `config.json`, `metrics.npz`, `params.msgpack` and `summary.json` were written; the 8 MB `params.msgpack` was then deleted to save disk.
- Its last 10 updates averaged 178 apples and 47 over-harvests per episode, 15 of them caught. Over-harvests per agent ranged from 5.2 to 8.5.
- The evaluation with the overseer on gave 193 apples, 53 over-harvests, 16 caught, 2,000 checks, orchard alive in 4 of 4 envs. With the overseer off it gave 178 apples and 50 over-harvests.
- Nothing is learned in 1e5 steps; this only shows the code runs. With a penalty and no freeze, the two evaluations differ only through the policy's sampled actions and the reveal channels (with the overseer off nobody is shown as checked, and "steps since I was checked" climbs to the cap).

## GPU jobs on CREATE

1e8 steps, seeds 1–3, Harvest, k = 2, P = 3.5 apples (average penalty 1 apple under random checks).

| List | Runs | What |
|---|---|---|
| `worstcase_pilot` | 2 | 2e7 steps: separate nets with central random; separate nets with visible checks, 2 strips sharing. To time both before the full lists |
| `separate` | 15 | separate nets, checks hidden: central random, central history, 2 strips random, history and sharing. The shared arms are stage 1 and the `DH_` runs |
| `worstcase` | 18 | separate nets, checks visible: central random, history, round robin and mixed (ε = 0.3); 2 strips random and sharing |

**Setup, once**, in its own checkout so other jobs keep their code:

```bash
WORK=/scratch/users/$USER/socialjax_work
git clone -b claude/separate-nets-worstcase https://github.com/TallowCatch/invasion-commons.git $WORK/invasion-commons-wc
cp $WORK/invasion-commons-wc/scripts/create/ippo_overseer_array.sbatch ~/ippo/ippo_overseer_array_wc.sbatch
cd $WORK/invasion-commons-wc
for L in worstcase_pilot separate worstcase; do
  python3 -m experiments.socialjax.decentralised_runs $L > ~/ippo/runs_$L.tsv
done
wc -l ~/ippo/runs_{worstcase_pilot,separate,worstcase}.tsv     # 2, 15, 18
```

**Pilot first:**

```bash
sbatch --array=1-2 --time=03:00:00 \
  --export=ALL,RUNS=$HOME/ippo/runs_worstcase_pilot.tsv,REPO=$WORK/invasion-commons-wc ~/ippo/ippo_overseer_array_wc.sbatch
```

- Read `train_seconds` from each pilot's `summary.json` (2e7 steps). Multiply by 5 for a 1e8 run, plus about 5 min of start-up, and set `--time` from that.
- If a run runs out of GPU memory, add `--num-envs 128` to its lines (NUM_MINIBATCHES = 500 still divides 128 × 1000).

**Then the full lists** (8 GPUs at a time):

```bash
for L in separate worstcase; do
  N=$(wc -l < ~/ippo/runs_$L.tsv)
  sbatch --array=1-$N%8 --time=04:00:00 \
    --export=ALL,RUNS=$HOME/ippo/runs_$L.tsv,REPO=$WORK/invasion-commons-wc ~/ippo/ippo_overseer_array_wc.sbatch
done
```

**Outputs:** `/scratch/users/$USER/ippo_overseer/<run name>/` (`config.json`, `metrics.npz`, `params.msgpack`, `summary.json`). Names start with `SN_` or `WC_`. `params.msgpack` holds 7 stacked networks (leading axis = agent).

**Total:** 35 runs (33 plus the 2 pilots). The GPU-hours are guesses until the pilot is timed: about 8–15 h for `separate` and 10–20 h for `worstcase`.
