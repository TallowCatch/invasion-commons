# Decentralised overseers in SocialJax: code, bot runs and GPU jobs

Built 21–22 Nov 2026 on branch `claude/decentralised-games`. Summary of results: `DECENTRALISED_SUMMARY.md`.

## What the code does

| File | What it is |
|---|---|
| `overseer/socialjax/regions.py` | Splits a map into R regions: vertical strips, horizontal bands, or a grid (R = 4 gives quadrants); which regions are neighbours |
| `overseer/socialjax/decentralised.py` | `DecentralisedOverseer`: one overseer per region sharing a total budget of k checks per step; independent or sharing; per-region logs. With R = 1 it is the central overseer and reproduces `HarvestOverseer` step for step |
| `overseer/socialjax/games.py` | The rule per game: `HarvestRules` (also the doubled 16 x 44 map with 14 agents), `CoinsRules`, `CleanupRules` |
| `overseer/socialjax/game_bots.py` | Honest, greedy, sneaky, random and migrator bots for each game |
| `overseer/socialjax/harvest_bots.py` | Harvest bots, now with the migrator |
| `overseer/socialjax/game_rollout.py` | Jitted episode runner (`run_seeds`) and episode summary with per-region measures |
| `tests/test_decentralised_overseer.py` | 26 tests (R = 1 equals the central overseer; regions; budgets; sharing; each game's rule; bots) |
| `experiments/socialjax/run_decentralised_bots.py` | The CPU bot sweeps |
| `experiments/socialjax/analyze_decentralised.py`, `make_decentralised_figures.py` | Tables and figures |
| `experiments/socialjax/train_ippo_overseer.py` | IPPO trainer, now with `--env`, `--regions`, `--share` and the budget split |
| `experiments/socialjax/decentralised_runs.py` | GPU job lists |

### The overseers

- **Regions and budget.** The map is split into R regions, with one overseer per region.
  - An overseer can only check agents standing in its region at that step.
  - The total budget is k checks per step. Each region gets k/R. When k is not a multiple of R, the spare checks go to randomly chosen regions each step, so the expected split is equal.
  - `--region-k 2,1` sets the split by hand.
  - A region with fewer agents than checks leaves the rest unused. This is logged as `unused_r`.
- **Independent overseers** (`--regions R`): each knows only its own checks and catches.
- **Sharing** (`--share`): every step, each overseer sends its suspicion scores (its decaying count of catches per agent) to each neighbouring region, and each keeps the higher score.
  - Each message costs `--cost-per-message` (default 0.25; a check costs 1).
  - `--share-coverage` also shares when each agent was last checked.
  - Sharing only changes the "history" chooser (repeat offenders first, then the longest unchecked).
- **Logs, per step and per region:** agents present, checks, catches, wrong flags, unused checks, and violations, detections and misses by the region where the violation happened; also check cost and communication cost.

### The rules

| Game | SocialJax env | Agents | Rule | Look a check needs |
|---|---|---|---|---|
| Harvest | `harvest_common_open` | 7 (14 on the doubled map) | don't eat an apple that leaves fewer than 3 within two squares | 1 step |
| Coins | `coin_game` | 2 (hard-coded in SocialJax) | don't take a coin of the other player's colour | 1 step |
| Clean Up | `clean_up` | 7 | don't eat while pollution > 0.2 unless you cleaned (removed dirt) in the last W = 100 steps | W steps |

- When a check reads fewer steps (`--window w`) than the rule needs, it cannot prove the violation.
- With `--presume`, the overseer instead flags any act that looks like a violation in what it can see. This catches more violations and also flags honest agents.

### Bigger maps

`--env harvest_large` tiles the Harvest map twice side by side (16 x 44) with 14 agents. SocialJax Harvest takes `num_agents`, `map_ASCII` and `grid_size`, so this needed no change to SocialJax.

Coins can't take more than 2 players without rewriting its reward code. Clean Up takes `num_agents` but has one river, so a tiled map would change the game; it was left at 7 agents.

## Local (CPU)

```bash
bash scripts/socialjax_local_setup.sh          # SocialJax at 18e2171 + .venv-socialjax with CPU JAX
.venv-socialjax/bin/python -m pytest -q tests/test_socialjax_overseer.py tests/test_decentralised_overseer.py
# bot sweeps (4-7 min each on a laptop; JSON under results/, git-ignored)
for g in harvest harvest_large coins cleanup; do
  .venv-socialjax/bin/python -m experiments.socialjax.run_decentralised_bots --game $g --out results/runs/socialjax_decentralised
  .venv-socialjax/bin/python -m experiments.socialjax.analyze_decentralised --game $g --run results/runs/socialjax_decentralised
done
.venv-socialjax/bin/python -m experiments.socialjax.make_decentralised_figures --run results/runs/socialjax_decentralised
```

**Trainer smoke tests on CPU** (22 Nov, M1 laptop, 4 envs). Every game ran end to end: training, saving params and metrics, and the with/without-overseer evaluation.

```bash
P=".venv-socialjax/bin/python -m experiments.socialjax.train_ippo_overseer"
$P --out results/runs/train_smoke/harvest_R2_1e5 --env harvest --k 2 --chooser history --penalty-apples 3.5 \
   --regions 2 --share --total-steps 1e5 --num-envs 4 --eval-envs 4          # 25 updates
$P --out results/runs/train_smoke/coins_R2 --env coins --k 1 --chooser random --penalty-apples 4 --regions 2 \
   --layout rows --total-steps 8e3 --num-envs 4 --eval-envs 2
$P --out results/runs/train_smoke/cleanup_presume --env cleanup --k 2 --chooser random --penalty-apples 3.5 \
   --window 50 --presume --cost-per-step-seen 0.01 --total-steps 8e3 --num-envs 4 --eval-envs 2
$P --out results/runs/train_smoke/harvest_large_R4 --env harvest_large --k 4 --chooser history --penalty-apples 3.5 \
   --regions 4 --share --total-steps 8e3 --num-envs 4 --eval-envs 2
```

SMOKE_TIMINGS

## GPU jobs on CREATE

**The job lists** come from `experiments/socialjax/decentralised_runs.py`, one line per run ("name<TAB>arguments"):
- 1e8 steps, 3 seeds (1–3);
- penalty per catch P = N c / k goods, so the average penalty per violation under random checks is c (as in stage 1).

| List | Runs | What |
|---|---|---|
| `pilot` | 4 | one 2e7-step run per game, to check time and memory first |
| `harvest` | 69 | k = 2, 4 × average penalty 1 and 2 apples: 2 strips (random; repeat offenders first, independent; repeat offenders first with sharing), 4 strips (random; sharing), and central history where stage 1 lacks it. Central random and central history at k = 2, c = 1 are stage 1's `B_random_*` and `D_history_k2_c1.0` (same code path), so they are not repeated |
| `harvest_large` | 21 | 14 agents, k = 4, c = 1: none, central random, central history, 2 and 4 strips (random; sharing) |
| `coins` | 45 | none; k = 1 at c = 1 and 2 (central random, central history, 2 halves random, independent, sharing); k = 2 (central = check both, 2 halves random) |
| `cleanup` | 24 | none; central random with a 150-step look at c = 1 and 2; central history; a 50-step look (cannot prove anything); a 50-step look presuming guilt; 2 strips random; 2 strips with sharing |

**Setup, once.** Use a separate checkout so the stage 2–3 jobs keep running on their own code. The array script reads `REPO`, which defaults to `$WORK/invasion-commons` as before.

```bash
# on a CREATE login node
WORK=/scratch/users/$USER/socialjax_work
git clone -b claude/decentralised-games https://github.com/TallowCatch/invasion-commons.git $WORK/invasion-commons-dec
# (later: cd $WORK/invasion-commons-dec && git pull)
cp $WORK/invasion-commons-dec/scripts/create/ippo_overseer_array.sbatch ~/ippo/ippo_overseer_array_dec.sbatch
```

**Make the job lists.** Plain Python, no JAX needed:

```bash
cd $WORK/invasion-commons-dec
for L in pilot harvest harvest_large coins cleanup; do
  python3 -m experiments.socialjax.decentralised_runs $L > ~/ippo/runs_dec_$L.tsv
done
wc -l ~/ippo/runs_dec_*.tsv     # 4, 69, 21, 45, 24
```

**Pilot first.** The time limit is raised because Clean Up and the 14-agent map are slower per step than Harvest; the script's default is 2 h:

```bash
sbatch --array=1-4 --time=04:00:00 \
  --export=ALL,RUNS=$HOME/ippo/runs_dec_pilot.tsv,REPO=$WORK/invasion-commons-dec ~/ippo/ippo_overseer_array_dec.sbatch
```

- Read the wall time from each pilot's `summary.json` (`train_seconds`, for 2e7 steps). Multiply by 5 for a 1e8 run, plus about 5 min of start-up.
- If `harvest_large` runs out of GPU memory, add `--num-envs 128` to its lines. NUM_MINIBATCHES (500) still divides the batch.

**Then the full lists** (8 GPUs at a time, the public_gpu limit). Set `--time` from the pilot:

```bash
for L in harvest harvest_large coins cleanup; do
  N=$(wc -l < ~/ippo/runs_dec_$L.tsv)
  sbatch --array=1-$N%8 --time=04:00:00 \
    --export=ALL,RUNS=$HOME/ippo/runs_dec_$L.tsv,REPO=$WORK/invasion-commons-dec ~/ippo/ippo_overseer_array_dec.sbatch
done
```

**Where the outputs go:** `/scratch/users/$USER/ippo_overseer/<run name>/`, holding `config.json`, `metrics.npz`, `params.msgpack` and `summary.json`.

- Run names start with `DH_`, `DL_`, `DC_` or `DU_`, so they don't collide with stage 1–3.
- `metrics.npz` has, per update, everything stage 1 logged, plus:
  - `comm_cost` and `unused_checks`;
  - per-region `checks_r`, `catches_r`, `violations_r`, `missed_r`;
  - `pollution_mean` and `cleaned` for Clean Up.
- The metric names keep Harvest's words: `apples_eaten` means goods taken, and `orchard_alive` means goods left on the map at the end.
- `summary.json` ends with the same switch-off test as stage 1.

**Total:** 159 runs. At stage 1's 18–38 min per Harvest run on an A100/H200, the Harvest list (69 runs) takes about 3–6 h on 8 GPUs. The other games need the pilot's timings first.
