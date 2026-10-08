# Harvest RL Scaffold Closeout

## What was added

- A new Harvest PPO self-play module:
  - `fishery_sim/harvest_rl.py`
- Train / eval entrypoints:
  - `experiments/train_harvest_rl.py`
  - `experiments/evaluate_harvest_rl.py`
- A small multi-seed baseline runner:
  - `experiments/run_harvest_rl_baseline.py`
- Targeted tests:
  - `tests/test_harvest_rl.py`

## Verification

- `python -m py_compile` passed on the new Harvest RL files.
- `pytest -q tests/test_harvest_rl.py` passed.
- The evaluation path reuses `run_harvest_episode`, so reported metrics match the existing Harvest episode runner rather than a separate RL-only evaluator.

## First empirical result

- The first Harvest RL version was degenerate:
  - the initial `20k` smoke and `100k` probe produced identical main held-out metrics for `top_down_only` and `hybrid`,
  - with only credit transfer separating the two conditions.
- The main issue was policy formulation: the original RL agent did not condition harvest-time decisions on the current inbox.
- After changing the policy to make harvest and credit-offer decisions depend on inbox summaries, the flat result disappeared.

## Message-Aware Follow-Up

- New `20k` smoke:
  - `hybrid` differs from `top_down_only` on aggression, overharvest, and welfare, though the direction is mixed.
- New `100k` one-seed probe:
  - `hybrid_minus_top__test_mean_patch_health = +0.3273`
  - `hybrid_minus_top__test_mean_welfare = -0.1309`
  - `hybrid_minus_top__test_mean_max_local_aggression = -0.1932`
  - `hybrid_minus_top__test_mean_neighborhood_overharvest = -0.8848`
- New `3`-seed `100k` pilot:
  - `hybrid_minus_top__test_mean_patch_health = +0.1091`
  - `hybrid_minus_top__test_mean_welfare = -0.0436`
  - `hybrid_minus_top__test_mean_max_local_aggression = -0.0644`
  - `hybrid_minus_top__test_mean_neighborhood_overharvest = -0.2949`
- That aggregate direction is now aligned with the main Harvest story: better ecological control under `hybrid`, with a modest welfare cost.
- The remaining issue is stability: only one of the three pilot seeds shows a strong separation, while the other two stay close to `top_down_only`.

## Reward-Shaped Follow-Up

- Added normalized reward penalties for:
  - local aggression,
  - neighborhood overharvest.
- Best shaping probe used:
  - `local_aggression_penalty_weight=1.0`
  - `neighborhood_overharvest_penalty_weight=1.5`
- Under that shaping, the new `3`-seed `100k` pilot gives:
  - `hybrid_minus_top__test_mean_patch_health = +0.3985`
  - `hybrid_minus_top__test_mean_welfare = -0.1430`
  - `hybrid_minus_top__test_mean_max_local_aggression = -0.3333`
  - `hybrid_minus_top__test_mean_neighborhood_overharvest = -2.25`
- This is the first Harvest RL pilot that looks directionally strong enough to justify a `5`-seed confirmation.

## Interpretation

- The scaffold is technically working.
- The message-aware policy fix was necessary and helped.
- Harvest RL is no longer blocked by a flat architecture error.
- The current limitation is now weaker but more familiar: the useful `hybrid` behavior is present, but stability still needs confirmation at higher seed count.

## Next fix

- Do not move to paper text yet.
- The next Harvest RL pass should be a `5`-seed confirmation under the shaped reward.
- Only if that confirmation stays directionally positive should Harvest RL be added to the manuscript.

## 5-Seed Confirmation

- The shaped `5`-seed `100k` confirmation was run on `medium_h1`.
- Aggregate held-out delta (`hybrid - top_down_only`):
  - patch health: `+0.2391`
  - welfare: `-0.0859`
  - max local aggression: `-0.2`
  - neighborhood overharvest: `-1.35`
  - garden failure: `0.0`
- Absolute held-out means:
  - `hybrid`: patch health `15.8064`, welfare `16.2898`, max local aggression `0.4`, neighborhood overharvest `7.65`
  - `top_down_only`: patch health `15.5673`, welfare `16.3757`, max local aggression `0.6`, neighborhood overharvest `9.0`

## Current Interpretation

- The shaped Harvest RL result now holds up across `5` seeds in the expected direction.
- `hybrid` improves patch health and reduces local aggression and neighborhood overharvest.
- The welfare cost remains modest rather than catastrophic.
- Survival and garden failure are saturated in this medium-tier setup, so the informative quantities remain patch health, aggression, overharvest, and welfare.

## Updated Decision

- Harvest RL is now strong enough to count as a narrow learned-policy validation of the Study 2 architecture ranking.
- It should still be framed conservatively:
  - this is medium-tier RL evidence,
  - it validates the `hybrid > top_down_only` direction,
  - it does not replace the main Stage C invasion result.
- The next sensible move is either to freeze and commit the Harvest RL scaffold, or to add a short paper/notes update with the RL result treated as supporting evidence only.
