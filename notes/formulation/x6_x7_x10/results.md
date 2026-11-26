# X6, X7, X10: results

Predictions: `../05_formulation.md` §5.6 (X10), §5.7 (X6), §5.8 (X7), and `predictions.md` here (implementation
choices and the extra unbiased-rounding X6 arm, written 26 Nov before any run). X10 scripted bots:
`../x10_strict_rule/bots.md`.

## Status (26 Nov 2026)

Code on branch `claude/x6-x7-x10`; CREATE checkout `/scratch/users/k26108758/socialjax_work/invasion-commons-x6`.

| list | runs | CREATE job | notes |
|---|---|---|---|
| X6 + X7 (Harvest) | 12 + 6 | 38899190 (array 1–18, %8) | `~/ippo/runs_x6_x7.tsv` |
| X10 (Clean Up) | 3 | 38899192 (array 1–3, h200/b200/l40s) | `~/ippo/runs_x10.tsv` |

All learner runs are queued behind the 101 X1–X4 runs. Tables below fill in with
`python -m experiments.socialjax.analyze_x6_x7_x10 --pull results/create_pull` once `metrics.npz` is pulled.

## X10 step 1 (done): scripted bots under W = 25

The condition for the X10 learner prediction holds. Seven rule-keeping bots keep the orchard to the end in 64 of
64 games under W = 25 (38% under W = 100), with 0.13 pollution crossings per game (6.1 under W = 100). Details and
caveats in `../x10_strict_rule/bots.md`.

## Baselines (last 10% of training, mean over seeds, per-seed values in brackets)

| arm | apples eaten | orchard alive | violations | share caught |
|---|---|---|---|---|
| central random `B_random_k2_c1.0` | 460 (441, 466, 473) | 1.00 | 38.6 | 0.29 |
| 2 regions, equal budgets `DH_R2_random_k2_c1.0` | 439 (475, 436, 408) | 1.00 | 41.1 | 0.27 |
| 4 regions, equal budgets `DH_R4_random_k2_c1.0` | 76 (77, 75, 77) | 0.00 | 31.4 | 0.18 |
| central repeat offenders first `D_history_k2_c1.0` | 474 (470, 465, 487) | 1.00 | 30.1 | 0.29 |
| Clean Up, W = 100, random, w = 150 `DU_central_random_w150_k2_c1.0` | 130 (75, 227, 89) | 0.52 | 11.9 | 1.00 (pollution 0.66) |

## Results

Pending.
