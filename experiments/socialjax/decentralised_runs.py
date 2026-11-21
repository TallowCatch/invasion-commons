"""GPU job lists for central vs decentralised overseers (notes/socialjax/DECENTRALISED_README.md).

One line per run, "name<TAB>arguments" for experiments.socialjax.train_ippo_overseer, read by
scripts/create/ippo_overseer_array.sbatch (task i runs line i).

  python -m experiments.socialjax.decentralised_runs pilot   > runs_dec_pilot.tsv     (4 short runs, one per game)
  python -m experiments.socialjax.decentralised_runs harvest > runs_dec_harvest.tsv   (69 runs)
  python -m experiments.socialjax.decentralised_runs harvest_large > runs_dec_harvest_large.tsv
  python -m experiments.socialjax.decentralised_runs coins   > runs_dec_coins.tsv
  python -m experiments.socialjax.decentralised_runs cleanup > runs_dec_cleanup.tsv
  python -m experiments.socialjax.decentralised_runs all     > runs_dec_all.tsv

Every arm at a given k has the same total budget: k checks per step, summed over the regions.
The penalty per catch is set so the average penalty per violation under random checks is c goods:
(k / N) x P = c, so P = N c / k (as in stage 1, phase2_runs.penalty). Decentralised random checks at the same k and
P have about the same average penalty (minus checks wasted on empty regions; the bots measured how much).
"""
from __future__ import annotations

import argparse

SEEDS = (1, 2, 3)
SHARE = "--share --cost-per-message 0.25"


def penalty(n, k, c):
    return round(n * c / k, 4)


def harvest():
    """Harvest (7 agents), k = 2, 4 at average penalties of 1 and 2 apples.
    Central random (all k, c) and central history (k = 2, c = 1) were run in stage 1 (phase2_runs.stage1:
    B_random_k{k}_c{c}, D_history_k2_c1.0) with the same code path, so they are not repeated here."""
    for s in SEEDS:
        for k in (2, 4):
            for c in (1.0, 2.0):
                p = f"--env harvest --k {k} --penalty-apples {penalty(7, k, c)} --seed {s}"
                if not (k == 2 and c == 1.0):
                    yield f"DH_central_history_k{k}_c{c}_seed{s}", f"{p} --chooser history"
                yield f"DH_R2_random_k{k}_c{c}_seed{s}", f"{p} --chooser random --regions 2"
                yield f"DH_R2_history_k{k}_c{c}_seed{s}", f"{p} --chooser history --regions 2"
                yield f"DH_R2_share_k{k}_c{c}_seed{s}", f"{p} --chooser history --regions 2 {SHARE}"
                yield f"DH_R4_random_k{k}_c{c}_seed{s}", f"{p} --chooser random --regions 4"
                yield f"DH_R4_share_k{k}_c{c}_seed{s}", f"{p} --chooser history --regions 4 {SHARE}"


def harvest_large():
    """Harvest on the doubled map (14 agents), k = 4 (the same checks per agent as k = 2 with 7), c = 1 apple."""
    for s in SEEDS:
        yield f"DL_none_seed{s}", f"--env harvest_large --k 0 --seed {s}"
        p = f"--env harvest_large --k 4 --penalty-apples {penalty(14, 4, 1.0)} --seed {s}"
        yield f"DL_central_random_k4_c1.0_seed{s}", f"{p} --chooser random"
        yield f"DL_central_history_k4_c1.0_seed{s}", f"{p} --chooser history"
        yield f"DL_R2_random_k4_c1.0_seed{s}", f"{p} --chooser random --regions 2"
        yield f"DL_R2_share_k4_c1.0_seed{s}", f"{p} --chooser history --regions 2 {SHARE}"
        yield f"DL_R4_random_k4_c1.0_seed{s}", f"{p} --chooser random --regions 4"
        yield f"DL_R4_share_k4_c1.0_seed{s}", f"{p} --chooser history --regions 4 {SHARE}"


def coins():
    """Coins (2 players), top and bottom halves; a theft gains the thief 1 coin, so c = 1 is break-even."""
    for s in SEEDS:
        yield f"DC_none_seed{s}", f"--env coins --k 0 --seed {s}"
        for c in (1.0, 2.0):
            k1 = f"--env coins --k 1 --penalty-apples {penalty(2, 1, c)} --seed {s}"
            yield f"DC_central_random_k1_c{c}_seed{s}", f"{k1} --chooser random"
            yield f"DC_central_history_k1_c{c}_seed{s}", f"{k1} --chooser history"
            yield f"DC_R2_random_k1_c{c}_seed{s}", f"{k1} --chooser random --regions 2 --layout rows"
            yield f"DC_R2_history_k1_c{c}_seed{s}", f"{k1} --chooser history --regions 2 --layout rows"
            yield f"DC_R2_share_k1_c{c}_seed{s}", f"{k1} --chooser history --regions 2 --layout rows {SHARE}"
            k2 = f"--env coins --k 2 --penalty-apples {penalty(2, 2, c)} --seed {s}"
            yield f"DC_central_all_k2_c{c}_seed{s}", f"{k2} --chooser random"
            yield f"DC_R2_random_k2_c{c}_seed{s}", f"{k2} --chooser random --regions 2 --layout rows"


def cleanup():
    """Clean Up (7 agents), rule W = 100 steps, pollution threshold 0.2; checks read w steps (cost 1 + 0.01 w)."""
    for s in SEEDS:
        yield f"DU_none_seed{s}", f"--env cleanup --k 0 --seed {s}"
        base = f"--env cleanup --k 2 --seed {s} --cost-per-step-seen 0.01"
        for c in (1.0, 2.0):
            p = f"{base} --penalty-apples {penalty(7, 2, c)}"
            yield f"DU_central_random_w150_k2_c{c}_seed{s}", f"{p} --chooser random --window 150"
        p = f"{base} --penalty-apples {penalty(7, 2, 1.0)}"
        yield f"DU_central_history_w150_k2_c1.0_seed{s}", f"{p} --chooser history --window 150"
        yield f"DU_central_random_w50_k2_c1.0_seed{s}", f"{p} --chooser random --window 50"
        yield f"DU_central_random_w50_presume_k2_c1.0_seed{s}", f"{p} --chooser random --window 50 --presume"
        yield f"DU_R2_random_w150_k2_c1.0_seed{s}", f"{p} --chooser random --window 150 --regions 2"
        yield f"DU_R2_share_w150_k2_c1.0_seed{s}", f"{p} --chooser history --window 150 --regions 2 {SHARE}"


def pilot():
    """One short run per game (2e7 steps) to check memory and time on the GPU before the full lists."""
    yield "pilot_DH_R2_share_k2_c1.0", f"--env harvest --k 2 --penalty-apples 3.5 --chooser history --regions 2 {SHARE} --seed 1 --total-steps 2e7"
    yield "pilot_DL_R4_random_k4_c1.0", f"--env harvest_large --k 4 --penalty-apples 3.5 --chooser random --regions 4 --seed 1 --total-steps 2e7"
    yield "pilot_DC_R2_random_k1_c2.0", "--env coins --k 1 --penalty-apples 4.0 --chooser random --regions 2 --layout rows --seed 1 --total-steps 2e7"
    yield "pilot_DU_central_random_w150_k2_c1.0", "--env cleanup --k 2 --penalty-apples 3.5 --chooser random --window 150 --cost-per-step-seen 0.01 --seed 1 --total-steps 2e7"


LISTS = {"pilot": pilot, "harvest": harvest, "harvest_large": harvest_large, "coins": coins, "cleanup": cleanup}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("list", choices=tuple(LISTS) + ("all",))
    a = ap.parse_args()
    names = [x for x in LISTS if x != "pilot"] if a.list == "all" else [a.list]
    for name in names:
        for run, args in LISTS[name]():
            print(f"{run}\t{args}")
