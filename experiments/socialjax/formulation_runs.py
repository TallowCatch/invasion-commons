"""GPU job lists for the formulation experiments X1-X4, X6, X7, X10 (notes/formulation/05_formulation.md, section 7).

Predictions for every list were written in 05_formulation.md (26 Nov 2026) before any of these runs.

  python -m experiments.socialjax.formulation_runs x1 > runs_x1.tsv   (56: seeds 4, 5 on the key arms)
  python -m experiments.socialjax.formulation_runs x2 > runs_x2.tsv   (33: randomisation sweep, Props 5.3-5.5)
  python -m experiments.socialjax.formulation_runs x3 > runs_x3.tsv   (6: k = 1 vs 7 at 3e8 steps, spread remark)
  python -m experiments.socialjax.formulation_runs x4 > runs_x4.tsv   (6: Clean Up full inspection, Prop 5.11)
  python -m experiments.socialjax.formulation_runs x6 > runs_x6.tsv   (12: headcount-proportional budgets, Prop 5.12)
  python -m experiments.socialjax.formulation_runs x7 > runs_x7.tsv   (6: public catches, section 5.8)
  python -m experiments.socialjax.formulation_runs x10 > runs_x10.tsv (3: Clean Up with W = 25, Prop 5.11)
"""
from __future__ import annotations

import argparse

from experiments.llm_overseer import error_pattern_runs as epr
from experiments.socialjax import decentralised_runs as dr
from experiments.socialjax import phase2_runs as p2

SEEDS = (1, 2, 3)
NEW_SEEDS = (4, 5)


def _with_seeds(gen, seeds, keep):
    """Run an existing list generator with other seeds and keep the arms whose name (minus _seedN) is in keep."""
    old = dr.SEEDS, p2.SEEDS
    dr.SEEDS = p2.SEEDS = seeds
    try:
        return [(n, a) for n, a in gen() if n.rsplit("_seed", 1)[0] in keep]
    finally:
        dr.SEEDS, p2.SEEDS = old


X1_STAGE1 = {"B_random_k1_c1.0", "B_random_k7_c1.0", "B_random_k2_c0.5", "B_random_k2_c1.0", "D_history_k2_c1.0"}
X1_WS = {"WS_central_random_k2_c1.0", "WS_central_history_k2_c1.0", "WS_central_roundrobin_k2_c1.0",
         "WS_central_mixed0.3_k2_c1.0", "OS_central_roundrobin_k2_c1.0", "OS_central_mixed0.3_k2_c1.0"}
X1_CU = {"DU_central_random_w150_k2_c1.0", "DU_central_random_w50_k2_c1.0", "DU_central_random_w50_presume_k2_c1.0",
         "DU_central_random_w150_k2_c2.0", "DU_none"}


def x1():
    """Seeds 4 and 5 on the 28 key arms (H1, H2, H3, H3b, H9): with 5 seeds an exact permutation test can reach
    p = 0.008; with 3 it cannot go below 0.10."""
    yield from _with_seeds(p2.stage1, NEW_SEEDS, X1_STAGE1)
    yield from _with_seeds(dr.worstcase_shared, NEW_SEEDS, X1_WS)
    yield from _with_seeds(dr.cleanup, NEW_SEEDS, X1_CU)
    tags = ["gpt-oss-20b_direct"]
    for name, args in epr.runs(tags, seeds=NEW_SEEDS):
        if "_coinov_" not in name:
            yield name, args
    yield from epr.inverted(seeds=NEW_SEEDS)


def x2():
    """Props 5.3-5.5: randomisation needed against agents who see the looks, and a bigger sanction cannot replace it."""
    for s in SEEDS:
        for c, epss in ((1.0, (0.5, 0.7, 0.9)), (2.0, (0.3, 0.5, 0.7))):
            p = f"--env harvest --k 2 --penalty-apples {dr.penalty(7, 2, c)} --seed {s}"
            for e in epss:
                yield f"WS_central_mixed{e}_k2_c{c}_seed{s}", f"{p} --reveal-checks --chooser mixed --mixed-eps {e}"
        for c in (2.0, 4.0):
            p = f"--env harvest --k 2 --penalty-apples {dr.penalty(7, 2, c)} --seed {s}"
            yield f"WS_central_history_k2_c{c}_seed{s}", f"{p} --reveal-checks --chooser history"
        p = f"--env harvest --k 2 --penalty-apples {dr.penalty(7, 2, 1.0)} --seed {s}"
        for e in (0.5, 0.7, 0.9):
            yield f"OS_central_mixed{e}_k2_c1.0_seed{s}", f"{p} --chooser mixed --mixed-eps {e}"


def x3():
    """Spread remark (Prop 5.1): does the k = 1 vs k = 7 gap at equal rho close with 3x longer training?"""
    for s in SEEDS:
        for k in (1, 7):
            yield (f"BL_random_k{k}_c1.0_seed{s}",
                   f"--k {k} --chooser random --penalty-apples {p2.penalty(k, 1.0)} --seed {s} --total-steps 3e8")


def x4():
    """Prop 5.11: Clean Up with every agent looked at every step by a perfect judge reading 150 steps."""
    for s in SEEDS:
        for c in (1.0, 2.0):
            yield (f"DU_central_random_w150_k7_c{c}_seed{s}",
                   f"--env cleanup --k 7 --seed {s} --cost-per-step-seen 0.01 --penalty-apples {dr.penalty(7, 7, c)} "
                   f"--chooser random --window 150")


def x6():
    """Prop 5.12: regional budgets split by headcount each step, 4 and 2 regions, random checks, c = 1. Compare
    DH_R4_random_k2_c1.0 and DH_R2_random_k2_c1.0 (equal budgets) and B_random_k2_c1.0 (central).
    DP_: largest-remainder rounding (the planned arm). DPU_ (added 26 Nov before any run): unbiased rounding, so
    p = k/N in every state as Prop 5.12 assumes; with 4 regions largest remainder leaves a lone agent unwatched in
    about 13% of agent-steps (notes/formulation/x6_x7_x10/predictions.md)."""
    for s in SEEDS:
        p = f"--env harvest --k 2 --penalty-apples {dr.penalty(7, 2, 1.0)} --seed {s} --chooser random"
        for R in (4, 2):
            yield f"DP_R{R}_random_k2_c1.0_seed{s}", f"{p} --regions {R} --region-alloc proportional"
        for R in (4, 2):
            yield f"DPU_R{R}_random_k2_c1.0_seed{s}", f"{p} --regions {R} --region-alloc proportional_random"


def x7():
    """Section 5.8: catches made public in the ordinary mode (looks still hidden). Compare D_history_k2_c1.0 and
    B_random_k2_c1.0 (same arms, catches hidden)."""
    for s in SEEDS:
        p = f"--env harvest --k 2 --penalty-apples {dr.penalty(7, 2, 1.0)} --seed {s} --public-catches"
        for chooser in ("history", "random"):
            yield f"PC_central_{chooser}_k2_c1.0_seed{s}", f"{p} --chooser {chooser}"


def x10():
    """Prop 5.11: Clean Up with the stricter rule W = 25 (clean at least once every 25 steps), looks reading w = 50
    steps (>= W, so every violation is provable). Scripted bots under W = 25 first: notes/formulation/x10_strict_rule/
    bots.md. Compare DU_central_random_w150_k2_c1.0 (W = 100)."""
    for s in SEEDS:
        yield (f"DS_W25_random_w50_k2_c1.0_seed{s}",
               f"--env cleanup --k 2 --seed {s} --cost-per-step-seen 0.01 --penalty-apples {dr.penalty(7, 2, 1.0)} "
               f"--chooser random --window 50 --clean-window 25")


LISTS = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x6": x6, "x7": x7, "x10": x10}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("list", choices=tuple(LISTS))
    for run, args in LISTS[ap.parse_args().list]():
        print(f"{run}\t{args}")
