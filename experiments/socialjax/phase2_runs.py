"""Phase 2 job list (notes/socialjax/phase2_ippo/protocol.md): one line per run, "name<TAB>arguments".

  python -m experiments.socialjax.phase2_runs stage1 > runs_stage1.tsv
  python -m experiments.socialjax.phase2_runs stage2 --checkpoints DIR > runs_stage2.tsv   (DIR holds A_seed*/)
  python -m experiments.socialjax.phase2_runs stage3 --checkpoints DIR > runs_stage3.tsv   (switch-off training)
  python -m experiments.socialjax.phase2_runs pilot > runs_pilot.tsv
"""
from __future__ import annotations

import argparse

SEEDS = (1, 2, 3)
KS = (1, 2, 4, 7)
LEVELS = (0.5, 1.0, 2.0)  # expected penalty per over-harvest, k x P, in units of 7 apples


def penalty(k, c):
    return round(7 * c / k, 4)


def stage1():
    for s in SEEDS:
        yield f"A_none_seed{s}", f"--k 0 --seed {s}"
        for k in KS:
            for c in LEVELS:
                yield f"B_random_k{k}_c{c}_seed{s}", f"--k {k} --chooser random --penalty-apples {penalty(k, c)} --seed {s}"
        yield f"C_freeze50_k2_seed{s}", f"--k 2 --chooser random --freeze 50 --seed {s}"
        for ch in ("history", "location"):
            yield f"D_{ch}_k2_c1.0_seed{s}", f"--k 2 --chooser {ch} --penalty-apples {penalty(2, 1.0)} --seed {s}"


def stage2(checkpoints):
    for s in SEEDS:
        init = f"--init-params {checkpoints}/A_none_seed{s}/params.msgpack --total-steps 5e7 --seed {s}"
        yield f"R_none_seed{s}", f"--k 0 {init}"
        for k in KS:
            yield f"R_random_k{k}_c1.0_seed{s}", f"--k {k} --chooser random --penalty-apples {penalty(k, 1.0)} {init}"
        yield f"R_freeze50_k2_seed{s}", f"--k 2 --chooser random --freeze 50 {init}"
        yield f"R_history_k2_c1.0_seed{s}", f"--k 2 --chooser history --penalty-apples {penalty(2, 1.0)} {init}"


def stage3(checkpoints):
    """Switch-off: agents trained under an overseer keep learning for 5e7 steps with no overseer (Amendment 1)."""
    arms = [f"B_random_k{k}_c1.0" for k in KS] + ["C_freeze50_k2", "D_history_k2_c1.0"]
    for s in SEEDS:
        for arm in arms:
            yield (f"S_off_after_{arm}_seed{s}",
                   f"--k 0 --init-params {checkpoints}/{arm}_seed{s}/params.msgpack --total-steps 5e7 --seed {s}")


def pilot():
    yield "pilot_A_none_seed1", "--k 0 --seed 1 --total-steps 2e7"
    yield "pilot_B_random_k2_c1.0_seed1", f"--k 2 --chooser random --penalty-apples {penalty(2, 1.0)} --seed 1 --total-steps 2e7"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("stage1", "stage2", "stage3", "pilot"))
    ap.add_argument("--checkpoints", default="/scratch/users/k26108758/ippo_overseer")
    a = ap.parse_args()
    runs = {"stage1": stage1, "pilot": pilot, "stage2": lambda: stage2(a.checkpoints),
            "stage3": lambda: stage3(a.checkpoints)}[a.stage]()
    for name, args in runs:
        print(f"{name}\t{args}")
