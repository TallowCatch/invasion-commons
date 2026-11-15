"""First look at the Harvest overseer with scripted bots (phase 1 preview; not a frozen study).

For each bot population and each way of choosing whom to check (k agents per step), runs `--seeds` episodes and
prints over-harvesting, how much of it was caught or missed, wrongly stopped harvesting, apples eaten and whether
the orchard survives. Writes one row per (population, chooser, seed) to --out as CSV.

  .venv-socialjax/bin/python -m experiments.socialjax.run_bots_preview --out results/runs/socialjax_bots_preview
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import numpy as np

from overseer.socialjax import CHOOSERS, HarvestOverseer, OverseerConfig, make_harvest_env
from overseer.socialjax.harvest_bots import kind_codes
from overseer.socialjax.rollout import run_episode, summarise

POPULATIONS = {
    "7 restrained": ("restrained",) * 7,
    "1 greedy + 6 restrained": ("greedy",) + ("restrained",) * 6,
    "7 sneaky": ("sneaky",) * 7,
    "7 greedy": ("greedy",) * 7,
}
COLUMNS = ("violations", "detected", "missed", "wrongly_stopped", "checks", "apples_eaten", "apples_end",
           "orchard_alive", "equality")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=999)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--freeze", type=int, default=50)
    ap.add_argument("--penalty", type=float, default=0.0)
    a = ap.parse_args()

    env = make_harvest_env(num_inner_steps=a.steps + 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pop, kinds in POPULATIONS.items():
        codes = kind_codes(kinds)
        for chooser in CHOOSERS:
            ov = HarvestOverseer(env, OverseerConfig(k=a.k, chooser=chooser, freeze_steps=a.freeze, penalty=a.penalty))
            for seed in range(a.seeds):
                m = summarise(*run_episode(ov, jax.random.PRNGKey(seed), codes, steps=a.steps))
                rows.append(dict(population=pop, chooser=chooser, seed=seed, k=a.k, freeze=a.freeze,
                                 penalty=a.penalty, **{c: m[c] for c in COLUMNS}))
            sub = [r for r in rows if r["population"] == pop and r["chooser"] == chooser]
            mean = {c: np.mean([r[c] for r in sub]) for c in COLUMNS}
            print(f"{pop:24s} {chooser:12s} " + " ".join(f"{c}={mean[c]:.2f}" for c in COLUMNS), flush=True)
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
