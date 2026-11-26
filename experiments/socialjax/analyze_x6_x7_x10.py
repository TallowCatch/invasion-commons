"""End-of-training tables for X6 (headcount budgets), X7 (public catches) and X10 (Clean Up, W = 25), with their
baselines. Predictions: notes/formulation/05_formulation.md sections 5.6-5.8 and notes/formulation/x6_x7_x10/
predictions.md (written before the runs). Write-up: notes/formulation/x6_x7_x10/results.md.

  .venv-socialjax/bin/python -m experiments.socialjax.analyze_x6_x7_x10 [--pull results/create_pull]

Every number is the mean over the last 10% of PPO updates of one run (per seed), then the mean over seeds.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from experiments.socialjax.analyze_formulation_runs import run_stats, tail

TABLES = {
    "X6 (Harvest, k = 2, c = 1, random looks)": [
        ("central (B_random)", "B_random_k2_c1.0"),
        ("2 regions, equal budgets", "DH_R2_random_k2_c1.0"),
        ("2 regions, largest remainder", "DP_R2_random_k2_c1.0"),
        ("2 regions, unbiased rounding", "DPU_R2_random_k2_c1.0"),
        ("4 regions, equal budgets", "DH_R4_random_k2_c1.0"),
        ("4 regions, largest remainder", "DP_R4_random_k2_c1.0"),
        ("4 regions, unbiased rounding", "DPU_R4_random_k2_c1.0"),
    ],
    "X7 (Harvest, k = 2, c = 1, central, looks hidden)": [
        ("repeat offenders first, catches hidden", "D_history_k2_c1.0"),
        ("repeat offenders first, catches public", "PC_central_history_k2_c1.0"),
        ("random, catches hidden", "B_random_k2_c1.0"),
        ("random, catches public", "PC_central_random_k2_c1.0"),
    ],
    "X10 (Clean Up, k = 2, c = 1, random looks)": [
        ("rule W = 100, look window 150", "DU_central_random_w150_k2_c1.0"),
        ("rule W = 25, look window 50", "DS_W25_random_w50_k2_c1.0"),
    ],
}


def seeds(pull, arm):
    out = []
    for path in sorted(glob.glob(os.path.join(pull, f"{arm}_seed*", "metrics.npz"))):
        r = run_stats(path)
        m = np.load(path)
        r["seed"] = path.split("_seed")[-1].split("/")[0]
        r["updates"] = int(len(m["apples_eaten"]))
        if "pollution_mean" in m:
            r["pollution"] = float(tail(m, "pollution_mean"))
            r["cleaned"] = float(tail(m, "cleaned"))
        out.append(r)
    return out


def fmt(rows, key, nd):
    xs = [r[key] for r in rows if key in r]
    if not xs:
        return "–"
    per = ", ".join(f"{x:.{nd}f}" for x in xs)
    return f"{np.nanmean(xs):.{nd}f} ({per})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", default="results/create_pull")
    ap.add_argument("--json", default="notes/formulation/x6_x7_x10/summary.json")
    a = ap.parse_args()
    summary = {}
    for title, arms in TABLES.items():
        cu = title.startswith("X10")
        print(f"\n## {title}\n")
        cols = ["apples eaten", "orchard alive", "violations", "share caught"] + (["pollution", "cleaned"] if cu else [])
        print("| arm | runs | " + " | ".join(cols) + " | drift |")
        print("|---|---|" + "---|" * (len(cols) + 1))
        for name, arm in arms:
            rows = seeds(a.pull, arm)
            summary[arm] = rows
            keys = [("eaten", 0), ("alive", 2), ("viol", 1), ("caught", 2)] + ([("pollution", 3), ("cleaned", 0)] if cu else [])
            cells = [fmt(rows, k, nd) for k, nd in keys]
            print(f"| {name} (`{arm}`) | {len(rows)} | " + " | ".join(cells) + f" | {fmt(rows, 'drift', 0)} |")
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
