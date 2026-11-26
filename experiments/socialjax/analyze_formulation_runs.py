"""End-of-training tables for the Phase 3 runs: H2 (visible checks), H3b (inverted inspector), H6 (separate nets, 3e8).

  .venv-socialjax/bin/python -m experiments.socialjax.analyze_formulation_runs [--pull results/create_pull]

Every number is the mean over the last 10% of PPO updates of each run, then mean and range over seeds.
Hypotheses and predictions: notes/formulation/03_hypotheses.md (written before the runs).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def tail(m, key, frac=0.1):
    x = m[key]
    n = max(1, int(round(len(x) * frac)))
    return x[-n:].mean(0)


def run_stats(path):
    m = np.load(path)
    v = float(tail(m, "violations"))
    out = dict(
        eaten=float(tail(m, "apples_eaten")),
        stock=float(tail(m, "apples_mean")),
        alive=float(tail(m, "orchard_alive")),
        viol=v,
        caught=float(tail(m, "detected")) / v if v else float("nan"),
    )
    if "violations_by_near" in m:
        vb = tail(m, "violations_by_near")
        out["near"] = (vb / vb.sum()).tolist() if vb.sum() else [float("nan")] * 3
    if "checks_on_violations" in m:
        cv = float(tail(m, "checks_on_violations"))
        out["alpha_real"] = 1 - float(tail(m, "detected")) / cv if cv else float("nan")
    # levelling off: last 10% vs the 10% before it (apples eaten)
    x = m["apples_eaten"]
    n = max(1, len(x) // 10)
    out["drift"] = float(x[-n:].mean() - x[-2 * n:-n].mean())
    return out


def group(pull, pattern):
    paths = sorted(glob.glob(os.path.join(pull, pattern, "metrics.npz")))
    return [run_stats(p) for p in paths]


def agg(rows, key):
    xs = [r[key] for r in rows]
    return float(np.mean(xs)), float(np.min(xs)), float(np.max(xs))


def line(name, rows, keys=("eaten", "alive", "viol", "caught")):
    if not rows:
        return f"| {name} | 0 |" + " – |" * len(keys)
    cells = []
    for k in keys:
        m, lo, hi = agg(rows, k)
        cells.append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]" if k in ("alive", "caught", "alpha_real") else f"{m:.0f} [{lo:.0f}, {hi:.0f}]" if k == "eaten" else f"{m:.1f}")
    return f"| {name} | {len(rows)} | " + " | ".join(cells) + " |"


H2 = [  # (look policy, ordinary pattern, visible-checks pattern)
    ("central random", "B_random_k2_c1.0_seed*", "WS_central_random_k2_c1.0_seed*"),
    ("central history", "D_history_k2_c1.0_seed*", "WS_central_history_k2_c1.0_seed*"),
    ("central round robin", "OS_central_roundrobin_k2_c1.0_seed*", "WS_central_roundrobin_k2_c1.0_seed*"),
    ("central mixed (30% random)", "OS_central_mixed0.3_k2_c1.0_seed*", "WS_central_mixed0.3_k2_c1.0_seed*"),
    ("2 overseers, random", "DH_R2_random_k2_c1.0_seed*", "WS_R2_random_k2_c1.0_seed*"),
    ("2 overseers, history + sharing", "DH_R2_share_k2_c1.0_seed*", "WS_R2_share_k2_c1.0_seed*"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", default="results/create_pull")
    ap.add_argument("--json", default="notes/formulation/phase3_runs_summary.json")
    a = ap.parse_args()
    summary = {}
    keys = ("eaten", "alive", "viol", "caught")
    hdr = "| arm | seeds | apples eaten | orchard alive | violations | share caught |\n|---|---|---|---|---|---|"

    print("## H2: ordinary vs visible checks (shared network, k = 2, c = 1)\n")
    print("| look policy | J ordinary (apples) | J visible | gap Δ | alive ord / vis | caught ord / vis |\n|---|---|---|---|---|---|")
    for name, op, wp in H2:
        o, w = group(a.pull, op), group(a.pull, wp)
        jo, jw = agg(o, "eaten")[0], agg(w, "eaten")[0]
        print(f"| {name} | {jo:.0f} (n={len(o)}) | {jw:.0f} [{agg(w,'eaten')[1]:.0f}, {agg(w,'eaten')[2]:.0f}] (n={len(w)}) | **{jo - jw:.0f}** "
              f"| {agg(o,'alive')[0]:.2f} / {agg(w,'alive')[0]:.2f} | {agg(o,'caught')[0]:.2f} / {agg(w,'caught')[0]:.2f} |")
        summary[f"H2/{name}"] = dict(ordinary=o, visible=w, delta=jo - jw)
    none = group(a.pull, "A_none_seed*")
    print(f"\nNo overseer (A_none): {agg(none,'eaten')[0]:.0f} apples, alive {agg(none,'alive')[0]:.2f}\n")

    print("## H3b: inverted inspector vs coin flip vs gpt-oss table\n")
    print("| k, c | inspector | seeds | apples eaten | alive | violations | caught | realised α | violations at 0 / 1 / 2 apples left |\n|---|---|---|---|---|---|---|---|---|")
    for k in (1, 7):
        for c in ("1.0", "2.0"):
            for arm in ("coin", "table", "inverted"):
                rows = group(a.pull, f"E_gptoss20b_{arm}_k{k}_c{c}_seed*")
                if not rows:
                    continue
                near = np.nanmean([r["near"] for r in rows], 0)
                print(f"| {k}, {c} | {arm} | {len(rows)} | {agg(rows,'eaten')[0]:.0f} [{agg(rows,'eaten')[1]:.0f}, {agg(rows,'eaten')[2]:.0f}] "
                      f"| {agg(rows,'alive')[0]:.2f} | {agg(rows,'viol')[0]:.1f} | {agg(rows,'caught')[0]:.2f} | {agg(rows,'alpha_real')[0]:.2f} "
                      f"| {near[0]:.2f} / {near[1]:.2f} / {near[2]:.2f} |")
                summary[f"H3b/k{k}_c{c}/{arm}"] = rows

    print("\n## H6: separate networks at 3e8 vs shared at 1e8 vs separate at 1e8\n")
    print(hdr.replace("| arm |", "| arm |") + "")
    for name, p in [("shared 1e8, central random", "B_random_k2_c1.0_seed*"), ("separate 1e8, central random", "SN_central_random_k2_c1.0_seed*"),
                    ("separate 3e8, central random", "SL_central_random_k2_c1.0_seed*"), ("shared 1e8, 2 overseers random", "DH_R2_random_k2_c1.0_seed*"),
                    ("separate 1e8, 2 overseers random", "SN_R2_random_k2_c1.0_seed*"), ("separate 3e8, 2 overseers random", "SL_R2_random_k2_c1.0_seed*")]:
        rows = group(a.pull, p)
        print(line(name, rows, keys) + (f" drift (last 10% − previous 10%): {np.mean([r['drift'] for r in rows]):+.1f}" if rows else ""))
        summary[f"H6/{name}"] = rows

    with open(a.json, "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
