"""Analysis for R3 (break-even confirmation) and T1 (audit targeting). Claude audit, October 2026.

Protocols: notes/claude_audit_20261005/studies/{R3_breakeven_confirmation,T1_audit_targeting}/protocol.md
Run:  PYTHONPATH=. python -m experiments.oversight.analyze_r3_t1 --r3 results/runs/claude_r3_v1 --t1 results/runs/claude_t1_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.oversight.analyze_s1b_s2 import load

B = 4000


def boot_diff(x, y, rng):
    d = np.asarray(x, float) - np.asarray(y, float)
    draws = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return dict(estimate=float(d.mean()), ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])


def boot_share_diff(num_x, den_x, num_y, den_y, rng):
    """Difference of pooled shares (sum num / sum den), paired context bootstrap."""
    nx, dx, ny, dy = (np.asarray(v, float) for v in (num_x, den_x, num_y, den_y))
    f = lambda i: nx[i].sum() / max(dx[i].sum(), 1) - ny[i].sum() / max(dy[i].sum(), 1)
    n = len(nx)
    draws = [f(rng.integers(0, n, n)) for _ in range(B)]
    return dict(estimate=float(f(np.arange(n))), ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])


def r3(run):
    rng = np.random.default_rng(20261017)
    cells = json.loads((run / "cells.json").read_text())
    E = load(run / "episodes.jsonl.gz")
    rows, S = [], {}
    for c in cells:
        rec = dict(cell=c["cell"], n_stress=c["n_stress"], regen=c["level"], g_star=c["g_star"], d_allow=c["d_allow"],
                   g_r2_definition=c["g_r2_definition"], testable=c["testable"], e_star=c["e_star"], extended=c.get("extended"))
        if c["testable"]:
            g, step = c["g_star"], 0.2 * c["g_star"]
            rec["in_band"] = c["e_star"] is not None and 0.8 * g - 1e-9 <= c["e_star"] <= 1.2 * g + 1e-9
            rec["e_star_over_g"] = None if c["e_star"] is None else c["e_star"] / g
            ds = [s["d_star"] for s in sorted(c["searches"], key=lambda s: s["e"])]
            rec["monotone"] = all(ds[i + 1] <= ds[i] for i in range(len(ds) - 1))
            cheating = [s for s in c["searches"] if s["d_star"] > 0]
            if cheating:
                last = max(cheating, key=lambda s: s["e"])
                comply = E[(E.cell == c["cell"]) & (E.arm == "comply")].sort_values("context")
                a = E[(E.cell == c["cell"]) & (E.arm == "adaptive") & np.isclose(E.e.astype(float), last["e"])].sort_values("context")
                n_cheat = a.n_cheaters.iloc[0]
                rec["last_cheating_e_over_g"] = last["e"] / g
                rec["heldout_gain_per_cheater_at_last_cheating_e"] = boot_diff(a.cheater_payoff.values / n_cheat,
                                                                                comply.cheater_payoff.values / n_cheat, rng)
            rec["d_star_by_e_over_g"] = {f'{s["e"] / g:.1f}': s["d_star"] for s in c["searches"]}
        rows.append(rec)
    testable = [r for r in rows if r["testable"]]
    S["cells"] = rows
    S["hypotheses"] = dict(
        R3_H1_all_in_band=all(r["in_band"] for r in testable), R3_H1_in_band=f'{sum(r["in_band"] for r in testable)} of {len(testable)}',
        R3_H2_all_positive=all(r.get("heldout_gain_per_cheater_at_last_cheating_e", {"ci": [0]})["ci"][0] > 0 for r in testable),
        R3_H3_all_monotone=all(r["monotone"] for r in testable))
    (run / "analysis").mkdir(exist_ok=True)
    (run / "analysis" / "r3_summary.json").write_text(json.dumps(S, indent=1, default=float))
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows]).to_csv(run / "analysis" / "r3_cells.csv", index=False)
    return S


def t1(run):
    rng = np.random.default_rng(20261018)
    E = load(run / "episodes.jsonl.gz")
    S, table = {}, []
    for game in ("fishery", "harvest", "river"):
        G = E[E.game == game]
        arms = {a: G[G.arm == a].sort_values("context") for a in G.arm.unique()}
        for a, d in arms.items():
            table.append(dict(game=game, arm=a, contexts=len(d), harm=float(d.harm.mean()),
                              audits_on_liars_share=float(d.audits_on_liars.sum() / max(d.audits.sum(), 1)),
                              catches_per_context=float(d.catches.mean()),
                              honest=float((d.honest_harvest if "honest_harvest" in d else d.honest_payoff).mean())))
        r, rp = arms["random"], arms["report"]
        S[game] = dict(report_minus_random_harm=boot_diff(rp.harm.values, r.harm.values, rng),
                       report_minus_random_liar_share=boot_share_diff(rp.audits_on_liars.values, rp.audits.values,
                                                                      r.audits_on_liars.values, r.audits.values, rng),
                       random_minus_trust_harm=boot_diff(r.harm.values, arms["trust"].harm.values, rng))
        trust = arms["trust"].harm.values
        S[game]["harm_relative_to_trust"] = {}
        for a_name, d in arms.items():
            if a_name == "trust":
                continue
            x = d.harm.values
            f = lambda i: x[i].mean() / max(trust[i].mean(), 1e-12)
            draws = [f(rng.integers(0, len(x), len(x))) for _ in range(B)]
            S[game]["harm_relative_to_trust"][a_name] = dict(estimate=float(f(np.arange(len(x)))),
                                                             ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])
        if "signal" in arms:
            S[game]["signal_minus_random_harm"] = boot_diff(arms["signal"].harm.values, r.harm.values, rng)
            S[game]["signal_minus_random_liar_share"] = boot_share_diff(arms["signal"].audits_on_liars.values,
                                                                        arms["signal"].audits.values,
                                                                        r.audits_on_liars.values, r.audits.values, rng)
    h1 = sum(S[g]["report_minus_random_harm"]["ci"][0] > 0 for g in ("fishery", "harvest", "river"))
    h2 = sum(S[g]["report_minus_random_liar_share"]["estimate"] < 0 for g in ("fishery", "harvest", "river"))
    S["hypotheses"] = dict(T1_H1_games=h1, T1_H1=h1 >= 2, T1_H2_games=h2, T1_H2=h2 >= 2,
                           T1_H3=not (S["harvest"]["signal_minus_random_harm"]["ci"][0] > 0))
    (run / "analysis").mkdir(exist_ok=True)
    (run / "analysis" / "t1_summary.json").write_text(json.dumps(S, indent=1, default=float))
    pd.DataFrame(table).to_csv(run / "analysis" / "t1_table.csv", index=False)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r3")
    ap.add_argument("--t1")
    a = ap.parse_args()
    out = {}
    if a.r3:
        out["R3"] = r3(Path(a.r3))["hypotheses"]
    if a.t1:
        out["T1"] = t1(Path(a.t1))["hypotheses"]
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
