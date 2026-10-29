"""G1b (post hoc, G1 protocol Amendment 3): does a lone over-taker need a larger fine than a cheating group?

Prediction on the 8 G1 training populations: g_lone = mean over agents i of
[i's net payoff when it alone over-takes (d = 1) - i's net payoff when it complies] / i's over-take rounds,
reviewer on, no audits; F_lone = g_lone / q. Test on the 64 G1 test populations: the Diagram-2 gain at every m
for F in {0, 1/2, 1, 3/2} x F_lone. Everything else is G1's code (run_g1_schelling).

  PYTHONPATH=. python3 -m experiments.oversight.run_g1b_lone_breakeven --out results/runs/claude_g1b_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.oversight import run_g1_schelling as g1

MULTS = (0.0, 0.5, 1.0, 1.5)


def predict_lone(setting, base, n_train):
    """g_lone on training populations (Amendment 3, item 1)."""
    per_pop = []
    for c in range(n_train):
        pop, sd = g1.population(setting, base, "train", c), g1.ctx_seeds(base, "train", setting, c)
        req = g1.paper_greedy(setting, pop)
        none = np.zeros(setting.n, bool)
        e0 = g1.episode(setting, pop, req, none, 0.0, 0.0, True, False, sd)
        gains = []
        for i in range(setting.n):
            alone = none.copy()
            alone[i] = True
            e = g1.episode(setting, pop, req, alone, 1.0, 0.0, True, False, sd)
            n_over = float(e["over_steps"][i])
            if n_over > 0:
                gains.append((g1.cheater_net(e, [i]) - g1.cheater_net(e0, [i])) / n_over)
        per_pop.append(float(np.mean(gains)) if gains else 0.0)
    return float(np.mean(per_pop)), per_pop


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--populations", type=int, default=64)
    ap.add_argument("--train", type=int, default=8)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    s = g1.Setting("fishery", horizon=80)
    g_lone, per_pop = predict_lone(s, g1.G1_BASE, a.train)
    F_lone = g_lone / g1.Q
    fines = [m * F_lone for m in MULTS]
    rows = g1.diagram2(s, g1.G1_BASE, a.populations, fines)
    res = dict(g_lone=g_lone, g_lone_by_train_pop=per_pop, F_lone=F_lone, F_group_paper=g1.FISHERY_F_STAR,
               ratio_lone_to_group=F_lone / g1.FISHERY_F_STAR, by_fine={})
    for mult, F in zip(MULTS, fines):
        sub = [r for r in rows if abs(r["fine"] - F) < 1e-12]
        returns = {k: [r["returns"] for r in sub if r["k"] == k] for k in range(s.n + 1)}
        greedy = {k: [r["defect"] for r in sub if r["k"] == k] for k in range(s.n + 1)}
        rc, rd = g1.schelling_curves(returns, greedy, s.n)
        t = g1.dilemma_test(rc, rd)
        res["by_fine"][f"{mult:g}"] = dict(fine=F, gain_by_m=[x["estimate"] for x in t["D3"]["by_m"]],
                                           gain_ci_by_m=[x["ci"] for x in t["D3"]["by_m"]])
    lone = {k: (v["gain_by_m"][s.n - 1], v["gain_ci_by_m"][s.n - 1]) for k, v in res["by_fine"].items()}
    res["lone_gain_m5"] = lone
    res["holds"] = bool(lone["0.5"][0] > 0 and lone["1"][0] <= 0 and lone["1.5"][1][1] <= 0)
    (out / "g1b_summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(dict(g_lone=g_lone, F_lone=F_lone, ratio=res["ratio_lone_to_group"], lone_gain_m5=lone,
                          holds=res["holds"]), default=float, indent=1))


if __name__ == "__main__":
    main()
