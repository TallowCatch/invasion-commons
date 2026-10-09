"""Post hoc (asked 9 Oct 2026, after all results): does one LLM agent's over-taking spread to the others?

LLM agents never see each other's actions, only their own last three rounds and the stock, so any spillover must run
through the shared stock or the reviewer's shared cut. For gpt-oss and Nemotron, in the cells where over-taking is
common (E0-E8, P0, S0), we regress an agent's over-taking in round t+1 on the number of OTHER LLM agents that over-took
in round t, with game and round fixed effects, the agent's own over-take in round t, and the stock. Then we add the
reviewer's cut in round t+1, and finally use only deliberate over-takes (catch above the agent's own request), which a
smaller allowance cannot create by itself. Intervals: 400 bootstrap resamples of games.
Run:  PYTHONPATH=. python -m experiments.oversight.l2_spillover --run STORE/claude_l2_v1
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from experiments.oversight import run_l2_llm_agents as l2
from fishery_sim import llm_actor as L

THR = l2.OVER_T / L.MAX_CATCH
CELLS = ("E0", "E1", "E2", "E4", "E8", "P0", "S0")


def rows_for(run, model):
    rows, gid = [], 0
    for p in sorted(glob.glob(f"{run}/{model}/episodes/*.json")):
        e = json.load(open(p))
        if e["cell"] not in CELLS:
            continue
        gid += 1
        st, A = e["steps"], e["llm_agents"]
        ov = [[st[t]["taken"][i] > st[t]["allowance"][i] + THR for i in A] for t in range(len(st))]
        for t in range(len(st) - 1):
            for k, i in enumerate(A):
                s1 = st[t + 1]
                rows.append(dict(g=gid, t=t + 1, y=float(ov[t + 1][k]), y2=float(s1["taken"][i] > s1["requests"][i] + 1e-6),
                                 own=float(ov[t][k]), oth=float(sum(ov[t][j] for j in range(len(A)) if j != k)),
                                 stock=s1["stock"] / 100, scale=s1["scale"]))
    return pd.DataFrame(rows)


def fit(D, yname, ctrl, B=400, seed=1):
    cols = [yname, "own", "oth"] + ctrl
    Z = D[cols].copy()
    for c in cols:  # two-way (game, round) demeaning
        Z[c] = D[c] - D.groupby("g")[c].transform("mean") - D.groupby("t")[c].transform("mean") + D[c].mean()
    X, Y = Z[["own", "oth"] + ctrl].to_numpy(), Z[yname].to_numpy()
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    games = D.g.unique()
    idx = {g: np.where(D.g.to_numpy() == g)[0] for g in games}
    dr = [np.linalg.lstsq(X[s], Y[s], rcond=None)[0][1]
          for s in (np.concatenate([idx[g] for g in rng.choice(games, len(games))]) for _ in range(B))]
    return dict(per_other_overtaker=float(b[1]), ci=[float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    out = {}
    for m in ("gpt-oss_120b-cloud", "nemotron-3-super_cloud"):
        D = rows_for(a.run, m)
        raw = {int(k): float(v) for k, v in D[D.own == 0].groupby("oth").y.mean().items()}
        out[m] = dict(agent_rounds=len(D), games=int(D.g.nunique()), raw_next_overtake_if_not_overtaking_by_others=raw,
                      fe_stock=fit(D, "y", ["stock"]), fe_stock_cut=fit(D, "y", ["stock", "scale"]),
                      deliberate_fe_stock_cut=fit(D, "y2", ["stock", "scale"]))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
