"""Experiment R3: does deterrence start at the break-even in every Fishery setting? (Claude audit, October 2026)

Protocol: notes/claude_audit_20261005/studies/R3_breakeven_confirmation/protocol.md
Reuses R2's non-compliance episode (S2/S3 cheating model, MSY target, Bernoulli audits, flat fine) per setting.
The predicted threshold g* = max over d of G(d) / n(d) is computed on the training contexts before any search.
Run:  PYTHONPATH=. python -m experiments.oversight.run_r3_breakeven --profile smoke|full --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.oversight import run_r2_robustness_reviewer_model as r2
from experiments.oversight.claude_oversight_common import sha256, write_jsonl_gz

SEEDS = dict(fishery_pop=1_700_000_000, audit=1_745_000_000, detect=1_746_000_000, train_fishery_pop=1_750_000_000)
CELLS = [r2.Cell("fishery", n, r) for n in (2, 4) for r in (0.5, 0.7, 0.9)]
Q = r2.DETER_Q  # 1/6
LEVELS = (0.25, 0.5, 0.75, 1.0)
N_GRID, N_EXT = 10, 20  # e_j = j * 0.2 g*, j = 0..10, extended to 20 if no grid value deters


def predict(cell, P, seeds=SEEDS):
    """g* = max over d with G(d) > 0 of G(d) / n(d), on the training contexts with no checks."""
    train = range(P["train_contexts"])
    comply = [r2.nc_episode(cell, c, "allow", 0.0, 0.0, P, seeds, True) for c in train]
    out = {}
    for d in LEVELS:
        eps = [r2.nc_episode(cell, c, "allow", d, 0.0, P, seeds, True) for c in train]
        G = float(np.mean([(a["cheater_payoff"] - b["cheater_payoff"]) / a["n_cheaters"] for a, b in zip(eps, comply)]))
        n = float(np.mean([a["cheater_over_steps"] for a in eps]))
        out[str(d)] = dict(G=G, n_over=n, g=G / n if n > 0 else None, payoff=float(np.mean([a["cheater_payoff"] for a in eps])))
    cands = [v["g"] for v in out.values() if v["G"] > 0 and v["g"] is not None]
    g_star = max(cands) if cands else 0.0
    comply_pay = float(np.mean([b["cheater_payoff"] for b in comply]))
    d_allow = max([0.0] + list(LEVELS), key=lambda d: (comply_pay if d == 0 else out[str(d)]["payoff"], -d))
    g_r2 = out[str(d_allow)]["g"] if d_allow > 0 else None  # R2's post hoc definition (preferred no-check level)
    return dict(by_level=out, g_star=g_star, d_allow=d_allow, g_r2_definition=g_r2)


def run_cell(cell, P, seeds=SEEDS):
    test = range(P["contexts"])
    pred = predict(cell, P, seeds)
    info = dict(**cell.tag(), **pred, testable=pred["g_star"] > 0, q=Q)
    eps = [{**e, "arm": "comply", "e": None} for e in (r2.nc_episode(cell, c, "allow", 0.0, 0.0, P, seeds) for c in test)]
    if pred["d_allow"] > 0:
        eps += [{**e, "arm": "no_check", "e": None}
                for e in (r2.nc_episode(cell, c, "allow", pred["d_allow"], 0.0, P, seeds) for c in test)]
    if not info["testable"]:
        info.update(searches=[], e_star=None, extended=False)
        return info, eps
    step = 0.2 * pred["g_star"]
    searches, e_star = [], None
    for j in range(N_EXT + 1):
        if j > N_GRID and e_star is not None:
            break
        ev = step * j
        best, scores = r2.search_d(cell, ev / Q, P, seeds)
        searches.append(dict(j=j, e=ev, fine=ev / Q, d_star=best, train_scores=scores, extended=j > N_GRID))
        eps += [{**r2.nc_episode(cell, c, "bern", best, ev / Q, P, seeds, q=Q), "arm": "adaptive", "e": ev, "j": j}
                for c in test]
        if best == 0 and e_star is None:
            e_star = ev
            if j >= N_GRID:
                break
    info.update(searches=searches, e_star=e_star, extended=any(s["extended"] for s in searches))
    return info, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = r2.profile(a.profile)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0, infos, episodes = time.time(), [], []
    for cell in CELLS:
        info, eps = run_cell(cell, P)
        infos.append(info)
        episodes += [{**e, "cell": cell.name} for e in eps]
        print(f"{cell.name} g*={info['g_star']:.4f} e*={info['e_star']} {time.time() - t0:.0f}s", flush=True)
    write_jsonl_gz(out / "episodes.jsonl.gz", episodes)
    (out / "cells.json").write_text(json.dumps(infos, indent=1, default=float))
    manifest = dict(experiment="R3", profile=a.profile, params=P, seeds=SEEDS, q=Q, seconds=time.time() - t0,
                    n_episodes=len(episodes), files={f: sha256(out / f) for f in ("episodes.jsonl.gz", "cells.json")},
                    source={p: sha256(Path(p)) for p in ["experiments/oversight/run_r2_robustness_reviewer_model.py",
                            "experiments/oversight/run_s3_threshold_timing_memory.py",
                            "experiments/oversight/run_r3_breakeven.py", "fishery_sim/reviewer_models.py"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(dict(n_episodes=len(episodes), seconds=manifest["seconds"])))


if __name__ == "__main__":
    main()
