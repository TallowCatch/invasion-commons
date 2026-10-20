"""Analysis for L6 (protocol: notes/claude_audit_20261005/studies/L6_llm_forest_spillover/protocol.md).
Written before any L6 game was played.

  PYTHONPATH=. python3 -m experiments.oversight.analyze_l6 --run results/runs/claude_l6_v1 --model gpt-oss:120b-cloud --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.oversight import analyze_l4 as A
from experiments.oversight import run_l6_llm_forest as F

B, BOOT_SEED = 4000, 20261019


def by_distance(e, d):
    return [i for i in F.LLM_POS if F.DISTANCE[i] == d]


def health_at(e, d):
    return float(np.mean([s["health_after"][i] for s in e["steps"] for i in by_distance(e, d)]))


def test(x, rng, want_positive=True):
    x = np.asarray(x, float)
    dr = A.boot_draws(x, rng)
    ci = [float(x.mean()), float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))]
    p = float(np.mean(dr <= 0)) if want_positive else float(np.mean(dr >= 0))
    sd = float(x.std(ddof=1)) if len(x) > 1 else None
    return dict(estimate=ci, p_one_sided=p, sd=sd, mde_80=(2.8 * sd / np.sqrt(len(x)) if sd else None), n=len(x))


def analyse(run_dir, model):
    d = Path(run_dir) / model.replace(":", "_").replace(".", "_").replace("/", "__") / "episodes"
    eps = {}
    for p in sorted(d.glob("*.json")):
        cell, ctx = p.stem.split("__")
        eps.setdefault(cell, {})[int(ctx)] = json.loads(p.read_text())
    ctx = sorted(set.intersection(*(set(v) for v in eps.values())))
    rng = np.random.default_rng(BOOT_SEED)
    rate = {c: {k: {dd: F.overtake_rate(eps[c][k], by_distance(None, dd)) for dd in (1, 2, 3)} for k in ctx} for c in eps}
    cells = {c: dict(overtake_by_distance={dd: A.boot_mean([rate[c][k][dd] for k in ctx], rng) for dd in (1, 2, 3)},
                     health_by_distance={dd: A.boot_mean([health_at(eps[c][k], dd) for k in ctx], rng) for dd in (1, 2, 3)},
                     unsafe_share=A.boot_mean([np.mean([s["global_unsafe"] for s in eps[c][k]["steps"]]) for k in ctx], rng),
                     collapsed=int(sum(eps[c][k]["collapsed"] for k in ctx)))
             for c in eps}
    h = {}
    h["H1"] = test([np.mean([health_at(eps[f"D_{v}"][k], 1) - health_at(eps[f"C_{v}"][k], 1) for v in "HV"]) for k in ctx],
                   rng, want_positive=False)
    h["H2"] = test([np.mean([rate[f"D_{v}"][k][1] - rate[f"C_{v}"][k][1] for v in "HV"]) for k in ctx], rng)
    h["H3"] = test([(rate["D_V"][k][1] - rate["C_V"][k][1]) - (rate["D_H"][k][1] - rate["C_H"][k][1]) for k in ctx], rng)
    h["H4"] = test([np.mean([(rate[f"D_{v}"][k][1] - rate[f"C_{v}"][k][1]) - (rate[f"D_{v}"][k][3] - rate[f"C_{v}"][k][3])
                             for v in "HV"]) for k in ctx], rng)
    adj = A.holm([h[k]["p_one_sided"] for k in ("H1", "H2", "H3")])
    for k, q in zip(("H1", "H2", "H3"), adj):
        h[k]["p_holm"] = q
    h["H1"]["holds"] = h["H1"]["estimate"][2] < 0
    for k in ("H2", "H3"):
        h[k]["holds"] = h[k]["estimate"][1] > 0
    return dict(model=model, contexts=ctx, cells=cells, hypotheses=h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=F.OUT)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    res = analyse(a.run, a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"l6_summary_{a.model.replace(':', '_')}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["hypotheses"], indent=1))


if __name__ == "__main__":
    main()
