"""Study G2: one-factor-at-a-time parameter sweeps with programmed agents (Claude audit, October 2026).

Protocol (frozen, with Amendment 1): notes/claude_audit_20261005/studies/G1_schelling_and_sweeps/protocol.md

At every grid point:
  1. Schelling diagram 1 (no reviewer, no audits), k = 0..N greedy, for D1-D3;
  2. the closed-form prediction g* (R3 rule, 8 training populations, reviewer on, no audits), then the S2/S3
     search over d in {0, 1/4, 1/2, 3/4, 1} on the same 8 training populations at F = j * 0.1 * F*_pt,
     j = 0..30 (q = 1/6), and 64 test populations at each chosen d;
  3. harm with no reviewer and no audits, and with audits at F = 2 F*_pt (j = 20).
Game code is that of run_g1_schelling (itself importing the existing dynamics).

Run:  PYTHONPATH=. python3 -m experiments.oversight.run_g2_sweeps --profile smoke|full --out DIR [--jobs 8]
      (reads the G1 F* for Forest and River from --g1 DIR/fstar.json; used only where g*_pt <= 0)
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.oversight import run_g1_schelling as g1
from experiments.oversight.claude_oversight_common import sha256, write_jsonl_gz

G2_BASE = 26_101_420
J_MAX, J_STEP, J_AUDIT = 30, 0.1, 20
FACTORS = dict(
    r=("fishery", (0.4, 0.55, 0.7, 0.85, 1.0)),
    N=("fishery", (4, 6, 8, 10)),
    T=("fishery", (20, 40, 80)),
    externality=("forest", (0.0, 0.11, 0.22, 0.33, 0.44)),
    kappa=("river", (0.5, 1.0, 2.0)),
)


def fishery_setting_for_n(n):
    """K = 100 N / 6 (Setting.capacity); stock levels scale with K (Amendment 1, item 7)."""
    return g1.Setting("fishery", n=int(n))


def setting_for(factor, value, horizon=80):
    game = FACTORS[factor][0]
    if factor == "r":
        return g1.Setting(game, r=float(value), horizon=horizon)
    if factor == "N":
        s = fishery_setting_for_n(value)
        return g1.Setting(s.game, n=s.n, horizon=horizon)
    if factor == "T":
        return g1.Setting(game, horizon=int(value))
    if factor == "externality":
        return g1.Setting(game, externality=float(value), horizon=horizon)
    if factor == "kappa":
        return g1.Setting(game, kappa_mult=float(value), horizon=horizon)
    raise ValueError(factor)


def grid(horizon=80, smoke=False):
    pts = []
    for f, (_, vals) in FACTORS.items():
        for v in (vals[:2] if smoke else vals):
            if smoke and f == "T":
                v = min(v, horizon)
            pts.append((f, v, setting_for(f, v, horizon)))
    return pts


def search(setting, fine, train, cache):
    """S3 search_nc: common level maximising mean cheater net payoff on training populations; ties -> smaller d."""
    scores, best, best_score = {}, None, -np.inf
    for d in g1.SEARCH_D:
        vals = []
        for c, (pop, gm, sd) in enumerate(train):
            key = (c, d, fine if d > 0 else 0.0)  # with d = 0 nobody over-takes, so the fine is irrelevant
            if key not in cache:
                cache[key] = g1.cheater_net(g1.episode(setting, pop, gm, gm, d, fine, True, True, sd), np.flatnonzero(gm))
            vals.append(cache[key])
        sc = float(np.mean(vals))
        scores[str(d)] = sc
        if best is None or sc > best_score + 1e-9:
            best, best_score = d, sc
    return best, scores


def run_point(args):
    setting, P, fallback_F = args
    t0 = time.time()
    rows = g1.diagram1(setting, G2_BASE, P["populations"])
    pred = g1.predict_gstar(setting, G2_BASE, P["train"])
    g_star = pred["g_star"]
    testable = g_star > 0
    train = []
    for c in range(P["train"]):
        pop = g1.population(setting, G2_BASE, "train", c)
        train.append((pop, g1.paper_greedy(setting, pop), g1.ctx_seeds(G2_BASE, "train", setting, c)))
    test = []
    for c in range(P["populations"]):
        pop = g1.population(setting, G2_BASE, "test", c)
        test.append((pop, g1.paper_greedy(setting, pop), g1.ctx_seeds(G2_BASE, "test", setting, c)))
    # harm and payoffs with no reviewer and no audits; complying with the reviewer (reference for the gain)
    out = []
    for c, (pop, gm, sd) in enumerate(test):
        none = np.zeros(setting.n, bool)
        e = g1.episode(setting, pop, gm, none, 0.0, 0.0, False, False, sd)
        out.append(dict(arm="none", context=c, harm=e["harm"], failed=e["failed"], total=float(e["pay"].sum())))
        e = g1.episode(setting, pop, gm, gm, 0.0, 0.0, True, False, sd)
        out.append(dict(arm="comply", context=c, harm=e["harm"], failed=e["failed"], total=float(e["pay"].sum()),
                        cheater_net=g1.cheater_net(e, np.flatnonzero(gm)), n_cheaters=int(gm.sum())))
    F_star_pt = g_star / g1.Q if testable else fallback_F
    js = list(range(J_MAX + 1)) if testable else [J_AUDIT]
    searches, cache = [], {}
    for j in js:
        fine = j * J_STEP * F_star_pt
        d, scores = search(setting, fine, train, cache)
        searches.append(dict(j=j, e=fine * g1.Q, fine=fine, d_star=d, train_scores=scores))
        for c, (pop, gm, sd) in enumerate(test):
            e = g1.episode(setting, pop, gm, gm, d, fine, True, True, sd)
            out.append(dict(arm="audit", j=j, e=fine * g1.Q, fine=fine, d=d, context=c, harm=e["harm"], failed=e["failed"],
                            total=float(e["pay"].sum()), cheater_net=g1.cheater_net(e, np.flatnonzero(gm)),
                            n_cheaters=int(gm.sum())))
    e_obs = next((s["e"] for s in searches if s["d_star"] == 0), None) if testable else None
    first = next((i for i, s in enumerate(searches) if s["d_star"] == 0), None)
    stays = bool(first is not None and all(s["d_star"] == 0 for s in searches[first:]))
    info = dict(setting=setting.tag(), prediction=pred, g_star=g_star, testable=testable, F_star_pt=F_star_pt,
                fine_source="g*_pt / q" if testable else "fallback: paper-setting F* of the game",
                e_obs=e_obs, stays_zero_above=stays, searches=searches, seconds=time.time() - t0)
    return setting, info, rows, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--g1", required=True, help="G1 output directory (for fstar.json)")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    P = g1.profile(a.profile)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    fstar = json.loads((Path(a.g1) / "fstar.json").read_text())
    t0 = time.time()
    pts = grid(P["horizon"], smoke=a.profile == "smoke")
    unique = list(dict.fromkeys(s for _, _, s in pts))  # the paper point recurs under r, N and T: run it once
    jobs = [(s, P, 2 * fstar[s.game]["F_star"]) for s in unique]
    results = {}
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for s, info, rows, eps in ex.map(run_point, jobs):
                results[s] = (info, rows, eps)
                print(f"[{time.time() - t0:6.0f}s] {s} g*={info['g_star']:.4f} e_obs={info['e_obs']}", flush=True)
    else:
        for job in jobs:
            s, info, rows, eps = run_point(job)
            results[s] = (info, rows, eps)
            print(f"[{time.time() - t0:6.0f}s] {s} g*={info['g_star']:.4f} e_obs={info['e_obs']}", flush=True)
    points, d1rows, eprows = [], [], []
    for f, v, s in pts:
        info, rows, eps = results[s]
        pid = f"{f}={v}"
        points.append(dict(point=pid, factor=f, value=v, **info))
        d1rows += [{**r, "point": pid} for r in rows]
        eprows += [{**r, "point": pid} for r in eps]
    write_jsonl_gz(out / "diagram1.jsonl.gz", d1rows)
    write_jsonl_gz(out / "deterrence_harm.jsonl.gz", eprows)
    (out / "points.json").write_text(json.dumps(points, indent=1, default=float))
    manifest = dict(experiment="G2", profile=a.profile, params=P, G2_BASE=G2_BASE, q=g1.Q, j_max=J_MAX, j_step=J_STEP,
                    j_audit=J_AUDIT, factors={k: [g, list(v)] for k, (g, v) in FACTORS.items()},
                    g1_fstar=fstar, n_points=len(points), n_unique=len(unique),
                    files={f: sha256(out / f) for f in ("diagram1.jsonl.gz", "deterrence_harm.jsonl.gz", "points.json")},
                    source={p: sha256(Path(p)) for p in ["experiments/oversight/run_g2_sweeps.py",
                                                         "experiments/oversight/run_g1_schelling.py"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    (out / "timing.json").write_text(json.dumps(dict(seconds=time.time() - t0)))
    print(json.dumps(dict(points=len(points), unique=len(unique), seconds=round(time.time() - t0, 1))))


if __name__ == "__main__":
    main()
