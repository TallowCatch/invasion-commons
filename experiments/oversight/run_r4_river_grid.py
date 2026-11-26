"""Experiment R4: River on a grid of settings, under no audits, audits without memory and audits with memory.

Protocol: notes/claude_audit_20261005/studies/R4_river_grid/protocol.md
Game: fishery_sim/two_reagent.py with C1's frozen parameters; the reviewer's target is quality >= 50 (half capacity).

Run:  PYTHONPATH=. python -m experiments.oversight.run_r4_river_grid --profile smoke|full --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.oversight.claude_oversight_common import sha256, write_jsonl_gz
from fishery_sim import two_reagent as TR
from fishery_sim.calibrated_oversight import stable_seed

SEEDS = dict(pop=1_880_000_000, weather=1_881_000_000, reviewer=1_882_000_000, audit=1_883_000_000)
GREEDY = (2, 4)
REGROWTH = (1.0, 0.85)
CONDITIONS = ("trust", "memoryless", "memory")
LIE = 0.5
TARGET = 50.0
B, BOOT_SEED = 4000, 20261019


def profile(name):
    if name == "smoke":
        return dict(contexts=2, horizon=20, reviewer_draws=400)
    if name == "full":
        return dict(contexts=64, horizon=80, reviewer_draws=400)
    raise ValueError(name)


@contextmanager
def river_target(q_line):
    """The reviewer's target and the one-step risk in two_reagent use the module constant Q_UNSAFE at call time."""
    old = TR.Q_UNSAFE
    TR.Q_UNSAFE = float(q_line)
    try:
        yield
    finally:
        TR.Q_UNSAFE = old


def make_population(context, pop_base, n_greedy):
    """2 greedy agents: exactly C1's population (one X and one Y stressed). 4: two X and two Y stressed."""
    pol = TR.make_population(context, pop_base)
    if n_greedy == 2:
        return pol
    R = TR.RANGE_SETS[TR.FROZEN_RANGES]
    rng = np.random.default_rng(pop_base + 7919 * context + 1)
    stressed = sorted(rng.choice(3, 2, replace=False).tolist() + (3 + rng.choice(3, 2, replace=False)).tolist())
    kind = ["stress" if i in stressed else "normal" for i in range(TR.N_AGENTS)]
    low = np.array([rng.uniform(*R[k][0]) for k in kind])
    high = np.array([rng.uniform(*R[k][1]) for k in kind])
    return dict(low=low, high=high, thresholds=pol["thresholds"], stressed=stressed)


def episode(ctx, n_greedy, regrowth_mult, condition, prof, seeds=SEEDS, c1_compat=False):
    """c1_compat: draw audits and reviewer weather exactly as C1 does (for the reproduction gate)."""
    P = TR.frozen_params("comp")
    P = replace(P, r=P.r * regrowth_mult)
    pol = make_population(ctx, seeds["pop"], n_greedy)
    liars = pol["stressed"]
    shocks = TR.weather_shocks(ctx, seeds["weather"], prof["horizon"]) * P.sigma
    q, records, rows, caught_liars = TR.Q_INIT, {}, [], set()
    for t in range(prof["horizon"]):
        req = TR.policy_requests(pol, q)
        reports = req.copy()
        reports[liars] *= (1.0 - LIE)
        audited = None
        if condition != "trust":
            aseed = stable_seed(seeds["audit"], "comp", ctx, t) if c1_compat else stable_seed(seeds["audit"], n_greedy, regrowth_mult, ctx, t)
            audited = int(np.random.default_rng(aseed).integers(0, TR.N_AGENTS))
        believed = TR.memory_belief(reports, records if condition == "memory" else {}, audited, req)
        if audited is not None and reports[audited] < req[audited] - 1e-12:
            caught_liars.add(audited)
            if condition == "memory":
                records.setdefault(audited, []).append(1.0 - reports[audited] / req[audited])
        rseed = (stable_seed(seeds["reviewer"], "closed", "comp", ctx, t) if c1_compat
                 else stable_seed(seeds["reviewer"], n_greedy, regrowth_mult, ctx, t))
        eps = TR.reviewer_draws(P, rseed, prof["reviewer_draws"])
        dec = TR.review(P, q, believed, req, "joint", eps)
        q_next = TR.next_q(P, q, dec["executed"], shocks[t])
        rows.append(dict(t=t, q=q, q_next=q_next, executed=float(np.sum(dec["executed"]))))
        q = q_next
    return dict(context=ctx, greedy=n_greedy, regrowth=regrowth_mult, condition=condition,
                below50=float(np.mean([r["q_next"] < 50 for r in rows])),
                below30=float(np.mean([r["q_next"] < 30 for r in rows])),
                discharge=float(np.sum([r["executed"] for r in rows])), liars=len(liars),
                liars_caught=len(caught_liars & set(liars)), rows=rows)


def analyse(eps, prof):
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    for g in GREEDY:
        for m in REGROWTH:
            sub = {c: {e["context"]: e for e in eps if e["greedy"] == g and e["regrowth"] == m and e["condition"] == c}
                   for c in CONDITIONS}
            ctx = np.array(sorted(sub["trust"]))
            mean = {c: float(np.mean([sub[c][k]["below50"] for k in ctx])) for c in CONDITIONS}
            def diff(a, b):
                d = np.array([sub[a][k]["below50"] - sub[b][k]["below50"] for k in ctx])
                draws = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
                return dict(estimate=float(d.mean()), ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])
            testable = mean["trust"] > 0
            h1, h2 = diff("memory", "memoryless"), diff("memory", "trust")
            out.append(dict(setting=f"{g}g {m}", greedy=g, regrowth=m, testable=testable, below50=mean,
                            below30={c: float(np.mean([sub[c][k]["below30"] for k in ctx])) for c in CONDITIONS},
                            memory_minus_memoryless=h1, memory_minus_trust=h2,
                            H1=bool(testable and h1["ci"][1] < 0), H2=bool(testable and h2["ci"][1] < 0),
                            memoryless_share_of_trust_harm_removed=(
                                float((mean["trust"] - mean["memoryless"]) / mean["trust"]) if testable else None),
                            liars_caught_share={c: float(np.sum([sub[c][k]["liars_caught"] for k in ctx]) /
                                                         np.sum([sub[c][k]["liars"] for k in ctx])) for c in CONDITIONS}))
    supported = all(r["H1"] for r in out if r["testable"])
    return dict(settings=out, supported=supported, profile=prof, target=TARGET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prof = profile(a.profile)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0, eps = time.time(), []
    with river_target(TARGET):
        for g in GREEDY:
            for m in REGROWTH:
                for c in range(prof["contexts"]):
                    for cond in CONDITIONS:
                        eps.append(episode(c, g, m, cond, prof))
                print(f"{g}g {m} {time.time() - t0:.0f}s", flush=True)
    write_jsonl_gz(out / "episodes.jsonl.gz", eps)
    res = analyse(eps, prof)
    (out / "r4_summary.json").write_text(json.dumps(res, indent=1, default=float))
    manifest = dict(experiment="R4", profile=a.profile, seeds=SEEDS, lie=LIE, target=TARGET, seconds=time.time() - t0,
                    frozen=TR.FROZEN.as_dict(), files={"episodes.jsonl.gz": sha256(out / "episodes.jsonl.gz")},
                    source={p: sha256(Path(p)) for p in ["experiments/oversight/run_r4_river_grid.py", "fishery_sim/two_reagent.py"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    print(json.dumps({r["setting"]: dict(testable=r["testable"], below50={k: round(v, 3) for k, v in r["below50"].items()},
                                         H1=r["H1"], H2=r["H2"]) for r in res["settings"]}, indent=1))
    print("supported:", res["supported"])


if __name__ == "__main__":
    main()
