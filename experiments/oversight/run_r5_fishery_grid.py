"""Experiment R5: the Fishery memory test on the 2x2 grid used for Forest (R2) and River (R4).

Protocol: notes/claude_audit_20261005/studies/R5_fishery_grid/protocol.md
Reuses R2's memory test unchanged (run_r2_robustness_reviewer_model.memory_episode), with new seed bases.
Run:  PYTHONPATH=. python -m experiments.oversight.run_r5_fishery_grid --profile smoke|full --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.oversight import run_r2_robustness_reviewer_model as r2
from experiments.oversight.claude_oversight_common import sha256, write_jsonl_gz

BASE_R = 0.7
GREEDY = (2, 4)
MULT = (1.0, 0.85)
SEEDS = {**r2.SEEDS, "fishery_pop": 1_890_000_000, "reviewer": 1_891_000_000, "reference": 1_892_000_000,
         "audit": 1_893_000_000}
B, BOOT_SEED = 4000, 20261019


def cell(n_greedy, mult):
    return r2.Cell("fishery", n_greedy, round(BASE_R * mult, 6))


def episode(n_greedy, mult, ctx, mode, P, seeds=SEEDS):
    e = r2.memory_episode(cell(n_greedy, mult), ctx, mode, "fixed", P, seeds)
    return dict(greedy=n_greedy, mult=mult, context=ctx, mode=mode, exec_risky=int(e["exec_risky"]),
                scored_steps=int(e["scored_steps"]), failure=int(e["failure"]))


def analyse(eps):
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    for g in GREEDY:
        for m in MULT:
            by = {md: {e["context"]: e for e in eps if e["greedy"] == g and e["mult"] == m and e["mode"] == md}
                  for md in r2.MEMORY_MODES}
            ctx = sorted(by["trust"])
            pooled = {md: sum(by[md][c]["exec_risky"] for c in ctx) / max(sum(by[md][c]["scored_steps"] for c in ctx), 1)
                      for md in r2.MEMORY_MODES}
            share = {md: np.array([by[md][c]["exec_risky"] / max(by[md][c]["scored_steps"], 1) for c in ctx])
                     for md in r2.MEMORY_MODES}
            def diff(a, b_):
                d = share[a] - share[b_]
                draws = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
                return dict(estimate=float(d.mean()), ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])
            testable = pooled["trust"] > 0
            h1, h2 = diff("memory", "memoryless"), diff("memory", "trust")
            out.append(dict(setting=f"{g}g {m}", greedy=g, mult=m, r=round(BASE_R * m, 6), testable=bool(testable),
                            target_breaking_rate=pooled, memory_minus_memoryless=h1, memory_minus_trust=h2,
                            H1=bool(testable and h1["ci"][1] < 0), H2=bool(testable and h2["ci"][1] < 0),
                            memoryless_share_of_trust_harm_removed=(
                                float((pooled["trust"] - pooled["memoryless"]) / pooled["trust"]) if testable else None)))
    return dict(settings=out, supported=all(r["H1"] for r in out if r["testable"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = r2.profile(a.profile)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0, eps = time.time(), []
    for g in GREEDY:
        for m in MULT:
            for c in range(P["contexts"]):
                for md in r2.MEMORY_MODES:
                    eps.append(episode(g, m, c, md, P))
            print(f"{g}g {m} {time.time() - t0:.0f}s", flush=True)
    write_jsonl_gz(out / "episodes.jsonl.gz", eps)
    res = analyse(eps)
    (out / "r5_summary.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "manifest.json").write_text(json.dumps(dict(experiment="R5", profile=a.profile, seeds=SEEDS, base_r=BASE_R,
        seconds=time.time() - t0, files={"episodes.jsonl.gz": sha256(out / "episodes.jsonl.gz")},
        source={p: sha256(Path(p)) for p in ["experiments/oversight/run_r5_fishery_grid.py",
                                              "experiments/oversight/run_r2_robustness_reviewer_model.py"]}), indent=1))
    print(json.dumps({r["setting"]: dict(testable=r["testable"], rate={k: round(v, 3) for k, v in r["target_breaking_rate"].items()},
                                         H1=r["H1"], H2=r["H2"]) for r in res["settings"]}, indent=1))
    print("supported:", res["supported"])


if __name__ == "__main__":
    main()
