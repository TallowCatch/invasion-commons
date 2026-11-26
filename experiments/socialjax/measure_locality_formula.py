"""Task 2 of notes/socialjax/locality/protocol.md: the catch formulas P_C and P_0 against simulation.

  .venv-socialjax/bin/python -m experiments.socialjax.measure_locality_formula --out results/locality/formula.json

Part 1 (abstract): exactly the formula's model, NumPy, 200,000 violations per cell. A violation is caught at the
first check made while the agent is in the region where it violated (chance q per step there); a check there that
misses (chance alpha) loses it; it is lost when it leaves the w-step window. Crossing patterns: drift (never back),
two regions, ring of four.
Part 2 (the real overseer code): LocalRecordOverseer with fixed k_m = K/R random checks per region, on the corridor
game (overseer/socialjax/corridor.py) with N = 8 crossing bots, K = 2, R = 2 or 4, and R = 1 (central) as the
reference. The catch share is detected / (violations - those still pending at the end).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from overseer.socialjax import locality as L
from overseer.socialjax.corridor import CorridorRules, crossing_policy
from overseer.socialjax.decentralised import DecConfig
from overseer.socialjax.local_records import LocalRecordOverseer

QS = (0.05, 1 / 7, 2 / 7)
CHIS = (0.0, 0.01, 0.05, 0.2, 0.5)
WS = (1, 2, 5, 10, 20, 50)
ALPHAS = (0.0, 0.5)
PATTERNS = ("drift", "two", "ring4")


def simulate_abstract(q, chi, w, alpha, pattern, n=200_000, rng=None):
    rng = rng or np.random.default_rng(0)
    S = {"drift": None, "two": 2, "ring4": 4}[pattern]
    pos = np.zeros(n, np.int64)  # 0 = home; drift: number of crossings so far
    alive = np.ones(n, bool)
    caught = np.zeros(n, bool)
    for j in range(w):
        if j > 0:
            cross = rng.random(n) < chi
            if pattern == "drift":
                pos = pos + cross
            elif pattern == "two":
                pos = np.where(cross, 1 - pos, pos)
            else:
                step = np.where(rng.random(n) < 0.5, 1, -1)
                pos = np.where(cross, (pos + step) % S, pos)
        check = alive & (pos == 0) & (rng.random(n) < q)
        hit = check & (rng.random(n) >= alpha)
        caught |= hit
        alive &= ~check
    return float(caught.mean())


def part1():
    rows = []
    rng = np.random.default_rng(1)
    for pattern in PATTERNS:
        for q in QS:
            for chi in CHIS:
                for w in WS:
                    for a in ALPHAS:
                        sim = simulate_abstract(q, chi, w, a, pattern, rng=rng)
                        rows.append(dict(pattern=pattern, q=q, chi=chi, w=w, alpha=a, sim=sim,
                                         p_c=L.p_central(q, w, a), p_0=L.p_independent(q, chi, w, a),
                                         p_markov=L.p0_markov(q, chi, w, a, pattern=pattern)))
    return rows


def part2(seeds=4, steps=5000, violate=0.05, n_agents=8, K=2):
    rows = []
    for R in (1, 2, 4):
        rules = CorridorRules(n_agents=n_agents, regions=max(R, 2) if R > 1 else 2)
        for w in WS:
            for chi in CHIS:
                t0 = time.time()
                cfg = DecConfig(k=K, chooser="random", regions=R, window=w)
                ov = LocalRecordOverseer(rules, cfg, records="local") if R > 1 else \
                    LocalRecordOverseer(rules, cfg, records="handover")
                act = crossing_policy(chi=chi, violate=violate, pattern="ring")
                det = viol = left = checks = 0
                occ_q = []
                for s in range(seeds):
                    t = L.collect_trace(ov, act, jax.random.PRNGKey(s), steps, keep_grid=False)
                    det += int(t["detected_count"].sum())
                    viol += int(t["violation"].sum())
                    left += t["left_in_window"]
                    checks += int(t["checked"].sum())
                    # each violator's chance of being checked in the step it violated (its region's occupancy)
                    if R > 1:
                        reg = t["region"]
                        occ = np.stack([(reg == r).sum(1) for r in range(R)], 1)  # (T, R)
                        k_r = K / R
                        q_here = np.minimum(1.0, k_r / np.maximum(occ[np.arange(len(reg))[:, None], reg], 1))
                        occ_q.append(q_here[t["violation"]])
                share = det / max(1, viol - left)
                q_nom = K / n_agents
                q_eff = checks / (seeds * steps * n_agents)
                pattern = "two" if R == 2 else "ring4"
                row = dict(R=R, w=w, chi=chi, sim=share, violations=viol, q_nominal=q_nom, q_measured=q_eff,
                           p_c=L.p_central(q_nom, w), p_0=L.p_independent(q_nom, chi, w) if R > 1 else None,
                           p_0_qeff=L.p_independent(q_eff, chi, w) if R > 1 else None,
                           p_markov=L.p0_markov(q_nom, chi, w, pattern=pattern) if R > 1 else None,
                           p_markov_qeff=L.p0_markov(q_eff, chi, w, pattern=pattern) if R > 1 else None,
                           q_at_violation=float(np.concatenate(occ_q).mean()) if occ_q else q_nom)
                rows.append(row)
                print(f"R={R} w={w} chi={chi}: sim {share:.3f}  P_C {row['p_c']:.3f}  P_0 {row['p_0']}  "
                      f"markov {row['p_markov']}  q_eff {q_eff:.3f} ({time.time() - t0:.0f} s)", flush=True)
            jax.clear_caches()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/locality/formula.json")
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    abstract = part1()
    print(f"abstract: {len(abstract)} cells, {time.time() - t0:.0f} s", flush=True)
    out.write_text(json.dumps(dict(abstract=abstract), indent=1))
    code = part2(seeds=a.seeds)
    out.write_text(json.dumps(dict(abstract=abstract, code=code), indent=1))


if __name__ == "__main__":
    main()
