"""Experiment C1: a third game where harm comes from a combination of actions (Claude audit, October 2026).

Protocol: notes/claude_audit_20261005/studies/C1_compositional_harm/protocol.md
Game and reviewers: fishery_sim/two_reagent.py

Run:
  PYTHONPATH=. python -m experiments.oversight.run_c1_compositional_harm --stage calibrate --out DIR
  PYTHONPATH=. python -m experiments.oversight.run_c1_compositional_harm --stage run --profile smoke|full --out DIR

Stage `calibrate` uses only the 8 calibration contexts (population base 1,490,000,000) and no reviewer.
Stage `run` refuses to start unless Amendment 1 (calibration) is in the protocol and the parameters are frozen.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from experiments.oversight.claude_oversight_common import sha256, write_jsonl_gz
from fishery_sim import two_reagent as TR
from fishery_sim.calibrated_oversight import stable_seed

PROTOCOL = Path("notes/claude_audit_20261005/studies/C1_compositional_harm/protocol.md")
SEEDS = dict(pop=1_400_000_000, weather=1_410_000_000, reviewer=1_420_000_000,
             reference=1_430_000_000, audit=1_440_000_000)
CALIB_SEEDS = dict(pop=1_490_000_000, weather=1_495_000_000, contexts=8, horizon=80)
CALIB_GRID = dict(ranges=tuple(TR.RANGE_SETS), kappa=(16, 20, 24, 28, 32), r=(0.7, 0.85, 1.0, 1.2), sigma=(3, 5, 7))
PART_A_ARMS = ("none",) + TR.REVIEWERS
PART_B_ARMS = ("trust", "random", "targeted")
LIE = 0.5  # stress agents report (1 - LIE) * p


def profile(name):
    if name == "smoke":
        return dict(contexts=2, horizon=20, closed_draws=2000, open_draws=4000, reviewer_draws=400)
    if name == "full":
        return dict(contexts=64, horizon=80, closed_draws=2000, open_draws=4000, reviewer_draws=400)
    raise ValueError(name)


# ------------------------------------------------------------------ calibration
def calibration_point(ranges, kappa, r, sigma):
    out = {}
    for game in TR.GAMES:
        P = TR.make_params(game, kappa, r, sigma)
        risky = pre = pre_risky = n = 0
        for c in range(CALIB_SEEDS["contexts"]):
            pol = TR.make_population(c, CALIB_SEEDS["pop"], ranges)
            w = TR.weather_shocks(c, CALIB_SEEDS["weather"], CALIB_SEEDS["horizon"]) * sigma
            q = TR.Q_INIT
            for t in range(CALIB_SEEDS["horizon"]):
                req = TR.policy_requests(pol, q)
                rk = TR.exact_risk(P, q, req) > TR.TOL
                risky += rk
                n += 1
                if q >= TR.Q_UNSAFE:
                    pre += 1
                    pre_risky += rk
                q = TR.next_q(P, q, req, w[t])
        out[game] = dict(risky_share=risky / n, presafe_share=pre / n, risky_share_presafe=pre_risky / max(pre, 1))
    P = TR.make_params("comp", kappa, r, sigma)
    return dict(ranges=ranges, kappa=kappa, lam=P.lam, lam_add=P.lam_add, r=r, sigma=sigma,
                safe_total_q80=TR.total_for_damage(P, TR.exact_headroom(r, sigma, TR.Q_INIT)), **out)


def feasible(pt):
    return all(0.30 <= pt[g]["risky_share"] <= 0.60 and pt[g]["presafe_share"] >= 0.5 for g in TR.GAMES)


def objective(pt):
    return abs(pt["comp"]["risky_share"] - 0.45) + abs(pt["add"]["risky_share"] - pt["comp"]["risky_share"])


def calibrate(out):
    t0 = time.time()
    pts = []
    for ranges, kappa, r, sigma in itertools.product(*CALIB_GRID.values()):
        pt = calibration_point(ranges, kappa, r, sigma)
        pt.update(feasible=feasible(pt), objective=objective(pt))
        pts.append(pt)
    ok = [p for p in pts if p["feasible"]]
    chosen = min(ok, key=lambda p: (p["objective"], p["kappa"], p["r"], p["sigma"])) if ok else None
    if chosen:  # how compositional is harm at the mean high request of the chosen range set?
        R = TR.RANGE_SETS[chosen["ranges"]]
        per_type = 2 * np.mean(R["normal"][1]) + np.mean(R["stress"][1])
        comp = chosen["kappa"] * per_type ** 2
        chosen["compositional_fraction_at_mean_high"] = comp / (comp + chosen["lam"] * 2 * per_type)
        per_type_low = 2 * np.mean(R["normal"][0]) + np.mean(R["stress"][0])
        comp = chosen["kappa"] * per_type_low ** 2
        chosen["compositional_fraction_at_mean_low"] = comp / (comp + chosen["lam"] * 2 * per_type_low)
    rule = ("feasible: no-reviewer risky share (exact one-step risk > 5%, all steps) in [0.30, 0.60] in both games "
            "and share of steps starting from Q >= 30 at least 0.5 in both games; choose the feasible point minimising "
            "|risky_comp - 0.45| + |risky_add - risky_comp|; ties to smaller kappa, r, sigma. lam = 0.05 kappa; "
            "lam_add from the Q = 80 typical-mix matching rule (analytic 95% normal quantile).")
    res = dict(grid=CALIB_GRID, seeds=CALIB_SEEDS, rule=rule, n_points=len(pts), n_feasible=len(ok), chosen=chosen,
               points=pts)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"calibration: {len(pts)} points, {len(ok)} feasible, {time.time() - t0:.0f}s")
    print(json.dumps(chosen, indent=1, default=float))


# ------------------------------------------------------------------ episodes
def episode(game, ctx, part, arm, prof):
    Par = TR.frozen_params(game)
    pol = TR.make_population(ctx, SEEDS["pop"])
    liars = pol["stressed"]
    shocks = TR.weather_shocks(ctx, SEEDS["weather"], prof["horizon"]) * Par.sigma
    q, records, rows = TR.Q_INIT, {}, []
    for t in range(prof["horizon"]):
        req = TR.policy_requests(pol, q)
        reports, audited = req.copy(), None
        if part == "A":
            reviewer, believed = arm, req.copy()
        else:
            reviewer = "joint"
            reports[liars] *= (1.0 - LIE)
            if arm == "random":
                audited = int(np.random.default_rng(stable_seed(SEEDS["audit"], game, ctx, t)).integers(0, TR.N_AGENTS))
            elif arm == "targeted":
                audited = TR.targeted_agent(Par, TR.memory_belief(reports, records))
            believed = TR.memory_belief(reports, records, audited, req)
            if audited is not None and reports[audited] < req[audited] - 1e-12:
                records.setdefault(audited, []).append(1.0 - reports[audited] / req[audited])
        eps = TR.reviewer_draws(Par, stable_seed(SEEDS["reviewer"], "closed", game, ctx, t), prof["reviewer_draws"])
        dec = TR.review(Par, q, believed, req, reviewer, eps)
        ref = TR.reference(Par, q, req, stable_seed(SEEDS["reference"], game, q, req.tolist()), prof["closed_draws"])
        q_next = TR.next_q(Par, q, dec["executed"], shocks[t])
        rows.append(dict(game=game, part=part, arm=arm, reviewer=reviewer, context=ctx, step=t, q=q,
                         pre_safe=int(q >= TR.Q_UNSAFE), requests=req.tolist(), reports=reports.tolist(),
                         believed=believed.tolist(), audited=audited,
                         caught=int(audited is not None and reports[audited] < req[audited] - 1e-12),
                         executed=dec["executed"].tolist(), scale=dec["scale"], approved=dec["approved"],
                         cap=dec.get("cap"), est_risk=dec["est_risk"], label=ref["label"], ref_risk=ref["risk"],
                         exact_risk_req=TR.exact_risk(Par, q, req), exact_risk_exec=TR.exact_risk(Par, q, dec["executed"]),
                         q_next=q_next, unsafe_next=int(q_next < TR.Q_UNSAFE)))
        q = q_next
    return summarize(game, ctx, part, arm, rows, liars), rows


def score(rows):
    """R1 scoring over steps that start from a safe state (Q >= 30)."""
    s = [r for r in rows if r["pre_safe"]]
    safe = [r for r in s if r["label"] == "safe"]
    risky = [r for r in s if r["label"] == "risky"]
    kept = [sum(r["executed"]) / sum(r["requests"]) if sum(r["requests"]) > 0 else 1.0 for r in safe]
    return dict(scored_steps=len(s), req_safe=len(safe), req_risky=len(risky),
                req_unresolved=len(s) - len(safe) - len(risky),
                risky_approved=sum(r["approved"] for r in risky), safe_cut=sum(1 - r["approved"] for r in safe),
                safe_kept_sum=float(sum(kept)), exec_risky=sum(int(r["exact_risk_exec"] > TR.TOL) for r in s))


def summarize(game, ctx, part, arm, rows, liars):
    pay = np.array([r["executed"] for r in rows])
    honest = [i for i in range(TR.N_AGENTS) if i not in liars]
    return dict(game=game, part=part, arm=arm, context=ctx, steps=len(rows), stressed=liars,
                total_payoff=float(pay.sum()), stress_payoff=float(pay[:, liars].sum()),
                honest_payoff=float(pay[:, honest].sum()),
                unsafe_share=float(np.mean([r["unsafe_next"] for r in rows])), final_q=rows[-1]["q_next"],
                mean_q=float(np.mean([r["q_next"] for r in rows])),
                audits=sum(r["audited"] is not None for r in rows), catches=sum(r["caught"] for r in rows),
                liars_caught=len({r["audited"] for r in rows if r["caught"]}), **score(rows))


# ------------------------------------------------------------------ open loop
def open_loop(none_rows, prof):
    out = []
    for r in none_rows:
        if not r["pre_safe"]:
            continue
        game, ctx, t, q = r["game"], r["context"], r["step"], r["q"]
        Par = TR.frozen_params(game)
        req = np.asarray(r["requests"])
        ref = TR.reference(Par, q, req, stable_seed(SEEDS["reference"] + 1, game, q, req.tolist()), prof["open_draws"])
        eps = TR.reviewer_draws(Par, stable_seed(SEEDS["reviewer"] + 1, "open", game, ctx, t), prof["reviewer_draws"])
        for rev in TR.REVIEWERS:
            dec = TR.review(Par, q, req, req, rev, eps)
            ex = dec["executed"]
            out.append(dict(game=game, context=ctx, step=t, reviewer=rev, q=q, label=ref["label"], ref_risk=ref["risk"],
                            exact_risk_req=TR.exact_risk(Par, q, req), approved=dec["approved"], scale=dec["scale"],
                            kept=float(ex.sum() / req.sum()) if req.sum() > 0 else 1.0,
                            exact_risk_exec=TR.exact_risk(Par, q, ex), exec_risky=int(TR.exact_risk(Par, q, ex) > TR.TOL),
                            x_req=float(req[list(TR.X_IDX)].sum()), y_req=float(req[list(TR.Y_IDX)].sum())))
    return out


def run(out, prof_name):
    if "Amendment 1 (calibration)" not in PROTOCOL.read_text() or TR.FROZEN is None:
        raise SystemExit("Gate 2: Amendment 1 (calibration) must be in protocol 21 and parameters frozen first.")
    prof = profile(prof_name)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    episodes, closed, none_rows = [], [], []
    for ctx in range(prof["contexts"]):
        for game in TR.GAMES:
            for part, arms in (("A", PART_A_ARMS), ("B", PART_B_ARMS)):
                for arm in arms:
                    ep, rows = episode(game, ctx, part, arm, prof)
                    episodes.append(ep)
                    (none_rows if arm == "none" else closed).extend(rows)
        print(f"context {ctx + 1}/{prof['contexts']} done, {time.time() - t0:.0f}s", flush=True)
    opened = open_loop(none_rows, prof)
    print(f"open loop done: {len(opened)} decisions, {time.time() - t0:.0f}s", flush=True)
    write_jsonl_gz(out / "episodes.jsonl.gz", episodes)
    write_jsonl_gz(out / "closed_loop_rows.jsonl.gz", closed)
    write_jsonl_gz(out / "none_rows.jsonl.gz", none_rows)
    write_jsonl_gz(out / "open_loop_decisions.jsonl.gz", opened)
    params = {g: TR.frozen_params(g).as_dict() for g in TR.GAMES}
    manifest = dict(experiment="C1", profile=prof_name, prof=prof, seeds=SEEDS, params=params,
                    ranges=TR.FROZEN_RANGES, range_values=TR.RANGE_SETS[TR.FROZEN_RANGES], lie=LIE,
                    n_episodes=len(episodes), n_closed=len(closed), n_none=len(none_rows), n_open=len(opened),
                    files={f.name: sha256(f) for f in sorted(out.glob("*.gz"))},
                    source={p: sha256(Path(p)) for p in ["fishery_sim/two_reagent.py", "fishery_sim/calibrated_oversight.py",
                                                         "experiments/oversight/claude_oversight_common.py",
                                                         "experiments/oversight/run_c1_compositional_harm.py"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    (out / "timing.json").write_text(json.dumps(dict(seconds=time.time() - t0)))  # kept out of the manifest so reruns are byte-identical
    print(json.dumps({k: manifest[k] for k in ("n_episodes", "n_closed", "n_none", "n_open")}), f"{time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("calibrate", "run"), required=True)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.stage == "calibrate":
        calibrate(Path(a.out))
    else:
        run(Path(a.out), a.profile)


if __name__ == "__main__":
    main()
