"""Experiment R1: the reviewer experiment, repaired (Claude audit, October 2026).

Protocol: notes/claude_audit_20261005/07_PROTOCOL_R1_REPAIRED_REVIEWER.md
Run:  PYTHONPATH=. python -m experiments.run_r1_repaired_reviewer --profile smoke|full --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.claude_oversight_common import (
    fishery_requests, fishery_setup, harvest_setup, sha256, write_jsonl_gz,
)
from fishery_sim.budgeted_oversight import inspection_order
from fishery_sim.calibrated_oversight import (
    REVIEWERS, fishery_choose_scale, fishery_reference, harvest_choose_scale,
    harvest_reference, stable_seed,
)
from fishery_sim.fishery_oversight import FisherySnapshot, safe as fishery_safe, transition
from fishery_sim.harvest import run_harvest_episode
from fishery_sim.oversight_protocol import harvest_safe

BUDGETS = (0, 3, 6)
FILLS = ("max", "previous")
TARGETS = ("one_step", "msy")
SEEDS = dict(fishery_pop=600_000_000, harvest_pop=610_000_000, weather=620_000_000,
             reviewer=630_000_000, reference=640_000_000)


def profile(name):
    if name == "smoke":
        return dict(contexts=2, horizon=20, closed_draws=2000, open_draws=4000, reviewer_draws=400)
    if name == "full":
        return dict(contexts=64, horizon=80, closed_draws=2000, open_draws=4000, reviewer_draws=400)
    raise ValueError(name)


def believed_vector(requests, budget, identity, fill, previous):
    order = inspection_order(len(requests), identity)[:budget]
    believed = np.ones(len(requests)) if (fill == "max" or previous is None) else np.asarray(previous, float).copy()
    believed[order] = np.asarray(requests)[order]
    return np.clip(believed, 0, 1)


# ------------------------------------------------------------------ Harvest
def harvest_episode(context, reviewer, budget, fill, P):
    cfg, specs, stressed = harvest_setup(context, SEEDS["harvest_pop"], SEEDS["weather"], P["horizon"])
    rows, state = [], dict(previous=None, step=0)

    def monitor(c, health, requests):
        t = state["step"]
        pre_safe = harvest_safe(c, health)
        row = dict(game="harvest", context=context, reviewer=reviewer, budget=budget, fill=fill,
                   step=t, pre_safe=int(pre_safe), state=health.tolist(), requests=requests.tolist(),
                   previous=None if state["previous"] is None else list(state["previous"]))
        if reviewer == "none":
            scale = 1.0
        else:
            identity = f"harvest__{context}__{t}"
            b = believed_vector(requests, budget, identity, fill, state["previous"])
            scale, est, cand = harvest_choose_scale(
                c, health, b, reviewer, stable_seed(SEEDS["reviewer"], "h", context, t, reviewer, budget, fill),
                draws=P["reviewer_draws"])
            row.update(est_risk=est, candidates=cand)
        ref = harvest_reference(c, health, requests, stable_seed(SEEDS["reference"], health.tolist(), requests.tolist()),
                                draws=P["closed_draws"])
        row.update(scale=scale, label=ref["label"], ref_risk=ref["risk"])
        rows.append(row)
        state["previous"], state["step"] = requests.copy(), t + 1
        return requests * scale

    result = run_harvest_episode(cfg, [s.to_agent() for s in specs], record_trace=True, action_filter=monitor)
    trace = result["episode_trace_rows"]
    unsafe = [int(r["global_unsafe"]) for r in trace]
    # fixed-horizon unsafe fraction: steps after garden failure count unsafe
    fixed = (sum(unsafe) + (P["horizon"] - len(trace))) / P["horizon"]
    ep = dict(game="harvest", context=context, reviewer=reviewer, budget=budget, fill=fill, target="one_step",
              total_harvest=result["total_welfare"], mean_health=result["mean_patch_health"],
              t_end=result["t_end"], failure=result["garden_failure_event"], unsafe_fixed=fixed,
              stressed=stressed)
    return ep, rows


# ------------------------------------------------------------------ Fishery
def fishery_episode(context, reviewer, budget, fill, target, P):
    cfg, pol, stressed = fishery_setup(context, SEEDS["fishery_pop"], P["horizon"])
    state, prev, rows, total, stocks, unsafe = FisherySnapshot(cfg.stock_init), None, [], 0.0, [], []
    for t in range(cfg.horizon):
        req = fishery_requests(pol, state.stock)
        row = dict(game="fishery", context=context, reviewer=reviewer, budget=budget, fill=fill, target=target,
                   step=t, pre_safe=int(fishery_safe(cfg, state)), state=state.stock, requests=req.tolist(),
                   previous=None if prev is None else prev.tolist())
        if reviewer == "none":
            scale = 1.0
        else:
            b = believed_vector(req, budget, f"fishery__{context}__{t}", fill, prev)
            scale, cand = fishery_choose_scale(cfg, state.stock, b, reviewer, target, state.collapsed)
            row.update(candidates=cand)
        ref = fishery_reference(cfg, state.stock, req, target)
        row.update(scale=scale, label=ref["label"])
        rows.append(row)
        future, payoffs, harvest = transition(cfg, state, req * scale)
        total += float(payoffs.sum())
        stocks.append(future.stock)
        unsafe.append(int(not fishery_safe(cfg, future)))
        prev, state = req, future
        if state.collapsed:
            break
    fixed = (sum(unsafe) + (cfg.horizon - len(unsafe))) / cfg.horizon
    ep = dict(game="fishery", context=context, reviewer=reviewer, budget=budget, fill=fill, target=target,
              total_harvest=total, mean_health=float(np.mean(stocks)), t_end=len(rows),
              failure=int(state.collapsed), unsafe_fixed=fixed, stressed=stressed)
    return ep, rows


# ------------------------------------------------------------------ open loop
def open_loop(none_rows, P):
    """Score every reviewer on requests recorded in no-reviewer runs (pre-state safe)."""
    from experiments.claude_oversight_common import harvest_setup as hs, fishery_setup as fs
    out = []
    for r in none_rows:
        if not r["pre_safe"]:
            continue
        g, ctx, t = r["game"], r["context"], r["step"]
        req = np.asarray(r["requests"])
        if g == "harvest":
            cfg = hs(ctx, SEEDS["harvest_pop"], SEEDS["weather"], P["horizon"])[0]
            h = np.asarray(r["state"])
            ref = harvest_reference(cfg, h, req, stable_seed(SEEDS["reference"] + 1, h.tolist(), req.tolist()),
                                    draws=P["open_draws"])
            for rev in REVIEWERS:
                for k in BUDGETS:
                    for fill in FILLS:
                        b = believed_vector(req, k, f"harvest__{ctx}__{t}", fill, r["previous"])
                        s, _, _ = harvest_choose_scale(cfg, h, b, rev,
                            stable_seed(SEEDS["reviewer"] + 1, "h", ctx, t, rev, k, fill), draws=P["reviewer_draws"])
                        out.append(dict(game=g, context=ctx, step=t, reviewer=rev, budget=k, fill=fill,
                                        target="one_step", label=ref["label"], ref_risk=ref["risk"], scale=s))
        else:
            cfg = fs(ctx, SEEDS["fishery_pop"], P["horizon"])[0]
            for target in TARGETS:
                ref = fishery_reference(cfg, r["state"], req, target)
                for rev in REVIEWERS:
                    for k in BUDGETS:
                        for fill in FILLS:
                            b = believed_vector(req, k, f"fishery__{ctx}__{t}", fill, r["previous"])
                            s, _ = fishery_choose_scale(cfg, r["state"], b, rev, target)
                            out.append(dict(game=g, context=ctx, step=t, reviewer=rev, budget=k, fill=fill,
                                            target=target, label=ref["label"], scale=s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = profile(a.profile)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    episodes, closed, none_rows = [], [], []
    for ctx in range(P["contexts"]):
        ep, rows = harvest_episode(ctx, "none", 0, "max", P); episodes.append(ep); none_rows += rows
        for rev in REVIEWERS:
            for k in BUDGETS:
                for fill in FILLS:
                    ep, rows = harvest_episode(ctx, rev, k, fill, P); episodes.append(ep); closed += rows
        for target in TARGETS:
            ep, rows = fishery_episode(ctx, "none", 0, "max", target, P)
            if target == "one_step":
                none_rows += rows
            episodes.append(ep)
            for rev in REVIEWERS:
                for k in BUDGETS:
                    for fill in FILLS:
                        ep, rows = fishery_episode(ctx, rev, k, fill, target, P); episodes.append(ep); closed += rows
        print(f"context {ctx} done, {time.time()-t0:.0f}s", flush=True)
    opened = open_loop(none_rows, P)
    write_jsonl_gz(out / "episodes.jsonl.gz", episodes)
    write_jsonl_gz(out / "closed_loop_decisions.jsonl.gz", closed)
    write_jsonl_gz(out / "none_rows.jsonl.gz", none_rows)
    write_jsonl_gz(out / "open_loop_decisions.jsonl.gz", opened)
    manifest = dict(experiment="R1", profile=a.profile, params=P, seeds=SEEDS, seconds=time.time() - t0,
                    n_episodes=len(episodes), n_closed=len(closed), n_open=len(opened),
                    files={f.name: sha256(f) for f in out.glob("*.gz")},
                    source={p: sha256(Path(p)) for p in ["fishery_sim/calibrated_oversight.py",
                            "experiments/claude_oversight_common.py", "experiments/run_r1_repaired_reviewer.py"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("n_episodes", "n_closed", "n_open", "seconds")}))


if __name__ == "__main__":
    main()
