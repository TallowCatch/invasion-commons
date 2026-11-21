"""Experiment S4: adaptive cheaters against a reviewer with memory, and a cost per audit (Claude audit, October 2026).

Protocol: notes/claude_audit_20261005/studies/S4_adaptive_cheaters_audit_cost/protocol.md

Part A: S2/S3 non-compliance in Fishery (MSY target) with Bernoulli audits under four regimes (none, fine,
memory, fine+memory). Memory gives a caught agent a targeted allowance so that its expected take is the planned
one. Cheaters choose a level and a reaction to being caught. Audit cost is applied in the analysis.
Part B: S3 Part D misreporting with a memory reviewer, adaptive lie level.
Run:  PYTHONPATH=. python -m experiments.oversight.run_s4_adaptive_memory_cost --part A|B --profile smoke|full --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.oversight import run_s1_reporting_audit as s1
from experiments.oversight import run_s2_compliance_deterrence as s2
from experiments.oversight import run_s3_threshold_timing_memory as s3
from experiments.oversight.claude_oversight_common import (
    fishery_requests, fishery_setup, harvest_setup, sha256, write_jsonl_gz,
)
from fishery_sim.calibrated_oversight import (
    fishery_choose_scale, fishery_reference, harvest_choose_scale, harvest_reference, stable_seed,
)
from fishery_sim.fishery_oversight import FisherySnapshot, safe as fishery_safe, transition
from fishery_sim.harvest import run_harvest_episode
from fishery_sim.oversight_protocol import harvest_nominal_next, harvest_safe

SEEDS = dict(fishery_pop=1_000_000_000, audit=1_045_000_000, detect=1_046_000_000, train_fishery_pop=1_050_000_000)
FINE = 6.0
REGIMES = ("fine", "memory", "fine+memory")
RATES = (0.01, 0.02, 0.05, 0.10, 1 / 6, 1 / 3)
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
REACTIONS = ("continue", "stop")
PART_B = (("harvest", "one_step"), ("fishery", "one_step"), ("fishery", "msy"))
B_LEVELS = s1.SEARCH_D


# ================================================================== Part A
def targeted_allowance(scale, dbar, p):
    """Allowance whose expected take, for an agent that keeps over-taking a share dbar of the cut, is scale * p."""
    if dbar >= 1 - 1e-12:
        return 0.0
    return max(0.0, (scale - dbar) / (1 - dbar)) * p


def observed_overtake(taken, allowance, p):
    return (taken - allowance) / (p - allowance) if p - allowance > 1e-12 else None


class Auditor(s3.Auditor):
    """S3 Bernoulli auditor (s = 1) plus memory of caught over-takes and a per-cheater reaction to being caught."""

    def __init__(self, regime, n, cheaters, d, context, seeds, q, reaction="continue"):
        audited = regime != "none"
        super().__init__("bern" if audited else "allow", n, cheaters, d, FINE if "fine" in regime else 0.0, context,
                         "fishery", seeds, q=q if audited else None, s=1.0)
        self.memory, self.reaction = "memory" in regime, reaction
        self.record = [[] for _ in range(n)]
        self.stopped = np.zeros(n, bool)

    def allowances(self, scale, p):
        a = scale * p
        if self.memory:
            for i in range(self.n):
                if self.record[i]:
                    a[i] = targeted_allowance(scale, float(np.mean(self.record[i])), p[i])
        return a

    def execute(self, p, allowance, t=None):
        self._p = p
        taken = allowance.copy()
        for i in self.cheat:
            if not self.stopped[i]:
                taken[i] = allowance[i] + self.d * (p[i] - allowance[i])
        self.over_steps += taken > allowance + 1e-12
        return taken

    def audit(self, t, allowance, taken):
        caught = super().audit(t, allowance, taken)
        for i in np.flatnonzero(caught):
            if self.memory:
                d_obs = observed_overtake(taken[i], allowance[i], self._p[i])
                if d_obs is not None:
                    self.record[i].append(d_obs)
            if self.reaction == "stop":
                self.stopped[i] = True
        return caught


def episode_a(context, regime, q, d, reaction, P, seeds=SEEDS, train=False):
    cfg, pol, cheat = fishery_setup(context, seeds["train_fishery_pop"] if train else seeds["fishery_pop"], P["horizon"])
    au = Auditor(regime, cfg.n_agents, cheat, d, context, seeds, q, reaction)
    state, pay, stocks, unsafe, rows = FisherySnapshot(cfg.stock_init), np.zeros(cfg.n_agents), [], [], []
    for t in range(cfg.horizon):
        req = fishery_requests(pol, state.stock)
        row = dict(step=t, pre_safe=int(fishery_safe(cfg, state)))
        scale, _ = fishery_choose_scale(cfg, state.stock, req, "joint", s3.FISHERY_TARGET, state.collapsed)
        allowance = au.allowances(scale, req)
        taken = au.execute(req, allowance, t)
        au.audit(t, allowance, taken)
        if not train:
            row["label_exec"] = fishery_reference(cfg, state.stock, taken, s3.FISHERY_TARGET)["label"]
        rows.append(row)
        future, payoffs, _ = transition(cfg, state, taken)
        pay += payoffs
        stocks.append(future.stock)
        unsafe.append(int(not fishery_safe(cfg, future)))
        state = future
        if state.collapsed:
            break
    e = s2.summarize("fishery", context, au.protocol, d, au.fine, cfg.horizon, pay, cheat, float(np.mean(stocks)),
                     unsafe, rows, au)
    e.update(regime=regime, q=q, reaction=reaction, cheater_over_steps=float(au.over_steps[cheat].mean()),
             flagged=int(sum(bool(r) for r in au.record)), stopped=int(au.stopped.sum()))
    return e


def search_a(regime, q, P):
    best, best_score, scores = None, -np.inf, {}
    for d in LEVELS:
        for reaction in REACTIONS:
            sc = float(np.mean([episode_a(c, regime, q, d, reaction, P, train=True)["cheater_payoff"]
                                for c in range(P["train_contexts"])]))
            scores[f"{d}:{reaction}"] = sc
            if best is None or sc > best_score + 1e-9:
                best, best_score = (d, reaction), sc
    return best, scores


def part_a(P):
    episodes, searches = [], []
    test = range(P["contexts"])
    for e in (episode_a(c, "none", None, 0.0, "continue", P) for c in test):
        episodes.append({**e, "actor": "comply"})
    (d0, r0), sc = search_a("none", None, P)
    searches.append(dict(regime="none", q=None, d_star=d0, reaction=r0, train_scores=sc))
    for e in (episode_a(c, "none", None, d0, r0, P) for c in test):
        episodes.append({**e, "actor": "adaptive"})
    for regime in REGIMES:
        for q in RATES:
            (d, r), sc = search_a(regime, q, P)
            searches.append(dict(regime=regime, q=q, d_star=d, reaction=r, train_scores=sc))
            for e in (episode_a(c, regime, q, d, r, P) for c in test):
                episodes.append({**e, "actor": "adaptive"})
            for e in (episode_a(c, regime, q, 0.0, "continue", P) for c in test):  # audits on compliers: cost baseline
                episodes.append({**e, "actor": "comply"})
            print(f"A {regime} q={q:.3f} d*={d} {r}", flush=True)
    return episodes, searches


# ================================================================== Part B
def harvest_b(context, mode, d, P, train=False):
    """S3 harvest_episode_d with optional training seeds (labels only on test contexts)."""
    cfg, specs, mis = harvest_setup(context, s1.SEEDS["train_harvest_pop" if train else "harvest_pop"],
                                    s1.SEEDS["train_weather" if train else "weather"], P["horizon"])
    ov = s3.MemoryOverseer("rand2", cfg.n_agents, mis, d, context, "harvest", mode, "fixed")
    st, rows = dict(t=0, pred=None), []

    def monitor(c, health, requests):
        t = st["t"]
        row = dict(step=t, pre_safe=int(harvest_safe(c, health)))
        p_eff, b, caught = ov.step(t, requests)
        scale, _, _ = harvest_choose_scale(c, health, b, "joint", stable_seed(s1.SEEDS["reviewer"], "h", train, context, t),
                                           draws=P["reviewer_draws"])
        executed = p_eff * scale
        st["pred"] = harvest_nominal_next(c, health, b * scale)
        if not train:
            row["label_exec"] = harvest_reference(c, health, executed, stable_seed(s1.SEEDS["reference"], health.tolist(),
                                                  executed.tolist()), draws=P["ref_draws"])["label"]
            row["label_req"] = harvest_reference(c, health, p_eff, stable_seed(s1.SEEDS["reference"], health.tolist(),
                                                 p_eff.tolist()), draws=P["ref_draws"])["label"]
        row["scale"] = scale
        rows.append(row)
        st["t"] = t + 1
        return executed

    res = run_harvest_episode(cfg, [x.to_agent() for x in specs], record_trace=True, action_filter=monitor)
    unsafe = [int(r["global_unsafe"]) for r in res["episode_trace_rows"]]
    return s1.summarize("harvest", context, "rand2", d, P["horizon"], np.asarray(res["final_payoffs"], float), mis,
                        res["mean_patch_health"], unsafe, rows, ov)


def fishery_b(context, mode, d, P, target, train=False):
    """S3 fishery_episode_d with optional training seeds."""
    cfg, pol, mis = fishery_setup(context, s1.SEEDS["train_fishery_pop" if train else "fishery_pop"], P["horizon"])
    ov = s3.MemoryOverseer("rand2", cfg.n_agents, mis, d, context, "fishery", mode, "fixed")
    state, pay, stocks, unsafe, rows = FisherySnapshot(cfg.stock_init), np.zeros(cfg.n_agents), [], [], []
    for t in range(cfg.horizon):
        req = fishery_requests(pol, state.stock)
        row = dict(step=t, pre_safe=int(fishery_safe(cfg, state)))
        p_eff, b, caught = ov.step(t, req)
        scale, _ = fishery_choose_scale(cfg, state.stock, b, "joint", target, state.collapsed)
        executed = p_eff * scale
        if not train:
            row["label_exec"] = fishery_reference(cfg, state.stock, executed, target)["label"]
            row["label_req"] = fishery_reference(cfg, state.stock, p_eff, target)["label"]
        row["scale"] = scale
        rows.append(row)
        future, payoffs, _ = transition(cfg, state, executed)
        pay += payoffs
        stocks.append(future.stock)
        unsafe.append(int(not fishery_safe(cfg, future)))
        state = future
        if state.collapsed:
            break
    return s1.summarize("fishery", context, "rand2", d, cfg.horizon, pay, mis, float(np.mean(stocks)), unsafe, rows, ov)


def part_b(P):
    episodes, searches = [], []
    for game, target in PART_B:
        fn = (lambda c, d, train=False: harvest_b(c, "memory", d, P, train)) if game == "harvest" else \
             (lambda c, d, train=False: fishery_b(c, "memory", d, P, target, train))
        scores = {d: float(np.mean([fn(c, d, True)["misreporter_payoff"] for c in range(P["train_contexts"])]))
                  for d in B_LEVELS}
        best = B_LEVELS[0]
        for d in B_LEVELS[1:]:
            if scores[d] > scores[best] + 1e-9:
                best = d
        searches.append(dict(game=game, target=target, d_star=best, train_scores=scores))
        for d in sorted({0.0, best}):
            for e in (fn(c, d) for c in range(P["contexts"])):
                episodes.append({**e, "target": target, "actor": "adaptive" if d == best else "comply"})
        print(f"B {game} {target} d*={best} {scores}", flush=True)
    return episodes, searches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["A", "B"])
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = s2.profile(a.profile)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    episodes, searches = part_a(P) if a.part == "A" else part_b(P)
    write_jsonl_gz(out / "episodes.jsonl.gz", episodes)
    (out / "adaptive_search.json").write_text(json.dumps(searches, indent=1))
    manifest = dict(experiment=f"S4-{a.part}", profile=a.profile, params=P, seeds=SEEDS, fine=FINE, rates=RATES,
                    seconds=time.time() - t0, n_episodes=len(episodes),
                    files={"episodes.jsonl.gz": sha256(out / "episodes.jsonl.gz"),
                           "adaptive_search.json": sha256(out / "adaptive_search.json")},
                    source={p: sha256(Path(p)) for p in ["fishery_sim/calibrated_oversight.py",
                            "experiments/oversight/claude_oversight_common.py",
                            "experiments/oversight/run_s1_reporting_audit.py",
                            "experiments/oversight/run_s2_compliance_deterrence.py",
                            "experiments/oversight/run_s3_threshold_timing_memory.py",
                            "experiments/oversight/run_s4_adaptive_memory_cost.py"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(dict(n_episodes=len(episodes), seconds=manifest["seconds"])))


if __name__ == "__main__":
    main()
