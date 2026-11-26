"""Experiment T1: do audits aimed at the largest report miss under-reporters? (Claude audit, October 2026)

Protocol: notes/claude_audit_20261005/studies/T1_audit_targeting/protocol.md
Fishery and Forest (the code's Harvest) reuse the S3 Part D memory reviewer (fixed lie d = 0.5, no sanctions,
one audit per step); River reuses C1 Part B with shifted seeds. Arms: trust, random, report, signal (Forest only).
Run:  PYTHONPATH=. python -m experiments.oversight.run_t1_audit_targeting --profile smoke|full --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.oversight import run_c1_compositional_harm as c1
from experiments.oversight import run_s1_reporting_audit as s1
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

SEEDS = dict(fishery_pop=1_800_000_000, harvest_pop=1_810_000_000, weather=1_820_000_000, reviewer=1_830_000_000,
             reference=1_840_000_000, audit=1_845_000_000)
RIVER_SHIFT = 1_860_000_000 - 1_400_000_000
LIE = 0.5
ARM_PROTOCOL = {"trust": "report", "random": "rand1", "report": "rep1", "signal": "targ1"}
ARMS = {"fishery": ("trust", "random", "report"), "harvest": ("trust", "random", "report", "signal"),
        "river": ("trust", "random", "report")}
FISHERY_TARGET = "msy"


class T1Overseer(s3.MemoryOverseer):
    """S3 Part D memory overseer (fixed lies, sanction none) with selectable seeds and a report-aimed audit ('rep1')."""

    def __init__(self, protocol, n, misreporters, context, game, seeds):
        mode = "trust" if protocol == "report" else "memory"
        super().__init__(protocol, n, misreporters, LIE, context, game, mode, "fixed")
        self.seeds = seeds
        self.audits_on_liars = 0

    def step(self, t, p):
        r = np.array([p[i] * (1 - LIE) if i in self.mis else p[i] for i in range(self.n)])
        b, caught = r.copy(), np.zeros(self.n, bool)
        rng = np.random.default_rng(stable_seed(self.seeds["audit"], self.game, self.context, t, self.protocol))
        if self.mode == "memory":
            for i in range(self.n):
                if self.dhat[i]:
                    keep = 1 - float(np.mean(self.dhat[i]))
                    b[i] = 1.0 if keep < 1e-9 else min(1.0, r[i] / keep)
        audited = []
        if self.protocol.startswith("rand"):
            audited = rng.permutation(np.arange(self.n))[:int(self.protocol[-1])].tolist()
        elif self.protocol.startswith("targ"):
            jitter = rng.uniform(0, 1e-9, self.n)
            audited = list(np.argsort(-(self.suspicion + jitter))[:int(self.protocol[-1])])
        elif self.protocol == "rep1":
            audited = [int(np.argmax(b))]  # the largest believed request (report corrected by memory); ties: lowest index
        elif self.protocol != "report":  # 'report' = trust: no audit
            raise ValueError(self.protocol)
        for i in audited:
            b[i] = p[i]
        for i in audited:
            self.audits += 1
            self.audits_on_liars += int(i in self.mis)
            if r[i] < p[i] - 1e-12:
                caught[i] = True
                self.catches += 1
                self.catch_by_agent[i] += 1
                if p[i] > 0:
                    self.dhat[i].append(1 - r[i] / p[i])
        return p.copy(), b, caught


def harvest_t1(context, arm, P, seeds=SEEDS):
    protocol = ARM_PROTOCOL[arm]
    cfg, specs, mis = harvest_setup(context, seeds["harvest_pop"], seeds["weather"], P["horizon"])
    ov = T1Overseer(protocol, cfg.n_agents, mis, context, "harvest", seeds)
    st, rows = dict(t=0, pred=None), []

    def monitor(c, health, requests):
        t = st["t"]
        if st["pred"] is not None and protocol.startswith("targ"):
            ov.suspicion = 0.8 * ov.suspicion + (st["pred"] - health)
        row = dict(step=t, pre_safe=int(harvest_safe(c, health)))
        p_eff, b, caught = ov.step(t, requests)
        scale, _, _ = harvest_choose_scale(c, health, b, "joint", stable_seed(seeds["reviewer"], "h", False, context, t),
                                           draws=P["reviewer_draws"])
        executed = p_eff * scale
        st["pred"] = harvest_nominal_next(c, health, b * scale)
        row["label_exec"] = harvest_reference(c, health, executed, stable_seed(seeds["reference"], health.tolist(),
                                              executed.tolist()), draws=P["ref_draws"])["label"]
        row["label_req"] = harvest_reference(c, health, p_eff, stable_seed(seeds["reference"], health.tolist(),
                                             p_eff.tolist()), draws=P["ref_draws"])["label"]
        row["scale"] = scale
        rows.append(row)
        st["t"] = t + 1
        return executed

    res = run_harvest_episode(cfg, [x.to_agent() for x in specs], record_trace=True, action_filter=monitor)
    unsafe = [int(r["global_unsafe"]) for r in res["episode_trace_rows"]]
    e = s1.summarize("harvest", context, protocol, LIE, P["horizon"], np.asarray(res["final_payoffs"], float), mis,
                     res["mean_patch_health"], unsafe, rows, ov)
    e.update(arm=arm, audits_on_liars=ov.audits_on_liars, harm=e["unsafe_fixed"])
    return e


def fishery_t1(context, arm, P, seeds=SEEDS):
    protocol = ARM_PROTOCOL[arm]
    cfg, pol, mis = fishery_setup(context, seeds["fishery_pop"], P["horizon"])
    ov = T1Overseer(protocol, cfg.n_agents, mis, context, "fishery", seeds)
    state, pay, stocks, unsafe, rows = FisherySnapshot(cfg.stock_init), np.zeros(cfg.n_agents), [], [], []
    for t in range(cfg.horizon):
        req = fishery_requests(pol, state.stock)
        row = dict(step=t, pre_safe=int(fishery_safe(cfg, state)))
        p_eff, b, caught = ov.step(t, req)
        scale, _ = fishery_choose_scale(cfg, state.stock, b, "joint", FISHERY_TARGET, state.collapsed)
        executed = p_eff * scale
        row["label_exec"] = fishery_reference(cfg, state.stock, executed, FISHERY_TARGET)["label"]
        row["label_req"] = fishery_reference(cfg, state.stock, p_eff, FISHERY_TARGET)["label"]
        row["scale"] = scale
        rows.append(row)
        future, payoffs, _ = transition(cfg, state, executed)
        pay += payoffs
        stocks.append(future.stock)
        unsafe.append(int(not fishery_safe(cfg, future)))
        state = future
        if state.collapsed:
            break
    e = s1.summarize("fishery", context, protocol, LIE, cfg.horizon, pay, mis, float(np.mean(stocks)), unsafe, rows, ov)
    e.update(arm=arm, audits_on_liars=ov.audits_on_liars,
             harm=e["exec_risky"] / max(e["scored_steps"], 1))  # steps breaking the MSY limit
    return e


def river_t1(context, arm, prof, seeds=None):
    """C1 Part B episode with C1's seed bases shifted (seeds=None) or C1's own seeds (seeds='c1', for gate 3)."""
    original = dict(c1.SEEDS)
    if seeds is None:
        c1.SEEDS.update({k: v + RIVER_SHIFT for k, v in original.items()})
    try:
        summ, rows = c1.episode("comp", context, "B", {"trust": "trust", "random": "random", "report": "targeted"}[arm], prof)
    finally:
        c1.SEEDS.clear(); c1.SEEDS.update(original)
    liars = set(summ["stressed"])
    summ.update(arm=arm, audits_on_liars=sum(1 for r in rows if r["audited"] is not None and r["audited"] in liars),
                harm=summ["unsafe_share"])
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = s1.profile(a.profile)
    prof = c1.profile(a.profile)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0, episodes = time.time(), []
    test = range(P["contexts"])
    for game, arms in ARMS.items():
        for arm in arms:
            for c in test:
                e = fishery_t1(c, arm, P) if game == "fishery" else harvest_t1(c, arm, P) if game == "harvest" else \
                    river_t1(c, arm, prof)
                episodes.append({**e, "game": game, "arm": arm})
            print(f"{game} {arm} {time.time() - t0:.0f}s", flush=True)
    write_jsonl_gz(out / "episodes.jsonl.gz", episodes)
    manifest = dict(experiment="T1", profile=a.profile, params=P, river_profile=prof, seeds=SEEDS, river_shift=RIVER_SHIFT,
                    lie=LIE, seconds=time.time() - t0, n_episodes=len(episodes),
                    files={"episodes.jsonl.gz": sha256(out / "episodes.jsonl.gz")},
                    source={p: sha256(Path(p)) for p in ["experiments/oversight/run_t1_audit_targeting.py",
                            "experiments/oversight/run_s3_threshold_timing_memory.py",
                            "experiments/oversight/run_c1_compositional_harm.py", "fishery_sim/two_reagent.py",
                            "fishery_sim/calibrated_oversight.py"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    print(json.dumps(dict(n_episodes=len(episodes), seconds=manifest["seconds"])))


if __name__ == "__main__":
    main()
