"""Experiment S3: deterrence threshold, imperfect audits, timed cheating, reviewer memory (Claude audit, October 2026).

Protocol: notes/claude_audit_20261005/15_PROTOCOL_S3_THRESHOLD_TIMING_MEMORY.md

Parts A-C reuse the S2 non-compliance model in Fishery (MSY target): a cheater with level d takes
a_i + d * (p_i - a_i); audits check actual extraction afterwards and charge a flat fine per catch.
Part D reuses the S1 misreporting model with no sanction and adds a reviewer that remembers caught lies.
Run:  PYTHONPATH=. python -m experiments.oversight.run_s3_threshold_timing_memory --part A|B|C|D --profile smoke|full --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.oversight import run_s1_reporting_audit as s1
from experiments.oversight import run_s2_compliance_deterrence as s2
from experiments.oversight.claude_oversight_common import (
    fishery_requests, fishery_setup, harvest_setup, sha256, write_jsonl_gz,
)
from fishery_sim.calibrated_oversight import (
    fishery_choose_scale, fishery_reference, harvest_choose_scale, harvest_reference, stable_seed,
)
from fishery_sim.fishery_oversight import FisherySnapshot, safe as fishery_safe, transition
from fishery_sim.harvest import run_harvest_episode
from fishery_sim.oversight_protocol import harvest_nominal_next, harvest_safe

SEEDS = dict(fishery_pop=900_000_000, reviewer=930_000_000, audit=945_000_000, detect=946_000_000,
             train_fishery_pop=950_000_000, noisy=947_000_000)
E_GRID = (0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0)
PART_A_Q = (1 / 6, 0.10, 0.05, 0.02)
PART_B_S = (0.5, 0.25)
PART_C_FINES = (0.0, 6.0, 24.0)
SEARCH_D = s2.SEARCH_D
TIMINGS = ("always", "avoid")
FISHERY_TARGET = s2.FISHERY_TARGET
PART_D_CELLS = [("harvest", "one_step", p) for p in ("rand1", "rand2", "targ1")] + \
               [("fishery", t, p) for t in ("one_step", "msy") for p in ("rand1", "rand2")]
MODES = ("memoryless", "memory", "memory_max")
LIARS = ("fixed", "noisy")


# ================================================================== Parts A-C: non-compliance
class Auditor(s2.Auditor):
    """S2 auditor plus Bernoulli audits with detection probability s and a known periodic schedule."""

    def __init__(self, protocol, n, cheaters, d, fine, context, game, seeds, q=None, s=1.0, timing="always"):
        super().__init__(protocol, n, cheaters, d, fine, context, game)
        self.seeds, self.q, self.s, self.timing = seeds, q, s, timing
        self.over_steps = np.zeros(n, int)

    def audit_chance(self, t):
        """Each agent's known chance of being audited at step t."""
        if self.protocol == "bern":
            return self.q
        if self.protocol == "periodic6":
            return 1.0 if t % 6 == 5 else 0.0
        if self.protocol.startswith("rand"):
            return int(self.protocol[-1]) / self.n
        return 0.0

    def execute(self, p, allowance, t=None):
        if self.timing == "avoid" and self.audit_chance(t) >= 0.5:
            return allowance.copy()
        taken = super().execute(p, allowance)
        self.over_steps += taken > allowance + 1e-12
        return taken

    def audit(self, t, allowance, taken):
        if self.protocol not in ("bern", "periodic6"):
            return super().audit(t, allowance, taken)
        over = taken > allowance + 1e-12
        if self.protocol == "bern":
            # audit draws depend on (context, step, q) only, so every fine and cheating level sees the same audits
            audited = np.random.default_rng(stable_seed(self.seeds["audit"], self.context, t, round(self.q, 6))).random(self.n) < self.q
        else:
            audited = np.full(self.n, t % 6 == 5)
        detected = np.random.default_rng(stable_seed(self.seeds["detect"], self.context, t)).random(self.n) < self.s
        caught = audited & over & detected
        self.audits += int(audited.sum())
        for i in np.flatnonzero(caught):
            self.catches += 1
            self.fines[i] += self.fine
            self.caught_honest += int(i not in self.cheat)
        return caught


def fishery_episode(context, protocol, d, fine, P, seeds, train=False, q=None, s=1.0, timing="always"):
    """S2 Fishery episode (s2.fishery_episode) with the extended auditor and selectable seeds."""
    cfg, pol, cheat = fishery_setup(context, seeds["train_fishery_pop"] if train else seeds["fishery_pop"], P["horizon"])
    au = Auditor(protocol, cfg.n_agents, cheat, d, fine, context, "fishery", seeds, q, s, timing)
    state, pay, stocks, unsafe, rows = FisherySnapshot(cfg.stock_init), np.zeros(cfg.n_agents), [], [], []
    for t in range(cfg.horizon):
        req = fishery_requests(pol, state.stock)
        row = dict(step=t, pre_safe=int(fishery_safe(cfg, state)))
        if protocol == "none":
            allowance = req.copy()
        else:
            scale, _ = fishery_choose_scale(cfg, state.stock, req, "joint", FISHERY_TARGET, state.collapsed)
            allowance = req * scale
        taken = au.execute(req, allowance, t) if protocol != "none" else allowance
        au.audit(t, allowance, taken)
        if not train:
            row["label_exec"] = fishery_reference(cfg, state.stock, taken, FISHERY_TARGET)["label"]
        rows.append(row)
        future, payoffs, _ = transition(cfg, state, taken)
        pay += payoffs
        stocks.append(future.stock)
        unsafe.append(int(not fishery_safe(cfg, future)))
        state = future
        if state.collapsed:
            break
    e = s2.summarize("fishery", context, protocol, d, fine, cfg.horizon, pay, cheat, float(np.mean(stocks)),
                     unsafe, rows, au)
    e.update(q=q, s=s, timing=timing, cheater_over_steps=float(au.over_steps[cheat].mean()))
    return e


def run_nc(protocol, d, fine, P, contexts, seeds, train=False, **kw):
    return [fishery_episode(c, protocol, d, fine, P, seeds, train, **kw) for c in contexts]


def search_nc(protocol, fine, P, seeds, timings=("always",), **kw):
    """Common cheating strategy maximising mean cheater net payoff on training contexts; ties -> smaller d, 'always'."""
    best, best_score, scores = None, -np.inf, {}
    for timing in timings:
        for d in SEARCH_D:
            sc = float(np.mean([e["cheater_payoff"] for e in run_nc(protocol, d, fine, P, range(P["train_contexts"]),
                                                                       seeds, True, timing=timing, **kw)]))
            scores[f"{timing}:{d}"] = sc
            if best is None or sc > best_score + 1e-9:
                best, best_score = (d, timing), sc
    return best, scores


def part_ab(part, P):
    episodes, searches = [], []
    test = range(P["contexts"])
    for e in run_nc("allow", 0.0, 0.0, P, test, SEEDS):
        episodes.append({**e, "actor": "comply"})
    (d_allow, _), sc = search_nc("allow", 0.0, P, SEEDS)
    searches.append(dict(protocol="allow", q=None, s=1.0, e=None, fine=0.0, d_star=d_allow, train_scores=sc))
    for e in run_nc("allow", d_allow, 0.0, P, test, SEEDS):
        episodes.append({**e, "actor": "adaptive"})
    combos = [(q, 1.0) for q in PART_A_Q] if part == "A" else [(1 / 6, s) for s in PART_B_S]
    for q, s in combos:
        for ev in E_GRID:
            fine = ev / (q * s)
            (d, _), sc = search_nc("bern", fine, P, SEEDS, q=q, s=s)
            searches.append(dict(protocol="bern", q=q, s=s, e=ev, fine=fine, d_star=d, train_scores=sc))
            for e in run_nc("bern", d, fine, P, test, SEEDS, q=q, s=s):
                episodes.append({**e, "actor": "adaptive", "e": ev})
            print(f"{part} q={q:.3f} s={s} e={ev} F={fine:.2f} d*={d}", flush=True)
    return episodes, searches


def part_c(P):
    episodes, searches = [], []
    test = range(P["contexts"])
    for e in run_nc("allow", 0.0, 0.0, P, test, SEEDS):
        episodes.append({**e, "actor": "comply"})
    (d_allow, _), sc = search_nc("allow", 0.0, P, SEEDS)
    searches.append(dict(protocol="allow", fine=0.0, d_star=d_allow, timing="always", train_scores=sc))
    for e in run_nc("allow", d_allow, 0.0, P, test, SEEDS):
        episodes.append({**e, "actor": "adaptive"})
    for protocol, kw in (("bern", dict(q=1 / 6)), ("periodic6", {})):
        for fine in PART_C_FINES:
            (d, timing), sc = search_nc(protocol, fine, P, SEEDS, timings=TIMINGS, **kw)
            searches.append(dict(protocol=protocol, fine=fine, d_star=d, timing=timing, train_scores=sc))
            for e in run_nc(protocol, d, fine, P, test, SEEDS, timing=timing, **kw):
                episodes.append({**e, "actor": "adaptive"})
            print(f"C {protocol} F={fine} d*={d} timing={timing}", flush=True)
    return episodes, searches


# ================================================================== Part D: reviewer memory
class MemoryOverseer(s1.Overseer):
    """S1 overseer, sanction none, with optional memory of caught lies and per-step noisy lies."""

    def __init__(self, protocol, n, misreporters, d, context, game, mode, liar):
        super().__init__(protocol, n, misreporters, d, context, game, belief=True, sanction="none")
        self.mode, self.liar = mode, liar
        self.dhat = [[] for _ in range(n)]

    def lie(self, t, i):
        if self.liar == "fixed":
            return self.d
        return float(np.random.default_rng(stable_seed(SEEDS["noisy"], self.game, self.context, t, i)).random())

    def step(self, t, p):
        if self.mode == "trust" and self.liar == "fixed":
            return super().step(t, p)  # identical to S1 'report' / S1b belief-on, sanction-none
        r = np.array([p[i] * (1 - self.lie(t, i)) if i in self.mis else p[i] for i in range(self.n)])
        b, caught = r.copy(), np.zeros(self.n, bool)
        rng = np.random.default_rng(stable_seed(s1.SEEDS["audit"], self.game, self.context, t, self.protocol))
        audited = []
        if self.protocol.startswith("rand"):
            audited = rng.permutation(np.arange(self.n))[:int(self.protocol[-1])].tolist()
        elif self.protocol.startswith("targ"):
            jitter = rng.uniform(0, 1e-9, self.n)
            audited = list(np.argsort(-(self.suspicion + jitter))[:int(self.protocol[-1])])
        elif self.protocol != "report":
            raise ValueError(self.protocol)
        if self.mode in ("memory", "memory_max"):
            for i in range(self.n):
                if self.dhat[i]:
                    if self.mode == "memory_max":
                        b[i] = 1.0
                    else:
                        keep = 1 - float(np.mean(self.dhat[i]))
                        b[i] = 1.0 if keep < 1e-9 else min(1.0, r[i] / keep)
        if self.mode != "trust":
            for i in audited:
                b[i] = p[i]
        for i in audited:
            self.audits += 1
            if r[i] < p[i] - 1e-12:
                caught[i] = True
                self.catches += 1
                self.catch_by_agent[i] += 1
                if p[i] > 0:
                    self.dhat[i].append(1 - r[i] / p[i])
        return p.copy(), b, caught


def harvest_episode_d(context, protocol, mode, liar, P, d=0.5):
    """s1.harvest_episode with the memory overseer (test contexts only)."""
    cfg, specs, mis = harvest_setup(context, s1.SEEDS["harvest_pop"], s1.SEEDS["weather"], P["horizon"])
    ov = MemoryOverseer(protocol, cfg.n_agents, mis, d, context, "harvest", mode, liar)
    st, rows = dict(t=0, pred=None), []

    def monitor(c, health, requests):
        t = st["t"]
        if st["pred"] is not None and protocol.startswith("targ"):
            ov.suspicion = 0.8 * ov.suspicion + (st["pred"] - health)
        row = dict(step=t, pre_safe=int(harvest_safe(c, health)))
        if protocol == "full":
            p_eff, b = requests.copy(), requests.copy()
            ov.audits += ov.n
        else:
            p_eff, b, caught = ov.step(t, requests)
            row["caught"] = int(caught.sum())
        scale, _, _ = harvest_choose_scale(c, health, b, "joint", stable_seed(s1.SEEDS["reviewer"], "h", False, context, t),
                                           draws=P["reviewer_draws"])
        executed = p_eff * scale
        st["pred"] = harvest_nominal_next(c, health, b * scale)
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
    return s1.summarize("harvest", context, protocol, d, P["horizon"], np.asarray(res["final_payoffs"], float), mis,
                        res["mean_patch_health"], unsafe, rows, ov)


def fishery_episode_d(context, protocol, mode, liar, P, target, d=0.5):
    """s1.fishery_episode with the memory overseer (test contexts only)."""
    cfg, pol, mis = fishery_setup(context, s1.SEEDS["fishery_pop"], P["horizon"])
    ov = MemoryOverseer(protocol, cfg.n_agents, mis, d, context, "fishery", mode, liar)
    state, pay, stocks, unsafe, rows = FisherySnapshot(cfg.stock_init), np.zeros(cfg.n_agents), [], [], []
    for t in range(cfg.horizon):
        req = fishery_requests(pol, state.stock)
        row = dict(step=t, pre_safe=int(fishery_safe(cfg, state)))
        if protocol == "full":
            p_eff, b = req.copy(), req.copy()
            ov.audits += ov.n
        else:
            p_eff, b, caught = ov.step(t, req)
            row["caught"] = int(caught.sum())
        scale, _ = fishery_choose_scale(cfg, state.stock, b, "joint", target, state.collapsed)
        executed = p_eff * scale
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
    return s1.summarize("fishery", context, protocol, d, cfg.horizon, pay, mis, float(np.mean(stocks)), unsafe, rows, ov)


def part_d(P):
    episodes, done_baselines = [], set()
    test = range(P["contexts"])
    for game, target, protocol in PART_D_CELLS:
        fn = (lambda c, pr, m, li: harvest_episode_d(c, pr, m, li, P)) if game == "harvest" else \
             (lambda c, pr, m, li: fishery_episode_d(c, pr, m, li, P, target))
        tag = dict(game=game, target=target)
        if (game, target) not in done_baselines:
            done_baselines.add((game, target))
            for e in (fn(c, "full", "trust", "fixed") for c in test):
                episodes.append({**e, **tag, "mode": "full", "liar": "none", "d": 0.0})
            for liar in LIARS:
                for e in (fn(c, "report", "trust", liar) for c in test):
                    episodes.append({**e, **tag, "mode": "trust", "liar": liar})
        for liar in LIARS:
            for mode in MODES:
                for e in (fn(c, protocol, mode, liar) for c in test):
                    episodes.append({**e, **tag, "mode": mode, "liar": liar})
                print(f"D {game} {target} {protocol} {liar} {mode}", flush=True)
    return episodes, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = s2.profile(a.profile)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    episodes, searches = {"A": lambda: part_ab("A", P), "B": lambda: part_ab("B", P), "C": lambda: part_c(P),
                          "D": lambda: part_d(P)}[a.part]()
    write_jsonl_gz(out / "episodes.jsonl.gz", episodes)
    (out / "adaptive_search.json").write_text(json.dumps(searches, indent=1))
    manifest = dict(experiment=f"S3-{a.part}", profile=a.profile, params=P, seeds=SEEDS, e_grid=E_GRID,
                    seconds=time.time() - t0, n_episodes=len(episodes),
                    files={"episodes.jsonl.gz": sha256(out / "episodes.jsonl.gz"),
                           "adaptive_search.json": sha256(out / "adaptive_search.json")},
                    source={p: sha256(Path(p)) for p in ["fishery_sim/calibrated_oversight.py",
                            "experiments/oversight/claude_oversight_common.py",
                            "experiments/oversight/run_s1_reporting_audit.py",
                            "experiments/oversight/run_s2_compliance_deterrence.py",
                            "experiments/oversight/run_s3_threshold_timing_memory.py"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(dict(n_episodes=len(episodes), seconds=manifest["seconds"])))


if __name__ == "__main__":
    main()
