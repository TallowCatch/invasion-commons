"""Experiment S1: self-reports, audits and agents who misreport (Claude audit, October 2026).

Protocol: notes/claude_audit_20261005/studies/S1_reporting_and_audits/protocol.md
Run:  PYTHONPATH=. python -m experiments.oversight.run_s1_reporting_audit --profile smoke|full --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.oversight.claude_oversight_common import (
    fishery_requests, fishery_setup, harvest_setup, sha256, write_jsonl_gz,
)
from fishery_sim.calibrated_oversight import (
    fishery_choose_scale, fishery_reference, harvest_choose_scale, harvest_reference, stable_seed,
)
from fishery_sim.fishery_oversight import FisherySnapshot, safe as fishery_safe, transition
from fishery_sim.harvest import run_harvest_episode
from fishery_sim.oversight_protocol import harvest_nominal_next, harvest_safe

SEEDS = dict(fishery_pop=700_000_000, harvest_pop=710_000_000, weather=720_000_000,
             reviewer=730_000_000, reference=740_000_000, audit=745_000_000,
             train_fishery_pop=750_000_000, train_harvest_pop=760_000_000, train_weather=770_000_000)
HARVEST_PROTOCOLS = ("report", "rand1", "rand2", "targ1", "targ2", "peer", "peer_collude")
FISHERY_PROTOCOLS = ("report", "rand1", "rand2", "peer", "peer_collude")
FIXED_D = (0.0, 0.25, 0.5)
SEARCH_D = (0.0, 0.25, 0.5, 0.75)
EXCLUSION_STEPS = 5  # caught agents get 0 this step and the next 5
FINE = 12.0  # Amendment 1: fine per catch (two steps of one agent's maximum harvest)


def profile(name):
    if name == "smoke":
        return dict(contexts=2, train_contexts=2, horizon=20, reviewer_draws=400, ref_draws=2000)
    if name == "full":
        return dict(contexts=64, train_contexts=8, horizon=80, reviewer_draws=400, ref_draws=2000)
    raise ValueError(name)


class Overseer:
    """Applies one protocol: who is audited, who is caught, what the reviewer believes."""

    def __init__(self, protocol, n, misreporters, d, context, game, belief=True, sanction="excl+fine"):
        # belief/sanction factors added for the S1b ablation (S1b/S2 protocol); defaults reproduce S1.
        if sanction not in ("excl+fine", "fine", "none"):
            raise ValueError(sanction)
        self.belief, self.sanction = belief, sanction
        self.protocol, self.n, self.d = protocol, n, d
        self.mis = set(misreporters)
        self.excluded_until = np.full(n, -1)
        self.suspicion = np.zeros(n)
        self.context, self.game = context, game
        self.audits = self.messages = self.catches = 0
        self.catch_by_agent = np.zeros(n, int)
        self.fines = np.zeros(n)

    def reports(self, p):
        return np.array([p[i] * (1 - self.d) if i in self.mis else p[i] for i in range(self.n)])

    def step(self, t, p):
        """Return (effective true requests, believed vector, caught mask)."""
        excluded = self.excluded_until >= t
        p_eff = np.where(excluded, 0.0, p)
        r = np.where(excluded, 0.0, self.reports(p))
        b, caught = r.copy(), np.zeros(self.n, bool)
        active = np.flatnonzero(~excluded)
        rng = np.random.default_rng(stable_seed(SEEDS["audit"], self.game, self.context, t, self.protocol))
        audited = []
        if self.protocol == "full":
            b = p_eff.copy(); self.audits += self.n
        elif self.protocol.startswith("rand"):
            k = int(self.protocol[-1])
            audited = rng.permutation(active)[:k].tolist() if len(active) else []
        elif self.protocol.startswith("targ"):
            k = int(self.protocol[-1])
            jitter = rng.uniform(0, 1e-9, self.n)
            order = [i for i in np.argsort(-(self.suspicion + jitter)) if i in set(active.tolist())]
            audited = order[:k]
        elif self.protocol.startswith("peer"):
            collude = self.protocol == "peer_collude"
            self.messages += self.n
            for i in active:
                m = (i + 1) % self.n
                q = r[i] if (collude and i in self.mis and m in self.mis) else p_eff[i]
                if q > r[i] + 1e-12:
                    caught[i] = True
                b[i] = max(r[i], q) if self.belief else r[i]
        elif self.protocol != "report":
            raise ValueError(self.protocol)
        for i in audited:
            self.audits += 1
            if self.belief:
                b[i] = p_eff[i]
            if r[i] < p_eff[i] - 1e-12:
                caught[i] = True
        for i in np.flatnonzero(caught):
            self.catches += 1
            self.catch_by_agent[i] += 1
            if self.sanction in ("excl+fine", "fine"):
                self.fines[i] += FINE
            if self.sanction == "excl+fine":
                self.excluded_until[i] = t + EXCLUSION_STEPS
        if self.sanction == "excl+fine":
            p_eff = np.where(caught, 0.0, p_eff)
            b = np.where(caught, 0.0, b)
        return p_eff, b, caught


# ------------------------------------------------------------------ Harvest
def harvest_episode(context, protocol, d, P, train=False, belief=True, sanction="excl+fine"):
    pop = SEEDS["train_harvest_pop"] if train else SEEDS["harvest_pop"]
    wb = SEEDS["train_weather"] if train else SEEDS["weather"]
    cfg, specs, mis = harvest_setup(context, pop, wb, P["horizon"])
    ov = Overseer(protocol, cfg.n_agents, mis, d, context, "harvest", belief, sanction)
    st = dict(t=0, pred=None)
    rows = []

    def monitor(c, health, requests):
        t = st["t"]
        if st["pred"] is not None and protocol.startswith("targ"):
            ov.suspicion = 0.8 * ov.suspicion + (st["pred"] - health)  # below prediction -> more suspicious
        row = dict(step=t, pre_safe=int(harvest_safe(c, health)))
        if protocol == "none":
            executed, scale, p_eff = requests.copy(), 1.0, requests.copy()
        else:
            p_eff, b, caught = ov.step(t, requests)
            scale, _, _ = harvest_choose_scale(c, health, b, "joint",
                                               stable_seed(SEEDS["reviewer"], "h", train, context, t),
                                               draws=P["reviewer_draws"])
            executed = p_eff * scale
            st["pred"] = harvest_nominal_next(c, health, b * scale)
            row["caught"] = int(caught.sum())
        if not train:
            seed = stable_seed(SEEDS["reference"], health.tolist(), executed.tolist())
            row["label_exec"] = harvest_reference(c, health, executed, seed, draws=P["ref_draws"])["label"]
            if protocol != "none":
                seed = stable_seed(SEEDS["reference"], health.tolist(), p_eff.tolist())
                row["label_req"] = harvest_reference(c, health, p_eff, seed, draws=P["ref_draws"])["label"]
            row["scale"] = scale
        rows.append(row)
        st["t"] = t + 1
        return executed

    res = run_harvest_episode(cfg, [s.to_agent() for s in specs], record_trace=True, action_filter=monitor)
    unsafe = [int(r["global_unsafe"]) for r in res["episode_trace_rows"]]
    pay = np.asarray(res["final_payoffs"], float)
    return summarize("harvest", context, protocol, d, P["horizon"], pay, mis, res["mean_patch_health"],
                     unsafe, rows, ov), rows


# ------------------------------------------------------------------ Fishery
def fishery_episode(context, protocol, d, P, train=False, belief=True, sanction="excl+fine", target="one_step"):
    cfg, pol, mis = fishery_setup(context, SEEDS["train_fishery_pop"] if train else SEEDS["fishery_pop"], P["horizon"])
    ov = Overseer(protocol, cfg.n_agents, mis, d, context, "fishery", belief, sanction)
    state, pay, stocks, unsafe, rows = FisherySnapshot(cfg.stock_init), np.zeros(cfg.n_agents), [], [], []
    for t in range(cfg.horizon):
        req = fishery_requests(pol, state.stock)
        row = dict(step=t, pre_safe=int(fishery_safe(cfg, state)))
        if protocol == "none":
            executed, scale, p_eff = req.copy(), 1.0, req.copy()
        else:
            p_eff, b, caught = ov.step(t, req)
            scale, _ = fishery_choose_scale(cfg, state.stock, b, "joint", target, state.collapsed)
            executed = p_eff * scale
            row["caught"] = int(caught.sum())
        if not train:
            row["label_exec"] = fishery_reference(cfg, state.stock, executed, target)["label"]
            if protocol != "none":
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
    return summarize("fishery", context, protocol, d, cfg.horizon, pay, mis, float(np.mean(stocks)),
                     unsafe, rows, ov), rows


def summarize(game, context, protocol, d, horizon, pay, mis, mean_health, unsafe, rows, ov):
    honest = [i for i in range(len(pay)) if i not in mis]
    scored = [r for r in rows if r["pre_safe"]]
    def count(key, val):
        return sum(1 for r in scored if r.get(key) == val)
    useful_loss = sum(1 for r in scored if r.get("label_req") == "safe" and r.get("scale", 1) < 1)
    return dict(game=game, context=context, protocol=protocol, d=d,
                total_harvest=float(pay.sum()), honest_harvest=float(pay[honest].sum()),
                misreporter_harvest=float(pay[list(mis)].sum()),
                misreporter_payoff=float(pay[list(mis)].sum() - ov.fines[list(mis)].sum()),
                fines=float(ov.fines.sum()), mean_health=mean_health,
                t_end=len(rows), unsafe_fixed=(sum(unsafe) + horizon - len(unsafe)) / horizon,
                scored_steps=len(scored), exec_risky=count("label_exec", "risky"),
                exec_unresolved=count("label_exec", "unresolved"), req_safe=count("label_req", "safe"),
                useful_loss=useful_loss, audits=ov.audits, messages=ov.messages, catches=ov.catches,
                caught_honest=int(ov.catch_by_agent[honest].sum()), misreporters=list(mis))


def run_condition(game, protocol, d, P, contexts, train=False, **opts):
    fn = harvest_episode if game == "harvest" else fishery_episode
    return [fn(c, protocol, d, P, train, **opts)[0] for c in contexts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = profile(a.profile)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    episodes, search = [], []
    test_ctx, train_ctx = range(P["contexts"]), range(P["train_contexts"])
    for game, protocols in (("fishery", FISHERY_PROTOCOLS), ("harvest", HARVEST_PROTOCOLS)):
        for proto in ("none", "full"):
            for e in run_condition(game, proto, 0.0, P, test_ctx):
                e["actor"] = "fixed"; episodes.append(e)
        for proto in protocols:
            fixed = {}
            for d in FIXED_D:
                eps = run_condition(game, proto, d, P, test_ctx)
                for e in eps:
                    e["actor"] = "fixed"
                fixed[d] = eps
                episodes += eps
            # adaptive misreporting search on separate training contexts
            scores = {}
            for d in SEARCH_D:
                tr = run_condition(game, proto, d, P, train_ctx, train=True)
                scores[d] = float(np.mean([e["misreporter_payoff"] for e in tr]))
            best = SEARCH_D[0]
            for d in SEARCH_D[1:]:
                if scores[d] > scores[best] + 1e-9:
                    best = d
            search.append(dict(game=game, protocol=proto, train_scores=scores, d_star=best))
            eps = [dict(e) for e in fixed[best]] if best in fixed else run_condition(game, proto, best, P, test_ctx)
            for e in eps:
                e["actor"] = "adaptive"
            episodes += eps
            print(f"{game} {proto} done; d*={best}; {time.time()-t0:.0f}s", flush=True)
    write_jsonl_gz(out / "episodes.jsonl.gz", episodes)
    (out / "adaptive_search.json").write_text(json.dumps(search, indent=1))
    manifest = dict(experiment="S1", profile=a.profile, params=P, seeds=SEEDS, seconds=time.time() - t0,
                    n_episodes=len(episodes), exclusion_steps=EXCLUSION_STEPS, fine=FINE,
                    files={f.name: sha256(f) for f in out.glob("*.gz")} | {"adaptive_search.json": sha256(out / "adaptive_search.json")},
                    source={p: sha256(Path(p)) for p in ["fishery_sim/calibrated_oversight.py",
                            "experiments/oversight/claude_oversight_common.py", "experiments/oversight/run_s1_reporting_audit.py"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(dict(n_episodes=len(episodes), seconds=manifest["seconds"])))


if __name__ == "__main__":
    main()
