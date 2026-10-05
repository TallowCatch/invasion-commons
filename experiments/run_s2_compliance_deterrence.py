"""Experiment S2: allowances, hidden non-compliance, ex-post audits and fines (Claude audit, October 2026).

Protocol: notes/claude_audit_20261005/11_PROTOCOL_S1B_S2_ABLATION_AND_DETERRENCE.md (Part C)

Each step: agents report their requests truthfully; the reviewer (calibrated joint chance constraint)
sets allowances a_i = scale * r_i assuming compliance. Honest agents take a_i. A non-complying agent with
level d takes a_i + d * (p_i - a_i). Audits check what was actually taken *after* extraction; a caught agent
pays fine F (no exclusion). Audits never change the current decision.
Run:  PYTHONPATH=. python -m experiments.run_s2_compliance_deterrence --profile smoke|full --out DIR
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
from fishery_sim.calibrated_oversight import (
    fishery_choose_scale, fishery_reference, harvest_choose_scale, harvest_reference, stable_seed,
)
from fishery_sim.fishery_oversight import FisherySnapshot, safe as fishery_safe, transition
from fishery_sim.harvest import run_harvest_episode
from fishery_sim.oversight_protocol import harvest_nominal_next, harvest_safe

SEEDS = dict(fishery_pop=800_000_000, harvest_pop=810_000_000, weather=820_000_000,
             reviewer=830_000_000, reference=840_000_000, audit=845_000_000,
             train_fishery_pop=850_000_000, train_harvest_pop=860_000_000, train_weather=870_000_000)
FINES = (0.0, 6.0, 12.0, 24.0)
SEARCH_D = (0.0, 0.25, 0.5, 0.75, 1.0)
PROTOCOLS = {"harvest": ("rand1", "rand2", "targ1", "peer", "peer_collude"),
             "fishery": ("rand1", "rand2", "peer", "peer_collude")}
FISHERY_TARGET = "msy"


def profile(name):
    if name == "smoke":
        return dict(contexts=2, train_contexts=2, horizon=20, reviewer_draws=400, ref_draws=2000)
    if name == "full":
        return dict(contexts=64, train_contexts=8, horizon=80, reviewer_draws=400, ref_draws=2000)
    raise ValueError(name)


class Auditor:
    def __init__(self, protocol, n, cheaters, d, fine, context, game):
        self.protocol, self.n, self.d, self.fine = protocol, n, d, fine
        self.cheat = set(cheaters)
        self.context, self.game = context, game
        self.suspicion = np.zeros(n)
        self.fines = np.zeros(n)
        self.audits = self.messages = self.catches = 0
        self.caught_honest = 0

    def execute(self, p, allowance):
        taken = allowance.copy()
        for i in self.cheat:
            taken[i] = allowance[i] + self.d * (p[i] - allowance[i])
        return taken

    def audit(self, t, allowance, taken):
        """Ex-post check of what was actually taken; fines caught agents."""
        over = taken > allowance + 1e-12
        caught = np.zeros(self.n, bool)
        rng = np.random.default_rng(stable_seed(SEEDS["audit"], self.game, self.context, t, self.protocol))
        if self.protocol in ("allow", "none"):
            return caught
        if self.protocol.startswith("rand"):
            idx = rng.permutation(self.n)[:int(self.protocol[-1])]
            self.audits += len(idx); caught[idx] = over[idx]
        elif self.protocol.startswith("targ"):
            jitter = rng.uniform(0, 1e-9, self.n)
            idx = np.argsort(-(self.suspicion + jitter))[:int(self.protocol[-1])]
            self.audits += len(idx); caught[idx] = over[idx]
        elif self.protocol.startswith("peer"):
            self.messages += self.n
            collude = self.protocol == "peer_collude"
            for i in range(self.n):
                m = (i + 1) % self.n
                if collude and i in self.cheat and m in self.cheat:
                    continue
                caught[i] = over[i]
        else:
            raise ValueError(self.protocol)
        for i in np.flatnonzero(caught):
            self.catches += 1
            self.fines[i] += self.fine
            self.caught_honest += int(i not in self.cheat)
        return caught


def harvest_episode(context, protocol, d, fine, P, train=False):
    cfg, specs, cheat = harvest_setup(context, SEEDS["train_harvest_pop"] if train else SEEDS["harvest_pop"],
                                      SEEDS["train_weather"] if train else SEEDS["weather"], P["horizon"])
    au = Auditor(protocol, cfg.n_agents, cheat, d, fine, context, "harvest")
    st, rows = dict(t=0, pred=None), []

    def monitor(c, health, requests):
        t = st["t"]
        if st["pred"] is not None:
            au.suspicion = 0.8 * au.suspicion + (st["pred"] - health)
        row = dict(step=t, pre_safe=int(harvest_safe(c, health)))
        if protocol == "none":
            allowance, scale = requests.copy(), 1.0
        else:
            scale, _, _ = harvest_choose_scale(c, health, requests, "joint",
                                               stable_seed(SEEDS["reviewer"], "h", train, context, t),
                                               draws=P["reviewer_draws"])
            allowance = requests * scale
        taken = au.execute(requests, allowance) if protocol != "none" else allowance
        st["pred"] = harvest_nominal_next(c, health, allowance)
        au.audit(t, allowance, taken)
        if not train:
            row["label_exec"] = harvest_reference(c, health, taken, stable_seed(SEEDS["reference"], health.tolist(),
                                                  taken.tolist()), draws=P["ref_draws"])["label"]
        rows.append(row)
        st["t"] = t + 1
        return taken

    res = run_harvest_episode(cfg, [s.to_agent() for s in specs], record_trace=True, action_filter=monitor)
    unsafe = [int(r["global_unsafe"]) for r in res["episode_trace_rows"]]
    return summarize("harvest", context, protocol, d, fine, P["horizon"], np.asarray(res["final_payoffs"], float),
                     cheat, res["mean_patch_health"], unsafe, rows, au)


def fishery_episode(context, protocol, d, fine, P, train=False):
    cfg, pol, cheat = fishery_setup(context, SEEDS["train_fishery_pop"] if train else SEEDS["fishery_pop"], P["horizon"])
    au = Auditor(protocol, cfg.n_agents, cheat, d, fine, context, "fishery")
    state, pay, stocks, unsafe, rows = FisherySnapshot(cfg.stock_init), np.zeros(cfg.n_agents), [], [], []
    for t in range(cfg.horizon):
        req = fishery_requests(pol, state.stock)
        row = dict(step=t, pre_safe=int(fishery_safe(cfg, state)))
        if protocol == "none":
            allowance = req.copy()
        else:
            scale, _ = fishery_choose_scale(cfg, state.stock, req, "joint", FISHERY_TARGET, state.collapsed)
            allowance = req * scale
        taken = au.execute(req, allowance) if protocol != "none" else allowance
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
    return summarize("fishery", context, protocol, d, fine, cfg.horizon, pay, cheat, float(np.mean(stocks)),
                     unsafe, rows, au)


def summarize(game, context, protocol, d, fine, horizon, pay, cheat, mean_health, unsafe, rows, au):
    cheat = list(cheat)
    honest = [i for i in range(len(pay)) if i not in cheat]
    scored = [r for r in rows if r["pre_safe"]]
    return dict(game=game, context=context, protocol=protocol, d=d, fine=fine,
                total_harvest=float(pay.sum()), honest_harvest=float(pay[honest].sum()),
                cheater_harvest=float(pay[cheat].sum()), cheater_payoff=float(pay[cheat].sum() - au.fines[cheat].sum()),
                fines=float(au.fines.sum()), mean_health=mean_health, t_end=len(rows),
                unsafe_fixed=(sum(unsafe) + horizon - len(unsafe)) / horizon, scored_steps=len(scored),
                exec_risky=sum(1 for r in scored if r.get("label_exec") == "risky"),
                exec_unresolved=sum(1 for r in scored if r.get("label_exec") == "unresolved"),
                audits=au.audits, messages=au.messages, catches=au.catches, caught_honest=au.caught_honest)


def run(game, protocol, d, fine, P, contexts, train=False):
    fn = harvest_episode if game == "harvest" else fishery_episode
    return [fn(c, protocol, d, fine, P, train) for c in contexts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = profile(a.profile)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0, episodes, searches = time.time(), [], []
    test, train = range(P["contexts"]), range(P["train_contexts"])
    for game in ("fishery", "harvest"):
        for e in run(game, "none", 0.0, 0.0, P, test):
            episodes.append({**e, "actor": "none"})
        for e in run(game, "allow", 0.0, 0.0, P, test):  # everyone complies: identical for every protocol
            episodes.append({**e, "actor": "comply"})
        cells = [("allow", 0.0)] + [(p, f) for p in PROTOCOLS[game] for f in FINES]
        for proto, fine in cells:
            scores = {}
            for d in SEARCH_D:
                scores[d] = float(np.mean([e["cheater_payoff"] for e in run(game, proto, d, fine, P, train, True)]))
            best = SEARCH_D[0]
            for d in SEARCH_D[1:]:
                if scores[d] > scores[best] + 1e-9:
                    best = d
            searches.append(dict(game=game, protocol=proto, fine=fine, d_star=best, train_scores=scores))
            full_cheat = run(game, proto, 1.0, fine, P, test)
            for e in full_cheat:
                episodes.append({**e, "actor": "d1"})
            adaptive = [dict(e) for e in full_cheat] if best == 1.0 else run(game, proto, best, fine, P, test)
            for e in adaptive:
                episodes.append({**e, "actor": "adaptive"})
            print(f"{game} {proto} F={fine} d*={best} {time.time()-t0:.0f}s", flush=True)
    write_jsonl_gz(out / "episodes.jsonl.gz", episodes)
    (out / "adaptive_search.json").write_text(json.dumps(searches, indent=1))
    manifest = dict(experiment="S2", profile=a.profile, params=P, seeds=SEEDS, fines=FINES, fishery_target=FISHERY_TARGET,
                    seconds=time.time() - t0, n_episodes=len(episodes),
                    files={"episodes.jsonl.gz": sha256(out / "episodes.jsonl.gz"),
                           "adaptive_search.json": sha256(out / "adaptive_search.json")},
                    source={p: sha256(Path(p)) for p in ["fishery_sim/calibrated_oversight.py",
                            "experiments/claude_oversight_common.py", "experiments/run_s2_compliance_deterrence.py"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(dict(n_episodes=len(episodes), seconds=manifest["seconds"])))


if __name__ == "__main__":
    main()
