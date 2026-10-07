"""Experiment S5: stronger attackers, and memory rules that close S4's gap (Claude audit, October 2026).

Protocol: notes/claude_audit_20261005/studies/S5_stronger_attackers/protocol.md

S2-S4 non-compliance model in Fishery (MSY target, 6 agents, 4 stress agents who may over-take, 80 steps).
Bernoulli audits with probability q per agent per step and perfect detection. Six reviewer regimes:
none, fine, memory (S4 targeted allowance), memory_cut (new: the shared cut accounts for caught agents' mean
over-take), memory_cap (new: after its first catch an agent is checked every step, so its take is enforced at its
allowance), fine+memory_cut. Three attacker tiers: T0 (S4's shared level + reaction), T1 (per-cheater,
state-dependent levels plus a lie-low length, iterated best response), T2 (shared PPO policy, see s5_ppo.py).

Run:  PYTHONPATH=. python -m experiments.oversight.run_s5_stronger_attackers --part T0T1|T2 --profile smoke|full --out DIR
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

from experiments.oversight import run_s2_compliance_deterrence as s2
from experiments.oversight.claude_oversight_common import fishery_requests, fishery_setup, sha256, write_jsonl_gz
from fishery_sim.calibrated_oversight import (
    SCALES, fishery_choose_scale, fishery_reference, fishery_target_ok, stable_seed,
)
from fishery_sim.fishery_oversight import FisherySnapshot, safe as fishery_safe

# Protocol seeds. 'detect' is never drawn (detection is perfect, s = 1); 'train_audit_tag' gives training
# contexts their own audit stream (see notes in the manifest). 'ppo' seeds the T2 runs (one per cell).
SEEDS = dict(fishery_pop=1_300_000_000, train_fishery_pop=1_350_000_000, audit=1_345_000_000,
             detect=1_346_000_000, train_audit_tag="train", ppo=1_347_000_000, ppo_eval=1_348_000_000)
FINE = 6.0
TARGET = "msy"
REGIMES = ("fine", "memory", "memory_cut", "memory_cap", "fine+memory_cut")
RATES = (0.02, 0.05, 0.10, 1 / 6)
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
REACTIONS = ("continue", "stop")
LIE_LOW = (0, 2, 5)
STOCK_SPLIT = 60.0
T1_ROUNDS = 3
T1_SWEEPS = 3  # max coordinate-ascent sweeps per cheater turn
T1_KEYS = ("hi_new", "hi_caught", "lo_new", "lo_caught")  # level index = 2 * (stock < 60) + caught_before
T2_CELLS = (("fine", 0.05), ("memory", 1 / 6), ("memory_cap", 1 / 6), ("fine+memory_cut", 0.05))
EPS = 1e-9


def profile(name):
    if name == "smoke":
        return dict(contexts=4, train_contexts=4, horizon=80, ppo_episodes=400, ppo_batch=40, ppo_eval_every=2)
    if name == "full":
        return dict(contexts=64, train_contexts=32, horizon=80, ppo_episodes=6000, ppo_batch=40, ppo_eval_every=5)
    raise ValueError(name)


# ================================================================== environment helpers
def fast_transition(cfg, state, requests):
    """Bit-identical to fishery_sim.fishery_oversight.transition for the S2-S5 Fishery config (no quota, no noise).

    It avoids building a FisheryEnv (and its RNG) every step. Equality is checked in tests/test_s5_attackers.py.
    """
    requests = np.asarray(requests, dtype=float)
    if np.any((requests < 0) | (requests > 1)):
        raise ValueError("Expected normalized requests")
    n = cfg.n_agents
    if state.collapsed:
        return FisherySnapshot(state.stock, state.below_count, True), np.zeros(n), 0.0
    maxh = float(cfg.max_harvest_per_agent)
    requested = np.clip(np.asarray(requests * cfg.max_harvest_per_agent, dtype=float), 0.0, maxh)
    harvests = requested.copy()
    fines = np.zeros(n)
    total = harvests.sum()
    stock = state.stock
    if total > stock and total > 0:
        harvests *= stock / total
        total = harvests.sum()
    payoffs = np.maximum(0.0, harvests - fines)
    remaining = max(0.0, stock - total)
    stock_max, regen = float(cfg.stock_max), float(cfg.regen_rate)
    growth = regen * remaining * (1.0 - remaining / stock_max)
    new = float(np.clip(remaining + max(0.0, growth), 0.0, stock_max))
    below = state.below_count + 1 if new < float(cfg.collapse_threshold) else 0
    collapsed = False
    if below >= int(cfg.collapse_patience):
        collapsed, new = True, 0.0
    harvest = float(total)
    return FisherySnapshot(new, below, collapsed), payoffs, harvest


_SETUP, _AUDIT = {}, {}


def setup(context, pop_base, horizon):
    key = (context, pop_base, horizon)
    if key not in _SETUP:
        _SETUP[key] = fishery_setup(context, pop_base, horizon)
    return _SETUP[key]


def audit_draws(seeds, context, t, q, n, train):
    """Uniform draws for the Bernoulli audits; same derivation as S3/S4 on test contexts (and with S4's seeds)."""
    tag = seeds.get("train_audit_tag") if train else None
    key = (seeds["audit"], tag, context, t, round(q, 6), n)
    if key not in _AUDIT:
        parts = (context, t, round(q, 6)) if tag is None else (tag, context, t, round(q, 6))
        _AUDIT[key] = np.random.default_rng(stable_seed(seeds["audit"], *parts)).random(n)
    return _AUDIT[key]


def targeted_allowance(scale, dbar, p):
    """S4's targeted allowance (S4 protocol)."""
    if dbar >= 1 - 1e-12:
        return 0.0
    return max(0.0, (scale - dbar) / (1 - dbar)) * p


def observed_overtake(taken, allowance, p):
    return (taken - allowance) / (p - allowance) if p - allowance > 1e-12 else None


def memory_cut_scale(cfg, stock, requests, dbar, flagged, target=TARGET, collapsed=False):
    """memory_cut: largest scale in SCALES whose *expected* total take meets the target.

    Unflagged agent i takes a_i = scale * p_i; a flagged agent takes a_i + dbar_i (p_i - a_i).
    With no flagged agent this is exactly fishery_choose_scale (joint reviewer).
    """
    flagged = np.asarray(flagged, bool)
    if not flagged.any():
        return fishery_choose_scale(cfg, stock, requests, "joint", target, collapsed)
    if collapsed:
        return 0.0, len(SCALES)
    p = np.asarray(requests, float)
    d = np.where(flagged, np.asarray(dbar, float), 0.0)
    for count, scale in enumerate(SCALES, start=1):
        a = p * scale
        expected = np.where(flagged, a + d * (p - a), a)
        if fishery_target_ok(cfg, stock, float((expected * cfg.max_harvest_per_agent).sum()), target):
            return scale, count
    return 0.0, len(SCALES)


# ================================================================== reviewer
class Reviewer:
    """Bernoulli auditor with the S5 regimes. Attribute names match s2.summarize."""

    def __init__(self, regime, q, n, cheaters, context, seeds, train):
        if regime not in ("none",) + REGIMES:
            raise ValueError(regime)
        self.regime, self.q, self.n, self.context, self.seeds, self.train = regime, q, n, context, seeds, train
        self.audited_regime = regime != "none"
        self.protocol = "bern" if self.audited_regime else "allow"
        self.fine = FINE if regime.startswith("fine") else 0.0
        self.memory = regime == "memory"
        self.cut = regime.endswith("memory_cut")
        self.cap = regime == "memory_cap"
        self.cheat = list(cheaters)
        self.record = [[] for _ in range(n)]
        self._dbar = np.zeros(n)  # float(np.mean(record[i])), refreshed on every new record
        self._flagged = np.zeros(n, bool)
        self.capped = np.zeros(n, bool)
        self.fines = np.zeros(n)
        self.over_steps = np.zeros(n, int)
        self.audits = self.messages = self.catches = self.caught_honest = self.enforced_checks = 0

    def choose_scale(self, cfg, stock, req, collapsed):
        if self.cut:
            return memory_cut_scale(cfg, stock, req, self._dbar, self._flagged, TARGET, collapsed)[0]
        return fishery_choose_scale(cfg, stock, req, "joint", TARGET, collapsed)[0]

    def allowances(self, scale, p):
        a = scale * p
        if self.memory:
            for i in np.flatnonzero(self._flagged):
                a[i] = targeted_allowance(scale, self._dbar[i], p[i])
        return a

    def execute(self, p, allowance, levels):
        """levels: per-agent over-take level (0 for honest agents). Capped agents are held at their allowance."""
        taken = allowance.copy()
        for i in self.cheat:
            if not self.capped[i] and levels[i] != 0.0:
                taken[i] = allowance[i] + levels[i] * (p[i] - allowance[i])
        self.over_steps += taken > allowance + 1e-12
        return taken

    def audit(self, t, p, allowance, taken):
        """Ex-post check after extraction. Returns (caught, audited) boolean arrays."""
        caught = np.zeros(self.n, bool)
        if not self.audited_regime:
            return caught, np.zeros(self.n, bool)
        audited = audit_draws(self.seeds, self.context, t, self.q, self.n, self.train) < self.q
        if self.cap and self.capped.any():
            self.enforced_checks += int(self.capped.sum())
            audited = audited | self.capped  # a capped agent is checked every step: one audit, never two
        over = taken > allowance + 1e-12
        caught = audited & over
        self.audits += int(audited.sum())
        for i in np.flatnonzero(caught):
            self.catches += 1
            self.fines[i] += self.fine
            self.caught_honest += int(i not in self.cheat)
            if self.memory or self.cut:
                d_obs = observed_overtake(taken[i], allowance[i], p[i])
                if d_obs is not None:
                    self.record[i].append(d_obs)
                    self._dbar[i] = float(np.mean(self.record[i]))
                    self._flagged[i] = True
        if self.cap:
            self.capped |= caught  # enforcement starts at the next step
        return caught, audited


# ================================================================== episode
class Obs:
    __slots__ = ("t", "horizon", "stock", "stock_max", "req", "allowance", "since", "caught", "capped")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


NEVER = 10 ** 6


class Episode:
    """One Fishery episode, stepped by an attacker policy that sets each cheater's level.

    Cheaters are indexed by rank k = position in the context's sorted list of stress agents.
    """

    def __init__(self, context, regime, q, P, seeds=SEEDS, train=False, trace=False):
        self.context, self.regime, self.q, self.train = context, regime, q, train
        self.cfg, pol, cheat = setup(context, seeds["train_fishery_pop"] if train else seeds["fishery_pop"], P["horizon"])
        self.pol, self.cheat = pol, list(cheat)
        n = self.cfg.n_agents
        self.rv = Reviewer(regime, q, n, cheat, context, seeds, train)
        self.state = FisherySnapshot(self.cfg.stock_init)
        self.pay = np.zeros(n)
        self.stocks, self.unsafe, self.rows = [], [], []
        self.last_audit = np.full(n, -NEVER)
        self.ever_caught = np.zeros(n, bool)
        self.t, self.done = 0, False
        self.trace = [] if trace else None
        self.level_sum = np.zeros(len(self.cheat))

    def observe(self):
        cfg, st = self.cfg, self.state
        self._req = fishery_requests(self.pol, st.stock)
        self._row = dict(step=self.t, pre_safe=int(fishery_safe(cfg, st)))
        self._scale = self.rv.choose_scale(cfg, st.stock, self._req, st.collapsed)
        self._allow = self.rv.allowances(self._scale, self._req)
        c = self.cheat
        return Obs(t=self.t, horizon=cfg.horizon, stock=st.stock, stock_max=cfg.stock_max, req=self._req[c],
                   allowance=self._allow[c], since=self.t - self.last_audit[c], caught=self.ever_caught[c].copy(),
                   capped=self.rv.capped[c].copy())

    def step(self, cheater_levels):
        """Apply the cheaters' levels for the current step. Returns each cheater's reward (payoff minus fine)."""
        cfg, rv = self.cfg, self.rv
        levels = np.zeros(cfg.n_agents)
        levels[self.cheat] = cheater_levels
        self.level_sum += np.asarray(cheater_levels, float)
        req, allowance = self._req, self._allow
        taken = rv.execute(req, allowance, levels)
        capped_before = rv.capped.copy()
        caught, audited = rv.audit(self.t, req, allowance, taken)
        self.last_audit[audited] = self.t
        self.ever_caught |= caught
        row = self._row
        if not self.train:
            row["label_exec"] = fishery_reference(cfg, self.state.stock, taken, TARGET)["label"]
        self.rows.append(row)
        if self.trace is not None:
            self.trace.append(dict(t=self.t, stock=self.state.stock, scale=self._scale, req=req.tolist(),
                                   allowance=allowance.tolist(), taken=taken.tolist(), audited=audited.tolist(),
                                   caught=caught.tolist(), capped_before=capped_before.tolist()))
        future, payoffs, _ = fast_transition(cfg, self.state, taken)
        self.pay += payoffs
        self.stocks.append(future.stock)
        self.unsafe.append(int(not fishery_safe(cfg, future)))
        self.state = future
        self.t += 1
        if self.state.collapsed or self.t >= cfg.horizon:
            self.done = True
        return payoffs[self.cheat] - rv.fine * caught[self.cheat]

    def summary(self):
        rv, cheat = self.rv, self.cheat
        e = s2.summarize("fishery", self.context, rv.protocol, None, rv.fine, self.cfg.horizon, self.pay, cheat,
                         float(np.mean(self.stocks)), self.unsafe, self.rows, rv)
        e.update(regime=self.regime, q=self.q, cheater_payoffs=(self.pay[cheat] - rv.fines[cheat]).tolist(),
                 n_honest=self.cfg.n_agents - len(cheat), cheater_over_steps=float(rv.over_steps[cheat].mean()),
                 flagged=int(sum(bool(r) for r in rv.record)), capped=int(rv.capped.sum()),
                 enforced_checks=rv.enforced_checks, mean_level=float(self.level_sum.mean() / max(self.t, 1)))
        return e


def run_episode(context, regime, q, policy, P, seeds=SEEDS, train=False, trace=False):
    ep = Episode(context, regime, q, P, seeds, train, trace)
    while not ep.done:
        ep.step(policy.act(ep.observe()))
    e = ep.summary()
    if trace:
        e["trace"] = ep.trace
    return e


# ================================================================== attackers
class T0Policy:
    """Shared level d for every cheater; 'stop' complies after the cheater's own first catch (S4)."""

    def __init__(self, d, reaction="continue"):
        self.d, self.reaction = float(d), reaction

    def act(self, obs):
        lv = np.full(len(obs.req), self.d)
        if self.reaction == "stop":
            lv[obs.caught] = 0.0
        return lv

    def describe(self):
        return dict(tier="T0", d=self.d, reaction=self.reaction)


class T1Policy:
    """Per-cheater levels for (stock >= 60 or < 60) x (caught before or not), plus lie-low L after own audits."""

    def __init__(self, levels, lie_low):
        self.levels = np.asarray(levels, float).reshape(-1, 4)
        self.L = np.asarray(lie_low, int)

    @classmethod
    def from_t0(cls, d, reaction, n_cheat=4):
        caught = 0.0 if reaction == "stop" else d
        return cls([[d, caught, d, caught]] * n_cheat, [0] * n_cheat)

    def params(self):
        return tuple(tuple(float(x) for x in row) for row in self.levels), tuple(int(x) for x in self.L)

    def act(self, obs):
        idx = 2 * int(obs.stock < STOCK_SPLIT) + obs.caught.astype(int)
        lv = self.levels[np.arange(len(idx)), idx]
        low = (obs.since >= 1) & (obs.since <= self.L)
        return np.where(low, 0.0, lv)

    def describe(self):
        return dict(tier="T1", levels=[dict(zip(T1_KEYS, map(float, r))) for r in self.levels],
                    lie_low=[int(x) for x in self.L])


def evaluate(regime, q, policy, P, contexts, seeds=SEEDS, train=True):
    return [run_episode(c, regime, q, policy, P, seeds, train) for c in contexts]


def search_t0(regime, q, P, seeds=SEEDS, train_contexts=None):
    """S4's search: maximise mean group cheater payoff on training contexts; ties -> smaller d, then 'continue'."""
    ctx = range(P["train_contexts"]) if train_contexts is None else train_contexts
    best, best_score, scores = None, -np.inf, {}
    for d in LEVELS:
        for reaction in REACTIONS:
            sc = float(np.mean([e["cheater_payoff"] for e in evaluate(regime, q, T0Policy(d, reaction), P, ctx, seeds)]))
            scores[f"{d}:{reaction}"] = sc
            if best is None or sc > best_score + EPS:
                best, best_score = (d, reaction), sc
    return best, best_score, scores


def search_t1(regime, q, d0, r0, P, seeds=SEEDS):
    """Iterated best response from the T0 solution. Each cheater (by rank) maximises its OWN mean net payoff
    on the training contexts by coordinate ascent over its 4 levels and L, others held fixed.
    A coordinate moves only on a strict improvement (> 1e-9); among improving values the best wins, ties to the
    smaller value. Up to T1_SWEEPS sweeps per turn; rounds stop after T1_ROUNDS or when a round changes nothing."""
    ctx = range(P["train_contexts"])
    cache = {}

    def score(levels, lie):
        key = (tuple(map(tuple, levels)), tuple(lie))
        if key not in cache:
            eps = evaluate(regime, q, T1Policy(levels, lie), P, ctx, seeds)
            cache[key] = (np.mean([e["cheater_payoffs"] for e in eps], axis=0), float(np.mean([e["cheater_payoff"] for e in eps])))
        return cache[key]

    start = T1Policy.from_t0(d0, r0)
    levels, lie = [list(r) for r in start.levels.tolist()], [int(x) for x in start.L]
    n_cheat = len(levels)
    trace = [dict(round=0, levels=[list(r) for r in levels], lie_low=list(lie), own=score(levels, lie)[0].tolist(),
                  group=score(levels, lie)[1])]
    rounds_done = 0
    for rnd in range(1, T1_ROUNDS + 1):
        changed_round = False
        for k in range(n_cheat):
            for _ in range(T1_SWEEPS):
                changed = False
                for coord in range(5):
                    values = LEVELS if coord < 4 else LIE_LOW
                    cur = levels[k][coord] if coord < 4 else lie[k]
                    cur_sc = score(levels, lie)[0][k]
                    best_v, best_sc = cur, cur_sc
                    for v in values:
                        if v == cur:
                            continue
                        lv = [list(r) for r in levels]
                        li = list(lie)
                        if coord < 4:
                            lv[k][coord] = float(v)
                        else:
                            li[k] = int(v)
                        sc = score(lv, li)[0][k]
                        if sc > cur_sc + EPS and sc > best_sc + EPS:
                            best_v, best_sc = v, sc
                    if best_v != cur:
                        changed = True
                        if coord < 4:
                            levels[k][coord] = float(best_v)
                        else:
                            lie[k] = int(best_v)
                if not changed:
                    break
                changed_round = True
            trace.append(dict(round=rnd, cheater=k, levels=[list(r) for r in levels], lie_low=list(lie),
                              own=score(levels, lie)[0].tolist(), group=score(levels, lie)[1]))
        rounds_done = rnd
        if not changed_round:
            break
    own, group = score(levels, lie)
    return T1Policy(levels, lie), dict(rounds=rounds_done, evaluations=len(cache), train_own=own.tolist(),
                                       train_group=group, trace=trace)


# ================================================================== parts
def cells():
    return [("none", None)] + [(r, q) for r in REGIMES for q in RATES]


def part_t0t1(P, out=None, log=print):
    test, train = range(P["contexts"]), range(P["train_contexts"])
    episodes, train_eps, searches = [], [], []
    t0 = time.time()
    all_cells = cells()
    for n_cell, (regime, q) in enumerate(all_cells, start=1):
        tag = f"[{n_cell}/{len(all_cells)}] {regime} q={q}"
        comply = T0Policy(0.0)
        for split, ctx, sink in (("test", test, episodes), ("train", train, train_eps)):
            for e in evaluate(regime, q, comply, P, ctx, train=split == "train"):
                sink.append({**e, "tier": "comply", "split": split})
        (d, r), sc0, scores = search_t0(regime, q, P)
        log(f"T0 {tag} d*={d} reaction={r} train_group={sc0:.2f} {time.time() - t0:.0f}s", flush=True)
        t0p = T0Policy(d, r)
        t1p, info = search_t1(regime, q, d, r, P)
        searches.append(dict(regime=regime, q=q, T0=dict(d_star=d, reaction=r, train_group=sc0, train_scores=scores),
                             T1=dict(**t1p.describe(), **info)))
        for tier, pol in (("T0", t0p), ("T1", t1p)):
            for split, ctx, sink in (("test", test, episodes), ("train", train, train_eps)):
                for e in evaluate(regime, q, pol, P, ctx, train=split == "train"):
                    sink.append({**e, "tier": tier, "split": split})
        log(f"T1 {tag} rounds={info['rounds']} evals={info['evaluations']} train_group={info['train_group']:.2f} "
            f"levels={t1p.levels.tolist()} L={t1p.L.tolist()} {time.time() - t0:.0f}s", flush=True)
        if out is not None:  # progress snapshot (not part of the manifest; the final files are written at the end)
            (out / "searches_partial.json").write_text(json.dumps(searches, indent=1))
    if out is not None and (out / "searches_partial.json").exists():
        (out / "searches_partial.json").unlink()
    return episodes, train_eps, searches


def part_t2(P, out, log=print):
    from experiments.oversight import s5_ppo
    test, train = range(P["contexts"]), range(P["train_contexts"])
    episodes, train_eps, logs = [], [], []
    hp = dict(s5_ppo.HYPERPARAMS)
    t0 = time.time()
    (out / "policies").mkdir(exist_ok=True)
    for idx, (regime, q) in enumerate(T2_CELLS):
        seed = SEEDS["ppo"] + idx
        res = s5_ppo.train(regime, q, P, hp, seed, log=log)
        pol = s5_ppo.TrainedPolicy(res["best_state"], hp)
        for split, ctx, sink in (("test", test, episodes), ("train", train, train_eps)):
            for e in s5_ppo.evaluate_lockstep(regime, q, pol, P, ctx, train=split == "train"):
                sink.append({**e, "tier": "T2", "split": split})
        name = f"{regime}_q{q:.4f}"
        s5_ppo.save_state(res["best_state"], out / "policies" / f"{name}.json.gz")
        logs.append(dict(regime=regime, q=q, seed=seed, best_iteration=res["best_iteration"],
                         best_train_group=res["best_score"], episodes_used=res["episodes_used"], log=res["log"]))
        log(f"T2 [{idx + 1}/{len(T2_CELLS)}] {regime} q={q} best_it={res['best_iteration']} train={res['best_score']:.3f} {time.time() - t0:.0f}s",
            flush=True)
    return episodes, train_eps, logs, hp


def write_json_gz(path, obj):
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        f.write(json.dumps(obj, indent=1).encode())


SOURCES = ["fishery_sim/calibrated_oversight.py", "fishery_sim/fishery_oversight.py", "fishery_sim/env.py",
           "experiments/oversight/claude_oversight_common.py", "experiments/oversight/run_s2_compliance_deterrence.py",
           "experiments/oversight/run_s5_stronger_attackers.py", "experiments/oversight/s5_ppo.py"]

DESIGN_NOTES = [
    "Cheater rank k = position of the stress agent in the context's sorted list; T1 parameters are indexed by rank.",
    "Training contexts use their own audit stream (stable_seed(audit, 'train', context, t, q)); test contexts use the "
    "S3/S4 derivation stable_seed(audit, context, t, q). Draws are shared across regimes and tiers (paired).",
    "Detection is perfect (s = 1), so the detection RNG is never drawn.",
    "memory_cut: flagged agents get the plain allowance scale*p; only the shared scale accounts for dbar.",
    "memory_cap: from the step after its first catch, an agent is checked every step and its take is held at its "
    "allowance scale*p. Each such step counts as one audit (its Bernoulli draw is not counted again). No over-take is "
    "possible, so a capped agent is never caught or fined again (the regime has no fine anyway).",
    "An agent observes every audit of itself (caught or not); lie-low covers the L steps after the audit step.",
    "T1 is iterated best response on each cheater's own net payoff (not the group payoff that T0 maximises).",
    "T2 is evaluated as the stochastic policy PPO optimises: levels are sampled with one RNG per episode, seeded by "
    "stable_seed(ppo_eval, split, context), on training and test contexts (checkpoint choice uses training only).",
    "T0 maximises the group's mean cheater payoff; T1 and T2 maximise each cheater's own payoff. Cheater gain (the "
    "outcome) is the group sum; per-rank individual gains are reported as exploratory.",
    "Cheater gain = group cheater net payoff minus the same contexts' complying payoff under the same regime and q.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["T0T1", "T2"])
    ap.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = profile(a.profile)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    extra = {}
    if a.part == "T0T1":
        episodes, train_eps, searches = part_t0t1(P, out)
        (out / "searches.json").write_text(json.dumps(searches, indent=1))
        files = ["episodes.jsonl.gz", "train_episodes.jsonl.gz", "searches.json"]
    else:
        import torch
        torch.set_num_threads(1)
        episodes, train_eps, logs, hp = part_t2(P, out)
        write_json_gz(out / "training_log.json.gz", logs)
        extra = dict(ppo_hyperparameters=hp, t2_cells=T2_CELLS, torch_version=torch.__version__)
        files = ["episodes.jsonl.gz", "train_episodes.jsonl.gz", "training_log.json.gz"] + \
                sorted(f"policies/{p.name}" for p in (out / "policies").iterdir())
    write_jsonl_gz(out / "episodes.jsonl.gz", episodes)
    write_jsonl_gz(out / "train_episodes.jsonl.gz", train_eps)
    manifest = dict(experiment=f"S5-{a.part}", profile=a.profile, params=P, seeds=SEEDS, fine=FINE, target=TARGET,
                    regimes=REGIMES, rates=RATES, levels=LEVELS, lie_low=LIE_LOW, stock_split=STOCK_SPLIT,
                    t1_rounds=T1_ROUNDS, t1_sweeps=T1_SWEEPS, n_episodes=len(episodes), n_train_episodes=len(train_eps),
                    design_notes=DESIGN_NOTES, **extra,
                    files={f: sha256(out / f) for f in files},
                    source={p: sha256(Path(p)) for p in SOURCES})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "timing.json").write_text(json.dumps(dict(seconds=time.time() - t0)))  # kept out of the manifest
    print(json.dumps(dict(n_episodes=len(episodes), seconds=time.time() - t0)), flush=True)


if __name__ == "__main__":
    main()
