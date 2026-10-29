"""Experiment R2: a predeclared settings grid (Part A) and a reviewer whose model is wrong (Part B).

Claude audit, October 2026. Protocol (frozen before this code):
notes/claude_audit_20261005/studies/R2_robustness_and_reviewer_model/protocol.md

Part A reruns, in every cell of the declared grid,
  1. R1 open loop at k = 6 (joint, local_bounded, local_optimistic);
  2. R1 closed loop, joint reviewer, k in {0, 3, 6}, fill = previous;
  3. S3 Part D memory (trust / memoryless / memory, rand2, fixed and noisy liars);
  4. (Fishery) the no-check gain G and S3 Part A's deterrence search over e in [0, 2G].
Part B runs the joint reviewer (closed loop, k = 6) and all three reviewers (open loop,
k = 6) with the reviewer's own, possibly wrong, model (fishery_sim/reviewer_models.py).

Run:  PYTHONPATH=. python -m experiments.oversight.run_r2_robustness_reviewer_model --part A|B --profile smoke|full --out DIR
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.oversight import run_s1_reporting_audit as s1
from experiments.oversight import run_s2_compliance_deterrence as s2
from experiments.oversight import run_s3_threshold_timing_memory as s3
from experiments.oversight.claude_oversight_common import fishery_requests, sha256, write_jsonl_gz
from experiments.oversight.run_r1_repaired_reviewer import believed_vector
from fishery_sim import reviewer_models as rm
from fishery_sim.calibrated_oversight import (
    REVIEWERS, fishery_choose_scale, harvest_choose_scale, harvest_reference, stable_seed,
)
from fishery_sim.config import FisheryConfig
from fishery_sim.fishery_oversight import FisherySnapshot, safe as fishery_safe
from fishery_sim.harvest import run_harvest_episode
from fishery_sim.harvest_benchmarks import make_harvest_cfg_for_scenario
from fishery_sim.harvest_evolution import adversarial_harvest_strategy, cooperative_harvest_strategy
from fishery_sim.oversight_protocol import harvest_nominal_next, harvest_safe

SEEDS = dict(fishery_pop=1_200_000_000, harvest_pop=1_210_000_000, weather=1_220_000_000,
             reviewer=1_230_000_000, reference=1_240_000_000, audit=1_245_000_000,
             train_fishery_pop=1_250_000_000,
             # not named in the protocol (design choices, see manifest): s = 1 makes detect unused
             detect=1_246_000_000, noisy=1_247_000_000)
BUDGETS = (0, 3, 6)
TARGETS = ("one_step", "msy")
OPEN_K = 6
MEMORY_PROTOCOL = "rand2"
MEMORY_MODES = ("trust", "memoryless", "memory")
LIARS = ("fixed", "noisy")
LIE = 0.5
CHEAT_LEVEL_FOR_G = 0.75
DETER_Q = 1 / 6
N_E = 11
DETER_TARGET = "msy"


def profile(name):
    if name == "smoke":
        return dict(contexts=2, horizon=20, train_contexts=2, closed_draws=2000, open_draws=4000, reviewer_draws=400)
    if name == "full":
        return dict(contexts=64, horizon=80, train_contexts=32, closed_draws=2000, open_draws=4000, reviewer_draws=400)
    raise ValueError(name)


# ================================================================== cells and setups
@dataclass(frozen=True)
class Cell:
    game: str
    n_stress: int
    level: float  # Fishery: regrowth rate r. Harvest: regrowth multiplier.

    @property
    def name(self):
        return f"{self.game}_s{self.n_stress}_{'r' if self.game == 'fishery' else 'm'}{self.level}"

    @property
    def piloted(self):
        return (self.n_stress, self.level) == ((4, 0.7) if self.game == "fishery" else (2, 0.85))

    def tag(self):
        return dict(cell=self.name, game=self.game, n_stress=self.n_stress, level=self.level, piloted=self.piloted)


PART_A_CELLS = [Cell("fishery", n, r) for n in (2, 4) for r in (0.5, 0.7, 0.9)] + \
               [Cell("harvest", n, m) for n in (2, 4) for m in (0.85, 1.0)]
PART_B_CELLS = [Cell("harvest", 2, 0.85), Cell("harvest", 4, 1.0), Cell("fishery", 4, 0.7), Cell("fishery", 2, 0.9)]


def harvest_setup_r2(context, pop_base, weather_base, horizon, n_stress, regen_mult):
    """claude_oversight_common.harvest_setup with the regrowth multiplier as a parameter (0.85 = identical)."""
    cfg = make_harvest_cfg_for_scenario("forest_co_management", horizon=horizon,
                                        seed=weather_base + 1000 * context,
                                        communication_enabled=False, side_payments_enabled=False)
    cfg.regen_rate *= regen_mult
    rng = np.random.default_rng(pop_base + context)
    stressed = sorted(rng.permutation(cfg.n_agents)[:n_stress].tolist())
    specs = [(adversarial_harvest_strategy if i in stressed else cooperative_harvest_strategy)(
        f"heldout_{i}", cfg.patch_max, rng) for i in range(cfg.n_agents)]
    return cfg, specs, stressed


def fishery_setup_r2(context, pop_base, horizon, n_stress, regen):
    """claude_oversight_common.fishery_setup with the regrowth rate as a parameter (0.7 = identical)."""
    if not 0 <= regen <= 1:
        raise ValueError("Fishery regen_rate must be in [0, 1]")
    cfg = FisheryConfig(n_agents=6, horizon=horizon, stock_init=70, stock_max=100,
                        regen_rate=regen, obs_noise_std=0, max_harvest_per_agent=6)
    rng = np.random.default_rng(pop_base + context)
    ranges = {"useful": ((.05, .20), (.20, .40)), "stress": ((.45, .65), (.75, .95))}
    stressed = sorted(rng.permutation(cfg.n_agents)[:n_stress].tolist())
    low = np.array([rng.uniform(*ranges["stress" if i in stressed else "useful"][0]) for i in range(cfg.n_agents)])
    high = np.array([rng.uniform(*ranges["stress" if i in stressed else "useful"][1]) for i in range(cfg.n_agents)])
    thresholds = rng.uniform(25, 65, cfg.n_agents)
    return cfg, dict(low=low, high=high, thresholds=thresholds), stressed


def setup(cell, context, P, seeds, train=False):
    if cell.game == "harvest":
        return harvest_setup_r2(context, seeds["harvest_pop"], seeds["weather"], P["horizon"], cell.n_stress, cell.level)
    return fishery_setup_r2(context, seeds["train_fishery_pop"] if train else seeds["fishery_pop"], P["horizon"],
                            cell.n_stress, cell.level)


# ================================================================== R1 closed loop (with a reviewer model)
def harvest_closed(cell, context, reviewer, budget, fill, P, seeds, condition="exact", keep_state=False, learn_none=False):
    """R1 harvest_episode; the reviewer predicts with its own model, labels use the true model."""
    cfg, specs, stressed = setup(cell, context, P, seeds)
    model = rm.ReviewerModel(cfg, "learned" if learn_none else condition)
    rows, st = [], dict(previous=None, step=0, last=None)

    def monitor(c, health, requests):
        t = st["step"]
        if st["last"] is not None:
            model.observe_harvest(st["last"][0], st["last"][1], health)
        row = dict(context=context, reviewer=reviewer, budget=budget, fill=fill, step=t,
                   pre_safe=int(harvest_safe(c, health)))
        if keep_state:
            row.update(state=health.tolist(), requests=requests.tolist(),
                       previous=None if st["previous"] is None else list(st["previous"]))
        if learn_none or condition == "learned":
            row["model"] = model.params()
        if reviewer == "none":
            scale = 1.0
        else:
            b = believed_vector(requests, budget, f"harvest__{context}__{t}", fill, st["previous"])
            scale, _, _ = harvest_choose_scale(
                model.config(), health, b, reviewer,
                stable_seed(seeds["reviewer"], "h", context, t, reviewer, budget, fill), draws=P["reviewer_draws"])
        ref = harvest_reference(c, health, requests, stable_seed(seeds["reference"], health.tolist(), requests.tolist()),
                                draws=P["closed_draws"])
        row.update(scale=scale, label=ref["label"], ref_risk=ref["risk"])
        rows.append(row)
        executed = requests * scale
        st["previous"], st["step"], st["last"] = requests.copy(), t + 1, (health.copy(), executed.copy())
        return executed

    result = run_harvest_episode(cfg, [s.to_agent() for s in specs], record_trace=True, action_filter=monitor)
    trace = result["episode_trace_rows"]
    unsafe = [int(r["global_unsafe"]) for r in trace]
    fixed = (sum(unsafe) + (P["horizon"] - len(trace))) / P["horizon"]
    ep = dict(context=context, reviewer=reviewer, budget=budget, fill=fill, target="one_step",
              total_harvest=result["total_welfare"], mean_health=result["mean_patch_health"],
              t_end=result["t_end"], failure=result["garden_failure_event"], unsafe_fixed=fixed, stressed=stressed)
    return ep, rows


def fishery_closed(cell, context, reviewer, budget, fill, target, P, seeds, condition="exact", keep_state=False,
                   learn_none=False):
    """R1 fishery_episode; true dynamics from reviewer_models.fishery_step (Allee only under 'allee')."""
    cfg, pol, stressed = setup(cell, context, P, seeds)
    allee = rm.env_allee(condition)
    model = rm.ReviewerModel(cfg, "learned" if learn_none else condition)
    state, prev, rows, total, stocks, unsafe = FisherySnapshot(cfg.stock_init), None, [], 0.0, [], []
    for t in range(cfg.horizon):
        req = fishery_requests(pol, state.stock)
        row = dict(context=context, reviewer=reviewer, budget=budget, fill=fill, target=target, step=t,
                   pre_safe=int(fishery_safe(cfg, state)))
        if keep_state:
            row.update(state=state.stock, requests=req.tolist(), previous=None if prev is None else prev.tolist())
        if learn_none or condition == "learned":
            row["model"] = model.params()
        if reviewer == "none":
            scale = 1.0
        else:
            b = believed_vector(req, budget, f"fishery__{context}__{t}", fill, prev)
            scale, _ = fishery_choose_scale(model.config(), state.stock, b, reviewer, target, state.collapsed)
        ref = rm.fishery_true_reference(cfg, state.stock, req, target or "one_step", allee)
        row.update(scale=scale, label=ref["label"])
        rows.append(row)
        executed = req * scale
        future, payoffs, _ = rm.fishery_step(cfg, state, executed, allee)
        model.observe_fishery(state.stock, executed, future)
        total += float(payoffs.sum())
        stocks.append(future.stock)
        unsafe.append(int(not fishery_safe(cfg, future)))
        prev, state = req, future
        if state.collapsed:
            break
    fixed = (sum(unsafe) + (cfg.horizon - len(unsafe))) / cfg.horizon
    ep = dict(context=context, reviewer=reviewer, budget=budget, fill=fill, target=target,
              total_harvest=total, mean_health=float(np.mean(stocks)), t_end=len(rows),
              failure=int(state.collapsed), unsafe_fixed=fixed, stressed=stressed)
    return ep, rows


# ================================================================== R1 open loop at k = 6
def learned_cfg(true_cfg, params):
    from dataclasses import replace
    return replace(true_cfg, **params)


def open_loop(cell, none_rows, P, seeds, conditions):
    """Score reviewers on requests recorded in no-reviewer runs (pre-state safe), at k = 6.

    The reviewer seed uses fill='max', exactly as R1's open-loop k = 6 decisions that R1 reported.
    ``learned`` uses the model it had learned from that no-reviewer run up to the step.
    """
    out = []
    cfgs = {}
    for r in none_rows:
        if not r["pre_safe"]:
            continue
        ctx, t = r["context"], r["step"]
        if ctx not in cfgs:
            cfgs[ctx] = setup(cell, ctx, P, seeds)[0]
        cfg = cfgs[ctx]
        req = np.asarray(r["requests"])
        b = believed_vector(req, OPEN_K, f"{cell.game}__{ctx}__{t}", "max", r["previous"])
        if cell.game == "harvest":
            h = np.asarray(r["state"])
            ref = harvest_reference(cfg, h, req, stable_seed(seeds["reference"] + 1, h.tolist(), req.tolist()),
                                    draws=P["open_draws"])
            for cond in conditions:
                rc = learned_cfg(cfg, r["model"]) if cond == "learned" else rm.reviewer_config(cfg, cond)
                for rev in REVIEWERS:
                    s, _, _ = harvest_choose_scale(rc, h, b, rev, stable_seed(seeds["reviewer"] + 1, "h", ctx, t, rev, OPEN_K, "max"),
                                                   draws=P["reviewer_draws"])
                    out.append(dict(cell=cell.name, condition=cond, context=ctx, step=t, reviewer=rev, budget=OPEN_K,
                                    target="one_step", label=ref["label"], ref_risk=ref["risk"], scale=s))
        else:
            for cond in conditions:
                allee = rm.env_allee(cond)
                if r.get("allee", 0.0) != allee:
                    continue  # Allee cases come from the Allee no-reviewer run, the others from the logistic run
                rc = learned_cfg(cfg, r["model"]) if cond == "learned" else rm.reviewer_config(cfg, cond)
                for target in TARGETS:
                    ref = rm.fishery_true_reference(cfg, r["state"], req, target, allee)
                    for rev in REVIEWERS:
                        s, _ = fishery_choose_scale(rc, r["state"], b, rev, target)
                        out.append(dict(cell=cell.name, condition=cond, context=ctx, step=t, reviewer=rev, budget=OPEN_K,
                                        target=target, label=ref["label"], scale=s))
    return out


# ================================================================== S3 Part D memory, with R2 seeds and cells
class R2MemoryOverseer(s3.MemoryOverseer):
    """s3.MemoryOverseer with R2 seed bases (audit, noisy lies). Sanction none, as in S3 Part D."""

    def __init__(self, protocol, n, misreporters, d, context, game, mode, liar, seeds):
        super().__init__(protocol, n, misreporters, d, context, game, mode, liar)
        self.seeds = seeds

    def lie(self, t, i):
        if self.liar == "fixed":
            return self.d
        return float(np.random.default_rng(stable_seed(self.seeds["noisy"], self.game, self.context, t, i)).random())

    def step(self, t, p):
        # identical logic to s3.MemoryOverseer.step (trust == S1 'report'), with self.seeds
        r = np.array([p[i] * (1 - self.lie(t, i)) if i in self.mis else p[i] for i in range(self.n)])
        b, caught = r.copy(), np.zeros(self.n, bool)
        rng = np.random.default_rng(stable_seed(self.seeds["audit"], self.game, self.context, t, self.protocol))
        audited = []
        if self.protocol.startswith("rand"):
            audited = rng.permutation(np.arange(self.n))[:int(self.protocol[-1])].tolist()
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


def memory_episode(cell, context, mode, liar, P, seeds):
    protocol = "report" if mode == "trust" else MEMORY_PROTOCOL
    if cell.game == "harvest":
        cfg, specs, mis = setup(cell, context, P, seeds)
        ov = R2MemoryOverseer(protocol, cfg.n_agents, mis, LIE, context, "harvest", mode, liar, seeds)
        st, rows = dict(t=0), []

        def monitor(c, health, requests):
            t = st["t"]
            row = dict(step=t, pre_safe=int(harvest_safe(c, health)))
            p_eff, b, caught = ov.step(t, requests)
            row["caught"] = int(caught.sum())
            scale, _, _ = harvest_choose_scale(c, health, b, "joint", stable_seed(seeds["reviewer"], "h", False, context, t),
                                               draws=P["reviewer_draws"])
            executed = p_eff * scale
            row["label_exec"] = harvest_reference(c, health, executed, stable_seed(seeds["reference"], health.tolist(),
                                                  executed.tolist()), draws=P["closed_draws"])["label"]
            row["label_req"] = harvest_reference(c, health, p_eff, stable_seed(seeds["reference"], health.tolist(),
                                                 p_eff.tolist()), draws=P["closed_draws"])["label"]
            row["scale"] = scale
            rows.append(row)
            st["t"] = t + 1
            return executed

        res = run_harvest_episode(cfg, [x.to_agent() for x in specs], record_trace=True, action_filter=monitor)
        unsafe = [int(r["global_unsafe"]) for r in res["episode_trace_rows"]]
        e = s1.summarize("harvest", context, protocol, LIE, P["horizon"], np.asarray(res["final_payoffs"], float), mis,
                         res["mean_patch_health"], unsafe, rows, ov)
        e.update(target="one_step", failure=int(res["garden_failure_event"]))
    else:
        target = "msy"
        cfg, pol, mis = setup(cell, context, P, seeds)
        ov = R2MemoryOverseer(protocol, cfg.n_agents, mis, LIE, context, "fishery", mode, liar, seeds)
        state, pay, stocks, unsafe, rows = FisherySnapshot(cfg.stock_init), np.zeros(cfg.n_agents), [], [], []
        for t in range(cfg.horizon):
            req = fishery_requests(pol, state.stock)
            row = dict(step=t, pre_safe=int(fishery_safe(cfg, state)))
            p_eff, b, caught = ov.step(t, req)
            row["caught"] = int(caught.sum())
            scale, _ = fishery_choose_scale(cfg, state.stock, b, "joint", target, state.collapsed)
            executed = p_eff * scale
            row["label_exec"] = rm.fishery_true_reference(cfg, state.stock, executed, target)["label"]
            row["label_req"] = rm.fishery_true_reference(cfg, state.stock, p_eff, target)["label"]
            row["scale"] = scale
            rows.append(row)
            future, payoffs, _ = rm.fishery_step(cfg, state, executed)
            pay += payoffs
            stocks.append(future.stock)
            unsafe.append(int(not fishery_safe(cfg, future)))
            state = future
            if state.collapsed:
                break
        e = s1.summarize("fishery", context, protocol, LIE, cfg.horizon, pay, mis, float(np.mean(stocks)), unsafe, rows, ov)
        e.update(target=target, failure=int(state.collapsed))
    e.update(mode=mode, liar=liar)
    return e


# ================================================================== deterrence (S3 Part A search, Fishery cells)
def nc_episode(cell, context, protocol, d, fine, P, seeds, train=False, q=None, s=1.0):
    """s3.fishery_episode with the cell's population and regrowth (MSY target, S2 cheating model)."""
    cfg, pol, cheat = setup(cell, context, P, seeds, train=train)
    au = s3.Auditor(protocol, cfg.n_agents, cheat, d, fine, context, "fishery", seeds, q, s, "always")
    state, pay, stocks, unsafe, rows = FisherySnapshot(cfg.stock_init), np.zeros(cfg.n_agents), [], [], []
    for t in range(cfg.horizon):
        req = fishery_requests(pol, state.stock)
        row = dict(step=t, pre_safe=int(fishery_safe(cfg, state)))
        scale, _ = fishery_choose_scale(cfg, state.stock, req, "joint", DETER_TARGET, state.collapsed)
        allowance = req * scale
        taken = au.execute(req, allowance, t)
        au.audit(t, allowance, taken)
        if not train:
            row["label_exec"] = rm.fishery_true_reference(cfg, state.stock, taken, DETER_TARGET)["label"]
        rows.append(row)
        future, payoffs, _ = rm.fishery_step(cfg, state, taken)
        pay += payoffs
        stocks.append(future.stock)
        unsafe.append(int(not fishery_safe(cfg, future)))
        state = future
        if state.collapsed:
            break
    e = s2.summarize("fishery", context, protocol, d, fine, cfg.horizon, pay, cheat, float(np.mean(stocks)),
                     unsafe, rows, au)
    e.update(q=q, s=s, cheater_over_steps=float(au.over_steps[cheat].mean()), n_cheaters=len(cheat),
             failure=int(state.collapsed))
    return e


def gain_per_step(cheat_eps, comply_eps, horizon):
    """No-check gain per cheater per step of the horizon, mean over contexts."""
    g = [(a["cheater_payoff"] - b["cheater_payoff"]) / a["n_cheaters"] / horizon for a, b in zip(cheat_eps, comply_eps)]
    return float(np.mean(g))


def search_d(cell, fine, P, seeds, protocol="bern"):
    """S3 Part A search: common level d maximising mean cheater payoff on the training contexts; ties -> smaller d."""
    scores, best, best_score = {}, None, -np.inf
    for d in s3.SEARCH_D:
        sc = float(np.mean([nc_episode(cell, c, protocol, d, fine, P, seeds, True, q=DETER_Q)["cheater_payoff"]
                            for c in range(P["train_contexts"])]))
        scores[str(d)] = sc
        if best is None or sc > best_score + 1e-9:
            best, best_score = d, sc
    return best, scores


def deterrence(cell, P, seeds):
    train, test = range(P["train_contexts"]), range(P["contexts"])
    eps = []
    tr = {d: [nc_episode(cell, c, "allow", d, 0.0, P, seeds, True) for c in train] for d in s3.SEARCH_D}
    tr_comply, tr_cheat = tr[0.0], tr[CHEAT_LEVEL_FOR_G]
    te_comply = [nc_episode(cell, c, "allow", 0.0, 0.0, P, seeds) for c in test]
    te_cheat = [nc_episode(cell, c, "allow", CHEAT_LEVEL_FOR_G, 0.0, P, seeds) for c in test]
    G = gain_per_step(tr_cheat, tr_comply, P["horizon"])
    # exploratory: the cheaters' best level with no checks (S3 tie rule) and its break-even per horizon step
    # and per over-take step (the steps on which a fine can be charged)
    allow_scores = {str(d): float(np.mean([e["cheater_payoff"] for e in tr[d]])) for d in s3.SEARCH_D}
    d_allow = s3.SEARCH_D[0]
    for d in s3.SEARCH_D[1:]:
        if allow_scores[str(d)] > allow_scores[str(d_allow)] + 1e-9:
            d_allow = d
    over = {d: float(np.mean([e["cheater_over_steps"] for e in tr[d]])) for d in s3.SEARCH_D}
    G_allow = gain_per_step(tr[d_allow], tr_comply, P["horizon"])
    info = dict(**cell.tag(), G=G, G_test=gain_per_step(te_cheat, te_comply, P["horizon"]),
                over_steps_train=over[CHEAT_LEVEL_FOR_G],
                over_steps_test=float(np.mean([e["cheater_over_steps"] for e in te_cheat])),
                q=DETER_Q, s=1.0, cheat_level_for_G=CHEAT_LEVEL_FOR_G, train_contexts=P["train_contexts"],
                exploratory_allow=dict(train_scores=allow_scores, d_allow=d_allow, over_steps_by_d=over,
                                       G_at_d_allow=G_allow,
                                       G_per_over_step_at_0_75=G * P["horizon"] / over[CHEAT_LEVEL_FOR_G]
                                       if over[CHEAT_LEVEL_FOR_G] else None,
                                       G_per_over_step_at_d_allow=G_allow * P["horizon"] / over[d_allow]
                                       if d_allow > 0 and over[d_allow] else None))
    for e in te_comply:
        eps.append({**e, "arm": "comply", "e": None})
    for e in te_cheat:
        eps.append({**e, "arm": "allow_d0.75", "e": None})
    if not G > 0:
        info.update(testable=False, e_grid=[], searches=[], e_star=None, extended_searches=[], e_star_extended=None)
        return info, eps
    step = 2 * G / (N_E - 1)
    grid = [step * j for j in range(N_E)]
    searches = []
    for ev in grid:
        fine = ev / DETER_Q
        best, scores = search_d(cell, fine, P, seeds)
        searches.append(dict(e=ev, fine=fine, d_star=best, train_scores=scores))
        for c in test:
            eps.append({**nc_episode(cell, c, "bern", best, fine, P, seeds, q=DETER_Q), "arm": "adaptive", "e": ev})
    e_star = next((r["e"] for r in searches if r["d_star"] == 0), None)
    # exploratory only (not used for A-H4): if no grid value deters, continue with the same step up to 10 G
    extended = []
    if e_star is None:
        for j in range(N_E, 5 * (N_E - 1) + 1):
            ev = step * j
            best, scores = search_d(cell, ev / DETER_Q, P, seeds)
            extended.append(dict(e=ev, fine=ev / DETER_Q, d_star=best, train_scores=scores))
            if best == 0:
                break
    info.update(testable=True, e_grid=grid, searches=searches, e_star=e_star, extended_searches=extended,
                e_star_extended=next((r["e"] for r in extended if r["d_star"] == 0), None))
    return info, eps


# ================================================================== parts
def tagged(rows, **kw):
    return [{**kw, **r} for r in rows]


def part_a(P, seeds, log):
    episodes, closed_rows, opened, deter = [], [], [], []
    for cell in PART_A_CELLS:
        tag = cell.tag()
        none_rows = []
        for ctx in range(P["contexts"]):
            if cell.game == "harvest":
                ep, rows = harvest_closed(cell, ctx, "none", 0, "previous", P, seeds, keep_state=True)
                episodes.append({**tag, "kind": "closed", **ep}); none_rows += rows
                for k in BUDGETS:
                    ep, rows = harvest_closed(cell, ctx, "joint", k, "previous", P, seeds)
                    episodes.append({**tag, "kind": "closed", **ep}); closed_rows += tagged(rows, cell=cell.name, target="one_step")
            else:
                ep, rows = fishery_closed(cell, ctx, "none", 0, "previous", None, P, seeds, keep_state=True)
                episodes.append({**tag, "kind": "closed", **ep}); none_rows += rows
                for target in TARGETS:
                    for k in BUDGETS:
                        ep, rows = fishery_closed(cell, ctx, "joint", k, "previous", target, P, seeds)
                        episodes.append({**tag, "kind": "closed", **ep}); closed_rows += tagged(rows, cell=cell.name)
            for liar in LIARS:
                for mode in MEMORY_MODES:
                    episodes.append({**tag, "kind": "memory", **memory_episode(cell, ctx, mode, liar, P, seeds)})
            log(f"A {cell.name} context {ctx} done")
        opened += open_loop(cell, none_rows, P, seeds, ["exact"])
        log(f"A {cell.name} open loop done ({len(opened)} decisions so far)")
        if cell.game == "fishery":
            info, eps = deterrence(cell, P, seeds)
            deter.append(info)
            episodes += [{**tag, "kind": "deter", **e} for e in eps]
            log(f"A {cell.name} deterrence: G={info['G']:.4f} e*={info['e_star']} "
                f"(exploratory: d_allow={info['exploratory_allow']['d_allow']}, e* extended={info['e_star_extended']})")
    return dict(episodes=episodes, closed_rows=closed_rows, open_loop=opened), dict(deterrence=deter)


def part_b(P, seeds, log):
    episodes, closed_rows, opened = [], [], []
    for cell in PART_B_CELLS:
        tag = cell.tag()
        conds = rm.HARVEST_CONDITIONS if cell.game == "harvest" else rm.FISHERY_CONDITIONS
        none_rows = []
        for ctx in range(P["contexts"]):
            if cell.game == "harvest":
                ep, rows = harvest_closed(cell, ctx, "none", 0, "previous", P, seeds, keep_state=True, learn_none=True)
                episodes.append({**tag, "condition": "none", **ep}); none_rows += rows
                for cond in conds:
                    ep, rows = harvest_closed(cell, ctx, "joint", OPEN_K, "previous", P, seeds, condition=cond)
                    episodes.append({**tag, "condition": cond, **ep})
                    closed_rows += tagged(rows, cell=cell.name, condition=cond, target="one_step")
            else:
                for env_cond in ("exact", "allee"):
                    ep, rows = fishery_closed(cell, ctx, "none", 0, "previous", None, P, seeds, condition=env_cond,
                                              keep_state=True, learn_none=True)
                    episodes.append({**tag, "condition": "none" if env_cond == "exact" else "none_allee", **ep})
                    none_rows += tagged(rows, allee=rm.env_allee(env_cond))
                for cond in conds:
                    for target in TARGETS:
                        ep, rows = fishery_closed(cell, ctx, "joint", OPEN_K, "previous", target, P, seeds, condition=cond)
                        episodes.append({**tag, "condition": cond, **ep})
                        closed_rows += tagged(rows, cell=cell.name, condition=cond)
            log(f"B {cell.name} context {ctx} done")
        opened += open_loop(cell, none_rows, P, seeds, list(conds))
        log(f"B {cell.name} open loop done ({len(opened)} decisions so far)")
    return dict(episodes=episodes, closed_rows=closed_rows, open_loop=opened), {}


SOURCES = ["fishery_sim/reviewer_models.py", "fishery_sim/calibrated_oversight.py", "fishery_sim/fishery_oversight.py",
           "experiments/oversight/claude_oversight_common.py", "experiments/oversight/run_r1_repaired_reviewer.py",
           "experiments/oversight/run_s1_reporting_audit.py", "experiments/oversight/run_s2_compliance_deterrence.py",
           "experiments/oversight/run_s3_threshold_timing_memory.py",
           "experiments/oversight/run_r2_robustness_reviewer_model.py"]


def git_hash():
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return dict(commit=h, dirty=bool(dirty))
    except Exception:  # noqa: BLE001 - provenance only
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["A", "B"])
    ap.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    P = profile(a.profile)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)

    log(f"R2 part {a.part}, profile {a.profile}, params {P}")
    data, extra = (part_a if a.part == "A" else part_b)(P, SEEDS, log)
    files = {}
    for name, rows in data.items():
        write_jsonl_gz(out / f"{name}.jsonl.gz", rows)
        files[f"{name}.jsonl.gz"] = sha256(out / f"{name}.jsonl.gz")
    for name, obj in extra.items():
        (out / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
        files[f"{name}.json"] = sha256(out / f"{name}.json")
    manifest = dict(experiment=f"R2-{a.part}", profile=a.profile, params=P, seeds=SEEDS,
                    cells=[c.tag() for c in (PART_A_CELLS if a.part == "A" else PART_B_CELLS)],
                    counts={k: len(v) for k, v in data.items()}, files=files,
                    git=git_hash(), python=sys.version.split()[0], numpy=np.__version__,
                    source={p: sha256(Path(p)) for p in SOURCES},
                    design_notes=dict(
                        detect_seed="unused: perfect audits (s = 1)",
                        noisy_seed="noisy-liar draws (protocol names no base; S3 used audit + 2,000,000)",
                        open_loop_reviewer_seed="R1's open-loop seed with fill='max' (R1 reported k = 6 with fill max)",
                        G="mean over 32 training contexts of (cheater payoff at d = 0.75 - at d = 0) / cheaters / horizon, "
                          "'allow' protocol (allowances, no checks), MSY target",
                        deterrence_exploratory="d_allow / break-even per over-take step, and a search continued beyond 2G "
                                               "(same step, up to 10G) when no grid value deters; not used for A-H4"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    seconds = time.time() - t0  # kept out of manifest.json so that the manifest is byte-identical on rerun
    (out / "timing.json").write_text(json.dumps(dict(seconds=seconds)))
    log(json.dumps(dict(counts=manifest["counts"], seconds=round(seconds, 1))))


if __name__ == "__main__":
    main()
