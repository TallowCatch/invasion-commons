"""Study G1: Schelling diagrams for Fishery, Forest and River with programmed agents (Claude audit, October 2026).

Protocol (frozen, with Amendment 1): notes/claude_audit_20261005/studies/G1_schelling_and_sweeps/protocol.md

Diagram 1 (base commons): no reviewer, no audits; k = 0..N agents use the greedy request policy, the rest
the restrained one. Diagram 2 (audited commons): the paper's population of request policies, the MSY /
half-capacity reviewer, Bernoulli audits q = 1/6 with a flat fine F in {0, F*/2, F*, 2F*}; k = 0..N agents
chosen at random over-take fully (d = 1), the rest take their allowance.

Dynamics, policies, reviewers and the auditor are imported, not re-implemented:
- Fishery: fishery_sim.fishery_oversight.transition, calibrated_oversight.fishery_choose_scale (MSY target);
- Forest: fishery_sim.harvest.run_harvest_episode with the harvest_evolution strategies and
  calibrated_oversight.harvest_choose_scale;
- River: fishery_sim.two_reagent (C1 frozen parameters), reviewer target 50 via run_r4_river_grid.river_target;
- Audits and fines: run_s3_threshold_timing_memory.Auditor ("bern" protocol, s = 1).

Run:  PYTHONPATH=. python3 -m experiments.oversight.run_g1_schelling --profile smoke|full --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from experiments.oversight import run_s3_threshold_timing_memory as s3
from experiments.oversight.claude_oversight_common import fishery_requests, sha256, write_jsonl_gz
from experiments.oversight.run_r4_river_grid import river_target
from fishery_sim import two_reagent as TR
from fishery_sim.calibrated_oversight import fishery_choose_scale, harvest_choose_scale, stable_seed
from fishery_sim.config import FisheryConfig
from fishery_sim.fishery_oversight import FisherySnapshot, transition
from fishery_sim.harvest import run_harvest_episode
from fishery_sim.harvest_benchmarks import make_harvest_cfg_for_scenario
from fishery_sim.harvest_evolution import adversarial_harvest_strategy, cooperative_harvest_strategy

G1_BASE = 26_101_400
GAMES = ("fishery", "forest", "river")
Q = 1 / 6
SEARCH_D = s3.SEARCH_D            # (0, 1/4, 1/2, 3/4, 1)
G_LEVELS = (0.25, 0.5, 0.75, 1.0)  # run_r3_breakeven.LEVELS
# Paper break-even, Fishery pilot setting: body.tex:96; studies/S3_threshold_timing_memory/results.md:49
FISHERY_G_STAR = 0.356
FISHERY_F_STAR = FISHERY_G_STAR / Q
REVIEWER_DRAWS = 400
RIVER_TARGET = 50.0
B, BOOT_SEED = 4000, 26101499


def profile(name):
    if name == "smoke":
        return dict(populations=3, train=2, horizon=20)
    if name == "full":
        return dict(populations=64, train=8, horizon=80)
    raise ValueError(name)


# ================================================================== settings
@dataclass(frozen=True)
class Setting:
    """One game at one parameter point. Defaults are the paper's values (Amendment 1, item 7)."""
    game: str
    n: int = 6
    horizon: int = 80
    r: float = 0.7             # Fishery regrowth (claude_oversight_common.fishery_setup)
    externality: float = 0.22  # Forest neighbour externality (harvest_benchmarks forest_co_management)
    kappa_mult: float = 1.0    # River synergy kappa as a multiple of the C1 value (lambda held fixed)

    @property
    def capacity(self):
        return 100.0 * self.n / 6 if self.game == "fishery" else None

    @property
    def n_stress(self):
        """The paper's number of greedy requesters (Fishery 4 of 6, scaled with N; Forest 2; River 1 X + 1 Y)."""
        return int(round(4 * self.n / 6)) if self.game == "fishery" else 2

    def tag(self):
        return {**asdict(self), "capacity": self.capacity, "n_stress": self.n_stress}


# ================================================================== seeds and populations
def population_seed(base, role, setting, c):
    return stable_seed(base, "pop", role, setting.game, setting.n, c)


def ctx_seeds(base, role, setting, c):
    """Weather, audit and reviewer draws depend only on (game, population, round), so are shared over k and F."""
    return {k: stable_seed(base, k, role, setting.game, c) for k in ("weather", "audit", "detect", "reviewer")}


def first_k(order, k):
    mask = np.zeros(len(order), bool)
    mask[np.asarray(order)[:k]] = True
    return mask


def population(setting, base, role, c):
    """Both request policies for every agent plus three random orders (nested across k).

    Ranges are those of claude_oversight_common.fishery_setup (lines 36-41), harvest_setup (strategies) and
    two_reagent.RANGE_SETS['wide_low'] (C1 frozen ranges); only the assignment to agents is new.
    """
    rng = np.random.default_rng(population_seed(base, role, setting, c))
    n = setting.n
    pop = dict(order=rng.permutation(n), order2=rng.permutation(n), stress_order=rng.permutation(n))
    if setting.game == "fishery":
        sc = setting.capacity / 100.0
        pop.update(rest_low=rng.uniform(.05, .20, n), rest_high=rng.uniform(.20, .40, n),
                   greedy_low=rng.uniform(.45, .65, n), greedy_high=rng.uniform(.75, .95, n),
                   thresholds=rng.uniform(25, 65, n) * sc)
    elif setting.game == "forest":
        pop.update(coop=[cooperative_harvest_strategy(f"g1_{i}", 20.0, rng) for i in range(n)],
                   adv=[adversarial_harvest_strategy(f"g1_{i}", 20.0, rng) for i in range(n)])
    elif setting.game == "river":
        R = TR.RANGE_SETS["wide_low"]
        pop.update(rest_low=rng.uniform(*R["normal"][0], n), rest_high=rng.uniform(*R["normal"][1], n),
                   greedy_low=rng.uniform(*R["stress"][0], n), greedy_high=rng.uniform(*R["stress"][1], n),
                   thresholds=rng.uniform(*R["thresholds"], n),
                   # C1 population: one X and one Y stress agent (two_reagent.make_population)
                   river_stress=[int(rng.integers(0, 3)), int(3 + rng.integers(0, 3))])
    else:
        raise ValueError(setting.game)
    return pop


def paper_greedy(setting, pop):
    """The paper's mix of request policies for this population (Diagram 2, deterrence)."""
    if setting.game == "river":
        mask = np.zeros(setting.n, bool)
        mask[pop["river_stress"]] = True
        return mask
    return first_k(pop["stress_order"], setting.n_stress)


def threshold_policy(pop, greedy):
    return dict(low=np.where(greedy, pop["greedy_low"], pop["rest_low"]),
                high=np.where(greedy, pop["greedy_high"], pop["rest_high"]), thresholds=pop["thresholds"])


# ================================================================== episodes
def episode(setting, pop, greedy, overtake, d, fine, reviewer, audits, sd):
    """One game. greedy: request-policy mask; overtake: agents that take a + d (p - a).

    Returns per-agent gross payoff, fines, over-take rounds, and the fixed-horizon harm share.
    """
    n, T = setting.n, setting.horizon
    cheat = np.flatnonzero(overtake).tolist()
    au = s3.Auditor("bern" if audits else "allow", n, cheat, d, fine, 0, setting.game,
                    dict(audit=sd["audit"], detect=sd["detect"]), q=Q if audits else None, s=1.0)
    fn = dict(fishery=_fishery, forest=_forest, river=_river)[setting.game]
    pay, harm_rounds, t_end, failed = fn(setting, pop, np.asarray(greedy, bool), reviewer, au, sd)
    harm = (sum(harm_rounds) + (T - len(harm_rounds))) / T  # rounds after a collapse count as harmful
    return dict(pay=np.asarray(pay, float), fines=au.fines.copy(), over_steps=au.over_steps.copy(), harm=float(harm),
                t_end=int(t_end), failed=int(failed))


def _fishery(s, pop, greedy, reviewer, au, sd):
    """Loop of run_r2_robustness_reviewer_model.nc_episode / run_s3 fishery_episode, with N, K and T as parameters."""
    K = s.capacity
    cfg = FisheryConfig(n_agents=s.n, horizon=s.horizon, stock_init=0.7 * K, stock_max=K, regen_rate=s.r,
                        obs_noise_std=0, max_harvest_per_agent=6, collapse_threshold=10.0 * K / 100.0)
    pol = threshold_policy(pop, greedy)
    state, pay, harm = FisherySnapshot(cfg.stock_init), np.zeros(s.n), []
    for t in range(cfg.horizon):
        req = fishery_requests(pol, state.stock)
        if reviewer:
            scale, _ = fishery_choose_scale(cfg, state.stock, req, "joint", "msy", state.collapsed)
            allowance = req * scale
        else:
            allowance = req.copy()
        taken = au.execute(req, allowance, t)
        au.audit(t, allowance, taken)
        # harm as in run_l2_llm_agents (msy_break): stock left after fishing below K/2
        harm.append(int(state.stock - taken.sum() * cfg.max_harvest_per_agent < K / 2 - 1e-9))
        future, payoffs, _ = transition(cfg, state, taken)
        pay += payoffs
        state = future
        if state.collapsed:
            break
    return pay, harm, len(harm), int(state.collapsed)


def _forest(s, pop, greedy, reviewer, au, sd):
    """claude_oversight_common.harvest_setup (regrowth x 0.85), the externality as a parameter."""
    cfg = make_harvest_cfg_for_scenario("forest_co_management", horizon=s.horizon, seed=sd["weather"],
                                        communication_enabled=False, side_payments_enabled=False)
    cfg.regen_rate *= .85
    cfg.neighbor_externality = s.externality
    specs = [pop["adv"][i] if greedy[i] else pop["coop"][i] for i in range(s.n)]
    st = dict(t=0)

    def monitor(c, health, requests):
        t = st["t"]
        if reviewer:
            scale, _, _ = harvest_choose_scale(c, health, requests, "joint", stable_seed(sd["reviewer"], t),
                                               draws=REVIEWER_DRAWS)
            allowance = requests * scale
        else:
            allowance = requests.copy()
        taken = au.execute(requests, allowance, t)
        au.audit(t, allowance, taken)
        st["t"] = t + 1
        return np.minimum(taken, requests)  # guards float rounding; taken <= requests by construction

    res = run_harvest_episode(cfg, [x.to_agent() for x in specs], record_trace=True, action_filter=monitor)
    harm = [int(r["mean_patch_health_after"] < cfg.global_min_mean_patch_health) for r in res["episode_trace_rows"]]
    return np.asarray(res["final_payoffs"], float), harm, res["t_end"], res["garden_failure_event"]


def river_params(s):
    P = TR.frozen_params("comp")
    return replace(P, kappa=P.kappa * s.kappa_mult)  # lambda stays at its paper value (Amendment 1, item 7)


def _river(s, pop, greedy, reviewer, au, sd):
    """run_r4_river_grid.episode loop (joint reviewer, target 50) with over-taking instead of misreporting."""
    P = river_params(s)
    pol = threshold_policy(pop, greedy)
    shocks = TR.weather_shocks(0, sd["weather"], s.horizon) * P.sigma
    q, pay, harm = TR.Q_INIT, np.zeros(s.n), []
    with river_target(RIVER_TARGET):
        for t in range(s.horizon):
            req = TR.policy_requests(pol, q)
            if reviewer:
                eps = TR.reviewer_draws(P, stable_seed(sd["reviewer"], t), REVIEWER_DRAWS)
                allowance = TR.review(P, q, req, req, "joint", eps)["executed"]
            else:
                allowance = req.copy()
            taken = au.execute(req, allowance, t)
            au.audit(t, allowance, taken)
            q_next = TR.next_q(P, q, taken, shocks[t])
            harm.append(int(q_next < RIVER_TARGET))
            pay += taken  # C1: each agent earns its discharge
            q = q_next
    return pay, harm, len(harm), 0


# ================================================================== break-even prediction and search
def cheater_net(e, cheat):
    return float(e["pay"][cheat].sum() - e["fines"][cheat].sum())


def predict_gstar(setting, base, n_train):
    """run_r3_breakeven.predict: g* = max over d with G(d) > 0 of G(d) / n(d), reviewer on, no audits."""
    out = {}
    eps0 = []
    for c in range(n_train):
        pop = population(setting, base, "train", c)
        g = paper_greedy(setting, pop)
        eps0.append((pop, g, episode(setting, pop, g, g, 0.0, 0.0, True, False, ctx_seeds(base, "train", setting, c))))
    for d in G_LEVELS:
        gains, over = [], []
        for c, (pop, g, e0) in enumerate(eps0):
            e = episode(setting, pop, g, g, d, 0.0, True, False, ctx_seeds(base, "train", setting, c))
            cheat = np.flatnonzero(g)
            gains.append((cheater_net(e, cheat) - cheater_net(e0, cheat)) / len(cheat))
            over.append(float(e["over_steps"][cheat].mean()))
        G, nn = float(np.mean(gains)), float(np.mean(over))
        out[str(d)] = dict(G=G, n_over=nn, g=G / nn if nn > 0 else None)
    cands = [v["g"] for v in out.values() if v["G"] > 0 and v["g"] is not None]
    return dict(by_level=out, g_star=max(cands) if cands else 0.0)


# ================================================================== Schelling curves and the dilemma test
def schelling_curves(returns, greedy, n):
    """returns[k], greedy[k]: (P, N) per-agent returns and greedy masks in populations with k greedy agents.

    m = number of OTHER restrained agents (Hughes et al. 2018). A restrained agent with m restrained others
    lives in a population with k = N - 1 - m greedy; a greedy agent with m restrained others in k = N - m.
    Returns rc, rd of shape (P, N): column m is the population's mean return of that type.
    """
    P = np.asarray(returns[0]).shape[0]
    rc, rd = np.full((P, n), np.nan), np.full((P, n), np.nan)
    for m in range(n):
        R, G = np.asarray(returns[n - 1 - m], float), np.asarray(greedy[n - 1 - m], bool)
        rc[:, m] = np.where(~G, R, 0).sum(1) / (~G).sum(1)
        R, G = np.asarray(returns[n - m], float), np.asarray(greedy[n - m], bool)
        rd[:, m] = np.where(G, R, 0).sum(1) / G.sum(1)
    return rc, rd


def _interval(draws):
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def dilemma_test(rc, rd, b=B, seed=BOOT_SEED):
    """D1-D3 (protocol), each requiring the 95% population-bootstrap interval of the difference to exclude 0."""
    rc, rd = np.asarray(rc, float), np.asarray(rd, float)
    P, N = rc.shape
    idx = np.random.default_rng(seed).integers(0, P, size=(b, P))
    bc, bd = rc[idx].mean(1), rd[idx].mean(1)  # (b, N), paired across m
    half = (N - 1) // 2

    def diff(est, draws):
        ci = _interval(draws)
        return dict(estimate=float(est), ci=ci, holds=bool(ci[0] > 0))

    d1 = diff(rc[:, N - 1].mean() - rd[:, 0].mean(), bc[:, N - 1] - bd[:, 0])
    d2 = diff(rc[:, N - 1].mean() - rc[:, 0].mean(), bc[:, N - 1] - bc[:, 0])
    temptation = [diff(rd[:, m].mean() - rc[:, m].mean(), bd[:, m] - bc[:, m]) for m in range(N)]
    fear = any(t["holds"] for t in temptation[:half + 1])
    greed = any(t["holds"] for t in temptation[half + 1:])
    d3 = dict(holds=bool(fear or greed), fear=bool(fear), greed=bool(greed), small_m_max=half,
              by_m=temptation, never_pays_point=bool(all(t["estimate"] <= 0 for t in temptation)))
    return dict(D1=d1, D2=d2, D3=d3, dilemma=bool(d1["holds"] and d2["holds"] and d3["holds"]),
                rc=rc.mean(0).tolist(), rd=rd.mean(0).tolist(),
                rc_ci=[_interval(bc[:, m]) for m in range(N)], rd_ci=[_interval(bd[:, m]) for m in range(N)])


# ================================================================== diagrams
def diagram1(setting, base, n_pop, role="test"):
    rows = []
    for c in range(n_pop):
        pop, sd = population(setting, base, role, c), ctx_seeds(base, role, setting, c)
        none = np.zeros(setting.n, bool)
        for k in range(setting.n + 1):
            g = first_k(pop["order"], k)
            e = episode(setting, pop, g, none, 0.0, 0.0, False, False, sd)
            rows.append(dict(diagram=1, game=setting.game, k=k, context=c, fine=None, returns=e["pay"].tolist(),
                             defect=g.astype(int).tolist(), harm=e["harm"], failed=e["failed"]))
    return rows


def diagram2(setting, base, n_pop, fines, role="test"):
    rows = []
    for c in range(n_pop):
        pop, sd = population(setting, base, role, c), ctx_seeds(base, role, setting, c)
        req = paper_greedy(setting, pop)
        for F in fines:
            for k in range(setting.n + 1):
                ot = first_k(pop["order2"], k)
                e = episode(setting, pop, req, ot, 1.0, F, True, True, sd)
                rows.append(dict(diagram=2, game=setting.game, k=k, context=c, fine=float(F),
                                 returns=(e["pay"] - e["fines"]).tolist(), gross=e["pay"].tolist(),
                                 fines=e["fines"].tolist(), defect=ot.astype(int).tolist(),
                                 greedy_requester=req.astype(int).tolist(), harm=e["harm"], failed=e["failed"]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    P = profile(a.profile)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0, rows, fstar = time.time(), [], {}
    for game in GAMES:
        s = Setting(game, horizon=P["horizon"])
        if game == "fishery":
            fstar[game] = dict(g_star=FISHERY_G_STAR, F_star=FISHERY_F_STAR, source="paper (body.tex:96; S3 results:49)")
        else:  # Amendment 1, item 1: computed before any run with a fine
            pred = predict_gstar(s, G1_BASE, P["train"])
            fstar[game] = dict(g_star=pred["g_star"], F_star=pred["g_star"] / Q, source="R3 rule, 8 training populations",
                               by_level=pred["by_level"])
        print(f"[{time.time() - t0:6.0f}s] {game} F* = {fstar[game]['F_star']:.4f}", flush=True)
    (out / "fstar.json").write_text(json.dumps(fstar, indent=1, default=float))
    for game in GAMES:
        s = Setting(game, horizon=P["horizon"])
        rows += diagram1(s, G1_BASE, P["populations"])
        print(f"[{time.time() - t0:6.0f}s] {game} diagram 1 done", flush=True)
        F = fstar[game]["F_star"]
        fines = tuple(dict.fromkeys((0.0, F / 2, F, 2 * F)))  # Amendment 2: F* = 0 -> F = 0 only
        rows += diagram2(s, G1_BASE, P["populations"], fines)
        print(f"[{time.time() - t0:6.0f}s] {game} diagram 2 done", flush=True)
    write_jsonl_gz(out / "episodes.jsonl.gz", rows)
    manifest = dict(experiment="G1", profile=a.profile, params=P, G1_BASE=G1_BASE, q=Q, fstar=fstar,
                    settings={g: Setting(g, horizon=P["horizon"]).tag() for g in GAMES}, n_rows=len(rows),
                    files={f: sha256(out / f) for f in ("episodes.jsonl.gz", "fstar.json")},
                    source={p: sha256(Path(p)) for p in [
                        "experiments/oversight/run_g1_schelling.py", "experiments/oversight/run_s3_threshold_timing_memory.py",
                        "experiments/oversight/run_s2_compliance_deterrence.py", "experiments/oversight/run_r4_river_grid.py",
                        "fishery_sim/two_reagent.py", "fishery_sim/harvest.py", "fishery_sim/calibrated_oversight.py",
                        "fishery_sim/fishery_oversight.py"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    (out / "timing.json").write_text(json.dumps(dict(seconds=time.time() - t0)))
    print(json.dumps(dict(n_rows=len(rows), seconds=round(time.time() - t0, 1))))


if __name__ == "__main__":
    main()
