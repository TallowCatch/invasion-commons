"""Engineering gates for experiment R2 (protocol file 19).

Gate 1: reviewer configuration copies; the least-squares estimator recovers r and K
from noise-free logistic data; the Allee transition with A = 0 equals the existing one.
Gate 2 (replication): in the piloted cells with the exact model, the R2 code path
reproduces R1's decisions for 4 sampled contexts given R1's seeds. R1's per-decision
outputs are not stored in the repository (only aggregate tables), so, as in S3
Amendment 1, the comparison is against R1's unchanged code run on this machine.
"""
from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from experiments.oversight import run_r1_repaired_reviewer as r1
from experiments.oversight import run_r2_robustness_reviewer_model as r2
from experiments.oversight import run_s3_threshold_timing_memory as s3
from experiments.oversight import run_s1_reporting_audit as s1
from experiments.oversight.claude_oversight_common import fishery_setup, harvest_setup
from fishery_sim import reviewer_models as rm
from fishery_sim.calibrated_oversight import fishery_reference
from fishery_sim.config import FisheryConfig
from fishery_sim.fishery_oversight import FisherySnapshot, transition

FISH = FisheryConfig(n_agents=6, horizon=80, stock_init=70, stock_max=100, regen_rate=.7, obs_noise_std=0,
                     max_harvest_per_agent=6)


def harvest_cfg():
    return harvest_setup(0, 1, 2)[0]


# ------------------------------------------------------------------ configuration copies
def _diff(a, b):
    return {f.name for f in fields(a) if getattr(a, f.name) != getattr(b, f.name)}


@pytest.mark.parametrize("cond,field,factor", [
    ("noise_low", "weather_noise_std", .5), ("noise_high", "weather_noise_std", 2.0),
    ("regen_low", "regen_rate", .75), ("regen_high", "regen_rate", 1.25)])
def test_harvest_reviewer_copies(cond, field, factor):
    true = harvest_cfg()
    before = repr(true)
    rc = rm.reviewer_config(true, cond)
    assert rc is not true and type(rc) is type(true)
    assert repr(true) == before  # the environment's configuration is untouched
    assert _diff(rc, true) == {field}
    assert np.isclose(getattr(rc, field), getattr(true, field) * factor)


@pytest.mark.parametrize("cond,changes", [
    ("regen_low", dict(regen_rate=.7 * .75)), ("regen_high", dict(regen_rate=.7 * 1.25)),
    ("K_low", dict(stock_max=75.0)), ("K_high", dict(stock_max=125.0)),
    ("learned", dict(regen_rate=.7 * 1.25, stock_max=125.0)), ("exact", {}), ("allee", {})])
def test_fishery_reviewer_copies(cond, changes):
    rc = rm.reviewer_config(FISH, cond)
    assert rc is not FISH and FISH.regen_rate == .7 and FISH.stock_max == 100
    assert _diff(rc, FISH) == set(changes)
    for k, v in changes.items():
        assert np.isclose(getattr(rc, k), v)


def test_regen_high_capped_at_one_and_learned_prior():
    hot = FisheryConfig(**{**FISH.__dict__, "regen_rate": .9})
    assert rm.reviewer_config(hot, "regen_high").regen_rate == 1.0
    assert rm.reviewer_config(hot, "learned").regen_rate == 1.0
    h = harvest_cfg()
    lp = rm.reviewer_config(h, "learned")
    assert np.isclose(lp.regen_rate, h.regen_rate * 1.25) and np.isclose(lp.weather_noise_std, h.weather_noise_std * .5)
    assert rm.env_allee("allee") == 20.0 and rm.env_allee("exact") == 0.0
    with pytest.raises(ValueError):
        rm.reviewer_config(h, "K_low")
    with pytest.raises(ValueError):
        rm.reviewer_config(FISH, "noise_low")


def test_reviewer_copy_changes_predictions_only():
    """The reviewer's copy feeds its prediction; the environment config stays the truth."""
    from fishery_sim.calibrated_oversight import fishery_choose_scale
    b = np.full(6, .9)
    exact = fishery_choose_scale(rm.reviewer_config(FISH, "exact"), 60.0, b, "joint", "msy")[0]
    klow = fishery_choose_scale(rm.reviewer_config(FISH, "K_low"), 60.0, b, "joint", "msy")[0]
    assert klow > exact  # K/2 = 37.5 instead of 50: less restrictive
    assert FISH.stock_max == 100


# ------------------------------------------------------------------ estimators
@pytest.mark.parametrize("r,K", [(.5, 100.0), (.7, 100.0), (.9, 100.0), (.6, 80.0), (1.0, 150.0)])
def test_least_squares_recovers_r_and_K(r, K):
    R = np.linspace(5, K * .95, 12)
    S = R + r * R * (1 - R / K)
    fit = rm.fit_logistic_rK(R, S)
    assert fit is not None
    assert np.isclose(fit[0], r, rtol=1e-9) and np.isclose(fit[1], K, rtol=1e-9)


def test_least_squares_recovers_from_simulated_fishery_trajectory():
    cfg = FisheryConfig(**{**FISH.__dict__, "regen_rate": .5})
    model = rm.ReviewerModel(cfg, "learned")
    assert model.config().regen_rate == .625 and model.config().stock_max == 125
    state = FisherySnapshot(70.0)
    rng = np.random.default_rng(0)
    for t in range(8):
        req = rng.uniform(0, 1, 6)
        nxt, _, _ = rm.fishery_step(cfg, state, req)
        model.observe_fishery(state.stock, req, nxt)
        if t < 4:
            assert model.config().regen_rate == .625  # optimistic prior until 5 steps are seen
        state = nxt
    assert np.isclose(model.config().regen_rate, .5, rtol=1e-8)
    assert np.isclose(model.config().stock_max, 100, rtol=1e-8)


def test_estimator_refuses_unidentified_data():
    assert rm.fit_logistic_rK([50, 50, 50], [60, 60, 60]) is None
    assert rm.fit_logistic_rK([10, 20], [5, 15]) is None  # r <= 0


def test_harvest_r_and_sigma_estimator():
    P, r = 20.0, .476
    R = np.linspace(1, 19, 300)
    g = R * (1 - R / P)
    fit = rm.fit_logistic_r(R, R + r * g, P)
    assert np.isclose(fit[0], r) and fit[1] < 1e-12
    w = np.random.default_rng(1).normal(0, .42, (200, 300))
    fits = np.array([rm.fit_logistic_r(R, R + r * g + e, P) for e in w])
    assert abs(fits[:, 0].mean() - r) < .005 and abs(fits[:, 1].mean() - .42) < .01


# ------------------------------------------------------------------ Allee transition
def test_allee_zero_equals_existing_transition():
    rng = np.random.default_rng(20261012)
    for regen in (.5, .7, .9, 1.0):
        cfg = FisheryConfig(**{**FISH.__dict__, "regen_rate": regen})
        for _ in range(60):
            a = b = FisherySnapshot(float(rng.uniform(0, 100)), int(rng.integers(0, 5)))
            for _ in range(40):  # long random sequences, including collapse and post-collapse steps
                req = rng.uniform(0, 1, 6) * rng.choice([0.2, 1.0])
                na, pa, ha = transition(cfg, a, req)
                nb, pb, hb = rm.fishery_step(cfg, b, req, allee=0.0)
                assert na == nb and ha == hb and np.array_equal(pa, pb)
                a, b = na, nb
            for stock in np.linspace(0, 100, 21):
                for req in (np.zeros(6), np.full(6, .5), np.ones(6)):
                    for target in ("one_step", "msy"):
                        assert rm.fishery_true_reference(cfg, stock, req, target, 0.0) == \
                            fishery_reference(cfg, stock, req, target)


def test_allee_dynamics():
    cfg = FISH
    zero = np.zeros(6)
    below, _, _ = rm.fishery_step(cfg, FisherySnapshot(15.0), zero, allee=20.0)
    assert below.stock < 15.0  # negative growth below A
    above, _, _ = rm.fishery_step(cfg, FisherySnapshot(50.0), zero, allee=20.0)
    assert np.isclose(above.stock, 50 + .7 * 50 * .5 * 1.5)
    assert rm.fishery_true_next(cfg, 15.0, 0.0, 20.0) < 15.0
    # collapse bookkeeping is replicated under Allee too
    s = FisherySnapshot(9.0)
    for _ in range(5):
        s, _, _ = rm.fishery_step(cfg, s, zero, allee=20.0)
    assert s.collapsed and s.stock == 0.0


# ------------------------------------------------------------------ setups
def test_new_setups_reproduce_existing_setups():
    for ctx in range(5):
        a = fishery_setup(ctx, 600_000_000, 80, n_stress=4)
        b = r2.fishery_setup_r2(ctx, 600_000_000, 80, 4, .7)
        assert a[0] == b[0] and a[2] == b[2] and all(np.array_equal(a[1][k], b[1][k]) for k in a[1])
        c = harvest_setup(ctx, 610_000_000, 620_000_000, 80, n_stress=2)
        d = r2.harvest_setup_r2(ctx, 610_000_000, 620_000_000, 80, 2, .85)
        assert c[0] == d[0] and c[2] == d[2] and [repr(x) for x in c[1]] == [repr(x) for x in d[1]]
    cfg = r2.harvest_setup_r2(0, 1, 2, 80, 4, 1.0)[0]
    assert np.isclose(cfg.regen_rate, .56)
    with pytest.raises(ValueError):
        r2.fishery_setup_r2(0, 1, 80, 2, 1.1)


# ------------------------------------------------------------------ Gate 2: replication of R1
R1_SEEDS = {**r2.SEEDS, **r1.SEEDS}
R1_P = r1.profile("full")
GATE_CONTEXTS = sorted(np.random.default_rng(20261012).choice(64, 4, replace=False).tolist())
H_PILOT, F_PILOT = r2.Cell("harvest", 2, .85), r2.Cell("fishery", 4, .7)
KEYS = ("step", "pre_safe", "scale", "label")


def _same_rows(a, b, keys=KEYS):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert {k: x[k] for k in keys} == {k: y[k] for k in keys}


def _same_episode(a, b):
    for k in ("total_harvest", "mean_health", "t_end", "failure", "unsafe_fixed", "stressed"):
        assert a[k] == b[k], k


@pytest.mark.parametrize("ctx", GATE_CONTEXTS)
def test_replication_gate_closed_loop(ctx):
    for k in r2.BUDGETS:
        a_ep, a_rows = r1.harvest_episode(ctx, "joint", k, "previous", R1_P)
        b_ep, b_rows = r2.harvest_closed(H_PILOT, ctx, "joint", k, "previous", R1_P, R1_SEEDS, condition="exact")
        _same_rows(a_rows, b_rows, KEYS + ("ref_risk",))
        _same_episode(a_ep, b_ep)
        for target in r2.TARGETS:
            a_ep, a_rows = r1.fishery_episode(ctx, "joint", k, "previous", target, R1_P)
            b_ep, b_rows = r2.fishery_closed(F_PILOT, ctx, "joint", k, "previous", target, R1_P, R1_SEEDS, condition="exact")
            _same_rows(a_rows, b_rows)
            _same_episode(a_ep, b_ep)


def test_replication_gate_open_loop():
    none_r1, none_h, none_f = [], [], []
    for ctx in GATE_CONTEXTS:
        none_r1 += r1.harvest_episode(ctx, "none", 0, "max", R1_P)[1]
        none_r1 += r1.fishery_episode(ctx, "none", 0, "max", "one_step", R1_P)[1]
        none_h += r2.harvest_closed(H_PILOT, ctx, "none", 0, "previous", R1_P, R1_SEEDS, keep_state=True)[1]
        none_f += r2.fishery_closed(F_PILOT, ctx, "none", 0, "previous", None, R1_P, R1_SEEDS, keep_state=True)[1]
    ref = {(r["game"], r["context"], r["step"], r["reviewer"], r["target"]): (r["scale"], r["label"])
           for r in r1.open_loop(none_r1, R1_P) if r["budget"] == 6 and r["fill"] == "max"}
    mine = r2.open_loop(H_PILOT, none_h, R1_P, R1_SEEDS, ["exact"]) + \
        r2.open_loop(F_PILOT, none_f, R1_P, R1_SEEDS, ["exact"])
    got = {("harvest" if r["cell"].startswith("harvest") else "fishery", r["context"], r["step"], r["reviewer"],
            r["target"]): (r["scale"], r["label"]) for r in mine}
    assert len(ref) > 100 and got == ref


def test_memory_and_deterrence_paths_reproduce_s3():
    """Supporting check: with S1/S3 seeds, R2's Part D and deterrence episodes equal S3's (piloted cells)."""
    P = dict(contexts=2, horizon=20, train_contexts=2, closed_draws=2000, open_draws=4000, reviewer_draws=400)
    s3P = dict(contexts=2, train_contexts=2, horizon=20, reviewer_draws=400, ref_draws=2000)
    seeds = {**r2.SEEDS, **s1.SEEDS, "noisy": s3.SEEDS["noisy"]}
    drop = ("protocol", "target", "failure", "mode", "liar")
    for ctx in range(2):
        for liar in ("fixed", "noisy"):
            for mode in r2.MEMORY_MODES:
                proto = "report" if mode == "trust" else "rand2"
                a = s3.harvest_episode_d(ctx, proto, mode, liar, s3P)
                b = r2.memory_episode(H_PILOT, ctx, mode, liar, P, seeds)
                assert {k: v for k, v in a.items() if k not in drop} == {k: v for k, v in b.items() if k not in drop}
                a = s3.fishery_episode_d(ctx, proto, mode, liar, s3P, "msy")
                b = r2.memory_episode(F_PILOT, ctx, mode, liar, P, seeds)
                assert {k: v for k, v in a.items() if k not in drop} == {k: v for k, v in b.items() if k not in drop}
        a = s3.fishery_episode(ctx, "bern", .75, 1.8, s3P, s3.SEEDS, q=1 / 6)
        b = r2.nc_episode(F_PILOT, ctx, "bern", .75, 1.8, P, {**r2.SEEDS, **s3.SEEDS}, q=1 / 6)
        assert {k: v for k, v in a.items() if k != "timing"} == \
            {k: v for k, v in b.items() if k not in ("n_cheaters", "failure")}
