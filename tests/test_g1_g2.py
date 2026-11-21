"""Tests for G1/G2 (Schelling diagrams and parameter sweeps).

Protocol: notes/claude_audit_20261005/studies/G1_schelling_and_sweeps/protocol.md
"""
import numpy as np
import pytest

from experiments.oversight import run_g1_schelling as g1
from experiments.oversight import run_g2_sweeps as g2


# ------------------------------------------------------------------ D1-D3 on hand-built curves
def _curves(rc_mean, rd_mean, P=64, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    rc = np.asarray(rc_mean, float)[None, :] + rng.normal(0, noise, (P, len(rc_mean)))
    rd = np.asarray(rd_mean, float)[None, :] + rng.normal(0, noise, (P, len(rd_mean)))
    return rc, rd


def test_prisoners_dilemma_curves_pass_all_three():
    N = 6
    m = np.arange(N)
    rc, rd = _curves(10 * m, 10 * m + 3)  # defecting always pays 3 more; everyone cooperating pays most
    res = g1.dilemma_test(rc, rd)
    assert res["D1"]["holds"] and res["D2"]["holds"] and res["D3"]["holds"]
    assert res["D3"]["fear"] and res["D3"]["greed"]
    assert res["dilemma"]


def test_no_temptation_fails_d3_only():
    N = 6
    m = np.arange(N)
    rc, rd = _curves(10 * m + 3, 10 * m)  # cooperating always pays more: no dilemma
    res = g1.dilemma_test(rc, rd)
    assert res["D1"]["holds"] and res["D2"]["holds"]
    assert not res["D3"]["holds"] and not res["dilemma"]


def test_mutual_defection_better_fails_d1():
    N = 6
    rc, rd = _curves([0, 1, 2, 3, 4, 5], [50, 51, 52, 53, 54, 55])
    res = g1.dilemma_test(rc, rd)
    assert not res["D1"]["holds"]  # R_c(5)=5 < R_d(0)=50
    assert res["D2"]["holds"]
    assert not res["dilemma"]


def test_fear_only_and_greed_only():
    # N = 6 -> small m are 0..2 (floor(5/2) = 2), large m are 3..5
    rc, rd = _curves([0, 2, 4, 6, 8, 10], [1, 3, 5, 6, 8, 10])
    res = g1.dilemma_test(rc, rd)
    assert res["D3"]["fear"] and not res["D3"]["greed"] and res["D3"]["holds"]
    rc, rd = _curves([0, 2, 4, 6, 8, 10], [0, 2, 4, 7, 9, 11])
    res = g1.dilemma_test(rc, rd)
    assert res["D3"]["greed"] and not res["D3"]["fear"] and res["D3"]["holds"]


def test_interval_must_exclude_zero():
    rc, rd = _curves([0, 2, 4, 6, 8, 10], [0, 2, 4, 6, 8, 10.05], noise=2.0)
    res = g1.dilemma_test(rc, rd)
    assert not res["D3"]["holds"]  # a tiny, noisy edge is not enough


# ------------------------------------------------------------------ Schelling indexing
def test_schelling_m_is_number_of_other_cooperators():
    """Hand-built returns: an agent earns 10 x (number of OTHER restrained agents), +1 if greedy."""
    N, P = 6, 3
    returns, greedy = {}, {}
    for k in range(N + 1):
        mask = np.zeros((P, N), bool)
        mask[:, :k] = True
        n_restrained = N - k
        others = np.where(mask, n_restrained, n_restrained - 1)
        returns[k] = 10.0 * others + mask
        greedy[k] = mask
    rc, rd = g1.schelling_curves(returns, greedy, N)
    assert rc.shape == (P, N) and rd.shape == (P, N)
    np.testing.assert_allclose(rc[0], 10.0 * np.arange(N))
    np.testing.assert_allclose(rd[0], 10.0 * np.arange(N) + 1)


def test_schelling_uses_k_equal_n_minus_1_minus_m_for_cooperators():
    N, P = 4, 1
    returns = {k: np.full((P, N), float(k)) for k in range(N + 1)}  # return encodes k
    greedy = {k: np.arange(N)[None, :] < k for k in range(N + 1)}
    rc, rd = g1.schelling_curves(returns, greedy, N)
    np.testing.assert_allclose(rc[0], [N - 1 - m for m in range(N)])
    np.testing.assert_allclose(rd[0], [N - m for m in range(N)])


# ------------------------------------------------------------------ seeds
def test_seed_bases_are_the_protocols_new_bases():
    assert g1.G1_BASE == 26_101_400
    assert g2.G2_BASE == 26_101_420


def test_population_seeds_come_from_the_base_and_do_not_collide():
    s = g1.Setting("fishery")
    a = g1.population_seed(g1.G1_BASE, "test", s, 0)
    b = g1.population_seed(g2.G2_BASE, "test", s, 0)
    assert a != b
    seeds1 = {g1.population_seed(g1.G1_BASE, "test", s, c) for c in range(64)}
    seeds2 = {g1.population_seed(g2.G2_BASE, "test", s, c) for c in range(64)}
    assert len(seeds1) == 64 and not seeds1 & seeds2
    # the seed is base + hash, so it starts at the base
    assert all(x >= g1.G1_BASE for x in seeds1)


def test_populations_are_nested_across_k():
    s = g1.Setting("fishery")
    pop = g1.population(s, g1.G1_BASE, "test", 3)
    m2, m3 = g1.first_k(pop["order"], 2), g1.first_k(pop["order"], 3)
    assert m2.sum() == 2 and m3.sum() == 3 and np.all(m3[m2])


# ------------------------------------------------------------------ games (smoke)
@pytest.mark.parametrize("game", ["fishery", "forest", "river"])
def test_episode_runs_and_greedy_take_more_without_reviewer(game):
    s = g1.Setting(game, horizon=10)
    pop = g1.population(s, g1.G1_BASE, "test", 0)
    mask = g1.first_k(pop["order"], 3)
    e = g1.episode(s, pop, mask, np.zeros(s.n, bool), 0.0, 0.0, False, False, g1.ctx_seeds(g1.G1_BASE, "test", s, 0))
    assert e["pay"].shape == (s.n,) and 0 <= e["harm"] <= 1
    assert np.all(e["fines"] == 0)


def test_full_overtake_with_no_reviewer_equals_complying():
    s = g1.Setting("fishery", horizon=10)
    pop = g1.population(s, g1.G1_BASE, "test", 1)
    mask = g1.first_k(pop["order"], 2)
    sd = g1.ctx_seeds(g1.G1_BASE, "test", s, 1)
    a = g1.episode(s, pop, mask, np.zeros(6, bool), 0.0, 0.0, False, False, sd)
    b = g1.episode(s, pop, mask, mask, 1.0, 0.0, False, False, sd)
    np.testing.assert_allclose(a["pay"], b["pay"])


def test_fines_only_charged_to_overtakers_under_audits():
    s = g1.Setting("fishery", horizon=30)
    pop = g1.population(s, g1.G1_BASE, "test", 2)
    req = g1.paper_greedy(s, pop)
    ot = g1.first_k(pop["order2"], 3)
    e = g1.episode(s, pop, req, ot, 1.0, 5.0, True, True, g1.ctx_seeds(g1.G1_BASE, "test", s, 2))
    assert np.all(e["fines"][~ot] == 0)


def test_n_scaling_keeps_k_over_n():
    s = g2.fishery_setting_for_n(10)
    assert s.n == 10 and abs(s.capacity / s.n - 100 / 6) < 1e-9 and s.n_stress == 7
    assert g2.fishery_setting_for_n(6) == g1.Setting("fishery")
