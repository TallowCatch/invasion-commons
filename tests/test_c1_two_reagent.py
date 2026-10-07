"""Engineering gates for C1 (protocol 21): the two-reagent river."""
from __future__ import annotations

import numpy as np

from fishery_sim import two_reagent as TR
from fishery_sim.calibrated_oversight import SCALES


def test_kappa_zero_damage_functions_agree():
    rng = np.random.default_rng(0)
    for lam in (0.3, 1.6, 7.0):
        comp = TR.Params("comp", 0.0, lam, lam, 1.0, 3.0)
        add = TR.Params("add", 0.0, lam, lam, 1.0, 3.0)
        for _ in range(200):
            p = rng.uniform(0, 1, TR.N_AGENTS)
            assert np.isclose(TR.damage_of(comp, p), TR.damage_of(add, p))
            q = rng.uniform(0, 100)
            assert np.isclose(TR.next_q(comp, q, p, 0.7), TR.next_q(add, q, p, 0.7))
            assert np.isclose(TR.exact_risk(comp, q, p), TR.exact_risk(add, q, p))


def test_compositional_damage_formula():
    P = TR.Params("comp", 32.0, 1.6, 22.9, 1.0, 3.0)
    p = np.array([0.2, 0.3, 0.5, 0.1, 0.1, 0.4])
    assert np.isclose(TR.damage_of(P, p), 32 * 1.0 * 0.6 + 1.6 * 1.6)


def test_matching_rule_equalises_safe_total_at_q80():
    P = TR.FROZEN
    A = TR.frozen_params("add")
    h = TR.exact_headroom(P.r, P.sigma, 80.0)
    assert np.isclose(TR.total_for_damage(P, h), TR.total_for_damage(A, h))


def test_joint_is_exact_with_sigma_zero():
    rng = np.random.default_rng(1)
    for game in TR.GAMES:
        P = TR.Params(game, 32.0, 1.6, 22.9, 1.0, 0.0)
        eps = TR.reviewer_draws(P, 5, 400)
        assert np.all(eps == 0)
        checked = 0
        for _ in range(500):
            q = rng.uniform(30, 100)
            req = rng.uniform(0, 1, TR.N_AGENTS)
            s, _, _ = TR.choose_scale(P, q, req, "joint", eps)
            # the executed action is truly safe ...
            assert TR.next_q(P, q, req * s, 0.0) >= TR.Q_UNSAFE
            # ... and every larger scale on the menu is truly unsafe (no unnecessary cut)
            for larger in [x for x in SCALES if x > s]:
                assert TR.next_q(P, q, req * larger, 0.0) < TR.Q_UNSAFE
                checked += 1
            # the reference label agrees with the truth
            ref = TR.reference(P, q, req * s, 3, draws=2000)
            truth_safe = TR.next_q(P, q, req * s, 0.0) >= TR.Q_UNSAFE
            assert ref["label"] == ("safe" if truth_safe else "risky")
        assert checked > 50


def test_quota_caps_at_safe_typical_total():
    P = TR.Params("comp", 32.0, 1.6, 22.9, 1.0, 0.0)
    eps = np.zeros(400)
    q = 60.0
    req = np.array([0.9, 0.1, 0.1, 0.9, 0.1, 0.1])
    dec = TR.review(P, q, req, req, "quota", eps)
    cap = dec["cap"]
    s_total = 6 * cap
    # the equal-mix total S is exactly at the safe boundary
    assert np.isclose(TR.next_q(P, q, np.full(6, s_total / 6), 0.0), TR.Q_UNSAFE)
    assert np.allclose(dec["executed"], np.minimum(req, cap))
    assert dec["approved"] == 0


def test_targeted_audit_picks_most_sensitive_agent():
    P = TR.Params("comp", 32.0, 1.6, 22.9, 1.0, 3.0)
    # X total 1.2 (largest X report 0.6), Y total 2.0 (largest Y report 0.9)
    b = np.array([0.6, 0.3, 0.3, 0.9, 0.6, 0.5])
    x, y = 1.2, 2.0
    scores = [(32 * y + 1.6) * b[i] if i < 3 else (32 * x + 1.6) * b[i] for i in range(6)]
    assert TR.targeted_agent(P, b) == int(np.argmax(scores)) == 0  # X agent facing the large Y total
    # numerical check: agent whose proportional report error moves predicted damage most
    for _ in range(100):
        b = np.random.default_rng(_).uniform(0, 1, 6)
        moves = []
        for i in range(6):
            bb = b.copy()
            bb[i] *= 1 + 1e-6
            moves.append(TR.damage_of(P, bb) - TR.damage_of(P, b))
        assert TR.targeted_agent(P, b) == int(np.argmax(moves))
    A = TR.Params("add", 32.0, 1.6, 22.9, 1.0, 3.0)
    b = np.array([0.2, 0.7, 0.1, 0.5, 0.3, 0.6])
    assert TR.targeted_agent(A, b) == 1


def test_memory_belief_corrects_fixed_liar():
    reports = np.array([0.2, 0.4, 0.3, 0.25, 0.1, 0.1])
    records = {1: [0.5]}
    b = TR.memory_belief(reports, records)
    assert np.isclose(b[1], 0.8) and np.isclose(b[0], 0.2)
    truth = np.array([0.2, 0.8, 0.3, 0.5, 0.1, 0.1])
    b = TR.memory_belief(reports, records, audited=3, true_requests=truth)
    assert np.isclose(b[3], 0.5)


def test_local_optimistic_ignores_other_type():
    P = TR.Params("comp", 32.0, 1.6, 22.9, 1.0, 3.0)
    b = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    assert np.isclose(TR.predicted_damage(P, b, 1.0, "local_optimistic"), 1.6 * 0.9)
    assert np.isclose(TR.predicted_damage(P, b, 1.0, "local_bounded"), 32 * 0.9 * 3 + 1.6 * 3.9)
