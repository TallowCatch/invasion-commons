"""Unit tests for each overseer part (overseer/DESIGN.md)."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.oversight import run_s5_stronger_attackers as s5
from experiments.oversight.claude_oversight_common import fishery_setup
from fishery_sim import calibrated_oversight as co
from overseer import Ledger, Overseer, WindowObs
from overseer import metrics as M
from overseer.adapters.fishery_text import FisheryText
from overseer.allowance import NoCap, SharedCut, TargetedCut
from overseer.audit import FixedCount, Targeted, UniformRandom
from overseer.budget import Budget
from overseer.detect import Noisy, Perfect
from overseer.sanction import Exclude, Fine, Leniency, Memory, targeted_allowance
from overseer.seeding import stable_seed

CFG = fishery_setup(0, 123, 20)[0]
GAME = FisheryText(CFG)
N = CFG.n_agents


def obs(t=0, stock=70.0, req=None, collapsed=False):
    from fishery_sim.fishery_oversight import FisherySnapshot
    req = np.full(N, 0.5) if req is None else np.asarray(req, float)
    return WindowObs(t, FisherySnapshot(stock, 0, collapsed), req)


def overseer(**kw):
    return Overseer(N, GAME.resource, **kw)


# ------------------------------------------------------------------ seeding
def test_stable_seed_is_the_runners_function():
    for parts in [(0, 1), (3, "E0", 2, "comp"), (5, 7, round(1 / 6, 6))]:
        assert stable_seed(1_965_000_000, *parts) == co.stable_seed(1_965_000_000, *parts)


# ------------------------------------------------------------------ allowances
def test_shared_cut_equals_fishery_choose_scale():
    rng = np.random.default_rng(0)
    ov = overseer(allowance=SharedCut())
    for _ in range(500):
        stock, req = float(rng.uniform(0, 100)), rng.uniform(0, 1, N)
        collapsed = bool(rng.random() < 0.05)
        d = ov.begin_window(obs(0, stock, req, collapsed))
        scale, _ = co.fishery_choose_scale(CFG, stock, req, "joint", "msy", collapsed)
        assert d.scale == scale and np.array_equal(d.caps, scale * req)


def test_safe_total_agrees_with_target_test():
    for stock in (20.0, 50.0, 70.0, 100.0):
        st = obs(0, stock).state
        cap = GAME.resource.safe_total(st)
        assert GAME.resource.within_target(st, cap) or stock < 50
        assert not GAME.resource.within_target(st, cap + 1e-6)
    assert GAME.resource.first_best_per_window() == pytest.approx(17.5)


def test_no_cap():
    assert overseer(allowance=NoCap()).begin_window(obs()).caps is None
    assert overseer().begin_window(obs()).caps is None


def test_targeted_cut_equals_s5_memory_cut():
    rng = np.random.default_rng(1)
    for _ in range(200):
        stock, req = float(rng.uniform(40, 100)), rng.uniform(0, 1, N)
        mem = Memory("share", N)
        flagged = rng.random(N) < 0.4
        dbar = np.zeros(N)
        for i in np.flatnonzero(flagged):
            mem.record[i] = [float(rng.uniform(0, 1))]
            dbar[i] = mem.record[i][0]
        ov = overseer(allowance=TargetedCut(), sanctions=(mem,))
        d = ov.begin_window(obs(0, stock, req))
        scale, _ = s5.memory_cut_scale(CFG, stock, req, dbar, flagged)
        assert d.scale == scale and np.array_equal(d.caps, scale * req)


def test_targeted_cut_needs_share_memory():
    with pytest.raises(ValueError):
        overseer(allowance=TargetedCut()).begin_window(obs())


# ------------------------------------------------------------------ audits
def test_uniform_audits_at_smaller_q_are_a_subset():
    qs = (1 / 12, 1 / 6, 1 / 3, 0.5)
    ov = overseer()
    for ctx in range(5):
        for t in range(40):
            sets = [UniformRandom(q, 1_965_000_000, prefix=(ctx,)).choose(obs(t), None, ov) for q in qs]
            for small, large in zip(sets, sets[1:]):
                assert not (small & ~large).any()


def test_uniform_audits_equal_l2_draws():
    ov = overseer()
    for ctx, t, q in [(0, 0, 1 / 6), (3, 7, 1 / 3), (9, 19, 1 / 12)]:
        want = np.random.default_rng(co.stable_seed(1_965_000_000, ctx, t)).random(N) < q
        assert np.array_equal(UniformRandom(q, 1_965_000_000, prefix=(ctx,)).choose(obs(t), None, ov), want)


def test_fixed_count_audits_exactly_k_like_s2():
    ov = overseer()
    for t in range(20):
        a = FixedCount(2, 845_000_000, prefix=("fishery", 4), suffix=("rand2",)).choose(obs(t), None, ov)
        idx = np.random.default_rng(co.stable_seed(845_000_000, "fishery", 4, t, "rand2")).permutation(N)[:2]
        assert a.sum() == 2 and set(np.flatnonzero(a)) == set(idx)


def test_targeted_audits_the_most_suspect():
    ov = overseer()
    ov.catches[:] = [0, 3, 1, 0, 5, 0]
    a = Targeted(2, 7).choose(obs(), None, ov)
    assert set(np.flatnonzero(a)) == {1, 4}
    a = Targeted(1, 7, score=lambda o, l, v: np.arange(N)).choose(obs(), None, ov)
    assert np.flatnonzero(a).tolist() == [N - 1]


# ------------------------------------------------------------------ detection
def ledger(taken, allowance):
    return Ledger(np.asarray(taken, float), np.asarray(allowance, float))


def test_perfect_finds_only_audited_over_takers_above_tolerance():
    from overseer.core import AuditPlan
    allow = np.full(N, 0.5)
    taken = allow + np.array([0.0, 0.005, 0.02, 0.5, 0.5, -0.1])
    plan = AuditPlan(np.array([1, 1, 1, 1, 0, 1], bool))
    f = Perfect(tol=0.01).detect(obs(), ledger(taken, allow), plan)
    assert f.over.tolist() == [False, False, True, True, False, False]
    assert np.allclose(f.excess, [0, 0, 0.02, 0.5, 0, 0])


def test_noisy_detection_limits_and_s3_draws():
    from overseer.core import AuditPlan
    allow, taken = np.zeros(N), np.r_[np.ones(3), np.zeros(3)]
    plan = AuditPlan(np.ones(N, bool))
    L_ = ledger(taken, allow)
    assert np.array_equal(Noisy(0, 0, 1).detect(obs(), L_, plan).over, Perfect().detect(obs(), L_, plan).over)
    assert not Noisy(1, 0, 1).detect(obs(), L_, plan).over.any()
    assert Noisy(1, 1, 1).detect(obs(), L_, plan).over.tolist() == [False] * 3 + [True] * 3
    for t in range(30):  # S3 Part B: caught = audited & over & (u < s), u from stable_seed(detect, context, t)
        u = np.random.default_rng(co.stable_seed(946_000_000, 2, t)).random(N)
        f = Noisy(0.5, 0, 946_000_000, prefix=(2,)).detect(obs(t), L_, plan)
        assert np.array_equal(f.over, (taken > 1e-12) & (u < 0.5))


# ------------------------------------------------------------------ sanctions
def test_fine_changes_score_only_not_stock():
    def play(F):
        ov = overseer(allowance=SharedCut(), audit=UniformRandom(1.0, 1), sanctions=(Fine(F),))
        state, path, caps = GAME.initial_state(), [], []
        for t in range(10):
            o = GAME.observe(t, state, np.full(N, 0.4))
            d = ov.begin_window(o)
            taken = d.caps.copy()
            taken[0] = min(1.0, taken[0] + 0.3)  # agent 0 over-takes every round and is always audited
            ov.end_window(o, Ledger(taken, d.caps, o.requests))
            caps.append(d.caps)
            state, _ = GAME.step(state, taken)
            path.append(state.stock)
        return ov, path, caps
    ov0, path0, caps0 = play(0.0)
    ov9, path9, caps9 = play(9.0)
    assert path0 == path9 and np.array_equal(caps0, caps9)
    assert ov0.fines.sum() == 0 and ov9.fines[0] == 90.0 and ov9.fines[1:].sum() == 0


def test_memory_share_rule():
    from overseer.core import Findings
    m = Memory("share", N)
    req, allow = np.full(N, 0.8), np.full(N, 0.4)
    taken = allow + np.array([0.2, 0.0, 0.8, 0, 0, 0])  # agent 0: half the cut; agent 2: more than all of it
    over = np.array([1, 0, 1, 0, 0, 0], bool)
    m.apply(obs(), Ledger(taken, allow, req), Findings(over, taken - allow))
    assert m.record[0] == [pytest.approx(0.5)] and m.record[2] == [1.0] and not m.record[1]
    caps = m.adjust(0.5 * req, 0.5, req)
    assert caps[0] == pytest.approx(targeted_allowance(0.5, 0.5, 0.8)) == 0.0
    assert caps[2] == 0.0 and caps[1] == 0.4
    m2 = Memory("share", N)  # nothing was cut (allowance = request): nothing is recorded
    m2.apply(obs(), Ledger(req + 0.1, req, req), Findings(np.ones(N, bool), np.full(N, 0.1)))
    assert not any(m2.record)


def test_memory_excess_rule():
    from overseer.core import Findings
    m = Memory("tonnes", N)  # the episode files' name for the rule
    assert m.rule == "excess"
    allow = np.full(N, 0.4)
    for extra in (0.2, 0.1):
        taken = allow + np.r_[extra, np.zeros(N - 1)]
        m.apply(obs(), Ledger(taken, allow, allow * 2), Findings(np.r_[True, np.zeros(N - 1, bool)], taken - allow))
    assert m.record[0] == [pytest.approx(0.2), pytest.approx(0.1)]
    caps = m.adjust(np.full(N, 0.3), 0.5, np.full(N, 0.6))
    assert caps[0] == pytest.approx(0.15) and (caps[1:] == 0.3).all()
    assert m.adjust(np.full(N, 0.1), 0.5, np.full(N, 0.2))[0] == 0.0  # never below 0
    with pytest.raises(ValueError):
        Memory("bogus")


def test_memory_lowers_later_allowances_through_shared_cut():
    ov = overseer(allowance=SharedCut(), audit=UniformRandom(1.0, 1), sanctions=(Memory("excess"),))
    o = obs(0, 100.0, np.full(N, 0.5))
    d = ov.begin_window(o)
    taken = d.caps + np.r_[0.25, np.zeros(N - 1)]
    out = ov.end_window(o, Ledger(taken, d.caps, o.requests))
    assert out.caught.tolist() == [True] + [False] * (N - 1) and ov.fines.sum() == 0
    d2 = ov.begin_window(obs(1, 100.0, np.full(N, 0.5)))
    assert d2.caps[0] == pytest.approx(d.caps[0] - 0.25) and np.array_equal(d2.caps[1:], d.caps[1:])


def test_exclude_for_some_windows():
    ov = overseer(allowance=SharedCut(), audit=UniformRandom(1.0, 1), sanctions=(Exclude(windows=2),))
    caps = []
    for t in range(5):
        o = obs(t, 100.0, np.full(N, 0.3))
        d = ov.begin_window(o)
        caps.append(d.caps[0])
        taken = d.caps + (np.r_[0.2, np.zeros(N - 1)] if t == 0 else 0.0)
        ov.end_window(o, Ledger(taken, d.caps, o.requests))
    assert caps[0] > 0 and caps[1] == caps[2] == 0.0 and caps[3] > 0 and caps[4] > 0


def test_leniency_is_a_stub():
    from overseer.core import Findings
    with pytest.raises(NotImplementedError):
        Leniency(6).apply(obs(), ledger(np.zeros(N), np.zeros(N)), Findings(np.zeros(N, bool), np.zeros(N)))


# ------------------------------------------------------------------ budget
def test_budget_percent_and_cost():
    ov = overseer(allowance=SharedCut(), audit=UniformRandom(1 / 3, 5, prefix=(0,)), budget=Budget(cost_per_audit=0.2))
    total = 0
    for t in range(30):
        o = obs(t, 100.0)
        d = ov.begin_window(o)
        out = ov.end_window(o, Ledger(d.caps, d.caps, o.requests))
        total += int(out.audited.sum())
        assert out.plan.cost == pytest.approx(0.2 * out.audited.sum())
    b = ov.budget
    assert b.audits == total and b.agent_windows == 30 * N and b.windows == 30
    assert b.percent == pytest.approx(100 * total / (30 * N)) and b.cost == pytest.approx(0.2 * total)
    assert M.audit_budget(b.audits, N, 30) * 100 == pytest.approx(b.percent)


def test_budget_caps():
    b = Budget(per_window=2, per_episode=3)
    a = b.limit(np.ones(N, bool))
    assert a.tolist() == [True, True] + [False] * (N - 2)
    b.charge(a)
    a = b.limit(np.ones(N, bool))
    assert a.sum() == 1
    b.charge(a)
    assert b.limit(np.ones(N, bool)).sum() == 0


# ------------------------------------------------------------------ metrics (W1 protocol, toy episode)
def test_metrics_on_a_toy_episode():
    fb = GAME.resource.first_best_per_window()
    w = M.welfare(harvest_total=350.0, fines_total=50.0)
    assert w == dict(W_transfer=350.0, W_lost=300.0)
    assert M.efficiency(w["W_transfer"], 20, fb) == pytest.approx(1.0)
    assert M.efficiency(175.0, 10, fb) == pytest.approx(1.0)  # collapsed after 10 rounds: actual length
    assert M.break_even_cost(300.0, 250.0, 20) == pytest.approx(2.5) and M.break_even_cost(1, 0, 0) is None
    assert M.oversight_efficiency(0.75, 0.5) == pytest.approx(0.5)
    assert M.oversight_efficiency(0.4, 0.5) < 0
    assert M.oversight_efficiency_net(300.0, 20, 0.0, 20, fb, 0.5) == pytest.approx(M.oversight_efficiency(300 / 350, 0.5))
    assert M.oversight_efficiency_net(300.0, 20, 0.2, 20, fb, 0.5) == pytest.approx(M.oversight_efficiency(296 / 350, 0.5))
    harm = [False] * 19 + [True]
    assert M.safety_per_round(harm) == pytest.approx(0.95)
    assert not M.safety_per_episode(harm, collapsed=False)  # 5% is not fewer than 5%
    assert M.safety_per_episode([False] * 20, collapsed=False) and not M.safety_per_episode([False] * 20, collapsed=True)
    assert M.harm_gap_recovered(0.8, 0.3, 0.3) == pytest.approx(1.0) and M.harm_gap_recovered(0.8, 0.8, 0.3) == 0
    assert M.usefulness(175.0, 20, fb) == pytest.approx(0.5)


def test_harm_line_is_l2_msy_break():
    st = obs(0, 70.0).state
    assert not GAME.resource.harm(st, np.r_[np.full(5, 20 / 30), 0.0])  # 20 t taken: 50 left
    assert GAME.resource.harm(st, np.full(N, 0.6))
