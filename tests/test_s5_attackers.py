"""Engineering gates for S5 (protocol file 20)."""
import numpy as np
import pytest

from experiments.oversight import run_s2_compliance_deterrence as s2
from experiments.oversight import run_s4_adaptive_memory_cost as s4
from experiments.oversight import run_s5_stronger_attackers as s5
from experiments.oversight.claude_oversight_common import fishery_setup
from fishery_sim.calibrated_oversight import SCALES, fishery_choose_scale, fishery_target_ok
from fishery_sim.fishery_oversight import FisherySnapshot, transition

FULL = s2.profile("full")  # horizon 80, as S4
SMOKE = s5.profile("smoke")
COMMON = ("total_harvest", "honest_harvest", "cheater_harvest", "cheater_payoff", "fines", "mean_health", "t_end",
          "unsafe_fixed", "scored_steps", "exec_risky", "exec_unresolved", "audits", "catches", "caught_honest",
          "cheater_over_steps", "flagged")


def strip(e):
    return {k: v for k, v in e.items() if k not in ("tier", "split")}


# ------------------------------------------------------------- environment
def test_fast_transition_matches_transition():
    cfg, _, _ = fishery_setup(0, 1, 80)
    rng = np.random.default_rng(0)
    for _ in range(3000):
        st = FisherySnapshot(float(rng.uniform(0, 100)), int(rng.integers(0, 5)), bool(rng.random() < 0.05))
        req = rng.uniform(0, 1, cfg.n_agents) * (rng.random(cfg.n_agents) < 0.9)
        a, pa, ha = transition(cfg, st, req)
        b, pb, hb = s5.fast_transition(cfg, st, req)
        assert (a.stock, a.below_count, a.collapsed) == (b.stock, b.below_count, b.collapsed)
        assert np.array_equal(pa, pb) and ha == hb


# ------------------------------------------------------------- memory_cut
@pytest.mark.parametrize("stock", [55.0, 70.0, 85.0, 100.0])
def test_memory_cut_expected_take_meets_target(stock):
    cfg, _, _ = fishery_setup(0, 1, 80)
    p = np.array([0.6, 0.9, 0.3, 0.8, 0.2, 0.7])
    flagged = np.array([True, True, False, True, False, False])
    dbar = np.array([0.75, 1.0, 0.0, 0.5, 0.0, 0.0])
    scale, _ = s5.memory_cut_scale(cfg, stock, p, dbar, flagged)
    expected = lambda s: float(np.where(flagged, p * s + dbar * (p - p * s), p * s).sum() * cfg.max_harvest_per_agent)
    if scale > 0 or fishery_target_ok(cfg, stock, expected(0.0), "msy"):
        assert fishery_target_ok(cfg, stock, expected(scale), "msy")
    for s in SCALES:  # it is the largest menu scale that meets the target
        if s > scale:
            assert not fishery_target_ok(cfg, stock, expected(s), "msy")
    # the cut is never looser than the plain reviewer's
    assert scale <= fishery_choose_scale(cfg, stock, p, "joint", "msy")[0]


def test_memory_cut_without_flags_is_plain_reviewer():
    cfg, _, _ = fishery_setup(0, 1, 80)
    rng = np.random.default_rng(1)
    for _ in range(500):
        p, stock = rng.uniform(0.05, 0.95, 6), float(rng.uniform(40, 100))
        assert s5.memory_cut_scale(cfg, stock, p, np.zeros(6), np.zeros(6, bool)) == \
            fishery_choose_scale(cfg, stock, p, "joint", "msy")


def test_memory_cut_episode_equals_fine_before_any_catch():
    a = s5.run_episode(0, "memory_cut", 0.02, s5.T0Policy(0.0), SMOKE, trace=True)
    b = s5.run_episode(0, "none", None, s5.T0Policy(0.0), SMOKE, trace=True)
    assert [r["scale"] for r in a["trace"]] == [r["scale"] for r in b["trace"]]


# ------------------------------------------------------------- memory_cap
def test_memory_cap_enforces_allowance_after_catch():
    e = s5.run_episode(1, "memory_cap", 1 / 6, s5.T0Policy(1.0), SMOKE, trace=True)
    tr, n_enforced, first = e["trace"], 0, {}
    for r in tr:
        for i in range(6):
            if r["capped_before"][i]:
                n_enforced += 1
                assert r["taken"][i] == r["allowance"][i] and r["audited"][i] and not r["caught"][i]
            elif r["caught"][i]:
                first.setdefault(i, r["t"])
    assert first and e["capped"] == len(first) and e["fines"] == 0
    assert n_enforced == e["enforced_checks"] > 0
    # audits = Bernoulli audits of uncapped agents + one check per capped agent-step
    assert e["audits"] == sum(sum(r["audited"]) for r in tr)
    for i, t in first.items():  # enforcement starts the step after the first catch
        assert all(r["capped_before"][i] == (r["t"] > t) for r in tr)


# ------------------------------------------------------------- T1 reproduces T0
@pytest.mark.parametrize("regime,q", [("none", None), ("fine", 0.05), ("memory", 1 / 6), ("memory_cut", 0.10),
                                      ("memory_cap", 1 / 6), ("fine+memory_cut", 0.02)])
@pytest.mark.parametrize("d,reaction", [(0.0, "continue"), (0.75, "continue"), (1.0, "stop"), (0.5, "stop")])
def test_t1_at_t0_level_reproduces_t0(regime, q, d, reaction):
    for train in (True, False):
        for c in range(2):
            a = s5.run_episode(c, regime, q, s5.T0Policy(d, reaction), SMOKE, train=train)
            b = s5.run_episode(c, regime, q, s5.T1Policy.from_t0(d, reaction), SMOKE, train=train)
            assert a == b


def test_lie_low_takes_allowance_after_own_audit():
    pol = s5.T1Policy([[1.0, 1.0, 1.0, 1.0]] * 4, [5] * 4)
    e = s5.run_episode(2, "fine", 1 / 6, pol, SMOKE, trace=True)
    ep = s5.Episode(2, "fine", 1 / 6, SMOKE)
    cheat, last = ep.cheat, {i: None for i in ep.cheat}
    for r in e["trace"]:
        for i in cheat:
            low = last[i] is not None and 1 <= r["t"] - last[i] <= 5
            assert (r["taken"][i] == r["allowance"][i]) == (low or r["req"][i] == r["allowance"][i])
            if r["audited"][i]:
                last[i] = r["t"]


# ------------------------------------------------------------- replication gate (S4)
@pytest.mark.parametrize("regime,q,d,reaction", [("memory", 1 / 6, 0.75, "continue"), ("fine", 0.05, 0.75, "continue"),
                                                 ("memory", 0.02, 1.0, "stop"), ("none", None, 0.75, "continue")])
def test_episodes_equal_s4(regime, q, d, reaction):
    for train in (True, False):
        for c in (0, 3, 7):
            a = s4.episode_a(c, regime, q, d, reaction, FULL, train=train)
            b = s5.run_episode(c, regime, q, s5.T0Policy(d, reaction), FULL, seeds=s4.SEEDS, train=train)
            for k in COMMON:
                assert a[k] == b[k], (k, c, train)


def test_replication_gate_s4_memory_q_one_sixth():
    """S4 (file 18): under memory at q = 1/6 the group chose d* = 0.75 and 'continue' on S4's 8 training contexts."""
    (d, r), score, scores = s5.search_t0("memory", 1 / 6, FULL, seeds=s4.SEEDS, train_contexts=range(8))
    assert (d, r) == (0.75, "continue")
    (d4, r4), sc4 = s4.search_a("memory", 1 / 6, FULL)
    assert (d4, r4) == (d, r) and sc4 == scores


# ------------------------------------------------------------- T2 (PPO) plumbing
def test_t2_lockstep_evaluation_equals_single_episodes():
    torch = pytest.importorskip("torch")
    from experiments.oversight import s5_ppo
    torch.manual_seed(0)
    state = s5_ppo.ActorCritic(s5_ppo.HYPERPARAMS).state_dict()
    for mode in ("sample", "greedy"):
        pol = s5_ppo.TrainedPolicy(state, mode=mode)
        batch = s5_ppo.evaluate_lockstep("fine", 0.05, pol, SMOKE, range(3), train=True)
        for c in range(3):
            rng = s5_ppo.eval_rng(c, True)
            ep = s5.Episode(c, "fine", 0.05, SMOKE, train=True)
            while not ep.done:
                ep.step(pol.act(ep.observe(), rng))
            assert ep.summary() == batch[c]


def test_t2_training_is_deterministic():
    pytest.importorskip("torch")
    from experiments.oversight import s5_ppo
    P = dict(SMOKE, ppo_episodes=8, ppo_batch=4, ppo_eval_every=1, train_contexts=2)
    runs = [s5_ppo.train("memory_cap", 1 / 6, P, s5_ppo.HYPERPARAMS, 7, log=lambda *a, **k: None) for _ in range(2)]
    assert runs[0]["log"] == runs[1]["log"]
    assert all(np.array_equal(runs[0]["best_state"][k].numpy(), runs[1]["best_state"][k].numpy())
               for k in runs[0]["best_state"])
