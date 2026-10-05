import numpy as np

from experiments.oversight import run_s2_compliance_deterrence as s2
from experiments.oversight import run_s4_adaptive_memory_cost as s4

P = s2.profile("smoke")


def test_targeted_allowance_restores_planned_take():
    for scale, d, p in ((0.75, 0.5, 0.8), (0.5, 0.25, 0.6), (1.0, 0.75, 0.4)):
        a = s4.targeted_allowance(scale, d, p)
        assert np.isclose(a + d * (p - a), scale * p)
    assert s4.targeted_allowance(0.25, 0.75, 0.8) == 0.0  # cannot go below zero
    assert s4.targeted_allowance(0.5, 1.0, 0.8) == 0.0


def test_observed_overtake_recovers_level():
    a, p, d = 0.3, 0.9, 0.6
    assert np.isclose(s4.observed_overtake(a + d * (p - a), a, p), d)
    assert s4.observed_overtake(0.5, 0.5, 0.5) is None


def test_stop_reaction_ends_cheating_after_first_catch():
    e = s4.episode_a(0, "fine", 1 / 3, 1.0, "stop", P)
    assert 0 < e["stopped"] <= 4 and e["catches"] == e["stopped"]


def test_memory_flags_caught_cheaters_only():
    e = s4.episode_a(0, "memory", 1 / 3, 0.75, "continue", P)
    assert e["fines"] == 0 and 0 < e["flagged"] <= 4 and e["caught_honest"] == 0
