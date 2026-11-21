import numpy as np

from experiments.oversight import run_s2_compliance_deterrence as s2
from experiments.oversight import run_s3_threshold_timing_memory as s3

P = s2.profile("smoke")


def test_bernoulli_audits_shared_across_fines_and_levels():
    allowance, taken = np.zeros(6), np.ones(6)
    for t in range(10):
        a = s3.Auditor("bern", 6, [0, 1], 0.75, 3.0, 0, "fishery", s3.SEEDS, q=0.5).audit(t, allowance, taken)
        b = s3.Auditor("bern", 6, [0, 1], 1.0, 30.0, 0, "fishery", s3.SEEDS, q=0.5).audit(t, allowance, taken)
        assert (a == b).all()


def test_avoid_cheaters_never_caught_by_periodic_schedule():
    e = s3.fishery_episode(1, "periodic6", 1.0, 24.0, P, s3.SEEDS, timing="avoid")
    assert e["catches"] == 0 and e["cheater_over_steps"] > 0
    always = s3.fishery_episode(1, "periodic6", 1.0, 24.0, P, s3.SEEDS, timing="always")
    assert always["catches"] > 0


def test_avoid_equals_always_under_random_audits():
    a = s3.fishery_episode(0, "bern", 0.5, 6.0, P, s3.SEEDS, q=1 / 6, timing="avoid")
    b = s3.fishery_episode(0, "bern", 0.5, 6.0, P, s3.SEEDS, q=1 / 6, timing="always")
    assert {k: v for k, v in a.items() if k != "timing"} == {k: v for k, v in b.items() if k != "timing"}


def test_imperfect_detection_never_catches_more():
    full = s3.fishery_episode(0, "bern", 1.0, 6.0, P, s3.SEEDS, q=0.5, s=1.0)
    half = s3.fishery_episode(0, "bern", 1.0, 6.0, P, s3.SEEDS, q=0.5, s=0.5)
    assert half["catches"] <= full["catches"]


def test_memory_corrects_a_caught_fixed_liar():
    ov = s3.MemoryOverseer("rand2", 6, [0, 1], 0.5, 0, "fishery", "memory", "fixed")
    ov.dhat[0] = [0.5]
    p = np.full(6, 0.8)
    _, b, _ = ov.step(0, p)
    assert np.isclose(b[0], 0.8)  # either audited (truth) or corrected r/(1-0.5) = 0.4/0.5
    assert b[1] in (0.4, 0.8)  # uncaught liar is believed unless audited this step


def test_memory_max_believes_maximum_after_catch():
    ov = s3.MemoryOverseer("report", 6, [0], 0.5, 0, "fishery", "memory_max", "fixed")
    ov.dhat[0] = [0.5]
    _, b, _ = ov.step(0, np.full(6, 0.3))
    assert b[0] == 1.0 and np.allclose(b[1:], 0.3)
