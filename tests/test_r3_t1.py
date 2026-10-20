import numpy as np

from experiments.oversight import run_r2_robustness_reviewer_model as r2
from experiments.oversight import run_r3_breakeven as r3
from experiments.oversight import run_t1_audit_targeting as t1


def test_report_aimed_audit_picks_largest_believed_request():
    ov = t1.T1Overseer("rep1", 6, [0], 0, "fishery", t1.SEEDS)
    p = np.array([0.9, 0.5, 0.3, 0.2, 0.4, 0.1])  # agent 0 lies: reports 0.45, so agent 1 (0.5) is audited
    _, b, caught = ov.step(0, p)
    assert ov.audits == 1 and ov.audits_on_liars == 0 and not caught.any() and b[1] == 0.5


def test_trust_arm_never_audits():
    ov = t1.T1Overseer("report", 6, [0], 0, "fishery", t1.SEEDS)
    _, b, _ = ov.step(0, np.full(6, 0.6))
    assert ov.audits == 0 and np.isclose(b[0], 0.3)


def test_r3_prediction_uses_gain_per_overtake_step():
    P = r2.profile("smoke")
    pred = r3.predict(r2.Cell("fishery", 4, 0.7), P)
    gs = [v["g"] for v in pred["by_level"].values() if v["G"] > 0 and v["g"] is not None]
    assert np.isclose(pred["g_star"], max(gs))
