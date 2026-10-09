import gzip, json
from pathlib import Path
import numpy as np
import pytest
from experiments.oversight.claude_oversight_common import fishery_setup, harvest_setup
from fishery_sim.calibrated_oversight import (fishery_choose_scale, fishery_reference, harvest_choose_scale,
                                              harvest_reference, SCALES)

RUN = Path("results/runs/budgeted_reviewer_confirmation_v1/blocks")


@pytest.mark.skipif(not RUN.exists(), reason="confirmation archive not unpacked")
def test_setup_reproduces_confirmation_populations():
    b = json.load(gzip.open(RUN / "fishery__mix4__deterministic__3__0__joint__6.json.gz"))
    _, pol, _ = fishery_setup(3, 500_000_000)
    assert np.allclose(pol["low"], b["policies"]["low"]) and np.allclose(pol["thresholds"], b["policies"]["thresholds"])
    h = json.load(gzip.open(RUN / "harvest__mix2__slow_regen__3__0__joint__6.json.gz"))
    cfg, specs, _ = harvest_setup(3, 510_000_000, 520_000_000)
    from dataclasses import asdict
    for mine, saved in zip(specs, h["policies"]):
        for key, value in asdict(mine).items():
            assert (abs(value - saved[key]) < 1e-12) if isinstance(value, float) else value == saved[key]
    assert abs(cfg.regen_rate - h["config"]["regen_rate"]) < 1e-12 and cfg.seed == h["config"]["seed"]


def test_fishery_joint_full_information_equals_reference():
    cfg, pol, _ = fishery_setup(0, 1)
    rng = np.random.default_rng(0)
    for _ in range(300):
        stock, req = rng.uniform(5, 100), rng.uniform(0, 1, 6)
        for target in ("one_step", "msy"):
            s, _ = fishery_choose_scale(cfg, stock, req, "joint", target)
            assert (s == 1.0) == (fishery_reference(cfg, stock, req, target)["label"] == "safe")


def test_harvest_bounded_never_less_cautious_than_joint():
    cfg, _, _ = harvest_setup(0, 1, 2)
    rng = np.random.default_rng(1)
    for i in range(60):
        h, req = rng.uniform(6, 16, 6), rng.uniform(0, 1, 6)
        sj = harvest_choose_scale(cfg, h, req, "joint", seed=i)[0]
        sb = harvest_choose_scale(cfg, h, req, "local_bounded", seed=i)[0]
        assert sb <= sj


def test_harvest_reference_labels_extremes():
    cfg, _, _ = harvest_setup(0, 1, 2)
    assert harvest_reference(cfg, np.full(6, 18.0), np.zeros(6), seed=1)["label"] == "safe"
    assert harvest_reference(cfg, np.full(6, 8.0), np.ones(6), seed=1)["label"] == "risky"
