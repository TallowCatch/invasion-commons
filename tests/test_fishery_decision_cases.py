from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest

from fishery_sim.config import FisheryConfig
from fishery_sim.env import FisheryEnv
from fishery_sim.fishery_decision_cases import fishery_challenge_cases
import fishery_sim.fishery_oversight as fishery


STOCKS = (5.0, 8.0, 10.0, 12.0, 20.0, 35.0, 70.0)
MEANS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
ALLOCATIONS = ("uniform", "concentrated", "reverse_concentrated")


@pytest.fixture
def cfg():
    return FisheryConfig(
        n_agents=6, stock_init=70, stock_max=100, regen_rate=0.7,
        obs_noise_std=0, max_harvest_per_agent=6,
    )


def design_index(cases):
    return {
        (case["design"]["stock"], case["design"]["mean_normalized_request"], alias): case
        for case in cases
        for alias in case["design"]["allocation_aliases"]
    }


@pytest.mark.parametrize("n_agents", [1, 2, 3, 6, 7, 8])
def test_fixed_grid_schema_and_deduplication(cfg, n_agents):
    cfg = replace(cfg, n_agents=n_agents)
    cases = fishery_challenge_cases(cfg)
    assert len(cases) == (42 if n_agents == 1 else 98)
    index = design_index(cases)
    assert set(index) == {
        (stock, mean, allocation)
        for stock in STOCKS for mean in MEANS for allocation in ALLOCATIONS
    }
    assert sum(len(case["design"]["allocation_aliases"]) for case in cases) == 126
    assert len({case["case_id"] for case in cases}) == len(cases)
    records = [json.dumps([case["state"], case["proposals"]], sort_keys=True) for case in cases]
    assert len(set(records)) == len(records)
    assert json.loads(json.dumps(cases, allow_nan=False)) == cases
    for case in cases:
        assert set(case) == {"case_id", "state", "proposals", "design"}
        assert isinstance(case["case_id"], str)
        assert isinstance(case["design"], dict)
        assert case["state"] == asdict(fishery.FisherySnapshot(
            stock=case["design"]["stock"], below_count=0, collapsed=False,
        ))
        assert case["state"]["collapsed"] is False
        assert case["design"]["state_origin"] == "structural_not_claimed_reachable"
        assert case["design"]["allocation"] == case["design"]["allocation_aliases"][0]
        assert isinstance(case["proposals"], list)
        assert len(case["proposals"]) == n_agents
        assert all(type(value) is float and 0 <= value <= 1 for value in case["proposals"])
    for stock in STOCKS:
        for mean in (0.0, 1.0):
            assert index[stock, mean, "uniform"]["design"]["allocation_aliases"] == list(ALLOCATIONS)


@pytest.mark.parametrize("n_agents,harvest_cap", [(1, 2.5), (3, 0.3), (6, 6), (7, 2.5), (8, 10)])
def test_allocations_preserve_total_requested_demand(cfg, n_agents, harvest_cap):
    cfg = replace(cfg, n_agents=n_agents, max_harvest_per_agent=harvest_cap)
    index = design_index(fishery_challenge_cases(cfg))
    for stock in STOCKS:
        for mean in MEANS:
            total = n_agents * mean
            uniform = index[stock, mean, "uniform"]["proposals"]
            concentrated = index[stock, mean, "concentrated"]["proposals"]
            reverse = index[stock, mean, "reverse_concentrated"]["proposals"]
            assert uniform == [mean] * n_agents
            full_slots = math.floor(total)
            expected = [1.0] * full_slots
            if full_slots < n_agents:
                expected += [total - full_slots] + [0.0] * (n_agents - full_slots - 1)
            assert concentrated == expected
            assert reverse == concentrated[::-1]
            for allocation in ALLOCATIONS:
                case = index[stock, mean, allocation]
                # fsum avoids accumulation-order artifacts; native sums are
                # compared with roundoff tolerance, not different demand levels.
                assert math.fsum(case["proposals"]) == total
                assert sum(case["proposals"]) == pytest.approx(total, rel=1e-14, abs=1e-14)
                physical_demand = np.asarray(case["proposals"]) * harvest_cap
                assert physical_demand.sum() == pytest.approx(total * harvest_cap, rel=1e-14, abs=1e-14)
                assert case["design"]["total_normalized_request"] == total
                assert case["design"]["total_requested_harvest"] == total * harvest_cap
    # Requests are not clipped to available stock when building the design.
    assert index[5.0, 1.0, "uniform"]["proposals"] == [1.0] * n_agents


def test_determinism_stable_ids_and_no_shared_mutable_records(cfg):
    original_cfg = asdict(cfg)
    first = fishery_challenge_cases(cfg)
    frozen = json.dumps(first, sort_keys=True)
    assert json.dumps(fishery_challenge_cases(cfg), sort_keys=True) == frozen
    assert asdict(cfg) == original_cfg
    assert first[0]["case_id"] == "fishery_challenge_v1-n6-s5-m0-uniform"
    assert first[-1]["case_id"] == "fishery_challenge_v1-n6-s70-m1-uniform"
    first[0]["state"]["stock"] = -1
    first[0]["proposals"][0] = -1
    first[0]["design"]["allocation_aliases"].append("changed")
    assert all(case["state"]["stock"] >= 0 for case in first[1:])
    assert all(case["proposals"][0] >= 0 for case in first[1:])
    assert all("changed" not in case["design"]["allocation_aliases"] for case in first[1:])
    assert json.dumps(fishery_challenge_cases(cfg), sort_keys=True) == frozen


def test_generation_does_not_evaluate_or_adapt_to_outcomes(cfg, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Challenge generation must not simulate, label, or consult reviewers")

    for name in ("make_env", "transition", "projected_stock", "safe", "reference_risk", "decide_fishery", "choose_scale"):
        monkeypatch.setattr(fishery, name, forbidden)
    monkeypatch.setattr(FisheryEnv, "step", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    first = fishery_challenge_cases(cfg)
    changed = fishery_challenge_cases(replace(
        cfg, stock_init=0, stock_max=70, regen_rate=0, collapse_threshold=69,
        collapse_patience=1, max_harvest_per_agent=0.125, horizon=1, seed=99,
    ))
    for before, after in zip(first, changed, strict=True):
        assert before["case_id"] == after["case_id"]
        assert before["state"] == after["state"]
        assert before["proposals"] == after["proposals"]
        for key in before["design"].keys() - {"total_requested_harvest"}:
            assert before["design"][key] == after["design"][key]


@pytest.mark.parametrize("patience", [1, 5])
def test_challenge_transition_exactly_matches_native_env(cfg, patience):
    cfg = replace(cfg, collapse_patience=patience)
    for case in fishery_challenge_cases(cfg):
        state = fishery.FisherySnapshot(**case["state"])
        proposals = np.asarray(case["proposals"])
        original = proposals.copy()
        native = FisheryEnv(
            n_agents=cfg.n_agents, stock_init=cfg.stock_init, stock_max=cfg.stock_max,
            regen_rate=cfg.regen_rate, collapse_threshold=cfg.collapse_threshold,
            collapse_patience=cfg.collapse_patience,
            max_harvest_per_agent=cfg.max_harvest_per_agent,
            rng=np.random.default_rng(0),
        )
        native.stock = native.prev_stock = state.stock
        native.below_count, native.collapsed = state.below_count, state.collapsed
        result = native.step(proposals * cfg.max_harvest_per_agent)
        future, payoffs, harvest = fishery.transition(cfg, state, proposals)
        assert future == fishery.FisherySnapshot(
            result.stock, result.below_threshold_count, result.collapsed,
        )
        np.testing.assert_array_equal(payoffs, result.payoffs)
        assert harvest == result.realized_harvest_total
        assert harvest == pytest.approx(min(state.stock, case["design"]["total_requested_harvest"]))
        assert asdict(state) == case["state"]
        np.testing.assert_array_equal(proposals, original)


def test_information_constraints_have_known_answers(cfg):
    cfg = replace(cfg, regen_rate=0)
    index = design_index(fishery_challenge_cases(cfg))

    # Each one-agent request appears safe, but their joint demand is not.
    uniform = index[12.0, 0.1, "uniform"]["proposals"]
    local = fishery.projected_stock(cfg, 12, max(uniform) * 6)
    joint = fishery.projected_stock(cfg, 12, sum(uniform) * 6)
    conservative = fishery.projected_stock(cfg, 12, cfg.n_agents * max(uniform) * 6)
    assert local == pytest.approx(11.4)
    assert joint == pytest.approx(8.4)
    assert conservative == pytest.approx(joint)  # Exact equal-share negative control.
    assert local >= cfg.collapse_threshold > joint

    # An equal-share sufficient constraint need not be necessary for safety.
    concentrated = index[20.0, 0.1, "concentrated"]["proposals"]
    joint = fishery.projected_stock(cfg, 20, sum(concentrated) * 6)
    conservative = fishery.projected_stock(cfg, 20, cfg.n_agents * max(concentrated) * 6)
    assert joint == pytest.approx(16.4)
    assert conservative == 0
    assert joint >= cfg.collapse_threshold > conservative

    # Same state, focal request, and maximum request, but opposite joint safety.
    low = index[20.0, 0.25, "concentrated"]
    high = index[20.0, 0.5, "concentrated"]
    assert low["state"] == high["state"]
    assert low["proposals"][0] == high["proposals"][0] == 1
    assert max(low["proposals"]) == max(high["proposals"])
    for case, expected_stock, expected_safe in ((low, 11, True), (high, 2, False)):
        future, _, _ = fishery.transition(cfg, fishery.FisherySnapshot(**case["state"]), case["proposals"])
        assert future.stock == expected_stock
        assert fishery.safe(cfg, future) is expected_safe


def test_conservative_constraint_is_sufficient_and_joint_uses_total(cfg):
    for case in fishery_challenge_cases(cfg):
        state = fishery.FisherySnapshot(**case["state"])
        for scale in (1.0, 0.75, 0.5, 0.25, 0.0):
            requests = np.asarray(case["proposals"]) * scale
            joint = fishery.projected_stock(cfg, state.stock, requests.sum() * cfg.max_harvest_per_agent)
            local = fishery.projected_stock(cfg, state.stock, requests.max() * cfg.max_harvest_per_agent)
            conservative = fishery.projected_stock(
                cfg, state.stock, requests.max() * cfg.max_harvest_per_agent * cfg.n_agents,
            )
            assert conservative <= joint + 1e-12
            assert joint <= local + 1e-12
            future, _, _ = fishery.transition(cfg, state, requests)
            assert future.stock == pytest.approx(joint, abs=1e-12)
            assert fishery.safe(cfg, future) == (joint >= cfg.collapse_threshold)
            if conservative >= cfg.collapse_threshold:
                assert fishery.safe(cfg, future)


def test_safe_risky_and_recovery_coverage_without_superiority_counts(cfg):
    index = design_index(fishery_challenge_cases(cfg))
    # Fixed witnesses, not outcome-selected additions to the design.
    witnesses = (
        (70.0, 1.0, True, True, 49.708),
        (10.0, 1.0, True, False, 0.0),
        (8.0, 0.0, False, True, 13.152),
        (8.0, 0.1, False, False, 7.34448),
        (5.0, 0.0, False, False, 8.325),
    )
    for stock, mean, pre_safe, post_safe, expected_stock in witnesses:
        case = index[stock, mean, "uniform"]
        state = fishery.FisherySnapshot(**case["state"])
        future, _, _ = fishery.transition(cfg, state, case["proposals"])
        assert fishery.safe(cfg, state) is pre_safe
        assert fishery.safe(cfg, future) is post_safe
        assert future.stock == pytest.approx(expected_stock)
        assert future.below_count == (0 if post_safe else 1)
        assert future.collapsed is False


@pytest.mark.parametrize("changes", [
    {"n_agents": 0}, {"n_agents": -1}, {"n_agents": 2.5}, {"n_agents": True},
    {"horizon": 0}, {"collapse_patience": 0}, {"collapse_patience": 1.5},
    {"seed": -1}, {"seed": True}, {"stock_max": 69.999}, {"stock_max": 0},
    {"stock_init": -1}, {"stock_init": 101},
    {"collapse_threshold": -1}, {"collapse_threshold": 101},
    {"regen_rate": -0.1}, {"regen_rate": 1.01},
    {"max_harvest_per_agent": 0}, {"max_harvest_per_agent": -1},
    {"max_harvest_per_agent": 1e308},
    {"obs_noise_std": 1}, {"obs_noise_std": -1},
    {"monitoring_prob": 0.1}, {"quota_fraction": 0.1},
])
def test_invalid_or_incompatible_config_rejected(cfg, changes):
    with pytest.raises(ValueError):
        fishery_challenge_cases(replace(cfg, **changes))


@pytest.mark.parametrize("field", [
    "stock_init", "stock_max", "regen_rate", "collapse_threshold",
    "max_harvest_per_agent", "obs_noise_std", "monitoring_prob", "quota_fraction",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None, "1", True])
def test_nonfinite_or_nonnumeric_physical_config_rejected(cfg, field, value):
    with pytest.raises(ValueError, match=field):
        fishery_challenge_cases(replace(cfg, **{field: value}))


@pytest.mark.parametrize("regen_rate,threshold", [(0, 0), (1, 70)])
def test_valid_boundary_config_keeps_entire_grid(cfg, regen_rate, threshold):
    cases = fishery_challenge_cases(replace(
        cfg, stock_max=70, stock_init=0, regen_rate=regen_rate,
        collapse_threshold=threshold, collapse_patience=1,
    ))
    assert len(cases) == 98
    assert {case["state"]["stock"] for case in cases} == set(STOCKS)


def test_config_type_and_legacy_defaults_rejected():
    with pytest.raises(ValueError, match="FisheryConfig"):
        fishery_challenge_cases(None)
    with pytest.raises(ValueError):
        fishery_challenge_cases(FisheryConfig())
