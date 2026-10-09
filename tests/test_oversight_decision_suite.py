"""The suite must preserve declared cases, native physics and matched inputs."""
import json
from dataclasses import asdict

import numpy as np
import pytest

from fishery_sim.harvest import HarvestCommonsConfig, SelfInterestedHarvestAgent, run_harvest_episode
from fishery_sim.harvest_decision_cases import harvest_challenge_cases, validate_native_replay
from fishery_sim.harvest_benchmarks import make_harvest_cfg_for_scenario
from fishery_sim.config import FisheryConfig
from fishery_sim.oversight_protocol import METHODS, harvest_safe
from experiments.archive.harvest_2026q2.run_oversight_decision_suite import collect_cases, evaluate_case, fingerprint, label_case
from experiments.common.run_matched_oversight import (
    FISHERY_METHODS, make_manifest, digest, read_json, write_json,
)
from fishery_sim.oversight_protocol import MonitorSettings


@pytest.fixture
def pilot_fixture(tmp_path):
    """Enough immutable prior-run structure to test collection without local data."""
    cfg = make_manifest(tmp_path, "smoke", 128, MonitorSettings())["protocol"]
    names = [f"harvest__{regime}__0__0__{method}"
             for regime in cfg["regimes"] for method in METHODS]
    names += [f"fishery__0__{method}" for method in FISHERY_METHODS]
    fish = asdict(FisheryConfig(n_agents=6, stock_init=70, stock_max=100,
        regen_rate=.7, obs_noise_std=0, max_harvest_per_agent=6))
    for name in names:
        game, regime = name.split("__")[:2]
        if game == "harvest":
            scenario = make_harvest_cfg_for_scenario("forest_co_management", horizon=12,
                seed=321, communication_enabled=False, side_payments_enabled=False)
            if regime == "slow_regen":
                scenario.regen_rate *= .85
            config = asdict(scenario)
            trace = [dict(step=0, patch_health_before_json=json.dumps([14.] * 6),
                          requested_fracs_json=json.dumps([.2] * 6))]
        else:
            regime = "deterministic"
            config = fish
            trace = [dict(step=0, state={"stock": 70., "below_count": 0, "collapsed": False},
                          requested_fracs_json=json.dumps([.2] * 6))]
        write_json(tmp_path / "blocks" / f"{name}.json.gz", dict(game=game, regime=regime,
            scenario="forest_co_management" if game == "harvest" else "single_stock",
            method=name.split("__")[-1], config=config, trace=trace, context=0, weather=0))
    write_json(tmp_path / "block_checksums.json", {
        name: digest(tmp_path / "blocks" / f"{name}.json.gz") for name in names})
    write_json(tmp_path / "frozen_judgments.json.gz", [])
    write_json(tmp_path / "completion.json", dict(expected_blocks=len(names),
        completed_blocks=len(names), frozen_judgment_rows=0))
    return tmp_path


def test_harvest_fixed_grid_is_deduplicated_and_preserves_demand():
    cfg = HarvestCommonsConfig(communication_enabled=False, side_payments_enabled=False)
    cases = harvest_challenge_cases(cfg)
    assert len(cases) == 238
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert sum(len(case["design"]["aliases"]) for case in cases) == 5 * 4 * 6 * 3
    assert json.loads(json.dumps(cases, allow_nan=False)) == cases
    for case in cases:
        health, proposals = np.asarray(case["state"]), np.asarray(case["proposals"])
        assert health.shape == proposals.shape == (6,)
        assert np.all((0 <= health) & (health <= cfg.patch_max))
        assert np.all((0 <= proposals) & (proposals <= 1))
        for alias in case["design"]["aliases"]:
            assert proposals.sum() == pytest.approx(cfg.n_agents * alias["demand_mean"])
        assert case["design"]["actual_mean_health"] == pytest.approx(health.mean())
    depleted = [c for c in cases if any(a["health_profile"] == "half_depleted"
                                      for a in c["design"]["aliases"])]
    assert depleted and all(not harvest_safe(cfg, c["state"]) for c in depleted)


def test_native_replay_checks_nonuniform_state_and_extraction():
    cfg = HarvestCommonsConfig(horizon=1, seed=48, communication_enabled=False,
                               side_payments_enabled=False)
    health = np.array([7., 8., 10., 12., 18., 19.])
    actions = np.array([1., .1, .7, .25, .8, 0.])
    before = health.copy()
    parity = validate_native_replay(cfg, health, actions, 504)
    assert parity["native_parity"]
    assert parity["extracted"] == pytest.approx(np.minimum(health, actions * cfg.max_harvest_per_agent).sum())
    np.testing.assert_array_equal(health, before)
    with pytest.raises(ValueError, match="Initial patch health"):
        validate_native_replay(cfg, [-1.] * 6, actions, 504)


def test_optional_initial_state_preserves_default_episode():
    cfg = HarvestCommonsConfig(horizon=4, seed=49, communication_enabled=False,
                               side_payments_enabled=False)
    def agents():
        return [SelfInterestedHarvestAgent() for _ in range(cfg.n_agents)]
    default = run_harvest_episode(cfg, agents(), record_trace=True)
    explicit = run_harvest_episode(cfg, agents(), record_trace=True,
                                   initial_patch_health=np.full(cfg.n_agents, cfg.patch_init))
    assert default["episode_trace_rows"] == explicit["episode_trace_rows"]
    assert default["total_welfare"] == explicit["total_welfare"]


def test_original_proposals_and_method_pairing_from_frozen_source(pilot_fixture):
    cases = collect_cases(pilot_fixture)
    assert len(cases) == 577
    assert len({c["case_id"] for c in cases}) == len(cases)
    recorded = [c for c in cases if c["source"] == "recorded"]
    assert len(recorded) == 3
    assert all(c["design"]["block"].endswith("__none") for c in recorded)
    assert all(c["design"]["step"] in (0, 5, 15, 30, 50, 70) for c in recorded)
    assert sum(c["game"] == "fishery" and c["source"] == "structural" for c in cases) == 98
    cfg = make_harvest_cfg_for_scenario("forest_co_management", horizon=80, seed=0,
        communication_enabled=False, side_payments_enabled=False)
    assert len(harvest_challenge_cases(cfg)) == 238
    first = next(c for c in cases if c["game"] == "harvest")
    altered = {**first, "config": {**first["config"], "seed": first["config"]["seed"] + 1}}
    assert fingerprint(first) == fingerprint(altered)
    altered = {**first, "proposals": [0.] * len(first["proposals"])}
    assert fingerprint(first) != fingerprint(altered) or not any(first["proposals"])
    selected = next(c for c in cases if c["game"] == "fishery" and c["source"] == "structural")
    label = label_case(selected)
    rows = evaluate_case(selected, label)
    assert set(row["method"] for row in rows) == set(FISHERY_METHODS)
    assert len({(row["reference_label"], row["original_risk"]) for row in rows}) == 1
    assert all(row["original_extraction"] >= row["retained_extraction"] for row in rows)


def test_still_includes_strong_local_harvest_controls(pilot_fixture):
    cases = collect_cases(pilot_fixture)
    selected = next(c for c in cases if c["game"] == "harvest" and c["source"] == "structural")
    label = label_case(selected)
    rows = evaluate_case(selected, label)
    assert set(row["method"] for row in rows) == set(METHODS)
    assert "local_uncertain" in {row["method"] for row in rows}
    assert all(row["reference_label"] == label["reference_label"] for row in rows)


def test_source_name_and_embedded_method_must_agree_even_with_valid_checksum(pilot_fixture):
    name = "fishery__0__none"
    target = pilot_fixture / "blocks" / f"{name}.json.gz"
    block = read_json(target)
    block["method"] = "joint_nominal"
    write_json(target, block)
    checksums = read_json(pilot_fixture / "block_checksums.json")
    checksums[name] = digest(target)
    write_json(pilot_fixture / "block_checksums.json", checksums)
    with pytest.raises(ValueError, match="Source block identity"):
        collect_cases(pilot_fixture)
