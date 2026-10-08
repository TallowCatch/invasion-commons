import numpy as np
import pytest

from fishery_sim.budgeted_oversight import (
    decide_budgeted_fishery, decide_budgeted_harvest,
    inspection_order, mask_requests,
)
from fishery_sim.config import FisheryConfig
from fishery_sim.fishery_oversight import FisherySnapshot, decide_fishery
from fishery_sim.harvest import HarvestCommonsConfig
from fishery_sim.oversight_protocol import MonitorSettings, decide_harvest


def test_nested_mask_and_no_hidden_action_leakage():
    n = 6
    actions = np.array([.1, .2, .3, .4, .5, .6])
    order = inspection_order(n, "same-context-step")
    three = mask_requests(actions, 3, "same-context-step")
    six = mask_requests(actions, 6, "same-context-step")
    assert np.array_equal(np.where(np.isfinite(three))[0], np.sort(order[:3]))
    np.testing.assert_equal(six, actions)
    changed = actions.copy()
    changed[order[3:]] = 1 - changed[order[3:]]
    np.testing.assert_equal(mask_requests(changed, 3, "same-context-step"), three)
    np.testing.assert_equal(order, inspection_order(n, "same-context-step"))


@pytest.mark.parametrize("mode", ["local_optimistic", "local_bounded", "joint", "local_coupled"])
def test_harvest_hidden_requests_cannot_change_verdict(mode):
    cfg = HarvestCommonsConfig()
    health = np.full(cfg.n_agents, 12.0)
    actions = np.array([.1, .2, .3, .4, .5, .6])
    visible = mask_requests(actions, 3, "frozen-case")
    baseline = decide_budgeted_harvest(cfg, health, visible, mode)
    changed = actions.copy()
    changed[np.isnan(visible)] = 1.0
    same_view = mask_requests(changed, 3, "frozen-case")
    assert decide_budgeted_harvest(cfg, health, same_view, mode) == baseline
    assert baseline.request_inspections == 3
    assert baseline.candidate_evaluations <= 5
    assert baseline.component_evaluations == cfg.n_agents * baseline.candidate_evaluations


@pytest.mark.parametrize("mode", ["local_optimistic", "local_bounded", "joint", "local_coupled"])
def test_fishery_hidden_requests_cannot_change_verdict(mode):
    cfg = FisheryConfig(n_agents=6, obs_noise_std=0, regen_rate=.7)
    state = FisherySnapshot(20)
    actions = np.array([.1, .2, .3, .4, .5, .6])
    visible = mask_requests(actions, 3, "frozen-case")
    baseline = decide_budgeted_fishery(cfg, state, visible, mode)
    changed = actions.copy()
    changed[np.isnan(visible)] = 1.0
    assert decide_budgeted_fishery(cfg, state, mask_requests(changed, 3, "frozen-case"), mode) == baseline
    assert baseline.request_inspections == 3


def test_full_information_matches_existing_predictions_and_authority():
    hcfg = HarvestCommonsConfig()
    health = np.full(hcfg.n_agents, 13.0)
    actions = np.linspace(.2, .9, hcfg.n_agents)
    settings = MonitorSettings()
    aliases = {"local_optimistic": "local_uncertain",
               "local_bounded": "local_conservative_uncertain", "joint": "joint_uncertain"}
    for mode, alias in aliases.items():
        new = decide_budgeted_harvest(hcfg, health, actions, mode, settings)
        old = decide_harvest(hcfg, health, actions, alias, settings)
        assert (new.scale, new.verdict, new.predicted_safe) == (
            old.scale, old.verdict, old.predicted_safe)
    fcfg = FisheryConfig(n_agents=6, obs_noise_std=0, regen_rate=.7)
    state = FisherySnapshot(19)
    aliases = {"local_optimistic": "local_nominal",
               "local_bounded": "local_conservative", "joint": "joint_nominal"}
    for mode, alias in aliases.items():
        new = decide_budgeted_fishery(fcfg, state, actions, mode, settings)
        old = decide_fishery(fcfg, state, actions, alias, settings)
        assert (new.scale, new.verdict, new.predicted_safe) == (
            old.scale, old.verdict, old.predicted_safe)


def test_invalid_visible_or_budget_is_rejected():
    cfg = HarvestCommonsConfig()
    with pytest.raises(ValueError):
        mask_requests(np.ones(6), 7, "x")
    with pytest.raises(ValueError):
        decide_budgeted_harvest(cfg, np.ones(6), np.full(6, np.inf), "joint")


@pytest.mark.parametrize("budget", [0, 3, 6])
def test_coupled_local_matches_joint_harvest_decision(budget):
    cfg = HarvestCommonsConfig()
    health = np.array([6.1, 9.4, 12.6, 7.8, 15.3, 10.2])
    requests = np.array([.15, .8, .3, .95, .5, .25])
    visible = mask_requests(requests, budget, "coupled-harvest")
    joint = decide_budgeted_harvest(cfg, health, visible, "joint")
    local = decide_budgeted_harvest(cfg, health, visible, "local_coupled")
    assert (joint.scale, joint.verdict, joint.status, joint.predicted_safe) == (
        local.scale, local.verdict, local.status, local.predicted_safe)
    assert local.request_inspections == joint.request_inspections == budget
    assert local.transmitted_scalars >= joint.transmitted_scalars


@pytest.mark.parametrize("budget", [0, 3, 6])
def test_coupled_local_matches_joint_fishery_decision(budget):
    cfg = FisheryConfig(n_agents=6, obs_noise_std=0, regen_rate=.7)
    state = FisherySnapshot(19)
    requests = np.array([.15, .8, .3, .95, .5, .25])
    visible = mask_requests(requests, budget, "coupled-fishery")
    joint = decide_budgeted_fishery(cfg, state, visible, "joint")
    local = decide_budgeted_fishery(cfg, state, visible, "local_coupled")
    assert (joint.scale, joint.verdict, joint.status, joint.predicted_safe) == (
        local.scale, local.verdict, local.status, local.predicted_safe)
    assert local.request_inspections == joint.request_inspections == budget
