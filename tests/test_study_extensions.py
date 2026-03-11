import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishery_sim.agents import GreedyAgent
from fishery_sim.config import FisheryConfig
from fishery_sim.evolution import make_strategy_injector
from fishery_sim.evolution import run_evolutionary_invasion
from fishery_sim.harvest import CreditSharingHarvestAgent
from fishery_sim.harvest import GovernmentAgent
from fishery_sim.harvest import HarvestCommonsConfig
from fishery_sim.harvest import HarvestMessage
from fishery_sim.harvest import HarvestObservation
from fishery_sim.harvest import ReciprocalHarvestAgent
from fishery_sim.harvest import SelfInterestedHarvestAgent
from fishery_sim.harvest import run_harvest_episode
from fishery_sim.harvest_benchmarks import get_harvest_tier_preset
from fishery_sim.harvest_benchmarks import make_harvest_cfg_for_tier
from fishery_sim.simulation import run_episode


def test_mechanism_logging_tracks_clipping_and_sanctions() -> None:
    cfg = FisheryConfig(
        n_agents=2,
        horizon=12,
        stock_init=100.0,
        stock_max=150.0,
        regen_rate=1.2,
        collapse_threshold=10.0,
        collapse_patience=4,
        obs_noise_std=0.0,
        max_harvest_per_agent=10.0,
        monitoring_prob=1.0,
        quota_fraction=0.03,
        base_fine_rate=2.0,
        fine_growth=0.5,
        seed=7,
    )
    out = run_episode(cfg, [GreedyAgent(max_h=10.0), GreedyAgent(max_h=10.0)])
    assert out["mean_requested_harvest"] > out["mean_realized_harvest"]
    assert out["mean_audit_rate"] == 1.0
    assert out["mean_quota_clipped_total"] > 0.0
    assert out["sanction_total"] > 0.0
    assert out["violation_events"] > 0


def test_new_injectors_work_with_partner_mix_presets() -> None:
    base_cfg = FisheryConfig(
        n_agents=6,
        horizon=24,
        stock_init=100.0,
        stock_max=180.0,
        regen_rate=1.5,
        collapse_threshold=10.0,
        collapse_patience=4,
        obs_noise_std=4.0,
        max_harvest_per_agent=8.0,
        seed=3,
    )
    for mode in ("random", "adversarial_heuristic", "search_mutation"):
        generation_df, strategy_df = run_evolutionary_invasion(
            base_cfg=copy.deepcopy(base_cfg),
            generations=3,
            population_size=6,
            seeds_per_generation=3,
            replacement_fraction=0.33,
            adversarial_pressure=0.5,
            rng_seed=11,
            partner_mix_preset="cooperative_heavy",
            injector=make_strategy_injector(mode),
        )
        assert len(generation_df) == 3
        assert generation_df["partner_mix_preset"].iloc[0] == "cooperative_heavy"
        assert strategy_df["generation"].max() == 2
        assert any(origin in set(strategy_df["origin"]) for origin in {mode, "random_injector", "adversarial_heuristic", "search_mutation"})


def test_harvest_tier_presets_increase_ecological_stress() -> None:
    easy = get_harvest_tier_preset("easy_h1")
    medium = get_harvest_tier_preset("medium_h1")
    hard = get_harvest_tier_preset("hard_h1")
    assert easy["regen_rate"] > medium["regen_rate"] > hard["regen_rate"]
    assert easy["weather_noise_std"] < medium["weather_noise_std"] < hard["weather_noise_std"]
    assert easy["neighbor_externality"] < medium["neighbor_externality"] < hard["neighbor_externality"]

    hard_cfg = make_harvest_cfg_for_tier("hard_h1", n_agents=6, seed=9)
    assert hard_cfg.patch_init == hard["patch_init"]
    assert hard_cfg.regen_rate == hard["regen_rate"]


def test_harvest_bottom_up_transfers_and_hybrid_cap() -> None:
    cfg = HarvestCommonsConfig(n_agents=4, horizon=20, seed=5)
    bottom_up_agents = [
        ReciprocalHarvestAgent(),
        CreditSharingHarvestAgent(),
        ReciprocalHarvestAgent(),
        CreditSharingHarvestAgent(),
    ]
    bottom_up = run_harvest_episode(cfg, bottom_up_agents, governor=None)
    assert bottom_up["total_credit_transferred"] > 0.0

    none_cfg = HarvestCommonsConfig(n_agents=4, horizon=20, seed=5, communication_enabled=False, side_payments_enabled=False)
    none_agents = [SelfInterestedHarvestAgent() for _ in range(4)]
    no_bottom_up = run_harvest_episode(none_cfg, none_agents, governor=None)
    assert no_bottom_up["total_credit_transferred"] == 0.0

    hybrid = run_harvest_episode(
        cfg,
        bottom_up_agents,
        governor=GovernmentAgent(
            trigger=15.0,
            strict_cap_frac=0.2,
            relaxed_cap_frac=0.4,
            soft_trigger=17.0,
            deterioration_threshold=0.2,
            activation_warmup=1,
            aggressive_request_threshold=0.4,
            aggressive_agent_fraction_trigger=0.25,
            enforcement_scope="local",
        ),
    )
    assert hybrid["mean_government_cap"] > 0.0
    assert hybrid["mean_capped_action_fraction"] > 0.0
    assert hybrid["mean_prevented_harvest"] > 0.0
    assert hybrid["mean_neighborhood_overharvest"] > 0.0


def test_bottom_up_signals_change_future_harvest_behavior() -> None:
    agent = SelfInterestedHarvestAgent()
    observation = HarvestObservation(
        local_patch=12.0,
        neighbor_mean=10.0,
        last_credit_received=0.0,
        government_cap_frac=None,
    )
    quiet_action = agent.act(observation, inbox=[], t=0)
    social_action = agent.act(
        HarvestObservation(
            local_patch=12.0,
            neighbor_mean=10.0,
            last_credit_received=0.4,
            government_cap_frac=None,
        ),
        inbox=[HarvestMessage(announced_restraint=0.8), HarvestMessage(announced_restraint=0.7)],
        t=1,
    )
    assert social_action.harvest_frac < quiet_action.harvest_frac
    capped_action = agent.act(
        HarvestObservation(
            local_patch=12.0,
            neighbor_mean=10.0,
            last_credit_received=0.0,
            government_cap_frac=0.2,
        ),
        inbox=[],
        t=2,
    )
    assert capped_action.harvest_frac <= 0.23


def test_local_aggressive_neighborhood_can_trigger_relaxed_cap() -> None:
    governor = GovernmentAgent(
        trigger=15.0,
        strict_cap_frac=0.18,
        relaxed_cap_frac=0.32,
        soft_trigger=17.5,
        deterioration_threshold=0.1,
        activation_warmup=1,
        aggressive_request_threshold=0.75,
        aggressive_agent_fraction_trigger=0.8,
        local_neighborhood_trigger=0.65,
        enforcement_scope="local",
    )
    assert governor.set_cap(mean_patch_health=18.5, t=0, n_agents=4) is None
    governor.observe_step(np.array([0.8, 0.8, 0.2, 0.2]))
    cap = governor.set_cap(mean_patch_health=17.0, t=2, n_agents=4)
    assert cap is not None
    assert np.isclose(cap[0], 0.32)
    assert np.isclose(cap[1], 0.32)
    assert np.isnan(cap[2])
    assert np.isnan(cap[3])


def test_harvest_hybrid_separates_from_bottom_up_under_mixed_pressure() -> None:
    cfg = HarvestCommonsConfig(n_agents=6, horizon=20, seed=11)
    mixed_agents = [
        SelfInterestedHarvestAgent(),
        ReciprocalHarvestAgent(),
        CreditSharingHarvestAgent(),
        SelfInterestedHarvestAgent(),
        ReciprocalHarvestAgent(),
        CreditSharingHarvestAgent(),
    ]
    bottom_up = run_harvest_episode(cfg, mixed_agents, governor=None)
    hybrid = run_harvest_episode(
        cfg,
        mixed_agents,
        governor=GovernmentAgent(
            trigger=16.0,
            strict_cap_frac=0.18,
            relaxed_cap_frac=0.35,
            soft_trigger=18.0,
            deterioration_threshold=0.35,
            activation_warmup=3,
            aggressive_request_threshold=0.7,
            aggressive_agent_fraction_trigger=0.3,
            enforcement_scope="local",
        ),
    )
    assert hybrid["mean_government_cap"] > 0.0
    assert hybrid["mean_patch_health"] != bottom_up["mean_patch_health"]
    assert hybrid["mean_capped_action_fraction"] > 0.0
    assert hybrid["mean_max_local_aggression"] > 0.0
    assert 0.0 < hybrid["mean_targeted_agent_fraction"] < 1.0


def test_harvest_selective_government_is_not_always_on_for_stable_cooperators() -> None:
    cfg = HarvestCommonsConfig(n_agents=6, horizon=20, seed=13)
    cooperative_agents = [
        ReciprocalHarvestAgent(),
        CreditSharingHarvestAgent(),
        ReciprocalHarvestAgent(),
        CreditSharingHarvestAgent(),
        ReciprocalHarvestAgent(),
        CreditSharingHarvestAgent(),
    ]
    top_down = run_harvest_episode(
        cfg,
        cooperative_agents,
        governor=GovernmentAgent(
            trigger=16.0,
            strict_cap_frac=0.18,
            relaxed_cap_frac=0.35,
            soft_trigger=18.0,
            deterioration_threshold=0.35,
            activation_warmup=3,
            aggressive_request_threshold=0.7,
            aggressive_agent_fraction_trigger=0.3,
            enforcement_scope="global",
        ),
    )
    assert 0.0 <= top_down["mean_government_cap"] < 0.35
    assert top_down["mean_capped_action_fraction"] < 0.2
