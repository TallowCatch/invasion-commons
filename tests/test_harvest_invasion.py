import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishery_sim.harvest import GovernmentAgent
from fishery_sim.harvest import HarvestCommonsConfig
from fishery_sim.harvest import SelfInterestedHarvestAgent
from fishery_sim.harvest import run_harvest_episode
from fishery_sim.harvest_benchmarks import get_harvest_regime_pack
from fishery_sim.harvest_benchmarks import make_harvest_cfg_for_tier
from fishery_sim.harvest_evolution import balanced_harvest_strategy
from fishery_sim.harvest_evolution import build_initial_harvest_population
from fishery_sim.harvest_evolution import evaluate_harvest_population
from fishery_sim.harvest_evolution import make_harvest_strategy_injector
from fishery_sim.harvest_evolution import mutate_harvest_strategy
from fishery_sim.harvest_evolution import run_harvest_invasion


def test_harvest_invasion_is_deterministic_for_random_and_mutation() -> None:
    cfg = make_harvest_cfg_for_tier("medium_h1", n_agents=6, seed=7)
    regimes = get_harvest_regime_pack("medium_h1")[:2]
    for mode in ("random", "mutation"):
        injector = make_harvest_strategy_injector(mode)
        gen_a, strat_a = run_harvest_invasion(
            base_cfg=copy.deepcopy(cfg),
            condition="hybrid",
            generations=3,
            population_size=6,
            seeds_per_generation=3,
            test_seeds_per_generation=3,
            replacement_fraction=0.2,
            adversarial_pressure=0.3,
            rng_seed=11,
            partner_mix_preset="balanced",
            injector=injector,
            test_regimes=regimes,
        )
        injector = make_harvest_strategy_injector(mode)
        gen_b, strat_b = run_harvest_invasion(
            base_cfg=copy.deepcopy(cfg),
            condition="hybrid",
            generations=3,
            population_size=6,
            seeds_per_generation=3,
            test_seeds_per_generation=3,
            replacement_fraction=0.2,
            adversarial_pressure=0.3,
            rng_seed=11,
            partner_mix_preset="balanced",
            injector=injector,
            test_regimes=regimes,
        )
        assert gen_a.equals(gen_b)
        assert strat_a.equals(strat_b)


def test_harvest_invasion_condition_gating_controls_caps_and_credits() -> None:
    cfg = make_harvest_cfg_for_tier("medium_h1", n_agents=6, seed=3)
    population = build_initial_harvest_population(6, patch_max=cfg.patch_max, rng=np.random.default_rng(2), partner_mix_preset="cooperative_heavy")

    none_df, _ = evaluate_harvest_population(cfg, "none", population, seeds=[1, 2])
    top_df, _ = evaluate_harvest_population(cfg, "top_down_only", population, seeds=[1, 2])
    bottom_df, _ = evaluate_harvest_population(cfg, "bottom_up_only", population, seeds=[1, 2])
    hybrid_df, _ = evaluate_harvest_population(cfg, "hybrid", population, seeds=[1, 2])

    assert float(none_df["mean_government_cap"].mean()) == 0.0
    assert float(bottom_df["mean_government_cap"].mean()) == 0.0
    assert float(top_df["mean_government_cap"].mean()) > 0.0
    assert float(hybrid_df["mean_government_cap"].mean()) > 0.0

    assert float(none_df["mean_credit_transferred"].mean()) == 0.0
    assert float(top_df["mean_credit_transferred"].mean()) == 0.0
    assert float(bottom_df["mean_credit_transferred"].mean()) > 0.0
    assert float(hybrid_df["mean_credit_transferred"].mean()) > 0.0


def test_harvest_mutation_pressure_increases_exploitation_and_reduces_restraint() -> None:
    rng = np.random.default_rng(5)
    parent = balanced_harvest_strategy("parent", patch_max=20.0, rng=rng)
    low_pressure_children = [
        mutate_harvest_strategy(parent, f"lp_{i}", patch_max=20.0, rng=np.random.default_rng(100 + i), adversarial_pressure=0.1)
        for i in range(64)
    ]
    high_pressure_children = [
        mutate_harvest_strategy(parent, f"hp_{i}", patch_max=20.0, rng=np.random.default_rng(300 + i), adversarial_pressure=0.7)
        for i in range(64)
    ]
    assert np.mean([c.high_harvest_frac for c in high_pressure_children]) > np.mean([c.high_harvest_frac for c in low_pressure_children])
    assert np.mean([c.restraint_low for c in high_pressure_children]) < np.mean([c.restraint_low for c in low_pressure_children])


def test_harvest_garden_failure_triggers_under_sustained_low_patch_health() -> None:
    cfg = HarvestCommonsConfig(
        n_agents=4,
        horizon=20,
        patch_init=3.0,
        patch_max=10.0,
        regen_rate=0.05,
        weather_noise_std=0.0,
        neighbor_externality=0.25,
        communication_enabled=False,
        side_payments_enabled=False,
        local_patch_failure_threshold=4.0,
        failure_fraction_threshold=0.5,
        failure_patience=2,
        seed=9,
    )
    agents = [SelfInterestedHarvestAgent() for _ in range(4)]
    out = run_harvest_episode(cfg, agents, governor=None)
    assert out["garden_failure_event"] == 1
    assert out["t_end"] <= cfg.horizon


def test_harvest_invasion_uses_full_held_out_pack() -> None:
    cfg = make_harvest_cfg_for_tier("medium_h1", n_agents=6, seed=11)
    regimes = get_harvest_regime_pack("medium_h1")[:2]
    generation_df, _ = run_harvest_invasion(
        base_cfg=cfg,
        condition="hybrid",
        generations=3,
        population_size=6,
        seeds_per_generation=3,
        test_seeds_per_generation=3,
        replacement_fraction=0.2,
        adversarial_pressure=0.3,
        rng_seed=17,
        partner_mix_preset="balanced",
        injector=make_harvest_strategy_injector("mutation"),
        test_regimes=regimes,
    )
    assert int(generation_df["test_regime_count"].iloc[0]) == 2
    assert "test_noisy_weather_garden_failure_rate" in generation_df.columns
    assert "test_strong_externality_garden_failure_rate" in generation_df.columns


def test_government_agent_interface_matches_previous_behavior() -> None:
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
    assert governor.act(mean_patch_health=18.5, t=0, n_agents=4) is None
    governor.observe(requested_fracs=np.array([0.8, 0.8, 0.2, 0.2]))
    cap = governor.act(mean_patch_health=17.0, t=2, n_agents=4)
    assert cap is not None
    capped, targeted = governor.apply_cap(np.array([0.9, 0.8, 0.2, 0.2]), cap)
    assert targeted.sum() == 2
    assert float(capped[0]) <= 0.32
    assert float(capped[1]) <= 0.32
