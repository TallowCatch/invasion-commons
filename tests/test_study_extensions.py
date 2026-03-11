import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishery_sim.agents import GreedyAgent
from fishery_sim.config import FisheryConfig
from fishery_sim.evolution import make_strategy_injector
from fishery_sim.evolution import run_evolutionary_invasion
from fishery_sim.harvest import CreditSharingHarvestAgent
from fishery_sim.harvest import GovernmentAgent
from fishery_sim.harvest import HarvestCommonsConfig
from fishery_sim.harvest import ReciprocalHarvestAgent
from fishery_sim.harvest import SelfInterestedHarvestAgent
from fishery_sim.harvest import run_harvest_episode
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
        ),
    )
    assert hybrid["mean_government_cap"] > 0.0


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
        ),
    )
    assert hybrid["mean_government_cap"] > 0.0
    assert hybrid["mean_patch_health"] != bottom_up["mean_patch_health"]


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
        ),
    )
    assert 0.0 <= top_down["mean_government_cap"] < 0.35
