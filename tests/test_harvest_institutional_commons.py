import copy
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.harvest_oversight_reporting import condition_label
from experiments.harvest_oversight_reporting import scenario_label
from experiments.summarize_harvest_invasion import _aggregate_with_ci
from fishery_sim.harvest import GovernmentAgent
from fishery_sim.harvest import harvest_global_safe
from fishery_sim.harvest import harvest_local_safety_mask
from fishery_sim.harvest_benchmarks import (
    get_harvest_actor_capability_preset,
    get_harvest_governance_friction_regime,
    get_harvest_overseer_capability_preset,
    get_harvest_scenario_preset,
    harvest_capability_gap,
    make_harvest_cfg_for_scenario,
    make_harvest_cfg_for_tier,
)
from fishery_sim.harvest_evolution import (
    _make_condition_setup,
    adversarial_harvest_strategy,
    balanced_harvest_strategy,
    cooperative_harvest_strategy,
)
from fishery_sim.harvest_llm_population import (
    build_harvest_bank_prompt,
    load_harvest_strategy_bank,
    sample_population_from_bank,
    strategy_spec_to_bank_row,
)


def test_harvest_scenario_presets_are_literature_backed() -> None:
    fishery = get_harvest_scenario_preset("regulated_fishery")
    irrigation = get_harvest_scenario_preset("community_irrigation")
    forest = get_harvest_scenario_preset("forest_co_management")

    assert fishery["tier"] == "medium_h1"
    assert fishery["preferred_governance_comparison"] == ["top_down_only", "hybrid"]
    assert len(fishery["citations"]) >= 2

    assert irrigation["cfg_overrides"]["communication_enabled"] is True
    assert irrigation["cfg_overrides"]["side_payments_enabled"] is False
    assert len(irrigation["citations"]) >= 3

    assert forest["tier"] == "hard_h1"
    assert forest["partner_mix"] == "adversarial_heavy"
    assert forest["cfg_overrides"]["side_payments_enabled"] is True


def test_governance_friction_regimes_match_plan_defaults() -> None:
    ideal = get_harvest_governance_friction_regime("ideal")
    constrained = get_harvest_governance_friction_regime("constrained")
    assert ideal["detection_recall"] == 1.0
    assert ideal["enforcement_delay_rounds"] == 0
    assert constrained["detection_recall"] == 0.7
    assert constrained["enforcement_delay_rounds"] == 1
    assert constrained["max_target_share"] == 0.5
    assert constrained["governance_budget_cost"] == 0.02


def test_harvest_oversight_capability_presets_match_stage_a_plan() -> None:
    low = get_harvest_actor_capability_preset("low_actor")
    medium = get_harvest_actor_capability_preset("medium_actor")
    high = get_harvest_actor_capability_preset("high_actor")
    weak = get_harvest_overseer_capability_preset("weak_overseer")
    strong = get_harvest_overseer_capability_preset("strong_overseer")

    assert low["injector_mode"] == "mutation"
    assert low["rank"] == 0
    assert medium["injector_mode"] == "search_mutation"
    assert medium["search_candidates"] == 6
    assert medium["search_eval_horizon"] == 30
    assert high["search_candidates"] == 12
    assert high["search_eval_horizon"] == 60
    assert weak["detection_recall"] == 0.5
    assert weak["enforcement_delay_rounds"] == 2
    assert strong["max_target_share"] == 1.0
    assert harvest_capability_gap("high_actor", "weak_overseer") == 2
    assert harvest_capability_gap("low_actor", "strong_overseer") == -2


def test_harvest_public_reporting_labels_hide_internal_names() -> None:
    assert condition_label("bottom_up_only") == "Local oversight"
    assert condition_label("top_down_only") == "Global signal"
    assert condition_label("hybrid") == "Hybrid oversight"
    assert scenario_label("community_irrigation") == "Moderate-coupling commons"
    assert scenario_label("forest_co_management") == "High-coupling commons"


def test_harvest_oversight_safety_predicates_count_compositional_failure() -> None:
    requested = np.array([0.38, 0.39, 0.40], dtype=float)
    local_mask = harvest_local_safety_mask(requested, sustainable_harvest_frac=0.35)
    assert local_mask.tolist() == [True, True, True]
    stricter_local_mask = harvest_local_safety_mask(requested, sustainable_harvest_frac=0.35, local_safety_margin=0.0)
    assert stricter_local_mask.tolist() == [False, False, False]

    globally_safe = harvest_global_safe(
        np.array([11.0, 10.5, 10.2], dtype=float),
        min_mean_patch_health=10.0,
        local_patch_failure_threshold=4.0,
        failure_fraction_threshold=0.5,
    )
    globally_unsafe = harvest_global_safe(
        np.array([9.0, 10.4, 10.1], dtype=float),
        min_mean_patch_health=10.0,
        local_patch_failure_threshold=4.0,
        failure_fraction_threshold=0.5,
    )
    globally_unsafe_at_stricter_threshold = harvest_global_safe(
        np.array([10.4, 10.5, 10.6], dtype=float),
        min_mean_patch_health=10.6,
        local_patch_failure_threshold=4.0,
        failure_fraction_threshold=0.5,
    )
    assert globally_safe
    assert not globally_unsafe
    assert not globally_unsafe_at_stricter_threshold
    assert bool(np.all(local_mask)) and not globally_unsafe


def test_bottom_up_condition_respects_scenario_comms_flags() -> None:
    base_cfg = make_harvest_cfg_for_scenario("regulated_fishery", n_agents=6, seed=7)
    cfg, governor = _make_condition_setup(base_cfg, "bottom_up_only", government_params=None)
    assert governor is None
    assert cfg.communication_enabled is False
    assert cfg.side_payments_enabled is False


def test_government_agent_friction_tracks_missed_targets_and_budget() -> None:
    governor = GovernmentAgent(
        enforcement_scope="global",
        detection_recall=0.0,
        enforcement_delay_rounds=0,
        max_target_share=1.0,
        governance_budget_cost=0.02,
    )
    governor.reset(seed=11)
    requested = np.array([0.9, 0.8, 0.2, 0.1], dtype=float)
    cap_fracs = np.array([0.2, 0.2, np.nan, np.nan], dtype=float)
    capped, targeted = governor.apply_cap(requested, cap_fracs)
    assert np.allclose(capped, requested)
    assert int(targeted.sum()) == 0
    assert governor._last_intended_target_count == 2
    assert governor._last_missed_target_count == 2
    assert governor._last_governance_budget_spent == 0.0

    governor = GovernmentAgent(
        enforcement_scope="global",
        detection_recall=1.0,
        enforcement_delay_rounds=0,
        max_target_share=0.5,
        governance_budget_cost=0.02,
    )
    governor.reset(seed=11)
    capped, targeted = governor.apply_cap(requested, np.full_like(requested, 0.25))
    assert int(targeted.sum()) == 2
    assert np.isclose(governor._last_governance_budget_spent, 0.04)
    assert np.all(capped[targeted] <= 0.25 + 1e-12)


def test_summarizer_keeps_scenarios_and_regimes_separate() -> None:
    per_run_df = pd.DataFrame(
        [
            {
                "scenario_preset": "regulated_fishery",
                "governance_friction_regime": "ideal",
                "tier": "medium_h1",
                "partner_mix": "balanced",
                "injector_mode_requested": "search_mutation",
                "adversarial_pressure": 0.3,
                "condition": "hybrid",
                "run_id": 0,
                "test_garden_failure_mean": 0.0,
                "test_mean_patch_health_mean": 15.0,
                "test_mean_welfare_mean": 10.0,
            },
            {
                "scenario_preset": "community_irrigation",
                "governance_friction_regime": "constrained",
                "tier": "medium_h1",
                "partner_mix": "balanced",
                "injector_mode_requested": "search_mutation",
                "adversarial_pressure": 0.3,
                "condition": "hybrid",
                "run_id": 0,
                "test_garden_failure_mean": 0.0,
                "test_mean_patch_health_mean": 12.0,
                "test_mean_welfare_mean": 9.0,
            },
        ]
    )
    table_df = _aggregate_with_ci(per_run_df)
    assert len(table_df) == 2
    assert set(table_df["scenario_preset"]) == {"regulated_fishery", "community_irrigation"}
    assert set(table_df["governance_friction_regime"]) == {"ideal", "constrained"}


def test_strategy_bank_helpers_generate_and_sample_by_attitude(tmp_path: Path) -> None:
    cfg = make_harvest_cfg_for_tier("medium_h1", n_agents=6, seed=3)
    prompt = build_harvest_bank_prompt("cooperative", patch_max=cfg.patch_max, prompt_nonce=99)
    assert "Attitude: cooperative" in prompt
    assert "Variation nonce: 99" in prompt

    model_label = "ollama__qwen2_5_3b_instruct"
    rows = []
    specs = [
        cooperative_harvest_strategy("c0", patch_max=cfg.patch_max, rng=np.random.default_rng(1)),
        cooperative_harvest_strategy("c1", patch_max=cfg.patch_max, rng=np.random.default_rng(2)),
        adversarial_harvest_strategy("e0", patch_max=cfg.patch_max, rng=np.random.default_rng(3)),
        balanced_harvest_strategy("e1", patch_max=cfg.patch_max, rng=np.random.default_rng(4)),
    ]
    attitudes = ["cooperative", "cooperative", "exploitative", "exploitative"]
    for i, (spec, attitude) in enumerate(zip(specs, attitudes, strict=True)):
        spec.origin = "llm_bank"
        rows.append(
            strategy_spec_to_bank_row(
                spec,
                bank_model_label=model_label,
                bank_provider="ollama",
                bank_model_name="qwen2.5:3b-instruct",
                bank_attitude=attitude,
                prompt_nonce=i,
            )
        )
    bank_csv = tmp_path / "bank.csv"
    pd.DataFrame(rows).to_csv(bank_csv, index=False)
    bank_df = load_harvest_strategy_bank(str(bank_csv))
    rng_a = np.random.default_rng(17)
    population_a, attitudes_a = sample_population_from_bank(
        bank_df,
        population_size=6,
        exploitative_share=0.5,
        rng=rng_a,
        model_label=model_label,
    )
    rng_b = np.random.default_rng(17)
    population_b, attitudes_b = sample_population_from_bank(
        bank_df,
        population_size=6,
        exploitative_share=0.5,
        rng=rng_b,
        model_label=model_label,
    )
    assert [spec.strategy_id for spec in population_a] == [spec.strategy_id for spec in population_b]
    assert attitudes_a == attitudes_b
    assert attitudes_a.count("exploitative") == 3
    assert attitudes_a.count("cooperative") == 3
