import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_harvest_invasion_local_shards import stage_config
from experiments.summarize_harvest_invasion import _build_capability_ladder_df
from experiments.summarize_harvest_invasion import _build_integrity_df
from experiments.summarize_harvest_invasion import _build_ranking_table
from experiments.summarize_harvest_invasion import _build_welfare_incidence_outputs
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
from fishery_sim.harvest_evolution import parse_harvest_policy_response
from fishery_sim.harvest_evolution import run_harvest_invasion
from fishery_sim.llm_adapter import FileReplayPolicyLLMClient


def test_harvest_invasion_is_deterministic_for_random_mutation_and_search() -> None:
    cfg = make_harvest_cfg_for_tier("medium_h1", n_agents=6, seed=7)
    regimes = get_harvest_regime_pack("medium_h1")[:2]
    for mode in ("random", "mutation", "search_mutation"):
        injector = make_harvest_strategy_injector(mode)
        gen_a, strat_a, agent_a = run_harvest_invasion(
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
        gen_b, strat_b, agent_b = run_harvest_invasion(
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
        assert agent_a.equals(agent_b)


def test_harvest_llm_replay_is_deterministic(tmp_path: Path) -> None:
    cfg = make_harvest_cfg_for_tier("medium_h1", n_agents=6, seed=7)
    regimes = get_harvest_regime_pack("medium_h1")[:2]
    replay_path = tmp_path / "harvest_llm_replay.jsonl"
    replay_path.write_text(
        "\n".join(
            [
                '{"rationale":"careful local trader","low_patch_threshold":5.0,"high_patch_threshold":12.0,"low_harvest_frac":0.10,"mid_harvest_frac":0.35,"high_harvest_frac":0.65,"restraint_low":0.70,"restraint_high":0.35,"credit_request_low":0.25,"credit_request_high":0.05,"credit_offer_threshold":10.0,"credit_offer_amount":0.25,"neighbor_reciprocity_weight":0.55,"credit_response_weight":0.50,"cap_compliance_margin":0.02}',
                '{"rationale":"more aggressive contender","low_patch_threshold":4.0,"high_patch_threshold":11.0,"low_harvest_frac":0.18,"mid_harvest_frac":0.45,"high_harvest_frac":0.82,"restraint_low":0.35,"restraint_high":0.18,"credit_request_low":0.12,"credit_request_high":0.02,"credit_offer_threshold":8.0,"credit_offer_amount":0.08,"neighbor_reciprocity_weight":0.20,"credit_response_weight":0.18,"cap_compliance_margin":0.06}',
            ]
        ),
        encoding="utf-8",
    )
    injector_a = make_harvest_strategy_injector("llm_json", llm_client=FileReplayPolicyLLMClient(str(replay_path)))
    gen_a, strat_a, agent_a = run_harvest_invasion(
        base_cfg=copy.deepcopy(cfg),
        condition="hybrid",
        generations=3,
        population_size=6,
        seeds_per_generation=3,
        test_seeds_per_generation=3,
        replacement_fraction=0.2,
        adversarial_pressure=0.3,
        rng_seed=17,
        partner_mix_preset="balanced",
        injector=injector_a,
        test_regimes=regimes,
    )
    injector_b = make_harvest_strategy_injector("llm_json", llm_client=FileReplayPolicyLLMClient(str(replay_path)))
    gen_b, strat_b, agent_b = run_harvest_invasion(
        base_cfg=copy.deepcopy(cfg),
        condition="hybrid",
        generations=3,
        population_size=6,
        seeds_per_generation=3,
        test_seeds_per_generation=3,
        replacement_fraction=0.2,
        adversarial_pressure=0.3,
        rng_seed=17,
        partner_mix_preset="balanced",
        injector=injector_b,
        test_regimes=regimes,
    )
    assert gen_a.equals(gen_b)
    assert strat_a.equals(strat_b)
    assert agent_a.equals(agent_b)
    assert "llm_json_fraction" in gen_a.columns
    assert float((strat_a["origin"] == "llm_json").mean()) > 0.0


def test_harvest_llm_invalid_json_falls_back_to_mutation(tmp_path: Path) -> None:
    cfg = make_harvest_cfg_for_tier("medium_h1", n_agents=6, seed=4)
    replay_path = tmp_path / "bad_harvest_llm.json"
    replay_path.write_text(
        '{"rationale":"broken","low_patch_threshold":3.0,"high_harvest_frac":0.9}',
        encoding="utf-8",
    )
    generation_df, strategy_df, agent_df = run_harvest_invasion(
        base_cfg=copy.deepcopy(cfg),
        condition="top_down_only",
        generations=2,
        population_size=6,
        seeds_per_generation=3,
        test_seeds_per_generation=3,
        replacement_fraction=0.2,
        adversarial_pressure=0.3,
        rng_seed=21,
        partner_mix_preset="balanced",
        injector=make_harvest_strategy_injector("llm_json", llm_client=FileReplayPolicyLLMClient(str(replay_path))),
        test_regimes=get_harvest_regime_pack("medium_h1")[:1],
    )
    assert float((strategy_df["origin"] == "llm_fallback_mutation").mean()) > 0.0
    assert float(generation_df["llm_fallback_fraction"].max()) > 0.0
    assert not agent_df.empty


def test_harvest_policy_parser_repairs_common_malformed_outputs() -> None:
    patch_max = 20.0
    cases = [
        (
            "fenced_json",
            """```json
{"rationale":"repair fence","low_patch_threshold":5.0,"high_patch_threshold":12.0,"low_harvest_frac":0.10,"mid_harvest_frac":0.35,"high_harvest_frac":0.65,"restraint_low":0.70,"restraint_high":0.35,"credit_request_low":0.25,"credit_request_high":0.05,"credit_offer_threshold":10.0,"credit_offer_amount":0.25,"neighbor_reciprocity_weight":0.55,"credit_response_weight":0.50,"cap_compliance_margin":0.02}
```""",
        ),
        (
            "trailing_prose",
            '{"rationale":"repair prose","low_patch_threshold":5.0,"high_patch_threshold":12.0,"low_harvest_frac":0.10,"mid_harvest_frac":0.35,"high_harvest_frac":0.65,"restraint_low":0.70,"restraint_high":0.35,"credit_request_low":0.25,"credit_request_high":0.05,"credit_offer_threshold":10.0,"credit_offer_amount":0.25,"neighbor_reciprocity_weight":0.55,"credit_response_weight":0.50,"cap_compliance_margin":0.02}\nThis policy is cautious.',
        ),
        (
            "single_quotes",
            "{'rationale':'repair quotes','low_patch_threshold':5.0,'high_patch_threshold':12.0,'low_harvest_frac':0.10,'mid_harvest_frac':0.35,'high_harvest_frac':0.65,'restraint_low':0.70,'restraint_high':0.35,'credit_request_low':0.25,'credit_request_high':0.05,'credit_offer_threshold':10.0,'credit_offer_amount':0.25,'neighbor_reciprocity_weight':0.55,'credit_response_weight':0.50,'cap_compliance_margin':0.02}",
        ),
        (
            "missing_outer_object",
            '"rationale":"repair wrapped","low_patch_threshold":5.0,"high_patch_threshold":12.0,"low_harvest_frac":0.10,"mid_harvest_frac":0.35,"high_harvest_frac":0.65,"restraint_low":0.70,"restraint_high":0.35,"credit_request_low":0.25,"credit_request_high":0.05,"credit_offer_threshold":10.0,"credit_offer_amount":0.25,"neighbor_reciprocity_weight":0.55,"credit_response_weight":0.50,"cap_compliance_margin":0.02',
        ),
        (
            "numeric_strings",
            '{"rationale":"repair numeric strings","low_patch_threshold":"5.0","high_patch_threshold":"12.0","low_harvest_frac":"0.10","mid_harvest_frac":"0.35","high_harvest_frac":"0.65","restraint_low":"0.70","restraint_high":"0.35","credit_request_low":"0.25","credit_request_high":"0.05","credit_offer_threshold":"10.0","credit_offer_amount":"0.25","neighbor_reciprocity_weight":"0.55","credit_response_weight":"0.50","cap_compliance_margin":"0.02"}',
        ),
    ]
    for expected_error_type, raw in cases:
        policy, parse_status, parse_error_type = parse_harvest_policy_response(raw, patch_max=patch_max)
        assert parse_status == "repaired_json"
        assert parse_error_type == expected_error_type
        assert 0.0 <= policy.low_harvest_frac <= 1.0
        assert 0.0 <= policy.high_harvest_frac <= 1.0


def test_harvest_llm_repaired_json_is_deterministic_and_clamped(tmp_path: Path) -> None:
    cfg = make_harvest_cfg_for_tier("medium_h1", n_agents=6, seed=12)
    replay_path = tmp_path / "repaired_harvest_llm.jsonl"
    replay_path.write_text(
        '```json {"rationale":"repair me","low_patch_threshold":"-2.0","high_patch_threshold":"30.0","low_harvest_frac":"-0.10","mid_harvest_frac":"0.60","high_harvest_frac":"1.40","restraint_low":"1.20","restraint_high":"-0.30","credit_request_low":"0.25","credit_request_high":"0.05","credit_offer_threshold":"25.0","credit_offer_amount":"1.20","neighbor_reciprocity_weight":"0.55","credit_response_weight":"0.50","cap_compliance_margin":"0.40"} ```',
        encoding="utf-8",
    )
    generation_df, strategy_df, agent_df = run_harvest_invasion(
        base_cfg=copy.deepcopy(cfg),
        condition="hybrid",
        generations=2,
        population_size=6,
        seeds_per_generation=3,
        test_seeds_per_generation=3,
        replacement_fraction=0.2,
        adversarial_pressure=0.3,
        rng_seed=17,
        partner_mix_preset="balanced",
        injector=make_harvest_strategy_injector("llm_json", llm_client=FileReplayPolicyLLMClient(str(replay_path))),
        test_regimes=get_harvest_regime_pack("medium_h1")[:1],
    )
    repaired = strategy_df[strategy_df["origin"] == "llm_json"].copy()
    assert not repaired.empty
    assert set(repaired["llm_parse_status"]) == {"repaired_json"}
    assert set(repaired["llm_parse_error_type"]) == {"fenced_json"}
    assert float(repaired["low_patch_threshold"].min()) >= 0.0
    assert float(repaired["high_patch_threshold"].max()) <= cfg.patch_max
    assert float(repaired["high_harvest_frac"].max()) <= 1.0
    assert float(repaired["cap_compliance_margin"].max()) <= 0.25
    assert float(generation_df["repaired_json_fraction"].max()) > 0.0
    assert float(generation_df["effective_llm_fraction"].max()) > 0.0
    assert not agent_df.empty


def test_harvest_integrity_summary_adds_gate_flags() -> None:
    per_run_df = pd.DataFrame(
        [
            {
                "tier": "medium_h1",
                "partner_mix": "balanced",
                "adversarial_pressure": 0.3,
                "condition": "hybrid",
                "injector_mode_requested": "llm_json",
                "run_id": 0,
                "llm_provider": "ollama",
                "llm_model": "qwen2.5:3b-instruct",
                "llm_json_fraction": 0.95,
                "llm_fallback_fraction": 0.05,
                "direct_json_fraction": 0.65,
                "repaired_json_fraction": 0.30,
                "effective_llm_fraction": 0.95,
                "unrepaired_fallback_fraction": 0.05,
                "llm_parse_error_count__fenced_json": 1,
                "llm_parse_error_count__numeric_strings": 2,
            },
            {
                "tier": "medium_h1",
                "partner_mix": "balanced",
                "adversarial_pressure": 0.3,
                "condition": "top_down_only",
                "injector_mode_requested": "llm_json",
                "run_id": 0,
                "llm_provider": "ollama",
                "llm_model": "qwen2.5:3b-instruct",
                "llm_json_fraction": 0.70,
                "llm_fallback_fraction": 0.30,
                "direct_json_fraction": 0.50,
                "repaired_json_fraction": 0.20,
                "effective_llm_fraction": 0.70,
                "unrepaired_fallback_fraction": 0.30,
                "llm_parse_error_count__fenced_json": 0,
                "llm_parse_error_count__numeric_strings": 1,
            },
        ]
    )
    integrity_df = _build_integrity_df(per_run_df)
    hybrid = integrity_df[integrity_df["condition"] == "hybrid"].iloc[0]
    top_down = integrity_df[integrity_df["condition"] == "top_down_only"].iloc[0]
    assert bool(hybrid["effective_llm_gate_pass"])
    assert bool(hybrid["fallback_gate_pass"])
    assert bool(hybrid["usable_for_evidence"])
    assert not bool(top_down["usable_for_evidence"])
    assert float(hybrid["repaired_json_fraction_mean"]) == 0.30


def test_harvest_local_shard_presets_cover_reliability_and_narrow_stages() -> None:
    stage_a = stage_config("stage_a_llm_reliability")
    stage_b = stage_config("stage_b_llm_narrow")
    stage_c = stage_config("stage_c_llm_narrow")
    arch_b = stage_config("architecture_stageB")
    arch_c = stage_config("architecture_stageC")
    ladder = stage_config("capability_ladder_stageB")
    assert len(stage_a["explicit_cells"]) == 3
    assert stage_a["generations"] == "4"
    assert stage_a["seeds_per_generation"] == "8"
    assert stage_b["pressures"] == ["0.3"]
    assert stage_b["injector_modes"] == ["mutation", "llm_json"]
    assert stage_b["n_runs"] == "3"
    assert stage_c["injector_modes"] == ["llm_json"]
    assert stage_c["n_runs"] == "5"
    assert arch_b["conditions"] == ["none", "bottom_up_only", "top_down_only", "hybrid"]
    assert arch_c["injector_modes"] == ["search_mutation"]
    assert ladder["injector_modes"] == ["random", "mutation", "adversarial_heuristic", "search_mutation"]


def test_harvest_invasion_condition_gating_controls_caps_and_credits() -> None:
    cfg = make_harvest_cfg_for_tier("medium_h1", n_agents=6, seed=3)
    population = build_initial_harvest_population(6, patch_max=cfg.patch_max, rng=np.random.default_rng(2), partner_mix_preset="cooperative_heavy")

    none_df, _, none_agents = evaluate_harvest_population(cfg, "none", population, seeds=[1, 2])
    top_df, _, top_agents = evaluate_harvest_population(cfg, "top_down_only", population, seeds=[1, 2])
    bottom_df, _, bottom_agents = evaluate_harvest_population(cfg, "bottom_up_only", population, seeds=[1, 2])
    hybrid_df, _, hybrid_agents = evaluate_harvest_population(cfg, "hybrid", population, seeds=[1, 2])

    assert float(none_df["mean_government_cap"].mean()) == 0.0
    assert float(bottom_df["mean_government_cap"].mean()) == 0.0
    assert float(top_df["mean_government_cap"].mean()) > 0.0
    assert float(hybrid_df["mean_government_cap"].mean()) > 0.0

    assert float(none_df["mean_credit_transferred"].mean()) == 0.0
    assert float(top_df["mean_credit_transferred"].mean()) == 0.0
    assert float(bottom_df["mean_credit_transferred"].mean()) > 0.0
    assert float(hybrid_df["mean_credit_transferred"].mean()) > 0.0
    assert "targeted_step_fraction" in top_agents.columns
    assert float(top_agents["targeted_step_fraction"].mean()) > 0.0
    assert float(bottom_agents["targeted_step_fraction"].mean()) == 0.0
    assert float(hybrid_agents["mean_credit_received"].mean()) > 0.0
    assert float(none_agents["mean_credit_received"].mean()) == 0.0


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
    generation_df, _, agent_df = run_harvest_invasion(
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
    assert set(agent_df["regime"].unique()) >= {"train", "noisy_weather", "strong_externality"}


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


def test_harvest_agent_history_records_targeting_and_credit_metrics() -> None:
    cfg = make_harvest_cfg_for_tier("medium_h1", n_agents=6, seed=19)
    population = build_initial_harvest_population(6, patch_max=cfg.patch_max, rng=np.random.default_rng(4), partner_mix_preset="balanced")
    _, _, top_agents = evaluate_harvest_population(cfg, "top_down_only", population, seeds=[1])
    _, _, hybrid_agents = evaluate_harvest_population(cfg, "hybrid", population, seeds=[1])
    assert {
        "total_welfare",
        "mean_requested_harvest",
        "mean_realized_harvest",
        "mean_prevented_harvest",
        "targeted_step_fraction",
        "capped_step_fraction",
        "mean_credit_sent",
        "mean_credit_received",
        "mean_local_patch_health",
    }.issubset(top_agents.columns)
    assert float(top_agents["targeted_step_fraction"].max()) >= 0.0
    assert float(hybrid_agents["mean_credit_received"].mean()) >= 0.0


def test_harvest_ranking_and_capability_tables_handle_multi_condition_inputs() -> None:
    table_df = pd.DataFrame(
        [
            {
                "tier": "medium_h1",
                "partner_mix": "balanced",
                "injector_mode_requested": "search_mutation",
                "adversarial_pressure": 0.3,
                "condition": "bottom_up_only",
                "test_garden_failure_mean_mean": 0.0,
                "test_mean_patch_health_mean_mean": 15.0,
                "test_mean_welfare_mean_mean": 10.0,
            },
            {
                "tier": "medium_h1",
                "partner_mix": "balanced",
                "injector_mode_requested": "search_mutation",
                "adversarial_pressure": 0.3,
                "condition": "top_down_only",
                "test_garden_failure_mean_mean": 0.0,
                "test_mean_patch_health_mean_mean": 16.0,
                "test_mean_welfare_mean_mean": 9.5,
            },
            {
                "tier": "medium_h1",
                "partner_mix": "balanced",
                "injector_mode_requested": "search_mutation",
                "adversarial_pressure": 0.3,
                "condition": "hybrid",
                "test_garden_failure_mean_mean": 0.0,
                "test_mean_patch_health_mean_mean": 17.0,
                "test_mean_welfare_mean_mean": 9.0,
            },
        ]
    )
    ranking_df = _build_ranking_table(table_df)
    assert ranking_df.loc[ranking_df["rank"] == 1, "condition"].iloc[0] == "hybrid"

    contrast_ci_df = pd.DataFrame(
        [
            {
                "tier": "medium_h1",
                "partner_mix": "balanced",
                "injector_mode_requested": "random",
                "adversarial_pressure": 0.3,
                "left_condition": "hybrid",
                "right_condition": "top_down_only",
                "contrast_name": "hybrid_minus_top_down_only",
                "delta__test_mean_patch_health_mean_mean": 0.1,
                "delta__test_mean_neighborhood_overharvest_mean_mean": -0.05,
                "delta__test_mean_welfare_mean_mean": -0.05,
            },
            {
                "tier": "medium_h1",
                "partner_mix": "balanced",
                "injector_mode_requested": "search_mutation",
                "adversarial_pressure": 0.3,
                "left_condition": "hybrid",
                "right_condition": "top_down_only",
                "contrast_name": "hybrid_minus_top_down_only",
                "delta__test_mean_patch_health_mean_mean": -0.01,
                "delta__test_mean_neighborhood_overharvest_mean_mean": 0.01,
                "delta__test_mean_welfare_mean_mean": -0.20,
            },
        ]
    )
    ladder_df = _build_capability_ladder_df(contrast_ci_df)
    row = ladder_df.iloc[0]
    assert row["first_ecological_break_injector"] == "search_mutation"
    assert row["first_control_break_injector"] == "search_mutation"
    assert row["first_costly_robustness_injector"] == "search_mutation"


def test_welfare_incidence_outputs_include_aggression_and_targeting_views() -> None:
    agent_history_df = pd.DataFrame(
        [
            {
                "tier": "medium_h1",
                "partner_mix": "balanced",
                "injector_mode_requested": "search_mutation",
                "adversarial_pressure": 0.3,
                "condition": condition,
                "run_id": 0,
                "generation": 1,
                "phase": "test",
                "regime": "default",
                "seed": 10,
                "agent_index": agent_idx,
                "aggressive_request_fraction": float(agent_idx) / 10.0,
                "mean_requested_harvest": 1.0 + agent_idx,
                "mean_welfare": 5.0 + agent_idx + (1.0 if condition == "hybrid" else 0.0),
                "mean_prevented_harvest": 0.2 * agent_idx,
                "mean_realized_harvest": 1.5 + agent_idx,
                "targeted_step_fraction": 1.0 if agent_idx < 2 else 0.0,
                "mean_local_patch_health": 12.0 - agent_idx,
            }
            for condition in ("top_down_only", "hybrid")
            for agent_idx in range(6)
        ]
    )
    outputs = _build_welfare_incidence_outputs(agent_history_df)
    assert not outputs["aggression_summary"].empty
    assert not outputs["targeting_summary"].empty
    assert set(outputs["aggression_summary"]["aggression_group"]) == {"low_aggression", "mid_aggression", "high_aggression"}
    assert set(outputs["targeting_summary"]["target_group"]) == {"targeted", "untargeted"}
