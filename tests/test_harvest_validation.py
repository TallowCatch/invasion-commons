import json

import numpy as np
import pandas as pd
import pytest

from experiments.common.extract_harvest_oversight_case import _matching_strategy_rows
from experiments.common.validate_harvest_mechanisms import joint_reference, local_cutoff, local_state, predict_health, setup
from experiments.paper_v5.run_overseer_limit_ablation import _summarise
from experiments.paper_v5.analyze_stagea_stress_regimes import STRESS_PREFIXES, BASE_METRICS, _summarise as summarise_stress
from fishery_sim.harvest import GovernmentAgent, HarvestCommonsConfig, SelfInterestedHarvestAgent, run_harvest_episode


def test_zero_capacity_has_explicit_legacy_and_floor_behaviour():
    requests, caps = np.ones(6), np.full(6, 0.2)
    legacy = GovernmentAgent(max_target_share=0)
    fixed = GovernmentAgent(max_target_share=0, capacity_rule="floor")
    assert legacy.apply_cap(requests, caps)[1].sum() == 1
    assert fixed.apply_cap(requests, caps)[1].sum() == 0
    assert GovernmentAgent(max_target_share=.33, capacity_rule="floor").apply_cap(requests, caps)[1].sum() == 1


def test_new_logging_and_identity_filter_leave_dynamics_unchanged():
    cfg = HarvestCommonsConfig(horizon=8, seed=12)
    first = run_harvest_episode(cfg, [SelfInterestedHarvestAgent() for _ in range(6)])
    second = run_harvest_episode(cfg, [SelfInterestedHarvestAgent() for _ in range(6)], record_trace=True,
                                 action_filter=lambda c, h, p: p)
    for metric in ["total_welfare", "mean_patch_health", "global_unsafe_rate", "t_end"]:
        assert first[metric] == second[metric]
    trace = second["episode_trace_rows"]
    assert second["approved_onset_count"] == sum(r["approved_onset"] for r in trace)
    assert second["approved_persistence_count"] == sum(r["approved_persistence"] for r in trace)
    assert first["local_pass_global_fail_rate"] == pytest.approx(
        (first["approved_onset_count"] + first["approved_persistence_count"]) / first["t_end"])
    assert trace[0]["pre_global_safe"] == 1


def test_invalid_action_filter_rejected():
    with pytest.raises(ValueError, match="Action filter"):
        run_harvest_episode(HarvestCommonsConfig(horizon=1), [SelfInterestedHarvestAgent() for _ in range(6)],
                            action_filter=lambda c, h, p: p + 1)


def test_prediction_matches_zero_weather_environment():
    cfg = HarvestCommonsConfig(horizon=1, weather_noise_std=0)
    out = run_harvest_episode(cfg, [SelfInterestedHarvestAgent() for _ in range(6)], record_trace=True)
    row = out["episode_trace_rows"][0]
    predicted = predict_health(cfg, np.array(json.loads(row["patch_health_before_json"])),
                               np.array(json.loads(row["allowed_fracs_json"])))
    np.testing.assert_allclose(predicted, json.loads(row["patch_health_after_json"]))


def test_state_filter_preserves_own_model_without_neighbour_damage():
    cfg = HarvestCommonsConfig(weather_noise_std=0, neighbor_externality=0)
    h = np.full(6, 12.)
    p = local_state(cfg, h, np.ones(6))
    np.testing.assert_allclose(predict_health(cfg, h, p), np.full(6, 10.))
    assert np.all(local_cutoff(cfg, h, np.ones(6)) <= .4 + 1e-12)


def test_joint_reference_accounts_for_spillovers():
    cfg = HarvestCommonsConfig(weather_noise_std=0, neighbor_externality=.3)
    h, p = np.full(6, 14.), np.ones(6)
    filtered = joint_reference(cfg, h, p)
    assert predict_health(cfg, h, filtered).mean() >= 10
    assert (filtered <= p).all()


def test_matched_setup_separates_messages_and_targeting():
    for mechanism in ["uniform_off", "uniform_on", "neighborhood_off", "neighborhood_on"]:
        cfg, governor, _ = setup(HarvestCommonsConfig(), mechanism)
        assert cfg.communication_enabled == mechanism.endswith("_on")
        assert not cfg.side_payments_enabled
        assert governor.enforcement_scope == ("global" if mechanism.startswith("uniform") else "local")
        assert governor.governance_budget_cost == 0
    _, signal, _ = setup(HarvestCommonsConfig(), "signal_only")
    requested = np.ones(6)
    allowed, targets = signal.apply_cap(requested, np.full(6, .1))
    np.testing.assert_equal(allowed, requested)
    assert not targets.any()


def test_restoration_uses_previous_rank_not_current_rank():
    df = pd.DataFrame({"strategy_id": ["g0_s0", "g0_s1", "g0_s2", "g0_s1", "g0_s0", "g1_s0"],
                       "generation": [0, 0, 0, 1, 1, 1], "rank": [1, 2, 3, 1, 2, 3]})
    out = _matching_strategy_rows(df, pd.Series({"generation": 1}))
    assert out.strategy_id.tolist() == ["g0_s0", "g0_s1", "g1_s0"]


def test_ablation_does_not_pool_scenarios(tmp_path):
    rows = [{"scenario_preset": scenario, "condition": "hybrid", "actor_capability_level": "high_actor",
             "overseer_capability_level": "weak_overseer", "capability_gap": 2, "run_id": run,
             "test_global_unsafe_rate_mean": value}
            for scenario, value in [("a", .1), ("b", .9)] for run in [0, 1]]
    pd.DataFrame(rows).to_csv(tmp_path / "check_runs.csv", index=False)
    _summarise(str(tmp_path / "check"), str(tmp_path / "summary.csv"))
    out = pd.read_csv(tmp_path / "summary.csv")
    assert len(out) == 2
    assert out.test_global_unsafe_rate_mean_mean.tolist() == pytest.approx([.1, .9])


def test_stress_generations_are_not_independent_runs():
    assert STRESS_PREFIXES["held_out_average"] == "test"
    assert "nominal" not in STRESS_PREFIXES
    rows = [{"stress_regime": "noisy_weather", "scenario_preset": "a", "condition": "hybrid",
             "condition_label": "Hybrid", "actor_capability_level": "high_actor",
             "overseer_capability_level": "weak_overseer", "capability_gap": 2,
             "run_id": run, "generation": gen, **dict.fromkeys(BASE_METRICS, float(run))}
            for run in [0, 1] for gen in [0, 1, 2]]
    result = summarise_stress(pd.DataFrame(rows)).iloc[0]
    assert result.n_runs == 2
    assert result.global_unsafe_rate_sem == pytest.approx(.5)
