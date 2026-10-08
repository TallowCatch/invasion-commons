from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from fishery_sim.harvest import HarvestCommonsConfig, SelfInterestedHarvestAgent, run_harvest_episode
from fishery_sim.oversight_protocol import (
    METHODS, MonitorSettings, decide_harvest, harvest_nominal_next, harvest_prediction,
    harvest_reference_risk, harvest_safe, local_patch_report, score_verdict, wilson_interval,
)
from fishery_sim.config import FisheryConfig
from fishery_sim.fishery_oversight import FisherySnapshot, decide_fishery, reference_risk, transition, safe
from experiments.analyze_matched_oversight import context_summary, METRICS
from experiments.run_matched_oversight import make_manifest, harvest_block, frozen_judgments, write_json, read_json


def test_prediction_matches_native_harvest_with_identical_weather():
    cfg = HarvestCommonsConfig(horizon=8, seed=531)
    rows = run_harvest_episode(cfg, [SelfInterestedHarvestAgent() for _ in range(cfg.n_agents)], record_trace=True)["episode_trace_rows"]
    rng = np.random.default_rng(cfg.seed)
    for row in rows:
        mean = harvest_nominal_next(cfg, json.loads(row["patch_health_before_json"]), json.loads(row["allowed_fracs_json"]))
        predicted = np.clip(mean + rng.normal(0, cfg.weather_noise_std, cfg.n_agents), 0, cfg.patch_max)
        np.testing.assert_allclose(predicted, json.loads(row["patch_health_after_json"]), atol=1e-12)


def test_local_report_has_no_neighbour_input_and_bounds_joint():
    cfg = HarvestCommonsConfig()
    rng = np.random.default_rng(15)
    for _ in range(40):
        h, a = rng.uniform(0, 20, 6), rng.uniform(0, 1, 6)
        scale = float(rng.uniform())
        lower = harvest_prediction(cfg, h, a, "local_conservative", scale, .05)
        joint = np.clip(harvest_nominal_next(cfg, h, a * scale), 0, 20)
        upper = harvest_prediction(cfg, h, a, "local_nominal", scale, .05)
        assert np.all(lower <= joint + 1e-12)
        assert np.all(joint <= upper + 1e-12)
        original = local_patch_report(cfg, h[0], a[0], scale, True)
        h[1:], a[1:] = 0, 1
        assert local_patch_report(cfg, h[0], a[0], scale, True) == original


@pytest.mark.parametrize("method", METHODS)
def test_authority_and_predicate_common(method):
    cfg = HarvestCommonsConfig()
    h, a = np.full(6, 11.0), np.linspace(.2, 1, 6)
    decision = decide_harvest(cfg, h, a, method)
    assert decision.scale in MonitorSettings().scales
    assert decision.candidate_evaluations <= 5
    assert np.all(a * decision.scale <= a)
    if decision.status == "feasible":
        predicted = harvest_prediction(cfg, h, a, method, decision.scale, .05)
        assert harvest_safe(cfg, predicted)


def test_nonmonotone_config_rejected():
    with pytest.raises(ValueError, match="monotone"):
        decide_harvest(HarvestCommonsConfig(regen_rate=1.5), np.ones(6), np.ones(6), "local_conservative")


def test_budget_exhaustion_and_infeasibility_not_claimed_safe():
    cfg = HarvestCommonsConfig()
    zero = decide_harvest(cfg, np.ones(6)*10, np.ones(6), "joint_nominal", MonitorSettings(candidate_budget=0))
    assert zero.verdict == "abstain" and zero.predicted_safe is None
    assert zero.candidate_evaluations == 0
    one = decide_harvest(cfg, np.ones(6)*10, np.ones(6), "joint_nominal", MonitorSettings(candidate_budget=1))
    assert one.status == "budget_exhausted" and one.predicted_safe is None
    impossible = decide_harvest(cfg, np.zeros(6), np.zeros(6), "joint_uncertain")
    assert impossible.status == "infeasible" and impossible.predicted_safe is False


def test_union_bound_margin_reduces_prediction():
    cfg = HarvestCommonsConfig()
    h, a = np.ones(6)*14, np.ones(6)*.4
    nominal = harvest_prediction(cfg, h, a, "joint_nominal", 1, .05)
    bounded = harvest_prediction(cfg, h, a, "joint_uncertain", 1, .05)
    assert np.all(bounded < nominal)
    exact = replace(cfg, weather_noise_std=0)
    np.testing.assert_equal(harvest_prediction(exact,h,a,"joint_uncertain",1,.05),
                            harvest_prediction(exact,h,a,"joint_nominal",1,.05))


def test_reference_is_reproducible_and_does_not_mutate_state_or_global_rng():
    cfg = HarvestCommonsConfig()
    h, a = np.ones(6)*10, np.ones(6)*.4
    old = h.copy()
    first = harvest_reference_risk(cfg, h, a, 20)
    second = harvest_reference_risk(cfg, h, a, 20)
    assert first == second
    np.testing.assert_equal(h, old)
    assert wilson_interval(0, 32)[1] > .05
    assert score_verdict("approve", "unresolved")["harmful_accepted"] == 0
    assert score_verdict("reject", "safe")["safe_rejected"] == 1
    deterministic = harvest_reference_risk(replace(cfg, weather_noise_std=0), h, a, 0)
    assert deterministic["risk_lower"] == deterministic["risk_upper"]


def test_local_and_global_share_threshold_change():
    cfg = HarvestCommonsConfig(weather_noise_std=0, neighbor_externality=0)
    h, a = np.ones(6)*14, np.ones(6)*.8
    for threshold in [9, 12, 16]:
        c = replace(cfg, global_min_mean_patch_health=threshold)
        local = decide_harvest(c, h, a, "local_nominal")
        joint = decide_harvest(c, h, a, "joint_nominal")
        assert local.scale == joint.scale


def test_fishery_snapshot_purity_and_exact_reference():
    cfg = FisheryConfig(n_agents=6, stock_init=70, stock_max=100, regen_rate=.7, obs_noise_std=0, max_harvest_per_agent=6)
    state = FisherySnapshot(15, 1)
    a = np.ones(6)
    future, payoff, harvest = transition(cfg, state, a)
    assert state == FisherySnapshot(15, 1)
    assert harvest == pytest.approx(15)
    assert payoff.sum() == pytest.approx(15)
    assert reference_risk(cfg, state, a)["risk"] == float(not safe(cfg, future))
    for method in ["local_conservative", "joint_nominal"]:
        decision = decide_fishery(cfg, state, a, method)
        next_state, _, _ = transition(cfg, state, a*decision.scale)
        assert safe(cfg, next_state)
    with pytest.raises(ValueError, match="no weather"):
        decide_fishery(cfg, state, a, "joint_uncertain")


def test_uncertainty_unit_is_context_not_weather():
    records = []
    for context, value in [(0, 0), (1, 1)]:
        for weather in range(20):
            records.append(dict(game="harvest", scenario="a", regime="base", method="none", context=context,
                                weather=weather, **dict.fromkeys(METRICS, value)))
    contexts, summary = context_summary(pd.DataFrame(records))
    assert len(contexts) == 2
    assert summary.iloc[0].global_unsafe_rate_n_contexts == 2
    assert summary.iloc[0].global_unsafe_rate == .5


def test_manifest_refuses_changed_protocol_and_atomic_gzip(tmp_path):
    first = make_manifest(tmp_path, "smoke", 32, MonitorSettings())
    assert make_manifest(tmp_path, "smoke", 32, MonitorSettings()) == first
    with pytest.raises(ValueError, match="Manifest differs"):
        make_manifest(tmp_path, "pilot", 32, MonitorSettings())
    target = tmp_path / "block.json.gz"
    write_json(target, {"a": [1,2]})
    assert read_json(target) == {"a": [1,2]}


def test_frozen_monitors_share_same_case_and_reference(tmp_path):
    cfg = make_manifest(tmp_path, "smoke", 128, MonitorSettings())["protocol"]
    block = harvest_block(0, 0, "base", "none", cfg)
    rows = frozen_judgments(block, cfg)
    frame = pd.DataFrame(rows)
    assert set(frame.method) == set(METHODS)
    assert (frame.groupby(["step", "proposal_scale"]).risk.nunique() == 1).all()
    for _, values in frame.groupby(["step", "proposal_scale"]):
        assert all(a == values.iloc[0].proposals for a in values.proposals)
    assert "safe" in set(frame[frame.pre_global_safe.eq(1)].reference_label)


def test_completed_resume_preserves_evidence_and_rejects_corruption(tmp_path, monkeypatch):
    import experiments.run_matched_oversight as runner

    cfg = make_manifest(tmp_path, "smoke", 128, MonitorSettings())["protocol"]
    names = [f"harvest__{regime}__0__0__{method}"
             for regime in cfg["regimes"] for method in METHODS]
    names += [f"fishery__0__{method}" for method in runner.FISHERY_METHODS]
    for name in names:
        write_json(tmp_path / "blocks" / f"{name}.json.gz", {"synthetic": name})
    checksums = {name: runner.digest(tmp_path / "blocks" / f"{name}.json.gz") for name in names}
    write_json(tmp_path / "block_checksums.json", checksums)
    frozen = tmp_path / "frozen_judgments.json.gz"
    write_json(frozen, [])
    completion = tmp_path / "completion.json"
    write_json(completion, dict(expected_blocks=len(names), completed_blocks=len(names),
        new_blocks=len(names), invocation_seconds=123.0, frozen_judgment_rows=0,
        frozen_sha256=runner.digest(frozen)))
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis/experiment.md").write_text("Synthetic completed analysis")
    original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (completion, frozen)}

    def forbidden(*args, **kwargs):
        pytest.fail("A completed run must not execute simulations or reference labeling")

    for function in ("harvest_block", "fishery_block", "frozen_judgments"):
        monkeypatch.setattr(runner, function, forbidden)
    runner.run(tmp_path, "smoke", 128)
    for path, expected in original.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == expected
    write_json(frozen, [{"unexpected": True}])
    with pytest.raises(ValueError, match="Corrupt completed frozen"):
        runner.run(tmp_path, "smoke", 128)
    frozen.write_bytes(original[frozen][0])
    write_json(tmp_path / "blocks" / f"{names[0]}.json.gz", {"modified": True})
    with pytest.raises(ValueError, match="Corrupt completed block"):
        runner.run(tmp_path, "smoke", 128)
