import json

import numpy as np
import pandas as pd
import pytest

from experiments.run_heldout_oversight import (
    FISHERY_METHODS, HARVEST_METHODS, evaluate_case, execute, jobs_for, reference_seed,
    plan, run, verify_completion,
)


def test_fixed_pilot_matrix_and_disjoint_context_seeds():
    cfg = plan("pilot")
    jobs = jobs_for(cfg)
    assert len(jobs) == 160
    assert sum(job[0] == "harvest" for job in jobs) == 128
    assert sum(job[0] == "fishery" for job in jobs) == 32
    assert cfg["population_seed_bases"]["useful"] != cfg["population_seed_bases"]["stress"]
    assert cfg["population_seed_bases"]["useful"] != 130_000_000
    assert set(HARVEST_METHODS) == {
        "none", "local_uncertain", "local_conservative_uncertain", "joint_uncertain"}
    assert set(FISHERY_METHODS) == {
        "none", "local_nominal", "local_conservative", "joint_nominal"}
    mixed = plan("mixed_pilot")
    assert len(jobs_for(mixed)) == 160
    assert mixed["population_seed_bases"] == {"mix2": 250_000_000, "mix4": 260_000_000}
    assert not set(mixed["population_seed_bases"].values()) & set(cfg["population_seed_bases"].values())


@pytest.mark.parametrize("group", ["useful", "stress"])
def test_new_policy_groups_are_paired_and_use_declared_ranges(group):
    cfg = plan("smoke")
    cfg["population_seed_bases"][group] = 220_000_000 if group == "stress" else 210_000_000
    fishery = [execute(("fishery", group, "deterministic", 0, 0, method), cfg)
               for method in FISHERY_METHODS]
    assert all(block["policies"] == fishery[0]["policies"] for block in fishery)
    low, high = np.array(fishery[0]["policies"]["low"]), np.array(fishery[0]["policies"]["high"])
    low_bounds = (.05, .20) if group == "useful" else (.45, .65)
    high_bounds = (.20, .40) if group == "useful" else (.75, .95)
    assert np.all((low_bounds[0] <= low) & (low <= low_bounds[1]))
    assert np.all((high_bounds[0] <= high) & (high <= high_bounds[1]))
    harvest = [execute(("harvest", group, "base", 0, 0, method), cfg)
               for method in HARVEST_METHODS]
    assert all(block["policies"] == harvest[0]["policies"] for block in harvest)
    assert all(block["config"] == harvest[0]["config"] for block in harvest)
    assert {policy["origin"] for policy in harvest[0]["policies"]} == {
        "cooperative_seed" if group == "useful" else "adversarial_seed"}


def test_same_frozen_original_is_judged_by_every_method():
    cfg = plan("smoke")
    block = execute(("harvest", "useful", "base", 0, 0, "none"), cfg)
    row = block["trace"][0]
    case = dict(case_id="harvest__useful__base__0__0__0", game="harvest",
        policy_group="useful", regime="base", context=0, weather=0, step=0,
        config=block["config"], pre_global_safe=row["pre_global_safe"],
        state=json.loads(row["patch_health_before_json"]),
        proposals=json.loads(row["requested_fracs_json"]))
    label, decisions = evaluate_case(case, cfg)
    assert label["reference_label"] in {"safe", "risky", "unresolved"}
    assert {d["method"] for d in decisions} == set(HARVEST_METHODS)
    assert len({d["case_id"] for d in decisions}) == 1
    assert max(d["candidate_evaluations"] for d in decisions) <= 5
    same_content = dict(case, case_id="other_weather_trace")
    same_content["config"] = dict(case["config"], seed=case["config"]["seed"] + 1)
    assert reference_seed(case, cfg["reference_seed_base"]) == reference_seed(
        same_content, cfg["reference_seed_base"])
    assert evaluate_case(same_content, cfg)[0]["reference_label"] == label["reference_label"]
    case["pre_global_safe"] = 1 - case["pre_global_safe"]
    with pytest.raises(ValueError, match="safety flag mismatch"):
        evaluate_case(case, cfg)


@pytest.mark.parametrize("group,count", [("mix2", 2), ("mix4", 4)])
def test_mixed_policy_composition_and_pairing(group, count):
    cfg = plan("mixed_pilot")
    harvest = [execute(("harvest", group, "base", 0, 0, method), cfg)
               for method in HARVEST_METHODS]
    assert all(block["policies"] == harvest[0]["policies"] for block in harvest)
    assert sum(policy["origin"] == "adversarial_seed" for policy in harvest[0]["policies"]) == count
    fishery = [execute(("fishery", group, "deterministic", 0, 0, method), cfg)
               for method in FISHERY_METHODS]
    assert all(block["policies"] == fishery[0]["policies"] for block in fishery)
    low = fishery[0]["policies"]["low"]
    assert sum(.45 <= value <= .65 for value in low) == count


def test_smoke_completes_and_completed_artifacts_are_immutable(tmp_path):
    run(tmp_path, "smoke", max_seconds=120)
    jobs = jobs_for(plan("smoke"))
    assert verify_completion(tmp_path, jobs)
    completion = json.loads((tmp_path / "completion.json").read_text())
    assert completion["episodes"] == 8
    assert completion["decisions"] == 4 * completion["frozen_cases"]
    outcomes = pd.read_csv(tmp_path / "analysis/outcomes.csv")
    decisions = pd.read_csv(tmp_path / "analysis/decision_quality.csv")
    assert set(outcomes.method) == set(HARVEST_METHODS) | set(FISHERY_METHODS)
    assert set(decisions.method) == set(HARVEST_METHODS) | set(FISHERY_METHODS)
    before = (tmp_path / "completion.json").read_bytes()
    run(tmp_path, "smoke", max_seconds=120)
    assert (tmp_path / "completion.json").read_bytes() == before
    with (tmp_path / "analysis/coverage.csv").open("a") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="Corrupt completed artifact"):
        run(tmp_path, "smoke", max_seconds=120)
