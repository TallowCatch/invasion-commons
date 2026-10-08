import json

import pandas as pd
import pytest

from experiments.run_budgeted_reviewer import (
    jobs_for, plan, run, verify_completion,
)


def test_fixed_matrix_and_fresh_population_seeds():
    smoke, pilot = plan("smoke"), plan("pilot")
    assert len(jobs_for(smoke)) == 14
    assert len(jobs_for(pilot)) == 400
    assert sum(job[0] == "harvest" for job in jobs_for(pilot)) == 320
    assert sum(job[0] == "fishery" for job in jobs_for(pilot)) == 80
    assert pilot["inspection_budgets"] == [0, 3, 6]
    assert pilot["population_seed_bases"] == {"mix2": 300_000_000, "mix4": 310_000_000}


def test_smoke_pairing_inventory_and_immutable_completion(tmp_path):
    run(tmp_path, "smoke", max_seconds=120)
    jobs = jobs_for(plan("smoke"))
    assert verify_completion(tmp_path, jobs)
    completion = json.loads((tmp_path / "completion.json").read_text())
    assert completion["episodes"] == 14
    assert completion["decisions"] == 6 * completion["frozen_cases"]
    quality = pd.read_csv(tmp_path / "analysis/decision_quality.csv")
    assert set(quality.inspection_budget) == {0, 6}
    assert set(quality["mode"]) == {"local_optimistic", "local_bounded", "joint"}
    assert quality[quality.inspection_budget.eq(0)].mean_request_inspections.eq(0).all()
    assert quality[quality.inspection_budget.eq(6)].mean_request_inspections.eq(6).all()
    before = (tmp_path / "completion.json").read_bytes()
    run(tmp_path, "smoke", max_seconds=120)
    assert (tmp_path / "completion.json").read_bytes() == before
    with (tmp_path / "analysis/outcomes.csv").open("a") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="Corrupt completed artifact"):
        run(tmp_path, "smoke", max_seconds=120)
