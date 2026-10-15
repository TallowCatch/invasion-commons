"""Inventory and paired-estimand checks for the fresh-seed reviewer study."""
import math

import numpy as np
import pandas as pd

from experiments.paper_v5.analyze_budgeted_reviewer_confirmation import _cluster_interval, _difference
from experiments.paper_v5.run_budgeted_reviewer_confirmation import jobs_for, plan


def test_confirmation_inventory_and_fresh_seeds():
    cfg = plan("confirm")
    jobs = jobs_for(cfg)
    assert len(jobs) == 1920
    assert sum(job[0] == "fishery" for job in jobs) == 640
    assert sum(job[0] == "harvest" for job in jobs) == 1280
    assert cfg["population_seed_bases"] == {"mix4": 500_000_000, "mix2": 510_000_000}
    assert cfg["inspection_budgets"] == [0, 3, 6]
    assert len({(job[0], job[1], job[2], job[3], job[4], job[5], job[6]) for job in jobs}) == len(jobs)


def test_paired_difference_uses_same_context_and_resolved_denominators():
    rows = []
    for context in range(4):
        for method, safe_rejected, harmful_accepted in (
            ("joint", context % 2, 0),
            ("local_bounded", 2, 0),
            ("local_optimistic", 0, 1),
        ):
            rows.append(dict(context=context, method=method, safe_resolved=2,
                             risky_resolved=1, unresolved=3, safe_rejected=safe_rejected,
                             harmful_accepted=harmful_accepted))
    table = pd.DataFrame(rows)
    risk, safe = _difference(table, "local_bounded")
    assert risk == 0
    assert safe == -0.75
    risk, safe = _difference(table, "local_optimistic")
    assert risk == -1
    assert safe == 0.25
    bounds = _cluster_interval(table, "local_bounded", np.random.default_rng(1), 200)
    assert bounds[0] == bounds[1] == 0
    assert bounds[2] <= -0.75 <= bounds[3]


def test_missing_risky_class_does_not_become_zero_risk():
    frame = pd.DataFrame([
        dict(method=method, safe_resolved=1, risky_resolved=0,
             safe_rejected=0, harmful_accepted=0)
        for method in ("joint", "local_bounded")
    ])
    risk, safe = _difference(frame, "local_bounded")
    assert math.isnan(risk)
    assert safe == 0
