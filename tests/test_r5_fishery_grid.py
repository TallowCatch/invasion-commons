import json
from pathlib import Path

from experiments.oversight import run_r2_robustness_reviewer_model as r2
from experiments.oversight import run_r5_fishery_grid as r5


def test_reproduces_r2_memory_episodes():
    ref = json.loads((Path(__file__).parent / "data_r2_memory_ref.json").read_text())
    P = r2.profile("full")
    for row in ref:
        e = r5.episode(4, 1.0, row["context"], row["mode"], P, seeds=r2.SEEDS)
        assert (e["exec_risky"], e["scored_steps"]) == (row["exec_risky"], row["scored_steps"]), row


def test_grid_rates():
    assert r5.cell(2, 0.85).level == 0.595 and r5.cell(4, 1.0).level == 0.7
