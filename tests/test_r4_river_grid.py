import numpy as np

from experiments.oversight import run_c1_compositional_harm as c1
from experiments.oversight import run_r4_river_grid as r4

SMOKE = dict(contexts=2, horizon=12, reviewer_draws=400)


def test_memory_equals_memoryless_when_nobody_lies(monkeypatch):
    monkeypatch.setattr(r4, "LIE", 0.0)
    with r4.river_target(50):
        a = r4.episode(0, 4, 1.0, "memory", SMOKE)
        b = r4.episode(0, 4, 1.0, "memoryless", SMOKE)
    assert [x["q_next"] for x in a["rows"]] == [x["q_next"] for x in b["rows"]]


def test_reproduces_c1_random_arm_at_the_30_line():
    prof = dict(contexts=1, horizon=12, reviewer_draws=400, closed_draws=200, open_draws=400)
    seeds = dict(pop=c1.SEEDS["pop"], weather=c1.SEEDS["weather"], reviewer=c1.SEEDS["reviewer"], audit=c1.SEEDS["audit"])
    _, rows_c1 = c1.episode("comp", 0, "B", "random", prof)
    with r4.river_target(30):
        e = r4.episode(0, 2, 1.0, "memory", prof, seeds=seeds, c1_compat=True)
    assert np.allclose([r["q_next"] for r in rows_c1], [r["q_next"] for r in e["rows"]])


def test_four_greedy_agents_are_two_of_each_type():
    for ctx in range(5):
        pol = r4.make_population(ctx, r4.SEEDS["pop"], 4)
        s = pol["stressed"]
        assert len(s) == 4 and sum(i < 3 for i in s) == 2 and sum(i >= 3 for i in s) == 2


def test_target_is_restored():
    from fishery_sim import two_reagent as TR
    old = TR.Q_UNSAFE
    with r4.river_target(50):
        assert TR.Q_UNSAFE == 50
    assert TR.Q_UNSAFE == old
