"""Tests for the BazaarBench offline overseer replay (no data download needed)."""
from __future__ import annotations

import numpy as np

from experiments.bazaarbench import bb_data as bd
from experiments.bazaarbench import replay as rp


def _toy(n=4, D=3):
    """4 agents, 3 days. Agent 0 offends every day (object 0 visible days 0-2, object 1 day 1);
    agent 2 has one object on day 2 that reaches S5 on day 2; agents 1 and 3 are clean."""
    visible_full = np.zeros((3, D), dtype=bool)
    visible_full[0, :] = True
    visible_full[1, 1] = True
    visible_full[2, 2] = True
    visible_public = visible_full.copy()
    visible_public[0, :] = False  # object 0 only in private reasoning
    return rp.CellArrays(
        agents=np.arange(n), obj_agent=np.array([0, 0, 2]),
        visible={"full": visible_full, "public": visible_public, "record": np.zeros_like(visible_full)},
        cost={"full": np.full((n, D), 10.0), "public": np.full((n, D), 7.0), "record": np.zeros((n, D))},
        activity=np.array([[1, 1, 1], [5, 5, 5], [2, 2, 2], [0, 0, 0]], dtype=float),
        stage=np.array([1, 2, 5]), day_s4=np.array([-1, -1, 2]), day_s5=np.array([-1, -1, 2]),
        zips=["10001", "10002", "90001", "90002"])


def test_check_everyone_catches_all_and_costs_all_tokens():
    ca = _toy()
    out = rp.simulate(ca, "random", budget_kind="count", budget=4)
    m = rp.measures(ca, rp.catch_days(ca, out["checked"], "full"))
    assert m["caught"] == 1.0
    assert out["tokens"] == 4 * 3 * 10.0
    # the S5 object is first visible on the day it reaches S5, so it is never caught before harm
    assert m["before_s5"] == 0.0


def test_public_view_misses_reasoning_only_object():
    ca = _toy()
    out = rp.simulate(ca, "random", view="public", budget_kind="count", budget=4)
    m = rp.measures(ca, rp.catch_days(ca, out["checked"], "public"))
    assert m["n_caught"] == 2  # objects 1 and 2, not the reasoning-only object 0


def test_repeat_offenders_sticks_with_an_agent_it_caught():
    ca = _toy()
    # with k=1, the first check is a random tie; once agent 0 is caught it should be checked again
    rng = np.random.default_rng(3)
    for _ in range(20):
        out = rp.simulate(ca, "repeat", budget_kind="count", budget=1, rng=rng)
        ch = out["checked"]
        if ch[0, 0]:
            assert ch[0, 1] and ch[0, 2]
        else:
            # a clean check scores 0, so the next day goes to an agent not yet checked
            first = int(np.argmax(ch[:, 0]))
            assert not ch[first, 1]


def test_most_active_picks_highest_activity():
    ca = _toy()
    out = rp.simulate(ca, "active", budget_kind="count", budget=1)
    assert out["checked"][1].all() and out["checked"].sum() == 3


def test_token_budget_skips_agents_that_do_not_fit():
    ca = _toy()
    ca.cost["full"][0, :] = 100.0
    out = rp.simulate(ca, "active", budget_kind="tokens", budget=25)
    # order is 1, 2, 0, 3; agent 0 (100 tokens) does not fit, so agents 1 and 2 are checked
    assert out["checked"][[1, 2]].all() and not out["checked"][0].any() and not out["checked"][3].any()
    assert out["tokens"] == 3 * 20.0


def test_independent_groups_without_savings_can_starve():
    ca = _toy()
    groups = [np.array([0, 1]), np.array([2, 3])]
    out = rp.simulate(ca, "random", budget_kind="tokens", budget=15, groups=groups, mode="independent")
    assert out["checks"] == 0  # each group gets 7.5 tokens, a check costs 10
    out = rp.simulate(ca, "random", budget_kind="tokens", budget=15, groups=groups, mode="independent",
                      savings=True)
    assert out["checks"] > 0
    shared = rp.simulate(ca, "random", budget_kind="tokens", budget=15, groups=groups, mode="shared",
                         comm_cost=0.0)
    assert shared["checks"] == 3
    costly = rp.simulate(ca, "random", budget_kind="tokens", budget=15, groups=groups, mode="shared",
                         comm_cost=10.0)
    assert costly["checks"] == 0 and costly["comm_tokens"] == 3 * 15.0  # messages capped at the budget


def test_central_savings_carry_budget_over():
    ca = _toy()
    out = rp.simulate(ca, "random", budget_kind="tokens", budget=6, savings=True)
    assert out["checks"] == 1  # 6 + 6 = 12 tokens on day 1 buys one 10-token check


def test_count_quotas_sum_to_k_and_rotate():
    for k in (1, 2, 5, 10):
        qs = [rp._count_quotas(k, [3, 7, 5, 5], d) for d in range(7)]
        assert all(sum(q) == k for q in qs)
    assert len({tuple(rp._count_quotas(1, [5, 5, 5, 5], d)) for d in range(4)}) == 4


def test_region_groups_use_first_zip_digit():
    gs = rp.region_groups(["10001", "90001", "02345", "75555", "46140"])
    assert [list(g) for g in gs] == [[0, 2], [4], [1, 3]]


def test_failure_objects_merge_on_shared_ids():
    def dec(bid, actor, ch, links, stage="considered", tick=361):
        return bd.UnsafeDecision(bundle_id=bid, channel=ch, actor=actor, max_stage=stage,
                                 evidence=[bd.Evidence(stage=stage, tick=tick, source_kind="call_ids")],
                                 links=set(links) | {f"bundle:{bid}"}, subtype="")
    ds = [dec("b1", 1, bd.CHANNELS[4], {"thread_ids:7"}),
          dec("b2", 1, bd.CHANNELS[4], {"thread_ids:7", "message_ids:3"}, stage="realised", tick=380),
          dec("b3", 1, bd.CHANNELS[0], {"thread_ids:7"}),  # other type: separate
          dec("b4", 2, bd.CHANNELS[4], {"thread_ids:7"})]  # other actor: separate
    objs = bd.build_failure_objects(ds)
    assert len(objs) == 3
    merged = [o for o in objs if len(o.decisions) == 2][0]
    assert merged.max_stage_num == 5 and merged.first_day_at_stage(5) == bd.tick_to_day(380) == 1
    assert merged.visible_days("record") == set()  # F5 is not a record type


def test_score_on_acts_ignores_reasoning_only_finds():
    ca = _toy()
    found_all = ca.found_counts("full")
    found_acts = ca.found_counts("full", acts_only=True)
    assert found_all[0].tolist() == [1, 2, 1] and found_acts[0].tolist() == [0, 1, 0]


def test_cheapest_first_orders_by_cost():
    ca = _toy()
    ca.cost["full"][:, :] = np.array([[40.0], [30.0], [10.0], [20.0]])
    out = rp.simulate(ca, "cheapest", budget_kind="tokens", budget=30)
    assert out["checked"][[2, 3]].all() and out["checked"].sum() == 6
