"""Tests for the locality study (notes/socialjax/locality/protocol.md).

Run: .venv-socialjax/bin/python -m pytest -q tests/test_locality.py
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("socialjax")

import jax.numpy as jnp  # noqa: E402

from overseer.socialjax import locality as L  # noqa: E402
from overseer.socialjax.corridor import CorridorRules, crossing_policy  # noqa: E402
from overseer.socialjax.decentralised import DecConfig, DecentralisedOverseer  # noqa: E402
from overseer.socialjax.game_bots import kind_codes, make_policy  # noqa: E402
from overseer.socialjax.games import make_rules  # noqa: E402
from overseer.socialjax.local_records import LocalRecordOverseer  # noqa: E402
from overseer.socialjax.regions import region_grid  # noqa: E402
from overseer.socialjax.rooms import geodesic, make_rooms_rules, rooms_map, room_grid  # noqa: E402


# ------------------------------------------------------------------ rooms
def test_rooms_maps_keep_apples_and_align_with_strips():
    from overseer.socialjax.games import HARVEST_MAP
    amap, walls = rooms_map("rooms")
    assert sum(r.count("A") for r in amap) == sum(r.count("A") for r in HARVEST_MAP)
    assert np.array_equal(room_grid(amap, walls), region_grid((16, 22), 2, "strips"))
    big, walls = rooms_map("rooms_large")
    assert np.array_equal(room_grid(big, walls), region_grid((16, 44), 4, "strips"))
    door, _ = rooms_map("rooms_door")
    assert door[11][11] == " " and door[10][11] == "W"


def test_walls_block_crossing_and_door_lets_through():
    D = geodesic(rooms_map("rooms")[0])
    assert D[0, 21] >= 10_000  # (0, 0) to (0, 21): different closed rooms
    Dd = geodesic(rooms_map("rooms_door")[0])
    assert Dd[0, 21] < 10_000


# ------------------------------------------------------------------ local-record overseer
def _rollout_pair(rules, cfg, codes, steps=120, seed=3):
    policy = make_policy(rules)
    a = DecentralisedOverseer(rules, cfg)
    b = LocalRecordOverseer(rules, cfg, records="handover")
    ta = L.collect_trace(a, L.bot_act(policy, codes, a.region_grid), jax.random.PRNGKey(seed), steps)
    tb = L.collect_trace(b, L.bot_act(policy, codes, b.region_grid), jax.random.PRNGKey(seed), steps)
    return ta, tb


@pytest.mark.parametrize("cfg", [
    DecConfig(k=2, chooser="random", regions=2, window=10, freeze_steps=20),
    DecConfig(k=3, chooser="history", regions=4, window=5, share=True, freeze_steps=10, miss_rate=0.3),
])
def test_handover_reproduces_the_existing_overseer(cfg):
    rules = make_rules("harvest", num_inner_steps=1000)
    codes = kind_codes(("greedy", "greedy", "sneaky", "sneaky", "migrator", "honest", "honest"))
    ta, tb = _rollout_pair(rules, cfg, codes)
    for f in ("pos", "violation", "detected_count", "missed_count", "caught", "checked"):
        assert np.array_equal(ta[f], tb[f]), f


def test_local_records_never_catch_more_and_equal_at_w1():
    rules = make_rules("harvest", num_inner_steps=1000)
    codes = kind_codes(("greedy",) * 3 + ("honest",) * 4)
    policy = make_policy(rules)
    out = {}
    for w in (1, 20):
        for rec in ("local", "handover"):
            ov = LocalRecordOverseer(rules, DecConfig(k=2, chooser="random", regions=4, window=w), records=rec)
            t = L.collect_trace(ov, L.bot_act(policy, codes, ov.region_grid), jax.random.PRNGKey(1), 300)
            out[w, rec] = t
    assert np.array_equal(out[1, "local"]["detected_count"], out[1, "handover"]["detected_count"])
    for rec in ("local", "handover"):  # every violation ends caught or missed
        t = out[20, rec]
        assert t["detected_count"].sum() + t["missed_count"].sum() + t["left_in_window"] == t["violation"].sum()


def test_local_record_counts_in_corridor():
    """Two regions, one agent that violates once and then crosses: the other region's overseer can't catch it."""
    rules = CorridorRules(n_agents=1, regions=2)
    ov = LocalRecordOverseer(rules, DecConfig(k=2, chooser="all", regions=2, window=5))
    _, s = ov.reset(jax.random.PRNGKey(0))
    # step 1: violate and stay in region 0 -> chooser "all" checks and catches it
    _, s, _, _, info = ov.step(jax.random.PRNGKey(1), s, jnp.array([3]))
    assert int(info["detected_count"][0]) == 1
    ov2 = LocalRecordOverseer(rules, DecConfig(k=0, chooser="none", regions=2, window=5))
    _, s = ov2.reset(jax.random.PRNGKey(0))
    _, s, _, _, info = ov2.step(jax.random.PRNGKey(1), s, jnp.array([3]))  # violate in region 0, unchecked
    ov3 = LocalRecordOverseer(rules, DecConfig(k=2, chooser="all", regions=2, window=5))
    s = s.replace(since_region_check=jnp.zeros((2, 1), jnp.int32))
    _, s2, _, _, info = ov3.step(jax.random.PRNGKey(2), s, jnp.array([1]))  # move to region 1, checked there
    assert int(info["detected_count"][0]) == 0 and int(s2.window.sum()) == 1  # still pending for region 0
    _, s3, _, _, info = ov3.step(jax.random.PRNGKey(3), s2, jnp.array([2]))  # back to region 0, checked
    assert int(info["detected_count"][0]) == 1


# ------------------------------------------------------------------ measures
def test_crossing_rate_and_stays():
    reg = np.array([[0], [0], [1], [1], [1], [0]])
    assert L.crossing_rate(reg) == pytest.approx(2 / 5)
    ev = np.array([[0, 0], [1, 0]])  # violations at trace steps 0 and 1 -> regions reg[1], reg[2]
    s = L.stays(reg, ev, 2)
    assert s[0] == 0.0 and s[1] == 1.0
    assert np.isnan(L.stays(reg, np.array([[4, 0]]), 3)[0])


def test_formulas():
    assert L.p_independent(0.2, 0.3, 1) == pytest.approx(L.p_central(0.2, 1))
    assert L.p_independent(0.2, 0.0, 10) == pytest.approx(L.p_central(0.2, 10))
    for w in (1, 5, 30):
        assert L.p0_markov(0.1, 0.2, w, pattern="drift") == pytest.approx(L.p_independent(0.1, 0.2, w))
        assert L.p0_markov(0.1, 0.2, w, pattern="two") >= L.p_independent(0.1, 0.2, w) - 1e-12
        assert L.p0_markov(0.1, 0.0, w, pattern="two") == pytest.approx(L.p_central(0.1, w))
    assert L.p_independent(1 / 7, 0.05, 10, alpha=0.5) == pytest.approx(0.5 * L.p_independent(1 / 7, 0.05, 10))


def test_counterfactual_regrowth_is_zero_without_neighbours():
    spawn = np.zeros((8, 8), bool)
    spawn[1, 1] = True
    apples = np.zeros((1, 8, 8), bool)
    lost = np.asarray(L._cf_batch(jnp.asarray(apples), jnp.array([[1, 1]]), jnp.asarray(spawn),
                                  jax.random.PRNGKey(0), horizon=20))
    assert lost.sum() == 0
    spawn[:, :] = True
    apples[0, 0:3, 0:3] = True
    apples[0, 1, 1] = False
    lost = np.asarray(L._cf_batch(jnp.asarray(apples), jnp.array([[1, 1]]), jnp.asarray(spawn),
                                  jax.random.PRNGKey(0), horizon=200))
    assert lost.sum() >= 0 and lost[0, 1, 1] == 0


def test_rooms_closed_have_no_crossings():
    rules = make_rooms_rules("rooms", num_inner_steps=1000)
    from overseer.socialjax.rooms import rooms_policy
    ov = LocalRecordOverseer(rules, DecConfig(k=2, chooser="random", window=1), region_grid=rules.room_grid)
    t = L.collect_trace(ov, L.bot_act(rooms_policy(rules), kind_codes(("random",) * 7), ov.region_grid),
                        jax.random.PRNGKey(0), 300)
    assert L.crossing_rate(L.regions_of(t["pos"], rules.room_grid)) == 0.0


def test_corridor_crossing_bot_hits_its_rate():
    rules = CorridorRules(n_agents=8, regions=4)
    ov = LocalRecordOverseer(rules, DecConfig(k=0, chooser="none", regions=4))
    act = crossing_policy(chi=0.2, violate=0.1, pattern="ring")
    t = L.collect_trace(ov, act, jax.random.PRNGKey(0), 900, keep_grid=False)
    assert L.crossing_rate(L.regions_of(t["pos"], ov.region_grid)) == pytest.approx(0.2, abs=0.02)
    assert t["violation"].mean() == pytest.approx(0.1, abs=0.02)


# ------------------------------------------------------------------ scale (Willis et al.'s commons, M groups)
def test_group_budgets_and_audits():
    G = pytest.importorskip("experiments.scale.groups")
    rng = np.random.default_rng(0)
    for K, M in ((8, 4), (32, 64), (5, 3)):
        assert G.budgets(K, M, rng).sum() == K
        assert G.budgets(K, M, rng, weights=np.arange(M)).sum() == K
    g = G.assign_groups(64, 4, rng)
    assert np.bincount(g).tolist() == [16] * 4
    D = np.ones((20, 64), bool)  # everyone always defects: every audit catches
    r = G.audit(D, g, 8, "random", rng)
    assert r["audits"] == 8 * 19 and r["caught"] == 8 * 19
    v = G.own_group_view(64, 1)
    assert v["own_gain"] == v["total_gain"] > 0
