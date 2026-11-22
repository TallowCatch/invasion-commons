"""Tests for decentralised overseers and the Coins / Clean Up ports (overseer/socialjax/decentralised.py, games.py,
game_bots.py). Skipped without JAX and SocialJax; run with
.venv-socialjax/bin/python -m pytest -q tests/test_decentralised_overseer.py (scripts/socialjax_local_setup.sh).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("socialjax")

import jax.numpy as jnp  # noqa: E402

from overseer.socialjax import HarvestOverseer, OverseerConfig  # noqa: E402
from overseer.socialjax.decentralised import DecConfig, DecentralisedOverseer  # noqa: E402
from overseer.socialjax.game_bots import kind_codes, make_policy  # noqa: E402
from overseer.socialjax.game_rollout import run_episode, summarise  # noqa: E402
from overseer.socialjax.games import make_rules  # noqa: E402
from overseer.socialjax.harvest_bots import bot_actions  # noqa: E402
from overseer.socialjax.harvest_bots import kind_codes as harvest_codes  # noqa: E402
from overseer.socialjax.regions import neighbours, region_grid  # noqa: E402

STEPS = 200
COMPARED = ("eaten", "violations", "checked", "caught", "detected", "false_alarms", "detected_count", "missed_count",
            "raw_reward", "penalty", "frozen", "apples", "cost")


@pytest.fixture(scope="module")
def rules():
    return {g: make_rules(g, num_inner_steps=STEPS + 1) for g in ("harvest", "coins", "cleanup")}


def episode(rules, kinds, seed=0, steps=STEPS, **cfg):
    ov = DecentralisedOverseer(rules, DecConfig(**cfg))
    s_end, trace = run_episode(ov, jax.random.PRNGKey(seed), kind_codes(kinds), make_policy(rules), steps=steps)
    return summarise(s_end, trace, rules.reward_unit), trace


# ------------------------------------------------------------------------------------------------ regions
def test_region_layouts():
    g = region_grid((16, 22), 2, "strips")
    assert (g[:, :11] == 0).all() and (g[:, 11:] == 1).all()
    q = region_grid((16, 22), 4, "grid")
    assert q[0, 0] == 0 and q[0, 21] == 1 and q[15, 0] == 2 and q[15, 21] == 3
    assert set(np.unique(region_grid((19, 28), 4, "strips"))) == {0, 1, 2, 3}
    assert (region_grid((16, 11), 2, "rows")[:8] == 0).all()
    nb = neighbours(4, "grid")
    assert nb[0, 1] and nb[0, 2] and not nb[0, 3] and (nb == nb.T).all()
    assert neighbours(4, "strips").sum() == 6 and neighbours(1).sum() == 0


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        DecConfig(region_k=(1, 1), regions=2, k=3)
    with pytest.raises(ValueError):
        DecConfig(layout="hexes")


# ------------------------------------------------------------------------------------------------ R = 1 is central
@pytest.mark.parametrize("cfg", [
    dict(k=2, chooser="random"),
    dict(k=3, chooser="history", window=3, penalty=5.0),
    dict(k=2, chooser="round_robin", freeze_steps=4),
    dict(k=1, chooser="location", miss_rate=0.3, false_alarm_rate=0.4),
    dict(chooser="all", window=2),
    dict(chooser="none"),
])
def test_one_region_reproduces_the_central_overseer(cfg):
    """Step for step, including an episode reset: same observations, rewards, logs and evidence windows."""
    rules = make_rules("harvest", num_inner_steps=41)
    central = HarvestOverseer(rules.env, OverseerConfig(**cfg))
    dec = DecentralisedOverseer(rules, DecConfig(**cfg))
    _, sa = central.reset(jax.random.PRNGKey(3))
    _, sb = dec.reset(jax.random.PRNGKey(3))
    codes = harvest_codes(("greedy", "greedy", "sneaky", "restrained", "restrained", "random", "greedy"))
    key = jax.random.PRNGKey(9)
    for t in range(50):
        key, k_bot, k_step = jax.random.split(key, 3)
        act = bot_actions(k_bot, sa, codes)
        oa, sa, ra, _, ia = central.step(k_step, sa, act)
        ob, sb, rb, _, ib = dec.step(k_step, sb, act)
        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(ra, rb)
        for f in COMPARED:
            np.testing.assert_array_equal(ia[f], ib[f], err_msg=f"{f} at step {t}")
        np.testing.assert_array_equal(sa.window, sb.window)
        np.testing.assert_array_equal(sa.since_check, sb.since_check)
        np.testing.assert_allclose(sa.suspicion, sb.suspicion[0])


# ------------------------------------------------------------------------------------------------ regions in action
@pytest.mark.parametrize("chooser", ["random", "round_robin", "history", "location"])
def test_overseers_check_only_their_own_region_within_budget(rules, chooser):
    r = rules["harvest"]
    ov = DecentralisedOverseer(r, DecConfig(k=3, chooser=chooser, regions=2))
    _, trace = run_episode(ov, jax.random.PRNGKey(1), kind_codes(("greedy",) * 7), make_policy(r), steps=60)
    checks_r = np.asarray(trace["checks_r"])
    agents_r = np.asarray(trace["agents_r"])
    assert (checks_r.sum(1) <= 3).all() and (checks_r <= 2).all()  # 3 checks split 1 + 1, one extra at random
    assert (checks_r <= agents_r).all()
    full = np.minimum(agents_r, 1).sum(1)  # each region with agents uses at least its base budget of 1
    assert (checks_r.sum(1) >= full).all()
    assert ((checks_r + np.asarray(trace["unused_r"])).sum(1) == 3).all()


def test_check_lands_in_the_agents_region(rules):
    r = rules["harvest"]
    ov = DecentralisedOverseer(r, DecConfig(k=2, chooser="random", regions=2))
    _, s = ov.reset(jax.random.PRNGKey(0))
    _, s, _, _, info = ov.step(jax.random.PRNGKey(1), s, jnp.full(7, 6))
    locs = np.asarray(s.env_state.agent_locs[:, :2])
    assert (np.asarray(info["region"]) == (locs[:, 1] >= 11)).all()
    checked = np.asarray(info["checked"])
    for reg in (0, 1):
        assert checked[np.asarray(info["region"]) == reg].sum() <= 1


@pytest.mark.parametrize("game,kinds,cfg", [
    ("harvest", ("greedy",) * 7, dict(k=2, chooser="history", regions=2, window=3, share=True)),
    ("harvest", ("migrator",) * 7, dict(k=4, chooser="random", regions=4, layout="grid", window=2)),
    ("coins", ("greedy", "greedy"), dict(k=1, chooser="random", regions=2, layout="rows")),
    ("cleanup", ("greedy",) + ("honest",) * 6, dict(k=2, chooser="random", regions=2, window=120)),
    ("cleanup", ("greedy",) + ("honest",) * 6, dict(k=2, chooser="random", regions=2, window=40, presume=True)),
])
def test_every_violation_is_detected_or_missed_in_some_region(rules, game, kinds, cfg):
    m, _ = episode(rules[game], kinds, **cfg)
    assert m["violations"] > 0
    assert m["violations"] == m["detected"] + m["missed"]
    reg = m["regions"]
    assert sum(reg["violations"]) == m["violations"]
    assert sum(reg["detected"]) == m["detected"] and sum(reg["missed"]) == m["missed"]
    assert sum(reg["checks"]) == m["checks"]


def test_sharing_costs_messages_and_does_not_change_random_checks(rules):
    r = rules["harvest"]
    kinds = ("greedy",) * 7
    a, _ = episode(r, kinds, k=2, chooser="random", regions=4, share=False)
    b, _ = episode(r, kinds, k=2, chooser="random", regions=4, share=True, share_every=5, cost_per_message=0.5)
    for f in ("violations", "detected", "checks", "taken"):
        assert a[f] == b[f]
    assert a["comm_cost"] == 0
    assert b["comm_cost"] == pytest.approx(6 * 0.5 * len(range(0, STEPS, 5)))  # 4 strips: 6 directed links


def test_sharing_passes_suspicion_to_neighbours(rules):
    r = rules["harvest"]
    ov = DecentralisedOverseer(r, DecConfig(k=2, chooser="history", regions=2, share=True))
    _, s = ov.reset(jax.random.PRNGKey(0))
    s = s.replace(suspicion=s.suspicion.at[0, 3].set(1.0))
    _, s, _, _, _ = ov.step(jax.random.PRNGKey(1), s, jnp.full(7, 6))
    assert float(s.suspicion[1, 3]) == pytest.approx(0.95)
    ov2 = DecentralisedOverseer(r, DecConfig(k=2, chooser="history", regions=2, share=False))
    _, s2 = ov2.reset(jax.random.PRNGKey(0))
    s2 = s2.replace(suspicion=s2.suspicion.at[0, 3].set(1.0))
    _, s2, _, _, _ = ov2.step(jax.random.PRNGKey(1), s2, jnp.full(7, 6))
    assert float(s2.suspicion[1, 3]) == 0.0


# ------------------------------------------------------------------------------------------------ bots
def test_migrator_is_sneaky_with_one_overseer(rules):
    r = rules["harvest"]
    a, _ = episode(r, ("sneaky",) * 7, k=1, chooser="random")
    b, _ = episode(r, ("migrator",) * 7, k=1, chooser="random")
    assert a["taken_per_agent"] == b["taken_per_agent"] and a["violations"] == b["violations"]


def test_migrator_cheats_more_when_overseers_are_local(rules):
    r = rules["harvest"]
    central, _ = episode(r, ("migrator",) * 7, k=2, chooser="random", steps=STEPS)
    local, _ = episode(r, ("migrator",) * 7, k=2, chooser="random", regions=2, steps=STEPS)
    assert local["violations"] > central["violations"]


def test_harvest_large_map():
    r = make_rules("harvest_large", num_inner_steps=60)
    assert r.env.num_agents == 14 and r.shape == (16, 44) and r.reward_unit == 14
    m, _ = episode(r, ("greedy",) * 14, steps=50, k=4, chooser="random", regions=4)
    assert m["taken"] > 0 and len(m["regions"]["checks"]) == 4


def test_coins_rule_matches_the_payoffs(rules):
    r = rules["coins"]
    m, trace = episode(r, ("greedy", "honest"), chooser="none")
    honest, _ = episode(r, ("honest", "honest"), chooser="none")
    v = np.asarray(trace["violations"])
    assert v[:, 0].sum() > 0 and v[:, 1].sum() <= 2  # honest only by a coin spawning under its step
    # the greedy player's thefts are exactly what the honest player lost
    assert m["stolen_from_per_agent"][1] == m["violations_per_agent"][0]
    raw = np.asarray(trace["raw_reward"]).sum(0)
    own = np.asarray(trace["taken"]).sum(0) - m["violations_per_agent"]
    np.testing.assert_allclose(raw, 2 * own + 2 * np.array(m["violations_per_agent"])
                               - 4 * np.array(m["stolen_from_per_agent"]))
    assert honest["welfare"] > m["welfare"]


def test_cleanup_honest_bots_clean_and_free_riders_do_not(rules):
    r = rules["cleanup"]
    m, _ = episode(r, ("greedy",) + ("honest",) * 6, chooser="none")
    assert m["cleaned_per_agent"][0] == 0
    assert min(m["cleaned_per_agent"][1:]) > 0
    assert m["violations_per_agent"][0] > 0 and sum(m["violations_per_agent"][1:]) == 0


def test_cleanup_short_looks_cannot_prove_free_riding(rules):
    """Proving 'no cleaning in the last W steps' needs a record of at least W steps."""
    r = rules["cleanup"]
    W = r.lookback
    kinds = ("greedy",) + ("honest",) * 6
    short, _ = episode(r, kinds, chooser="all", window=W - 1)
    long, _ = episode(r, kinds, chooser="all", window=W + 20)
    presumed, _ = episode(r, kinds, chooser="all", window=10, presume=True)
    assert short["violations"] > 0 and short["detected"] == 0
    assert long["detected"] > 0
    assert presumed["detected"] == presumed["violations"]  # presuming guilt catches every free-ride ...
    assert presumed["presumed_wrong"] > 0  # ... and flags honest agents who cleaned before the look began


# ------------------------------------------------------------------------------------------------ GPU job lists
def test_gpu_job_lists_parse_with_the_trainer():
    """Every line of the decentralised job lists (and stage 1) parses with the trainer's own arguments and gives a
    valid overseer config; names are unique; random checks get the stated average penalty."""
    import re
    import shlex

    from experiments.socialjax import decentralised_runs, phase2_runs
    from experiments.socialjax.train_ippo_overseer import parser

    n_agents = {"harvest": 7, "harvest_large": 14, "coins": 2, "cleanup": 7}
    names = []
    for name, fn in decentralised_runs.LISTS.items():
        if name in ("worstcase_pilot", "separate", "worstcase"):  # tests/test_worstcase.py
            continue
        for run, args in fn():
            a = parser().parse_args(["--out", "x"] + shlex.split(args))
            DecConfig(k=a.k, chooser=a.chooser if a.k > 0 else "none", window=a.window, regions=a.regions,
                      layout=a.layout, share=a.share, presume=a.presume)
            c = re.search(r"_c([0-9.]+)", run)
            if c and a.chooser == "random":
                assert a.k / n_agents[a.env] * a.penalty_apples == pytest.approx(float(c.group(1)), rel=1e-3)
            names.append(run)
    assert len(names) == len(set(names)) == 4 + 69 + 21 + 45 + 24
    for _, args in phase2_runs.stage1():
        parser().parse_args(["--out", "x"] + shlex.split(args))
