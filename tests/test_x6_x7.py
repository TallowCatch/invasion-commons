"""Tests for X6 (headcount-proportional regional budgets, Prop 5.12) and X7 (public catches, section 5.8) in
overseer/socialjax/decentralised.py and reveal.py (notes/formulation/05_formulation.md; added 26 Nov 2026).
Skipped without JAX and SocialJax; run with .venv-socialjax/bin/python -m pytest -q tests/test_x6_x7.py.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("socialjax")

import jax.numpy as jnp  # noqa: E402

from overseer.socialjax.decentralised import DecConfig, DecentralisedOverseer  # noqa: E402
from overseer.socialjax.games import make_rules  # noqa: E402
from overseer.socialjax.harvest_bots import bot_actions, kind_codes  # noqa: E402
from overseer.socialjax.reveal import CATCH_CAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "48ad9a7"  # decentralised.py before region_alloc and public_catches
INFO = ("violations", "checked", "caught", "detected", "false_alarms", "detected_count", "missed_count", "raw_reward",
        "penalty", "frozen", "cost", "region", "checks_r", "catches_r", "unused_r", "violations_r", "detected_r",
        "missed_r")
KINDS = ("greedy", "greedy", "sneaky", "restrained", "restrained", "random", "greedy")


@pytest.fixture(scope="module")
def harvest():
    return make_rules("harvest", num_inner_steps=61)


def _base_module():
    try:
        src = subprocess.run(["git", "show", f"{BASE_COMMIT}:overseer/socialjax/decentralised.py"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history not available")
    import sys
    spec = importlib.util.spec_from_loader("base_decentralised", loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["base_decentralised"] = mod  # dataclasses look their module up here
    exec(compile(src, "base_decentralised.py", "exec"), mod.__dict__)
    return mod


# ------------------------------------------------------------------------------------------------ defaults unchanged
@pytest.mark.parametrize("cfg", [
    dict(k=2, chooser="random", regions=4, penalty=24.5),
    dict(k=3, chooser="history", regions=2, window=3, share=True),
    dict(k=2, chooser="mixed", regions=1, reveal_checks=True, freeze_steps=3),
])
def test_defaults_match_the_overseer_before_x6_x7(harvest, cfg):
    """With region_alloc "fixed" and public_catches off, every observation, reward and logged field is the same as
    decentralised.py at 48ad9a7, step for step through an episode reset."""
    base = _base_module()
    old = base.DecentralisedOverseer(harvest, base.DecConfig(**cfg))
    new = DecentralisedOverseer(harvest, DecConfig(**cfg))
    assert new.obs_shape() == old.obs_shape()
    oa, sa = old.reset(jax.random.PRNGKey(3))
    ob, sb = new.reset(jax.random.PRNGKey(3))
    np.testing.assert_array_equal(oa, ob)
    codes = kind_codes(KINDS)
    key = jax.random.PRNGKey(9)
    for t in range(70):
        key, k_bot, k_step = jax.random.split(key, 3)
        np.testing.assert_array_equal(old.check_prob(sa), new.check_prob(sb))
        act = bot_actions(k_bot, sb, codes)
        oa, sa, ra, _, ia = old.step(k_step, sa, act)
        ob, sb, rb, _, ib = new.step(k_step, sb, act)
        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(ra, rb)
        for f in INFO:
            np.testing.assert_array_equal(ia[f], ib[f], err_msg=f"{f} at step {t}")


def test_bad_options_rejected():
    with pytest.raises(ValueError):
        DecConfig(region_alloc="equal")
    with pytest.raises(ValueError):
        DecConfig(k=2, regions=2, region_k=(1, 1), region_alloc="proportional")


# ------------------------------------------------------------------------------------------------ X6 allocation
@pytest.mark.parametrize("alloc", ["proportional", "proportional_random"])
def test_allocations_sum_to_k_and_round_the_headcount_share(harvest, alloc):
    """For every k and every way of placing 7 agents in R regions: the budgets sum to exactly k, no region gets more
    checks than agents, and each region gets floor or ceil of k N_r / N."""
    rng = np.random.default_rng(0)
    for R in (2, 3, 4):
        for k in range(1, 8):
            ov = DecentralisedOverseer(harvest, DecConfig(k=k, chooser="random", regions=R, region_alloc=alloc))
            regions = jnp.asarray(rng.integers(0, R, (300, 7)))
            regions = regions.at[0].set(0).at[1].set(jnp.arange(7) % R)  # everyone in one region; spread out
            keys = jax.random.split(jax.random.PRNGKey(R * 10 + k), 300)
            b = np.asarray(jax.vmap(ov.budget)(keys, regions))
            counts = np.stack([np.bincount(np.asarray(r), minlength=R) for r in regions])
            q = k * counts / 7
            assert (b.sum(1) == k).all(), (R, k)
            assert (b <= counts).all() and (b >= np.floor(q)).all() and (b <= np.ceil(q)).all(), (R, k)
            if alloc == "proportional":  # largest remainder: a region with a smaller remainder never gets more
                frac = q - np.floor(q)
                up = b > np.floor(q)
                for row_up, row_frac in zip(up, frac):
                    if row_up.any() and (~row_up).any():
                        assert row_frac[row_up].min() >= row_frac[~row_up].max() - 1e-9


def test_largest_remainder_example(harvest):
    """3, 2, 1, 1 agents in 4 regions with k = 2: shares 6/7, 4/7, 2/7, 2/7, so the two crowded regions get one each."""
    ov = DecentralisedOverseer(harvest, DecConfig(k=2, chooser="random", regions=4, region_alloc="proportional"))
    b = ov.budget(jax.random.PRNGKey(0), jnp.array([0, 0, 0, 1, 1, 2, 3]))
    np.testing.assert_array_equal(b, [1, 1, 0, 0])


def _played(ov, steps, seed, kinds):
    """Per-step checked masks and check_prob with the given bot population, from the start of an episode."""
    codes = kind_codes(kinds)

    @jax.jit
    def go(key):
        _, s0 = ov.reset(key)

        def body(s, key):
            k_bot, k_step = jax.random.split(key)
            p = ov.check_prob(s)
            _, s, _, _, info = ov.step(k_step, s, bot_actions(k_bot, s, codes))
            return s, (info["checked"], p, info["checks_r"].sum())
        return jax.lax.scan(body, s0, jax.random.split(key, steps))[1]
    return [np.asarray(x) for x in go(jax.random.PRNGKey(seed))]


@pytest.mark.parametrize("alloc,R", [("proportional", 4), ("proportional", 2), ("proportional_random", 4)])
def test_each_agents_look_chance_is_about_k_over_n(alloc, R):
    """Proportional budgets with random choice and 7 agents who all behave alike: every check is used (k each step),
    and over an episode each agent is looked at in about k / N = 2/7 of steps. check_prob averages to k / N too; for
    the unbiased rounding it is exactly k / N in every state, whoever the agents are."""
    rules = make_rules("harvest", num_inner_steps=1001)
    ov = DecentralisedOverseer(rules, DecConfig(k=2, chooser="random", regions=R, region_alloc=alloc))
    checked, p, used = _played(ov, 1000, seed=R, kinds=("random",) * 7)
    assert (used == 2).all()
    np.testing.assert_allclose(checked.mean(0), 2 / 7, atol=0.035)
    np.testing.assert_allclose(p.sum(1), 2.0, atol=1e-5)
    np.testing.assert_allclose(p.mean(0), 2 / 7, atol=0.035)
    if alloc == "proportional_random":
        _, p_mixed, _ = _played(ov, 300, seed=R, kinds=KINDS)
        np.testing.assert_allclose(p_mixed, 2 / 7, atol=1e-6)


def test_largest_remainder_favours_crowded_regions(harvest):
    """The known bias of largest remainder (why proportional_random exists): with 3, 2, 1, 1 agents and k = 2, the
    agents in the crowded regions are looked at with chance 1/3 and 1/2, and the two lone agents never; with
    unbiased rounding every agent's chance is 2/7."""
    region = jnp.array([0, 0, 0, 1, 1, 2, 3])
    for alloc, want in (("proportional", [1 / 3] * 3 + [1 / 2] * 2 + [0, 0]), ("proportional_random", [2 / 7] * 7)):
        ov = DecentralisedOverseer(harvest, DecConfig(k=2, chooser="random", regions=4, region_alloc=alloc))
        _, s = ov.reset(jax.random.PRNGKey(0))
        s = s.replace(env_state=s.env_state.replace(agent_locs=_locs_in(ov, s, region)))
        np.testing.assert_allclose(ov.check_prob(s), want, atol=1e-6)
        draw = jax.vmap(lambda k: ov._choose(k, s, region, jnp.zeros(ov.n))[0])
        freq = np.asarray(draw(jax.random.split(jax.random.PRNGKey(1), 4000))).mean(0)
        np.testing.assert_allclose(freq, want, atol=0.035)


def _locs_in(ov, s, region):
    """Agent locations (row, col, facing) putting agent i in region[i]: the first free cells of each strip."""
    grid = np.asarray(ov.region_grid)
    free = np.asarray(s.env_state.grid) == 0
    locs = np.asarray(s.env_state.agent_locs).copy()
    used = set()
    for i, r in enumerate(np.asarray(region)):
        cells = [(a, b) for a, b in zip(*np.nonzero((grid == r) & free)) if (a, b) not in used]
        locs[i, :2] = cells[0]
        used.add(cells[0])
    return jnp.asarray(locs)


@pytest.mark.parametrize("alloc", ["proportional", "proportional_random"])
@pytest.mark.parametrize("chooser", ["random", "history"])
def test_check_prob_matches_how_often_agents_are_checked(harvest, alloc, chooser):
    ov = DecentralisedOverseer(harvest, DecConfig(k=3, chooser=chooser, regions=4, region_alloc=alloc))
    _, s = ov.reset(jax.random.PRNGKey(0))
    codes = kind_codes(("greedy",) * 3 + ("restrained",) * 4)
    key = jax.random.PRNGKey(1)
    for _ in range(40):
        key, k_bot, k_step = jax.random.split(key, 3)
        _, s, _, _, _ = ov.step(k_step, s, bot_actions(k_bot, s, codes))
    region = ov.regions_of(s.env_state)
    draw = jax.jit(jax.vmap(lambda k: ov._choose(k, s, region, jnp.zeros(ov.n))[0]))
    freq = np.asarray(draw(jax.random.split(jax.random.PRNGKey(5), 4000))).mean(0)
    np.testing.assert_allclose(freq, ov.check_prob(s), atol=0.035)


# ------------------------------------------------------------------------------------------------ X7 public catches
def test_public_catches_off_leaves_everything_alone(harvest):
    a = DecentralisedOverseer(harvest, DecConfig(k=2, chooser="history", penalty=24.5))
    b = DecentralisedOverseer(harvest, DecConfig(k=2, chooser="history", penalty=24.5, public_catches=True))
    assert b.obs_shape()[-1] == a.obs_shape()[-1] + 14
    oa, sa = a.reset(jax.random.PRNGKey(0))
    ob, sb = b.reset(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(oa, ob[..., :-14])
    codes = kind_codes(KINDS)
    key = jax.random.PRNGKey(4)
    for t in range(70):
        key, k_bot, k_step = jax.random.split(key, 3)
        act = bot_actions(k_bot, sa, codes)
        oa, sa, ra, _, ia = a.step(k_step, sa, act)
        ob, sb, rb, _, ib = b.step(k_step, sb, act)
        np.testing.assert_array_equal(oa, ob[..., :-14])
        np.testing.assert_array_equal(ra, rb)
        for f in INFO:
            np.testing.assert_array_equal(ia[f], ib[f], err_msg=f"{f} at step {t}")


@pytest.mark.parametrize("extra", [dict(), dict(reveal_checks=True)])
def test_public_catch_planes_show_every_agents_catches(harvest, extra):
    """Plane j of agent i is agent (i + j) mod N: "caught last step", then min(steps since caught, cap) / cap.
    At the start (and after the episode resets) nobody has been caught: 0 and 1."""
    n = 7
    ov = DecentralisedOverseer(harvest, DecConfig(k=3, chooser="random", penalty=10.0, public_catches=True, **extra))
    obs, s = ov.reset(jax.random.PRNGKey(0))
    assert obs.shape[1:] == ov.obs_shape()
    pc = np.asarray(obs[..., -2 * n:])
    assert (pc[..., :n] == 0).all() and (pc[..., n:] == 1).all()
    codes = kind_codes(("greedy",) * 7)
    since = np.full(n, CATCH_CAP)
    key = jax.random.PRNGKey(2)
    caught_any = False
    for t in range(80):  # the episode (60 steps) resets at step 60
        key, k_bot, k_step = jax.random.split(key, 3)
        obs, s, _, dones, info = ov.step(k_step, s, bot_actions(k_bot, s, codes))
        caught = np.asarray(info["caught"])
        since = np.where(caught, 0, since + 1)
        if bool(dones["__all__"]):
            since = np.full(n, CATCH_CAP)
            caught = np.zeros(n, bool)
        caught_any |= caught.any()
        pc = np.asarray(obs[..., -2 * n:])
        assert (pc == pc[:, :1, :1, :]).all()  # constant planes
        for i in range(n):
            order = (i + np.arange(n)) % n
            np.testing.assert_array_equal(pc[i, 0, 0, :n], caught[order].astype(np.float32), err_msg=f"t {t}")
            np.testing.assert_allclose(pc[i, 0, 0, n:], np.minimum(since[order], CATCH_CAP) / CATCH_CAP)
    assert caught_any


def test_public_catches_trainer_and_switch_off_eval(monkeypatch):
    """The trainer builds a network on the wider observation, and the switch-off test (overseer off) keeps it."""
    import importlib
    tr = importlib.import_module("experiments.socialjax.train_ippo_overseer")
    for k, v in dict(NUM_STEPS=20, NUM_MINIBATCHES=4, UPDATE_EPOCHS=1).items():
        monkeypatch.setitem(tr.HPARAMS, k, v)
    a = tr.parser().parse_args(["--out", "x", "--k", "2", "--chooser", "history", "--public-catches",
                                "--regions", "2", "--region-alloc", "proportional"])
    assert a.public_catches and a.region_alloc == "proportional"
    cfg = dict(env="harvest", rules=dict(min_neighbours=3), total_steps=80, num_envs=2, seed=1)
    ov = DecConfig(k=2, chooser="history", penalty=24.5, public_catches=True, regions=2, region_alloc="proportional")
    params, m = jax.jit(tr.make_train(cfg, ov)[0])(jax.random.PRNGKey(0))
    first_conv = [x for x in jax.tree.leaves(params) if x.ndim == 4][0]
    assert first_conv.shape[2] == 15 + 14
    assert all(np.isfinite(np.asarray(v)).all() for v in m.values())
    off = DecConfig(chooser="none", k=0, regions=2, public_catches=True)
    out = tr.make_eval(cfg, off, 2)(params, jax.random.PRNGKey(1))
    assert np.isfinite(float(out["apples_eaten"]))
    assert tr.parser().parse_args(["--out", "x"]).region_alloc == "fixed"
    assert not tr.parser().parse_args(["--out", "x"]).public_catches
