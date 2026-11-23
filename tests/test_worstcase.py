"""Tests for the worst-case additions (22-23 Nov 2026): separate networks in the IPPO trainer, the reveal_checks
observation, the "mixed" chooser, check_prob, and the just_checked and dodger bots.
Skipped without JAX and SocialJax; run with
.venv-socialjax/bin/python -m pytest -q tests/test_worstcase.py (scripts/socialjax_local_setup.sh).
"""
from __future__ import annotations

import importlib
import importlib.util
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("socialjax")

import flax.serialization  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from overseer.socialjax import HarvestOverseer, OverseerConfig  # noqa: E402
from overseer.socialjax.decentralised import DecConfig, DecentralisedOverseer  # noqa: E402
from overseer.socialjax.games import make_rules  # noqa: E402
from overseer.socialjax.harvest_bots import bot_actions, kind_codes  # noqa: E402
from overseer.socialjax.harvest_overseer import apples_near  # noqa: E402
from overseer.socialjax.reveal import SINCE_CAP, egocentric, topk_chance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "fca047d"  # the trainer before separate nets and reveal_checks
COMPARED = ("eaten", "violations", "checked", "caught", "detected", "false_alarms", "detected_count", "missed_count",
            "raw_reward", "penalty", "frozen", "apples", "cost")
TINY = dict(NUM_STEPS=20, NUM_MINIBATCHES=4, UPDATE_EPOCHS=1)


def items_of(env):
    return importlib.import_module(type(env).__module__).Items


# ------------------------------------------------------------------------------------------------ reveal channels
@pytest.mark.parametrize("game", ["harvest", "coins", "cleanup"])
def test_reveal_map_lines_up_with_the_game_channels(game):
    """The checked map is cut and rotated exactly as the game's own channels: the egocentric() view of the
    resource map equals the game's resource channel, and under "all" the checked map covers the agent's own cell."""
    rules = make_rules(game, num_inner_steps=60)
    Items = items_of(rules.env)
    # Items.apple in Harvest and Clean Up; in Coins both coins (each agent sees its own colour in the first coin
    # channel, so the two coin channels swap for agent 1 and only their sum is the same map for everyone)
    resource = (3, 4) if game == "coins" else (3,)
    self_ch = len(Items) - 1  # the channel the game uses for the observing agent itself
    for chooser in ("random", "all", "none"):
        ov = DecentralisedOverseer(rules, DecConfig(k=1, chooser=chooser, reveal_checks=True))
        obs, s = ov.reset(jax.random.PRNGKey(0))
        assert obs.shape[1:] == ov.obs_shape()
        key = jax.random.PRNGKey(1)
        for t in range(25):
            key, k_a, k_s = jax.random.split(key, 3)
            act = jax.random.randint(k_a, (ov.n,), 0, rules.env.action_space().n)
            obs, s, _, _, info = ov.step(k_s, s, act)
            g = s.env_state.grid
            locs = s.env_state.agent_locs.astype(jnp.int32)
            size, pad = rules.env.OBS_SIZE, rules.env.PADDING
            mine = egocentric(jnp.isin(g, jnp.array(resource)).astype(jnp.float32), locs, size, pad)
            theirs = sum(obs[..., v - 1] for v in resource)
            np.testing.assert_array_equal(mine, theirs, err_msg=f"{game} step {t}")
            rev = obs[..., -3:]
            marked = jnp.zeros(g.shape).at[locs[:, 0], locs[:, 1]].max(info["checked"].astype(jnp.float32))
            np.testing.assert_array_equal(rev[..., 0], egocentric(marked, locs, size, pad))
            np.testing.assert_array_equal(rev[..., 1], np.broadcast_to(np.asarray(info["checked"], np.float32)
                                                                       [:, None, None], rev[..., 1].shape))
            since = np.minimum(np.asarray(s.since_check), SINCE_CAP) / SINCE_CAP
            np.testing.assert_allclose(rev[..., 2], np.broadcast_to(since[:, None, None], rev[..., 2].shape))
            me = np.asarray(obs[..., self_ch]) > 0
            if chooser == "all":
                assert me.any(axis=(1, 2)).all() and (np.asarray(rev[..., 0])[me] == 1).all()
            if chooser == "none":
                assert (np.asarray(rev)[..., :2] == 0).all()


def test_reveal_off_leaves_the_observation_alone():
    rules = make_rules("harvest", num_inner_steps=30)
    a = DecentralisedOverseer(rules, DecConfig(k=2, chooser="random"))
    b = DecentralisedOverseer(rules, DecConfig(k=2, chooser="random", reveal_checks=True))
    oa, sa = a.reset(jax.random.PRNGKey(0))
    ob, sb = b.reset(jax.random.PRNGKey(0))
    assert ob.shape[-1] == oa.shape[-1] + 3
    for t in range(10):
        act = jnp.full((a.n,), t % 8)
        oa, sa, ra, _, _ = a.step(jax.random.PRNGKey(t), sa, act)
        ob, sb, rb, _, _ = b.step(jax.random.PRNGKey(t), sb, act)
        np.testing.assert_array_equal(oa, ob[..., :-3])
        np.testing.assert_array_equal(ra, rb)


def test_topk_chance_handles_ties():
    p = topk_chance(jnp.array([5.0, 1.0, 1.0, 1.0, 0.0]), jnp.array([True, True, True, True, False]), 2)
    np.testing.assert_allclose(p, [1.0, 1 / 3, 1 / 3, 1 / 3, 0.0])
    np.testing.assert_allclose(topk_chance(jnp.ones(4), jnp.ones(4, bool), 0), 0.0)


# ------------------------------------------------------------------------------------------------ R = 1 is central
@pytest.mark.parametrize("cfg", [
    dict(k=2, chooser="mixed", mixed_eps=0.5, penalty=5.0),
    dict(k=3, chooser="mixed", mixed_eps=0.3, reveal_checks=True, window=3),
    dict(k=2, chooser="history", reveal_checks=True, freeze_steps=4),
    dict(k=2, chooser="round_robin", reveal_checks=True),
])
def test_one_region_reproduces_the_central_overseer(cfg):
    """With the new options, R = 1 is still the central overseer step for step (through an episode reset),
    and both give the same check_prob."""
    from overseer.socialjax.harvest_bots import kind_codes as harvest_codes
    rules = make_rules("harvest", num_inner_steps=41)
    central = HarvestOverseer(rules.env, OverseerConfig(**cfg))
    dec = DecentralisedOverseer(rules, DecConfig(**cfg))
    oa, sa = central.reset(jax.random.PRNGKey(3))
    ob, sb = dec.reset(jax.random.PRNGKey(3))
    np.testing.assert_array_equal(oa, ob)
    codes = harvest_codes(("greedy", "greedy", "sneaky", "restrained", "dodger", "random", "just_checked"))
    key = jax.random.PRNGKey(9)
    for t in range(50):
        key, k_bot, k_step = jax.random.split(key, 3)
        np.testing.assert_allclose(central.check_prob(sa), dec.check_prob(sb), atol=1e-6)
        act = bot_actions(k_bot, sa, codes, check_prob=central.check_prob(sa))
        oa, sa, ra, _, ia = central.step(k_step, sa, act)
        ob, sb, rb, _, ib = dec.step(k_step, sb, act)
        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(ra, rb)
        for f in COMPARED:
            np.testing.assert_array_equal(ia[f], ib[f], err_msg=f"{f} at step {t}")
        np.testing.assert_allclose(sa.suspicion, sb.suspicion[0])


# ------------------------------------------------------------------------------------------------ check_prob
def _played_state(ov, steps=60, seed=0):
    """A state some way into an episode with greedy bots, so there are catches and suspicion to target."""
    _, s = ov.reset(jax.random.PRNGKey(seed))
    codes = kind_codes(("greedy",) * 3 + ("restrained",) * 4)
    key = jax.random.PRNGKey(seed + 1)
    for _ in range(steps):
        key, k_bot, k_step = jax.random.split(key, 3)
        _, s, _, _, _ = ov.step(k_step, s, bot_actions(k_bot, s, codes))
    return s


@pytest.mark.parametrize("chooser", ["random", "round_robin", "history", "mixed"])
@pytest.mark.parametrize("regions,k", [(1, 2), (2, 3)])
def test_check_prob_matches_how_often_agents_are_checked(chooser, regions, k):
    rules = make_rules("harvest", num_inner_steps=200)
    if regions == 1:
        ov = HarvestOverseer(rules.env, OverseerConfig(k=k, chooser=chooser, mixed_eps=0.4))
        s = _played_state(ov)
        near = apples_near(s.env_state.grid, s.env_state.agent_locs[:, :2])
        draw = jax.jit(jax.vmap(lambda key: ov._choose(key, s, near)))
    else:
        ov = DecentralisedOverseer(rules, DecConfig(k=k, chooser=chooser, regions=regions, mixed_eps=0.4))
        s = _played_state(ov)
        region = ov.regions_of(s.env_state)
        draw = jax.jit(jax.vmap(lambda key: ov._choose(key, s, region, jnp.zeros(ov.n))[0]))
    freq = np.asarray(draw(jax.random.split(jax.random.PRNGKey(5), 4000))).mean(0)
    np.testing.assert_allclose(freq, ov.check_prob(s), atol=0.035)


# ------------------------------------------------------------------------------------------------ bots
def _run(ov, kinds, steps=300, seed=0, skip=0, **bot_kw):
    codes = kind_codes(kinds)

    @jax.jit
    def go(key):
        _, s0 = ov.reset(key)

        def body(s, key):
            k_bot, k_step = jax.random.split(key)
            act = bot_actions(k_bot, s, codes, check_prob=ov.check_prob(s), **bot_kw)
            _, s, _, _, info = ov.step(k_step, s, act)
            return s, (info["violations"], info["caught"])
        return jax.lax.scan(body, s0, jax.random.split(key, steps))[1]
    v, c = go(jax.random.PRNGKey(seed))
    return int(v[skip:].sum()), int(c[skip:].sum())


def test_dodgers_and_just_checked_are_never_caught_by_round_robin():
    """Round robin is fully predictable: a dodger (and, with k = 2 of 7, an agent cheating in the 2 steps after its
    check) over-harvests and is never caught. Random checks at the same k catch the dodger whenever it cheats."""
    rules = make_rules("harvest", num_inner_steps=400)
    rr = HarvestOverseer(rules.env, OverseerConfig(k=2, chooser="round_robin", penalty=24.5))
    for kind in ("dodger", "just_checked"):
        # an episode starts with "steps since checked" = 0, so just_checked cheats in steps 0 and 1 regardless
        violations, caught = _run(rr, (kind,) * 7, explore=0.0, dodge_below=0.5, skip=2)
        assert violations > 0 and caught == 0, kind
    rnd = HarvestOverseer(rules.env, OverseerConfig(k=2, chooser="random", penalty=24.5))
    violations, caught = _run(rnd, ("dodger",) * 7, explore=0.0, dodge_below=0.5)  # 2/7 < 0.5: always cheats
    assert violations > 0 and caught > 0
    violations, _ = _run(rnd, ("dodger",) * 7, explore=0.0, dodge_below=0.25)  # 2/7 > 0.25: never cheats
    assert violations == 0


# ------------------------------------------------------------------------------------------------ trainer
@pytest.fixture(scope="module")
def trainer():
    return importlib.import_module("experiments.socialjax.train_ippo_overseer")


@pytest.fixture()
def tiny(trainer, monkeypatch):
    for k, v in TINY.items():
        monkeypatch.setitem(trainer.HPARAMS, k, v)
    return trainer


def _cfg(**kw):
    return dict(dict(env="harvest", rules=dict(min_neighbours=3), total_steps=80, num_envs=2, seed=1), **kw)


def test_shared_training_matches_the_base_trainer(tiny, tmp_path, monkeypatch):
    """The default path (shared net, no reveal) gives the same params and measures as the trainer at fca047d."""
    try:
        src = subprocess.run(["git", "show", f"{BASE_COMMIT}:experiments/socialjax/train_ippo_overseer.py"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history not available")
    path = tmp_path / "base_trainer.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("base_trainer", path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    for k, v in TINY.items():
        monkeypatch.setitem(base.HPARAMS, k, v)
    ov = DecConfig(k=2, chooser="history", penalty=3.5 * 7)
    cfg = _cfg()
    new_p, new_m = jax.jit(tiny.make_train(cfg, ov)[0])(jax.random.PRNGKey(1))
    old_p, old_m = jax.jit(base.make_train(cfg, ov)[0])(jax.random.PRNGKey(1))
    for a, b in zip(jax.tree.leaves(new_p), jax.tree.leaves(old_p)):
        np.testing.assert_array_equal(a, b)
    assert set(old_m) <= set(new_m)
    for k in old_m:
        np.testing.assert_array_equal(new_m[k], old_m[k], err_msg=k)
    assert new_m["apples_agent"].shape == (2, 7)
    np.testing.assert_allclose(new_m["apples_agent"].sum(-1), new_m["apples_eaten"], rtol=1e-5)


def test_separate_nets_train_one_network_per_agent(tiny):
    ov = DecConfig(k=2, chooser="mixed", penalty=3.5 * 7, reveal_checks=True)
    params, m = jax.jit(tiny.make_train(_cfg(separate_nets=True), ov)[0])(jax.random.PRNGKey(2))
    leaves = jax.tree.leaves(params)
    assert all(x.shape[0] == 7 for x in leaves)
    first_conv = [x for x in leaves if x.ndim == 5][0]  # (agents, kh, kw, in, out)
    assert first_conv.shape[3] == 15 + 3  # Harvest's 15 channels plus the 3 reveal channels
    assert not np.allclose(leaves[0][0], leaves[0][1])  # agents start (and stay) different
    assert all(np.isfinite(np.asarray(v)).all() for v in m.values())
    ev = tiny.make_eval(_cfg(separate_nets=True), DecConfig(chooser="none", reveal_checks=True), 2)
    out = ev(params, jax.random.PRNGKey(0))
    assert np.isfinite(float(out["apples_eaten"]))


def test_agent_minibatches_hold_only_their_own_transitions(trainer):
    n, e, T, M = 3, 4, 5, 2
    agent = jnp.broadcast_to(jnp.repeat(jnp.arange(n), e)[None, :], (T, n * e))  # stored agent-major
    stamp = jnp.arange(T * n * e).reshape(T, n * e)
    mbs = trainer.agent_minibatches(jax.random.PRNGKey(0), dict(agent=agent, stamp=stamp), n, M)
    assert mbs["agent"].shape == (M, n, T * e // M)
    for i in range(n):
        assert (np.asarray(mbs["agent"][:, i]) == i).all()
        mine = np.sort(np.asarray(mbs["stamp"][:, i]).ravel())
        np.testing.assert_array_equal(mine, np.sort(np.asarray(stamp)[np.asarray(agent) == i]))


def test_separate_policy_with_copied_params_is_the_shared_policy(trainer):
    rules = make_rules("harvest", num_inner_steps=10)
    env = DecentralisedOverseer(rules, DecConfig())
    net = trainer.socialjax_actor_critic()(env.env.action_space().n, activation="relu")
    shape = env.obs_shape()
    shared = net.init(jax.random.PRNGKey(0), jnp.zeros((1, *shape)))
    copied = trainer.init_network(net, jax.random.PRNGKey(1), shape, env.n, True,
                                  flax.serialization.to_bytes(shared))
    obs = jax.random.uniform(jax.random.PRNGKey(2), (env.n * 3, *shape))
    pi_a, v_a = trainer.policy_fn(net, env.n, shape, False)(shared, obs)
    pi_b, v_b = trainer.policy_fn(net, env.n, shape, True)(copied, obs)
    np.testing.assert_allclose(v_a, v_b, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(jax.nn.log_softmax(pi_a.logits), pi_b.logits, rtol=1e-5, atol=1e-6)
    again = trainer.init_network(net, jax.random.PRNGKey(3), shape, env.n, True, flax.serialization.to_bytes(copied))
    for a, b in zip(jax.tree.leaves(again), jax.tree.leaves(copied)):
        np.testing.assert_array_equal(a, b)


def test_worstcase_job_lists_parse_with_the_trainer(trainer):
    """Every worst-case line parses; separate-nets arms all use --separate-nets; worst-case arms also reveal checks;
    the penalty gives an average penalty of 1 apple under random checks."""
    from experiments.socialjax import decentralised_runs as d
    names = []
    for lst, flags in (("separate", ("separate_nets",)), ("worstcase", ("separate_nets", "reveal_checks"))):
        for run, args in d.LISTS[lst]():
            a = trainer.parser().parse_args(["--out", "x"] + shlex.split(args))
            DecConfig(k=a.k, chooser=a.chooser, regions=a.regions, share=a.share, mixed_eps=a.mixed_eps,
                      reveal_checks=a.reveal_checks)
            assert all(getattr(a, f) for f in flags), run
            assert a.k / 7 * a.penalty_apples == pytest.approx(1.0)
            names.append(run)
    assert len(names) == len(set(names)) == 15 + 18


def test_redesigned_job_lists_parse_with_the_trainer(trainer):
    """H2 redesign (worstcase_shared) and H6 reduced rerun (separate_long), 23 Nov: every line parses; WS_ arms reveal
    checks with a SHARED network; OS_ arms are the ordinary-mode shared baselines (no reveal, no separate nets);
    SL_ arms use separate nets at 3e8 steps; all at an average penalty of 1 apple under random checks."""
    from experiments.socialjax import decentralised_runs as d
    names = []
    for lst in ("worstcase_shared", "separate_long"):
        for run, args in d.LISTS[lst]():
            a = trainer.parser().parse_args(["--out", "x"] + shlex.split(args))
            DecConfig(k=a.k, chooser=a.chooser, regions=a.regions, share=a.share, mixed_eps=a.mixed_eps,
                      reveal_checks=a.reveal_checks)
            if run.startswith("WS_"):
                assert a.reveal_checks and not a.separate_nets, run
            elif run.startswith("OS_"):
                assert not a.reveal_checks and not a.separate_nets, run
            else:
                assert run.startswith("SL_") and a.separate_nets and not a.reveal_checks and a.total_steps == 3e8, run
            assert a.k / 7 * a.penalty_apples == pytest.approx(1.0)
            names.append(run)
    assert len(names) == len(set(names)) == 24 + 6
