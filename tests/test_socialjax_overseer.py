"""Tests for the budgeted overseer on SocialJax Harvest (overseer/socialjax). Skipped without JAX and SocialJax;
run with .venv-socialjax/bin/python -m pytest -q tests/test_socialjax_overseer.py (scripts/socialjax_local_setup.sh).
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("socialjax")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from overseer.socialjax import CHOOSERS, HarvestOverseer, OverseerConfig, apples_near, make_harvest_env  # noqa: E402
from overseer.socialjax.harvest_bots import bot_actions, kind_codes  # noqa: E402
from overseer.socialjax.harvest_overseer import APPLE, STAY  # noqa: E402
from overseer.socialjax.rollout import run_episode, summarise  # noqa: E402

STEPS = 200


@pytest.fixture(scope="module")
def env():
    return make_harvest_env(num_inner_steps=STEPS + 1)


def episode(env, kinds, seed=0, steps=STEPS, **cfg):
    ov = HarvestOverseer(env, OverseerConfig(**cfg))
    return summarise(*run_episode(ov, jax.random.PRNGKey(seed), kind_codes(kinds), steps=steps))


def test_neighbourhood_matches_harvest_regrowth():
    grid = jnp.zeros((7, 7), jnp.int16)
    for r, c in [(3, 4), (3, 5), (1, 3), (4, 4), (0, 0), (3, 3)]:
        grid = grid.at[r, c].set(APPLE)
    # around (3, 3): (3, 4) one step, (3, 5) two steps, (1, 3) two steps, (4, 4) diagonal; (0, 0) too far;
    # the cell itself is not counted
    assert int(apples_near(grid, jnp.array([[3, 3]]))[0]) == 4
    assert int(apples_near(grid, jnp.array([[0, 0]]))[0]) == 0


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        OverseerConfig(chooser="psychic")
    with pytest.raises(ValueError):
        OverseerConfig(window=0)


def test_every_violation_is_detected_or_missed(env):
    for chooser in CHOOSERS:
        m = episode(env, ("greedy",) * 7, k=2, chooser=chooser, window=3)
        assert m["violations"] == m["detected"] + m["missed"], chooser
        assert m["violations"] > 0


def test_no_checks_misses_everything_and_full_checks_miss_nothing(env):
    none = episode(env, ("greedy",) * 7, chooser="none")
    assert none["checks"] == 0 and none["detected"] == 0 and none["missed"] == none["violations"]
    full = episode(env, ("greedy",) * 7, chooser="all")
    assert full["missed"] == 0 and full["detected"] == full["violations"] and full["checks"] == 7 * STEPS


@pytest.mark.parametrize("chooser", ["random", "round_robin", "history", "location"])
def test_exactly_k_checks_per_step(env, chooser):
    ov = HarvestOverseer(env, OverseerConfig(k=3, chooser=chooser))
    _, trace = run_episode(ov, jax.random.PRNGKey(1), kind_codes(("greedy",) * 7), steps=50)
    assert (np.asarray(trace["checked"]).sum(1) == 3).all()


def test_round_robin_checks_everyone_in_turn(env):
    ov = HarvestOverseer(env, OverseerConfig(k=2, chooser="round_robin"))
    _, trace = run_episode(ov, jax.random.PRNGKey(2), kind_codes(("restrained",) * 7), steps=7)
    assert (np.asarray(trace["checked"]).sum(0) == 2).all()


def test_penalty_comes_off_the_reward_and_freeze_holds_the_agent(env):
    ov = HarvestOverseer(env, OverseerConfig(chooser="all", penalty=5.0, freeze_steps=4))
    _, s = ov.reset(jax.random.PRNGKey(3))
    codes = kind_codes(("greedy",) * 7)
    key = jax.random.PRNGKey(4)
    for _ in range(STEPS - 1):
        key, k_bot, k_step = jax.random.split(key, 3)
        _, s, reward, _, info = ov.step(k_step, s, bot_actions(k_bot, s, codes))
        if bool(info["caught"].any()):
            break
    else:
        pytest.fail("nobody was caught")
    caught = np.asarray(info["caught"])
    np.testing.assert_allclose(np.asarray(reward), np.asarray(info["raw_reward"]) - 5.0 * caught)
    assert (np.asarray(s.frozen)[caught] == 4).all()
    before = np.asarray(s.env_state.agent_locs[:, :2])
    _, s2, _, _, _ = ov.step(jax.random.PRNGKey(5), s, jnp.full(7, 4))  # everyone tries to move
    after = np.asarray(s2.env_state.agent_locs[:, :2])
    np.testing.assert_array_equal(after[caught], before[caught])


def test_imperfect_inspector(env):
    blind = episode(env, ("greedy",) * 7, chooser="all", miss_rate=1.0)
    assert blind["detected"] == 0 and blind["missed"] == blind["violations"]
    jumpy = episode(env, ("restrained",) * 7, chooser="all", false_alarm_rate=1.0)
    assert jumpy["wrongly_stopped"] > 0
    assert jumpy["wrongly_stopped"] + jumpy["detected"] <= jumpy["checks"]
    # false alarms only fall on honest harvests, never on agents that took nothing
    assert jumpy["wrongly_stopped"] <= jumpy["apples_eaten"]


def test_restrained_bots_keep_the_orchard_greedy_bots_strip_it(env):
    restrained = episode(env, ("restrained",) * 7, chooser="none")
    greedy = episode(env, ("greedy",) * 7, chooser="none")
    assert restrained["apples_end"] > greedy["apples_end"]
    assert restrained["violations"] < greedy["violations"]


def test_sneaky_bot_cheats_less_when_checked_often(env):
    kinds = ("sneaky",) * 7
    unwatched = episode(env, kinds, chooser="none")
    watched = episode(env, kinds, chooser="all")  # checked every step, so never unchecked for long
    assert watched["violations"] < unwatched["violations"]


def test_state_resets_with_the_episode():
    short = make_harvest_env(num_inner_steps=10)
    ov = HarvestOverseer(short, OverseerConfig(k=1, chooser="history", window=5))
    _, s = ov.reset(jax.random.PRNGKey(6))
    codes = kind_codes(("greedy",) * 7)
    key = jax.random.PRNGKey(7)
    for t in range(10):
        key, k_bot, k_step = jax.random.split(key, 3)
        _, s, _, dones, _ = ov.step(k_step, s, bot_actions(k_bot, s, codes))
    assert bool(dones["__all__"])
    assert int(s.window.sum()) == 0 and int(s.since_check.sum()) == 0 and float(s.suspicion.sum()) == 0.0


def test_stay_action_matches_socialjax():
    from socialjax.environments.common_harvest.harvest_open import Actions, Items
    assert int(Actions.stay) == STAY and int(Items.apple) == APPLE
