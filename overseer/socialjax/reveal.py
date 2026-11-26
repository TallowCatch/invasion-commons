"""What agents see of the overseer's checks when `reveal_checks` is on (the worst case; written 22-23 Nov 2026).

By default agents see only their own 11 x 11 view and cannot tell who is checked. With `reveal_checks=True` every
agent's observation gets 3 extra channels, appended after the game's own channels so a CNN reads them like the rest:
  0. checked map: 1 on the cells (in the agent's own rotated view) of every agent that was checked last step;
  1. "I was checked last step": a constant plane of 0 or 1;
  2. "steps since I was last checked": a constant plane, min(steps, SINCE_CAP) / SINCE_CAP (0 = checked last step).
"Last step" is the step that has just been played: the observation returned by step() is the one the agents act on
next, so the checks it shows are the ones made after their previous action.

SocialJax's observation is egocentric: a window of the padded map cut at a point that depends on the agent's
position and facing, then rotated so the agent faces up (harvest_open.py, clean_up.py and coin_game.py share the code:
_get_obs_point and rotate_grid inside __init__). `egocentric` repeats those steps for any (rows, cols) map, so the
checked map lines up with the game's own channels (tests/test_worstcase.py checks it against the apple channel).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

REVEAL_CHANNELS = 3
SINCE_CAP = 20  # steps; "steps since I was last checked" saturates here
CATCH_CAP = 100  # steps; "steps since agent j was last caught" saturates here (public catches, below)


def obs_point(agent_loc, obs_size, padding):
    """Top-left corner of the agent's view in the padded map (SocialJax's _get_obs_point)."""
    x, y, d = agent_loc[0], agent_loc[1], agent_loc[2]
    x, y = x + padding - obs_size // 2, y + padding - obs_size // 2
    x = jnp.where(d == 0, x + obs_size // 2 - 1, x)
    y = jnp.where(d == 1, y + obs_size // 2 - 1, y)
    x = jnp.where(d == 2, x - obs_size // 2 + 1, x)
    y = jnp.where(d == 3, y - obs_size // 2 + 1, y)
    return x, y


def rotate(agent_loc, view):
    """SocialJax's rotate_grid: turn the view so the agent faces the same way in every observation."""
    d = agent_loc[2]
    view = jnp.where(d == 1, jnp.rot90(view, k=1, axes=(0, 1)), view)
    view = jnp.where(d == 2, jnp.rot90(view, k=2, axes=(0, 1)), view)
    return jnp.where(d == 3, jnp.rot90(view, k=3, axes=(0, 1)), view)


def egocentric(map2d, agent_locs, obs_size, padding, fill=0.0):
    """(N, obs, obs): each agent's rotated view of a (rows, cols) map, cut exactly as SocialJax cuts its own."""
    padded = jnp.pad(map2d, ((padding, padding), (padding, padding)), constant_values=fill)

    def one(loc):
        x, y = obs_point(loc, obs_size, padding)
        view = jax.lax.dynamic_slice(padded, (x, y), (obs_size, obs_size))
        return rotate(loc, view)

    return jax.vmap(one)(agent_locs.astype(jnp.int32))


def reveal_channels(env, env_state, checked, since_check):
    """(N, obs, obs, 3) extra channels: the checked map, 'checked last step' and 'steps since checked'."""
    locs = env_state.agent_locs.astype(jnp.int32)
    rows, cols = env_state.grid.shape
    marked = jnp.zeros((rows, cols), jnp.float32).at[locs[:, 0], locs[:, 1]].max(checked.astype(jnp.float32))
    size = env.OBS_SIZE
    view = egocentric(marked, locs, size, env.PADDING)
    me = jnp.broadcast_to(checked.astype(jnp.float32)[:, None, None], view.shape)
    since = jnp.minimum(since_check, SINCE_CAP).astype(jnp.float32) / SINCE_CAP
    since = jnp.broadcast_to(since[:, None, None], view.shape)
    return jnp.stack([view, me, since], axis=-1)


def augment(obs, env, env_state, checked, since_check):
    """The game's observation with the 3 reveal channels appended (dtype of obs, float32 in SocialJax)."""
    extra = reveal_channels(env, env_state, checked, since_check).astype(obs.dtype)
    return jnp.concatenate([obs, extra], axis=-1)


def public_catch_channels(since_caught, view_shape):
    """Public catches (X7, added 26 Nov 2026; notes/formulation/05_formulation.md section 5.8): (N, obs, obs, 2N)
    constant planes. For agent i, plane j is "agent (i + j) mod N was caught last step" (0 or 1) and plane N + j is
    min(steps since that agent was last caught, CATCH_CAP) / CATCH_CAP, so plane 0 and plane N are always the
    observer's own (the same order for every agent, which a shared network needs). An agent not caught yet this
    episode reads 1. The cap is longer than SINCE_CAP because catches are rarer than checks and the "history"
    chooser's suspicion (decay 0.95 per step) still ranks an agent caught 80 steps ago above one never caught."""
    n = since_caught.shape[0]
    order = (jnp.arange(n)[:, None] + jnp.arange(n)[None, :]) % n  # (N, N): observer i's j-th agent
    caught = (since_caught == 0).astype(jnp.float32)[order]
    since = (jnp.minimum(since_caught, CATCH_CAP).astype(jnp.float32) / CATCH_CAP)[order]
    planes = jnp.concatenate([caught, since], axis=1)  # (N, 2N)
    return jnp.broadcast_to(planes[:, None, None, :], (n,) + tuple(view_shape) + (2 * n,))


def augment_catches(obs, since_caught):
    """The observation with the 2N public-catch planes appended."""
    extra = public_catch_channels(since_caught, obs.shape[1:3]).astype(obs.dtype)
    return jnp.concatenate([obs, extra], axis=-1)


def topk_chance(score, member, budget):
    """Chance each member is among the `budget` highest scores, ties broken uniformly at random (as the choosers'
    1e-6 noise does). Non-members get 0. score (N,), member (N,) bool, budget a scalar (may be traced)."""
    s = jnp.where(member, score, -jnp.inf)
    higher = (member[None, :] & (s[None, :] > s[:, None])).sum(1)
    equal = (member[None, :] & (s[None, :] == s[:, None])).sum(1)
    p = jnp.clip((budget - higher) / jnp.maximum(equal, 1), 0.0, 1.0)
    return jnp.where(member, p, 0.0)
