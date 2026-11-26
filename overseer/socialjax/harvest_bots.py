"""Scripted Harvest players for testing the overseer before any training (phase 1).

All bots see the whole map (they are scripted, so privileged information is fine) and walk to the nearest apple
they are willing to eat:
- restrained: only apples that leave at least m apples within two squares, and it never steps onto any other apple.
  It can still break the rule by accident when another agent takes a neighbouring apple in the same step;
- greedy: any apple;
- sneaky: greedy once it has gone `sneaky_after` steps without being checked, restrained otherwise
  (a first stand-in for an agent that learns where the overseer is not looking);
- random: uniform random moves;
- migrator (added 22 Nov for decentralised overseers): greedy only inside regions whose own overseer has not checked
  it for `sneaky_after` steps, restrained elsewhere, and drawn to apples in those regions (they count as
  `migrate_pull` squares nearer). With one overseer for the whole map it behaves exactly like sneaky.
- just_checked (added 22 Nov, the worst case): greedy for the `dodge_steps` steps right after it was checked,
  restrained otherwise. It uses only what reveal_checks shows an agent about itself ("steps since I was last
  checked"), and it is safe exactly when the overseer cannot check the same agent twice in a row (round robin, or
  repeat-offenders-first while nobody has been caught);
- dodger (added 22 Nov, the worst case): greedy only when its chance of being checked this step, given the
  overseer's rule and everything public (HarvestOverseer.check_prob / DecentralisedOverseer.check_prob, passed in as
  `check_prob`), is below `dodge_below`; restrained otherwise. This is the Stackelberg follower of security games:
  it knows the overseer's commitment and the record, not the coin flips. With dodge_below = gain / penalty it cheats
  exactly when the expected penalty is below the apple it gains. Without `check_prob` it never cheats.
With probability `explore` a bot makes a random move instead, which also breaks deadlocks between agents.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from overseer.socialjax.harvest_overseer import APPLE, STAY, apple_count_grid

KINDS = ("restrained", "greedy", "sneaky", "random", "migrator", "just_checked", "dodger")
MOVES = jnp.array([4, 5, 2, 3])  # row+1, row-1, col+1, col-1
DELTA = jnp.array([[0, 0], [0, 0], [0, 1], [0, -1], [1, 0], [-1, 0], [0, 0], [0, 0]])  # harvest_open.py STEP_MOVE


def kind_codes(kinds):
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown bot kinds {sorted(unknown)}; choose from {KINDS}")
    return jnp.array([KINDS.index(k) for k in kinds], jnp.int32)


@partial(jax.jit, static_argnames=("min_neighbours", "sneaky_after", "explore", "migrate_pull", "dodge_steps"))
def bot_actions(key, state, codes, min_neighbours=3, sneaky_after=20, explore=0.1, region_grid=None,
                migrate_pull=8, check_prob=None, dodge_below=0.25, dodge_steps=2):
    """Actions (N,) for scripted bots; `state` is an OverseerState or a DecState, `codes` from kind_codes().
    `region_grid` (rows, cols) gives each cell's region for the migrator (a DecentralisedOverseer's region_grid).
    `check_prob` (N,) is each agent's chance of being checked this step (the overseer's check_prob(state)), for the
    dodger."""
    grid = state.env_state.grid
    locs = state.env_state.agent_locs[:, :2].astype(jnp.int32)
    n_rows, n_cols = grid.shape
    apples = grid == APPLE
    allowed = apples & (apple_count_grid(grid) >= min_neighbours)
    forbidden = apples & ~allowed
    rows, cols = jnp.meshgrid(jnp.arange(n_rows), jnp.arange(n_cols), indexing="ij")
    if check_prob is None:
        check_prob = jnp.ones(locs.shape[0])
    by_region = getattr(state, "since_region_check", None)
    if by_region is None or region_grid is None:  # one overseer for the whole map
        by_region, region_grid = state.since_check[None, :], jnp.zeros((n_rows, n_cols), jnp.int32)

    def act(loc, code, unchecked_for, unchecked_by_region, p_check, key):
        unwatched = (unchecked_by_region >= sneaky_after)[region_grid]  # (rows, cols): cells it would cheat in
        migrator = code == 4
        dodging = (code == 5) | (code == 6)
        greedy_now = ((code == 1) | ((code == 2) & (unchecked_for >= sneaky_after))
                      | (migrator & unwatched[loc[0], loc[1]])
                      | ((code == 5) & (unchecked_for < dodge_steps)) | ((code == 6) & (p_check < dodge_below)))
        careful = (code == 0) | ((code == 2) & ~greedy_now) | (migrator & ~greedy_now) | (dodging & ~greedy_now)
        targets = jnp.where(migrator, (apples & unwatched) | allowed, jnp.where(greedy_now, apples, allowed))
        pull = jnp.where(migrator & unwatched, migrate_pull, 0)
        dist = jnp.where(targets, jnp.abs(rows - loc[0]) + jnp.abs(cols - loc[1]) - pull, 10_000).ravel()
        i = jnp.argmin(dist)
        dr, dc = i // n_cols - loc[0], i % n_cols - loc[1]
        row_move, col_move = jnp.where(dr > 0, 4, 5), jnp.where(dc > 0, 2, 3)
        primary = jnp.where(jnp.abs(dr) >= jnp.abs(dc), row_move, col_move)
        secondary = jnp.where(jnp.abs(dr) >= jnp.abs(dc), jnp.where(dc != 0, col_move, STAY),
                              jnp.where(dr != 0, row_move, STAY))
        k_move, k_explore = jax.random.split(key)
        explore_now = ((dist[i] >= 10_000) | (code == 3)
                       | (jax.random.uniform(k_explore) < explore))
        first = jnp.where(explore_now, MOVES[jax.random.randint(k_move, (), 0, 4)], primary)

        def safe(a):
            dest = jnp.clip(loc + DELTA[a], 0, jnp.array([n_rows - 1, n_cols - 1]))
            return ~(careful & forbidden[dest[0], dest[1]])

        return jnp.where(safe(first), first, jnp.where(safe(secondary), secondary, STAY))

    keys = jax.random.split(key, locs.shape[0])
    return jax.vmap(act)(locs, codes, state.since_check, by_region.T, check_prob, keys).astype(jnp.int32)
