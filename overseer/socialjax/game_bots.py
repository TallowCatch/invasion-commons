"""Scripted players for every game the decentralised overseer covers (Harvest, Coins, Clean Up).

Every game has the same five kinds, with the same codes as harvest_bots.KINDS:
- honest (code 0; "restrained" in Harvest): keeps the rule;
- greedy (1): breaks the rule whenever it gains from it;
- sneaky (2): greedy once no overseer has checked it for `sneaky_after` steps, honest otherwise;
- random (3): random moves;
- migrator (4): greedy only inside regions whose own overseer has not checked it for `sneaky_after` steps, honest
  elsewhere, and drawn to goods in those regions (they count as `migrate_pull` squares nearer). With one overseer
  for the whole map it is sneaky.
Bots are scripted, see the whole map and the overseers' check records (privileged information is fine for
scripted bots), and do not learn: a penalty does not change what they do; a freeze holds them still.

A policy is called as policy(key, state, codes, mem, region_grid) -> (actions, mem), where state is a DecState and
mem is the bots' own memory (only Clean Up bots use it).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from overseer.socialjax import harvest_bots
from overseer.socialjax.games import CU_APPLE, CU_STEP, CU_ZAP_CLEAN

KINDS = ("honest", "greedy", "sneaky", "random", "migrator")
MOVES = jnp.array([4, 5, 2, 3])  # row+1, row-1, col+1, col-1 (STEP_MOVE in harvest_open, coin_game, clean_up)
DELTA = jnp.array([[0, 0], [0, 0], [0, 1], [0, -1], [1, 0], [-1, 0], [0, 0], [0, 0], [0, 0]])
STAY, TURN_LEFT, TURN_RIGHT = 6, 0, 1  # turn_left: facing d -> d + 1; turn_right: d -> d - 1
FAR = 10_000


def kind_codes(kinds):
    kinds = tuple("honest" if k == "restrained" else k for k in kinds)
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown bot kinds {sorted(unknown)}; choose from {KINDS}")
    return jnp.array([KINDS.index(k) for k in kinds], jnp.int32)


def _grids(shape):
    return jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]), indexing="ij")


def walk(key, loc, cost, forbidden, careful, random_now, explore):
    """One move towards the cheapest cell of `cost` (FAR = not a target), avoiding `forbidden` cells if careful."""
    n_rows, n_cols = cost.shape
    flat = cost.ravel()
    i = jnp.argmin(flat)
    dr, dc = i // n_cols - loc[0], i % n_cols - loc[1]
    row_move, col_move = jnp.where(dr > 0, 4, 5), jnp.where(dc > 0, 2, 3)
    primary = jnp.where(jnp.abs(dr) >= jnp.abs(dc), row_move, col_move)
    secondary = jnp.where(jnp.abs(dr) >= jnp.abs(dc), jnp.where(dc != 0, col_move, STAY),
                          jnp.where(dr != 0, row_move, STAY))
    primary = jnp.where((dr == 0) & (dc == 0), STAY, primary)
    k_move, k_explore = jax.random.split(key)
    explore_now = (flat[i] >= FAR) | random_now | (jax.random.uniform(k_explore) < explore)
    first = jnp.where(explore_now, MOVES[jax.random.randint(k_move, (), 0, 4)], primary)

    def safe(a):
        dest = jnp.clip(loc + DELTA[a], 0, jnp.array([n_rows - 1, n_cols - 1]))
        return ~(careful & forbidden[dest[0], dest[1]])

    return jnp.where(safe(first), first, jnp.where(safe(secondary), secondary, STAY))


def _watch(state, region_grid, sneaky_after):
    """Per agent: unchecked by anyone for long (N,), and the cells of regions that have not checked it (N, rows, cols)."""
    unwatched_any = state.since_check >= sneaky_after
    unwatched_cells = (state.since_region_check >= sneaky_after).T[:, region_grid]
    return unwatched_any, unwatched_cells


def init_mem(n):
    return dict(burst_left=jnp.zeros(n, jnp.int32))


# ------------------------------------------------------------------------------------------------ Harvest
def harvest_policy(min_neighbours=3, sneaky_after=20, explore=0.1, migrate_pull=8):
    def policy(key, state, codes, mem, region_grid):
        return harvest_bots.bot_actions(key, state, codes, min_neighbours=min_neighbours, sneaky_after=sneaky_after,
                                        explore=explore, region_grid=region_grid, migrate_pull=migrate_pull), mem
    return policy


# ------------------------------------------------------------------------------------------------ Coins
def coins_policy(sneaky_after=20, explore=0.1, migrate_pull=8):
    """Honest coins bots take only coins of their own colour and never step onto the other's; greedy take any."""
    @jax.jit
    def policy(key, state, codes, mem, region_grid):
        grid = state.env_state.grid
        locs = state.env_state.agent_locs[:, :2].astype(jnp.int32)
        rows, cols = _grids(grid.shape)
        unwatched_any, unwatched_cells = _watch(state, region_grid, sneaky_after)
        own_code = jnp.array([3, 4])
        coins = (grid == 3) | (grid == 4)

        def act(i, loc, code, un_any, un_cells, key):
            own = grid == own_code[i]
            other = coins & ~own
            migrator = code == 4
            greedy_now = (code == 1) | ((code == 2) & un_any) | (migrator & un_cells[loc[0], loc[1]])
            targets = jnp.where(migrator, own | (other & un_cells), jnp.where(greedy_now, coins, own))
            pull = jnp.where(migrator & un_cells, migrate_pull, 0)
            cost = jnp.where(targets, jnp.abs(rows - loc[0]) + jnp.abs(cols - loc[1]) - pull, FAR)
            return walk(key, loc, cost, other, ~greedy_now, code == 3, explore)

        keys = jax.random.split(key, locs.shape[0])
        acts = jax.vmap(act)(jnp.arange(locs.shape[0]), locs, codes, unwatched_any, unwatched_cells, keys)
        return acts.astype(jnp.int32), mem
    return policy


# ------------------------------------------------------------------------------------------------ Clean Up
def cleanup_policy(rules, sneaky_after=20, explore=0.05, migrate_pull=8, burst=5, slack=30):
    """Clean Up bots.

    Honest: when the river is polluted above the rule's threshold and it last cleaned at least W - slack steps ago,
    it goes to the river and fires the cleaning beam until it has removed dirt `burst` times, then goes back to
    eat. While it would break the rule by eating (polluted, no cleaning in the last W steps) it never steps onto an
    apple. Free-riders (greedy, and sneaky or migrator while unwatched) never clean and eat whatever is nearest.
    """
    W, thr = rules.lookback, rules.threshold
    step = CU_STEP
    shape = rules.shape
    orchard = jnp.zeros(shape, bool).at[rules.env.POTENTIAL_APPLE[:, 0], rules.env.POTENTIAL_APPLE[:, 1]].set(True)

    def hits(dirt):
        """(4, rows, cols): dirt cells a beam fired from each cell facing each direction would reach."""
        pad = jnp.pad(dirt.astype(jnp.int32), 2)

        def shifted(dr, dc):
            return pad[2 + dr:2 + dr + shape[0], 2 + dc:2 + dc + shape[1]]
        out = []
        for d in range(4):
            a, r, l_ = step[d], step[(d + 1) % 4], step[(d - 1) % 4]
            out.append(sum(shifted(int(x[0]), int(x[1])) for x in (a, 2 * a, a + r, a + l_)))
        return jnp.stack(out)

    @jax.jit
    def policy(key, state, codes, mem, region_grid):
        env_state = state.env_state
        grid = env_state.grid
        locs = env_state.agent_locs[:, :2].astype(jnp.int32)
        facing = env_state.agent_locs[:, 2].astype(jnp.int32)
        rows, cols = _grids(shape)
        apples = grid == CU_APPLE
        dirt = rules.dirt_grid(env_state)
        H = hits(dirt)
        can_clean = H.max(0) > 0
        polluted = rules.pollution(env_state) > thr
        since_clean = state.aux["since_clean"]
        unwatched_any, unwatched_cells = _watch(state, region_grid, sneaky_after)
        burst_left = jnp.where(since_clean == 0, jnp.maximum(mem["burst_left"] - 1, 0), mem["burst_left"])

        def act(loc, d, code, sc, bl, un_any, un_cells, key):
            migrator = code == 4
            free_ride = (code == 1) | ((code == 2) & un_any) | (migrator & un_cells[loc[0], loc[1]])
            owing = polluted & (sc + 1 >= W)  # eating now would break the rule
            start = polluted & (sc >= W - slack) & (bl == 0)
            bl = jnp.where(free_ride | (code == 3), 0, jnp.where(start, burst, bl))
            cleaning = (bl > 0) | (owing & ~free_ride)
            dist = jnp.abs(rows - loc[0]) + jnp.abs(cols - loc[1])
            # cleaning: fire if facing dirt, turn if dirt is in reach another way, else walk to the nearest spot
            here = H[:, loc[0], loc[1]]
            want = jnp.argmax(here > 0)
            diff = (want - d) % 4
            turn = jnp.where(diff == 3, TURN_RIGHT, TURN_LEFT)
            k_walk, k_eat = jax.random.split(key)
            to_river = walk(k_walk, loc, jnp.where(can_clean, dist, FAR), apples, owing, False, explore)
            clean_act = jnp.where(here[d] > 0, CU_ZAP_CLEAN, jnp.where(here.max() > 0, turn, to_river))
            # eating: nearest apple (migrators prefer unwatched regions); with no apples, wait in the orchard
            pull = jnp.where(migrator & un_cells, migrate_pull, 0)
            apple_cost = jnp.where(apples, dist - pull, FAR)
            wait_cost = jnp.where(orchard, dist, FAR)
            any_apple = apples.any()
            in_orchard = orchard[loc[0], loc[1]]
            eat_act = walk(k_eat, loc, jnp.where(any_apple, apple_cost, wait_cost), apples, owing & ~free_ride,
                           (code == 3) | (~any_apple & in_orchard), explore)
            return jnp.where(cleaning, clean_act, eat_act), bl

        keys = jax.random.split(key, locs.shape[0])
        acts, bl = jax.vmap(act)(locs, facing, codes, since_clean, burst_left, unwatched_any, unwatched_cells, keys)
        return acts.astype(jnp.int32), dict(burst_left=bl)
    return policy


def make_policy(rules, **kw):
    if rules.name == "harvest":
        return harvest_policy(min_neighbours=rules.m, **kw)
    if rules.name == "coins":
        return coins_policy(**kw)
    if rules.name == "cleanup":
        return cleanup_policy(rules, **kw)
    raise ValueError(rules.name)
