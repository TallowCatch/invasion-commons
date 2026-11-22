"""Harvest with rooms: SocialJax Harvest Open on maps with walls that cut crossings (written 22 Nov 2026).

The mechanics are Harvest Open's (overseer/socialjax/games.py, HarvestRules); only the map changes.
SocialJax's `harvest_common_open` takes a `map_ASCII` in which `W` is a wall, and walls block movement
(harvest_open.py: cells that are not empty, an apple or an agent are `blocked` before `resolve_movement`).

Maps:
- `rooms`: the default 16 x 22 map with a wall down column 11, so two closed rooms (columns 0-10 and 12-21).
  No apple cell is lost (column 11 is empty above the spawn rows); 3 spawn cells are. Each room has one of the
  map's two "inside" spawn cells (Q, row 6), so every episode starts with at least one agent per room.
- `rooms_door`: the same wall with a one-cell door at row 11 (`door_rows` sets other doors).
- `rooms_large`: the 14-agent map (the default map tiled twice, 16 x 44) with walls at columns 11, 22 and 33, so
  four closed rooms. The wall at column 22 replaces the 4 apple cells of the right tile's column 0 (the seam patch).

`room_grid` gives every cell's room (walls belong to the room on their right, as in regions.region_grid's strips,
so that for `rooms` the rooms are exactly strips R = 2 and for `rooms_large` exactly strips R = 4).
"""
from __future__ import annotations

import numpy as np

from overseer.socialjax.games import HARVEST_MAP, HarvestRules
from overseer.socialjax.harvest_overseer import make_harvest_env

ROOM_MAPS = ("rooms", "rooms_door", "rooms_large")


def _wall(rows, col, door_rows=()):
    out = []
    for r, row in enumerate(rows):
        ch = " " if r in door_rows else "W"
        out.append(row[:col] + ch + row[col + 1:])
    return out


def rooms_map(name="rooms", door_rows=None):
    """(map rows, wall columns) for one of ROOM_MAPS."""
    if name == "rooms":
        return _wall(HARVEST_MAP, 11), (11,)
    if name == "rooms_door":
        return _wall(HARVEST_MAP, 11, door_rows=(11,) if door_rows is None else tuple(door_rows)), (11,)
    if name == "rooms_large":
        amap = [row * 2 for row in HARVEST_MAP]
        for c in (11, 22, 33):
            amap = _wall(amap, c)
        return amap, (11, 22, 33)
    raise ValueError(f"unknown rooms map {name!r}; choose from {ROOM_MAPS}")


def room_grid(amap, walls):
    """(rows, cols) int: the room of every cell, counting walls from the left; a wall cell joins the room to its
    right."""
    rows, cols = len(amap), len(amap[0])
    col_room = np.searchsorted(np.asarray(walls), np.arange(cols), side="right")
    return np.broadcast_to(col_room[None, :], (rows, cols)).astype(np.int32).copy()


class RoomsHarvestRules(HarvestRules):
    """HarvestRules on a rooms map. Same rule, same judge; `room_grid` and `rooms` describe the rooms."""

    def __init__(self, name="rooms", num_inner_steps=1000, min_neighbours=3, num_agents=None, door_rows=None):
        amap, walls = rooms_map(name, door_rows)
        n = num_agents or (14 if name == "rooms_large" else 7)
        self.env = make_harvest_env(num_inner_steps=num_inner_steps, num_agents=n, map_ASCII=amap,
                                    grid_size=(len(amap), len(amap[0])))
        self.shape = (len(amap), len(amap[0]))
        self.m = min_neighbours
        self.reward_unit = float(self.env.num_agents)
        self.map_ascii = amap
        self.walls = walls
        self.room_grid = room_grid(amap, walls)
        self.rooms = len(walls) + 1
        self.map_name = name


def make_rooms_rules(name="rooms", num_inner_steps=1000, **kw):
    return RoomsHarvestRules(name=name, num_inner_steps=num_inner_steps, **kw)


# ------------------------------------------------------------------------------------------------ bots
def geodesic(amap):
    """(C, C) int32 shortest-path distances (4-neighbour moves, walls `W` impassable) between the C = rows x cols
    cells of a map, by breadth-first search from every cell; FAR where unreachable or for wall cells."""
    rows, cols = len(amap), len(amap[0])
    wall = np.array([[ch == "W" for ch in row] for row in amap])
    C = rows * cols
    D = np.full((C, C), FAR, np.int32)
    for s in range(C):
        r0, c0 = divmod(s, cols)
        if wall[r0, c0]:
            continue
        dist = np.full((rows, cols), FAR, np.int32)
        dist[r0, c0] = 0
        frontier = [(r0, c0)]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for r, c in frontier:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and not wall[rr, cc] and dist[rr, cc] == FAR:
                        dist[rr, cc] = d
                        nxt.append((rr, cc))
            frontier = nxt
        D[s] = dist.ravel()
    return D


FAR = 10_000


def rooms_policy(rules, min_neighbours=3, sneaky_after=20, explore=0.1, migrate_pull=8):
    """Harvest bots (overseer/socialjax/harvest_bots.py: restrained/honest, greedy, sneaky, random, migrator, the
    same codes and the same choice of target) that measure distance along paths around walls instead of straight
    across them, and step to the neighbouring cell nearest the target along such a path. On a map with walls the
    straight-line bots walk into the wall and stay there; these go round, or through a door if there is one.
    Called as policy(key, state, codes, mem, region_grid) -> (actions, mem), like game_bots' policies."""
    import jax
    import jax.numpy as jnp

    from overseer.socialjax.harvest_overseer import APPLE, STAY, apple_count_grid

    rows, cols = rules.shape
    D = jnp.asarray(geodesic(getattr(rules, "map_ascii", HARVEST_MAP)))
    moves = jnp.array([4, 5, 2, 3])  # row+1, row-1, col+1, col-1 (harvest_open STEP_MOVE)
    delta = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    @jax.jit
    def policy(key, state, codes, mem, region_grid):
        grid = state.env_state.grid
        locs = state.env_state.agent_locs[:, :2].astype(jnp.int32)
        apples = grid == APPLE
        allowed = apples & (apple_count_grid(grid) >= min_neighbours)
        forbidden = apples & ~allowed

        def act(loc, code, unchecked_for, unchecked_by_region, key):
            unwatched = (unchecked_by_region >= sneaky_after)[region_grid]
            migrator = code == 4
            greedy_now = ((code == 1) | ((code == 2) & (unchecked_for >= sneaky_after))
                          | (migrator & unwatched[loc[0], loc[1]]))
            careful = (code == 0) | ((code == 2) & ~greedy_now) | (migrator & ~greedy_now)
            targets = jnp.where(migrator, (apples & unwatched) | allowed, jnp.where(greedy_now, apples, allowed))
            pull = jnp.where(migrator & unwatched, migrate_pull, 0)
            here = loc[0] * cols + loc[1]
            dist = jnp.where(targets.ravel(), D[here] - pull.ravel(), FAR)
            i = jnp.argmin(dist)
            dest = jnp.clip(loc[None, :] + delta, 0, jnp.array([rows - 1, cols - 1]))
            dflat = dest[:, 0] * cols + dest[:, 1]
            to_target = D[dflat, i]
            ok = (to_target < FAR) & ~(careful & forbidden[dest[:, 0], dest[:, 1]])
            best = jnp.argmin(jnp.where(ok, to_target, FAR + 1))
            k_move, k_explore = jax.random.split(key)
            no_target = dist[i] >= FAR - migrate_pull
            explore_now = no_target | (code == 3) | (jax.random.uniform(k_explore) < explore)
            rnd = jax.random.randint(k_move, (), 0, 4)
            rnd_ok = ~(careful & forbidden[dest[rnd, 0], dest[rnd, 1]])
            go = jnp.where(ok[best], moves[best], STAY)
            return jnp.where(explore_now, jnp.where(rnd_ok, moves[rnd], STAY), go)

        by_region = state.since_region_check
        keys = jax.random.split(key, locs.shape[0])
        acts = jax.vmap(act)(locs, codes, state.since_check, by_region.T, keys)
        return acts.astype(jnp.int32), mem

    return policy
