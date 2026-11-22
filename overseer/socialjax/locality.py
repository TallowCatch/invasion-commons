"""Measuring locality in SocialJax games (written 22 Nov 2026).

Definitions: notes/phd_plan/2026-11-23_decentralised_overseer_formulation.md, section 5; protocol:
notes/socialjax/locality/protocol.md.

- chi (crossing rate): region changes / agent-steps.
- S_w (evidence stays): per violation, 1 if the violator is in the same region at each of the next w - 1 steps.
- zeta (harm containment): per violation, the share of its harm that lands in the violator's region.
    Harvest: footprint (empty apple cells within two squares whose regrowth chance falls) or counterfactual
    regrowth (the map with and without the eaten apple, regrowth only, same random numbers);
    Clean Up: the region's share of the empty orchard cells (regrowth depends only on global dirt);
    Coins: 1 if the victim is in the thief's region.
- Lambda_w = (1 - chi)^(w-1) * zeta (formula) or mean of S_w * zeta over violations (direct); xi_loc = 1 - Lambda_w.
Also the catch probabilities of section 5.2 (P_C, P_0) and an exact Markov version of P_0 with returns.

Traces come from `collect_trace`, which plays one episode under any overseer with any policy
act(key, obs, state, mem) -> (actions, mem) and keeps positions, violations and (optionally) the map.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from overseer.socialjax.harvest_overseer import APPLE, NEIGHBOURHOOD, apple_count_grid


# ------------------------------------------------------------------------------------------------ traces
@partial(jax.jit, static_argnums=(0, 1), static_argnames=("steps", "keep_grid"))
def _run(overseer, act, key, mem0, steps, keep_grid=True):
    k_reset, k_run = jax.random.split(key)
    obs, s0 = overseer.reset(k_reset)

    def body(carry, key):
        s, obs, mem = carry
        k_act, k_step = jax.random.split(key)
        actions, mem = act(k_act, obs, s, mem)
        obs, s, _, _, info = overseer.step(k_step, s, actions)
        out = dict(pos=overseer.rules.positions(s.env_state), violation=info["violations"], taken=info["taken"],
                   region=info["region"], detected_count=info["detected_count"], caught=info["caught"],
                   missed_count=info["missed_count"], checked=info["checked"], raw_reward=info["raw_reward"],
                   resource=info["resource"], unused_r=info["unused_r"])
        if "handover_steps" in info:
            out["handover_steps"] = info["handover_steps"]
        if "stolen_from" in info:
            out["stolen_from"] = info["stolen_from"]
        if keep_grid:
            out["grid"] = s.env_state.grid.astype(jnp.int8)
        return (s, obs, mem), out

    (s_end, _, _), trace = jax.lax.scan(body, (s0, obs, mem0), jax.random.split(k_run, steps))
    pos0 = overseer.rules.positions(s0.env_state)
    return s_end, trace, pos0


def collect_trace(overseer, act, key, steps, mem0=None, keep_grid=True):
    """One episode; numpy dict with pos (T+1, N, 2) including the start, and per step (T, ...) the rest.
    `steps` must be below the env's episode length so the episode never resets inside the rollout."""
    if mem0 is None:
        mem0 = dict(burst_left=jnp.zeros(overseer.n, jnp.int32))
    s_end, trace, pos0 = _run(overseer, act, key, mem0, steps=steps, keep_grid=keep_grid)
    t = {k: np.asarray(v) for k, v in trace.items()}
    t["pos"] = np.concatenate([np.asarray(pos0)[None], t["pos"]], 0)
    t["left_in_window"] = int(np.asarray(s_end.window).sum())
    return t


def bot_act(policy, codes, region_grid):
    """Adapter: a game_bots policy -> act(key, obs, state, mem)."""
    def act(key, obs, state, mem):
        return policy(key, state, codes, mem, region_grid)
    return act


# ------------------------------------------------------------------------------------------------ regions
def regions_of(pos, grid):
    """(T+1, N) region ids from positions (T+1, N, 2) and a region grid (rows, cols)."""
    grid = np.asarray(grid)
    return grid[pos[..., 0], pos[..., 1]]


def crossing_rate(reg):
    """chi: share of agent-steps (t >= 1) at which the agent is in another region than at t - 1."""
    return float((reg[1:] != reg[:-1]).mean())


def violation_events(trace):
    """(K, 2) array of (t, i): step t (0-based, the step's index in the trace) and agent i of every violation."""
    return np.argwhere(trace["violation"])


def stays(reg, events, w):
    """Per violation: 1.0 if the violator is in the region where it violated at each of the next w - 1 steps,
    NaN if the episode ends before (censored). reg is (T+1, N); a violation at trace step t happened at reg[t+1]."""
    T1 = reg.shape[0]
    out = np.full(len(events), np.nan)
    for j, (t, i) in enumerate(events):
        a = t + 1
        if a + w - 1 >= T1:
            continue
        out[j] = float(np.all(reg[a:a + w, i] == reg[a, i]))
    return out


# ------------------------------------------------------------------------------------------------ zeta
def _neigh_cells(r, c, shape):
    rows, cols = shape
    for dr, dc in NEIGHBOURHOOD:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc


def harvest_footprint(trace, events, rgrid, spawn_mask):
    """Per violation, (dynamic zeta, static zeta, footprint size).
    Dynamic: of the apple spawn cells within two squares of the eaten apple that are empty right after the step
    (their regrowth chance falls: one apple fewer nearby), the share in the violator's region.
    Static: the same over all apple spawn cells within two squares, whatever they hold.
    NaN where the footprint is empty."""
    rgrid = np.asarray(rgrid)
    shape = rgrid.shape
    dyn, sta, size = [], [], []
    for t, i in events:
        r, c = trace["pos"][t + 1, i]
        g = trace["grid"][t]
        home = rgrid[r, c]
        cells = [(rr, cc) for rr, cc in _neigh_cells(r, c, shape) if spawn_mask[rr, cc]]
        empty = [(rr, cc) for rr, cc in cells if g[rr, cc] != APPLE]
        sta.append(np.mean([rgrid[x] == home for x in cells]) if cells else np.nan)
        dyn.append(np.mean([rgrid[x] == home for x in empty]) if empty else np.nan)
        size.append(len(empty))
    return np.asarray(dyn, float), np.asarray(sta, float), np.asarray(size, float)


def harvest_regrow(grid, spawn_mask, u):
    """One step of Harvest Open's regrowth (harvest_open.py, regrow_apple) on an apples-only map, with uniforms u
    (rows, cols): an empty spawn cell regrows with chance 2.5% / 0.5% / 0.1% for 3+ / 2 / 1 apples within two
    squares, never with none."""
    near = apple_count_grid(jnp.where(grid, APPLE, 0))
    p = jnp.where(near >= 3, 0.025, jnp.where(near == 2, 0.005, jnp.where(near == 1, 0.001, 0.0)))
    return grid | (spawn_mask & (u < p))


@partial(jax.jit, static_argnames=("horizon",))
def _cf_batch(apples0, cells, spawn_mask, key, horizon):
    """apples0 (B, rows, cols) bool maps after the violation; cells (B, 2) eaten cells. Returns (B, rows, cols)
    lost apple-steps (without - with the violation) over `horizon` steps, same uniforms in both branches."""
    def one(a0, cell, key):
        b0 = a0.at[cell[0], cell[1]].set(True)

        def body(carry, k):
            a, b = carry
            u = jax.random.uniform(k, a.shape)
            a, b = harvest_regrow(a, spawn_mask, u), harvest_regrow(b, spawn_mask, u)
            return (a, b), (b.astype(jnp.int32) - a.astype(jnp.int32))

        _, diff = jax.lax.scan(body, (a0, b0), jax.random.split(key, horizon))
        lost = diff.sum(0)
        return lost.at[cell[0], cell[1]].set(0)  # the eaten apple itself is the gain, not the harm
    return jax.vmap(one)(apples0, cells, jax.random.split(key, apples0.shape[0]))


def harvest_counterfactual(trace, events, rgrid, spawn_mask, key, horizon=100, max_events=200):
    """Counterfactual regrowth zeta: (in-region lost apple-steps, total lost apple-steps) per sampled violation.
    Agents are removed from both copies (cells they stand on count as empty)."""
    rgrid = np.asarray(rgrid)
    if len(events) == 0:
        return np.zeros(0), np.zeros(0)
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    idx = rng.choice(len(events), size=min(max_events, len(events)), replace=False)
    ev = events[idx]
    maps = np.stack([trace["grid"][t] == APPLE for t, _ in ev])
    cells = np.stack([trace["pos"][t + 1, i] for t, i in ev]).astype(np.int32)
    lost = np.asarray(_cf_batch(jnp.asarray(maps), jnp.asarray(cells), jnp.asarray(spawn_mask), key,
                                horizon=horizon))
    home = rgrid[cells[:, 0], cells[:, 1]]
    inside = np.array([lost[j][rgrid == home[j]].sum() for j in range(len(ev))], float)
    return inside, lost.reshape(len(ev), -1).sum(1).astype(float)


def cleanup_zeta(trace, events, rgrid, orchard_cells, apple_code=3):
    """Per violation: the violator's region's share of the empty orchard cells right after the step."""
    rgrid = np.asarray(rgrid)
    orc_r = rgrid[orchard_cells[:, 0], orchard_cells[:, 1]]
    out = []
    for t, i in events:
        r, c = trace["pos"][t + 1, i]
        empty = trace["grid"][t][orchard_cells[:, 0], orchard_cells[:, 1]] != apple_code
        out.append((orc_r[empty] == rgrid[r, c]).mean() if empty.any() else np.nan)
    return np.asarray(out, float)


def coins_zeta(trace, events, rgrid):
    """Per theft: 1 if the victim (the other player) is in the thief's region right after the step."""
    reg = regions_of(trace["pos"], rgrid)
    return np.asarray([float(reg[t + 1, i] == reg[t + 1, 1 - i]) for t, i in events], float)


# ------------------------------------------------------------------------------------------------ Lambda, xi
def lambda_formula(chi, zeta, w):
    return (1.0 - chi) ** (w - 1) * zeta


def lambda_direct(stay_w, zeta_v):
    m = ~np.isnan(stay_w) & ~np.isnan(zeta_v)
    return float(np.mean(stay_w[m] * zeta_v[m])) if m.any() else float("nan")


# ------------------------------------------------------------------------------------------------ catch formulas
def p_central(q, w, alpha=0.0):
    """P_C = (1 - alpha)[1 - (1 - q)^w]."""
    return (1 - alpha) * (1 - (1 - q) ** w)


def p_independent(q, chi, w, alpha=0.0):
    """P_0 = (1 - alpha) q [1 - ((1 - q)(1 - chi))^w] / [1 - (1 - q)(1 - chi)] (no return within the window)."""
    x = (1 - q) * (1 - chi)
    if np.isclose(x, 1.0):
        return (1 - alpha) * q * w
    return (1 - alpha) * q * (1 - x ** w) / (1 - x)


def p0_markov(q, chi, w, alpha=0.0, pattern="two"):
    """Exact P_0 for independent overseers when the agent can come back, under the formula's other assumptions
    (each check finds the agent with chance q while it is in the violation's region; a miss loses the record).
    pattern: "drift" (never back: the formula), "two" (two regions; a crossing always goes to the other one),
    "ring4" (four regions in a ring; a crossing goes left or right with equal chance).
    The record is caught at the first check made in the home region; it is lost at a missed check there."""
    if pattern == "drift":
        return p_independent(q, chi, w, alpha)
    if pattern == "two":
        P = np.array([[1 - chi, chi], [chi, 1 - chi]])
    elif pattern == "ring4":
        P = np.zeros((4, 4))
        for s in range(4):
            P[s, s] = 1 - chi
            P[s, (s + 1) % 4] += chi / 2
            P[s, (s - 1) % 4] += chi / 2
    else:
        raise ValueError(pattern)
    S = P.shape[0]
    home = np.zeros(S)
    home[0] = 1.0
    # alive[s]: chance the record is still pending (not yet checked in the home region) and the agent is in s
    alive = np.zeros(S)
    alive[0] = 1.0
    caught = 0.0
    for j in range(w):
        if j > 0:
            alive = alive @ P
        checked_home = alive * home * q
        caught += (1 - alpha) * checked_home.sum()
        alive = alive - checked_home  # a check at home ends the record either way (caught or missed)
    return float(caught)
