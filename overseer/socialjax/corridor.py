"""A toy corridor game for testing the catch formulas against the real overseer code (written 22 Nov 2026).

N agents live on R cells in a ring, one cell per region (map shape (1, R), so strips layout = one region per
cell). Each step an agent may move one cell right or left and may break the rule. Nothing else happens: there is
no resource. It exists so that `LocalRecordOverseer` / `DecentralisedOverseer` can be run with a controllable
crossing rate chi and violation rate (`crossing_policy`), and the simulated catch rates compared with P_C and P_0
(notes/phd_plan/2026-11-23_decentralised_overseer_formulation.md, section 5.2).

Actions: a = move + 3 * violate, with move 0 = stay, 1 = right, 2 = left.
Agents start spread evenly (agent i in region i mod R).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class CorridorState:
    agent_locs: jnp.ndarray  # (N, 3): row (always 0), column (= region), facing (unused)
    t: jnp.ndarray


class CorridorEnv:
    def __init__(self, n_agents=8, regions=2, num_inner_steps=100_000):
        self.num_agents, self.R, self.T = n_agents, regions, num_inner_steps

    def _obs(self, state):
        return jnp.zeros((self.num_agents, 1), jnp.float32)

    def reset(self, key):
        cols = jnp.arange(self.num_agents) % self.R
        locs = jnp.stack([jnp.zeros_like(cols), cols, jnp.zeros_like(cols)], 1).astype(jnp.int32)
        s = CorridorState(agent_locs=locs, t=jnp.zeros((), jnp.int32))
        return self._obs(s), s

    def step(self, key, state, actions):
        move = actions % 3
        shift = jnp.where(move == 1, 1, jnp.where(move == 2, -1, 0))
        cols = (state.agent_locs[:, 1] + shift) % self.R
        locs = state.agent_locs.at[:, 1].set(cols)
        s = CorridorState(agent_locs=locs, t=state.t + 1)
        done = s.t >= self.T
        return self._obs(s), s, jnp.zeros(self.num_agents), {"__all__": done}, {}


class CorridorRules:
    name = "corridor"
    lookback = 1
    stay = 0
    reward_unit = 1.0

    def __init__(self, n_agents=8, regions=2, num_inner_steps=100_000):
        self.env = CorridorEnv(n_agents, regions, num_inner_steps)
        self.shape = (1, regions)

    def positions(self, env_state):
        return env_state.agent_locs[:, :2].astype(jnp.int32)

    def init(self, env_state):
        return {}

    def judge(self, aux, prev, actions, env_state, raw, info):
        v = jnp.asarray(actions) >= 3
        z = jnp.zeros_like(v)
        return aux, dict(violation=v, suspect=v, clear=z, honest=z, taken=v,
                         location_score=jnp.zeros(v.shape, jnp.float32), resource=jnp.zeros((), jnp.int32))


def crossing_policy(chi, violate, pattern="ring"):
    """act(key, obs, state, mem): each agent crosses with chance chi (pattern "ring": left or right at random;
    "drift": always right) and breaks the rule with chance `violate`, independently."""
    def act(key, obs, state, mem):
        n = state.env_state.agent_locs.shape[0]
        k1, k2, k3 = jax.random.split(key, 3)
        cross = jax.random.uniform(k1, (n,)) < chi
        right = jax.random.uniform(k2, (n,)) < 0.5 if pattern == "ring" else jnp.ones(n, bool)
        move = jnp.where(cross, jnp.where(right, 1, 2), 0)
        v = jax.random.uniform(k3, (n,)) < violate
        return (move + 3 * v).astype(jnp.int32), mem
    return act
