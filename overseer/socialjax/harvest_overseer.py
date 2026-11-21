"""A budgeted overseer wrapped around SocialJax Harvest Open (`harvest_common_open`).

The rule: eating an apple that leaves fewer than `min_neighbours` apples within two squares is over-harvesting.
It is judged on the map as it was when the agent moved (the previous state), not after everyone has moved.
Harvest's regrowth (harvest_open.py, `regrow_apple`) uses the same 12-cell neighbourhood: an empty cell regrows with
probability 2.5% / 0.5% / 0.1% per step when 3+ / 2 / 1 apples are near it, and never when none are.

Each step, after the agents act:
  1. every agent's over-harvesting this step goes into its evidence window (the last `window` steps);
  2. the overseer picks k agents to check (`chooser`), and sees each checked agent's window;
  3. a check finds the violations with probability 1 - miss_rate, and wrongly flags an honest harvest (an apple
     taken within the rule this step, with no violation in the window) with probability false_alarm_rate (both 0 by
     default: a perfect inspector). Changed 21 Nov 2026 from "any clean agent": a flag on an agent that did nothing
     is noise it cannot act on, while misjudging honest harvesting is the mistake that discourages honest work;
  4. a caught agent pays `penalty` (subtracted from its reward) and/or is frozen for `freeze_steps` steps
     (its actions are replaced by `stay`);
  5. violations that leave the window unchecked, or that a check fails to see, count as missed.
Cost (the "spend" dial): each check costs cost_per_check plus cost_per_step_seen for every step of history it reads,
so a longer look (a bigger window, e.g. an LLM reading more of an agent's trajectory) costs more.

Logged per step (info): violations, checks, catches, false alarms ("normal harvesting wrongly stopped"), missed and
detected violation counts, apples on the map, penalties and cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

APPLE = 3  # socialjax.environments.common_harvest.harvest_open.Items.apple
STAY = 6  # Actions.stay
NEIGHBOURHOOD = ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2),
                 (-1, -1), (-1, 1), (1, -1), (1, 1))  # harvest_open.py, count_apple
CHOOSERS = ("none", "all", "random", "round_robin", "history", "location")


def make_harvest_env(num_inner_steps=1000, **kwargs):
    """Harvest Open with individual rewards (each apple pays num_agents to the eater), so it is a social dilemma."""
    import socialjax
    return socialjax.make("harvest_common_open", num_inner_steps=num_inner_steps, shared_rewards=False, **kwargs)


def apple_count_grid(grid):
    """For every cell, the number of apples among its 12 neighbours (the cell itself excluded)."""
    rows, cols = grid.shape
    padded = jnp.pad((grid == APPLE).astype(jnp.int32), 2)
    return sum(padded[2 + dr:2 + dr + rows, 2 + dc:2 + dc + cols] for dr, dc in NEIGHBOURHOOD)


def apples_near(grid, locs):
    """Apples within two squares of each (row, col) in `locs`."""
    return apple_count_grid(grid)[locs[:, 0], locs[:, 1]]


@dataclass(frozen=True)
class OverseerConfig:
    k: int = 2  # agents checked per step
    chooser: str = "random"  # one of CHOOSERS
    min_neighbours: int = 3  # the rule's m
    window: int = 1  # steps of history a check sees (1 = only what the agent did this step)
    penalty: float = 0.0  # reward removed from a caught agent (one apple pays num_agents)
    freeze_steps: int = 0  # steps a caught agent is held still
    miss_rate: float = 0.0  # chance a check fails to see violations that are there
    false_alarm_rate: float = 0.0  # chance a check flags an honest harvest as a violation
    suspicion_decay: float = 0.95  # for the "history" chooser
    cost_per_check: float = 1.0
    cost_per_step_seen: float = 0.0

    def __post_init__(self):
        if self.chooser not in CHOOSERS:
            raise ValueError(f"chooser must be one of {CHOOSERS}, not {self.chooser!r}")
        if self.k < 0 or self.window < 1:
            raise ValueError("need k >= 0 and window >= 1")


@struct.dataclass
class OverseerState:
    env_state: object
    frozen: jnp.ndarray  # (N,) steps of freeze left
    since_check: jnp.ndarray  # (N,) steps since each agent was last checked
    window: jnp.ndarray  # (N, W) unchecked violations by age (column 0 = this step)
    suspicion: jnp.ndarray  # (N,) decaying count of catches
    rr_ptr: jnp.ndarray  # next agent for round robin


class HarvestOverseer:
    """Wraps a SocialJax Harvest env; reset/step mirror the env's but carry the overseer's state."""

    def __init__(self, env, config: OverseerConfig):
        self.env, self.cfg, self.n = env, config, env.num_agents
        self.k = self.n if config.chooser == "all" else min(config.k, self.n)

    def _fresh(self, env_state):
        n, w = self.n, self.cfg.window
        return OverseerState(env_state=env_state, frozen=jnp.zeros(n, jnp.int32), since_check=jnp.zeros(n, jnp.int32),
                             window=jnp.zeros((n, w), jnp.int32), suspicion=jnp.zeros(n, jnp.float32),
                             rr_ptr=jnp.zeros((), jnp.int32))

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self.env.reset(key)
        return obs, self._fresh(env_state)

    def _choose(self, key, s, near):
        n, k, c = self.n, self.k, self.cfg.chooser
        mask = jnp.zeros(n, bool)
        if c == "none" or k == 0:
            return mask
        if c == "all":
            return jnp.ones(n, bool)
        if c == "random":
            return mask.at[jax.random.permutation(key, n)[:k]].set(True)
        if c == "round_robin":
            return mask.at[(s.rr_ptr + jnp.arange(k)) % n].set(True)
        noise = 1e-6 * jax.random.uniform(key, (n,))  # random tie-break
        if c == "history":  # agents caught before, then the longest unchecked
            score = s.suspicion + 1e-3 * s.since_check + noise
        else:  # "location": agents standing where the fewest apples are left
            score = -near.astype(jnp.float32) + noise
        return mask.at[jax.lax.top_k(score, k)[1]].set(True)

    @partial(jax.jit, static_argnums=0)
    def step(self, key, s, actions):
        cfg, n = self.cfg, self.n
        k_env, k_choose, k_miss, k_alarm = jax.random.split(key, 4)
        actions = jnp.where(s.frozen > 0, STAY, jnp.asarray(actions))
        obs, env_state, raw, dones, info = self.env.step(k_env, s.env_state, actions)
        raw = raw.astype(jnp.float32)
        done = dones["__all__"]

        eaten = raw > 0
        locs = env_state.agent_locs[:, :2]
        # judged on the map the agent saw when it moved, so an apple taken by someone else in the same step
        # does not turn a careful eat into a violation
        violation = eaten & (apples_near(s.env_state.grid, locs) < cfg.min_neighbours)
        near = apples_near(env_state.grid, locs)  # apples left around each agent now (the "location" chooser)

        aged_out = s.window[:, -1]
        window = jnp.concatenate([violation[:, None].astype(jnp.int32), s.window[:, :-1]], axis=1)
        pending = window.sum(1)

        checked = self._choose(k_choose, s, near)
        seen = jax.random.uniform(k_miss, (n,)) >= cfg.miss_rate
        alarm = jax.random.uniform(k_alarm, (n,)) < cfg.false_alarm_rate
        detected = checked & (pending > 0) & seen
        false_alarm = checked & (pending == 0) & eaten & alarm  # an honest harvest misjudged
        caught = detected | false_alarm

        detected_count = jnp.where(detected, pending, 0)
        missed_count = aged_out + jnp.where(checked & ~seen, pending, 0)
        window = jnp.where(checked[:, None], 0, window)
        missed_count = missed_count + jnp.where(done, window.sum(1), 0)  # left unchecked at the end

        penalty = jnp.where(caught, cfg.penalty, 0.0)
        frozen = jnp.where(caught, cfg.freeze_steps, jnp.maximum(s.frozen - 1, 0))
        s_next = OverseerState(
            env_state=env_state, frozen=frozen, since_check=jnp.where(checked, 0, s.since_check + 1), window=window,
            suspicion=cfg.suspicion_decay * s.suspicion + caught, rr_ptr=(s.rr_ptr + self.k) % n)
        s_next = jax.tree.map(lambda fresh, cur: jnp.where(done, fresh, cur), self._fresh(env_state), s_next)

        info = dict(info, eaten=eaten, violations=violation, checked=checked, caught=caught, detected=detected,
                    false_alarms=false_alarm, detected_count=detected_count, missed_count=missed_count,
                    raw_reward=raw, penalty=penalty, frozen=frozen > 0,
                    apples=jnp.sum(env_state.grid == APPLE),
                    cost=checked.sum() * (cfg.cost_per_check + cfg.cost_per_step_seen * cfg.window))
        return obs, s_next, raw - penalty, dones, info
