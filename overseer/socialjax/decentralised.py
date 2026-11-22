"""Decentralised budgeted overseers for any SocialJax game (written 21-22 Nov 2026).

The map is split into R regions (overseer/socialjax/regions.py), with one overseer per region. Each step, after the
agents act:
  1. the game's rule (a GameRules object, overseer/socialjax/games.py) says which agents broke it this step;
     each agent's record of the last `window` steps is kept (violations, acts that look like violations, acts that
     clear an agent, and the region each violation happened in);
  2. overseer r checks up to k_r agents, chosen only among the agents standing in region r now. The total budget is
     k checks per step: k_r = k / R each (the remainder goes to randomly chosen regions each step) unless
     `region_k` sets the split;
  3. a check reads the agent's record. A violation is provable only if the record covers the act and the `lookback`
     steps of history the rule needs (lookback 1 in Harvest and Coins: the act itself; W in Clean Up: "no cleaning
     in the last W steps"). With `presume=True` the overseer also flags acts whose visible part of the lookback shows
     nothing that clears the agent, so it catches unprovable violations and wrongly flags some honest agents;
  4. a check finds what it can with probability 1 - miss_rate, and wrongly flags an honest act (with nothing pending)
     with probability false_alarm_rate, exactly as in the central wrapper (harvest_overseer.py);
  5. a caught agent pays `penalty` and/or is frozen for `freeze_steps` steps;
  6. with `share=True`, every `share_every` steps each overseer sends its suspicion scores to each neighbouring
     region (one message per neighbour, `cost_per_message` each), and each overseer keeps the higher score. With
     `share_coverage=True` the message also carries "steps since I last checked each agent", and each overseer
     keeps the more recent check, so overseers stop re-checking agents a neighbour has just checked. Only the
     "history" chooser uses these scores, so sharing changes nothing under random or round-robin checks.

Independent overseers know only their own checks and catches. With R = 1 this is the central overseer, and it
reproduces HarvestOverseer step for step (tests/test_decentralised_overseer.py).

Logged per step, besides the central wrapper's fields: each agent's region, and per region the checks, catches,
false alarms, unused checks (too few agents in the region), violations, detections and misses by the region where the
violation happened, the check cost, and the communication cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from overseer.socialjax.regions import LAYOUTS, neighbours, region_grid

CHOOSERS = ("none", "all", "random", "round_robin", "history", "location")


@dataclass(frozen=True)
class DecConfig:
    k: int = 2  # total checks per step, summed over regions
    chooser: str = "random"  # one of CHOOSERS
    window: int = 1  # steps of an agent's record a check reads
    penalty: float = 0.0  # reward removed from a caught agent
    freeze_steps: int = 0  # steps a caught agent is held still
    miss_rate: float = 0.0
    false_alarm_rate: float = 0.0
    suspicion_decay: float = 0.95  # for the "history" chooser
    cost_per_check: float = 1.0
    cost_per_step_seen: float = 0.0
    regions: int = 1
    layout: str = "strips"
    region_k: tuple = field(default_factory=tuple)  # per-region budgets; empty = k split equally
    share: bool = False
    share_every: int = 1
    share_coverage: bool = False  # also share when each agent was last checked
    cost_per_message: float = 1.0
    presume: bool = False  # flag acts that look like violations in what the record shows

    def __post_init__(self):
        if self.chooser not in CHOOSERS:
            raise ValueError(f"chooser must be one of {CHOOSERS}, not {self.chooser!r}")
        if self.k < 0 or self.window < 1 or self.regions < 1 or self.share_every < 1:
            raise ValueError("need k >= 0, window >= 1, regions >= 1 and share_every >= 1")
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        if self.region_k and (len(self.region_k) != self.regions or sum(self.region_k) != self.k
                              or min(self.region_k) < 0):
            raise ValueError("region_k needs one non-negative budget per region, summing to k")

    def budgets(self):
        """(base budget per region, number of extra checks handed to random regions each step)."""
        if self.region_k:
            return tuple(int(x) for x in self.region_k), 0
        return (self.k // self.regions,) * self.regions, self.k % self.regions


@struct.dataclass
class DecState:
    env_state: object
    aux: object  # the game rule's own state (e.g. steps since each agent last cleaned)
    frozen: jnp.ndarray  # (N,) steps of freeze left
    since_check: jnp.ndarray  # (N,) steps since any overseer checked the agent
    since_region_check: jnp.ndarray  # (R, N) steps since overseer r checked the agent (what bots can see)
    belief_since: jnp.ndarray  # (R, N) overseer r's belief about it (its own checks, or shared ones)
    suspicion: jnp.ndarray  # (R, N) overseer r's decaying count of catches (own, or shared)
    rr_ptr: jnp.ndarray  # (R,) next agent for round robin
    window: jnp.ndarray  # (N, W) unchecked violations by age (column 0 = this step)
    win_suspect: jnp.ndarray  # (N, W) unchecked acts that would be violations if nothing clears them
    win_clear: jnp.ndarray  # (N, W) acts that clear the agent (e.g. cleaning), by age
    win_region: jnp.ndarray  # (N, W) region where each violation happened
    t: jnp.ndarray  # step counter within the episode


class DecentralisedOverseer:
    """Wraps a SocialJax env and a GameRules; reset/step mirror the env's but carry the overseers' state."""

    def __init__(self, rules, config: DecConfig):
        self.rules, self.env, self.cfg = rules, rules.env, config
        self.n = self.env.num_agents
        self.R = config.regions
        self.region_grid = jnp.asarray(region_grid(rules.shape, config.regions, config.layout))
        self.nbr = neighbours(config.regions, config.layout)
        self.links = int(self.nbr.sum())  # directed messages per sharing round
        base, self.extra = config.budgets()
        self.kmax = tuple(min(b + (1 if self.extra else 0), self.n) for b in base)
        self.base = jnp.asarray(base, jnp.int32)
        self.lookback = int(rules.lookback)
        w, L = config.window, self.lookback
        ages = np.arange(w)
        self.provable_age = jnp.asarray(ages + L - 1 <= w - 1)  # the record covers the act and its lookback
        self.lookback_end = jnp.asarray(np.minimum(ages + L - 1, w - 1))

    # ---------------------------------------------------------------- state
    def _fresh(self, env_state):
        n, w, R = self.n, self.cfg.window, self.R
        zi = jnp.zeros((n, w), jnp.int32)
        return DecState(env_state=env_state, aux=self.rules.init(env_state), frozen=jnp.zeros(n, jnp.int32),
                        since_check=jnp.zeros(n, jnp.int32), since_region_check=jnp.zeros((R, n), jnp.int32),
                        belief_since=jnp.zeros((R, n), jnp.int32), suspicion=jnp.zeros((R, n), jnp.float32),
                        rr_ptr=jnp.zeros(R, jnp.int32), window=zi, win_suspect=zi, win_clear=zi, win_region=zi,
                        t=jnp.zeros((), jnp.int32))

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self.env.reset(key)
        return obs, self._fresh(env_state)

    def regions_of(self, env_state):
        locs = self.rules.positions(env_state)
        return self.region_grid[locs[:, 0], locs[:, 1]]

    # ---------------------------------------------------------------- choosing
    def _choose(self, key, s, region, loc_score):
        """(N,) bool checked, (R,) budget used this step."""
        n, c, R = self.n, self.cfg.chooser, self.R
        budget = self.base
        if self.extra:
            extra = jax.random.permutation(jax.random.fold_in(key, 7919), R)[:self.extra]
            budget = budget.at[extra].add(1)
        mask = jnp.zeros(n, bool)
        if c == "none":
            return mask, jnp.zeros(R, jnp.int32)
        if c == "all":
            return jnp.ones(n, bool), jnp.zeros(R, jnp.int32)
        idx = jnp.arange(n)
        if c == "random":
            perm = jax.random.permutation(key, n)
            base_score = -jnp.zeros(n, jnp.int32).at[perm].set(idx).astype(jnp.float32)  # earlier in perm first
        elif c in ("history", "location"):
            noise = 1e-6 * jax.random.uniform(key, (n,))  # random tie-break
        for r in range(R):
            kk = self.kmax[r]
            if kk == 0:
                continue
            in_r = region == r
            if c == "random":
                score = base_score
            elif c == "round_robin":
                score = -((idx - s.rr_ptr[r]) % n).astype(jnp.float32)
            elif c == "history":  # caught before, then the longest unchecked (as this overseer knows it)
                score = s.suspicion[r] + 1e-3 * s.belief_since[r] + noise
            else:  # "location": the game's own sign of where violations happen
                score = loc_score + noise
            vals, top = jax.lax.top_k(jnp.where(in_r, score, -jnp.inf), kk)
            take = (jnp.arange(kk) < budget[r]) & jnp.isfinite(vals)
            mask = mask | jnp.zeros(n, bool).at[top].set(take)
        return mask, budget

    def _next_rr(self, s, checked, region):
        if self.cfg.chooser != "round_robin":
            return s.rr_ptr
        n, idx = self.n, jnp.arange(self.n)
        ptrs = []
        for r in range(self.R):
            sel = checked & (region == r)
            off = jnp.where(sel, (idx - s.rr_ptr[r]) % n, -1).max()
            ptrs.append(jnp.where(off >= 0, (s.rr_ptr[r] + off + 1) % n, s.rr_ptr[r]))
        return jnp.stack(ptrs).astype(jnp.int32)

    # ---------------------------------------------------------------- step
    @partial(jax.jit, static_argnums=0)
    def step(self, key, s, actions):
        cfg, n, R = self.cfg, self.n, self.R
        k_env, k_choose, k_miss, k_alarm = jax.random.split(key, 4)
        actions = jnp.where(s.frozen > 0, self.rules.stay, jnp.asarray(actions))
        obs, env_state, raw, dones, info = self.env.step(k_env, s.env_state, actions)
        raw = raw.astype(jnp.float32)
        done = dones["__all__"]

        aux, sig = self.rules.judge(s.aux, s.env_state, actions, env_state, raw, info)
        violation = sig["violation"]
        region = self.regions_of(env_state)
        onehot = jax.nn.one_hot(region, R, dtype=jnp.int32)  # (N, R)

        def push(win, new):
            return jnp.concatenate([new[:, None].astype(jnp.int32), win[:, :-1]], axis=1)

        aged_out, aged_region = s.window[:, -1], s.win_region[:, -1]
        window = push(s.window, violation)
        win_suspect = push(s.win_suspect, sig["suspect"])
        win_clear = push(s.win_clear, sig["clear"])
        win_region = push(s.win_region, jnp.where(violation, region, 0))
        pending = window.sum(1)
        provable = window * self.provable_age[None, :]
        if cfg.presume:  # flag what looks like a violation in the part of the lookback the record shows
            cs = jnp.cumsum(win_clear, axis=1)
            before = jnp.concatenate([jnp.zeros((n, 1), jnp.int32), cs[:, :-1]], axis=1)
            visible_clear = cs[:, self.lookback_end] - before
            flagged = win_suspect * (visible_clear == 0)
            detectable = window
            wrongful = flagged * (1 - window)
        else:
            detectable = provable
            wrongful = jnp.zeros_like(window)
        detectable_n = detectable.sum(1)

        checked, budget = self._choose(k_choose, s, region, sig["location_score"])
        seen = jax.random.uniform(k_miss, (n,)) >= cfg.miss_rate
        alarm = jax.random.uniform(k_alarm, (n,)) < cfg.false_alarm_rate
        detected = checked & (detectable_n > 0) & seen
        false_alarm = checked & (pending == 0) & sig["honest"] & alarm  # an honest act misjudged
        presumed_wrong = checked & seen & (wrongful.sum(1) > 0)  # flagged on incomplete evidence
        caught = detected | false_alarm | presumed_wrong

        found = jnp.where(detected[:, None], detectable, 0)  # (N, W) violations caught, by age
        lost = jnp.where(checked[:, None], window - found, 0)  # checked but unseen or unprovable: never caught
        detected_count = found.sum(1)
        missed_count = aged_out + lost.sum(1)
        window = jnp.where(checked[:, None], 0, window)
        win_suspect = jnp.where(checked[:, None], 0, win_suspect)
        at_end = jnp.where(done, window, 0)
        missed_count = missed_count + at_end.sum(1)  # left unchecked at the end

        def by_region(counts_nw, reg_nw):  # (N, W) counts with (N, W) regions -> (R,)
            return (jax.nn.one_hot(reg_nw, R, dtype=jnp.int32) * counts_nw[..., None]).sum((0, 1))

        missed_r = (jnp.zeros(R, jnp.int32).at[aged_region].add(aged_out) + by_region(lost, win_region)
                    + by_region(at_end, win_region))
        detected_r = by_region(found, win_region)

        penalty = jnp.where(caught, cfg.penalty, 0.0)
        frozen = jnp.where(caught, cfg.freeze_steps, jnp.maximum(s.frozen - 1, 0))
        checked_by = onehot.T.astype(bool) & checked[None, :]  # (R, N)
        caught_by = (onehot.T * caught[None, :]).astype(jnp.float32)
        since_region_check = jnp.where(checked_by, 0, s.since_region_check + 1)
        belief_since = jnp.where(checked_by, 0, s.belief_since + 1)
        suspicion = cfg.suspicion_decay * s.suspicion + caught_by
        comm = jnp.zeros((), jnp.float32)
        if cfg.share and R > 1:
            share_now = (s.t % cfg.share_every) == 0
            nb = jnp.asarray(self.nbr)
            nb_susp = jnp.where(nb[:, :, None], suspicion[None, :, :], -jnp.inf).max(1)
            nb_since = jnp.where(nb[:, :, None], belief_since[None, :, :], jnp.iinfo(jnp.int32).max).min(1)
            suspicion = jnp.where(share_now, jnp.maximum(suspicion, nb_susp), suspicion)
            if cfg.share_coverage:
                belief_since = jnp.where(share_now, jnp.minimum(belief_since, nb_since), belief_since)
            comm = jnp.where(share_now, self.links * cfg.cost_per_message, 0.0)

        s_next = DecState(
            env_state=env_state, aux=aux, frozen=frozen, since_check=jnp.where(checked, 0, s.since_check + 1),
            since_region_check=since_region_check, belief_since=belief_since, suspicion=suspicion,
            rr_ptr=self._next_rr(s, checked, region), window=window, win_suspect=win_suspect, win_clear=win_clear,
            win_region=win_region, t=s.t + 1)
        s_next = jax.tree.map(lambda fresh, cur: jnp.where(done, fresh, cur), self._fresh(env_state), s_next)

        per_check = cfg.cost_per_check + cfg.cost_per_step_seen * cfg.window
        checks_r = (onehot * checked[:, None]).sum(0)
        agents_r = onehot.sum(0)
        info = dict(info, **sig.get("extra", {}), taken=sig["taken"], violations=violation, checked=checked,
                    caught=caught, detected=detected, false_alarms=false_alarm, presumed_wrong=presumed_wrong,
                    detected_count=detected_count, missed_count=missed_count, raw_reward=raw, penalty=penalty,
                    frozen=frozen > 0, resource=sig["resource"], cost=checked.sum() * per_check,
                    region=region, agents_r=agents_r, checks_r=checks_r,
                    catches_r=(onehot * caught[:, None]).sum(0),
                    false_alarms_r=(onehot * (false_alarm | presumed_wrong)[:, None]).sum(0),
                    unused_r=(jnp.zeros(R, jnp.int32) if cfg.chooser in ("none", "all") else budget - checks_r),
                    violations_r=(onehot * violation[:, None]).sum(0), detected_r=detected_r, missed_r=missed_r,
                    cost_r=checks_r * per_check, comm_cost=comm)
        return obs, s_next, raw - penalty, dones, info
