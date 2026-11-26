"""Independent regional overseers that see only what happened in their own region (written 22 Nov 2026).

The existing DecentralisedOverseer (overseer/socialjax/decentralised.py) keeps each agent's record of its last w
steps with the agent, and a check by any overseer reads all of it, including steps the agent spent elsewhere. That
is a full hand-over between overseers at no message cost (kappa = infinity in
notes/phd_plan/2026-11-23_decentralised_overseer_formulation.md, section 3.3).

`LocalRecordOverseer(records="local")` is the independent version that section's catch formula P_0 is about
(kappa = 0): overseer m's check reads only the steps the agent spent in region m (rho_m in section 3.2).
- A violation committed in region m stays pending in the agent's record until it ages out of the window. Only
  overseer m can catch it, and only while the agent is back in m. A check by another overseer neither sees nor
  clears it.
- A check by m clears only what m could see; a miss (miss_rate) loses what m saw, as in the parent class.
- `records="handover"` is the parent class's behaviour step for step (tests/test_locality.py).
- `region_grid` replaces the layout's regions with any (rows, cols) map of region ids, e.g. the rooms of
  overseer/socialjax/rooms.py; R is then the number of ids. Neighbours (only used for sharing) are left as the
  layout's.
- The handover message cost is logged: each time an agent with pending records crosses into another region, the
  number of pending record-steps it carries (`handover_steps`), so communication can be compared with central
  collection (N x w record-steps per step).
`presume` (flagging on incomplete evidence) is not supported here.
"""
from __future__ import annotations

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp

from overseer.socialjax.decentralised import DecConfig, DecentralisedOverseer, DecState  # noqa: F401

RECORDS = ("local", "handover")


class LocalRecordOverseer(DecentralisedOverseer):

    def __init__(self, rules, config: DecConfig, records="local", region_grid=None):
        if records not in RECORDS:
            raise ValueError(f"records must be one of {RECORDS}")
        if config.presume:
            raise ValueError("presume is not supported by LocalRecordOverseer")
        if region_grid is not None:
            grid = jnp.asarray(region_grid, jnp.int32)
            R = int(grid.max()) + 1
            if R != config.regions:
                config = replace(config, regions=R, layout="strips")
        super().__init__(rules, config)
        if region_grid is not None:
            self.region_grid = grid
        self.records = records

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
        prev_region = self.regions_of(s.env_state)
        region = self.regions_of(env_state)
        onehot = jax.nn.one_hot(region, R, dtype=jnp.int32)

        def push(win, new):
            return jnp.concatenate([new[:, None].astype(jnp.int32), win[:, :-1]], axis=1)

        crossed = region != prev_region
        handover_steps = jnp.where(crossed, s.window.sum(1), 0)  # pending record-steps carried across

        aged_out, aged_region = s.window[:, -1], s.win_region[:, -1]
        window = push(s.window, violation)
        win_suspect = push(s.win_suspect, sig["suspect"])
        win_clear = push(s.win_clear, sig["clear"])
        win_region = push(s.win_region, jnp.where(violation, region, 0))
        pending = window.sum(1)
        provable = window * self.provable_age[None, :]
        if self.records == "local":
            visible = (win_region == region[:, None]).astype(jnp.int32)  # what the current region's overseer saw
        else:
            visible = jnp.ones_like(window)
        detectable = provable * visible
        detectable_n = detectable.sum(1)

        checked, budget = self._choose(k_choose, s, region, sig["location_score"])
        seen = jax.random.uniform(k_miss, (n,)) >= cfg.miss_rate
        alarm = jax.random.uniform(k_alarm, (n,)) < cfg.false_alarm_rate
        detected = checked & (detectable_n > 0) & seen
        # an honest act misjudged: nothing pending that this overseer can see
        false_alarm = checked & ((window * visible).sum(1) == 0) & sig["honest"] & alarm
        caught = detected | false_alarm

        found = jnp.where(detected[:, None], detectable, 0)
        lost = jnp.where(checked[:, None], window * visible - found, 0)  # seen by this check but not caught
        detected_count = found.sum(1)
        missed_count = aged_out + lost.sum(1)
        window = jnp.where(checked[:, None], window * (1 - visible), window)
        win_suspect = jnp.where(checked[:, None], win_suspect * (1 - visible), win_suspect)
        at_end = jnp.where(done, window, 0)
        missed_count = missed_count + at_end.sum(1)

        def by_region(counts_nw, reg_nw):
            return (jax.nn.one_hot(reg_nw, R, dtype=jnp.int32) * counts_nw[..., None]).sum((0, 1))

        missed_r = (jnp.zeros(R, jnp.int32).at[aged_region].add(aged_out) + by_region(lost, win_region)
                    + by_region(at_end, win_region))
        detected_r = by_region(found, win_region)

        penalty = jnp.where(caught, cfg.penalty, 0.0)
        frozen = jnp.where(caught, cfg.freeze_steps, jnp.maximum(s.frozen - 1, 0))
        checked_by = onehot.T.astype(bool) & checked[None, :]
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
            win_region=win_region, t=s.t + 1, since_caught=jnp.where(caught, 0, s.since_caught + 1))
        s_next = jax.tree.map(lambda fresh, cur: jnp.where(done, fresh, cur), self._fresh(env_state), s_next)

        per_check = cfg.cost_per_check + cfg.cost_per_step_seen * cfg.window
        checks_r = (onehot * checked[:, None]).sum(0)
        agents_r = onehot.sum(0)
        info = dict(info, **sig.get("extra", {}), taken=sig["taken"], violations=violation, checked=checked,
                    caught=caught, detected=detected, false_alarms=false_alarm,
                    presumed_wrong=jnp.zeros(n, bool), detected_count=detected_count, missed_count=missed_count,
                    raw_reward=raw, penalty=penalty, frozen=frozen > 0, resource=sig["resource"],
                    cost=checked.sum() * per_check, region=region, crossed=crossed, handover_steps=handover_steps,
                    agents_r=agents_r, checks_r=checks_r, catches_r=(onehot * caught[:, None]).sum(0),
                    false_alarms_r=(onehot * false_alarm[:, None]).sum(0),
                    unused_r=(jnp.zeros(R, jnp.int32) if cfg.chooser in ("none", "all") else budget - checks_r),
                    violations_r=(onehot * violation[:, None]).sum(0), detected_r=detected_r, missed_r=missed_r,
                    cost_r=checks_r * per_check, comm_cost=comm)
        return obs, s_next, raw - penalty, dones, info
