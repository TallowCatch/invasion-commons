"""Run one episode of any game with scripted bots under (decentralised) overseers, and summarise it."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from overseer.socialjax.game_bots import init_mem
from overseer.socialjax.rollout import gini

STEP_FIELDS = ("taken", "violations", "checked", "caught", "detected", "false_alarms", "presumed_wrong",
               "detected_count", "missed_count", "raw_reward", "penalty", "resource", "cost", "comm_cost",
               "agents_r", "checks_r", "catches_r", "false_alarms_r", "unused_r", "violations_r", "detected_r",
               "missed_r")
EXTRA_FIELDS = {"cleanup": ("pollution", "cleaned"), "coins": ("stolen_from",), "harvest": ()}


@partial(jax.jit, static_argnums=(0, 3), static_argnames=("steps",))
def run_episode(overseer, key, codes, policy, steps):
    """`steps` must be below the env's num_inner_steps, so the episode never resets inside the rollout."""
    k_reset, k_run = jax.random.split(key)
    _, s0 = overseer.reset(k_reset)
    fields = STEP_FIELDS + EXTRA_FIELDS[overseer.rules.name]

    def body(carry, key):
        s, mem = carry
        k_bot, k_step = jax.random.split(key)
        actions, mem = policy(k_bot, s, codes, mem, overseer.region_grid)
        _, s, _, _, info = overseer.step(k_step, s, actions)
        return (s, mem), {f: info[f] for f in fields}

    (s_end, _), trace = jax.lax.scan(body, (s0, init_mem(overseer.n)), jax.random.split(k_run, steps))
    return s_end, trace


def run_seeds(overseer, seeds, codes, policy, steps):
    """Summaries of one episode per seed, run together with vmap."""
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(list(seeds)))
    s_end, trace = jax.vmap(lambda k: run_episode(overseer, k, codes, policy, steps=steps))(keys)
    out = []
    for i, seed in enumerate(seeds):
        r = summarise(jax.tree.map(lambda x, i=i: x[i], s_end), {f: v[i] for f, v in trace.items()},
                      overseer.rules.reward_unit)
        r["seed"] = int(seed)
        out.append(r)
    return out


def summarise(s_end, trace, unit=1.0):
    """Episode metrics. Violations still in the evidence window at the end count as missed (in their region)."""
    t = {k: np.asarray(v) for k, v in trace.items()}
    taken = t["taken"].sum(0)
    violations = int(t["violations"].sum())
    detected = int(t["detected_count"].sum())
    left = np.asarray(s_end.window)
    left_r = np.bincount(np.asarray(s_end.win_region)[left > 0], minlength=t["checks_r"].shape[1])
    missed = int(t["missed_count"].sum()) + int(left.sum())
    raw = t["raw_reward"].sum(0)
    take_times = np.nonzero(t["taken"])[0]
    out = dict(
        steps=int(t["resource"].shape[0]),
        taken=int(taken.sum()),
        taken_per_agent=taken.tolist(),
        reward_per_agent=(raw / unit).tolist(),  # in goods (apples or coins)
        violations_per_agent=t["violations"].sum(0).tolist(),
        caught_per_agent=t["caught"].sum(0).tolist(),
        welfare=float(raw.sum() / unit),
        violations=violations,
        detected=detected,
        missed=missed,
        caught_share=detected / violations if violations else float("nan"),
        wrongly_stopped=int(t["false_alarms"].sum() + t["presumed_wrong"].sum()),
        presumed_wrong=int(t["presumed_wrong"].sum()),
        catches=int(t["caught"].sum()),
        checks=int(t["checked"].sum()),
        cost=float(t["cost"].sum()),
        comm_cost=float(t["comm_cost"].sum()),
        resource_end=int(t["resource"][-1]),
        resource_mean=float(t["resource"].mean()),
        alive=bool(t["resource"][-1] > 0),
        equality=1.0 - gini(np.maximum(raw, 0)),
        sustainability=float(take_times.mean()) if take_times.size else 0.0,
        regions=dict(
            agents_mean=t["agents_r"].mean(0).tolist(),
            checks=t["checks_r"].sum(0).tolist(),
            catches=t["catches_r"].sum(0).tolist(),
            wrong_flags=t["false_alarms_r"].sum(0).tolist(),
            unused_checks=t["unused_r"].sum(0).tolist(),
            violations=t["violations_r"].sum(0).tolist(),
            detected=t["detected_r"].sum(0).tolist(),
            missed=(t["missed_r"].sum(0) + left_r).tolist(),
        ),
    )
    if "pollution" in t:
        out["pollution_mean"] = float(t["pollution"].mean())
        out["cleaned_per_agent"] = t["cleaned"].sum(0).tolist()
    if "stolen_from" in t:
        out["stolen_from_per_agent"] = t["stolen_from"].sum(0).tolist()
    return out
