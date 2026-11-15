"""Run one Harvest episode with scripted bots under the overseer, and summarise it."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from overseer.socialjax.harvest_bots import bot_actions

STEP_FIELDS = ("eaten", "violations", "checked", "caught", "detected", "false_alarms", "detected_count",
               "missed_count", "raw_reward", "penalty", "apples", "cost")


@partial(jax.jit, static_argnums=(0,), static_argnames=("steps", "min_neighbours", "sneaky_after", "explore"))
def run_episode(overseer, key, codes, steps, min_neighbours=3, sneaky_after=20, explore=0.1):
    """`steps` must be below the env's num_inner_steps, so the episode never resets inside the rollout."""
    k_reset, k_run = jax.random.split(key)
    _, s0 = overseer.reset(k_reset)

    def body(s, key):
        k_bot, k_step = jax.random.split(key)
        actions = bot_actions(k_bot, s, codes, min_neighbours=min_neighbours, sneaky_after=sneaky_after,
                              explore=explore)
        _, s, _, _, info = overseer.step(k_step, s, actions)
        return s, {f: info[f] for f in STEP_FIELDS}

    s_end, trace = jax.lax.scan(body, s0, jax.random.split(k_run, steps))
    return s_end, trace


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def summarise(s_end, trace):
    """Episode metrics. Violations still in the evidence window at the end count as missed."""
    t = {k: np.asarray(v) for k, v in trace.items()}
    eaten = t["eaten"].sum(0)
    steps = t["apples"].shape[0]
    violations = int(t["violations"].sum())
    detected = int(t["detected_count"].sum())
    missed = int(t["missed_count"].sum()) + int(np.asarray(s_end.window).sum())
    eat_times = np.nonzero(t["eaten"])[0]
    return dict(
        steps=steps,
        apples_eaten=int(eaten.sum()),
        eaten_per_agent=eaten.tolist(),
        net_reward=float((t["raw_reward"] - t["penalty"]).sum()),
        violations=violations,
        detected=detected,
        missed=missed,
        missed_share=missed / violations if violations else 0.0,
        wrongly_stopped=int(t["false_alarms"].sum()),
        catches=int(t["caught"].sum()),
        checks=int(t["checked"].sum()),
        cost=float(t["cost"].sum()),
        apples_start=int(t["apples"][0]),
        apples_end=int(t["apples"][-1]),
        apples_min=int(t["apples"].min()),
        apples_mean=float(t["apples"].mean()),
        orchard_alive=bool(t["apples"][-1] > 0),
        equality=1.0 - gini(eaten),
        sustainability=float(eat_times.mean()) if eat_times.size else 0.0,  # Perolat et al. 2017: mean eating time
    )
