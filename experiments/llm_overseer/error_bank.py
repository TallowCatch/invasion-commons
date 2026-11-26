"""Situation bank for the patterned-inspector study (notes/llm_overseer/error_patterns/protocol.md, section 1).

Plays trained IPPO agents (stage 1 params) in Harvest with no overseer, and keeps every apple eaten as one judging
item at w = 1: the wrapper's label, the features the trainer's inspector table uses (overseer/socialjax/
inspector_table.py) and the raw rendering the LLM reads (overseer/llm/harvest_records.render_raw).

  PYTHONPATH=. .venv-socialjax/bin/python -m experiments.llm_overseer.error_bank \
      --params-root results/create_pull --out experiments/llm_overseer/data/error_bank.jsonl.gz

Default sources: A_none_seed1 (32 episodes), B_random_k1_c1.0_seed1, B_random_k2_c1.0_seed1, B_random_k7_c2.0_seed1,
D_history_k2_c1.0_seed1 (6 episodes each); 1,000 steps per episode, actions sampled from the policy.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

SOURCES = (("A_none_seed1", 32), ("B_random_k1_c1.0_seed1", 6), ("B_random_k2_c1.0_seed1", 6),
           ("B_random_k7_c2.0_seed1", 6), ("D_history_k2_c1.0_seed1", 6))
STEPS = 1000


def make_runner(steps=STEPS):
    """jit(run(params, key)) -> per-step numpy-able dict for one episode of trained agents under no overseer."""
    import jax
    import jax.numpy as jnp

    from experiments.socialjax.train_ippo_overseer import socialjax_actor_critic
    from overseer.socialjax.decentralised import DecConfig, DecentralisedOverseer
    from overseer.socialjax.games import make_rules
    from overseer.socialjax.inspector_table import features

    rules = make_rules("harvest", num_inner_steps=steps + 1)  # never resets inside the recorded steps
    env = DecentralisedOverseer(rules, DecConfig(k=0, chooser="none"))
    network = socialjax_actor_critic()(env.env.action_space().n, activation="relu")
    obs_shape = env.obs_shape()

    def run(params, key):
        k_reset, k_run = jax.random.split(key)
        obs, s0 = env.reset(k_reset)

        def body(carry, key):
            s, obs = carry
            k_act, k_step = jax.random.split(key)
            pi, _ = network.apply(params, obs)
            actions = pi.sample(seed=k_act)
            pre_grid, pre_locs = s.env_state.grid, s.env_state.agent_locs[:, :2]
            obs, s2, _, _, info = env.step(k_step, s, actions)
            post = s2.env_state.agent_locs[:, :2]
            out = dict(pre_grid=pre_grid.astype(jnp.int8), pre_locs=pre_locs, post_locs=post, eaten=info["taken"],
                       violations=info["violations"], **features(pre_grid, post.astype(jnp.int32)))
            return (s2, obs), out

        _, trace = jax.lax.scan(body, (s0, obs), jax.random.split(k_run, steps))
        return trace

    def init_params(raw):
        import flax.serialization
        p = network.init(jax.random.PRNGKey(0), jnp.zeros((1, *obs_shape)))
        return flax.serialization.from_bytes(p, raw)

    return jax.jit(run), init_params


def items_from_episode(tr, source, episode):
    from overseer.llm.harvest_records import render_raw
    out = []
    ts, agents = np.nonzero(tr["eaten"])
    for t, a in zip(ts, agents):
        t, a = int(t), int(a)
        out.append(dict(id=f"bank_{source}_e{episode}_t{t}_a{a}", dataset="harvest", window=1, source=source,
                        episode=episode, agent=a, t=t + 1, label=int(tr["violations"][t, a]),
                        near=int(tr["near"][t, a]), outer=int(tr["outer"][t, a]),
                        players=int(tr["players"][t, a]), walls=int(tr["walls"][t, a]),
                        renderings=dict(raw=render_raw(tr, a, t, t))))
    return out


def main():
    import jax
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-root", default="results/create_pull")
    ap.add_argument("--out", default="experiments/llm_overseer/data/error_bank.jsonl.gz")
    ap.add_argument("--scale-episodes", type=float, default=1.0, help="multiply every source's episode count")
    a = ap.parse_args()
    run, init_params = make_runner()
    items, per_source = [], {}
    for si, (source, n_ep) in enumerate(SOURCES):
        params = init_params((Path(a.params_root) / source / "params.msgpack").read_bytes())
        t0 = time.time()
        got = []
        for e in range(max(1, round(n_ep * a.scale_episodes))):
            tr = {k: np.asarray(v) for k, v in run(params, jax.random.PRNGKey(30_000 + 100 * si + e)).items()}
            got += items_from_episode(tr, source, e)
        per_source[source] = dict(episodes=e + 1, eats=len(got), violations=sum(x["label"] for x in got))
        print(f"{source}: {per_source[source]} ({time.time() - t0:.0f} s)", flush=True)
        items += got
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as fh:
        for x in items:
            fh.write(json.dumps(x) + "\n")
    manifest = dict(n_items=len(items), violations=sum(x["label"] for x in items), per_source=per_source,
                    steps=STEPS, near=dict(sorted(Counter(x["near"] for x in items).items())),
                    near_by_label={str(lab): dict(sorted(Counter(x["near"] for x in items if x["label"] == lab
                                                                 ).items())) for lab in (0, 1)})
    Path(str(out).replace(".jsonl.gz", "_manifest.json")).write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
