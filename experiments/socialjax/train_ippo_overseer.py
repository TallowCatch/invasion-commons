"""IPPO on SocialJax Harvest Open with the budgeted overseer (phase 2).

The training loop is SocialJax's IPPO (algorithms/IPPO/ippo_cnn_harvest_open.py at commit 18e2171: parameter
sharing, the same CNN actor-critic, PPO loss, GAE and hyperparameters from ippo_base.yaml), with three changes:
  1. the env is wrapped in overseer.socialjax.HarvestOverseer (individual rewards; a caught agent's penalty comes off
     its reward; a frozen agent's action is replaced by `stay`);
  2. each update logs the overseer's measures for its episode (NUM_STEPS = episode length, so one update = one
     episode in every parallel env) instead of calling wandb;
  3. it can start from saved parameters (the "agents already over-harvest" start), saves its final parameters, and
     ends with a switch-off test: the trained policy played with the overseer on and with it off.

Protocol: notes/socialjax/phase2_ippo/protocol.md

  python -m experiments.socialjax.train_ippo_overseer --out DIR --k 2 --chooser random --penalty-apples 3.5 \
      [--freeze 0] [--total-steps 1e8] [--num-envs 256] [--seed 1] [--init-params PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import asdict
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from overseer.socialjax import HarvestOverseer, OverseerConfig, make_harvest_env

APPLE_REWARD = 7.0  # with individual rewards, one apple pays num_agents = 7
HPARAMS = dict(LR=5e-4, NUM_STEPS=1000, UPDATE_EPOCHS=2, NUM_MINIBATCHES=500, GAMMA=0.99, GAE_LAMBDA=0.95,
               CLIP_EPS=0.2, ENT_COEF=0.01, VF_COEF=0.5, MAX_GRAD_NORM=0.5, ACTIVATION="relu", ANNEAL_LR=True)
EPISODE_FIELDS = ("apples_eaten", "net_reward", "violations", "detected", "missed", "catches", "wrongly_stopped",
                  "penalty_apples", "checks", "apples_mean", "apples_end", "orchard_alive")


def socialjax_actor_critic():
    """SocialJax's ActorCritic, loaded from its file so the package's wandb import is not needed."""
    import socialjax
    path = Path(socialjax.__file__).resolve().parents[1] / "algorithms" / "utils" / "networks.py"
    spec = importlib.util.spec_from_file_location("socialjax_networks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ActorCritic


def episode_measures(info, last=-2):
    """Per-episode measures from per-step info stacked as (steps, envs, ...); averaged over envs.
    The last step of an episode is the reset step, so the end-of-episode orchard is read one step earlier."""
    def per_env(x):
        return x.reshape(x.shape[0], x.shape[1], -1).sum(axis=(0, 2))
    apples = info["apples"]
    m = dict(
        apples_eaten=per_env(info["eaten"]).mean(),
        net_reward=per_env(info["raw_reward"] - info["penalty"]).mean(),
        violations=per_env(info["violations"]).mean(),
        detected=per_env(info["detected_count"]).mean(),
        missed=per_env(info["missed_count"]).mean(),
        catches=per_env(info["caught"]).mean(),
        wrongly_stopped=per_env(info["false_alarms"]).mean(),
        penalty_apples=per_env(info["penalty"]).mean() / APPLE_REWARD,
        checks=per_env(info["checked"]).mean(),
        apples_mean=apples[:last].mean(),
        apples_end=apples[last].mean(),
        orchard_alive=(apples[last] > 0).mean(),
    )
    return {k: jnp.asarray(v, jnp.float32) for k, v in m.items()}


def make_train(cfg, ov_cfg, init_params=None):
    h = HPARAMS
    env = HarvestOverseer(make_harvest_env(num_inner_steps=h["NUM_STEPS"]), ov_cfg)
    n, e = env.n, cfg["num_envs"]
    actors = n * e
    num_updates = int(cfg["total_steps"]) // h["NUM_STEPS"] // e
    minibatch = actors * h["NUM_STEPS"] // h["NUM_MINIBATCHES"]
    network = socialjax_actor_critic()(env.env.action_space().n, activation=h["ACTIVATION"])
    obs_shape = env.env.observation_space()[0].shape

    def lr(count):
        return h["LR"] * (1.0 - (count // (h["NUM_MINIBATCHES"] * h["UPDATE_EPOCHS"])) / num_updates)

    def train(rng):
        rng, k_init, k_reset, k_run = jax.random.split(rng, 4)
        params = network.init(k_init, jnp.zeros((1, *obs_shape)))
        if init_params is not None:
            params = flax.serialization.from_bytes(params, init_params)
        tx = optax.chain(optax.clip_by_global_norm(h["MAX_GRAD_NORM"]),
                         optax.adam(learning_rate=lr if h["ANNEAL_LR"] else h["LR"], eps=1e-5))
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        obs, state = jax.vmap(env.reset)(jax.random.split(k_reset, e))

        def update(runner, _):
            def env_step(carry, _):
                ts, s, last_obs, rng = carry
                rng, k_act, k_step = jax.random.split(rng, 3)
                obs_b = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape)  # agent-major
                pi, value = network.apply(ts.params, obs_b)
                action = pi.sample(seed=k_act)
                log_prob = pi.log_prob(action)
                obs, s, reward, done, info = jax.vmap(env.step)(jax.random.split(k_step, e), s,
                                                                 action.reshape(n, e).T)
                trans = dict(done=jnp.tile(done["__all__"], n).astype(jnp.float32), action=action, value=value,
                             reward=reward.T.reshape(-1), log_prob=log_prob, obs=obs_b)
                keep = {f: info[f] for f in ("eaten", "raw_reward", "penalty", "violations", "detected_count",
                                             "missed_count", "caught", "false_alarms", "checked", "apples")}
                return (ts, s, obs, rng), (trans, keep)

            ts, s, last_obs, rng, step = runner
            (ts, s, last_obs, rng), (traj, info) = jax.lax.scan(env_step, (ts, s, last_obs, rng), None,
                                                                h["NUM_STEPS"])
            _, last_val = network.apply(ts.params, jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape))

            def gae_step(carry, t):
                gae, next_value = carry
                delta = t["reward"] + h["GAMMA"] * next_value * (1 - t["done"]) - t["value"]
                gae = delta + h["GAMMA"] * h["GAE_LAMBDA"] * (1 - t["done"]) * gae
                return (gae, t["value"]), gae

            _, adv = jax.lax.scan(gae_step, (jnp.zeros_like(last_val), last_val), traj, reverse=True, unroll=16)
            targets = adv + traj["value"]

            def epoch(carry, _):
                ts, rng = carry
                rng, k_perm = jax.random.split(rng)
                batch = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), (traj, adv, targets))
                perm = jax.random.permutation(k_perm, actors * h["NUM_STEPS"])
                mbs = jax.tree.map(lambda x: jnp.take(x, perm, axis=0).reshape(
                    (h["NUM_MINIBATCHES"], minibatch) + x.shape[1:]), batch)

                def minibatch_step(ts, mb):
                    t, gae, tgt = mb

                    def loss_fn(p):
                        pi, value = network.apply(p, t["obs"])
                        log_prob = pi.log_prob(t["action"])
                        v_clip = t["value"] + (value - t["value"]).clip(-h["CLIP_EPS"], h["CLIP_EPS"])
                        v_loss = 0.5 * jnp.maximum(jnp.square(value - tgt), jnp.square(v_clip - tgt)).mean()
                        ratio = jnp.exp(log_prob - t["log_prob"])
                        g = (gae - gae.mean()) / (gae.std() + 1e-8)
                        a_loss = -jnp.minimum(ratio * g, jnp.clip(ratio, 1 - h["CLIP_EPS"], 1 + h["CLIP_EPS"]) * g
                                              ).mean()
                        return a_loss + h["VF_COEF"] * v_loss - h["ENT_COEF"] * pi.entropy().mean()

                    grads = jax.grad(loss_fn)(ts.params)
                    return ts.apply_gradients(grads=grads), None

                ts, _ = jax.lax.scan(minibatch_step, ts, mbs)
                return (ts, rng), None

            (ts, rng), _ = jax.lax.scan(epoch, (ts, rng), None, h["UPDATE_EPOCHS"])
            m = episode_measures(info)
            jax.debug.callback(report, step, m)
            return (ts, s, last_obs, rng, step + 1), m

        runner = (train_state, state, obs, k_run, jnp.zeros((), jnp.int32))
        runner, metrics = jax.lax.scan(update, runner, None, num_updates)
        return runner[0].params, metrics

    return train, num_updates


def report(step, m):
    step = int(step)
    if step % 10 == 0:
        print(f"update {step:4d}  apples {float(m['apples_eaten']):7.1f}  violations {float(m['violations']):6.1f}  "
              f"caught {float(m['detected']):6.1f}  orchard alive {float(m['orchard_alive']):.2f}", flush=True)


def make_eval(ov_cfg, num_envs):
    """The trained policy (sampling its actions) for one episode in each of num_envs envs, no learning."""
    env = HarvestOverseer(make_harvest_env(num_inner_steps=HPARAMS["NUM_STEPS"]), ov_cfg)
    n = env.n
    network = socialjax_actor_critic()(env.env.action_space().n, activation=HPARAMS["ACTIVATION"])
    obs_shape = env.env.observation_space()[0].shape

    def run(params, rng):
        k_reset, k_run = jax.random.split(rng)
        obs, s = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))

        def step(carry, _):
            s, obs, rng = carry
            rng, k_act, k_step = jax.random.split(rng, 3)
            pi, _ = network.apply(params, jnp.transpose(obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape))
            action = pi.sample(seed=k_act).reshape(n, num_envs).T
            obs, s, _, _, info = jax.vmap(env.step)(jax.random.split(k_step, num_envs), s, action)
            return (s, obs, rng), {f: info[f] for f in ("eaten", "raw_reward", "penalty", "violations",
                                                        "detected_count", "missed_count", "caught", "false_alarms",
                                                        "checked", "apples")}

        _, info = jax.lax.scan(step, (s, obs, k_run), None, HPARAMS["NUM_STEPS"])
        return episode_measures(info)

    return jax.jit(run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--chooser", default="none")
    ap.add_argument("--penalty-apples", type=float, default=0.0)
    ap.add_argument("--freeze", type=int, default=0)
    ap.add_argument("--min-neighbours", type=int, default=3)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--total-steps", type=float, default=1e8)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-params", default=None, help="msgpack of a previous run's final params")
    ap.add_argument("--eval-envs", type=int, default=64)
    ap.add_argument("--miss-rate", type=float, default=0.0, help="chance a check misses a real over-harvest")
    ap.add_argument("--false-alarm", type=float, default=0.0, help="chance a check flags an honest harvest")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    chooser = a.chooser if a.k > 0 or a.chooser == "all" else "none"
    ov_cfg = OverseerConfig(k=a.k, chooser=chooser, min_neighbours=a.min_neighbours, window=a.window,
                            penalty=a.penalty_apples * APPLE_REWARD, freeze_steps=a.freeze,
                            miss_rate=a.miss_rate, false_alarm_rate=a.false_alarm)
    cfg = dict(total_steps=a.total_steps, num_envs=a.num_envs, seed=a.seed, init_params=a.init_params)
    init = Path(a.init_params).read_bytes() if a.init_params else None
    (out / "config.json").write_text(json.dumps(dict(cfg, overseer=asdict(ov_cfg), penalty_apples=a.penalty_apples,
                                                     hparams=HPARAMS, devices=str(jax.devices())), indent=1))

    t0 = time.time()
    train, num_updates = make_train(cfg, ov_cfg, init)
    print(f"{num_updates} updates of {a.num_envs} envs x {HPARAMS['NUM_STEPS']} steps; overseer {ov_cfg}", flush=True)
    params, metrics = jax.jit(train)(jax.random.PRNGKey(a.seed))
    params = jax.block_until_ready(params)
    train_seconds = time.time() - t0
    (out / "params.msgpack").write_bytes(flax.serialization.to_bytes(params))
    np.savez(out / "metrics.npz", **{k: np.asarray(v) for k, v in metrics.items()})

    switch = {}
    off = OverseerConfig(chooser="none", min_neighbours=a.min_neighbours)
    for name, c in (("with_overseer", ov_cfg), ("overseer_off", off)):
        m = make_eval(c, a.eval_envs)(params, jax.random.PRNGKey(10_000 + a.seed))
        switch[name] = {k: float(v) for k, v in m.items()}
    last = {k: float(np.asarray(v)[-10:].mean()) for k, v in metrics.items()}
    (out / "summary.json").write_text(json.dumps(dict(train_seconds=train_seconds, num_updates=num_updates,
                                                      last_10_updates=last, switch_off_test=switch), indent=1))
    print(json.dumps(dict(train_seconds=round(train_seconds), last_10_updates=last, switch_off_test=switch),
                     indent=1))


if __name__ == "__main__":
    main()
