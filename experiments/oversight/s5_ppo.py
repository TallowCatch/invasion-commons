"""T2 attacker for S5: one shared PPO policy (Schulman et al. 2017) for all cheaters.

Observation (per cheater): stock / K, own request, own allowance, steps since own last audit (capped at 10, / 10;
never audited = 1), caught-before flag, t / T. Action: a level in {0, 0.25, 0.5, 0.75, 1}. Reward: the cheater's
own payoff minus its fines that step. Episodes cycle through the training contexts (episode e uses context
e mod n_train) and run in lockstep batches so that one forward pass serves every live cheater.
Deterministic: torch.set_num_threads(1), torch.manual_seed(seed) for the weights, a numpy Generator(seed) for action
sampling and minibatch order.
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import torch
from torch import nn

from experiments.oversight import run_s5_stronger_attackers as s5

OBS_DIM, N_ACT = 6, len(s5.LEVELS)
LEVELS = np.asarray(s5.LEVELS)

# Final hyperparameters (fixed at the smoke stage on training payoff only).
HYPERPARAMS = dict(hidden=64, layers=2, activation="tanh", lr=3e-4, gamma=1.0, gae_lambda=0.95, clip=0.2,
                   epochs=4, minibatch=512, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
                   reward_scale=0.1, adv_norm=True, init="torch default; policy head scaled by 0.01",
                   batch_episodes="profile ppo_batch", eval_mode="sample",
                   eval="policy (eval_mode) on all training contexts every ppo_eval_every iterations, at iteration 0 "
                   "and at the end; best = highest mean group cheater payoff, ties -> earliest")


def mlp(out, hp):
    return nn.Sequential(nn.Linear(OBS_DIM, hp["hidden"]), nn.Tanh(), nn.Linear(hp["hidden"], hp["hidden"]), nn.Tanh(),
                         nn.Linear(hp["hidden"], out))


class ActorCritic(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.pi, self.v = mlp(N_ACT, hp), mlp(1, hp)
        with torch.no_grad():
            self.pi[-1].weight.mul_(0.01)
            self.pi[-1].bias.zero_()

    def forward(self, x):
        return self.pi(x), self.v(x).squeeze(-1)


def obs_matrix(obs):
    since = np.minimum(np.asarray(obs.since, float), 10.0) / 10.0
    k = len(obs.req)
    return np.column_stack([np.full(k, obs.stock / obs.stock_max), obs.req, obs.allowance, since,
                            obs.caught.astype(float), np.full(k, obs.t / obs.horizon)]).astype(np.float32)


class TrainedPolicy:
    """The trained network as an attacker. mode 'sample' draws the level from the policy's distribution (what PPO
    optimises); mode 'greedy' takes the most likely level. Sampling uses one numpy Generator per episode, seeded by
    (eval seed, split, context), so an episode does not depend on which other episodes share its batch."""

    def __init__(self, state, hp=HYPERPARAMS, mode=None):
        self.net = ActorCritic(hp)
        self.net.load_state_dict(state)
        self.net.eval()
        self.mode = mode or hp["eval_mode"]

    def act_batch(self, X, u=None):
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(X))
        if self.mode == "greedy":
            return logits.argmax(-1).numpy()
        return sample_actions(torch.log_softmax(logits, -1).numpy(), u)

    def act(self, obs, rng=None):
        X = obs_matrix(obs)
        return LEVELS[self.act_batch(X, None if rng is None else rng.random(len(X)))]


GreedyPolicy = TrainedPolicy  # backwards-compatible name


def sample_actions(logp, u):
    """Inverse-CDF sampling (float64) from log-probabilities with uniforms u; identical in training and evaluation."""
    probs = np.exp(logp.astype(np.float64))
    cdf = np.cumsum(probs, axis=1)
    return np.minimum((cdf < (u * cdf[:, -1])[:, None]).sum(axis=1), N_ACT - 1)


def eval_rng(context, train, seeds=s5.SEEDS):
    return np.random.default_rng(s5.stable_seed(seeds["ppo_eval"], "train" if train else "test", int(context)))


def evaluate_lockstep(regime, q, policy, P, contexts, train=True, seeds=s5.SEEDS):
    """Evaluation of a trained policy; each episode equals running its context alone with its own sampling stream."""
    eps = [s5.Episode(c, regime, q, P, seeds, train) for c in contexts]
    rngs = [eval_rng(c, train, seeds) for c in contexts]
    while True:
        live = [j for j, e in enumerate(eps) if not e.done]
        if not live:
            break
        mats = [obs_matrix(eps[j].observe()) for j in live]
        u = np.concatenate([rngs[j].random(len(m)) for j, m in zip(live, mats)])
        acts = policy.act_batch(np.concatenate(mats), u)
        i = 0
        for j, m in zip(live, mats):
            eps[j].step(LEVELS[acts[i:i + len(m)]])
            i += len(m)
    return [e.summary() for e in eps]


def state_copy(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def save_state(state, path):
    obj = {k: v.tolist() for k, v in state.items()}
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        f.write(json.dumps(obj).encode())


def load_state(path):
    with gzip.open(path, "rt") as f:
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in json.load(f).items()}


def rollout(net, regime, q, P, contexts, rng, seeds=s5.SEEDS):
    eps = [s5.Episode(c, regime, q, P, seeds, True) for c in contexts]
    k = [len(e.cheat) for e in eps]
    buf = [[dict(obs=[], act=[], logp=[], val=[], rew=[]) for _ in range(n)] for n in k]
    while True:
        live = [j for j, e in enumerate(eps) if not e.done]
        if not live:
            break
        mats = [obs_matrix(eps[j].observe()) for j in live]
        X = np.concatenate(mats)
        with torch.no_grad():
            logits, vals = net(torch.from_numpy(X))
            logp_all = torch.log_softmax(logits, -1)
        acts = sample_actions(logp_all.numpy(), rng.random(len(X)))
        lp = logp_all.numpy()[np.arange(len(X)), acts]
        v = vals.numpy()
        i = 0
        for j, m in zip(live, mats):
            r = eps[j].step(LEVELS[acts[i:i + len(m)]])
            for c in range(len(m)):
                b = buf[j][c]
                b["obs"].append(m[c]); b["act"].append(int(acts[i + c])); b["logp"].append(float(lp[i + c]))
                b["val"].append(float(v[i + c])); b["rew"].append(float(r[c]))
            i += len(m)
    trajs = [b for per in buf for b in per]
    return trajs, [e.summary() for e in eps]


def gae(traj, hp):
    r = np.asarray(traj["rew"]) * hp["reward_scale"]
    v = np.asarray(traj["val"], float)
    adv = np.zeros(len(r))
    last = 0.0
    for t in reversed(range(len(r))):
        nv = v[t + 1] if t + 1 < len(r) else 0.0  # episodes end at the horizon or at collapse: terminal
        delta = r[t] + hp["gamma"] * nv - v[t]
        last = delta + hp["gamma"] * hp["gae_lambda"] * last
        adv[t] = last
    return adv, adv + v


def train(regime, q, P, hp, seed, log=print):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = ActorCritic(hp)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    n_train, batch, budget = P["train_contexts"], P["ppo_batch"], P["ppo_episodes"]
    pol = TrainedPolicy(state_copy(net), hp)

    def eval_now():
        pol.net.load_state_dict(state_copy(net))
        eps = evaluate_lockstep(regime, q, pol, P, range(n_train), True)
        return float(np.mean([e["cheater_payoff"] for e in eps])), float(np.mean([e["mean_level"] for e in eps]))

    sc, ml = eval_now()
    best = dict(score=sc, iteration=0, state=state_copy(net))
    history = [dict(iteration=0, episodes=0, eval_train_group=sc, eval_mean_level=ml)]
    used, it = 0, 0
    while used < budget:
        nb = min(batch, budget - used)
        contexts = [(used + j) % n_train for j in range(nb)]
        trajs, summ = rollout(net, regime, q, P, contexts, rng)
        used += nb
        it += 1
        obs = np.concatenate([np.asarray(t["obs"]) for t in trajs]).astype(np.float32)
        act = np.concatenate([t["act"] for t in trajs])
        old = np.concatenate([t["logp"] for t in trajs]).astype(np.float32)
        advs, rets = zip(*[gae(t, hp) for t in trajs])
        adv, ret = np.concatenate(advs), np.concatenate(rets)
        if hp["adv_norm"]:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        O, A, OL = torch.from_numpy(obs), torch.from_numpy(act), torch.from_numpy(old)
        AD, RT = torch.from_numpy(adv.astype(np.float32)), torch.from_numpy(ret.astype(np.float32))
        n = len(obs)
        stats = []
        for _ in range(hp["epochs"]):
            perm = rng.permutation(n)
            for s in range(0, n, hp["minibatch"]):
                idx = torch.from_numpy(perm[s:s + hp["minibatch"]])
                logits, v = net(O[idx])
                logp_all = torch.log_softmax(logits, -1)
                lp = logp_all.gather(1, A[idx].unsqueeze(1)).squeeze(1)
                ratio = torch.exp(lp - OL[idx])
                pg = -torch.min(ratio * AD[idx], torch.clamp(ratio, 1 - hp["clip"], 1 + hp["clip"]) * AD[idx]).mean()
                vl = ((v - RT[idx]) ** 2).mean()
                ent = -(logp_all.exp() * logp_all).sum(-1).mean()
                loss = pg + hp["value_coef"] * vl - hp["entropy_coef"] * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), hp["max_grad_norm"])
                opt.step()
                stats.append((pg.item(), vl.item(), ent.item()))
        row = dict(iteration=it, episodes=used, rollout_group=float(np.mean([e["cheater_payoff"] for e in summ])),
                   pg_loss=float(np.mean([s[0] for s in stats])), v_loss=float(np.mean([s[1] for s in stats])),
                   entropy=float(np.mean([s[2] for s in stats])))
        if it % P["ppo_eval_every"] == 0 or used >= budget:
            sc, ml = eval_now()
            row.update(eval_train_group=sc, eval_mean_level=ml)
            if sc > best["score"] + s5.EPS:
                best = dict(score=sc, iteration=it, state=state_copy(net))
            log(f"  T2 {regime} q={q:.4f} it={it} eps={used} rollout={row['rollout_group']:.2f} eval={sc:.2f} "
                f"level={ml:.2f} ent={row['entropy']:.3f}", flush=True)
        history.append(row)
    return dict(best_state=best["state"], best_score=best["score"], best_iteration=best["iteration"],
                episodes_used=used, log=history)
